"""Extra cameras beyond the one that scores.

The scoring engine owns exactly one camera, because scoring is a single
sequential thing: one board, one settling state machine, one game. But a second
camera has to be *set up* long before it can contribute anything - it has to be
pointed, focused and calibrated, and all of that needs its live picture on
screen and its own four clicks saved somewhere.

So a secondary view is deliberately much less than the primary. It opens its
camera, hands out frames, and keeps its own calibration. It does not detect, it
does not score, and it cannot end a visit. That keeps it impossible for a
half-configured second camera to disturb a game in progress, which is the
failure worth designing against - and it is all that is needed until the fusion
step in fusion.py is wired into scoring.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from . import render
from .calibration import Calibration, measure_pose
from .detector import DartDetector


class SecondaryView:
    """One extra camera: opened, streamed and calibrated, but not scoring."""

    def __init__(self, cfg, calibration_path: str, open_source,
                 detector_cfg=None) -> None:
        self.cfg = cfg
        self.name = cfg.name
        self.calibration_path = calibration_path
        self._open_source = open_source
        self._detector_cfg = detector_cfg
        self._watch_cfg = None

        # This view watches for darts, but only so it can answer when
        # asked. It never publishes one: the scoring camera alone decides
        # whether a dart happened, so a shadow or a passing hand here
        # cannot invent a throw, and this camera falling over cannot stop
        # a game.
        self.detector: DartDetector | None = None
        self._observations: deque = deque(maxlen=16)
        # A few seconds of prepared greyscale, so a dart can be measured
        # at the moment the scoring camera saw it rather than whenever
        # this one happened to settle.
        self._recent: deque = deque(maxlen=32)
        self._next_watch = 0.0

        self._lock = threading.Lock()
        self._cond = threading.Condition()
        self._source = None
        self._raw: np.ndarray | None = None
        self._seq = 0
        self._fps = 0.0
        self._state = "starting"
        self._error: str | None = None
        self._running = False
        self._thread: threading.Thread | None = None

        self.calibration: Calibration | None = None
        self._load_calibration()

    # -- lifecycle -------------------------------------------------------- #
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop,
                                        name=f"view:{self.name}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None
        with self._lock:
            if self._source is not None:
                try:
                    self._source.release()
                except Exception:
                    pass
                self._source = None
        with self._cond:
            self._cond.notify_all()

    def _loop(self) -> None:
        last = time.time()
        while self._running:
            try:
                with self._lock:
                    source = self._source
                if source is None:
                    try:
                        source = self._open_source(self.cfg)
                    except Exception as exc:
                        source = None
                        self._error = f"{type(exc).__name__}: {exc}"
                    if source is None:
                        self._state = "no camera"
                        self._error = self._error or \
                            f"cannot open camera {self.cfg.source!r}"
                        time.sleep(2.0)
                        continue
                    with self._lock:
                        self._source = source
                    self._error = None

                frame = source.read()
                if frame is None:
                    self._state = "no signal"
                    with self._lock:
                        self._source = None
                    try:
                        source.release()
                    except Exception:
                        pass
                    time.sleep(1.0)
                    continue

                now = time.time()
                self._fps = 0.9 * self._fps + 0.1 / max(now - last, 1e-6)
                last = now
                self._state = "live"
                with self._cond:
                    self._raw = frame
                    self._seq += 1
                    self._cond.notify_all()
                self._watch(frame)
            except Exception as exc:               # never let this thread die
                self._state = "error"
                self._error = f"{type(exc).__name__}: {exc}"
                time.sleep(1.0)

    # -- watching (never scoring) ----------------------------------------- #
    def _ensure_detector(self):
        if self.calibration is None:
            self.detector = None
            return None
        if self.detector is None or self.detector.calib is not self.calibration:
            d = self._detector_cfg
            kw = {} if d is None else {
                "diff_threshold": d.diff_threshold, "min_area": d.min_area,
                "max_area": d.max_area, "settle_frames": d.settle_frames,
                "motion_threshold": d.motion_threshold,
                "learn_frames": d.learn_frames,
                "min_elongation": d.min_elongation,
            }
            # Always "centre" here: which end of the blob is the point gets
            # decided by where the two views cross, not by a rule about where
            # this camera happens to be mounted.
            self.detector = DartDetector(self.calibration, tip_mode="centre", **kw)
            self._observations.clear()
        return self.detector

    def _watch(self, frame) -> None:
        detector = self._ensure_detector()
        if detector is None:
            return
        # Rate-limited on purpose. A dart stays put for the best part of a
        # second, so watching at the full frame rate finds nothing extra and
        # takes the CPU the scoring camera needs.
        now = time.monotonic()
        rate = max(int(getattr(self._watch_cfg, "watch_fps", 8) or 0), 1)
        if now < self._next_watch:
            return
        self._next_watch = max(now, self._next_watch) + 1.0 / rate
        try:
            result = detector.update(frame)
            for dart in result.darts:
                self._observations.append((now, dart))
            gray = detector._prev
        except Exception as exc:
            self._error = f"watch failed: {type(exc).__name__}: {exc}"
            return
        with self._cond:
            self._recent.append((now, gray))
            self._cond.notify_all()

    def observation_near(self, when: float, window: float):
        """A dart this view settled on around ``when``, if there was one."""
        best, best_gap = None, window
        for stamp, dart in list(self._observations):
            gap = abs(stamp - when)
            if gap <= best_gap:
                best, best_gap = dart, gap
        return best

    def measure_at(self, when: float, window: float = 1.5, wait: float = 0.5):
        """Measure this view at the moment the scoring camera saw the dart.

        The two cameras settle independently and their exposure-to-frame delays
        differ by tens of milliseconds, so rather than demanding they agree on
        timing, this walks the recent frames for the stillest one near `when`.
        A dart stays put for the best part of a second, so any quiet frame in
        that window is looking at the same thing.
        """
        detector = self._ensure_detector()
        if detector is None or detector.reference is None:
            return None

        # This camera watches at a few frames a second, so when the scoring
        # camera settles first there may be nothing here from after the dart
        # landed yet. Wait briefly for one rather than answering from frames
        # that predate it - the dart is not going anywhere, and this is the
        # only place fusion costs any time at all.
        deadline = time.monotonic() + max(wait, 0.0)
        with self._cond:
            while time.monotonic() < deadline and                     not any(t >= when for t, _ in self._recent):
                if not self._running:
                    break
                self._cond.wait(0.05)
            frames = [(t, g) for t, g in self._recent if abs(t - when) <= window]
        if not frames:
            return None

        scored, previous = [], None
        for stamp, gray in frames:
            motion = 0.0
            if gray is not None and previous is not None and \
                    gray.shape == previous.shape:
                changed = cv2.absdiff(gray, previous)
                _, changed = cv2.threshold(changed, detector.diff_threshold,
                                           255, cv2.THRESH_BINARY)
                motion = float(cv2.countNonZero(changed))
            # Frames from *before* the scoring camera saw the dart are the
            # quietest ones in the buffer, and they do not contain it - sorting
            # on stillness alone picks exactly the frames that cannot answer.
            # So the dart having landed comes first, stillness second.
            scored.append((stamp < when - 0.25,
                           motion > detector.motion_threshold,
                           abs(stamp - when), gray))
            previous = gray
        scored.sort(key=lambda row: row[:3])
        for _, _, _, gray in scored[:6]:
            dart = detector.measure_gray(gray)
            if dart is not None:
                return dart
        return None

    def note_scored(self, when: float) -> None:
        """The scoring camera accepted a dart; move past it here too."""
        detector = self.detector
        if detector is None:
            return
        with self._cond:
            after = [g for t, g in self._recent if t >= when]
        if after:
            # Move this view's reference past the dart too, so the next throw
            # is measured against a board that already has this one in it.
            try:
                detector._last_stable = after[-1]
            except Exception:
                pass
        self._observations.clear()

    # -- frames ----------------------------------------------------------- #
    def latest(self):
        with self._cond:
            return self._raw

    def _encode(self, frame):
        if frame is None:
            return None
        scale = getattr(self.cfg, "stream_scale", 1.0) or 1.0
        if scale != 1.0:
            frame = cv2.resize(frame, None, fx=scale, fy=scale)
        quality = int(getattr(self.cfg, "stream_quality", 75) or 75)
        ok, buf = cv2.imencode(".jpg", frame,
                               [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return buf.tobytes() if ok else None

    def _annotate(self, frame):
        """The board's rings, wires and numbers drawn over the picture.

        The same overlay the scoring camera gets, and for the same reason: it
        is the only way to see whether a calibration actually landed on the
        board. Drawn on the live stream, not just on a still, because that is
        where anyone setting a camera up is looking.
        """
        calib = self.calibration
        if calib is None:
            return frame
        view = frame.copy()
        render.draw_board_overlay(view, calib, (0, 200, 255), 1)
        label = f"{self.name.upper()}  {self._fps:.0f} fps"
        for colour, weight in (((0, 0, 0), 4), ((255, 255, 255), 1)):
            cv2.putText(view, label, (14, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, colour, weight, cv2.LINE_AA)
        return view

    def snapshot(self, annotated: bool = True):
        frame = self.latest()
        if frame is None:
            return None
        return self._encode(self._annotate(frame) if annotated else frame)

    def rectified(self):
        """The board warped flat - the quickest check a calibration is right."""
        frame = self.latest()
        if frame is None or self.calibration is None:
            return None
        flat = cv2.resize(self.calibration.warp(frame), (480, 480))
        return self._encode(flat)

    def stream(self, fps_cap: int | None = None):
        """JPEG frames for an <img multipart/x-mixed-replace>.

        Encoded on demand rather than continuously: a secondary is only watched
        while someone is setting it up, so there is no point spending a Pi's CPU
        compressing frames nobody is looking at.
        """
        cap = fps_cap if fps_cap is not None else \
            int(getattr(self.cfg, "stream_fps", 20) or 0)
        interval = 1.0 / cap if cap else 0.0
        seen, next_at = -1, 0.0
        while self._running:
            with self._cond:
                if self._seq == seen:
                    self._cond.wait(2.0)
                    if self._seq == seen:
                        continue
                seen = self._seq
                frame = self._raw
            now = time.monotonic()
            if interval and now < next_at:
                continue
            next_at = max(now, next_at) + interval
            data = self._encode(self._annotate(frame))
            if data:
                yield data

    # -- calibration ------------------------------------------------------ #
    def _load_calibration(self) -> None:
        path = Path(self.calibration_path)
        if not path.exists():
            return
        try:
            self.calibration = Calibration.load(path)
        except Exception as exc:
            self._error = f"could not load {path}: {exc}"

    def set_calibration(self, points, save: bool = True) -> dict:
        frame = self.latest()
        size = None if frame is None else frame.shape[1::-1]
        calib = Calibration.from_points(points, size)
        self.calibration = calib
        self.detector = None                   # rebuilt against the new one
        if save:
            calib.save(self.calibration_path)
        return {"points": calib.image_points, "saved": bool(save)}

    def clear_calibration(self) -> None:
        self.calibration = None
        self.detector = None
        Path(self.calibration_path).unlink(missing_ok=True)

    # -- reporting -------------------------------------------------------- #
    def info(self) -> dict:
        frame = self.latest()
        measured = None
        if self.calibration is not None:
            try:
                measured = measure_pose(self.calibration)
            except Exception:
                measured = None
        return {
            "open": self._source is not None,
            "state": self._state,
            "error": self._error,
            "fps": round(self._fps, 1),
            "frame_size": list(frame.shape[1::-1]) if frame is not None else None,
            "calibrated": self.calibration is not None,
            # Calibrated is not the same as ready: this view also has to have
            # learned what the empty board looks like before it can tell a dart
            # from the wall behind it.
            "watching": bool(self.detector is not None
                             and self.detector.reference is not None),
            "calibration_points": (self.calibration.image_points
                                   if self.calibration else []),
            "measured": measured,
        }


class ViewManager:
    """The set of secondary views, kept in step with the configuration."""

    def __init__(self, config, open_source) -> None:
        self.config = config
        self._open_source = open_source
        self.detector_cfg = config.detector
        self._views: dict[str, SecondaryView] = {}
        self._lock = threading.Lock()

    def calibration_path_for(self, cfg) -> str:
        """Where this view's landmarks live.

        Beside the primary's calibration, not beside the working directory:
        deployed, the primary's is in /var/lib/dart-scorer while the working
        directory is /opt/dart-scorer, which the service cannot write to at all
        under ProtectSystem=strict. A bare relative name would fail to save
        there and, worse, would be picked up from whatever directory the
        process happened to start in.
        """
        if cfg.calibration_path:
            return cfg.calibration_path
        primary = Path(self.config.calibration_path)
        return str(primary.with_name(f"{primary.stem}.{cfg.name}{primary.suffix}"))

    def sync(self) -> None:
        """Open, close and reopen views so they match the configuration."""
        with self._lock:
            wanted = {v.name: v for v in self.config.views}
            for name in list(self._views):
                if name not in wanted:
                    self._views.pop(name).stop()
            for name, cfg in wanted.items():
                existing = self._views.get(name)
                if existing is not None and _same_capture(existing.cfg, cfg):
                    existing.cfg = cfg          # cheap settings, no reopen
                    continue
                if existing is not None:
                    existing.stop()
                view = SecondaryView(cfg, self.calibration_path_for(cfg),
                                     self._open_source, self.detector_cfg)
                view._watch_cfg = self.config.fusion
                view.start()
                self._views[name] = view

    def get(self, name: str) -> SecondaryView | None:
        with self._lock:
            return self._views.get(name)

    def info(self) -> dict:
        with self._lock:
            return {name: view.info() for name, view in self._views.items()}

    def stop_all(self) -> None:
        with self._lock:
            for view in self._views.values():
                view.stop()
            self._views.clear()


# Changing any of these means the camera has to be opened again; the rest are
# settings that can be picked up in place.
_CAPTURE_FIELDS = ("source", "width", "height", "fourcc", "fps", "backend",
                   "buffer_size", "stream")


def _same_capture(a, b) -> bool:
    if any(getattr(a, f, None) != getattr(b, f, None) for f in _CAPTURE_FIELDS):
        return False
    pa, pb = getattr(a, "placement", None), getattr(b, "placement", None)
    return all(getattr(pa, f, None) == getattr(pb, f, None)
               for f in ("rotate", "flip_h", "flip_v"))
