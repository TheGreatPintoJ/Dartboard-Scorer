"""The scoring engine: owns the camera, the detector and the game state.

Runs on its own thread so the web layer never blocks on a camera read, and so
a browser disconnecting cannot interrupt scoring. Everything the HTTP layer
needs is exposed through small, locked accessors.
"""

from __future__ import annotations

import csv
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from . import geometry as geo
from . import render
from .calibration import Calibration, measure_pose, tip_mode_for_bearing
from .camera import GEOMETRY_CONTROLS, open_capture
from .config import AppConfig
from .detector import DartDetector
from .kinect import is_kinect_source, kinect_status
from .session import Session
from .synthetic import DemoSource


class EventHub:
    """Fan-out of scoring events to any number of browser connections."""

    def __init__(self, capacity: int = 200) -> None:
        self._cond = threading.Condition()
        self._events: deque = deque(maxlen=capacity)
        self._seq = 0

    def publish(self, kind: str, **data) -> None:
        with self._cond:
            self._seq += 1
            self._events.append({"seq": self._seq, "type": kind,
                                 "time": round(time.time(), 3), **data})
            self._cond.notify_all()

    def since(self, cursor: int, timeout: float = 25.0):
        """Events newer than `cursor`, waiting up to `timeout` for one."""
        with self._cond:
            if self._seq <= cursor:
                self._cond.wait(timeout)
            return [e for e in self._events if e["seq"] > cursor], self._seq

    @property
    def cursor(self) -> int:
        with self._cond:
            return self._seq


def open_source(camera_cfg):
    """A camera, a video file, a stream URL, or the built-in demo board."""
    source = str(camera_cfg.source).strip()
    if source.lower() in ("demo", "synthetic", "test"):
        return DemoSource(fps=30.0, stream=getattr(camera_cfg, "stream", "rgb"))
    if is_kinect_source(source):
        # Imported here, never at module scope: libfreenect is optional, and a
        # half-finished install must not stop the webcam user from serving.
        from .kinect import open_kinect
        return open_kinect(camera_cfg)
    return open_capture(camera_cfg)


# How far apart two views' bearings should be. Below the minimum their shadows
# on the board cross too shallowly to trust (see fusion.MIN_SIN_THETA); much
# beyond the maximum and they start to disagree about which dart is which.
GOOD_SEPARATION = (55.0, 125.0)
DRIFT_WARN_DEG = 15.0


def _bearing_drift(expected: dict, measured: dict | None) -> float | None:
    """How far the camera actually is from where it was said to be."""
    if not measured:
        return None
    diff = (measured["bearing_deg"] - expected["bearing_deg"] + 180.0) % 360.0 - 180.0
    return round(abs(diff), 1)


def _view_warnings(view, measured: dict | None) -> list[str]:
    out = []
    drift = _bearing_drift(
        {"bearing_deg": view.placement.bearing_deg}, measured)
    if drift is not None and drift > DRIFT_WARN_DEG:
        out.append(
            f"measured bearing is {drift:.0f} deg from the {view.placement.bearing_deg:.0f} "
            "you entered - the camera has moved, or this is not the camera you think")
    if is_kinect_source(view.source) and view.placement.distance_mm < 800:
        out.append("a Kinect v1 cannot focus closer than about 800 mm; below "
                   "that its depth comes back as holes")
    if view.stream in ("ir", "depth") and not is_kinect_source(view.source):
        out.append(f"{view.stream} is a Kinect stream; an ordinary camera ignores it")
    return out


def _fusion_outlook(views: list[dict], cfg) -> dict:
    """Whether these cameras are placed well enough to fuse.

    Worth saying at setup time rather than letting it quietly degrade later:
    two cameras close together produce shadows that cross at a shallow angle,
    and that is the case fusion has to refuse.
    """
    placed = [v for v in views
              if (v.get("measured") or v.get("placement")) is not None]
    if len(placed) < 2:
        return {"ready": False, "reason": "fusion needs a second view"}

    def bearing(v):
        return (v.get("measured") or v["placement"])["bearing_deg"]

    best = None
    for i in range(len(placed)):
        for j in range(i + 1, len(placed)):
            gap = abs((bearing(placed[i]) - bearing(placed[j]) + 180.0) % 360.0 - 180.0)
            pair = {"views": [placed[i]["name"], placed[j]["name"]],
                    "separation_deg": round(gap, 1)}
            if best is None or gap > best["separation_deg"]:
                best = pair
    lo, hi = GOOD_SEPARATION
    gap = best["separation_deg"]
    if gap < lo:
        best["reason"] = (f"{gap:.0f} deg apart is too close - their shadows on the "
                          f"board cross too shallowly to trust. Aim for {lo:.0f}-{hi:.0f}.")
    elif gap > hi:
        best["reason"] = (f"{gap:.0f} deg apart is wide - they will often be looking "
                          "at opposite sides of the same dart.")
    else:
        best["reason"] = f"{gap:.0f} deg apart - good for fusing."
    best["ready"] = lo <= gap <= hi
    return best


class ScoringEngine:
    def __init__(self, config: AppConfig, config_path: str | None = None) -> None:
        self.config = config
        self.config_path = config_path
        self.events = EventHub()

        self._lock = threading.RLock()
        self._frame_cond = threading.Condition()
        self._running = False
        self._thread: threading.Thread | None = None

        self._source = None
        self._reopen = True
        self._raw: np.ndarray | None = None
        self._jpeg: bytes | None = None
        self._jpeg_seq = 0
        self._viewers = 0
        self._last_pull = 0.0
        self._next_publish = 0.0

        self.calibration: Calibration | None = None
        self.detector: DartDetector | None = None
        self.session = Session(list(config.game.players),
                               start_score=config.game.start_score,
                               double_out=config.game.double_out)
        self._tips: list = []
        self._state = "starting"
        self._error: str | None = None
        self._fps = 0.0
        self._log_writer = None
        self._log_file = None

        self._frame_size: tuple[int, int] | None = None
        self._open_error: str | None = None
        self._load_calibration()

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="scoring", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        with self._lock:
            if self._source:
                self._source.release()
                self._source = None
            if self._log_file:
                self._log_file.close()
                self._log_file = None

    # ------------------------------------------------------------------ #
    # calibration
    # ------------------------------------------------------------------ #
    def _load_calibration(self) -> None:
        path = Path(self.config.calibration_path)
        if not path.exists():
            self._state = "no calibration"
            return
        try:
            self.calibration = Calibration.load(path)
            self._rebuild_detector()
        except Exception as exc:                       # corrupt or hand-edited
            self._error = f"could not load {path}: {exc}"
            self._state = "no calibration"

    def _fit_calibration_to_frame(self, size) -> None:
        """Check the calibration actually belongs to the frames arriving.

        The homography is in raw pixel coordinates, so a calibration marked at
        one resolution silently scores every dart in the wrong place at another
        - a 1920x1080 calibration fed 1280x720 frames puts the bull about
        250 mm out, which rejects every throw as off-board with no clue why.
        Nothing used to compare the two. A uniform rescale is recoverable and
        is applied; anything else refuses to score and says so.
        """
        with self._lock:
            calib = self.calibration
            if calib is None:
                return
            try:
                fitted = calib.for_frame_size(size)
            except ValueError as exc:
                self.calibration = None
                self.detector = None
                self._state = "no calibration"
                self._error = str(exc)
                self.events.publish("state", state=self._state, error=self._error)
                return
            if fitted is calib:
                return
            self.calibration = fitted
            self._rebuild_detector()
            note = (f"calibration rescaled from {tuple(calib.frame_size)} to "
                    f"{tuple(size)}; recalibrate at this resolution for best "
                    "accuracy")
            self._error = note
        self.events.publish("calibration", points=fitted.image_points,
                            saved=False, note=note)

    def _rebuild_detector(self) -> None:
        if self.calibration is None:
            self.detector = None
            return
        d = self.config.detector
        tip_mode = d.tip_mode
        if tip_mode == "auto":
            # Which end of the blob is the point depends only on where the
            # camera is, and the calibration already says where that is - so
            # there is nothing here for the operator to get wrong.
            try:
                pose = measure_pose(self.calibration)
                tip_mode = tip_mode_for_bearing(pose["bearing_deg"],
                                                pose["elevation_deg"])
            except Exception:
                # A degenerate homography cannot say where the camera is; the
                # rule that holds for almost every mounting is the safe default.
                tip_mode = "centre"
        self.detector = DartDetector(
            self.calibration,
            diff_threshold=d.diff_threshold,
            min_area=d.min_area,
            max_area=d.max_area,
            settle_frames=d.settle_frames,
            motion_threshold=d.motion_threshold,
            learn_frames=d.learn_frames,
            min_elongation=d.min_elongation,
            tip_mode=tip_mode,
        )
        self._tips.clear()

    def set_calibration(self, points, save: bool = True) -> dict:
        """Install a calibration from clicked image points."""
        with self._lock:
            size = None if self._raw is None else self._raw.shape[1::-1]
            calib = Calibration.from_points(points, size)
            self.calibration = calib
            self._rebuild_detector()
            if save:
                calib.save(self.config.calibration_path)
            self._state = "learning"
        self.events.publish("calibration", points=calib.image_points,
                            saved=bool(save))
        return {"points": calib.image_points, "saved": bool(save)}

    def clear_calibration(self) -> None:
        with self._lock:
            self.calibration = None
            self.detector = None
            self._tips.clear()
            self._state = "no calibration"
            Path(self.config.calibration_path).unlink(missing_ok=True)
        self.events.publish("calibration", points=[], saved=False)

    def auto_points(self) -> list | None:
        """Best-effort marker placement from the board outline."""
        from .calibration import ellipse_reference_guess, fit_board_ellipse

        frame = self.latest_raw()
        if frame is None:
            return None
        ellipse = fit_board_ellipse(frame)
        return None if ellipse is None else ellipse_reference_guess(ellipse)

    # ------------------------------------------------------------------ #
    # configuration
    # ------------------------------------------------------------------ #
    # Changing any of these means tearing the capture down and opening it
    # again; the rest can be pushed at a running camera.
    REOPEN_ON = frozenset({"source", "width", "height", "fourcc", "fps",
                           "backend", "buffer_size", "stream",
                           # How the frame is turned before anything sees it.
                           "placement.rotate", "placement.flip_h",
                           "placement.flip_v"})

    # Changing any of these moves the board within the frame, so the homography
    # stops describing where the board is - the same problem as zoom.
    RECALIBRATE_ON = frozenset({"stream", "placement.rotate",
                                "placement.flip_h", "placement.flip_v"})

    def apply_config(self, patch: dict, save: bool = True) -> dict:
        with self._lock:
            changed = self.config.apply(patch)
            camera = {k.split(".", 1)[1] for k in changed if k.startswith("camera.")}
            if camera & self.REOPEN_ON:
                self._reopen = True
            elif "controls" in camera:
                self._push_controls()
            recalibrate = sorted(
                {k.split(".", 1)[1] for k in changed if k.startswith("camera.")}
                & self.RECALIBRATE_ON)
            if any(k.startswith("detector.") for k in changed):
                self._rebuild_detector()
            if any(k.startswith("game.") for k in changed):
                self._new_game()
            if save and self.config_path:
                self.config.save(self.config_path)
        self.events.publish("config", changed=changed, recalibrate=recalibrate)
        result = self.config.to_dict()
        result["changed"] = changed
        result["recalibrate"] = recalibrate
        return result

    # ------------------------------------------------------------------ #
    # camera controls
    # ------------------------------------------------------------------ #
    def _push_controls(self) -> dict:
        """Send the configured controls to the open camera. Caller holds the lock."""
        source = self._source
        if source is None or not hasattr(source, "apply_controls"):
            return {}
        readback = source.apply_controls(dict(self.config.camera.controls))
        # Focus, exposure and the rest all change how the board looks, so the
        # empty-board reference the detector diffs against is now wrong.
        if self.detector is not None:
            self.detector.reset_background()
            self._tips.clear()
            self._state = "learning"
        return readback

    def set_camera_controls(self, controls: dict, save: bool = True) -> dict:
        """Apply camera controls live, without reopening the camera."""
        geometry_moved = sorted(
            name for name, value in (controls or {}).items()
            if name in GEOMETRY_CONTROLS
            and float(value or 0) != float(self.config.camera.controls.get(name, 0))
        )
        with self._lock:
            self.config.apply({"camera": {"controls": controls}})
            readback = self._push_controls()
            if save and self.config_path:
                self.config.save(self.config_path)
        result = {"controls": readback, "camera": self.camera_info(),
                  "recalibrate": geometry_moved}
        self.events.publish("camera", recalibrate=geometry_moved)
        return result

    def views_info(self) -> dict:
        """Every configured view, for the Cameras tab.

        Each entry carries the placement the operator *expects* alongside the
        one measured from the calibration. Those diverging is the signal that a
        camera has been knocked, or that two identical devices came up in the
        opposite order after a reboot - neither of which is otherwise visible
        until the scores quietly go wrong.
        """
        with self._lock:
            cfg = self.config
            calib = self.calibration
            primary = cfg.camera.name
            source = self._source
            open_now = source is not None

        views = []
        for view in cfg.views_all():
            is_primary = view.name == primary
            # Only the primary is actually opened today; the rest are
            # configuration waiting for the multi-camera runtime.
            measured = None
            if is_primary and calib is not None:
                try:
                    measured = measure_pose(calib)
                except Exception:                      # a degenerate homography
                    measured = None
            expected = {"bearing_deg": view.placement.bearing_deg,
                        "elevation_deg": view.placement.elevation_deg,
                        "distance_mm": view.placement.distance_mm}
            views.append({
                "name": view.name,
                "role": "primary" if is_primary else "secondary",
                "source": view.source,
                "stream": view.stream,
                "kinect": is_kinect_source(view.source),
                "calibrated": bool(is_primary and calib is not None),
                "calibration_path": view.calibration_path
                                    or (cfg.calibration_path if is_primary
                                        else f"calibration.{view.name}.json"),
                "open": bool(is_primary and open_now),
                "placement": {
                    **expected,
                    "rotate": view.placement.rotate,
                    "flip_h": view.placement.flip_h,
                    "flip_v": view.placement.flip_v,
                },
                "measured": measured,
                "drift_deg": _bearing_drift(expected, measured),
                "warnings": _view_warnings(view, measured),
            })

        return {"views": views, "primary": primary,
                "kinect": kinect_status(),
                "fusion": _fusion_outlook(views, cfg.fusion)}

    def camera_info(self) -> dict:
        """What the camera is actually doing, plus every control it exposes."""
        with self._lock:
            source = self._source
            requested = dict(self.config.camera.controls)
        if source is None or not hasattr(source, "describe"):
            return {"open": False, "demo": isinstance(source, DemoSource),
                    "controls": {}, "actual": {}}
        return {"open": True, "demo": False, "actual": source.describe(),
                "controls": source.read_controls(requested)}

    def _new_game(self) -> None:
        g = self.config.game
        self.session = Session(list(g.players), start_score=g.start_score,
                               double_out=g.double_out)
        self._tips.clear()

    # ------------------------------------------------------------------ #
    # commands
    # ------------------------------------------------------------------ #
    def command(self, name: str, **kwargs) -> dict:
        with self._lock:
            if name == "undo":
                self.session.undo_dart()
                if self._tips:
                    self._tips.pop()
            elif name == "end_turn":
                self.session.end_turn()
                self._tips.clear()
            elif name == "new_game":
                self._new_game()
            elif name == "relearn":
                if self.detector:
                    # Learn from the frames that follow, not from the last one
                    # captured - that may still show darts being pulled out.
                    self.detector.reset_background()
                    self._tips.clear()
                    self._state = "learning"
            elif name == "reconnect":
                self._reopen = True
            elif name == "throw" and isinstance(self._source, DemoSource):
                self._source.throw(str(kwargs.get("target", "T20")))
            elif name == "pull_darts" and isinstance(self._source, DemoSource):
                self._source.clear()
            else:
                raise ValueError(f"unknown command {name!r}")
        self.events.publish("command", command=name)
        return self.status()

    # ------------------------------------------------------------------ #
    # frames
    # ------------------------------------------------------------------ #
    def latest_raw(self) -> np.ndarray | None:
        with self._lock:
            return None if self._raw is None else self._raw.copy()

    def _encode(self, frame) -> bytes | None:
        cam = self.config.camera
        if cam.stream_scale and abs(cam.stream_scale - 1.0) > 1e-3:
            frame = cv2.resize(frame, None, fx=cam.stream_scale, fy=cam.stream_scale)
        ok, buf = cv2.imencode(".jpg", frame,
                               [cv2.IMWRITE_JPEG_QUALITY, int(cam.stream_quality)])
        return buf.tobytes() if ok else None

    def snapshot(self, annotated: bool = True) -> bytes | None:
        """One JPEG, encoded on demand."""
        self._last_pull = time.time()
        frame = self.latest_raw()
        if frame is None:
            return None
        if annotated:
            frame = self._annotate(frame)
        return self._encode(frame)

    def preview(self, points) -> bytes | None:
        """Render the board overlay for candidate calibration points."""
        frame = self.latest_raw()
        if frame is None:
            return None
        try:
            calib = Calibration.from_points(points, frame.shape[1::-1])
        except Exception:
            return None
        render.draw_board_overlay(frame, calib, (0, 220, 255), 2)
        for i, p in enumerate(points, 1):
            cv2.drawMarker(frame, (int(p[0]), int(p[1])), (0, 255, 255),
                           cv2.MARKER_CROSS, 20, 2)
            cv2.putText(frame, str(i), (int(p[0]) + 10, int(p[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        return self._encode(frame)

    def rectified(self) -> bytes | None:
        """The board warped flat - the quickest check that a calibration is good."""
        frame = self.latest_raw()
        with self._lock:
            calib = self.calibration
        if frame is None or calib is None:
            return None
        return self._encode(cv2.resize(calib.warp(frame), (480, 480)))

    def stream(self, timeout: float = 10.0):
        """Yield annotated JPEG frames as they arrive."""
        with self._frame_cond:
            self._viewers += 1
        try:
            seq = -1
            while self._running:
                with self._frame_cond:
                    if self._jpeg_seq == seq:
                        self._frame_cond.wait(timeout)
                    if self._jpeg is None or self._jpeg_seq == seq:
                        continue
                    seq = self._jpeg_seq
                    frame = self._jpeg
                yield frame
        finally:
            with self._frame_cond:
                self._viewers -= 1

    def _annotate(self, frame) -> np.ndarray:
        with self._lock:
            calib, tips, state, fps = self.calibration, list(self._tips), self._state, self._fps
        view = frame.copy()
        if calib is not None:
            render.draw_board_overlay(view, calib, (0, 200, 255), 1)
            for i, dart in enumerate(tips, 1):
                colour = (0, 255, 120) if dart.confidence > 0.6 else (0, 180, 255)
                render.draw_marker(view, dart.tip_image, f"{i}: {dart.label}", colour)
        cv2.putText(view, f"{state.upper()}  {fps:.0f} fps", (14, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(view, f"{state.upper()}  {fps:.0f} fps", (14, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        return view

    # ------------------------------------------------------------------ #
    # status
    # ------------------------------------------------------------------ #
    def status(self) -> dict:
        with self._lock:
            s = self.session
            visit = [{"label": d.label, "points": d.points,
                      "confidence": d.confidence,
                      "near_wire": d.score.near_wire,
                      "radius_mm": round(geo.radius_of(*d.board_mm), 1),
                      "board_mm": [round(d.board_mm[0], 1), round(d.board_mm[1], 1)],
                      "tip": [round(d.tip_image[0], 1), round(d.tip_image[1], 1)]}
                     for d in s.turn.darts]
            demo = isinstance(self._source, DemoSource)
            return {
                "state": self._state,
                "error": self._error,
                "fps": round(self._fps, 1),
                "calibrated": self.calibration is not None,
                "calibration_points": self.calibration.image_points if self.calibration else [],
                "frame_size": list(self._raw.shape[1::-1]) if self._raw is not None else None,
                "demo": demo,
                "demo_darts": list(self._source.darts) if demo else [],
                "config": self.config.to_dict(),
                "cursor": self.events.cursor,
                "session": {
                    "players": [{"name": p.name, "remaining": p.remaining,
                                 "darts": p.darts_thrown} for p in s.players],
                    "current": s.current,
                    "visit": visit,
                    "visit_total": s.turn.points,
                    "busted": s.turn.busted,
                    "complete": s.turn_complete,
                    "messages": list(s.messages),
                    "game": s.start_score,
                },
            }

    # ------------------------------------------------------------------ #
    # the loop
    # ------------------------------------------------------------------ #
    def _log(self, dart) -> None:
        if not self.config.log_path:
            return
        if self._log_writer is None:
            path = Path(self.config.log_path)
            new = not path.exists()
            self._log_file = open(path, "a", newline="", encoding="utf-8")
            self._log_writer = csv.writer(self._log_file)
            if new:
                self._log_writer.writerow(
                    ["timestamp", "player", "label", "points", "x_mm", "y_mm",
                     "radius_mm", "angle_deg", "confidence"])
        self._log_writer.writerow([
            f"{time.time():.3f}", self.session.player.name, dart.label, dart.points,
            f"{dart.board_mm[0]:.1f}", f"{dart.board_mm[1]:.1f}",
            f"{geo.radius_of(*dart.board_mm):.1f}", f"{dart.score.angle_deg:.1f}",
            dart.confidence])
        self._log_file.flush()

    def _publish_frame(self, frame) -> None:
        now = time.time()
        if self._viewers <= 0 and now - self._last_pull > 5.0:
            return                                   # nobody is watching
        # Detection still runs on every captured frame; only what the browser
        # is asked to decode is capped. Past ~25 fps the browser's MJPEG path
        # is the bottleneck, not the camera or the encoder.
        cap = self.config.camera.stream_fps
        if cap:
            interval = 1.0 / cap
            # Aim at a deadline that advances by a fixed step, and accept a
            # frame that arrives slightly early. Comparing against "when did I
            # last publish" instead makes any timing jitter push the decision
            # onto the following captured frame, so a 60 fps camera capped at
            # 20 delivers a lurching 15.
            if now < self._next_publish - 0.25 * interval:
                return
            self._next_publish = max(now, self._next_publish) + interval
        jpeg = self._encode(self._annotate(frame))
        if jpeg is None:
            return
        with self._frame_cond:
            self._jpeg = jpeg
            self._jpeg_seq += 1
            self._frame_cond.notify_all()

    def _loop(self) -> None:
        # Any exception escaping here used to end scoring for the life of the
        # process while _running stayed true, so the browser sat on a frozen
        # status and /healthz still reported ok. Whatever breaks, say so and
        # keep going.
        while self._running:
            try:
                self._scoring_loop()
            except Exception as exc:                       # pragma: no cover
                with self._lock:
                    self._state = "error"
                    self._error = f"{type(exc).__name__}: {exc}"
                    if self._source is not None:
                        try:
                            self._source.release()
                        except Exception:
                            pass
                        self._source = None
                self.events.publish("state", state="error", error=self._error)
                time.sleep(1.0)

    def _scoring_loop(self) -> None:
        last = time.time()
        while self._running:
            with self._lock:
                need_source = self._source is None or self._reopen
            if need_source:
                with self._lock:
                    if self._source:
                        self._source.release()
                    try:
                        self._source = open_source(self.config.camera)
                    except Exception as exc:
                        # A backend that raises rather than returning None used
                        # to kill this thread outright: _running stayed true, so
                        # the browser saw a frozen status and /healthz still
                        # said ok, forever. Treat it as "no camera" and retry.
                        self._source = None
                        self._open_error = f"{type(exc).__name__}: {exc}"
                    self._reopen = False
                    if self._source is None:
                        self._state = "no camera"
                        detail = getattr(self, "_open_error", None)
                        self._error = (
                            f"cannot open camera {self.config.camera.source!r}"
                            + (f": {detail}" if detail else ""))
                        self._open_error = None
                    else:
                        self._error = None
                        self._state = "learning" if self.calibration else "no calibration"
                        if self.detector:
                            self.detector.reset_background()
                        if self.config.camera.controls:
                            self._push_controls()
                if self._source is None:
                    self.events.publish("state", state="no camera", error=self._error)
                    time.sleep(2.0)
                    continue

            frame = self._source.read()
            if frame is None:
                with self._lock:
                    self._state = "no signal"
                    self._source.release()
                    self._source = None
                self.events.publish("state", state="no signal")
                time.sleep(1.0)
                continue

            now = time.time()
            self._fps = 0.9 * self._fps + 0.1 / max(now - last, 1e-6)
            last = now

            size = (frame.shape[1], frame.shape[0])
            if size != self._frame_size:
                self._frame_size = size
                self._fit_calibration_to_frame(size)

            with self._lock:
                self._raw = frame
                detector, calib = self.detector, self.calibration

            if detector is not None and calib is not None:
                result = detector.update(frame)
                with self._lock:
                    previous, self._state = self._state, result.state.value
                for dart in result.darts:
                    with self._lock:
                        self.session.add_dart(dart)
                        self._tips.append(dart)
                        self._log(dart)
                        message = self.session.messages[-1] if self.session.messages else ""
                    self.events.publish(
                        "dart", label=dart.label, points=dart.points,
                        confidence=dart.confidence, near_wire=dart.score.near_wire,
                        board_mm=[round(v, 1) for v in dart.board_mm],
                        message=message, status=self.status())
                if result.cleared:
                    with self._lock:
                        scored = self.session.turn.points
                        had = bool(self.session.turn.darts)
                        if had and self.config.game.auto_end_turn:
                            self.session.end_turn()
                        self._tips.clear()
                    if had:
                        self.events.publish("cleared", scored=scored,
                                            status=self.status())
                if previous != self._state:
                    self.events.publish("state", state=self._state)

            self._publish_frame(frame)
