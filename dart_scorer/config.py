"""Runtime configuration, editable from the web UI and persisted to disk."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path

DEFAULT_CONFIG_PATH = "config.json"


STREAMS = ("rgb", "ir", "depth")
ROTATIONS = (0, 90, 180, 270)


@dataclass
class Placement:
    """Where a camera sits in relation to the board.

    Angles use the board's own convention (see geometry.py): bearing 0 is off to
    the right of the board, 90 straight above it, 180 to the left, 270 below.
    Elevation is the angle up out of the board's plane - 0 means level with the
    face, looking straight across it, 90 means square on.

    These are what you *expect*; the real values are measured from the
    calibration by calibration.measure_pose, and the two being far apart is how
    you find out a camera has been knocked. Distance is the exception: it cannot
    be derived without a focal length, so it is only ever what you type, and is
    used for guidance (a Kinect closer than 800 mm returns holes, not depth).
    """

    bearing_deg: float = 0.0
    elevation_deg: float = 25.0
    distance_mm: float = 1200.0
    # Applied to incoming frames, for a camera that is physically mounted
    # sideways. Changing it moves the board in frame, so it invalidates the
    # calibration exactly like zoom does.
    rotate: int = 0            # 0 | 90 | 180 | 270
    flip_h: bool = False
    flip_v: bool = False


@dataclass
class CameraConfig:
    source: str = "0"          # camera index, video file, URL, or "demo"
    width: int = 1280
    height: int = 720
    # MJPG rather than the usual YUY2 default: uncompressed 720p does not fit
    # down USB 2.0 at 30 fps, so the camera silently drops to 5-10 and the feed
    # looks broken. Blank leaves the driver's choice alone.
    fourcc: str = "MJPG"
    fps: int = 0               # 0 = whatever the camera offers
    backend: str = "auto"      # auto | any | dshow | msmf | v4l2 | avfoundation
    buffer_size: int = 1       # 1 = always read the newest frame, never a queued one
    stream_quality: int = 75   # JPEG quality for the browser stream
    stream_scale: float = 1.0  # shrink the stream to save bandwidth
    # Cap what we push at the browser. A dartboard does not need 60 fps, and
    # an <img multipart/x-mixed-replace> decoding 60 full-size JPEGs a second
    # stalls, batches and tears - which looks exactly like a jittery camera
    # even when the capture and the server are both perfectly smooth.
    # 0 = uncapped, which is only sensible for a camera at 30 fps or below.
    stream_fps: int = 20
    # Only the controls present here are pushed at the camera; anything absent
    # is left exactly as the camera had it.
    controls: dict = field(default_factory=dict)

    # --- per-view identity and geometry ---------------------------------- #
    name: str = "primary"
    # Which of a Kinect's cameras to read. Ignored by an ordinary webcam.
    # rgb and ir are physically different sensors about 25 mm apart, so they do
    # not share a homography: switching means recalibrating.
    stream: str = "rgb"        # rgb | ir | depth
    # Blank means calibration.<name>.json, so views cannot collide.
    calibration_path: str = ""
    placement: Placement = field(default_factory=Placement)


@dataclass
class DetectorConfig:
    diff_threshold: int = 26
    min_area: int = 220
    max_area: int = 26000
    settle_frames: int = 4
    motion_threshold: int = 120
    learn_frames: int = 25
    min_elongation: float = 2.0
    # "auto" works it out from where the calibration says the camera is.
    tip_mode: str = "auto"


@dataclass
class GameConfig:
    players: list[str] = field(default_factory=lambda: ["Player 1"])
    start_score: int = 0       # 0 = free scoring, else 301/501/...
    double_out: bool = True
    auto_end_turn: bool = True  # end the visit when the board is cleared


@dataclass
class FusionConfig:
    """Combining two views of the same dart.

    A single camera measures the end of the dart's *visible* blob, which stands
    proud of the board, so perspective carries it sideways. A second view fixes
    that - but only when the geometry is sound, hence the gates.
    """

    enabled: bool = True
    # Two views' shadows crossing at a shallower angle than this are too
    # ill-conditioned to trust; below it we keep the primary's own answer.
    min_sin_theta: float = 0.42            # sin(25 degrees)
    # If fusing moves the point further than this, something is wrong with the
    # match. Keeping the single-view answer bounds the worst case to "no worse
    # than one camera".
    max_correction_mm: float = 30.0
    # Two views whose own estimates are further apart than this are looking at
    # different darts.
    max_pair_mm: float = 60.0
    min_segment_mm: float = 40.0
    endpoint_tolerance_mm: float = 60.0
    # How far apart two views' settle times may be and still be the same dart.
    match_window: float = 0.8
    # How often a watching camera runs detection. It only has to notice a
    # dart that then sits still for the best part of a second, so running
    # this at the full frame rate buys nothing and costs a great deal -
    # on a Pi 3 it was the difference between 10 fps and 4 on the camera
    # that actually scores.
    watch_fps: int = 8


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    game: GameConfig = field(default_factory=GameConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    # The *additional* views. `camera` above is the primary one, and stays where
    # it is so that /api/config, the --source flag and every existing caller
    # keep working; the UI is handed a uniform list either way (see `views_all`).
    views: list[CameraConfig] = field(default_factory=list)
    calibration_path: str = "calibration.json"
    log_path: str = "throws.csv"

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        return asdict(self)

    def views_all(self) -> list[CameraConfig]:
        """Every view, primary first. The order the UI shows them in."""
        return [self.camera] + list(self.views)

    def view(self, name: str | None) -> CameraConfig | None:
        """Look a view up by name; no name means the primary."""
        if not name:
            return self.camera
        for v in self.views_all():
            if v.name == name:
                return v
        return None

    @classmethod
    def from_dict(cls, data: dict) -> "AppConfig":
        cfg = cls()
        cfg.apply(data)
        return cfg

    def apply(self, patch: dict) -> list[str]:
        """Merge a (possibly partial) dict in.

        Returns the dotted names of the values that actually *changed*
        ("camera.focus", "game.start_score"). The web UI submits the whole form
        every time, and reopening the camera or restarting the game because
        someone nudged an unrelated slider would be maddening - so callers act
        on precisely what moved.
        """
        changed = []
        # Derived from our own fields rather than listed by hand: a hardcoded
        # list silently discards any section added later, which is a miserable
        # thing to debug from the browser side.
        sections = {f.name: getattr(self, f.name) for f in fields(self)
                    if is_dataclass(getattr(self, f.name))}
        for key, value in (patch or {}).items():
            if key in sections and isinstance(value, dict):
                changed += _merge(sections[key], value, key)
            elif key == "views" and isinstance(value, list):
                changed += self._merge_views(value)
            elif key in ("calibration_path", "log_path") and isinstance(value, str):
                if getattr(self, key) != value:
                    setattr(self, key, value)
                    changed.append(key)
        return changed

    def _merge_views(self, patch: list) -> list[str]:
        """Merge a list of extra views, matched by name.

        Matching by name rather than position means the browser can send the
        views in any order, and adding one does not renumber the rest. A view
        given as null (or with ``"remove": true``) is dropped - that is how the
        UI deletes one.
        """
        changed: list[str] = []
        by_name = {v.name: v for v in self.views}
        for entry in patch:
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name") or "").strip()
            if not name or name == self.camera.name:
                continue                      # the primary is patched as `camera`
            if entry.get("remove"):
                if by_name.pop(name, None) is not None:
                    self.views = [v for v in self.views if v.name != name]
                    changed.append(f"views.{name}.remove")
                continue
            view = by_name.get(name)
            if view is None:
                view = CameraConfig(name=name)
                self.views.append(view)
                by_name[name] = view
                changed.append(f"views.{name}.added")
            changed += _merge(view, entry, f"views.{name}")
        return changed

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "AppConfig":
        p = Path(path)
        if not p.exists():
            return cls()
        return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))


def _merge(section, patch: dict, prefix: str) -> list[str]:
    """Merge a dict into one dataclass section, reporting what moved.

    Nested dataclasses (a view's `placement`) recurse, so the dotted names come
    back as e.g. "views.kinect.placement.bearing_deg".
    """
    changed = []
    allowed = {f.name for f in fields(section)}
    for name, raw in (patch or {}).items():
        if name not in allowed:
            continue
        current = getattr(section, name)
        if is_dataclass(current):
            if isinstance(raw, dict):
                changed += _merge(current, raw, f"{prefix}.{name}")
            continue
        new = _coerce(current, raw)
        if name == "stream" and new not in STREAMS:
            continue
        if name == "rotate" and new not in ROTATIONS:
            continue
        if new != current:
            setattr(section, name, new)
            changed.append(f"{prefix}.{name}")
    return changed


def _coerce(current, raw):
    """Keep the declared type - the browser sends everything as strings."""
    if isinstance(current, dict):
        # Merge rather than replace, so setting one camera control does not
        # wipe the rest. An explicit null removes a control (back to "leave
        # whatever the camera had").
        merged = dict(current)
        for key, value in (raw or {}).items():
            if value is None or value == "":
                merged.pop(key, None)
            else:
                merged[key] = float(value)
        return merged
    if isinstance(current, bool):
        if isinstance(raw, str):
            return raw.strip().lower() in ("1", "true", "yes", "on")
        return bool(raw)
    if isinstance(current, int):
        return int(float(raw))
    if isinstance(current, float):
        return float(raw)
    if isinstance(current, list):
        if isinstance(raw, str):
            return [p.strip() for p in raw.split(",") if p.strip()]
        return [str(v) for v in raw]
    return str(raw)
