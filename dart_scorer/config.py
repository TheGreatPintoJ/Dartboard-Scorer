"""Runtime configuration, editable from the web UI and persisted to disk."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

DEFAULT_CONFIG_PATH = "config.json"


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
    stream_fps: int = 0        # 0 = publish every captured frame (smoothest)
    # Only the controls present here are pushed at the camera; anything absent
    # is left exactly as the camera had it.
    controls: dict = field(default_factory=dict)


@dataclass
class DetectorConfig:
    diff_threshold: int = 26
    min_area: int = 220
    max_area: int = 26000
    settle_frames: int = 4
    motion_threshold: int = 120
    learn_frames: int = 25
    min_elongation: float = 2.0
    tip_mode: str = "centre"


@dataclass
class GameConfig:
    players: list[str] = field(default_factory=lambda: ["Player 1"])
    start_score: int = 0       # 0 = free scoring, else 301/501/...
    double_out: bool = True
    auto_end_turn: bool = True  # end the visit when the board is cleared


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    game: GameConfig = field(default_factory=GameConfig)
    calibration_path: str = "calibration.json"
    log_path: str = "throws.csv"

    # ------------------------------------------------------------------ #
    def to_dict(self) -> dict:
        return asdict(self)

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
        sections = {"camera": self.camera, "detector": self.detector, "game": self.game}
        for key, value in (patch or {}).items():
            if key in sections and isinstance(value, dict):
                section = sections[key]
                allowed = {f.name for f in fields(section)}
                for name, raw in value.items():
                    if name not in allowed:
                        continue
                    current = getattr(section, name)
                    new = _coerce(current, raw)
                    if new != current:
                        setattr(section, name, new)
                        changed.append(f"{key}.{name}")
            elif key in ("calibration_path", "log_path") and isinstance(value, str):
                if getattr(self, key) != value:
                    setattr(self, key, value)
                    changed.append(key)
        return changed

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "AppConfig":
        p = Path(path)
        if not p.exists():
            return cls()
        return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))


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
