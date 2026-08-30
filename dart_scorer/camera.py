"""Opening a camera and driving its controls.

Three things here matter far more than they look:

* **Pixel format.** Most USB webcams offer uncompressed YUY2 and compressed
  MJPG. OpenCV takes whatever the driver offers first, which is usually YUY2 -
  and 1280x720 YUY2 does not fit down USB 2.0 at 30 fps, so the camera quietly
  drops to 5-10 fps. Asking for MJPG is normally the difference between a
  stuttering feed and a smooth one.
* **Buffer depth.** With the default queue, a consumer that falls behind reads
  older and older frames, which shows up as lag followed by bursts. Depth 1
  means we always get the newest frame.
* **Autofocus and auto-exposure.** Both hunt: the picture pulses and drifts,
  which looks like jitter and, worse, moves the background reference the
  detector diffs against. For scoring, both should be off and fixed.
"""

from __future__ import annotations

import re
import subprocess
import sys
import threading
import time

import cv2

# Controls exposed to the UI, in the order they are shown.
CONTROLS: dict[str, int] = {
    "autofocus": cv2.CAP_PROP_AUTOFOCUS,
    "focus": cv2.CAP_PROP_FOCUS,
    "zoom": cv2.CAP_PROP_ZOOM,
    "auto_exposure": cv2.CAP_PROP_AUTO_EXPOSURE,
    "exposure": cv2.CAP_PROP_EXPOSURE,
    "gain": cv2.CAP_PROP_GAIN,
    "brightness": cv2.CAP_PROP_BRIGHTNESS,
    "contrast": cv2.CAP_PROP_CONTRAST,
    "saturation": cv2.CAP_PROP_SATURATION,
    "sharpness": cv2.CAP_PROP_SHARPNESS,
    "auto_wb": cv2.CAP_PROP_AUTO_WB,
    "wb_temperature": cv2.CAP_PROP_WB_TEMPERATURE,
    "pan": cv2.CAP_PROP_PAN,
    "tilt": cv2.CAP_PROP_TILT,
    "backlight": cv2.CAP_PROP_BACKLIGHT,
}

# Changing any of these moves the board within the frame, so the homography no
# longer describes where the board is.
GEOMETRY_CONTROLS = frozenset({"zoom", "pan", "tilt"})

BACKENDS: dict[str, int] = {
    "any": cv2.CAP_ANY,
    "dshow": getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY),
    "msmf": getattr(cv2, "CAP_MSMF", cv2.CAP_ANY),
    "v4l2": getattr(cv2, "CAP_V4L2", cv2.CAP_ANY),
    "avfoundation": getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY),
    "gstreamer": getattr(cv2, "CAP_GSTREAMER", cv2.CAP_ANY),
    "ffmpeg": getattr(cv2, "CAP_FFMPEG", cv2.CAP_ANY),
}


def default_backend() -> str:
    """DirectShow beats Media Foundation on Windows for webcam latency; V4L2 is
    the sane default on Linux."""
    if sys.platform.startswith("win"):
        return "dshow"
    if sys.platform.startswith("linux"):
        return "v4l2"
    if sys.platform == "darwin":
        return "avfoundation"
    return "any"


def open_capture(cfg):
    """Open the configured camera, or None if it will not open.

    Order matters: several drivers only honour a resolution change after the
    pixel format is settled, and only honour the format while stopped.
    """
    source = str(cfg.source).strip()
    is_index = source.isdigit()

    if is_index:
        name = (cfg.backend or "auto").lower()
        if name in ("", "auto"):
            name = default_backend()
        cap = cv2.VideoCapture(int(source), BACKENDS.get(name, cv2.CAP_ANY))
    else:
        # Files and URLs: let OpenCV choose, a webcam backend would refuse.
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        cap.release()
        return None

    if is_index and cfg.fourcc:
        code = cfg.fourcc.upper()[:4].ljust(4)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*code))
    if cfg.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
    if cfg.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
    if cfg.fps:
        cap.set(cv2.CAP_PROP_FPS, cfg.fps)
    if cfg.buffer_size:
        # Best effort - not every backend implements it.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, cfg.buffer_size)

    # A file is not live: read it at the pace the caller asks for.
    live = is_index or "://" in source
    return CameraSource(cap, source, drain=live,
                        index=int(source) if is_index else None)


# OpenCV can set a control but cannot say what range it accepts, so the web UI
# would be reduced to "type a number and see what happens". On Linux v4l2-ctl
# knows the real min/max/step, which turns those boxes into sliders. The names
# differ between kernel versions, hence the alternatives.
V4L2_NAMES: dict[str, tuple[str, ...]] = {
    "autofocus": ("focus_automatic_continuous", "focus_auto"),
    "focus": ("focus_absolute",),
    "zoom": ("zoom_absolute",),
    "auto_exposure": ("auto_exposure", "exposure_auto"),
    "exposure": ("exposure_time_absolute", "exposure_absolute"),
    "gain": ("gain",),
    "brightness": ("brightness",),
    "contrast": ("contrast",),
    "saturation": ("saturation",),
    "sharpness": ("sharpness",),
    "auto_wb": ("white_balance_automatic", "white_balance_temperature_auto"),
    "wb_temperature": ("white_balance_temperature",),
    "pan": ("pan_absolute",),
    "tilt": ("tilt_absolute",),
    "backlight": ("backlight_compensation",),
}

_CONTROL_LINE = re.compile(
    r"^\s*(?P<name>\w+)\s+0x[0-9a-fA-F]+\s+\((?P<kind>\w+)\)\s*:\s*(?P<rest>.*)$")


def v4l2_ranges(index: int, timeout: float = 2.0) -> dict:
    """min/max/step/default per control, from v4l2-ctl. {} when unavailable."""
    if not sys.platform.startswith("linux"):
        return {}
    try:
        out = subprocess.run(
            ["v4l2-ctl", "-d", f"/dev/video{index}", "--list-ctrls"],
            capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return {}                                  # v4l2-utils not installed
    if out.returncode != 0:
        return {}
    return parse_v4l2_controls(out.stdout)


def parse_v4l2_controls(text: str) -> dict:
    """Pull min/max/step/default out of `v4l2-ctl --list-ctrls` output."""
    found = {}
    for line in text.splitlines():
        match = _CONTROL_LINE.match(line)
        if not match:
            continue
        fields = dict(part.split("=", 1) for part in match.group("rest").split()
                      if "=" in part)
        entry = {"kind": match.group("kind")}
        for key in ("min", "max", "step", "default"):
            if key in fields:
                try:
                    entry[key] = float(fields[key])
                except ValueError:
                    pass
        found[match.group("name")] = entry

    ranges = {}
    for ours, candidates in V4L2_NAMES.items():
        for name in candidates:
            if name in found and "min" in found[name] and "max" in found[name]:
                ranges[ours] = found[name]
                break
    return ranges


def fourcc_name(value: float) -> str:
    code = int(value)
    if code <= 0:
        return ""
    return "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4)).strip()


class CameraSource:
    """A cv2.VideoCapture plus the controls the web UI drives.

    For a live camera the frames are pulled by a thread of their own. Not for
    speed - so that the driver's queue is always drained. CAP_PROP_BUFFERSIZE
    is ignored by several backends, and when the consumer is slower than the
    camera the queue backs up: the picture lags, then catches up in a burst.
    Draining continuously and keeping only the newest frame makes that
    impossible whether or not the backend honours the buffer setting.

    Video files are read straight, with no drain thread - racing through a file
    as fast as it decodes is not what anyone wants.
    """

    def __init__(self, cap, name: str, drain: bool = False,
                 index: int | None = None) -> None:
        self._cap = cap
        self.name = name
        self._index = index
        self._ranges: dict | None = None        # looked up once, then cached
        self._cap_lock = threading.Lock()      # VideoCapture is not thread-safe
        self._latest = None
        self._latest_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        if drain:
            self._thread = threading.Thread(target=self._drain, name="capture",
                                            daemon=True)
            self._thread.start()

    def _drain(self) -> None:
        while not self._stop.is_set():
            with self._cap_lock:
                ok, frame = self._cap.read()
            if not ok:
                self._stop.wait(0.05)
                continue
            with self._latest_lock:
                self._latest = frame

    def read(self):
        if self._thread is None:
            with self._cap_lock:
                ok, frame = self._cap.read()
            return frame if ok else None
        # Hand back each frame once, so the caller keeps the camera's own
        # pacing instead of spinning on a frame it has already scored.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not self._stop.is_set():
            with self._latest_lock:
                frame, self._latest = self._latest, None
            if frame is not None:
                return frame
            time.sleep(0.002)
        return None                            # camera has gone away

    def release(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        with self._cap_lock:
            self._cap.release()

    # ------------------------------------------------------------------ #
    def apply_controls(self, controls: dict) -> dict:
        """Push control values at the camera and report what it actually took.

        Cameras routinely accept a value and then ignore it, or clamp it to
        their own range, so every value is read back rather than assumed. Auto
        modes are applied first: most drivers refuse a manual focus or exposure
        while the corresponding auto mode is still on.
        """
        ordered = sorted(controls.items(),
                         key=lambda kv: not kv[0].startswith(("auto", "autofocus")))
        for name, value in ordered:
            prop = CONTROLS.get(name)
            if prop is None or value is None:
                continue
            try:
                with self._cap_lock:
                    self._cap.set(prop, float(value))
            except Exception:
                pass
        return self.read_controls(controls)

    def read_controls(self, requested: dict | None = None) -> dict:
        """Current value of every control, with what was asked for alongside."""
        requested = requested or {}
        if self._ranges is None:
            self._ranges = v4l2_ranges(self._index) if self._index is not None else {}
        out = {}
        for name, prop in CONTROLS.items():
            try:
                with self._cap_lock:
                    actual = self._cap.get(prop)
            except Exception:
                actual = -1.0
            out[name] = {
                "actual": None if actual is None else round(float(actual), 3),
                "requested": requested.get(name),
                # A property the backend does not implement reads back as -1.
                "supported": actual is not None and float(actual) != -1.0,
                "range": self._ranges.get(name),
            }
        return out

    def describe(self) -> dict:
        """What the camera is really doing, as opposed to what we asked for."""
        with self._cap_lock:
            get = self._cap.get
            return {
                "width": int(get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
                "height": int(get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
                "fps": round(float(get(cv2.CAP_PROP_FPS) or 0), 1),
                "fourcc": fourcc_name(get(cv2.CAP_PROP_FOURCC) or 0),
                "buffer_size": int(get(cv2.CAP_PROP_BUFFERSIZE) or 0),
                "backend": self._cap.getBackendName(),
            }
