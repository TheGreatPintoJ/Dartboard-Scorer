"""Kinect v1 as a camera, when one is plugged in.

A Kinect is not one camera but three: a colour camera, an infrared camera, and
the depth map the infrared one produces. Which of them is the right input
depends on the room - colour is the most detailed, infrared is lit by the
Kinect's own emitter so it does not care whether the lights are on, and depth
sees the dart standing out of the board rather than merely looking different
from it.

Two things to know before switching between them:

* **Colour and infrared are different sensors**, about 25 mm apart, so they do
  not share a homography. Changing the stream means calibrating again; the
  engine flags that the same way it flags a zoom change.
* **libfreenect is optional.** The project's whole dependency list is OpenCV
  and NumPy, and it stays that way: with no Kinect, no libfreenect, or on
  Windows, everything here reports itself unavailable and the app behaves
  exactly as it always did.
"""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np

# Deliberately broad. The cython wrapper raises OSError when libfreenect.so is
# not on the loader path and ImportError on a numpy ABI mismatch, and a bare
# ImportError guard would take the whole service down on a half-finished
# install - the exact situation this guard exists for.
try:
    import freenect as _freenect
    FREENECT_ERROR: str | None = None
except Exception as exc:                       # pragma: no cover - env specific
    _freenect = None
    FREENECT_ERROR = f"{type(exc).__name__}: {exc}"

# Kinect v1's own limits, worth stating where they are used rather than in a
# comment somewhere else.
MIN_RANGE_MM = 800
MAX_RANGE_MM = 4000
STREAMS = ("rgb", "ir", "depth")


def is_kinect_source(source) -> bool:
    """Whether this source string asks for a Kinect."""
    return str(source).strip().lower().startswith("kinect")


def device_index(source) -> int:
    """The device number in "kinect", "kinect:0", "kinect:1"."""
    text = str(source).strip().lower()
    _, _, tail = text.partition(":")
    return int(tail) if tail.strip().isdigit() else 0


def available() -> bool:
    return _freenect is not None


def unavailable_reason() -> str:
    if _freenect is not None:
        return ""
    return (FREENECT_ERROR or "libfreenect is not installed")


def kinect_status() -> dict:
    """What the UI needs to decide whether to offer the stream selector."""
    if _freenect is None:
        return {"available": False, "devices": 0,
                "reason": unavailable_reason(), "streams": list(STREAMS)}
    try:
        count = int(_freenect.num_devices())
    except Exception as exc:
        return {"available": False, "devices": 0,
                "reason": f"libfreenect is installed but unusable: {exc}",
                "streams": list(STREAMS)}
    reason = "" if count else (
        "no Kinect on the bus. Check `lsusb | grep 045e` - if the motor (02b0) "
        "shows but the camera (02ae) does not, the 12 V adapter is not plugged in")
    return {"available": bool(count), "devices": count, "reason": reason,
            "streams": list(STREAMS)}


def depth_to_bgr(depth, lo: int = MIN_RANGE_MM, hi: int = MAX_RANGE_MM):
    """A depth map as something the rest of the pipeline can treat as a picture.

    Everything downstream expects 3-channel 8-bit, and near-means-bright is the
    reading that makes a dart standing out of the board obvious. Zero means "no
    reading" on a Kinect, not "touching the lens", so it has to stay black
    rather than becoming the brightest thing in frame.
    """
    depth = np.asarray(depth)
    valid = depth > 0
    scaled = np.zeros(depth.shape, np.uint8)
    if valid.any():
        span = max(hi - lo, 1)
        near = np.clip((depth.astype(np.float32) - lo) / span, 0.0, 1.0)
        scaled[valid] = (255 * (1.0 - near[valid])).astype(np.uint8)
    return cv2.cvtColor(scaled, cv2.COLOR_GRAY2BGR)


class KinectSource:
    """One Kinect stream, behind the same duck type as a webcam.

    Frames are pulled by a thread of their own into a single-slot mailbox, the
    same shape as CameraSource, so the consumer always gets the newest frame
    and never a queued one.
    """

    def __init__(self, index: int = 0, stream: str = "rgb", orient=None) -> None:
        if _freenect is None:
            raise RuntimeError(unavailable_reason())
        if stream not in STREAMS:
            raise ValueError(f"{stream!r} is not one of {', '.join(STREAMS)}")
        self.index = index
        self.stream = stream
        self.name = f"kinect:{index}"
        self._orient = orient
        self._latest = None
        self._latest_depth = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._error: str | None = None
        self._thread = threading.Thread(target=self._drain, name="capture",
                                        daemon=True)
        self._thread.start()

    # -- capture ---------------------------------------------------------- #
    def _grab(self):
        """One (picture, depth) pair from the device."""
        if self.stream == "rgb":
            rgb, _ = _freenect.sync_get_video(self.index)
            return cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR), None
        if self.stream == "ir":
            ir, _ = _freenect.sync_get_video(
                self.index, _freenect.VIDEO_IR_8BIT)
            return cv2.cvtColor(np.asarray(ir, np.uint8), cv2.COLOR_GRAY2BGR), None
        # Registered depth comes back already aligned to the colour camera and
        # already in millimetres, so there is no 11-bit table to undo here.
        depth, _ = _freenect.sync_get_depth(
            self.index, _freenect.DEPTH_REGISTERED)
        depth = np.asarray(depth, np.uint16)
        return depth_to_bgr(depth), depth

    def _drain(self) -> None:
        from .camera import orient_frame
        while not self._stop.is_set():
            try:
                frame, depth = self._grab()
                self._error = None
            except Exception as exc:
                # Never let this thread die: a Kinect that drops packets should
                # degrade to "no signal", not take the scoring loop with it.
                self._error = f"{type(exc).__name__}: {exc}"
                self._stop.wait(0.2)
                continue
            with self._lock:
                self._latest = orient_frame(frame, self._orient)
                self._latest_depth = depth

    def read(self):
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not self._stop.is_set():
            with self._lock:
                frame, self._latest = self._latest, None
            if frame is not None:
                return frame
            time.sleep(0.002)
        return None

    def read_pair(self):
        """The picture and, on the depth stream, the millimetres behind it."""
        frame = self.read()
        if frame is None:
            return None
        with self._lock:
            return frame, self._latest_depth

    def release(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        try:
            _freenect.sync_stop()
        except Exception:                              # already gone
            pass

    # -- the bits the web UI asks every source for ------------------------ #
    def describe(self) -> dict:
        rotate = int(getattr(self._orient, "rotate", 0) or 0)
        width, height = (480, 640) if rotate in (90, 270) else (640, 480)
        return {"width": width, "height": height, "fps": 30.0,
                "fourcc": self.stream.upper()[:4], "buffer_size": 1,
                "backend": "libfreenect", "rotate": rotate,
                "stream": self.stream, "error": self._error}

    def apply_controls(self, controls: dict) -> dict:
        """A Kinect exposes no webcam controls, but it does have a motor."""
        tilt = (controls or {}).get("tilt")
        if tilt is not None:
            try:
                _freenect.sync_set_tilt_degs(int(tilt), self.index)
            except Exception as exc:
                self._error = f"tilt failed: {exc}"
        return self.read_controls(controls)

    def read_controls(self, requested: dict | None = None) -> dict:
        requested = requested or {}
        from .camera import CONTROLS
        out = {}
        for name in CONTROLS:
            supported = name == "tilt"
            out[name] = {
                "actual": None, "requested": requested.get(name),
                "supported": supported,
                "range": ({"min": -30, "max": 30, "step": 1, "default": 0}
                          if supported else None),
            }
        return out


def open_kinect(cfg):
    """Open the Kinect named by a config's source, or raise saying why not."""
    return KinectSource(index=device_index(cfg.source),
                        stream=getattr(cfg, "stream", "rgb") or "rgb",
                        orient=getattr(cfg, "placement", None))
