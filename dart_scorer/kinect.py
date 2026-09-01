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

The library is reached through ctypes rather than libfreenect's own Cython
wrapper. That wrapper has to be compiled against the exact Python and NumPy in
use and is not packaged for either, so on anything current it is a build that
fails. ``libfreenect_sync`` is four C functions with plain integer arguments -
get a frame, set the tilt, stop - so binding it directly costs about sixty
lines, needs no compiler, and cannot break on a NumPy ABI change. Installing is
then just ``apt install libfreenect-dev``.
"""

from __future__ import annotations

import ctypes
import threading
import time
from pathlib import Path

import cv2
import numpy as np

# Kinect v1's own limits, worth stating where they are used rather than in a
# comment somewhere else.
MIN_RANGE_MM = 800
MAX_RANGE_MM = 4000
STREAMS = ("rgb", "ir", "depth")

# libfreenect's own enums, from libfreenect.h. Spelled out because they are the
# only part of its API this needs and they never change.
VIDEO_RGB = 0
VIDEO_IR_8BIT = 2
DEPTH_REGISTERED = 4        # millimetres, already aligned to the colour camera

# The Kinect's three USB interfaces. The camera is the one that matters; the
# motor turning up without it means the 12 V supply is not connected.
USB_VENDOR = "045e"
USB_CAMERA = "02ae"
USB_MOTOR = "02c2"

_SONAMES = ("libfreenect_sync.so.0.5", "libfreenect_sync.so.0",
            "libfreenect_sync.so")


def _load():
    """The sync library, or the reason it could not be loaded."""
    last = "libfreenect is not installed"
    for name in _SONAMES:
        try:
            lib = ctypes.CDLL(name)
        except OSError as exc:
            last = str(exc)
            continue
        ptr = ctypes.POINTER(ctypes.c_void_p)
        stamp = ctypes.POINTER(ctypes.c_uint32)
        lib.freenect_sync_get_video.argtypes = [ptr, stamp, ctypes.c_int, ctypes.c_int]
        lib.freenect_sync_get_video.restype = ctypes.c_int
        lib.freenect_sync_get_depth.argtypes = [ptr, stamp, ctypes.c_int, ctypes.c_int]
        lib.freenect_sync_get_depth.restype = ctypes.c_int
        lib.freenect_sync_set_tilt_degs.argtypes = [ctypes.c_int, ctypes.c_int]
        lib.freenect_sync_set_tilt_degs.restype = ctypes.c_int
        lib.freenect_sync_stop.argtypes = []
        lib.freenect_sync_stop.restype = None
        return lib, None
    return None, last


_lib, FREENECT_ERROR = _load()

# libfreenect is not thread-safe and its sync API keeps one global device
# handle, so every call goes through one lock.
_call_lock = threading.Lock()


def _grab(index: int, stream: str):
    """One frame, as a numpy array. Raises if the device will not give one."""
    if _lib is None:
        raise RuntimeError(unavailable_reason())
    buf = ctypes.c_void_p()
    stamp = ctypes.c_uint32()
    if stream == "depth":
        fn, fmt, shape, dtype = (_lib.freenect_sync_get_depth, DEPTH_REGISTERED,
                                 (480, 640), np.uint16)
    elif stream == "ir":
        fn, fmt, shape, dtype = (_lib.freenect_sync_get_video, VIDEO_IR_8BIT,
                                 (480, 640), np.uint8)
    else:
        fn, fmt, shape, dtype = (_lib.freenect_sync_get_video, VIDEO_RGB,
                                 (480, 640, 3), np.uint8)
    with _call_lock:
        rc = fn(ctypes.byref(buf), ctypes.byref(stamp), int(index), fmt)
        if rc != 0 or not buf:
            raise RuntimeError(f"the Kinect returned no {stream} frame (rc={rc})")
        # Copied out deliberately: libfreenect hands back a pointer into a
        # buffer it reuses on the very next call.
        nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        raw = ctypes.string_at(buf, nbytes)
    return np.frombuffer(raw, dtype=dtype).reshape(shape), int(stamp.value)


def usb_devices() -> list[str]:
    """Kinect cameras on the USB bus, found without libfreenect.

    Whether a Kinect is plugged in should not depend on whether its driver is
    installed - otherwise a missing driver and missing hardware look identical.
    """
    root = Path("/sys/bus/usb/devices")
    if not root.is_dir():
        return []
    found = []
    for dev in sorted(root.iterdir()):
        try:
            if (dev / "idVendor").read_text().strip() == USB_VENDOR and \
                    (dev / "idProduct").read_text().strip() == USB_CAMERA:
                found.append(dev.name)
        except OSError:
            continue
    return found


def usb_motors() -> int:
    root = Path("/sys/bus/usb/devices")
    if not root.is_dir():
        return 0
    count = 0
    for dev in sorted(root.iterdir()):
        try:
            if (dev / "idVendor").read_text().strip() == USB_VENDOR and \
                    (dev / "idProduct").read_text().strip() == USB_MOTOR:
                count += 1
        except OSError:
            continue
    return count


def is_kinect_source(source) -> bool:
    """Whether this source string asks for a Kinect."""
    return str(source).strip().lower().startswith("kinect")


def device_index(source) -> int:
    """The device number in "kinect", "kinect:0", "kinect:1"."""
    text = str(source).strip().lower()
    _, _, tail = text.partition(":")
    return int(tail) if tail.strip().isdigit() else 0


def available() -> bool:
    """Whether a Kinect could actually be opened: driver present and hardware on."""
    return _lib is not None and bool(usb_devices())


def unavailable_reason() -> str:
    if _lib is None:
        return (FREENECT_ERROR or "libfreenect is not installed")
    if not usb_devices():
        if usb_motors():
            return ("the Kinect's motor is on the USB bus but its camera is not "
                    "- the 12 V power adapter is not connected")
        return "no Kinect on the USB bus"
    return ""


def kinect_status() -> dict:
    """What the UI needs to decide whether to offer the stream selector."""
    plugged = usb_devices()
    return {"available": available(), "devices": len(plugged),
            "plugged_in": len(plugged), "reason": unavailable_reason(),
            "driver": _lib is not None, "streams": list(STREAMS)}


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
        if _lib is None or not usb_devices():
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
        frame, _ = _grab(self.index, self.stream)
        if self.stream == "rgb":
            return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), None
        if self.stream == "ir":
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR), None
        # Registered depth arrives already aligned to the colour camera and
        # already in millimetres, so there is no 11-bit table to undo here.
        return depth_to_bgr(frame), frame

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
            with _call_lock:
                _lib.freenect_sync_stop()
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
                with _call_lock:
                    _lib.freenect_sync_set_tilt_degs(int(tilt), self.index)
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
