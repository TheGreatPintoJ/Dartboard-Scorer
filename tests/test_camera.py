"""Capture-layer checks, driven by a stand-in for cv2.VideoCapture.

The point of these is the drain thread: it is what stops a slow consumer from
reading a backlog of stale frames, which is the difference between a smooth
feed and one that lags then jumps.
"""

import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2                                          # noqa: E402
import numpy as np                                  # noqa: E402

from dart_scorer.camera import CONTROLS, CameraSource, fourcc_name  # noqa: E402


def check(condition, message):
    assert condition, message


class FakeCapture:
    """Behaves like a webcam: hands out numbered frames at a fixed rate."""

    def __init__(self, interval=0.01, fail_after=None):
        self.interval = interval
        self.fail_after = fail_after
        self.count = 0
        self.props = {}
        self.released = False
        self._lock = threading.Lock()

    def read(self):
        time.sleep(self.interval)
        with self._lock:
            if self.fail_after is not None and self.count >= self.fail_after:
                return False, None
            self.count += 1
            frame = np.full((4, 4, 3), self.count % 256, np.uint8)
        return True, frame

    def get(self, prop):
        return self.props.get(prop, -1.0)

    def set(self, prop, value):
        self.props[prop] = float(value)
        return True

    def getBackendName(self):                       # noqa: N802 - OpenCV's spelling
        return "fake"

    def release(self):
        self.released = True


# --------------------------------------------------------------------------- #
def test_plain_read_without_drain():
    source = CameraSource(FakeCapture(), "test", drain=False)
    check(source.read() is not None, "a frame should come back")
    source.release()


def test_drain_hands_each_frame_out_once():
    """Otherwise a fast consumer would score the same frame repeatedly."""
    cap = FakeCapture(interval=0.02)
    source = CameraSource(cap, "test", drain=True)
    try:
        first = source.read()
        check(first is not None, "the drain thread should produce a frame")
        seen = [int(first[0, 0, 0])]
        for _ in range(4):
            seen.append(int(source.read()[0, 0, 0]))
        check(len(set(seen)) == len(seen), f"frames should all differ, got {seen}")
    finally:
        source.release()


def test_drain_keeps_only_the_newest_frame():
    """A slow consumer must not work through a backlog."""
    cap = FakeCapture(interval=0.002)               # camera much faster than us
    source = CameraSource(cap, "test", drain=True)
    try:
        source.read()
        time.sleep(0.25)                            # fall a long way behind
        with cap._lock:
            produced = cap.count
        served = int(source.read()[0, 0, 0])
        check(abs(served - produced % 256) <= 2,
              f"expected the newest frame (~{produced % 256}), got {served}")
    finally:
        source.release()


def test_dead_camera_reports_no_frame():
    source = CameraSource(FakeCapture(interval=0.005, fail_after=3), "test", drain=True)
    try:
        deadline = time.time() + 4
        while time.time() < deadline and source.read() is not None:
            pass
        check(source.read() is None, "a dead camera should read as no frame")
    finally:
        source.release()


def test_release_stops_the_thread():
    source = CameraSource(FakeCapture(), "test", drain=True)
    source.read()
    source.release()
    time.sleep(0.1)
    check(not source._thread.is_alive(), "the drain thread should stop")
    check(source._cap.released, "the capture should be released")


def test_controls_round_trip_and_report_readback():
    cap = FakeCapture()
    source = CameraSource(cap, "test", drain=False)
    readback = source.apply_controls({"focus": 42, "autofocus": 0})
    check(cap.props[CONTROLS["focus"]] == 42, "focus should reach the camera")
    check(readback["focus"]["actual"] == 42, "the readback should be reported")
    check(readback["focus"]["requested"] == 42, "the request should be echoed")
    check(readback["focus"]["supported"], "a property that reads back is supported")
    # An untouched property reads -1 on this fake, as on a real backend that
    # does not implement it.
    check(not readback["zoom"]["supported"], "zoom should report as unsupported")
    source.release()


def test_auto_modes_are_set_before_manual_values():
    """Most drivers ignore a manual focus while autofocus is still on."""
    order = []
    cap = FakeCapture()
    original = cap.set
    cap.set = lambda prop, value: (order.append(prop), original(prop, value))[1]
    source = CameraSource(cap, "test", drain=False)
    source.apply_controls({"focus": 10, "autofocus": 0,
                           "exposure": -6, "auto_exposure": 1})
    autos = [CONTROLS["autofocus"], CONTROLS["auto_exposure"]]
    manuals = [CONTROLS["focus"], CONTROLS["exposure"]]
    check(max(order.index(p) for p in autos) < min(order.index(p) for p in manuals),
          f"auto modes must be applied first, order was {order}")
    source.release()


def test_describe_reports_what_the_camera_is_doing():
    cap = FakeCapture()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    source = CameraSource(cap, "test", drain=False)
    info = source.describe()
    check(info["width"] == 1280 and info["height"] == 720, "resolution reported")
    check(info["fourcc"] == "MJPG", f"pixel format reported, got {info['fourcc']}")
    check(info["backend"] == "fake", "backend reported")
    source.release()


def test_fourcc_name_decoding():
    check(fourcc_name(cv2.VideoWriter_fourcc(*"MJPG")) == "MJPG", "MJPG decodes")
    check(fourcc_name(cv2.VideoWriter_fourcc(*"YUYV")) == "YUYV", "YUYV decodes")
    check(fourcc_name(0) == "", "nothing decodes to nothing")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)
