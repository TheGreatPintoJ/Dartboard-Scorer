"""A fake camera, for testing the pipeline and demoing the web UI.

Renders the canonical board through a perspective warp so it looks like a
camera off to one side of the board, and lets you "throw" darts at named
targets. Everything downstream - detection, tip finding, scoring - runs exactly
as it does on a real feed.
"""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np

from . import geometry as geo
from . import render
from .calibration import CANON_SIZE, Calibration, board_to_canon, reference_canon_points


class SyntheticView:
    """Turns the canonical board into what an off-axis camera would see.

    The two halves are separate because they change at different rates: the
    perspective warp only changes when a dart lands, while the sensor noise
    must differ every frame. Drawing two million gaussian samples per frame
    costs more than the whole scoring pipeline, so the noise is generated once
    and cycled - otherwise the demo source looks like the pipeline is slow.
    """

    def __init__(self, H_cam, width, height, rng, pool=8):
        self.H_cam = H_cam
        self.size = (width, height)
        self._noise = [rng.normal(0, 1.6, (height, width, 3)).astype(np.int16)
                       for _ in range(pool)]
        self._n = 0

    def warp(self, canvas):
        return cv2.warpPerspective(canvas, self.H_cam, self.size)

    def noise(self, image):
        self._n += 1
        return cv2.add(image, self._noise[self._n % len(self._noise)], dtype=cv2.CV_8U)

    def __call__(self, canvas):
        return self.noise(self.warp(canvas))


def synthetic_camera(width=960, height=720, skew=0.16, seed=7):
    """Return (board image, view function, the matching perfect calibration)."""
    rng = np.random.default_rng(seed)
    src = np.float32([[0, 0], [CANON_SIZE, 0], [CANON_SIZE, CANON_SIZE], [0, CANON_SIZE]])
    m = min(width, height) * 0.92
    cx, cy = width / 2, height / 2
    dst = np.float32([
        [cx - m / 2, cy - m / 2 * (1 - skew)],
        [cx + m / 2, cy - m / 2 * (1 + skew)],
        [cx + m / 2, cy + m / 2 * (1 - skew)],
        [cx - m / 2, cy + m / 2 * (1 + skew)],
    ])
    H_cam = cv2.getPerspectiveTransform(src, dst)          # canonical -> camera
    board = render.render_board()

    view = SyntheticView(H_cam, width, height, rng)

    image_points = cv2.perspectiveTransform(
        reference_canon_points().reshape(-1, 1, 2), H_cam).reshape(-1, 2)
    calib = Calibration.from_points(image_points, (width, height))
    return board, view, calib


def draw_dart_on_board(canvas, x_mm, y_mm, length_mm=95.0, jitter_deg=0.0):
    """Draw a dart lying in the board plane with its point at (x_mm, y_mm)."""
    angle = geo.angle_of(x_mm, y_mm) + jitter_deg
    r = max(geo.radius_of(x_mm, y_mm), 1.0)
    tail = geo.polar_to_board(r + length_mm, angle)
    p0 = tuple(int(v) for v in board_to_canon(x_mm, y_mm))
    p1 = tuple(int(v) for v in board_to_canon(*tail))
    cv2.line(canvas, p0, p1, (60, 60, 60), 7, cv2.LINE_AA)          # barrel
    cv2.line(canvas, p0, p1, (210, 210, 210), 3, cv2.LINE_AA)       # highlight
    cv2.circle(canvas, p1, 9, (40, 200, 240), -1, cv2.LINE_AA)      # flight
    return canvas


def target_for_label(label: str) -> tuple[float, float]:
    """Board point that scores `label`: T20, D16, 5, BULL, 50, MISS."""
    label = label.strip().upper()
    if label in ("BULL", "25"):
        return geo.polar_to_board(10.0, 0.0)
    if label in ("50", "DBULL", "D25"):
        return geo.polar_to_board(2.0, 0.0)
    if label == "MISS":
        return geo.polar_to_board(190.0, 45.0)
    if label[0] in "DT" and label[1:].isdigit():
        mult, bed = label[0], int(label[1:])
    elif label.isdigit():
        mult, bed = "S", int(label)
    else:
        raise ValueError(f"cannot read the target {label!r}")
    if bed not in geo.SECTORS:
        raise ValueError(f"{bed} is not a bed on a dartboard")
    radius = {"S": 140.0, "D": 166.0, "T": 103.0}[mult]
    return geo.polar_to_board(radius, 90.0 - geo.SECTOR_DEG * geo.SECTORS.index(bed))


class DemoSource:
    """Frame source that behaves like a camera pointed at a board.

    Throw darts into it with :meth:`throw`; pull them out with :meth:`clear`.
    Used by ``--source demo`` so the service is fully exercisable with no
    hardware attached.
    """

    def __init__(self, width=960, height=720, fps=30.0):
        self._board, self._view, self.calibration = synthetic_camera(width, height)
        self._canvas = self._board.copy()
        self._lock = threading.Lock()
        self._interval = 1.0 / max(fps, 1.0)
        self._next = 0.0
        self.name = "demo"
        self.darts: list[str] = []
        self._dirty = True
        self._warped: np.ndarray | None = None

    def read(self):
        now = time.monotonic()
        if now < self._next:
            time.sleep(min(self._next - now, self._interval))
        self._next = max(now, self._next) + self._interval
        with self._lock:
            # The board only changes when a dart lands or the darts come out,
            # so the perspective warp is cached; only the noise is per frame.
            if self._dirty or self._warped is None:
                self._warped = self._view.warp(self._canvas)
                self._dirty = False
            return self._view.noise(self._warped)

    def throw(self, label: str, jitter_deg: float = 5.0) -> str:
        x, y = target_for_label(label)
        with self._lock:
            if len(self.darts) >= 3:
                self._canvas = self._board.copy()
                self.darts.clear()
                self._dirty = True
            draw_dart_on_board(self._canvas, x, y, jitter_deg=jitter_deg)
            self.darts.append(label.upper())
            self._dirty = True
        return label.upper()

    def clear(self) -> None:
        with self._lock:
            self._canvas = self._board.copy()
            self.darts.clear()
            self._dirty = True

    def release(self) -> None:
        pass
