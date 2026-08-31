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
from .calibration import (A_BOARD_TO_CANON, CANON_SIZE, REFERENCE_POINTS,
                          Calibration, board_to_canon, reference_canon_points)


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


# ---------------------------------------------------------------------- #
# 3D: a dart that actually stands proud of the board
# ---------------------------------------------------------------------- #
# The 2D synthetic camera above draws darts *lying in* the board plane, which
# is fine for exercising detection and scoring but has zero parallax - so it
# cannot exercise the two-view geometry at all. These build a real pinhole
# camera instead, with a centre to project off-plane points from.
#
# World frame is the board frame of geometry.py: millimetres, origin at the
# bull, +x right, +y down, and +z out of the face towards the thrower.

class Camera3D:
    """A pinhole camera looking at the board."""

    def __init__(self, K, R, C):
        self.K = np.asarray(K, dtype=np.float64)
        self.R = np.asarray(R, dtype=np.float64)       # world -> camera
        self.C = np.asarray(C, dtype=np.float64)       # centre, world mm

    def project(self, points):
        """3D board-frame millimetres -> image pixels."""
        pts = np.atleast_2d(np.asarray(points, dtype=np.float64))
        cam = (self.R @ (pts - self.C).T).T
        img = (self.K @ cam.T).T
        return img[:, :2] / img[:, 2:3]

    def board_matrix(self) -> np.ndarray:
        """Board millimetres (z = 0) -> image pixels, as a homography."""
        return self.K @ np.column_stack([self.R[:, 0], self.R[:, 1], -self.R @ self.C])


def look_at(distance_mm, azimuth_deg, elevation_deg, fov_deg, width, height):
    """A camera at the given bearing from the board, aimed at the bull."""
    az, el = np.radians(azimuth_deg), np.radians(elevation_deg)
    # Same angle convention as geo.polar_to_board: 0 = east, 90 = north, y down.
    c = distance_mm * np.array([np.cos(el) * np.cos(az),
                                -np.cos(el) * np.sin(az),
                                np.sin(el)])
    f = -c / np.linalg.norm(c)                       # towards the bull
    # The camera's own axes, in board coordinates: x right across the frame,
    # y *down* the frame, z forward. Building y from x and f (rather than the
    # other way round) is what keeps the rendered board the right way up - the
    # board frame has +y already pointing down, and +z out towards the thrower,
    # so the usual right-handed cross-product order renders it upside down.
    up_board = np.array([0.0, -1.0, 0.0])
    x = np.cross(up_board, f)
    if np.linalg.norm(x) < 1e-9:                     # looking straight along y
        x = np.cross(np.array([0.0, 0.0, 1.0]), f)
    x /= np.linalg.norm(x)
    y = np.cross(x, f)
    fx = (width / 2.0) / np.tan(np.radians(fov_deg) / 2.0)
    K = np.array([[fx, 0.0, width / 2.0],
                  [0.0, fx, height / 2.0],
                  [0.0, 0.0, 1.0]])
    return Camera3D(K, np.vstack([x, y, f]), c)


def synthetic_camera_3d(width=1280, height=720, *, distance_mm=1200.0,
                        azimuth_deg=35.0, elevation_deg=25.0, fov_deg=55.0,
                        seed=7, click_px=0.0):
    """Return (board image, view, calibration, camera) for a real 3D camera.

    ``click_px`` adds gaussian noise to the four landmarks the calibration is
    solved from, so tests can ask what an imperfectly clicked calibration does.
    """
    rng = np.random.default_rng(seed)
    cam = look_at(distance_mm, azimuth_deg, elevation_deg, fov_deg, width, height)

    # canonical board pixels -> image, for warping the rendered board in.
    H_cam = cam.board_matrix() @ np.linalg.inv(A_BOARD_TO_CANON)
    board = render.render_board()
    view = SyntheticView(H_cam, width, height, rng)

    marks = np.array([geo.polar_to_board(r, a) for _, r, a in REFERENCE_POINTS])
    image_points = cam.project(np.column_stack([marks, np.zeros(len(marks))]))
    if click_px:
        image_points = image_points + rng.normal(0, click_px, image_points.shape)
    calib = Calibration.from_points(image_points, (width, height))
    return board, view, calib, cam


def synthetic_pair(width=1280, height=720, *, separation_deg=90.0,
                   azimuth_deg=35.0, **kw):
    """Two cameras around the board, both aimed at the bull.

    Returns ``(board, [view_a, view_b], [calib_a, calib_b], [cam_a, cam_b])``.
    """
    a = synthetic_camera_3d(width, height, azimuth_deg=azimuth_deg, **kw)
    b = synthetic_camera_3d(width, height,
                            azimuth_deg=azimuth_deg + separation_deg, **kw)
    return a[0], [a[1], b[1]], [a[2], b[2]], [a[3], b[3]]


def dart_axis_3d(x_mm, y_mm, azimuth_deg, elevation_deg):
    """Unit vector along a dart's shaft, pointing away from the board."""
    az, el = np.radians(azimuth_deg), np.radians(elevation_deg)
    return np.array([np.cos(el) * np.cos(az),
                     -np.cos(el) * np.sin(az),
                     np.sin(el)])


def draw_dart_3d(image, cam, x_mm, y_mm, *, azimuth_deg=None, elevation_deg=20.0,
                 length_mm=150.0, buried_mm=8.0, flight=True):
    """Draw a dart standing out of the board, as ``cam`` would see it.

    ``buried_mm`` is how much of the point is hidden in the board, i.e. how far
    up the shaft the *visible* end starts. That gap is the whole reason a single
    camera reads the dart in the wrong place, so it is not a detail: it is the
    thing under test.
    """
    if azimuth_deg is None:                 # point away from the bull by default
        azimuth_deg = geo.angle_of(x_mm, y_mm) if (x_mm or y_mm) else 0.0
    d = dart_axis_3d(x_mm, y_mm, azimuth_deg, elevation_deg)
    tip = np.array([x_mm, y_mm, 0.0])
    visible = tip + buried_mm * d
    tail = tip + length_mm * d

    p_vis, p_tail = cam.project([visible, tail])
    shaft = (tuple(np.round(p_vis).astype(int)), tuple(np.round(p_tail).astype(int)))
    cv2.line(image, shaft[0], shaft[1], (60, 60, 60), 7, cv2.LINE_AA)     # barrel
    cv2.line(image, shaft[0], shaft[1], (210, 210, 210), 3, cv2.LINE_AA)  # highlight

    if flight:
        # A real flight is wide (about 35 mm) and sits at the far end. It is the
        # reason a contour-fitted principal axis misses the shaft, so model it
        # with actual width rather than as a dot.
        perp = np.cross(d, [0.0, 0.0, 1.0])
        if np.linalg.norm(perp) < 1e-6:
            perp = np.array([1.0, 0.0, 0.0])
        perp /= np.linalg.norm(perp)
        base = tip + (length_mm - 42.0) * d
        corners = cam.project([base, base + 17.0 * perp + 20.0 * d,
                               tail, base - 17.0 * perp + 20.0 * d])
        cv2.fillConvexPoly(image, np.round(corners).astype(np.int32), (40, 200, 240),
                           cv2.LINE_AA)
    return image


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

    def __init__(self, width=960, height=720, fps=30.0, stream="rgb"):
        self._board, self._view, self.calibration = synthetic_camera(width, height)
        self._canvas = self._board.copy()
        self._lock = threading.Lock()
        self._interval = 1.0 / max(fps, 1.0)
        self._next = 0.0
        # Pretending to be a Kinect's other cameras, so the stream selector and
        # everything downstream of it can be exercised with nothing plugged in.
        self.stream = stream if stream in ("rgb", "ir", "depth") else "rgb"
        self.name = "demo" if self.stream == "rgb" else f"demo:{self.stream}"
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
                self._warped = self._as_stream(self._view.warp(self._canvas))
                self._dirty = False
            return self._view.noise(self._warped)

    def _as_stream(self, frame):
        """Fake what a Kinect's infrared or depth camera would hand back.

        Infrared is a monochrome view lit by the sensor's own emitter, so it is
        the colour view with the colour taken out. Depth is flat across the
        board because the board *is* flat - which is exactly why depth is good
        at spotting a dart standing out of it, and useless for saying where on
        the board that dart landed.
        """
        if self.stream == "rgb":
            return frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.stream == "ir":
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        lit = cv2.GaussianBlur(gray, (9, 9), 0)
        board = (lit > 8).astype(np.uint8) * 150      # the board face, one plane
        proud = (cv2.absdiff(gray, lit) > 18).astype(np.uint8) * 105
        return cv2.cvtColor(np.clip(board + proud, 0, 255).astype(np.uint8),
                            cv2.COLOR_GRAY2BGR)

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
