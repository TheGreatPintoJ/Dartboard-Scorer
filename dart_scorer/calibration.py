"""Mapping between camera pixels and the board plane.

A dartboard is a flat disc, so a single homography is enough to undo the
perspective of any camera pose - no lens-independent 3D calibration needed.
The user marks four known landmarks (the outer edge of the double ring at the
centre of beds 20, 6, 3 and 11) and we solve for that homography.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import geometry as geo

# Canonical top-down view of the board used for display and for the mm mapping.
PX_PER_MM = 2.0
CANON_SIZE = int(round(2 * geo.R_BOARD * PX_PER_MM))          # 902 px
CANON_CENTRE = (CANON_SIZE / 2.0, CANON_SIZE / 2.0)

# The landmarks the operator is asked to click, in order.
REFERENCE_POINTS: tuple[tuple[str, float, float], ...] = (
    ("outer edge of DOUBLE 20  (top)", geo.R_DOUBLE_OUT, 90.0),
    ("outer edge of DOUBLE 6   (right)", geo.R_DOUBLE_OUT, 0.0),
    ("outer edge of DOUBLE 3   (bottom)", geo.R_DOUBLE_OUT, 270.0),
    ("outer edge of DOUBLE 11  (left)", geo.R_DOUBLE_OUT, 180.0),
    ("centre of the BULL  (optional, improves accuracy)", 0.0, 0.0),
)


# The same mm -> canonical mapping as board_to_canon, as a matrix, so it can be
# composed with a homography. See Calibration.board_matrix.
A_BOARD_TO_CANON = np.array([
    [PX_PER_MM, 0.0, CANON_CENTRE[0]],
    [0.0, PX_PER_MM, CANON_CENTRE[1]],
    [0.0, 0.0, 1.0],
], dtype=np.float64)


def board_to_canon(x_mm: float, y_mm: float) -> tuple[float, float]:
    return CANON_CENTRE[0] + x_mm * PX_PER_MM, CANON_CENTRE[1] + y_mm * PX_PER_MM


def canon_to_board(px: float, py: float) -> tuple[float, float]:
    return (px - CANON_CENTRE[0]) / PX_PER_MM, (py - CANON_CENTRE[1]) / PX_PER_MM


def reference_canon_points() -> np.ndarray:
    pts = []
    for _, r, a in REFERENCE_POINTS:
        pts.append(board_to_canon(*geo.polar_to_board(r, a)))
    return np.array(pts, dtype=np.float32)


@dataclass
class Calibration:
    """Image <-> board-plane transform."""

    H: np.ndarray                       # image px -> canonical px
    image_points: list[list[float]] = field(default_factory=list)
    frame_size: tuple[int, int] | None = None

    @property
    def H_inv(self) -> np.ndarray:
        # Cached: this is on the per-frame drawing path. A Calibration is
        # replaced wholesale when the landmarks change, never edited in place,
        # so the cache cannot go stale.
        inv = getattr(self, "_H_inv", None)
        if inv is None:
            inv = np.linalg.inv(self.H)
            object.__setattr__(self, "_H_inv", inv)
        return inv

    # -- construction ------------------------------------------------------
    @classmethod
    def from_points(cls, image_points, frame_size=None) -> "Calibration":
        src = np.array(image_points, dtype=np.float32)
        dst = reference_canon_points()[: len(src)]
        if len(src) < 4:
            raise ValueError("need at least 4 reference points")
        if len(src) == 4:
            H = cv2.getPerspectiveTransform(src, dst)
        else:
            H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
        if H is None:
            raise ValueError("could not solve a homography from those points")
        return cls(H=np.asarray(H, dtype=np.float64),
                   image_points=[list(map(float, p)) for p in src],
                   frame_size=frame_size)

    # -- transforms --------------------------------------------------------
    def to_canon(self, pts) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)

    def to_image(self, pts) -> np.ndarray:
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H_inv).reshape(-1, 2)

    def to_board_mm(self, pt) -> tuple[float, float]:
        cx, cy = self.to_canon([pt])[0]
        return canon_to_board(cx, cy)

    def board_mm_to_image(self, x_mm: float, y_mm: float) -> tuple[float, float]:
        px, py = board_to_canon(x_mm, y_mm)
        ix, iy = self.to_image([(px, py)])[0]
        return float(ix), float(iy)

    def score(self, image_point) -> geo.Score:
        x, y = self.to_board_mm(image_point)
        return geo.score_at(x, y)

    # -- lines on the board plane -----------------------------------------
    def board_matrix(self) -> np.ndarray:
        """``G``: homogeneous board millimetres -> image pixels.

        This is :meth:`board_mm_to_image` written as a matrix, which is what
        lets a whole *line* be pushed onto the board rather than a point.
        """
        g = getattr(self, "_G", None)
        if g is None:
            g = self.H_inv @ A_BOARD_TO_CANON
            object.__setattr__(self, "_G", g)
        return g

    def image_line_to_board(self, line) -> np.ndarray:
        """Push an image line ``(a, b, c)`` onto the board plane.

        A board point ``X`` lies on the result exactly when its image ``G @ X``
        lies on ``line``, so the answer is ``G.T @ line``: the shadow the line
        casts on the board from this camera's viewpoint.

        Returned normalised (``a^2 + b^2 == 1``) so that ``L @ (x, y, 1)`` is a
        signed distance in millimetres.
        """
        board = self.board_matrix().T @ np.asarray(line, dtype=np.float64)
        return normalise_line(board)

    # -- frame size --------------------------------------------------------
    def for_frame_size(self, size) -> "Calibration":
        """This calibration, valid for a frame of ``size`` (width, height).

        The homography is in raw pixel coordinates, so feeding it frames at a
        resolution other than the one it was marked at silently scores every
        dart in the wrong place - a 1920x1080 calibration fed 1280x720 puts the
        bull a quarter of a metre out. A uniform rescale is recoverable and is
        applied here; anything else is not, and raises.
        """
        want = (int(size[0]), int(size[1]))
        if not self.frame_size:
            # Saved before the size was recorded. The homography is unchanged;
            # we simply now know what it was marked at, so note it and carry on
            # rather than pretending this is a different calibration.
            self.frame_size = want
            return self
        have = (int(self.frame_size[0]), int(self.frame_size[1]))
        if have == want:
            return self
        sx, sy = want[0] / have[0], want[1] / have[1]
        if abs(sx - sy) > 0.01:
            raise ValueError(
                f"calibration was marked at {have[0]}x{have[1]} but the camera "
                f"is delivering {want[0]}x{want[1]}; the aspect ratio differs, "
                "so it cannot be rescaled - recalibrate at this resolution")
        scale = np.array([[1 / sx, 0.0, 0.0], [0.0, 1 / sy, 0.0], [0.0, 0.0, 1.0]])
        return Calibration(
            H=self.H @ scale,
            image_points=[[p[0] * sx, p[1] * sy] for p in self.image_points],
            frame_size=want,
        )

    def warp(self, frame) -> np.ndarray:
        """Rectify a camera frame into the canonical top-down board view."""
        return cv2.warpPerspective(frame, self.H, (CANON_SIZE, CANON_SIZE))

    # -- persistence -------------------------------------------------------
    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({
            "H": self.H.tolist(),
            "image_points": self.image_points,
            "frame_size": list(self.frame_size) if self.frame_size else None,
            "px_per_mm": PX_PER_MM,
            "canon_size": CANON_SIZE,
        }, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Calibration":
        data = json.loads(Path(path).read_text())
        size = data.get("frame_size")
        return cls(
            H=np.array(data["H"], dtype=np.float64),
            image_points=data.get("image_points", []),
            frame_size=tuple(size) if size else None,
        )


def measure_pose(calib, at=(0.0, 0.0), step_mm: float = 1.0) -> dict:
    """Where the camera is, read straight out of the calibration.

    A circle viewed off-axis images as an ellipse, squashed along the direction
    pointing at the viewer. So take the board -> image map's Jacobian at the
    bull and look at its two principal scales: the direction that shrinks most
    is the direction the camera lies in, and how much it shrinks is the sine of
    the camera's height above the board plane.

    Both fall out of one SVD, and because elevation is a *ratio* of the two
    scales, the focal length and the distance cancel - which is why this works
    with no intrinsics and no extra setup step. Verified exact against known
    synthetic camera poses; a millimetre of slop in the calibration clicks costs
    about a third of a degree.

    Distance is the one thing not recoverable: without a focal length, a close
    wide-angle camera and a distant narrow one produce the same picture.

    Returns bearing/elevation/roll in degrees, plus ``squash`` (the raw scale
    ratio) for anyone who wants the unrounded number.
    """
    g = calib.board_matrix()

    def image_of(p):
        v = g @ [p[0], p[1], 1.0]
        return v[:2] / v[2]

    def jacobian(p):
        h = step_mm
        return np.column_stack([
            (image_of((p[0] + h, p[1])) - image_of((p[0] - h, p[1]))) / (2 * h),
            (image_of((p[0], p[1] + h)) - image_of((p[0], p[1] - h))) / (2 * h),
        ])

    j = jacobian(at)
    _, sv, vt = np.linalg.svd(j)
    axis = vt[1]                      # board direction with the smallest scale

    # That axis is a line, not an arrow: it points at the camera or directly
    # away from it. The near half of the board images larger, so compare.
    probe = 120.0
    if abs(np.linalg.det(jacobian(axis * probe))) < \
            abs(np.linalg.det(jacobian(-axis * probe))):
        axis = -axis

    squash = float(sv[1] / sv[0]) if sv[0] > 1e-12 else 0.0
    up = j @ np.array([0.0, -1.0])     # where "up on the board" points in frame

    return {
        "bearing_deg": round(float(geo.angle_of(axis[0], axis[1]) % 360.0), 1),
        "elevation_deg": round(float(np.degrees(np.arcsin(np.clip(squash, 0.0, 1.0)))), 1),
        "roll_deg": round(float(np.degrees(np.arctan2(up[0], -up[1]))), 1),
        "squash": round(squash, 4),
    }


# Which end of the blob is the point, given where the camera sits. The barrel
# and flight stand away from the board, so a camera low to the board's plane
# sees them displaced towards itself - and the tip is the end on the far side.
# This is the table the README used to ask the operator to pick by hand.
def tip_mode_for_bearing(bearing_deg: float, elevation_deg: float = 90.0,
                         near_plane_deg: float = 35.0) -> str:
    """The ``tip_mode`` a camera at this bearing should use."""
    if elevation_deg >= near_plane_deg:
        # Looking down at the board rather than across it: the barrel always
        # projects outwards from the bull, so the usual rule holds.
        return "centre"
    quadrant = round((bearing_deg % 360.0) / 90.0) % 4
    return ("leftmost", "lowest", "rightmost", "highest")[quadrant]


def normalise_line(line) -> np.ndarray:
    """Scale ``(a, b, c)`` so ``a^2 + b^2 == 1``.

    Then evaluating the line at a point gives a signed distance in whatever
    units the point is in, which is what makes every threshold downstream
    physically meaningful rather than an arbitrary scale.
    """
    line = np.asarray(line, dtype=np.float64)
    n = float(np.hypot(line[0], line[1]))
    if n < 1e-12:
        raise ValueError("degenerate line: it has no direction")
    return line / n


def line_through(p, q) -> np.ndarray:
    """Normalised homogeneous line through two 2D points."""
    return normalise_line(np.cross([p[0], p[1], 1.0], [q[0], q[1], 1.0]))


def line_from_point_direction(point, direction) -> np.ndarray:
    """Normalised homogeneous line through ``point`` along ``direction``."""
    dx, dy = float(direction[0]), float(direction[1])
    n = float(np.hypot(dx, dy))
    if n < 1e-12:
        raise ValueError("degenerate direction")
    dx, dy = dx / n, dy / n
    px, py = float(point[0]), float(point[1])
    return np.array([dy, -dx, dx * py - dy * px], dtype=np.float64)


def fit_board_ellipse(frame) -> tuple | None:
    """Best-effort automatic outline of the board.

    Finds the red+green ring pixels (doubles and trebles are the only strongly
    saturated colours on a board) and fits an ellipse to their outer hull.
    Used only to pre-place the calibration markers - the operator still decides
    where bed 20 is, because the red/green pattern repeats every 36 degrees and
    cannot resolve rotation on its own.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red = cv2.inRange(hsv, (0, 90, 60), (10, 255, 255)) | \
          cv2.inRange(hsv, (170, 90, 60), (180, 255, 255))
    green = cv2.inRange(hsv, (35, 60, 40), (90, 255, 255))
    mask = cv2.morphologyEx(red | green, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pts = [c for c in contours if cv2.contourArea(c) > 200]
    if not pts:
        return None
    hull = cv2.convexHull(np.vstack(pts))
    if len(hull) < 5:
        return None
    return cv2.fitEllipse(hull)


def ellipse_reference_guess(ellipse) -> list[list[float]]:
    """Four starting markers on an ellipse, at its axis endpoints (top/right/
    bottom/left).  Rotation is a guess; the operator drags them into place."""
    (cx, cy), (w, h), ang = ellipse
    rad = np.radians(ang)
    ax = np.array([np.cos(rad), np.sin(rad)]) * (w / 2.0)      # major-ish axis
    ay = np.array([-np.sin(rad), np.cos(rad)]) * (h / 2.0)
    c = np.array([cx, cy])
    return [list(c - ay), list(c + ax), list(c + ay), list(c - ax)]
