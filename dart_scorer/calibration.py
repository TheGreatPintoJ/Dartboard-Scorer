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
        return np.linalg.inv(self.H)

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
