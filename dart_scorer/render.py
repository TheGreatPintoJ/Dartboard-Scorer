"""Drawing helpers: a synthetic board, and overlays on the live camera view."""

from __future__ import annotations

import cv2
import numpy as np

from . import geometry as geo
from .calibration import CANON_CENTRE, CANON_SIZE, PX_PER_MM, board_to_canon

BLACK = (28, 28, 28)
CREAM = (196, 222, 236)
RED = (40, 40, 205)
GREEN = (70, 160, 60)
WIRE = (170, 170, 170)


def render_board(size: int = CANON_SIZE) -> np.ndarray:
    """A clean top-down dartboard in canonical coordinates.

    Painted per pixel straight from the same radius/bed maths the scorer uses,
    so the picture can never disagree with the scoring.
    """
    grid = (np.arange(CANON_SIZE, dtype=np.float32) - CANON_SIZE / 2.0) / PX_PER_MM
    x = grid[None, :]
    y = grid[:, None]
    r = np.hypot(x, y)
    angle = np.degrees(np.arctan2(-y, x)) % 360.0
    idx = (((geo.SECTOR_ORIGIN_DEG - angle) % 360.0) // geo.SECTOR_DEG).astype(np.int32)
    dark = (idx % 2) == 0                      # bed 20 is black, and they alternate

    img = np.zeros((CANON_SIZE, CANON_SIZE, 3), np.uint8)
    img[r <= geo.R_BOARD] = (18, 18, 18)       # the surround outside the doubles

    body = np.where(dark[..., None], np.array(BLACK), np.array(CREAM)).astype(np.uint8)
    ring = np.where(dark[..., None], np.array(RED), np.array(GREEN)).astype(np.uint8)
    for lo, hi, colour in (
        (geo.R_OUTER_BULL, geo.R_TRIPLE_IN, body),
        (geo.R_TRIPLE_IN, geo.R_TRIPLE_OUT, ring),
        (geo.R_TRIPLE_OUT, geo.R_DOUBLE_IN, body),
        (geo.R_DOUBLE_IN, geo.R_DOUBLE_OUT, ring),
    ):
        band = (r > lo) & (r <= hi)
        img[band] = colour[band]
    img[r <= geo.R_OUTER_BULL] = GREEN
    img[r <= geo.R_INNER_BULL] = RED

    c = (int(CANON_SIZE / 2), int(CANON_SIZE / 2))

    # Spider wires and the number ring.
    for idx in range(20):
        a = np.radians(geo.SECTOR_ORIGIN_DEG - idx * geo.SECTOR_DEG)
        d = np.array([np.cos(a), -np.sin(a)])
        p0 = np.array(board_to_canon(*(d * geo.R_OUTER_BULL)))
        p1 = np.array(board_to_canon(*(d * geo.R_DOUBLE_OUT)))
        cv2.line(img, tuple(p0.astype(int)), tuple(p1.astype(int)), WIRE, 1, cv2.LINE_AA)
    for r in (geo.R_OUTER_BULL, geo.R_TRIPLE_IN, geo.R_TRIPLE_OUT,
              geo.R_DOUBLE_IN, geo.R_DOUBLE_OUT):
        cv2.circle(img, c, int(r * PX_PER_MM), WIRE, 1, cv2.LINE_AA)
    for idx, bed in enumerate(geo.SECTORS):
        a = geo.SECTOR_ORIGIN_DEG - (idx + 0.5) * geo.SECTOR_DEG
        px, py = board_to_canon(*geo.polar_to_board(geo.R_DOUBLE_OUT + 25, a))
        cv2.putText(img, str(bed), (int(px) - 14, int(py) + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (235, 235, 235), 2, cv2.LINE_AA)
    return img


RING_RADII = (geo.R_OUTER_BULL, geo.R_TRIPLE_IN, geo.R_TRIPLE_OUT,
              geo.R_DOUBLE_IN, geo.R_DOUBLE_OUT)
_RING_STEPS = 181


def _project_mm(calib, points_mm):
    """Board millimetres -> image pixels, in one batched transform."""
    canon = np.asarray(points_mm, dtype=np.float32) * PX_PER_MM + np.float32(CANON_CENTRE)
    return calib.to_image(canon)


def _overlay_geometry(calib):
    """Rings, wires and number positions in image pixels.

    Cached on the calibration object: the board does not move between frames,
    so projecting it again every frame is pure waste. Projecting the ~950
    points one at a time costs ~10 ms; batched into three calls it is ~0.2 ms,
    and cached it is free. A Calibration is replaced wholesale whenever the
    landmarks change, so the cache invalidates itself.
    """
    cached = getattr(calib, "_overlay_geometry", None)
    if cached is not None:
        return cached

    angles = np.linspace(0.0, 360.0, _RING_STEPS)
    rings_mm = np.array([geo.polar_to_board(r, t) for r in RING_RADII for t in angles])
    rings = _project_mm(calib, rings_mm).reshape(len(RING_RADII), _RING_STEPS, 2)
    rings = [np.round(r).astype(np.int32).reshape(-1, 1, 2) for r in rings]

    wire_angles = [geo.SECTOR_ORIGIN_DEG - i * geo.SECTOR_DEG for i in range(20)]
    wires_mm = np.array(
        [geo.polar_to_board(r, a) for a in wire_angles
         for r in (geo.R_OUTER_BULL, geo.R_DOUBLE_OUT)])
    wires = np.round(_project_mm(calib, wires_mm)).astype(np.int32).reshape(20, 2, 2)

    label_mm = np.array([
        geo.polar_to_board(geo.R_DOUBLE_OUT + 18,
                           geo.SECTOR_ORIGIN_DEG - (i + 0.5) * geo.SECTOR_DEG)
        for i in range(20)])
    label_xy = np.round(_project_mm(calib, label_mm)).astype(np.int32)
    labels = [(str(bed), (int(x) - 10, int(y) + 5))
              for bed, (x, y) in zip(geo.SECTORS, label_xy)]

    geometry = (rings, wires, labels)
    calib._overlay_geometry = geometry
    return geometry


def draw_board_overlay(frame, calib, colour=(0, 220, 255), thickness=1):
    """Project the board's rings and wires back onto the camera image - the
    quickest way to eyeball whether a calibration is any good."""
    rings, wires, labels = _overlay_geometry(calib)
    cv2.polylines(frame, rings, True, colour, thickness, cv2.LINE_AA)
    for (x0, y0), (x1, y1) in wires:
        cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)),
                 colour, thickness, cv2.LINE_AA)
    for text, origin in labels:
        cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, colour, 1, cv2.LINE_AA)
    return frame


def draw_marker(frame, pt, label, colour=(0, 255, 120)):
    x, y = int(pt[0]), int(pt[1])
    cv2.drawMarker(frame, (x, y), colour, cv2.MARKER_CROSS, 18, 2)
    cv2.circle(frame, (x, y), 11, colour, 1, cv2.LINE_AA)
    if label:
        cv2.putText(frame, label, (x + 14, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, label, (x + 14, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, colour, 1, cv2.LINE_AA)
    return frame


def draw_panel(frame, lines, origin=(12, 12), width=280, colour=(255, 255, 255)):
    """Translucent HUD box, one entry per line."""
    x, y = origin
    h = 26 * len(lines) + 16
    box = frame[y:y + h, x:x + width]
    if box.size:
        frame[y:y + h, x:x + width] = cv2.addWeighted(
            box, 0.35, np.zeros_like(box), 0.0, 0)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x + 10, y + 28 + 26 * i),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, colour, 1, cv2.LINE_AA)
    return frame
