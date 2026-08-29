"""Canonical dartboard geometry and score lookup.

Board coordinate system used everywhere below:

    origin  = centre of the bullseye
    units   = millimetres on the physical board plane
    +x      = to the right
    +y      = DOWNWARDS  (image convention, so it lines up with pixel space)

All radii are taken from the official WDF/BDO specification for a
standard 20-bed bristle board.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# --- Physical dimensions (millimetres, radius from the centre of the bull) ---
R_INNER_BULL = 6.35     # "double bull" / 50
R_OUTER_BULL = 15.9     # "single bull" / 25
R_TRIPLE_IN = 99.0      # inner edge of the treble ring
R_TRIPLE_OUT = 107.0    # outer edge of the treble ring
R_DOUBLE_IN = 162.0     # inner edge of the double ring
R_DOUBLE_OUT = 170.0    # outer edge of the double ring -> scoring boundary
R_BOARD = 225.5         # outside edge of the board incl. the number ring

WIRE_MM = 1.0           # nominal spider wire thickness, used for "close call" flags

# Bed numbers in clockwise order starting at 20, which sits at the top of the
# board.  Each bed spans exactly 360/20 = 18 degrees.
SECTORS: tuple[int, ...] = (
    20, 1, 18, 4, 13, 6, 10, 15, 2, 17,
    3, 19, 7, 16, 8, 11, 14, 9, 12, 5,
)
SECTOR_DEG = 360.0 / len(SECTORS)

# Bed 20 is centred on 90 degrees (straight up) so its clockwise leading edge
# sits at 90 + 9 = 99 degrees.
SECTOR_ORIGIN_DEG = 90.0 + SECTOR_DEG / 2.0

# Beds alternate black / white starting with 20 = black.  Black beds carry red
# doubles and trebles, white beds carry green ones.  Handy for rendering and
# for sanity-checking a calibration.
BLACK_BEDS = frozenset(SECTORS[0::2])


@dataclass(frozen=True)
class Score:
    """The result of looking up one landing point."""

    base: int           # 0 (miss), 1..20, or 25 (bull)
    multiplier: int     # 0 (miss), 1, 2 or 3
    ring: str           # miss | single | treble | double | bull | inner_bull
    radius_mm: float
    angle_deg: float
    near_wire: bool = False   # tip landed within a wire's width of a boundary

    @property
    def points(self) -> int:
        return self.base * self.multiplier

    @property
    def label(self) -> str:
        if self.multiplier == 0:
            return "MISS"
        if self.base == 25:
            return "BULL" if self.multiplier == 1 else "50"
        return f"{'  DT'[self.multiplier]}{self.base}".strip()

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return f"{self.label} ({self.points})"


def angle_of(x: float, y: float) -> float:
    """Angle of a board point in degrees, 0 = east, 90 = north (y is down)."""
    return math.degrees(math.atan2(-y, x)) % 360.0


def radius_of(x: float, y: float) -> float:
    return math.hypot(x, y)


def bed_at_angle(angle_deg: float) -> int:
    """Bed number whose wedge contains ``angle_deg``."""
    offset = (SECTOR_ORIGIN_DEG - angle_deg) % 360.0
    return SECTORS[int(offset // SECTOR_DEG)]


def bed_bounds(number: int) -> tuple[float, float]:
    """(start, end) angle in degrees of a bed, measured counter-clockwise."""
    idx = SECTORS.index(number)
    end = (SECTOR_ORIGIN_DEG - idx * SECTOR_DEG) % 360.0
    return (end - SECTOR_DEG) % 360.0, end


_RING_EDGES = (
    R_INNER_BULL, R_OUTER_BULL, R_TRIPLE_IN,
    R_TRIPLE_OUT, R_DOUBLE_IN, R_DOUBLE_OUT,
)


def score_at(x: float, y: float, wire_mm: float = WIRE_MM) -> Score:
    """Score the board point ``(x, y)`` given in millimetres."""
    r = radius_of(x, y)
    a = angle_of(x, y)

    # Distance to the nearest ring boundary, plus the nearest wedge wire,
    # so the caller can flag judgement calls near a wire.
    ring_gap = min(abs(r - edge) for edge in _RING_EDGES)
    wedge_gap = abs(((a - SECTOR_ORIGIN_DEG) % SECTOR_DEG) - SECTOR_DEG / 2.0)
    wedge_gap = (SECTOR_DEG / 2.0 - wedge_gap) * math.pi / 180.0 * max(r, 1e-6)
    near_wire = min(ring_gap, wedge_gap) < wire_mm and r > R_OUTER_BULL

    if r <= R_INNER_BULL:
        return Score(25, 2, "inner_bull", r, a, near_wire)
    if r <= R_OUTER_BULL:
        return Score(25, 1, "bull", r, a, near_wire)
    if r > R_DOUBLE_OUT:
        return Score(0, 0, "miss", r, a, False)

    bed = bed_at_angle(a)
    if R_TRIPLE_IN < r <= R_TRIPLE_OUT:
        return Score(bed, 3, "treble", r, a, near_wire)
    if R_DOUBLE_IN < r <= R_DOUBLE_OUT:
        return Score(bed, 2, "double", r, a, near_wire)
    return Score(bed, 1, "single", r, a, near_wire)


def polar_to_board(radius_mm: float, angle_deg: float) -> tuple[float, float]:
    """Inverse of :func:`angle_of`/:func:`radius_of` - handy for tests/rendering."""
    rad = math.radians(angle_deg)
    return radius_mm * math.cos(rad), -radius_mm * math.sin(rad)
