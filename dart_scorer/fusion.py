"""Recovering a dart's tip from two camera views.

The idea
--------
A dart's point is stuck *in* the board, so it sits exactly on the board plane.
What a single camera actually measures is the end of the dart's visible blob,
which is a few millimetres proud of the face; perspective then carries that
point sideways, and the error grows with how far off-axis the camera sits. That
is the "one camera cannot see depth" limit in the README.

Back-projecting a camera's view of the dart gives a plane through the camera
centre containing the whole dart.  Where that plane meets the board is the
*shadow* the dart casts from that viewpoint - and because the point lies on the
board, the point lies on its own shadow, in every view.  So:

    two shadows cross at the point.

Crucially this needs no camera intrinsics, no relative pose and no new
calibration step.  Pushing an image line onto the board is
``L = G.T @ line`` (:meth:`Calibration.image_line_to_board`), and a wrong focal
length would only distort the reconstruction by a map that fixes the board plane
pointwise - which leaves the shadow, and therefore the answer, untouched.

It also buys two things beyond parallax: the intersection extrapolates the shaft
through a buried or occluded point, and whichever end of a view's shadow segment
is nearer the crossing *is* the tip end - so which end is the point becomes a
measurement rather than the ``tip_mode`` guess.

Conditioning
------------
Two shadows that are nearly parallel cross at a point that is wildly sensitive
to noise in either.  ``sin_theta`` between them falls out of the same cross
product used to intersect them, so it is free; callers must gate on it. Ungated,
the answer is better than one camera on average and occasionally off the board
entirely.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import geometry as geo
from .calibration import line_from_point_direction

# Below this the two shadows are too close to parallel to be trusted; fall back
# to a single view instead. sin(25 degrees) ~ 0.42.
MIN_SIN_THETA = 0.42
# A fused point further than this from the single-view answer means something is
# wrong - a mismatched pair, a bad line fit. Keeping the single-view answer
# bounds the worst case to "no worse than one camera".
MAX_CORRECTION_MM = 30.0
# A dart seen nearly end-on casts a short, poorly determined shadow.
MIN_SEGMENT_MM = 40.0
# The crossing must land at the tip end of each view's shadow segment.
ENDPOINT_TOLERANCE_MM = 60.0


@dataclass
class AxisFit:
    """A dart's axis in one image."""

    line: np.ndarray                       # normalised (a, b, c), image pixels
    ends: tuple[tuple[float, float], tuple[float, float]]
    centroid: tuple[float, float]
    elongation: float
    sigma_deg: float                       # angular uncertainty of the fit
    area: float


@dataclass
class ViewAxis:
    """One view's contribution: its shadow on the board."""

    name: str
    line: np.ndarray                       # normalised (a, b, c), board mm
    ends_mm: tuple[tuple[float, float], tuple[float, float]]
    single_mm: tuple[float, float]         # that view's own tip estimate
    sigma_deg: float
    weight: float = 1.0


@dataclass
class FusionResult:
    board_mm: tuple[float, float] | None = None
    mode: str = "single"                   # fused | single | disagreement | rejected
    residual_mm: float = 0.0
    sin_theta: float = 0.0
    views: list[str] = field(default_factory=list)
    tip_end: dict[str, int] = field(default_factory=dict)
    parallax_mm: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    confidence_factor: float = 1.0


# ---------------------------------------------------------------------- #
# fitting the axis in one image
# ---------------------------------------------------------------------- #
def image_axis(mask, contour=None, *, min_area: float = 1.0) -> AxisFit | None:
    """Fit the dart's shaft axis in a binary mask.

    Fitted to mask *pixels* weighted by ``1 / (1 + distanceTransform)`` rather
    than to the contour.  A dart's silhouette is not symmetric about its shaft -
    the flight is around 35 mm across and the barrel 6 mm - so the silhouette's
    principal axis is pulled off the true axis by degrees.  Weighting by inverse
    thickness lets the thin shaft, which *is* the axis, dominate the fit and
    suppresses the flight.

    Fusion accuracy is entirely the accuracy of this line, so it is worth the
    one extra distance transform on a small ROI.
    """
    if contour is not None:
        region = np.zeros(mask.shape, np.uint8)
        cv2.drawContours(region, [contour], -1, 255, cv2.FILLED)
        region &= mask
    else:
        region = mask

    ys, xs = np.nonzero(region)
    if len(xs) < 3:
        return None
    area = float(len(xs))

    # Thin parts of the blob get a large weight, wide parts a small one.
    dist = cv2.distanceTransform(region, cv2.DIST_L2, 3)
    weights = 1.0 / (1.0 + dist[ys, xs].astype(np.float64))
    pts = np.column_stack((xs, ys)).astype(np.float64)

    axis, mean, sv = _weighted_axis(pts, weights)
    if axis is None:
        return None

    # Second pass: drop the pixels furthest from the first fit - in practice the
    # flight - and refit on what is left.
    perp = np.abs((pts - mean) @ np.array([-axis[1], axis[0]]))
    keep = perp <= max(1.5 * float(np.median(perp)), 1.0)
    if keep.sum() >= max(3, 0.2 * len(pts)):
        axis2, mean2, sv2 = _weighted_axis(pts[keep], weights[keep])
        if axis2 is not None:
            axis, mean, sv = axis2, mean2, sv2

    elongation = float(sv[0] / max(sv[1], 1e-6))
    # Angular uncertainty of a line fit: the perpendicular spread relative to the
    # along-axis spread, shrinking with the number of pixels behind it.
    sigma_deg = float(np.degrees(
        (sv[1] / max(sv[0], 1e-9)) / max(np.sqrt(len(pts)), 1.0)))

    proj = (pts - mean) @ axis
    ends = []
    for extreme in (proj.min(), proj.max()):
        # Average the few pixels at each end so one stray pixel cannot move the
        # answer by half a segment.
        near = pts[np.abs(proj - extreme) <= 2.0]
        end = near.mean(axis=0) if len(near) else pts[int(np.argmin(np.abs(proj - extreme)))]
        ends.append((float(end[0]), float(end[1])))

    return AxisFit(
        line=line_from_point_direction(mean, axis),
        ends=(ends[0], ends[1]),
        centroid=(float(mean[0]), float(mean[1])),
        elongation=elongation,
        sigma_deg=sigma_deg,
        area=area,
    )


def _weighted_axis(pts, weights):
    """Principal direction of ``pts`` under ``weights``."""
    total = float(weights.sum())
    if total <= 0 or len(pts) < 3:
        return None, None, (1.0, 1.0)
    mean = (pts * weights[:, None]).sum(axis=0) / total
    centred = pts - mean
    cov = (centred * weights[:, None]).T @ centred / total
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    axis = vecs[:, 0]
    sv = (float(np.sqrt(max(vals[0], 0.0))), float(np.sqrt(max(vals[1], 0.0))))
    return axis, mean, sv


# ---------------------------------------------------------------------- #
# intersecting the shadows
# ---------------------------------------------------------------------- #
def intersect(lines, weights=None) -> tuple[tuple[float, float], float, float] | None:
    """Weighted least-squares point closest to every line.

    Returns ``(point_mm, residual_mm, sin_theta)``.  Two lines give the exact
    crossing; a third (say a depth-derived one) turns the residual into a free,
    calibrated confidence number.  ``sin_theta`` is the smallest angle between
    any pair, which is the conditioning of the whole system.
    """
    lines = [np.asarray(l, dtype=np.float64) for l in lines]
    if len(lines) < 2:
        return None
    w = np.ones(len(lines)) if weights is None else np.asarray(weights, dtype=np.float64)

    a = np.array([l[0] for l in lines])
    b = np.array([l[1] for l in lines])
    c = np.array([l[2] for l in lines])

    m = np.array([[float((w * a * a).sum()), float((w * a * b).sum())],
                  [float((w * a * b).sum()), float((w * b * b).sum())]])
    rhs = -np.array([float((w * a * c).sum()), float((w * b * c).sum())])
    det = float(np.linalg.det(m))

    sin_theta = min(abs(float(lines[i][0] * lines[j][1] - lines[j][0] * lines[i][1]))
                    for i in range(len(lines)) for j in range(i + 1, len(lines)))
    if abs(det) < 1e-12:
        return None

    point = np.linalg.solve(m, rhs)
    res = a * point[0] + b * point[1] + c
    residual = float(np.sqrt(float((w * res * res).sum()) / max(float(w.sum()), 1e-9)))
    return (float(point[0]), float(point[1])), residual, sin_theta


def segment_length_mm(ends_mm) -> float:
    (x0, y0), (x1, y1) = ends_mm
    return float(np.hypot(x1 - x0, y1 - y0))


def tip_end_index(point_mm, ends_mm) -> int:
    """Which end of a view's shadow segment the crossing sits at.

    This is what replaces ``tip_mode``: rather than guessing which end of the
    blob is the point from where the camera happens to be mounted, the crossing
    tells us, in every view at once.
    """
    (x0, y0), (x1, y1) = ends_mm
    d0 = np.hypot(point_mm[0] - x0, point_mm[1] - y0)
    d1 = np.hypot(point_mm[0] - x1, point_mm[1] - y1)
    return 0 if d0 <= d1 else 1


def _endpoint_ok(point_mm, ends_mm, tolerance) -> bool:
    """The crossing must be at one end of the shadow, not stranded in the middle."""
    (x0, y0), (x1, y1) = ends_mm
    seg = np.array([x1 - x0, y1 - y0])
    length = float(np.hypot(*seg))
    if length < 1e-6:
        return False
    t = float((np.array(point_mm) - np.array([x0, y0])) @ seg / (length * length))
    slack = tolerance / length
    # Inside the segment is fine only near an end; beyond either end is fine
    # within tolerance, because the true point is past the visible blob.
    return (-slack <= t <= 0.35) or (0.65 <= t <= 1.0 + slack)


# ---------------------------------------------------------------------- #
# the whole decision
# ---------------------------------------------------------------------- #
def fuse(views, *, primary: str | None = None,
         min_sin_theta: float = MIN_SIN_THETA,
         max_correction_mm: float = MAX_CORRECTION_MM,
         min_segment_mm: float = MIN_SEGMENT_MM,
         endpoint_tolerance_mm: float = ENDPOINT_TOLERANCE_MM,
         max_pair_mm: float = 60.0) -> FusionResult:
    """Combine several views' shadows into one board position.

    Falls back to the primary view's own estimate whenever the geometry is not
    trustworthy, so the result is never worse than a single camera.
    """
    views = list(views)
    if not views:
        return FusionResult(mode="rejected", reasons=["no views"])

    order = {v.name: i for i, v in enumerate(views)}
    lead = views[order[primary]] if primary in order else views[0]
    fallback = FusionResult(board_mm=lead.single_mm, mode="single",
                            views=[lead.name], confidence_factor=1.0)

    usable = [v for v in views if segment_length_mm(v.ends_mm) >= min_segment_mm]
    if len(usable) < 2:
        fallback.reasons.append("only one usable view")
        return fallback

    # Two views that disagree wildly are looking at different darts.
    for i in range(len(usable)):
        for j in range(i + 1, len(usable)):
            gap = np.hypot(usable[i].single_mm[0] - usable[j].single_mm[0],
                           usable[i].single_mm[1] - usable[j].single_mm[1])
            if gap > max_pair_mm:
                fallback.mode = "single"
                fallback.reasons.append(
                    f"views disagree by {gap:.0f} mm - probably different darts")
                fallback.confidence_factor = 0.85
                return fallback

    # Weight by how well each view pins down the crossing: a confident line
    # (many pixels, thin residual) close to the tip is worth more.
    weights = [v.weight / max(v.sigma_deg, 1e-3) ** 2 for v in usable]
    solved = intersect([v.line for v in usable], weights)
    if solved is None:
        fallback.reasons.append("shadows are parallel")
        fallback.confidence_factor = 0.85
        return fallback

    point, residual, sin_theta = solved
    if sin_theta < min_sin_theta:
        fallback.reasons.append(
            f"shadows cross at {np.degrees(np.arcsin(min(sin_theta, 1.0))):.0f} "
            "degrees - too shallow to trust")
        fallback.confidence_factor = 0.85
        return fallback

    for v in usable:
        if not _endpoint_ok(point, v.ends_mm, endpoint_tolerance_mm):
            fallback.reasons.append(f"crossing is not at an end of {v.name}'s shadow")
            fallback.confidence_factor = 0.85
            return fallback

    # The clamp that bounds the damage: if fusing moved the answer further than
    # a plausible parallax error, something is wrong with the match, and the
    # single-view answer is the safer one.
    correction = float(np.hypot(point[0] - lead.single_mm[0],
                                point[1] - lead.single_mm[1]))
    if correction > max_correction_mm:
        return FusionResult(
            board_mm=lead.single_mm, mode="disagreement",
            residual_mm=residual, sin_theta=sin_theta,
            views=[v.name for v in usable], confidence_factor=0.60,
            reasons=[f"fusion moved the point {correction:.0f} mm; kept the "
                     f"{lead.name} answer"])

    return FusionResult(
        board_mm=point,
        mode="fused",
        residual_mm=residual,
        sin_theta=sin_theta,
        views=[v.name for v in usable],
        tip_end={v.name: tip_end_index(point, v.ends_mm) for v in usable},
        # How far each view's own answer sat from the truth: this is exactly the
        # parallax bias the README's "known limits" is about, now measurable.
        parallax_mm={v.name: float(np.hypot(point[0] - v.single_mm[0],
                                            point[1] - v.single_mm[1]))
                     for v in usable},
        confidence_factor=_confidence(residual, sin_theta),
    )


def _confidence(residual_mm: float, sin_theta: float) -> float:
    good_residual = float(np.clip(1.0 - residual_mm / 20.0, 0.5, 1.0))
    good_angle = float(np.clip(sin_theta / 0.57, 0.5, 1.0))     # 0.57 ~ 35 deg
    return round(good_residual * good_angle, 3)


def view_axis(name, calib, fit: AxisFit, single_mm, *, weight: float = 1.0) -> ViewAxis:
    """Turn one view's image-space axis fit into its shadow on the board."""
    return ViewAxis(
        name=name,
        line=calib.image_line_to_board(fit.line),
        ends_mm=(calib.to_board_mm(fit.ends[0]), calib.to_board_mm(fit.ends[1])),
        single_mm=tuple(single_mm),
        sigma_deg=max(fit.sigma_deg, 1e-3),
        weight=weight,
    )


def score_fused(result: FusionResult) -> geo.Score | None:
    if result.board_mm is None:
        return None
    return geo.score_at(*result.board_mm)
