"""Detecting darts in a fixed camera view of a board.

Strategy
--------
The camera does not move, so everything reduces to differencing:

* base         - a reference frame of the *empty* board. Differencing against
                 it tells us how many darts are in the board right now, and
                 when the board has been cleared.
* last_stable  - the frame as it looked after the previous dart settled.
                 Differencing against *that* isolates only the newest dart,
                 which is what keeps tight groupings (three in the treble 20)
                 resolvable: the new blob is separated in time even when it
                 touches its neighbours in space.

A dart is only registered once the image has stopped changing for a few frames,
so we never measure a dart mid-flight or a hand that is still in shot.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from . import fusion
from . import geometry as geo
from .calibration import Calibration


class State(str, Enum):
    LEARNING = "learning"       # building the empty-board reference
    READY = "ready"             # board quiet, waiting for a dart
    MOVING = "moving"           # something is moving (dart in flight, a hand)
    OCCLUDED = "occluded"       # too much of the board is covered to judge


@dataclass
class Dart:
    tip_image: tuple[float, float]
    board_mm: tuple[float, float]
    score: geo.Score
    area: float
    elongation: float
    confidence: float
    # The dart's axis in this image, and both ends of it. A single view cannot
    # tell where along this line the point actually is - that is the parallax
    # limit - so it is kept for a second view to intersect against.
    axis_image: tuple[float, float, float] | None = None
    ends_image: tuple[tuple[float, float], tuple[float, float]] | None = None
    axis_sigma_deg: float = 0.0
    fusion: dict | None = None

    @property
    def label(self) -> str:
        return self.score.label

    @property
    def points(self) -> int:
        return self.score.points


@dataclass
class Result:
    """What happened in a single frame."""

    state: State
    darts: list[Dart]           # newly registered this frame (usually 0 or 1)
    cleared: bool = False       # the board was emptied
    motion: float = 0.0
    change_area: int = 0
    debug_mask: np.ndarray | None = None


class DartDetector:
    def __init__(
        self,
        calibration: Calibration,
        *,
        diff_threshold: int = 26,
        min_area: int = 220,
        max_area: int = 26000,
        settle_frames: int = 4,
        motion_threshold: int = 120,   # changed pixels per frame that count as movement
        learn_frames: int = 25,
        min_elongation: float = 2.0,
        # Which end of the blob is the point:
        #   centre    - the end nearer the bull. Right for almost every setup,
        #               because the barrel and flight extend away from the board.
        #   lowest    - the end lowest in frame; for a camera above the board.
        #   highest   - the end highest in frame; for a camera below it.
        #   leftmost  - the end furthest left; for a camera off to the right.
        #   rightmost - the end furthest right; for a camera off to the left.
        tip_mode: str = "centre",
        # Slack on the "outside the board" reject, for when a second view is
        # about to correct the point. 0 keeps the single-camera behaviour.
        reject_margin_mm: float = 0.0,
    ) -> None:
        self.calib = calibration
        self.diff_threshold = diff_threshold
        self.min_area = min_area
        self.max_area = max_area
        self.settle_frames = settle_frames
        self.motion_threshold = motion_threshold
        self.learn_frames = learn_frames
        self.min_elongation = min_elongation
        self.tip_mode = tip_mode
        self.reject_margin_mm = reject_margin_mm

        self.state = State.LEARNING
        self._base: np.ndarray | None = None
        self._last_stable: np.ndarray | None = None
        self._prev: np.ndarray | None = None
        self._learned = 0
        self._still = 0
        self._base_area = 0          # change-vs-base area at the last stable point
        self._roi: np.ndarray | None = None
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _prepare(frame) -> np.ndarray:
        # Deliberately uint8, not float32. Every operation downstream of this
        # is a difference, a threshold or a blur, all of which OpenCV runs with
        # SIMD on 8-bit data; promoting to float doubles the memory traffic for
        # no accuracy that matters at a 26-level threshold. On a slow machine
        # this is the difference between keeping up with the camera and not.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (5, 5), 0)

    def _region_of_interest(self, shape) -> np.ndarray:
        """Everything outside the board (walls, the thrower) is ignored."""
        if self._roi is not None and self._roi.shape == shape:
            return self._roi
        mask = np.zeros(shape, np.uint8)
        ring = [
            self.calib.board_mm_to_image(*geo.polar_to_board(geo.R_BOARD * 1.35, t))
            for t in np.linspace(0, 360, 121)
        ]
        cv2.fillPoly(mask, [np.array(ring, np.int32)], 255)
        self._roi = mask
        return mask

    def _mask_against(self, gray, reference) -> tuple[np.ndarray, int]:
        diff = cv2.absdiff(gray, reference)
        _, mask = cv2.threshold(diff, self.diff_threshold, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel, iterations=2)
        mask = cv2.bitwise_and(mask, self._region_of_interest(mask.shape))
        return mask, int(cv2.countNonZero(mask))

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #
    def reset_background(self, frame=None) -> None:
        """Declare the board empty as of now."""
        self.state = State.LEARNING
        self._base = None if frame is None else self._prepare(frame)
        self._last_stable = None if self._base is None else self._base.copy()
        self._learned = 0 if frame is None else self.learn_frames
        self._still = 0
        self._base_area = 0
        if frame is not None:
            self.state = State.READY

    def update(self, frame) -> Result:
        gray = self._prepare(frame)
        # Movement is measured as the number of pixels that changed since the
        # previous frame - far less sensitive to sensor noise and to slow
        # lighting drift than a whole-frame mean would be.
        if self._prev is None:
            motion = 0.0
        else:
            changed = cv2.absdiff(gray, self._prev)
            _, changed = cv2.threshold(changed, self.diff_threshold, 255,
                                       cv2.THRESH_BINARY)
            motion = float(cv2.countNonZero(changed))
        self._prev = gray

        # --- learn the empty board ------------------------------------- #
        if self._base is None or self._learned < self.learn_frames:
            self._base = gray if self._base is None else \
                cv2.addWeighted(self._base, 0.8, gray, 0.2, 0)
            self._learned += 1
            if self._learned >= self.learn_frames:
                self._last_stable = self._base.copy()
                self.state = State.READY
            else:
                self.state = State.LEARNING
            return Result(self.state, [], motion=motion)

        total_mask, total_area = self._mask_against(gray, self._base)

        # --- wait for the scene to hold still --------------------------- #
        if motion > self.motion_threshold:
            self._still = 0
            self.state = State.MOVING
            return Result(self.state, [], motion=motion,
                          change_area=total_area, debug_mask=total_mask)

        self._still += 1
        if self._still < self.settle_frames:
            return Result(self.state, [], motion=motion,
                          change_area=total_area, debug_mask=total_mask)

        # A hand or an arm covers far more of the board than three darts do.
        if total_area > self.max_area * 3:
            self.state = State.OCCLUDED
            return Result(self.state, [], motion=motion,
                          change_area=total_area, debug_mask=total_mask)

        # --- board emptied ---------------------------------------------- #
        if total_area < self.min_area:
            cleared = self._base_area >= self.min_area
            self._base_area = 0
            self._last_stable = gray.copy()
            self.state = State.READY
            return Result(self.state, [], cleared=cleared, motion=motion,
                          change_area=total_area, debug_mask=total_mask)

        new_mask, new_area = self._mask_against(gray, self._last_stable)
        self.state = State.READY

        if new_area < self.min_area:
            # Nothing has changed since the last settled frame.
            return Result(self.state, [], motion=motion,
                          change_area=total_area, debug_mask=total_mask)

        if total_area < self._base_area - self.min_area:
            # Fewer darts than before: one was pulled out, or knocked out.
            self._base_area = total_area
            self._last_stable = gray.copy()
            return Result(self.state, [], motion=motion,
                          change_area=total_area, debug_mask=total_mask)

        dart = self._measure(new_mask)
        self._base_area = total_area
        self._last_stable = gray.copy()
        return Result(self.state, [dart] if dart else [], motion=motion,
                      change_area=total_area, debug_mask=new_mask)

    # ------------------------------------------------------------------ #
    # measurement
    # ------------------------------------------------------------------ #
    def _measure(self, mask) -> Dart | None:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = [c for c in contours
                    if self.min_area <= cv2.contourArea(c) <= self.max_area]
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))

        # Fitted to the mask's pixels weighted by inverse thickness, not to the
        # contour: a dart's silhouette is not symmetric about its shaft (the
        # flight is ~35 mm across, the barrel ~6 mm), so a contour fit is pulled
        # off the true axis by degrees. See fusion.image_axis.
        fit = fusion.image_axis(mask, contour)
        if fit is None:
            return None
        ends = [np.array(fit.ends[0]), np.array(fit.ends[1])]
        elongation = fit.elongation

        boards = [self.calib.to_board_mm(tuple(e)) for e in ends]
        radii = [geo.radius_of(*b) for b in boards]

        if self.tip_mode == "lowest":
            pick = int(np.argmax([e[1] for e in ends]))
        elif self.tip_mode == "highest":
            pick = int(np.argmin([e[1] for e in ends]))
        elif self.tip_mode == "leftmost":
            pick = int(np.argmin([e[0] for e in ends]))
        elif self.tip_mode == "rightmost":
            pick = int(np.argmax([e[0] for e in ends]))
        else:
            # "centre": the barrel and flight always point away from the bull,
            # so the end nearer the centre of the board is the point.
            pick = int(np.argmin(radii))

        tip = (float(ends[pick][0]), float(ends[pick][1]))
        board_mm = boards[pick]

        # Nothing outside the physical board can be a dart stuck in it. Movement
        # at the edge of frame - an arm reaching in, someone walking past, a
        # shifting shadow - otherwise lands in the visit as a phantom MISS.
        # A dart in the number ring is still a legitimate zero.
        #
        # When a second view is going to correct this point, the margin gives
        # fusion room to pull a borderline reading back onto the board; the hard
        # reject is then applied to the fused answer instead.
        if radii[pick] > geo.R_BOARD + self.reject_margin_mm:
            return None

        score = geo.score_at(*board_mm)

        confidence = 1.0
        if elongation < self.min_elongation:
            confidence *= 0.55                  # a blob, not obviously a dart
        if radii[pick] > geo.R_DOUBLE_OUT + 25:
            confidence *= 0.40                  # landed off the scoring area
        if score.near_wire:
            confidence *= 0.75                  # a wire call, worth an eyeball
        if area < self.min_area * 2:
            confidence *= 0.80

        return Dart(tip_image=tip, board_mm=board_mm, score=score, area=area,
                    elongation=elongation, confidence=round(confidence, 2),
                    axis_image=tuple(float(v) for v in fit.line),
                    ends_image=(fit.ends[0], fit.ends[1]),
                    axis_sigma_deg=fit.sigma_deg)
