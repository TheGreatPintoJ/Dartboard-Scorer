"""Tip-end selection, on synthetic darts with a known orientation.

A dart lies along a line: one end is the point in the board, the other is the
flight. Which end is which depends on where the camera sits, so the mode has to
be selectable - and each mode has to pick the end it claims to.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dart_scorer import geometry as geo                    # noqa: E402
from dart_scorer.detector import DartDetector              # noqa: E402
from dart_scorer.synthetic import (                        # noqa: E402
    draw_dart_on_board, synthetic_camera)

TIP_R = 103.0          # where the point goes in
TAIL_R = TIP_R + 95.0  # where the flight ends up, 95 mm of dart later


def check(condition, message):
    assert condition, message


def throw(tip_mode, angle_deg):
    """Throw one dart along `angle_deg` and return the radius that was measured."""
    board, view, calib = synthetic_camera(960, 720)
    detector = DartDetector(calib, settle_frames=3, min_area=150,
                            tip_mode=tip_mode)
    canvas = board.copy()
    for _ in range(30):
        detector.update(view(canvas))
    draw_dart_on_board(canvas, *geo.polar_to_board(TIP_R, angle_deg), jitter_deg=0.0)
    for _ in range(8):
        result = detector.update(view(canvas))
        if result.darts:
            return geo.radius_of(*result.darts[0].board_mm)
    raise AssertionError(f"no dart detected in {tip_mode} mode")


def near(value, target, tolerance=12.0):
    return abs(value - target) <= tolerance


# --------------------------------------------------------------------------- #
def test_centre_picks_the_end_nearer_the_bull():
    for angle in (90.0, 0.0, 180.0, 270.0):
        r = throw("centre", angle)
        check(near(r, TIP_R), f"at {angle} deg expected the point at {TIP_R}, got {r:.1f}")


def test_lowest_and_highest():
    # A dart at the top of the board points up: the flight is higher in frame
    # than the point, so "lowest" is the point and "highest" is the flight.
    check(near(throw("lowest", 90.0), TIP_R),
          "lowest should be the point for a dart in the top half")
    check(near(throw("highest", 90.0), TAIL_R),
          "highest should be the flight for a dart in the top half")


def test_leftmost_and_rightmost():
    # A dart in bed 6 points right: the flight is further right than the point.
    check(near(throw("leftmost", 0.0), TIP_R),
          "leftmost should be the point for a dart on the right of the board")
    check(near(throw("rightmost", 0.0), TAIL_R),
          "rightmost should be the flight for a dart on the right of the board")

    # Mirror it: a dart in bed 11 points left, so the roles swap.
    check(near(throw("rightmost", 180.0), TIP_R),
          "rightmost should be the point for a dart on the left of the board")
    check(near(throw("leftmost", 180.0), TAIL_R),
          "leftmost should be the flight for a dart on the left of the board")


def test_unknown_mode_falls_back_to_centre():
    check(near(throw("nonsense", 90.0), TIP_R),
          "an unrecognised mode should behave like centre, not crash")


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
