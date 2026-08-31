"""The calibration's two jobs beyond scoring a point.

1. Pushing a whole *line* onto the board, which is what two-view fusion needs.
2. Noticing when it is being fed frames at a resolution it was not marked at.

(2) used to fail silently and catastrophically: the homography is in raw pixel
coordinates, so a 1920x1080 calibration fed 1280x720 frames puts the bull about
a quarter of a metre off the board and every throw is rejected as a miss with no
indication why. Nothing compared the two.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dart_scorer import geometry as geo                      # noqa: E402
from dart_scorer import synthetic as S                       # noqa: E402
from dart_scorer.calibration import Calibration, line_through  # noqa: E402
from dart_scorer.config import AppConfig                     # noqa: E402


def check(condition, message):
    assert condition, message


# --------------------------------------------------------------------------- #
# lines on the board
# --------------------------------------------------------------------------- #
def test_board_matrix_agrees_with_board_mm_to_image():
    _, _, calib, _ = S.synthetic_camera_3d(1280, 720)
    g = calib.board_matrix()
    for x, y in [(0.0, 0.0), (100.0, -40.0), (-150.0, 90.0)]:
        want = calib.board_mm_to_image(x, y)
        v = g @ [x, y, 1.0]
        got = (v[0] / v[2], v[1] / v[2])
        # board_mm_to_image goes through cv2.perspectiveTransform, which is
        # float32; board_matrix stays in float64, so it is the more exact of
        # the two. A thousandth of a pixel is far below anything that matters.
        check(abs(got[0] - want[0]) < 1e-3 and abs(got[1] - want[1]) < 1e-3,
              f"board_matrix disagrees at ({x}, {y}): {got} vs {want}")


def test_an_image_line_becomes_the_right_board_line():
    """Two board points on a line must still be on it after the push."""
    _, _, calib, _ = S.synthetic_camera_3d(1280, 720)
    a_mm, b_mm = (-120.0, 30.0), (140.0, -70.0)
    a_px = calib.board_mm_to_image(*a_mm)
    b_px = calib.board_mm_to_image(*b_mm)

    board_line = calib.image_line_to_board(line_through(a_px, b_px))
    for pt in (a_mm, b_mm):
        check(abs(board_line @ [pt[0], pt[1], 1.0]) < 1e-3,
              f"{pt} should lie on the pushed line")
    check(abs(np.hypot(board_line[0], board_line[1]) - 1.0) < 1e-9,
          "the line should come back normalised, so it measures millimetres")


# --------------------------------------------------------------------------- #
# frame size
# --------------------------------------------------------------------------- #
def test_the_same_size_is_returned_untouched():
    _, _, calib, _ = S.synthetic_camera_3d(1280, 720)
    check(calib.for_frame_size((1280, 720)) is calib,
          "a matching size should not copy anything")


def test_a_uniform_rescale_keeps_the_scoring_correct():
    """Half-resolution frames must still score the same board position."""
    _, _, calib, cam = S.synthetic_camera_3d(1280, 720)
    small = calib.for_frame_size((640, 360))
    for r, a in [(0.0, 0.0), (103.0, 18.0), (166.0, 234.0), (60.0, 300.0)]:
        x, y = geo.polar_to_board(r, a)
        px = cam.project([[x, y, 0.0]])[0]
        got = small.to_board_mm((px[0] / 2.0, px[1] / 2.0))
        check(np.hypot(got[0] - x, got[1] - y) < 0.05,
              f"({x:.0f}, {y:.0f}) scored {got} after rescaling")


def test_an_unrescaled_calibration_would_have_been_catastrophic():
    """Shows the size of the bug this guards against."""
    _, _, calib, cam = S.synthetic_camera_3d(1280, 720)
    px = cam.project([[0.0, 0.0, 0.0]])[0]                # the bull
    wrong = calib.to_board_mm((px[0] / 2.0, px[1] / 2.0))  # fed half-size frames
    check(geo.radius_of(*wrong) > geo.R_BOARD,
          "the whole point is that the error puts the bull off the board")


def test_a_different_aspect_ratio_refuses_rather_than_guesses():
    _, _, calib, _ = S.synthetic_camera_3d(1280, 720)
    try:
        calib.for_frame_size((1280, 1024))
    except ValueError as exc:
        check("recalibrate" in str(exc),
              f"the message should say what to do, got {exc}")
        return
    raise AssertionError("a squashed frame cannot be rescaled and must not be")


def test_a_calibration_with_no_recorded_size_adopts_the_live_one():
    """Calibrations saved before the size was recorded must still work."""
    _, _, calib, _ = S.synthetic_camera_3d(1280, 720)
    old = Calibration(H=calib.H.copy(), image_points=calib.image_points)
    fitted = old.for_frame_size((1280, 720))
    check(fitted.frame_size == (1280, 720), "it should adopt the live size")
    check(np.allclose(fitted.H, calib.H), "and leave the homography alone")


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def test_every_section_is_settable_not_just_the_original_three():
    """The section list is derived, so a new section cannot be forgotten."""
    cfg = AppConfig()
    changed = cfg.apply({"fusion": {"enabled": "false", "max_correction_mm": "25"}})
    check("fusion.enabled" in changed, f"fusion should be applied, got {changed}")
    check(cfg.fusion.enabled is False, "the browser's 'false' should coerce to bool")
    check(cfg.fusion.max_correction_mm == 25.0, "and numbers should coerce too")


def test_config_round_trips_through_json(tmp=None):
    import json
    import tempfile
    cfg = AppConfig()
    cfg.fusion.min_sin_theta = 0.5
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "config.json"
        cfg.save(path)
        back = AppConfig.load(path)
    check(back.fusion.min_sin_theta == 0.5, "fusion settings should survive a save")
    check(json.loads(json.dumps(cfg.to_dict()))["fusion"]["enabled"] is True,
          "to_dict should be JSON-clean")


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
