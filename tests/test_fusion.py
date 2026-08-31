"""Two-view fusion: does intersecting two shadows actually find the point?

The dart's point is stuck in the board, so it lies on the board plane, so it
lies on the shadow its own shaft casts there - from every viewpoint. Two
shadows therefore cross at the point. These tests check that claim end to end
on synthetic 3D darts, and that the fallbacks fire when the geometry is bad.

The headline assertion is the accuracy comparison at the bottom: fusion must
beat a single camera by a wide margin on the median, and must not make the tail
worse. If that stops holding, the feature is not earning its complexity.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dart_scorer import fusion                              # noqa: E402
from dart_scorer import geometry as geo                     # noqa: E402
from dart_scorer import synthetic as S                      # noqa: E402


def check(condition, message):
    assert condition, message


def shadow(calib, cam, tip_mm, azimuth, elevation, *, buried_mm=8.0,
           length_mm=150.0, noise_px=0.0, rng=None):
    """What one view contributes for a known dart: its board-plane shadow."""
    d = S.dart_axis_3d(tip_mm[0], tip_mm[1], azimuth, elevation)
    tip3 = np.array([tip_mm[0], tip_mm[1], 0.0])
    seen = cam.project([tip3 + buried_mm * d, tip3 + length_mm * d])
    if noise_px and rng is not None:
        seen = seen + rng.normal(0, noise_px, seen.shape)

    line_img = np.cross([seen[0][0], seen[0][1], 1.0], [seen[1][0], seen[1][1], 1.0])
    single = calib.to_board_mm(tuple(seen[0]))          # what one camera reports
    return fusion.ViewAxis(
        name="v", line=calib.image_line_to_board(line_img),
        ends_mm=(calib.to_board_mm(tuple(seen[0])), calib.to_board_mm(tuple(seen[1]))),
        single_mm=single, sigma_deg=0.5)


def pair(tip_mm, azimuth, elevation, *, separation=90.0, noise_px=0.0,
         click_px=0.0, rng=None, **kw):
    _, _, calibs, cams = S.synthetic_pair(1280, 720, separation_deg=separation,
                                          click_px=click_px)
    out = []
    for i, (calib, cam) in enumerate(zip(calibs, cams)):
        v = shadow(calib, cam, tip_mm, azimuth, elevation,
                   noise_px=noise_px, rng=rng, **kw)
        v.name = f"cam{i}"
        out.append(v)
    return out


# --------------------------------------------------------------------------- #
# the geometry itself
# --------------------------------------------------------------------------- #
def test_the_point_lies_on_its_own_shadow():
    """The claim the whole feature rests on."""
    _, _, calib, cam = S.synthetic_camera_3d(1280, 720)
    for tip in [(0.0, 0.0), (100.0, -40.0), (-60.0, 120.0), (150.0, 150.0)]:
        v = shadow(calib, cam, tip, azimuth=25.0, elevation=30.0)
        offset = abs(v.line @ [tip[0], tip[1], 1.0])
        check(offset < 1e-3,
              f"tip {tip} sits {offset:.4f} mm off its own shadow")


def test_two_shadows_cross_at_the_point():
    """With perfect data a fused answer should be exact, not merely close."""
    fused = 0
    total = 0
    for tip in [(0.0, 0.0), (103.0, 0.0), (-80.0, 60.0), (40.0, -140.0),
                (150.0, 20.0), (-30.0, -160.0)]:
        for elevation in (20.0, 30.0, 45.0):
            # Darts point away from the bull, which is what makes the "end
            # nearer the centre is the point" rule work in the first place.
            azimuth = geo.angle_of(*tip) if any(tip) else 0.0
            views = pair(tip, azimuth=azimuth, elevation=elevation)
            got = fusion.fuse(views, primary="cam0")
            total += 1
            if got.mode != "fused":
                continue
            fused += 1
            err = np.hypot(got.board_mm[0] - tip[0], got.board_mm[1] - tip[1])
            check(err < 0.05,
                  f"tip {tip} at {elevation} deg recovered {err:.4f} mm out")
    check(fused >= 0.7 * total,
          f"only {fused}/{total} throws fused; the gates are too tight")


def test_a_flat_dart_cannot_be_fused():
    """An honest limit, not a bug.

    Fusion works because each camera sees the dart's shadow from a different
    place. A dart lying almost flat in the board casts nearly the same shadow
    from everywhere, so the two lines become parallel and there is no crossing
    to find. That has to degrade to one camera rather than guess.
    """
    views = pair((103.0, 0.0), azimuth=0.0, elevation=4.0)
    got = fusion.fuse(views, primary="cam0")
    check(got.mode != "fused",
          f"a dart 4 degrees off the board should not fuse, got {got.mode}")
    check(got.board_mm == views[0].single_mm, "it should keep the primary's answer")


def test_fusion_beats_one_camera_on_a_buried_point():
    """A camera reads the dart where its *visible* end is, several mm adrift."""
    tip = (103.0, 0.0)
    views = pair(tip, azimuth=20.0, elevation=30.0, buried_mm=12.0)
    single = np.hypot(views[0].single_mm[0] - tip[0], views[0].single_mm[1] - tip[1])
    got = fusion.fuse(views, primary="cam0")
    fused = np.hypot(got.board_mm[0] - tip[0], got.board_mm[1] - tip[1])
    check(single > 3.0, f"the single-view bias should be real, got {single:.2f} mm")
    check(fused < 0.05, f"fusion should remove it, left {fused:.3f} mm")


def test_intersection_is_exact_for_two_lines():
    a = np.array([1.0, 0.0, -50.0])         # x = 50
    b = np.array([0.0, 1.0, -20.0])         # y = 20
    point, residual, sin_theta = fusion.intersect([a, b])
    check(abs(point[0] - 50.0) < 1e-9 and abs(point[1] - 20.0) < 1e-9,
          f"expected (50, 20), got {point}")
    check(residual < 1e-9, f"two lines should meet exactly, residual {residual}")
    check(abs(sin_theta - 1.0) < 1e-9, "perpendicular lines have sin_theta 1")


def test_parallel_shadows_are_rejected_not_guessed():
    a = np.array([1.0, 0.0, -50.0])
    b = np.array([1.0, 0.0, -51.0])
    check(fusion.intersect([a, b]) is None,
          "parallel lines have no crossing and must not return one")


# --------------------------------------------------------------------------- #
# the axis fit, which is what limits the whole thing
# --------------------------------------------------------------------------- #
def _drawn_dart(cam, angle, elevation=25.0):
    """Render one dart and return (mask, true image direction, span in px)."""
    import cv2
    x, y = geo.polar_to_board(103.0, angle)
    canvas = np.zeros((720, 1280, 3), np.uint8)
    S.draw_dart_3d(canvas, cam, x, y, elevation_deg=elevation, flight=True)
    mask = (cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY) > 8).astype(np.uint8) * 255

    d = S.dart_axis_3d(x, y, geo.angle_of(x, y), elevation)
    tip3 = np.array([x, y, 0.0])
    truth = cam.project([tip3 + 8 * d, tip3 + 150 * d])
    want = truth[1] - truth[0]
    span = float(np.linalg.norm(want))
    return mask, want / span, span


def _axis_error_deg(line, want):
    got = np.array([-line[1], line[0]])
    return abs(np.degrees(np.arcsin(
        np.clip(abs(want[0] * got[1] - want[1] * got[0]), 0, 1))))


def test_axis_fit_is_not_dragged_off_by_the_flight():
    """A dart's flight is ~6x wider than its barrel; the axis must ignore it."""
    _, _, _, cam = S.synthetic_camera_3d(1280, 720)
    worst = 0.0
    for angle in (0.0, 120.0, 200.0, 300.0):
        mask, want, span = _drawn_dart(cam, angle)
        fit = fusion.image_axis(mask)
        check(fit is not None, "the dart should be fittable")
        worst = max(worst, _axis_error_deg(fit.line, want))
    check(worst < 1.0, f"axis fit is {worst:.2f} deg off the true shaft")


def test_a_foreshortened_dart_is_reported_as_uncertain():
    """A dart pointing at the camera is genuinely hard to fit.

    The fit cannot be accurate there and must not pretend otherwise - the
    weighting in fuse() is 1/sigma^2, so an honest sigma is what stops a
    badly-seen view outvoting a well-seen one.
    """
    _, _, _, cam = S.synthetic_camera_3d(1280, 720)
    clear, _, clear_span = _drawn_dart(cam, 120.0)
    end_on, _, end_on_span = _drawn_dart(cam, 45.0)
    check(end_on_span < clear_span / 2,
          "the 45 degree dart should be the foreshortened one")
    a, b = fusion.image_axis(clear), fusion.image_axis(end_on)
    check(b.sigma_deg > a.sigma_deg * 1.5,
          f"the foreshortened fit should own up: sigma {b.sigma_deg:.3f} "
          f"vs {a.sigma_deg:.3f}")


# --------------------------------------------------------------------------- #
# the gates
# --------------------------------------------------------------------------- #
def test_shallow_crossings_fall_back_to_one_camera():
    """Nearly parallel shadows must not be trusted - that is the 249 mm tail."""
    views = pair((103.0, 0.0), azimuth=20.0, elevation=30.0, separation=4.0)
    got = fusion.fuse(views, primary="cam0")
    check(got.mode != "fused",
          f"a 4-degree camera separation should not fuse, got {got.mode}")
    check(got.board_mm == views[0].single_mm, "it should keep the primary's answer")


def test_a_wild_correction_is_clamped_to_the_single_view():
    """Bounding the damage: fusion may never be worse than one camera."""
    views = pair((103.0, 0.0), azimuth=20.0, elevation=30.0)
    views[1].line = np.array([0.0, 1.0, -200.0])     # nonsense second view
    views[1].ends_mm = ((-100.0, 200.0), (100.0, 200.0))
    views[1].single_mm = (0.0, 200.0)
    got = fusion.fuse(views, primary="cam0")
    check(got.mode in ("single", "disagreement"),
          f"a bad partner should not be believed, got {got.mode}")
    check(got.board_mm == views[0].single_mm, "it should keep the primary's answer")


def test_views_looking_at_different_darts_are_rejected():
    views = pair((103.0, 0.0), azimuth=20.0, elevation=30.0)
    views[1].single_mm = (-150.0, 120.0)                    # a different dart
    got = fusion.fuse(views, primary="cam0")
    check(got.mode == "single", f"expected a fallback, got {got.mode}")
    check(any("disagree" in r for r in got.reasons),
          f"it should say why: {got.reasons}")


def test_a_single_view_still_scores():
    views = pair((103.0, 0.0), azimuth=20.0, elevation=30.0)[:1]
    got = fusion.fuse(views, primary="cam0")
    check(got.mode == "single", "one view is a legitimate install, not an error")
    check(got.board_mm == views[0].single_mm, "it should report that view's answer")
    check(got.confidence_factor == 1.0,
          "a deliberate one-camera setup must not be penalised")


def test_tip_end_is_measured_not_guessed():
    """What replaces tip_mode: the crossing says which blob end is the point."""
    tip = (103.0, 0.0)
    views = pair(tip, azimuth=20.0, elevation=30.0)
    got = fusion.fuse(views, primary="cam0")
    for v in views:
        end = v.ends_mm[got.tip_end[v.name]]
        check(np.hypot(end[0] - tip[0], end[1] - tip[1]) < 30.0,
              f"{v.name}: picked the wrong end of the shadow")


# --------------------------------------------------------------------------- #
# the number that decides whether this was worth building
# --------------------------------------------------------------------------- #
def _accuracy_sweep(n=600, seed=5):
    rng = np.random.default_rng(seed)
    _, _, calibs, cams = S.synthetic_pair(1280, 720, separation_deg=90.0,
                                          click_px=1.0)
    single, fused = [], []
    for _ in range(n):
        r = rng.uniform(5.0, 165.0)
        a = rng.uniform(0.0, 360.0)
        tip = geo.polar_to_board(r, a)
        azimuth = geo.angle_of(*tip) + rng.normal(0, 25.0)
        elevation = rng.uniform(8.0, 45.0)
        buried = rng.uniform(4.0, 16.0)

        views = []
        for i, (calib, cam) in enumerate(zip(calibs, cams)):
            v = shadow(calib, cam, tip, azimuth, elevation,
                       buried_mm=buried, noise_px=0.7, rng=rng)
            v.name = f"cam{i}"
            views.append(v)

        single.append(np.hypot(views[0].single_mm[0] - tip[0],
                               views[0].single_mm[1] - tip[1]))
        got = fusion.fuse(views, primary="cam0")
        fused.append(np.hypot(got.board_mm[0] - tip[0], got.board_mm[1] - tip[1]))
    return np.array(single), np.array(fused)


def test_fusion_is_more_accurate_than_one_camera():
    single, fused = _accuracy_sweep()
    s_med, f_med = np.median(single), np.median(fused)
    s_p90, f_p90 = np.percentile(single, 90), np.percentile(fused, 90)
    print(f"     single: median {s_med:5.2f} mm  p90 {s_p90:5.2f} mm")
    print(f"     fused : median {f_med:5.2f} mm  p90 {f_p90:5.2f} mm")
    check(f_med < s_med * 0.7,
          f"fusion should clearly beat one camera: {f_med:.2f} vs {s_med:.2f} mm")
    # The gates exist to stop an ill-conditioned crossing throwing the answer
    # off the board, so the tail must not get worse either.
    check(f_p90 <= s_p90 * 1.1,
          f"fusion must not worsen the tail: p90 {f_p90:.2f} vs {s_p90:.2f} mm")
    check(f_p90 < 25.0, f"p90 of {f_p90:.2f} mm is too loose to score a treble")


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
