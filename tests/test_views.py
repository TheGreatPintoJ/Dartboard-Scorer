"""Cameras as things with positions: the config, the pose, and /api/views.

The point of recording where a camera is, is that the calibration already knows
- so what the operator types is an *expectation*, and the two disagreeing is how
you find out a camera has been knocked or that two identical devices came up in
the opposite order. These tests hold that behaviour down, along with the config
shape that lets there be more than one camera at all.
"""

import json
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dart_scorer import geometry as geo                      # noqa: E402
from dart_scorer import synthetic as S                       # noqa: E402
from dart_scorer.calibration import measure_pose, tip_mode_for_bearing  # noqa: E402
from dart_scorer.config import AppConfig                     # noqa: E402
from dart_scorer.engine import ScoringEngine                 # noqa: E402
from dart_scorer.synthetic import DemoSource                 # noqa: E402
from dart_scorer.webapp import serve                         # noqa: E402


def check(condition, message):
    assert condition, message


# --------------------------------------------------------------------------- #
# reading the camera's position out of the calibration
# --------------------------------------------------------------------------- #
def test_pose_matches_where_the_camera_actually_is():
    for azimuth in (0.0, 35.0, 90.0, 150.0, 220.0, 300.0):
        for elevation in (10.0, 25.0, 45.0):
            _, _, calib, _ = S.synthetic_camera_3d(
                1280, 720, azimuth_deg=azimuth, elevation_deg=elevation)
            pose = measure_pose(calib)
            bearing_err = abs((pose["bearing_deg"] - azimuth + 180) % 360 - 180)
            check(bearing_err < 0.5,
                  f"bearing read {pose['bearing_deg']} for a camera at {azimuth}")
            check(abs(pose["elevation_deg"] - elevation) < 0.5,
                  f"elevation read {pose['elevation_deg']} for {elevation}")


def test_pose_does_not_need_to_know_the_lens():
    """Elevation is a ratio of two scales, so the focal length cancels."""
    poses = [measure_pose(S.synthetic_camera_3d(
        1280, 720, azimuth_deg=35, elevation_deg=25, fov_deg=fov,
        distance_mm=d)[2]) for fov, d in ((35, 600), (55, 1200), (95, 4000))]
    for pose in poses[1:]:
        check(abs(pose["bearing_deg"] - poses[0]["bearing_deg"]) < 0.5
              and abs(pose["elevation_deg"] - poses[0]["elevation_deg"]) < 0.5,
              f"the answer moved with the lens: {poses}")


def test_pose_survives_imperfect_calibration_clicks():
    worst = 0.0
    for seed in range(20):
        _, _, calib, _ = S.synthetic_camera_3d(
            1280, 720, azimuth_deg=35, elevation_deg=25, click_px=2.0, seed=seed)
        pose = measure_pose(calib)
        worst = max(worst, abs((pose["bearing_deg"] - 35 + 180) % 360 - 180))
    check(worst < 5.0, f"2 px of click slop moved the bearing {worst:.1f} deg")


def test_an_upright_camera_reads_no_roll():
    _, _, calib, _ = S.synthetic_camera_3d(1280, 720, azimuth_deg=40)
    check(abs(measure_pose(calib)["roll_deg"]) < 1.0,
          "a level camera should not report itself tilted")


def test_tip_mode_follows_the_readme_table():
    # "lowest - the camera is above the board", and so on round the board.
    for bearing, want in ((0, "leftmost"), (90, "lowest"),
                          (180, "rightmost"), (270, "highest")):
        got = tip_mode_for_bearing(bearing, elevation_deg=15.0)
        check(got == want, f"a camera at {bearing} deg should use {want}, got {got}")
    check(tip_mode_for_bearing(0, elevation_deg=70.0) == "centre",
          "a camera looking down at the board uses the usual rule")


# --------------------------------------------------------------------------- #
# the config shape
# --------------------------------------------------------------------------- #
def test_a_second_camera_can_be_added_and_removed():
    cfg = AppConfig()
    changed = cfg.apply({"views": [{"name": "side", "source": "1",
                                    "placement": {"bearing_deg": "90"}}]})
    check("views.side.added" in changed, f"expected an add, got {changed}")
    check([v.name for v in cfg.views_all()] == ["primary", "side"],
          "the primary should come first")
    check(cfg.view("side").placement.bearing_deg == 90.0,
          "nested placement should be merged, not dropped")
    cfg.apply({"views": [{"name": "side", "remove": True}]})
    check([v.name for v in cfg.views_all()] == ["primary"], "removal should stick")


def test_the_primary_is_not_reachable_through_the_views_list():
    """It lives at config.camera, so patching it twice would be ambiguous."""
    cfg = AppConfig()
    cfg.apply({"views": [{"name": "primary", "source": "99"}]})
    check(cfg.camera.source == "0", "the views list must not touch the primary")


def test_nonsense_values_are_refused():
    cfg = AppConfig()
    cfg.apply({"camera": {"stream": "sideways", "placement": {"rotate": "45"}}})
    check(cfg.camera.stream == "rgb", "an unknown stream should be ignored")
    check(cfg.camera.placement.rotate == 0, "45 degrees is not a rotation we can do")


def test_an_empty_views_list_is_exactly_todays_config():
    check(AppConfig().views == [], "views must default to empty")
    check(AppConfig().camera.name == "primary", "the primary needs a stable name")


# --------------------------------------------------------------------------- #
# the HTTP surface
# --------------------------------------------------------------------------- #
class _Server:
    def __init__(self, bearing=90.0):
        self._tmp = tempfile.TemporaryDirectory()
        d = Path(self._tmp.name)
        cfg = AppConfig()
        cfg.camera.source = "demo"
        cfg.camera.placement.bearing_deg = bearing
        cfg.calibration_path = str(d / "calibration.json")
        cfg.log_path = str(d / "throws.csv")
        self.engine = ScoringEngine(cfg, str(d / "config.json"))
        self.engine.set_calibration(DemoSource().calibration.image_points)

    def __enter__(self):
        self.engine.start()
        self.httpd = serve(self.engine, "127.0.0.1", 0)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        # Building the demo board takes several seconds on a Raspberry Pi.
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline and self.engine._raw is None:
            time.sleep(0.02)
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.engine.stop()
        self._tmp.cleanup()

    def get(self, path):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}") as r:
            return json.loads(r.read())

    def post(self, path, body):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}{path}", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read())


def test_views_endpoint_reports_expected_beside_measured():
    with _Server() as s:
        info = s.get("/api/views")
        view = info["views"][0]
        check(view["role"] == "primary", f"expected a primary, got {view['role']}")
        check(view["measured"] is not None,
              "a calibrated view should report where it actually is")
        check(view["placement"]["bearing_deg"] == 90.0,
              "the expected bearing should come back as entered")
        check(view["drift_deg"] is not None, "drift should be computed")


def test_a_camera_that_is_not_where_you_said_is_flagged():
    # The demo board's camera is nowhere near a bearing of 90.
    with _Server(bearing=90.0) as s:
        view = s.get("/api/views")["views"][0]
        check(view["drift_deg"] > 15.0,
              f"expected a large drift, got {view['drift_deg']}")
        check(any("moved" in w for w in view["warnings"]),
              f"it should say so: {view['warnings']}")


def test_entering_the_measured_bearing_clears_the_warning():
    with _Server() as s:
        measured = s.get("/api/views")["views"][0]["measured"]["bearing_deg"]
        info = s.post("/api/views", {"name": "primary",
                                     "placement": {"bearing_deg": measured}})
        view = info["views"][0]
        check(view["drift_deg"] < 1.0, f"drift should be gone, got {view['drift_deg']}")
        check(not any("moved" in w for w in view["warnings"]),
              f"no warning expected: {view['warnings']}")


def test_a_second_view_gets_a_fusion_verdict():
    with _Server() as s:
        measured = s.get("/api/views")["views"][0]["measured"]["bearing_deg"]
        # Put the second camera a quarter turn away: good geometry for fusing.
        info = s.post("/api/views", {
            "name": "side", "add": True, "source": "1",
            "placement": {"bearing_deg": (measured + 90) % 360}})
        check(len(info["views"]) == 2, "the view should have been added")
        check(info["fusion"]["ready"], f"90 deg apart should fuse: {info['fusion']}")

        info = s.post("/api/views", {"name": "side",
                                     "placement": {"bearing_deg": measured + 5}})
        check(not info["fusion"]["ready"],
              "5 deg apart is too close to fuse")
        check("too close" in info["fusion"]["reason"],
              f"it should say why: {info['fusion']['reason']}")


def test_kinect_stream_selection_is_offered_only_for_a_kinect():
    with _Server() as s:
        info = s.post("/api/views", {"name": "k", "add": True,
                                     "source": "kinect:0", "stream": "ir"})
        kinect = [v for v in info["views"] if v["name"] == "k"][0]
        check(kinect["kinect"], "a kinect: source should be recognised")
        check(kinect["stream"] == "ir", f"stream should be ir, got {kinect['stream']}")
        check(not info["views"][0]["kinect"],
              "the demo board is not a Kinect")
        check("available" in info["kinect"],
              "the UI needs to know whether libfreenect is there")


def test_a_kinect_too_close_to_the_board_is_flagged():
    with _Server() as s:
        info = s.post("/api/views", {"name": "k", "add": True,
                                     "source": "kinect:0",
                                     "placement": {"distance_mm": 500}})
        kinect = [v for v in info["views"] if v["name"] == "k"][0]
        check(any("800" in w for w in kinect["warnings"]),
              f"expected a minimum-range warning: {kinect['warnings']}")


def test_turning_the_picture_asks_for_a_fresh_calibration():
    with _Server() as s:
        result = s.engine.apply_config({"camera": {"placement": {"rotate": 90}}})
        check("placement.rotate" in result["recalibrate"],
              f"a quarter turn moves the board in frame: {result['recalibrate']}")


def test_an_unknown_view_is_a_404_not_a_silent_no_op():
    with _Server() as s:
        try:
            s.post("/api/views", {"name": "nope", "stream": "ir"})
        except urllib.error.HTTPError as err:
            check(err.code == 404, f"expected 404, got {err.code}")
            return
        raise AssertionError("patching a view that does not exist should fail")


def test_the_camera_probe_still_reports_the_old_shape():
    with _Server() as s:
        data = s.get("/api/cameras")
        check(isinstance(data.get("cameras"), list), "the old key must survive")
        check(isinstance(data.get("devices"), list), "the new one carries labels")


# --------------------------------------------------------------------------- #
# tip_mode, end to end
# --------------------------------------------------------------------------- #
def test_automatic_tip_mode_picks_from_the_calibration():
    cfg = AppConfig()
    cfg.camera.source = "demo"
    cfg.detector.tip_mode = "auto"
    with tempfile.TemporaryDirectory() as d:
        cfg.calibration_path = str(Path(d) / "c.json")
        engine = ScoringEngine(cfg, str(Path(d) / "cfg.json"))
        engine.set_calibration(DemoSource().calibration.image_points)
        pose = measure_pose(engine.calibration)
        want = tip_mode_for_bearing(pose["bearing_deg"], pose["elevation_deg"])
        check(engine.detector.tip_mode == want,
              f"detector took {engine.detector.tip_mode}, expected {want}")
        check(engine.detector.tip_mode != "auto",
              "'auto' has to be resolved to a real mode, not passed through")


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
