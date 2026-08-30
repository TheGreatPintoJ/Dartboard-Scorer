"""End-to-end test of the web service against the built-in demo board.

Starts the real server on an ephemeral port, throws darts at the synthetic
board and checks that the HTTP surface reports them. No camera, no browser.
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

from dart_scorer.config import AppConfig                # noqa: E402
from dart_scorer.engine import ScoringEngine            # noqa: E402
from dart_scorer.synthetic import DemoSource            # noqa: E402
from dart_scorer.webapp import Handler, serve           # noqa: E402

BASE = None
ENGINE = None


def check(condition, message):
    assert condition, message


def get(path, raw=False, timeout=10):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as res:
        body = res.read()
        return (res, body) if raw else json.loads(body)


def post(path, payload=None, raw=False, timeout=10):
    request = urllib.request.Request(
        BASE + path, data=json.dumps(payload or {}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as res:
        body = res.read()
        return (res, body) if raw else json.loads(body)


def wait_for(predicate, timeout=15.0, what="condition"):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = get("/api/status")
        if predicate(status):
            return status
        time.sleep(0.1)
    raise AssertionError(f"timed out waiting for {what}; state={status['state']}")


# --------------------------------------------------------------------------- #
def test_health_and_page():
    check(get("/healthz")["ok"], "healthz should report ok")
    res, body = get("/", raw=True)
    check(b"<title>Dart Scorer</title>" in body, "the control page should be served")
    check(res.headers["Content-Type"].startswith("text/html"), "page is HTML")


def test_status_reports_demo_and_calibration():
    status = get("/api/status")
    check(status["demo"], "the demo source should be active")
    check(status["calibrated"], "the demo calibration should be installed")
    check(len(status["calibration_points"]) >= 4, "landmarks should be recorded")


def test_camera_feeds_frames():
    wait_for(lambda s: s["fps"] > 0, what="frames to arrive")
    res, body = get("/snapshot.jpg", raw=True)
    check(body[:2] == b"\xff\xd8", "snapshot should be a JPEG")
    check(res.headers["Content-Type"] == "image/jpeg", "content type should be JPEG")
    _, flat = get("/rectified.jpg", raw=True)
    check(flat[:2] == b"\xff\xd8", "the flat view should be a JPEG")


def test_mjpeg_stream():
    with urllib.request.urlopen(BASE + "/stream.mjpg", timeout=10) as res:
        check("multipart/x-mixed-replace" in res.headers["Content-Type"],
              "stream should be multipart")
        chunk = res.read(40000)
    check(b"--dartframe" in chunk, "stream should carry frame boundaries")
    check(b"\xff\xd8" in chunk, "stream should carry JPEG data")


def test_events_stream_opens_with_status():
    # Read line by line: the stream stays open, so a fixed-size read would block.
    with urllib.request.urlopen(BASE + "/api/events", timeout=10) as res:
        check(res.headers["Content-Type"].startswith("text/event-stream"), "SSE type")
        lines = [res.readline().decode("utf-8", "replace") for _ in range(4)]
    payload = "".join(lines)
    check(": connected" in payload, "the stream should greet the client")
    check("data:" in payload, "the stream should open with a status payload")
    check('"state"' in payload, "that payload should carry the engine state")


def test_thrown_dart_is_scored():
    wait_for(lambda s: s["state"] == "ready", what="the board to be learned")
    post("/api/command", {"command": "throw", "target": "T20"})
    status = wait_for(lambda s: len(s["session"]["visit"]) == 1, what="the dart")
    dart = status["session"]["visit"][0]
    check(dart["label"] == "T20", f"expected T20, got {dart['label']}")
    check(status["session"]["visit_total"] == 60, "T20 is 60")

    post("/api/command", {"command": "throw", "target": "D16"})
    status = wait_for(lambda s: len(s["session"]["visit"]) == 2, what="the second dart")
    check(status["session"]["visit"][1]["label"] == "D16", "second dart is D16")
    check(status["session"]["visit_total"] == 92, "60 + 32 = 92")


def test_undo_and_end_visit():
    status = post("/api/command", {"command": "undo"})
    check(len(status["session"]["visit"]) == 1, "undo should drop the last dart")
    status = post("/api/command", {"command": "end_turn"})
    check(status["session"]["visit"] == [], "ending the visit clears it")
    check(status["session"]["players"][0]["darts"] == 1, "one dart was recorded")


def test_pulling_darts_ends_the_visit():
    post("/api/command", {"command": "pull_darts"})
    post("/api/command", {"command": "relearn"})
    wait_for(lambda s: s["state"] == "ready", what="a clean board")
    post("/api/command", {"command": "throw", "target": "19"})
    wait_for(lambda s: len(s["session"]["visit"]) == 1, what="the dart")
    before = get("/api/status")["session"]["players"][0]["darts"]
    post("/api/command", {"command": "pull_darts"})
    status = wait_for(lambda s: s["session"]["players"][0]["darts"] > before,
                      what="the visit to close when the darts came out")
    check(status["session"]["visit"] == [], "the new visit starts empty")


def test_config_round_trip():
    config = post("/api/config", {"game": {"players": ["Kyle", "Sam"],
                                           "start_score": 501}})
    check(config["game"]["players"] == ["Kyle", "Sam"], "players should be set")
    status = get("/api/status")
    check(status["session"]["players"][0]["remaining"] == 501, "501 start")
    check(len(status["session"]["players"]) == 2, "two players")

    post("/api/config", {"detector": {"min_area": 200}})
    check(get("/api/config")["detector"]["min_area"] == 200, "detector setting applied")
    check(ENGINE.detector.min_area == 200, "the live detector should be rebuilt")


def test_unchanged_settings_do_not_disturb_the_game():
    """The UI submits the whole form; only real changes should have effects."""
    post("/api/config", {"game": {"players": ["Kyle", "Sam"], "start_score": 501}})
    post("/api/command", {"command": "relearn"})
    wait_for(lambda s: s["state"] == "ready", what="the board to be learned")
    post("/api/command", {"command": "throw", "target": "T20"})
    wait_for(lambda s: len(s["session"]["visit"]) == 1, what="a dart in the visit")

    # Same values as are already set, plus one unrelated tweak.
    post("/api/config", {"game": {"players": ["Kyle", "Sam"], "start_score": 501,
                                  "double_out": True},
                         "camera": {"stream_quality": 70}})
    status = get("/api/status")
    check(len(status["session"]["visit"]) == 1,
          "resubmitting identical game settings must not restart the game")
    check(status["config"]["camera"]["stream_quality"] == 70, "the tweak applied")

    post("/api/config", {"game": {"start_score": 301}})
    check(get("/api/status")["session"]["visit"] == [],
          "a real game change does start a new game")
    post("/api/config", {"game": {"players": ["Player 1"], "start_score": 0}})


def test_config_coerces_browser_strings():
    post("/api/config", {"detector": {"settle_frames": "6"},
                         "game": {"double_out": "false"}})
    config = get("/api/config")
    check(config["detector"]["settle_frames"] == 6, "numbers arrive as strings")
    check(config["game"]["double_out"] is False, "checkboxes arrive as strings")
    post("/api/config", {"detector": {"settle_frames": 4}, "game": {"double_out": True}})


def test_camera_endpoint_reports_controls():
    info = get("/api/camera")
    check("controls" in info, "the camera endpoint should list controls")
    # The demo board has no real capture device behind it.
    check(info["demo"] or info["open"], "it should say whether a camera is open")


def test_camera_controls_are_rejected_when_empty():
    try:
        post("/api/camera", {"controls": {}})
        raise AssertionError("an empty control set should be rejected")
    except urllib.error.HTTPError as exc:
        check(exc.code == 400, f"expected 400, got {exc.code}")


def test_camera_controls_persist_in_config():
    post("/api/camera", {"controls": {"focus": 30, "autofocus": 0}})
    controls = get("/api/config")["camera"]["controls"]
    check(controls.get("focus") == 30, f"focus should be stored, got {controls}")
    check(controls.get("autofocus") == 0, "autofocus should be stored")

    # Setting one control must not wipe the others.
    post("/api/camera", {"controls": {"brightness": 128}})
    controls = get("/api/config")["camera"]["controls"]
    check(controls.get("focus") == 30, "focus should survive an unrelated change")
    check(controls.get("brightness") == 128, "brightness should be stored")

    # Blanking a control stops us setting it at all.
    post("/api/camera", {"controls": {"focus": None}})
    check("focus" not in get("/api/config")["camera"]["controls"],
          "a null should remove the control")
    post("/api/config", {"camera": {"controls": {"autofocus": None,
                                                 "brightness": None}}})


def test_zoom_change_warns_that_calibration_is_stale():
    result = post("/api/camera", {"controls": {"zoom": 2}})
    check(result["recalibrate"] == ["zoom"],
          f"zoom moves the board in frame, expected a warning, got {result['recalibrate']}")
    result = post("/api/camera", {"controls": {"brightness": 100}})
    check(result["recalibrate"] == [], "brightness does not move the board")
    post("/api/config", {"camera": {"controls": {"zoom": None, "brightness": None}}})


def test_capture_settings_round_trip():
    post("/api/config", {"camera": {"fourcc": "MJPG", "buffer_size": 1,
                                    "backend": "auto", "stream_fps": 0}})
    cam = get("/api/config")["camera"]
    check(cam["fourcc"] == "MJPG", "MJPG is the default pixel format")
    check(cam["buffer_size"] == 1, "buffer depth 1 keeps the newest frame")
    check(cam["stream_fps"] == 0, "0 means publish every captured frame")


def test_calibration_endpoints():
    points = get("/api/calibration")["points"]
    _, preview = post("/api/calibration/preview", {"points": points}, raw=True)
    check(preview[:2] == b"\xff\xd8", "preview should render a JPEG")

    try:
        post("/api/calibration", {"points": points[:2]})
        raise AssertionError("three landmarks should be rejected")
    except urllib.error.HTTPError as exc:
        check(exc.code == 400, f"expected 400, got {exc.code}")

    result = post("/api/calibration", {"points": points})
    check(len(result["points"]) == len(points), "calibration should be accepted")
    check(get("/api/status")["calibrated"], "still calibrated")


def test_unknown_command_is_rejected():
    try:
        post("/api/command", {"command": "explode"})
        raise AssertionError("unknown commands should be rejected")
    except urllib.error.HTTPError as exc:
        check(exc.code == 400, f"expected 400, got {exc.code}")


def test_missing_route():
    try:
        get("/api/nope")
        raise AssertionError("unknown routes should 404")
    except urllib.error.HTTPError as exc:
        check(exc.code == 404, f"expected 404, got {exc.code}")


def test_token_is_enforced_when_set():
    Handler.token = "secret"
    try:
        try:
            get("/api/status")
            raise AssertionError("a token should be required")
        except urllib.error.HTTPError as exc:
            check(exc.code == 401, f"expected 401, got {exc.code}")
        check(get("/api/status?token=secret")["state"], "the token in the URL works")
        request = urllib.request.Request(BASE + "/api/status",
                                         headers={"X-Auth-Token": "secret"})
        with urllib.request.urlopen(request, timeout=10) as res:
            check(json.loads(res.read())["state"], "the header works too")
        check(get("/healthz")["ok"], "the health probe stays open")
    finally:
        Handler.token = None


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    workdir = Path(tempfile.mkdtemp(prefix="dartweb-"))
    config = AppConfig()
    config.camera.source = "demo"
    config.calibration_path = str(workdir / "calibration.json")
    config.log_path = str(workdir / "throws.csv")

    ENGINE = ScoringEngine(config, config_path=str(workdir / "config.json"))
    ENGINE.set_calibration(DemoSource().calibration.image_points)
    ENGINE.start()

    httpd = serve(ENGINE, "127.0.0.1", 0)
    BASE = f"http://127.0.0.1:{httpd.server_address[1]}"
    threading.Thread(target=httpd.serve_forever, daemon=True).start()

    order = [
        test_health_and_page, test_status_reports_demo_and_calibration,
        test_camera_feeds_frames, test_mjpeg_stream,
        test_events_stream_opens_with_status, test_thrown_dart_is_scored,
        test_undo_and_end_visit, test_pulling_darts_ends_the_visit,
        test_config_round_trip, test_unchanged_settings_do_not_disturb_the_game,
        test_config_coerces_browser_strings,
        test_camera_endpoint_reports_controls,
        test_camera_controls_are_rejected_when_empty,
        test_camera_controls_persist_in_config,
        test_zoom_change_warns_that_calibration_is_stale,
        test_capture_settings_round_trip,
        test_calibration_endpoints, test_unknown_command_is_rejected,
        test_missing_route, test_token_is_enforced_when_set,
    ]
    failures = 0
    for fn in order:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:
            failures += 1
            print(f"FAIL {fn.__name__}: {exc}")

    httpd.shutdown()
    httpd.server_close()
    ENGINE.stop()
    print(f"\n{failures} failure(s)")
    sys.exit(1 if failures else 0)
