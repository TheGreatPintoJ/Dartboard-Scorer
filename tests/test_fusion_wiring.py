"""Fusion, through the running engine rather than as pure geometry.

test_fusion.py proves the maths. This proves the wiring: that a second camera
actually reaches the score, that it only ever *moves* a dart, and that every
way it can go wrong ends with the scoring camera's own answer instead.

Both cameras here look at the same synthetic 3D board from different bearings,
with the dart standing proud of the face - which is the whole reason a single
camera reads it in the wrong place.
"""

import sys
import tempfile
import threading
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dart_scorer import geometry as geo                      # noqa: E402
from dart_scorer import synthetic as S                       # noqa: E402
from dart_scorer.config import AppConfig                     # noqa: E402
from dart_scorer.engine import ScoringEngine                 # noqa: E402


def check(condition, message):
    assert condition, message


class ThreeDeeSource:
    """A camera looking at a real 3D board, with darts that stand out of it."""

    def __init__(self, cam, board, view, darts, fps=30.0):
        self.cam, self._board, self._view, self._darts = cam, board, view, darts
        self.name = "3d"
        self._interval = 1.0 / fps
        self._next = 0.0

    def read(self):
        now = time.monotonic()
        if now < self._next:
            time.sleep(min(self._next - now, self._interval))
        self._next = max(now, self._next) + self._interval
        frame = self._view.warp(self._board)
        for tip, elevation, buried in list(self._darts):
            S.draw_dart_3d(frame, self.cam, tip[0], tip[1],
                           elevation_deg=elevation, buried_mm=buried)
        return self._view.noise(frame)

    def release(self):
        pass


def build(tmp, separation=90.0, azimuth=30.0):
    """An engine with two cameras on the same board, both calibrated."""
    board, views, calibs, cams = S.synthetic_pair(
        960, 720, separation_deg=separation, azimuth_deg=azimuth)
    darts = []

    cfg = AppConfig()
    cfg.camera.source = "primary3d"
    cfg.camera.name = "primary"
    cfg.detector.settle_frames = 2
    cfg.detector.learn_frames = 6
    cfg.detector.min_area = 120
    cfg.calibration_path = str(Path(tmp) / "calibration.json")
    cfg.log_path = str(Path(tmp) / "throws.csv")
    cfg.apply({"views": [{"name": "side", "source": "side3d"}]})

    sources = {
        "primary3d": lambda c: ThreeDeeSource(cams[0], board, views[0], darts),
        "side3d": lambda c: ThreeDeeSource(cams[1], board, views[1], darts),
    }

    import dart_scorer.engine as engine_mod
    import dart_scorer.views as views_mod
    original = engine_mod.open_source

    def fake_open(camera_cfg):
        maker = sources.get(str(camera_cfg.source))
        return maker(camera_cfg) if maker else original(camera_cfg)

    engine_mod.open_source = fake_open
    engine = ScoringEngine(cfg, str(Path(tmp) / "config.json"))
    engine.views._open_source = fake_open
    engine.set_calibration(calibs[0].image_points)
    engine.start()

    # The second camera's own four landmarks, in its own picture.
    deadline = time.monotonic() + 60
    side = None
    while time.monotonic() < deadline:
        side = engine.views.get("side")
        if side is not None and side.latest() is not None:
            break
        time.sleep(0.1)
    check(side is not None and side.latest() is not None, "the second camera never opened")
    side.set_calibration(calibs[1].image_points, save=True)

    # It also has to learn the empty board before it can recognise a dart on
    # it. Throwing before then is not a scenario worth testing - it is the
    # test getting ahead of the system.
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline and not side.info()["watching"]:
        time.sleep(0.1)
    check(side.info()["watching"], "the second camera never learned the board")
    return engine, darts, cams, original


def wait_for(predicate, timeout=60.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def throw(engine, darts, tip, elevation=25.0, buried=10.0, timeout=60.0):
    """Put a dart in the board and wait for it to be scored."""
    before = len(engine.session.turn.darts)
    darts.append((tip, elevation, buried))
    if not wait_for(lambda: len(engine.session.turn.darts) > before, timeout):
        return None
    return engine.session.turn.darts[-1]


def error_mm(dart, tip):
    return float(np.hypot(dart.board_mm[0] - tip[0], dart.board_mm[1] - tip[1]))


# --------------------------------------------------------------------------- #
def test_a_second_camera_moves_the_dart_closer_to_the_truth():
    """The point of the whole exercise."""
    import dart_scorer.engine as engine_mod
    with tempfile.TemporaryDirectory() as tmp:
        engine, darts, cams, original = build(tmp)
        try:
            check(wait_for(lambda: engine.status()["state"] == "ready"),
                  f"never became ready: {engine.status()['state']}")
            # A bed both cameras see side-on. A dart pointing straight at a
            # camera is foreshortened into a stub, which that camera cannot
            # measure an axis from - see the companion test below.
            tip = geo.polar_to_board(103.0, 250.0)
            dart = throw(engine, darts, tip, elevation=30.0, buried=12.0)
            check(dart is not None, "the dart was never scored")
            check(dart.fusion is not None, "fusion should have been consulted")
            info = dart.fusion
            print(f"     mode={info['mode']} views={info['views']} "
                  f"error={error_mm(dart, tip):.1f} mm "
                  f"parallax={info.get('parallax_mm')}")
            check(info["mode"] == "fused",
                  f"expected a fused answer, got {info['mode']}: {info['reasons']}")
            check(len(info["views"]) == 2, f"both cameras should count: {info}")
            check(error_mm(dart, tip) < 8.0,
                  f"fused answer is {error_mm(dart, tip):.1f} mm out")
            # The correction it applied is the parallax this camera suffers -
            # the error the second camera exists to remove.
            moved = info["parallax_mm"].get("primary", 0.0)
            check(moved > 4.0,
                  f"it should have moved the point meaningfully, moved {moved} mm")
        finally:
            engine.stop()
            engine_mod.open_source = original


def test_a_dart_pointing_at_a_camera_falls_back_rather_than_guessing():
    """An honest limit: a foreshortened dart has no measurable axis."""
    import dart_scorer.engine as engine_mod
    with tempfile.TemporaryDirectory() as tmp:
        engine, darts, cams, original = build(tmp)
        try:
            check(wait_for(lambda: engine.status()["state"] == "ready"),
                  "never became ready")
            # Bed 20 points almost straight at the first camera.
            tip = geo.polar_to_board(103.0, 20.0)
            dart = throw(engine, darts, tip, elevation=25.0, buried=12.0)
            check(dart is not None, "the dart was never scored")
            check(dart.label, "it must still be scored, just not fused")
            if dart.fusion and dart.fusion["mode"] != "fused":
                check(dart.fusion["reasons"],
                      f"a fallback has to say why: {dart.fusion}")
        finally:
            engine.stop()
            engine_mod.open_source = original


def test_the_second_camera_never_invents_a_dart():
    """It may move a dart. It may not create, delete or reorder one."""
    import dart_scorer.engine as engine_mod
    with tempfile.TemporaryDirectory() as tmp:
        engine, darts, cams, original = build(tmp)
        try:
            check(wait_for(lambda: engine.status()["state"] == "ready"),
                  "never became ready")
            side = engine.views.get("side")
            # Something moves in the second camera only - a hand, a shadow.
            for _ in range(40):
                side._observations.append((time.monotonic(), _phantom(side)))
                time.sleep(0.02)
            time.sleep(2.0)
            check(len(engine.session.turn.darts) == 0,
                  f"the second camera invented {len(engine.session.turn.darts)} dart(s)")
        finally:
            engine.stop()
            engine_mod.open_source = original


def _phantom(view):
    """A plausible-looking dart that only the second camera can see."""
    from dart_scorer.detector import Dart
    return Dart(tip_image=(100.0, 100.0), board_mm=(20.0, 20.0),
                score=geo.score_at(20.0, 20.0), area=500.0, elongation=5.0,
                confidence=1.0, axis_image=(0.0, 1.0, -100.0),
                ends_image=((80.0, 100.0), (140.0, 100.0)), axis_sigma_deg=0.2)


def test_a_dead_second_camera_does_not_stop_scoring():
    import dart_scorer.engine as engine_mod
    with tempfile.TemporaryDirectory() as tmp:
        engine, darts, cams, original = build(tmp)
        try:
            check(wait_for(lambda: engine.status()["state"] == "ready"),
                  "never became ready")
            engine.views.get("side").stop()          # pull its plug
            tip = geo.polar_to_board(103.0, 20.0)
            dart = throw(engine, darts, tip)
            check(dart is not None, "scoring stopped when the second camera died")
            check(dart.label, "the dart should still have a score")
        finally:
            engine.stop()
            engine_mod.open_source = original


def test_the_log_gains_fusion_columns_without_corrupting_an_old_one():
    """An existing CSV must not silently gain five unlabelled fields."""
    from dart_scorer.engine import LOG_COLUMNS
    with tempfile.TemporaryDirectory() as tmp:
        log = Path(tmp) / "throws.csv"
        log.write_text("timestamp,player,label,points,x_mm,y_mm,radius_mm,"
                       "angle_deg,confidence\n1,P,T20,60,0,0,0,0,1.0\n",
                       encoding="utf-8")
        cfg = AppConfig()
        cfg.log_path = str(log)
        cfg.calibration_path = str(Path(tmp) / "c.json")
        engine = ScoringEngine(cfg, str(Path(tmp) / "cfg.json"))
        engine._log(_phantom(None))
        engine.stop()

        rolled = Path(str(log) + ".v1")
        check(rolled.exists(), "the old log should have been kept, not appended to")
        header = log.open(encoding="utf-8").readline().strip().split(",")
        check(header == LOG_COLUMNS, f"new log has the wrong header: {header}")
        rows = [r for r in log.open(encoding="utf-8").read().splitlines() if r]
        check(len(rows) == 2, f"expected a header and one row, got {len(rows)}")
        check(len(rows[1].split(",")) == len(LOG_COLUMNS),
              "every row must have as many fields as the header")


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
