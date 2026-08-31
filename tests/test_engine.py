"""The scoring thread has to survive things going wrong.

It is a daemon thread that nothing joins and nothing watches. If an exception
escapes it, scoring stops for the life of the process while ``_running`` stays
true - so the browser keeps polling a status that will never change again and
``/healthz`` still answers ok. A camera backend that raises on open rather than
returning None is the obvious way in, and adding one (a Kinect behind an
optional import, say) is exactly when it would bite.
"""

import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dart_scorer import engine as engine_mod                 # noqa: E402
from dart_scorer.config import AppConfig                     # noqa: E402
from dart_scorer.engine import ScoringEngine                 # noqa: E402


def check(condition, message):
    assert condition, message


def wait_for(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def scoring_threads():
    return [t for t in threading.enumerate() if t.name == "scoring" and t.is_alive()]


class _Engine:
    """A started engine in a scratch directory, cleaned up on exit."""

    def __init__(self, source="demo", calibrate=False):
        self._tmp = tempfile.TemporaryDirectory()
        cfg = AppConfig()
        cfg.camera.source = source
        cfg.calibration_path = str(Path(self._tmp.name) / "calibration.json")
        cfg.log_path = str(Path(self._tmp.name) / "throws.csv")
        self.engine = ScoringEngine(cfg, str(Path(self._tmp.name) / "config.json"))
        if calibrate:
            # What cmd_serve does for a demo board: there is no physical camera
            # to mark up, so install the calibration that matches it. Note this
            # happens before any frame has arrived, so it carries no frame size
            # until the loop fills one in.
            from dart_scorer.synthetic import DemoSource
            self.engine.set_calibration(DemoSource().calibration.image_points)

    def __enter__(self):
        self.engine.start()
        return self.engine

    def __exit__(self, *exc):
        self.engine.stop()
        self._tmp.cleanup()


# --------------------------------------------------------------------------- #
def test_a_backend_that_raises_does_not_kill_scoring():
    original = engine_mod.open_source
    calls = []

    def exploding(cfg):
        calls.append(cfg)
        raise RuntimeError("libfreenect is not installed")

    engine_mod.open_source = exploding
    try:
        with _Engine() as eng:
            check(wait_for(lambda: eng.status()["state"] == "no camera"),
                  f"expected 'no camera', got {eng.status()['state']!r}")
            err = eng.status()["error"] or ""
            check("libfreenect is not installed" in err,
                  f"the reason should reach the browser, got {err!r}")
            # The real regression: it must still be trying, not dead.
            before = len(calls)
            check(wait_for(lambda: len(calls) > before, timeout=6.0),
                  "the engine stopped retrying - the scoring thread died")
            check(scoring_threads(), "the scoring thread should still be alive")
    finally:
        engine_mod.open_source = original


def test_the_demo_board_still_scores():
    """The single-camera path must be untouched by any of this."""
    with _Engine(calibrate=True) as eng:
        check(wait_for(lambda: eng._raw is not None), "no frames arrived")
        check(wait_for(lambda: eng.status()["calibrated"]),
              "the demo source should install its own calibration")
        check(wait_for(lambda: eng.status()["state"] == "ready", timeout=10.0),
              f"expected 'ready', got {eng.status()['state']!r}")

        eng.command("throw", label="T20")
        check(wait_for(lambda: eng.status()["session"]["visit"], timeout=10.0),
              "a thrown dart should be scored")
        labels = [d["label"] for d in eng.status()["session"]["visit"]]
        check(labels == ["T20"], f"expected a T20, got {labels}")


def test_no_scoring_thread_is_left_behind():
    with _Engine():
        pass
    check(wait_for(lambda: not scoring_threads()),
          "stop() should join the scoring thread")


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
