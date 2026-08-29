"""Visit and X01 bookkeeping checks - no camera or OpenCV needed."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dart_scorer import geometry as geo      # noqa: E402
from dart_scorer.session import Session      # noqa: E402


class FakeDart:
    """Stands in for a detection - only the score matters here."""

    def __init__(self, radius, angle):
        self.score = geo.score_at(*geo.polar_to_board(radius, angle))
        self.tip_image = (0.0, 0.0)
        self.board_mm = geo.polar_to_board(radius, angle)
        self.confidence = 1.0

    @property
    def label(self):
        return self.score.label

    @property
    def points(self):
        return self.score.points


def dart(label):
    """Build a dart that scores `label`, e.g. T20, D16, 5, 50."""
    lookup = {"BULL": (10.0, 0.0), "50": (2.0, 0.0)}
    if label in lookup:
        return FakeDart(*lookup[label])
    mult, bed = (label[0], int(label[1:])) if label[0] in "DT" else ("S", int(label))
    radius = {"S": 140.0, "D": 166.0, "T": 103.0}[mult]
    angle = 90.0 - 18.0 * geo.SECTORS.index(bed)
    return FakeDart(radius, angle)


def check(condition, message):
    assert condition, message


def test_dart_helper_is_honest():
    for label in ("T20", "D16", "5", "BULL", "50", "T19", "D1"):
        check(dart(label).label == label, f"helper built {dart(label).label} not {label}")


def test_free_scoring_visit():
    s = Session()
    for label in ("T20", "T20", "T20"):
        s.add_dart(dart(label))
    check(s.turn.points == 180, f"maximum should be 180, got {s.turn.points}")
    check(s.turn_complete, "three darts complete a visit")
    s.add_dart(dart("T20"))
    check(s.turn.points == 180, "a fourth dart must not count")


def test_x01_subtracts_on_visit_end():
    s = Session(["A"], start_score=501)
    for label in ("T20", "T20", "T20"):
        s.add_dart(dart(label))
    check(s.player.remaining == 501, "score only changes when the visit ends")
    s.end_turn()
    check(s.player.remaining == 321, f"501-180 should be 321, got {s.player.remaining}")


def test_bust_below_zero():
    s = Session(["A"], start_score=40)
    s.add_dart(dart("T20"))
    check(s.turn.busted, "going below zero is a bust")
    s.end_turn()
    check(s.player.remaining == 40, "a busted visit scores nothing")


def test_bust_on_one_left():
    s = Session(["A"], start_score=20)
    s.add_dart(dart("19"))
    check(s.turn.busted, "leaving 1 with double-out is a bust")


def test_checkout_needs_a_double():
    s = Session(["A"], start_score=40)
    s.add_dart(dart("D20"))
    check(not s.turn.busted, "D20 checks out 40")
    check("GAME SHOT" in s.messages[-1], f"expected a game shot, got {s.messages[-1]}")

    s = Session(["A"], start_score=40)
    s.add_dart(dart("T20"))
    check(s.turn.busted, "finishing on a single busts under double-out")

    s = Session(["A"], start_score=50)
    s.add_dart(dart("50"))
    check(not s.turn.busted, "the bull counts as a double for the out")


def test_straight_out():
    s = Session(["A"], start_score=20, double_out=False)
    s.add_dart(dart("20"))
    check(not s.turn.busted, "single 20 checks out when double-out is off")


def test_players_alternate():
    s = Session(["A", "B"], start_score=501)
    s.add_dart(dart("20"))
    s.end_turn()
    check(s.player.name == "B", "turn should pass to B")
    check(s.players[0].remaining == 481, "A scored 20")


def test_undo_within_and_across_visits():
    s = Session(["A", "B"], start_score=501)
    s.add_dart(dart("T20"))
    s.add_dart(dart("T20"))
    s.undo_dart()
    check(s.turn.points == 60, f"one dart left in the visit, got {s.turn.points}")

    s.end_turn()
    check(s.players[0].remaining == 441 and s.player.name == "B", "A threw 60")
    s.undo_dart()
    check(s.player.name == "A", "undo on an empty visit reopens A's visit")
    check(s.players[0].remaining == 501, "A's 60 was given back")
    check(s.turn.darts == [], "the reopened visit lost its last dart")


def test_undo_clears_a_bust():
    s = Session(["A"], start_score=40)
    s.add_dart(dart("T20"))
    check(s.turn.busted, "busted")
    s.undo_dart()
    check(not s.turn.busted, "undo clears the bust")


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
