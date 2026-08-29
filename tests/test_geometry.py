"""Scoring geometry checks - no camera or OpenCV needed."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dart_scorer import geometry as geo   # noqa: E402


def at(radius, angle):
    return geo.score_at(*geo.polar_to_board(radius, angle))


def check(condition, message):
    assert condition, message


def test_bull():
    check(at(0, 0).label == "50", "centre is the inner bull")
    check(at(5, 123).points == 50, "inner bull scores 50")
    check(at(10, 40).label == "BULL", "outer bull is 25")
    check(at(15.5, 200).points == 25, "outer bull scores 25")


def test_rings_on_bed_20():
    check(at(50, 90).label == "20", "single 20")
    check(at(103, 90).label == "T20", "treble 20")
    check(at(140, 90).label == "20", "outer single 20")
    check(at(166, 90).label == "D20", "double 20")
    check(at(103, 90).points == 60, "treble 20 is 60")
    check(at(166, 90).points == 40, "double 20 is 40")


def test_off_board():
    check(at(171, 90).label == "MISS", "outside the double ring scores nothing")
    check(at(400, 33).points == 0, "way off the board scores nothing")


def test_bed_order_is_clockwise_from_20():
    beds = [geo.bed_at_angle(90 - 18 * i) for i in range(20)]
    check(beds == list(geo.SECTORS), f"bed order wrong: {beds}")


def test_bed_boundaries():
    # 20 spans 81..99 degrees; either side of the wire must differ.
    check(geo.bed_at_angle(98.9) == 20, "just inside the 20/1 wire")
    check(geo.bed_at_angle(99.1) == 5, "just past the 20/5 wire")
    check(geo.bed_at_angle(81.1) == 20, "just inside the 20/1 wire")
    check(geo.bed_at_angle(80.9) == 1, "just past the 20/1 wire")
    for bed in geo.SECTORS:
        lo, hi = geo.bed_bounds(bed)
        mid = (lo + (hi - lo) % 360 / 2) % 360
        check(geo.bed_at_angle(mid) == bed, f"midpoint of bed {bed} lands elsewhere")


def test_near_wire_flag():
    check(at(99.2, 90).near_wire, "just inside the treble wire is a close call")
    check(not at(60, 90).near_wire, "middle of a big single is not a close call")


def test_all_beds_reachable():
    got = {at(140, 90 - 18 * i).base for i in range(20)}
    check(got == set(geo.SECTORS), f"missing beds: {set(geo.SECTORS) - got}")


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
