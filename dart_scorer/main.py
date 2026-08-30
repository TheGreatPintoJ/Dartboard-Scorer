"""Command line entry point: calibrate, run, or self-test the scorer."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from . import geometry as geo
from . import render
from .calibration import (
    Calibration, REFERENCE_POINTS, ellipse_reference_guess, fit_board_ellipse,
)
from .config import DEFAULT_CONFIG_PATH, AppConfig
from .detector import DartDetector, State
from .session import Session
from .synthetic import draw_dart_on_board, synthetic_camera

DEFAULT_CALIBRATION = "calibration.json"


# --------------------------------------------------------------------------- #
# capture helpers
# --------------------------------------------------------------------------- #
def open_capture(source: str, width: int = 0, height: int = 0) -> cv2.VideoCapture:
    src: object = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(src)
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise SystemExit(f"could not open video source {source!r}")
    return cap


def grab(cap: cv2.VideoCapture):
    ok, frame = cap.read()
    if not ok:
        raise SystemExit("camera returned no frame")
    return frame


# --------------------------------------------------------------------------- #
# calibrate
# --------------------------------------------------------------------------- #
CALIBRATION_HELP = """
Click the four landmarks in this order:

  1. outer edge of the DOUBLE ring in the middle of bed 20   (top of the board)
  2. outer edge of the DOUBLE ring in the middle of bed 6    (right)
  3. outer edge of the DOUBLE ring in the middle of bed 3    (bottom)
  4. outer edge of the DOUBLE ring in the middle of bed 11   (left)
  5. the centre of the bull  (optional - a 5th point sharpens the fit)

Drag any marker to nudge it. Keys:
  a  auto-place the four markers on the detected board outline (then drag)
  u  undo last point      r  reset      n  grab a fresh frame
  s  save and quit        q  quit without saving
"""


def cmd_calibrate(args) -> int:
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise SystemExit(f"could not read {args.image}")
        cap = None
    else:
        cap = open_capture(args.source, args.width, args.height)
        for _ in range(10):        # let auto-exposure settle
            frame = grab(cap)

    print(CALIBRATION_HELP)
    points: list[list[float]] = []
    drag = {"index": -1}
    window = "calibrate - click the four double-ring landmarks"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, min(1280, frame.shape[1]), min(800, frame.shape[0]))

    def on_mouse(event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for i, p in enumerate(points):
                if (p[0] - x) ** 2 + (p[1] - y) ** 2 < 225:
                    drag["index"] = i
                    return
            if len(points) < len(REFERENCE_POINTS):
                points.append([float(x), float(y)])
        elif event == cv2.EVENT_MOUSEMOVE and drag["index"] >= 0:
            points[drag["index"]] = [float(x), float(y)]
        elif event == cv2.EVENT_LBUTTONUP:
            drag["index"] = -1

    cv2.setMouseCallback(window, on_mouse)

    while True:
        view = frame.copy()
        calib = None
        if len(points) >= 4:
            try:
                calib = Calibration.from_points(points, frame.shape[1::-1])
                render.draw_board_overlay(view, calib)
            except Exception as exc:                      # degenerate quad
                cv2.putText(view, str(exc), (12, view.shape[0] - 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        for i, p in enumerate(points):
            colour = (0, 255, 255) if i < 4 else (255, 160, 0)
            cv2.drawMarker(view, (int(p[0]), int(p[1])), colour,
                           cv2.MARKER_CROSS, 22, 2)
            cv2.circle(view, (int(p[0]), int(p[1])), 9, colour, 1, cv2.LINE_AA)
            cv2.putText(view, str(i + 1), (int(p[0]) + 12, int(p[1]) - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2, cv2.LINE_AA)

        nxt = REFERENCE_POINTS[len(points)][0] if len(points) < len(REFERENCE_POINTS) \
            else "all set - press s to save"
        render.draw_panel(view, [f"next: {nxt}",
                                 "a auto  u undo  r reset  s save  q quit"],
                          width=560)
        cv2.imshow(window, view)
        if calib is not None:
            cv2.imshow("rectified board (rings should line up)",
                       cv2.resize(calib.warp(frame), (450, 450)))

        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            cv2.destroyAllWindows()
            return 1
        if key == ord("u") and points:
            points.pop()
        elif key == ord("r"):
            points.clear()
        elif key == ord("a"):
            ellipse = fit_board_ellipse(frame)
            if ellipse is None:
                print("could not find the board automatically - click manually")
            else:
                points[:] = ellipse_reference_guess(ellipse)
                print("markers placed on the detected outline - now drag marker 1 "
                      "onto bed 20 and the rest will follow round the board")
        elif key == ord("n") and cap is not None:
            frame = grab(cap)
        elif key in (ord("s"), 13) and calib is not None:
            calib.save(args.calibration)
            print(f"saved {args.calibration}")
            cv2.imwrite(str(Path(args.calibration).with_suffix(".reference.png")), frame)
            break

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    return 0


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
RUN_KEYS = ("q quit   b re-learn empty board   u undo dart   "
            "n end visit   m mask   p pause")


def cmd_run(args) -> int:
    if not Path(args.calibration).exists():
        raise SystemExit(f"no calibration at {args.calibration!r} - "
                         f"run 'python -m dart_scorer calibrate' first")
    calib = Calibration.load(args.calibration)
    cap = open_capture(args.source, args.width, args.height)
    detector = DartDetector(
        calib,
        diff_threshold=args.diff,
        min_area=args.min_area,
        settle_frames=args.settle,
        motion_threshold=args.motion,
        tip_mode=args.tip,
    )
    session = Session(args.players, start_score=args.game, double_out=not args.straight_out)

    log = log_file = None
    if args.log:
        new = not Path(args.log).exists()
        log_file = open(args.log, "a", newline="", encoding="utf-8")
        log = csv.writer(log_file)
        if new:
            log.writerow(["timestamp", "player", "label", "points", "x_mm", "y_mm",
                          "radius_mm", "angle_deg", "confidence"])

    show = not args.no_display
    try:
        _run_loop(cap, calib, detector, session, args, log, show)
    except KeyboardInterrupt:
        print()
    finally:
        if log_file:
            log_file.close()
        cap.release()
        cv2.destroyAllWindows()

    session.end_turn()
    if args.game:
        for p in session.players:
            print(f"{p.name}: {p.remaining} left after {p.darts_thrown} darts")
    return 0


def _run_loop(cap, calib, detector, session, args, log, show) -> None:
    show_mask = False
    paused = False
    tips: list = []
    fps, last = 0.0, time.time()
    frame = result = None

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break
            result = detector.update(frame)

            for dart in result.darts:
                session.add_dart(dart)
                tips.append(dart)
                print(f"{dart.label:>5}  {dart.points:>3} pts   "
                      f"r={geo.radius_of(*dart.board_mm):6.1f}mm  "
                      f"conf={dart.confidence:.2f}"
                      + ("  [near wire]" if dart.score.near_wire else ""))
                if log:
                    log.writerow([f"{time.time():.3f}", session.player.name,
                                  dart.label, dart.points,
                                  f"{dart.board_mm[0]:.1f}", f"{dart.board_mm[1]:.1f}",
                                  f"{geo.radius_of(*dart.board_mm):.1f}",
                                  f"{dart.score.angle_deg:.1f}", dart.confidence])
            if result.cleared and session.turn.darts:
                print(f"-- board cleared: {session.turn.points} scored --")
                session.end_turn()
                tips.clear()

            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
            last = now

        if not show:
            continue

        view = frame.copy()
        render.draw_board_overlay(view, calib, (0, 200, 255), 1)
        for i, dart in enumerate(tips, 1):
            render.draw_marker(view, dart.tip_image, f"{i}: {dart.label}",
                               (0, 255, 120) if dart.confidence > 0.6 else (0, 180, 255))
        lines = [f"{result.state.value.upper():9s} {fps:4.1f} fps"]
        lines += session.scoreboard() + session.messages[-2:]
        render.draw_panel(view, lines, width=340)
        cv2.putText(view, RUN_KEYS, (12, view.shape[0] - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
        cv2.imshow("dart scorer", view)
        if show_mask and result.debug_mask is not None:
            cv2.imshow("change mask", result.debug_mask)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("b"):
            detector.reset_background(frame)
            tips.clear()
            print("re-learned the empty board")
        elif key == ord("u"):
            session.undo_dart()
            if tips:
                tips.pop()
        elif key == ord("n"):
            session.end_turn()
            tips.clear()
        elif key == ord("m"):
            show_mask = not show_mask
            if not show_mask:
                cv2.destroyWindow("change mask")
        elif key == ord("p"):
            paused = not paused


# --------------------------------------------------------------------------- #
# selftest - the whole pipeline on synthetic frames
# --------------------------------------------------------------------------- #
def cmd_selftest(args) -> int:
    board, view, calib = synthetic_camera()
    detector = DartDetector(calib, settle_frames=3, min_area=150)

    targets = [
        (geo.polar_to_board(103, 90.0), "T20"),
        (geo.polar_to_board(166, 90.0), "D20"),
        (geo.polar_to_board(140, 90.0 - 18 * 5), "6"),
        (geo.polar_to_board(103, 90.0 - 18 * 10), "T3"),
        (geo.polar_to_board(3, 0.0), "50"),
        (geo.polar_to_board(12, 210.0), "BULL"),
        (geo.polar_to_board(166, 90.0 - 18 * 3), "D4"),
        (geo.polar_to_board(166, 90.0 + 18 * 3), "D9"),
        (geo.polar_to_board(60, 90.0 - 18 * 15), "11"),
        (geo.polar_to_board(103, 90.0 + 18 * 7), "T16"),
        (geo.polar_to_board(190, 45.0), "MISS"),
        # Beyond the board itself: movement, not a dart. Must be ignored.
        (geo.polar_to_board(260, 200.0), "none"),
    ]

    canvas = board.copy()
    for _ in range(30):                       # learn the empty board
        detector.update(view(canvas))

    passed, results = 0, []
    for (x, y), expected in targets:
        canvas = draw_dart_on_board(canvas, x, y, jitter_deg=4.0)
        got = None
        for _ in range(8):
            res = detector.update(view(canvas))
            if res.darts:
                got = res.darts[0]
                break
        label = got.label if got else "none"
        err = ""
        if got:
            dx = got.board_mm[0] - x
            dy = got.board_mm[1] - y
            err = f"  tip error {np.hypot(dx, dy):5.1f} mm  conf {got.confidence:.2f}"
        ok = label == expected
        passed += ok
        results.append(f"  {'PASS' if ok else 'FAIL'}  expected {expected:>5} "
                       f"got {label:>5}{err}")
        if args.show and got:
            frame = view(canvas)
            render.draw_board_overlay(frame, calib)
            render.draw_marker(frame, got.tip_image, label)
            cv2.imshow("selftest", frame)
            cv2.waitKey(400)

    print("\n".join(results))
    print(f"\n{passed}/{len(targets)} synthetic throws scored correctly")
    if args.show:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return 0 if passed == len(targets) else 1


# --------------------------------------------------------------------------- #
# serve - the web interface
# --------------------------------------------------------------------------- #
def cmd_serve(args) -> int:
    from .engine import ScoringEngine
    from .webapp import serve

    config = AppConfig.load(args.config)
    if args.source:
        config.camera.source = args.source
    if args.width:
        config.camera.width = args.width
    if args.height:
        config.camera.height = args.height
    if args.calibration:
        config.calibration_path = args.calibration
    if args.log:
        config.log_path = args.log
    config.save(args.config)

    engine = ScoringEngine(config, config_path=args.config)

    # A demo board has no physical camera to calibrate against, so install the
    # calibration that matches it and the UI is usable the moment it loads.
    if str(config.camera.source).lower() == "demo" and engine.calibration is None:
        from .synthetic import DemoSource

        engine.set_calibration(DemoSource().calibration.image_points)
        print("demo source: installed the matching calibration")

    engine.start()
    httpd = serve(engine, args.host, args.port, token=args.token, verbose=args.verbose)
    shown = args.host if args.host not in ("0.0.0.0", "::") else "<this machine>"
    suffix = f"?token={args.token}" if args.token else ""
    print(f"dart scorer on http://{shown}:{args.port}/{suffix}")
    if args.host == "0.0.0.0":
        print("listening on every interface - anyone on the network can watch "
              "the camera" + ("" if args.token else "; consider --token"))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        httpd.shutdown()
        httpd.server_close()
        engine.stop()
    return 0


# --------------------------------------------------------------------------- #
def cmd_doctor(args) -> int:
    """Work out which stage a jittery picture is coming from."""
    from .diagnose import report

    index = None if args.no_camera else (
        int(args.source) if str(args.source).isdigit() else None)
    if index is None and not args.no_camera:
        print(f"skipping camera checks: {args.source!r} is not a camera index")
    return report(index, args.url, args.token)


# --------------------------------------------------------------------------- #
def cmd_board(args) -> int:
    """Write out a reference image of the canonical board."""
    cv2.imwrite(args.out, render.render_board())
    print(f"wrote {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dart_scorer", description="Score darts from a camera pointed at a board.")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("--source", default="0",
                        help="camera index or video file (default: 0)")
        sp.add_argument("--width", type=int, default=1280)
        sp.add_argument("--height", type=int, default=720)
        sp.add_argument("--calibration", default=DEFAULT_CALIBRATION)

    c = sub.add_parser("calibrate", help="mark the board landmarks")
    common(c)
    c.add_argument("--image", help="calibrate from a still image instead of a camera")
    c.set_defaults(func=cmd_calibrate)

    r = sub.add_parser("run", help="score a live feed")
    common(r)
    r.add_argument("--game", type=int, default=0,
                   help="X01 start score, e.g. 501 (default: free scoring)")
    r.add_argument("--players", nargs="+", help="player names")
    r.add_argument("--straight-out", action="store_true",
                   help="X01 without the double-out rule")
    r.add_argument("--diff", type=int, default=26, help="pixel change threshold")
    r.add_argument("--min-area", type=int, default=220,
                   help="smallest blob accepted as a dart, in pixels")
    r.add_argument("--settle", type=int, default=4,
                   help="still frames required before a dart is scored")
    r.add_argument("--motion", type=int, default=120,
                   help="changed pixels per frame that count as movement")
    r.add_argument("--tip", choices=("centre", "lowest", "highest"), default="centre",
                   help="which end of the blob is the point of the dart")
    r.add_argument("--log", help="append every detection to this CSV file")
    r.add_argument("--no-display", action="store_true", help="headless")
    r.set_defaults(func=cmd_run)

    w = sub.add_parser("serve", help="run the web interface (no display needed)")
    w.add_argument("--host", default="127.0.0.1",
                   help="0.0.0.0 to reach it from other machines")
    w.add_argument("--port", type=int, default=8080)
    w.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    w.add_argument("--source", help="camera index, video file, URL, or 'demo'")
    w.add_argument("--width", type=int)
    w.add_argument("--height", type=int)
    w.add_argument("--calibration", help="path to calibration.json")
    w.add_argument("--log", help="path to the detections CSV")
    w.add_argument("--token", help="require this token in the URL or "
                                   "the X-Auth-Token header")
    w.add_argument("--verbose", action="store_true", help="log every request")
    w.set_defaults(func=cmd_serve)

    d = sub.add_parser("doctor", help="diagnose a jittery or slow picture")
    d.add_argument("--source", default="0", help="camera index to test (default: 0)")
    d.add_argument("--no-camera", action="store_true",
                   help="skip the camera checks")
    d.add_argument("--url", help="also measure a running server, "
                                 "e.g. http://127.0.0.1:8080")
    d.add_argument("--token", help="token, if the server needs one")
    d.set_defaults(func=cmd_doctor)

    t = sub.add_parser("selftest", help="run the pipeline on synthetic throws")
    t.add_argument("--show", action="store_true", help="display each throw")
    t.set_defaults(func=cmd_selftest)

    b = sub.add_parser("board", help="write a reference board image")
    b.add_argument("--out", default="board.png")
    b.set_defaults(func=cmd_board)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
