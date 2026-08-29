# Dartboard Scorer

Reads a camera pointed at a dartboard and calls the score of every dart that
lands: `T20`, `D16`, `BULL`, `MISS`. Keeps a running visit total and, if you
want it, a game of 501.

```
  T20   60 pts   r= 103.4mm  conf=1.00
    1    1 pts   r= 139.8mm  conf=1.00
   D4    8 pts   r= 165.6mm  conf=1.00
-- board cleared: 69 scored --
```

## Install

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate elsewhere
pip install -r requirements.txt
```

## Use it

**1. Calibrate** (once per camera position):

```bash
python -m dart_scorer calibrate --source 0
```

Click four landmarks, in this order:

1. the outer edge of the double ring in the middle of bed **20** (top)
2. the same for bed **6** (right)
3. bed **3** (bottom)
4. bed **11** (left)
5. optionally, the centre of the bull - a fifth point tightens the fit

Drag any marker to nudge it. The rings drawn over your camera view, and the
rectified board in the second window, tell you when it is right: the yellow
rings should sit exactly on the real wires. Press `a` to have the board outline
found automatically and the markers pre-placed, then drag them into position.
`s` saves to `calibration.json`.

**2. Score:**

```bash
python -m dart_scorer run --source 0                 # free scoring
python -m dart_scorer run --game 501 --players Kyle Sam
python -m dart_scorer run --log throws.csv           # log every dart
```

Keys while running: `b` re-learn the empty board, `u` undo a dart, `n` end the
visit, `m` show the change mask, `p` pause, `q` quit.

**3. Check it without a camera:**

```bash
python -m dart_scorer selftest    # synthetic throws through the whole pipeline
python tests/test_geometry.py     # scoring geometry
python tests/test_session.py      # visits, busts, checkouts, undo
python -m dart_scorer board       # write a reference board image
```

## How it works

**Calibration.** A dartboard is flat, so one homography undoes any camera angle.
Four known landmarks give that homography; from then on any pixel can be turned
into a millimetre position on the board, and radius plus angle give the score
using the official ring dimensions.

**Detection.** The camera does not move, so everything is differencing, against
two references:

- an **empty-board** frame - tells us how many darts are in the board and when
  the board has been cleared;
- the frame as it looked **after the previous dart settled** - differencing
  against this isolates only the newest dart, which is what keeps three darts
  in the treble 20 resolvable. They touch in space, but they are separated in
  time.

Nothing is measured until the image has held still for a few frames, so darts
in flight and hands in shot never produce a score.

**Finding the point.** The new blob is the dart. Its principal axis is the shaft;
the two ends of that axis are the point and the flight. The barrel always
extends away from the centre of the board, so the end nearer the bull is the
point. The last few pixels at that end are averaged, which keeps the answer
stable to about 1.5 mm - comfortably finer than the 8 mm treble ring.

Each dart carries a confidence, knocked down when the blob is not dart-shaped,
when it lands off the scoring area, or when it lands within a wire's width of a
boundary. Low-confidence calls are drawn in amber - those are the ones to check.

## Tuning

Defaults suit a 1280x720 webcam a metre or so from the board. If it misses
darts, lower `--min-area` and `--diff`; if it invents them, raise both.

| flag | default | what it does |
|---|---|---|
| `--diff` | 26 | grey-level change that counts as "different" |
| `--min-area` | 220 | smallest blob accepted as a dart, in pixels |
| `--settle` | 4 | still frames required before scoring |
| `--motion` | 120 | changed pixels per frame that count as movement |
| `--tip` | centre | which end of the blob is the point |

Practical rig notes: bolt the camera down (any shift needs re-calibration), light
the board evenly and avoid a light that swings when a dart hits, and keep the
camera off to one side rather than straight on so darts stand out from the face
of the board.

## Known limits

- **One camera cannot see depth.** A dart at a steep angle is measured where its
  point *appears*, so an extreme angle can read a millimetre or two off. The fix
  is a second camera at ~90 degrees, intersecting two rays - the geometry and
  calibration here already support that; the fusion step is not written.
- **A dart hidden behind another** will not be seen as a separate blob. It is
  logged as a low-confidence detection rather than silently scored wrong.
- **Wire calls** are flagged, not resolved. Anything landing within a millimetre
  of a wire is worth a glance.
- Bouncers and darts that fall out are handled as removals, not as scores.

## Layout

```
dart_scorer/
  geometry.py     board dimensions, bed order, point -> score
  calibration.py  camera <-> board homography, save/load, board outline finder
  detector.py     background differencing, settle logic, tip estimation
  session.py      visits, 3-dart turns, X01 with double-out
  render.py       board drawing and the live overlay
  main.py         calibrate / run / selftest / board
tests/
  test_geometry.py
  test_session.py
```

`board.png` is a reference render of the canonical board; `example_overlay.png`
shows the live view - overlay rings on the board, marked tips, running score.
