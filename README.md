# Dartboard Scorer

Reads a camera pointed at a dartboard and calls the score of every dart that
lands: `T20`, `D16`, `BULL`, `MISS`. Runs as a headless service with a web
interface, so a machine in the corner with a webcam does the scoring and any
phone or laptop on the network watches the game.

Everything is driven from the browser - camera selection, calibration, detection
tuning, the game itself. Nothing needs a screen attached to the scoring machine.

## Install

```bash
python -m venv .venv
.venv/Scripts/activate           # Windows;  source .venv/bin/activate elsewhere
pip install -r requirements.txt
```

Only OpenCV and NumPy. The web server is standard library.

## Run it

```bash
python -m dart_scorer serve --source 0                    # http://127.0.0.1:8080
python -m dart_scorer serve --host 0.0.0.0 --token hunter2 # reachable on the LAN
python -m dart_scorer serve --source demo                 # no camera needed
```

Open the address it prints. Then:

1. **Calibrate** - press Calibrate and click four landmarks on the video:
   the outer edge of the double ring in the middle of bed **20** (top), then
   **6** (right), **3** (bottom), **11** (left). A fifth click on the bull
   sharpens the fit. Press Preview to check the rings sit on the real wires,
   then Save. This holds until the camera moves.
2. **Throw.** Darts are scored as they land. Pull them out and the visit ends.

`--source demo` runs a synthetic board with no hardware: throw darts from the
browser and watch them go through the same detection path as a real feed. It is
the quickest way to see the whole thing working.

### Command line, without the web interface

```bash
python -m dart_scorer calibrate --source 0      # OpenCV window, click landmarks
python -m dart_scorer run --game 501 --players Kyle Sam
python -m dart_scorer selftest                  # synthetic throws, no camera
python -m dart_scorer board --out board.png     # reference board image
```

## Running as a Linux service

```bash
sudo ./deploy/install.sh --source 0 --port 8080
```

That creates a `dartscorer` system user, installs the code to `/opt/dart-scorer`
with its own virtualenv, writes settings to `/etc/default/dart-scorer`, and
enables `dart-scorer.service`. Config, calibration and the throw log live in
`/var/lib/dart-scorer` and survive reinstalls.

```bash
systemctl status dart-scorer
journalctl -u dart-scorer -f
systemctl restart dart-scorer         # after editing /etc/default/dart-scorer
```

Settings in `/etc/default/dart-scorer`: `DART_HOST`, `DART_PORT`, `DART_SOURCE`,
`DART_TOKEN`. The unit runs with `ProtectSystem=strict`, a restricted syscall
set and access to video devices only. `deploy/nginx-dart-scorer.conf` is there
if you want TLS or a hostname in front - it turns off proxy buffering, which
both the video stream and the event stream need.

On a Raspberry Pi or similar, install `opencv-python-headless`; the install
script prefers it already. Drop `stream_scale` to 0.5 and `stream_quality` to
60 in Settings if the network is the bottleneck.

## HTTP interface

| endpoint | what it does |
|---|---|
| `GET /` | the control page |
| `GET /stream.mjpg` | live annotated video (multipart JPEG) |
| `GET /snapshot.jpg`, `GET /rectified.jpg` | one frame; the board warped flat |
| `GET /api/status` | state, scoreboard, current visit, config |
| `GET /api/events` | server-sent events: darts, state changes |
| `GET`/`POST /api/config` | read or change any setting |
| `GET`/`POST`/`DELETE /api/calibration` | the board landmarks |
| `GET`/`POST /api/camera` | what the camera is doing; focus, zoom, exposure, ... |
| `POST /api/calibration/auto`, `/preview` | suggested landmarks; overlay preview |
| `POST /api/command` | `undo`, `end_turn`, `new_game`, `relearn`, `reconnect`, `throw` |
| `GET /api/throws.csv` | every detection, logged |
| `GET /healthz` | liveness probe, no token needed |

So a scoreboard on a TV, a Home Assistant card or a league bot is a matter of
reading `/api/events`. Example:

```bash
curl -N http://darts.local:8080/api/events
curl -X POST -d '{"command":"undo"}' http://darts.local:8080/api/command
```

With `--token` set, pass it as `?token=...` or an `X-Auth-Token` header. Opening
the page once with `?token=...` sets a cookie, so the browser keeps working.
There is no user accounts system: the token is a shared secret for a home LAN,
and the video stream is unencrypted unless you put nginx in front.

## How it works

**Calibration.** A dartboard is flat, so one homography undoes any camera angle.
Four known landmarks give that homography; from then on any pixel becomes a
millimetre position on the board, and radius plus angle give the score using the
official ring dimensions.

**Detection.** The camera does not move, so everything is differencing, against
two references:

- an **empty-board** frame - tells us how many darts are in the board and when
  the board has been cleared;
- the frame as it looked **after the previous dart settled** - differencing
  against this isolates only the newest dart, which is what keeps three darts in
  the treble 20 resolvable. They touch in space, but they are separated in time.

Nothing is measured until the image has held still for a few frames, so darts in
flight and hands in shot never produce a score.

**Finding the point.** The new blob is the dart. Its principal axis is the shaft;
the two ends of that axis are the point and the flight. The barrel always extends
away from the centre of the board, so the end nearer the bull is the point. The
last few pixels at that end are averaged, which keeps the answer stable to about
1.5 mm - comfortably finer than the 8 mm treble ring.

A blob whose point falls outside the physical board is discarded rather than
scored: an arm reaching in, someone walking past or a shifting shadow cannot be
a dart stuck in the board. A dart in the number ring is still a legitimate zero.

Each dart carries a confidence, knocked down when the blob is not dart-shaped,
when it lands off the scoring area, or when it lands within a wire's width of a
boundary. Low-confidence calls are marked in amber - those are the ones to check.

## Camera settings

Open **Camera controls** under the video. Every row shows what the camera
*actually* reports after a change, because cameras routinely accept a value and
then ignore or clamp it - if the reading does not move, that camera does not
support that control. Blank a box to stop setting it at all.

- **Lock focus & white balance** freezes the lens where it is. Do this before
  scoring: autofocus hunts, which pulses the picture and, worse, shifts the
  reference frame the detector compares against, producing phantom darts.
- **Zoom, pan and tilt move the board within the frame**, so they invalidate the
  calibration. The UI says so when you change one; calibrate again afterwards.
- Auto-exposure is worth turning off for the same reason as autofocus. Its
  manual value is backend-specific (often `1` on V4L2, `0.25` on DirectShow),
  which is why the box takes a number rather than a checkbox.

Capture settings live in Settings:

| setting | default | what it does |
|---|---|---|
| pixel format | MJPG | see below - the single biggest cause of a stuttering feed |
| capture fps | 0 | 0 leaves the camera's own rate alone |
| backend | auto | dshow on Windows, v4l2 on Linux, avfoundation on macOS |
| buffer depth | 1 | 1 = always the newest frame, never a queued one |
| stream fps cap | 20 | what the browser is asked to decode; 0 = uncapped |

## If the picture is jittery

Run the doctor. It measures each stage separately, because the camera, this
machine, the server and the browser fail differently and need different fixes:

```bash
python -m dart_scorer doctor --source 0 --url http://127.0.0.1:8080
```

It reports what this machine can process per frame, what the camera delivers at
each resolution and pixel format, and what the running server actually serves -
including a **horizontal wrap check**. Frames read from the buffer at the wrong
offset come out shifted sideways with the content wrapping around the edge;
comparing consecutive frames of a still scene by circular cross-correlation
finds it. A healthy camera scores 0.

Then work down this list:

1. **If the doctor says the camera never exceeds ~10 fps**, it is the pixel
   format. Most USB webcams offer uncompressed YUY2 and compressed MJPG, and
   OpenCV takes whatever the driver lists first - usually YUY2, which does not
   fit down USB 2.0 at 720p, so the camera quietly drops to 5-10 fps. Set the
   pixel format to MJPG in Settings.
2. **If the doctor says the camera returns wrapped frames**, that is a driver or
   cable problem: update the camera driver, use a different USB port, and avoid
   hubs.
3. **If the doctor says the served stream is clean and steady but the picture on
   screen is not**, the fault is past the socket and no change to this program
   will fix it. See below.

### When the stream is clean but the screen is not

A browser renders an MJPEG stream through the GPU compositor, and that path can
tear, stall or wrap even when every byte arriving is correct. To tell the
difference:

- Open `http://<host>:8080/stream.mjpg` **directly** in a tab, with no page
  around it. If that is smooth, the problem is the page or the compositor rather
  than the stream.
- Compare what you see with what a screen recording shows. If your eyes see
  smooth video and only the recording is torn, the recorder is at fault - it is
  capturing a hardware surface - and there is nothing to fix.

If it really is on screen:

- **Turn off browser hardware acceleration.** Chrome and Edge:
  `chrome://settings/system` / `edge://settings/system`, uncheck "Use graphics
  acceleration when available", restart the browser. This is the usual fix for a
  torn or sideways-wrapped video surface.
- **Try another browser.** Firefox and Chromium handle MJPEG differently.
- **Update the GPU driver.** Stride and offset bugs in the compositor are driver
  bugs.
- **Close anything else using the camera** - OBS, Teams, Zoom. Windows shares a
  camera between applications through its Frame Server, and the format can be
  renegotiated underneath a running capture.
- **Lower the stream fps cap** in Settings. The default is 20; a browser decodes
  that comfortably, where 60 full-size JPEGs a second stalls and batches.

## Tuning

Defaults suit a 1280x720 webcam a metre or so from the board. All of it is in
Settings in the web UI; the CLI has matching flags.

| setting | default | what it does |
|---|---|---|
| change threshold | 26 | grey-level change that counts as "different" |
| min blob px | 220 | smallest blob accepted as a dart |
| max blob px | 26000 | largest; beyond 3x this the board is treated as occluded |
| settle frames | 4 | still frames required before scoring |
| motion px | 120 | changed pixels per frame that count as movement |
| tip end | nearer the bull | which end of the blob is the point |

If it misses darts, lower the threshold and the minimum blob size; if it invents
them, raise both. Bolt the camera down - any shift needs re-calibration - light
the board evenly, and put the camera off to one side rather than straight on so
darts stand out from the face of the board.

## Testing

```bash
python -m dart_scorer selftest    # synthetic throws through the whole pipeline
python tests/test_geometry.py     # scoring geometry
python tests/test_session.py      # visits, busts, checkouts, undo
python -m dart_scorer doctor      # camera, machine and stream diagnostics
python tests/test_camera.py       # capture layer: drain thread, controls
python tests/test_webapp.py       # the live service, end to end, no camera
```

`test_webapp.py` starts the real server on a spare port, throws darts at the
demo board and checks the HTTP surface reports them.

## Known limits

- **One camera cannot see depth.** A dart at a steep angle is measured where its
  point *appears*, so an extreme angle can read a millimetre or two off. The fix
  is a second camera at ~90 degrees, intersecting two rays - the geometry and
  calibration support that; the fusion step is not written.
- **A dart hidden behind another** will not be seen as a separate blob. It is
  logged as a low-confidence detection rather than silently scored wrong.
- **Wire calls** are flagged, not resolved.
- Bouncers and darts that fall out are handled as removals, not as scores.
- The server is `http.server` with threads: fine for a handful of viewers on a
  home network, not a public deployment.

## Layout

```
dart_scorer/
  geometry.py     board dimensions, bed order, point -> score
  calibration.py  camera <-> board homography, save/load, board outline finder
  detector.py     background differencing, settle logic, tip estimation
  session.py      visits, 3-dart turns, X01 with double-out
  camera.py       opening a camera: pixel format, drain thread, focus/zoom/...
  diagnose.py     the doctor: per-stage timing, wrap detection, verdicts
  engine.py       the scoring thread: camera, detection, state, events
  webapp.py       HTTP routes, MJPEG stream, server-sent events
  web/index.html  the browser interface
  config.py       runtime settings, persisted to config.json
  synthetic.py    the demo board, used by the tests and --source demo
  render.py       board drawing and the live overlay
  main.py         serve / calibrate / run / selftest / board
deploy/
  dart-scorer.service      systemd unit
  install.sh               one-shot installer
  nginx-dart-scorer.conf   optional reverse proxy
tests/
  test_geometry.py  test_session.py  test_camera.py  test_webapp.py
```

`board.png` is a reference render of the canonical board; `example_overlay.png`
and `web_snapshot.png` show the live view - overlay rings on the board, marked
tips, running score.
