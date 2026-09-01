"""HTTP interface to the scoring engine.

Standard library only - no web framework - so deploying it is a matter of
copying the tree and pointing systemd at it. Serves:

  /                         the control page
  /stream.mjpg              live annotated video (multipart JPEG)
  /snapshot.jpg             one frame
  /rectified.jpg            the board warped flat, to check a calibration
  /api/status               everything the UI needs, as JSON
  /api/events               server-sent events: darts, state changes, config
  /api/config               GET / POST the runtime configuration
  /api/calibration          GET / POST / DELETE the board landmarks
  /api/calibration/auto     suggested landmarks from the board outline
  /api/calibration/preview  overlay rendered for candidate landmarks
  /api/camera               GET what the camera is doing + every control it
                            exposes; POST to change focus, zoom, exposure, ...
  /api/command              undo, end_turn, new_game, relearn, reconnect, throw
  /api/throws.csv           the detection log
  /healthz                  liveness probe
"""

from __future__ import annotations

import json
import mimetypes
import posixpath
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import render

WEB_ROOT = Path(__file__).parent / "web"
BOUNDARY = "dartframe"
MAX_BODY = 1 << 20


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "DartScorer"
    engine = None            # set by serve()
    token = None
    verbose = False

    # ------------------------------------------------------------------ #
    # plumbing
    # ------------------------------------------------------------------ #
    def log_message(self, fmt, *args):                       # noqa: A003
        if self.verbose:
            super().log_message(fmt, *args)

    def _authorised(self, query) -> bool:
        if not self.token:
            return True
        supplied = (self.headers.get("X-Auth-Token")
                    or (query.get("token") or [None])[0]
                    or self._cookie("dart_token"))
        return supplied == self.token

    def _cookie(self, name):
        for part in (self.headers.get("Cookie") or "").split(";"):
            key, _, value = part.strip().partition("=")
            if key == name:
                return value
        return None

    def _send(self, code, body=b"", content_type="text/plain; charset=utf-8",
              extra_headers=None):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for key, value in (extra_headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        if self.command != "HEAD" and body:
            self.wfile.write(body)

    def _json(self, data, code=HTTPStatus.OK):
        self._send(code, json.dumps(data, default=str), "application/json")

    def _error(self, code, message):
        self._json({"error": message}, code)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        if length > MAX_BODY:
            raise ValueError("request body too large")
        raw = self.rfile.read(length)
        if not raw.strip():
            return {}
        return json.loads(raw.decode("utf-8"))

    # ------------------------------------------------------------------ #
    # routing
    # ------------------------------------------------------------------ #
    def do_GET(self):                                        # noqa: N802
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        path = posixpath.normpath(parsed.path)

        if path == "/healthz":
            return self._json({"ok": True, "state": self.engine.status()["state"]})
        if not self._authorised(query):
            return self._error(HTTPStatus.UNAUTHORIZED, "token required")

        cookie = None
        if self.token and (query.get("token") or [None])[0] == self.token:
            cookie = {"Set-Cookie": f"dart_token={self.token}; Path=/; SameSite=Strict"}

        try:
            if path == "/":
                return self._file(WEB_ROOT / "index.html", cookie)
            if path.startswith("/static/"):
                return self._file(WEB_ROOT / path[len("/static/"):], cookie)
            # ?view= names an extra camera; absent means the one that scores,
            # which is what keeps every existing URL behaving as it always did.
            view = self._view(query)
            if path == "/stream.mjpg":
                return self._stream(view)
            if path == "/snapshot.jpg":
                annotated = (query.get("annotated") or ["1"])[0] != "0"
                return self._image(view.snapshot(annotated) if view
                                   else self.engine.snapshot(annotated))
            if path == "/rectified.jpg":
                return self._image(view.rectified() if view
                                   else self.engine.rectified())
            if path == "/api/status":
                return self._json(self.engine.status())
            if path == "/api/events":
                return self._events(int((query.get("cursor") or ["0"])[0]))
            if path == "/api/config":
                return self._json(self.engine.config.to_dict())
            if path == "/api/calibration":
                if view is not None:
                    info = view.info()
                    return self._json({"calibrated": info["calibrated"],
                                       "points": info["calibration_points"],
                                       "frame_size": info["frame_size"]})
                status = self.engine.status()
                return self._json({"calibrated": status["calibrated"],
                                   "points": status["calibration_points"],
                                   "frame_size": status["frame_size"]})
            if path == "/api/throws.csv":
                return self._file(Path(self.engine.config.log_path), None,
                                  "text/csv; charset=utf-8")
            if path == "/api/cameras":
                return self._json(probe_devices())
            if path == "/api/camera":
                return self._json(self.engine.camera_info())
            if path == "/api/views":
                return self._json(self.engine.views_info())
        except BrokenPipeError:
            return
        except ValueError as exc:            # an unknown ?view=, mostly
            return self._error(HTTPStatus.NOT_FOUND, str(exc))
        except Exception as exc:                             # keep the service up
            return self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
        return self._error(HTTPStatus.NOT_FOUND, f"no route for {parsed.path}")

    def do_HEAD(self):                                       # noqa: N802
        self.do_GET()

    def do_POST(self):                                       # noqa: N802
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        if not self._authorised(query):
            return self._error(HTTPStatus.UNAUTHORIZED, "token required")
        path = posixpath.normpath(parsed.path)

        try:
            body = self._body()
        except ValueError as exc:
            return self._error(HTTPStatus.BAD_REQUEST, f"bad JSON: {exc}")

        try:
            if path == "/api/config":
                return self._json(self.engine.apply_config(body))

            target = self._view(query)
            if path == "/api/calibration":
                points = body.get("points") or []
                if len(points) < 4:
                    return self._error(HTTPStatus.BAD_REQUEST,
                                       "four landmarks are needed")
                save = body.get("save", True)
                result = (target.set_calibration(points, save=save) if target
                          else self.engine.set_calibration(points, save=save))
                return self._json(result)

            if path == "/api/calibration/preview":
                jpeg = (self._view_preview(target, body.get("points") or [])
                        if target else self.engine.preview(body.get("points") or []))
                if jpeg is None:
                    return self._error(HTTPStatus.BAD_REQUEST,
                                       "need a frame and four usable landmarks")
                return self._image(jpeg)

            if path == "/api/calibration/auto":
                points = (self._view_auto(target) if target
                          else self.engine.auto_points())
                if points is None:
                    return self._error(HTTPStatus.NOT_FOUND,
                                       "could not find the board - place the "
                                       "landmarks by hand")
                return self._json({"points": points})

            if path == "/api/camera":
                controls = body.get("controls", body)
                if not isinstance(controls, dict) or not controls:
                    return self._error(HTTPStatus.BAD_REQUEST, "no controls given")
                return self._json(self.engine.set_camera_controls(controls))

            if path == "/api/views":
                # One view at a time, named. The primary lives at config.camera
                # so that every existing caller keeps working; the browser does
                # not need to know that, and addresses them all the same way.
                view = body.get("view", body)
                name = str(view.get("name") or "").strip()
                if not name:
                    return self._error(HTTPStatus.BAD_REQUEST, "which view?")
                target = self.engine.config.view(name)
                if name == self.engine.config.camera.name:
                    patch = {"camera": {k: v for k, v in view.items() if k != "name"}}
                elif target is None and not view.get("add"):
                    return self._error(HTTPStatus.NOT_FOUND, f"no view named {name!r}")
                else:
                    patch = {"views": [view]}
                self.engine.apply_config(patch)
                return self._json(self.engine.views_info())

            if path == "/api/command":
                name = body.get("command", "")
                kwargs = {k: v for k, v in body.items() if k != "command"}
                return self._json(self.engine.command(name, **kwargs))
        except ValueError as exc:
            return self._error(HTTPStatus.BAD_REQUEST, str(exc))
        except BrokenPipeError:
            return
        except Exception as exc:
            return self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))
        return self._error(HTTPStatus.NOT_FOUND, f"no route for {parsed.path}")

    def do_DELETE(self):                                     # noqa: N802
        parsed = urlparse(self.path)
        if not self._authorised(parse_qs(parsed.query)):
            return self._error(HTTPStatus.UNAUTHORIZED, "token required")
        if posixpath.normpath(parsed.path) == "/api/calibration":
            view = self._view(parse_qs(parsed.query))
            if view is not None:
                view.clear_calibration()
            else:
                self.engine.clear_calibration()
            return self._json({"calibrated": False})
        return self._error(HTTPStatus.NOT_FOUND, f"no route for {parsed.path}")

    # ------------------------------------------------------------------ #
    # views
    # ------------------------------------------------------------------ #
    def _view(self, query):
        """The extra camera named by ?view=, or None for the one that scores.

        Naming the primary explicitly is the same as not naming anything, so
        the browser can address every camera the same way without having to
        know which one happens to be doing the scoring.
        """
        name = (query.get("view") or [""])[0].strip()
        if not name or name == self.engine.config.camera.name:
            return None
        view = self.engine.views.get(name)
        if view is None:
            raise ValueError(f"no camera named {name!r}")
        return view

    @staticmethod
    def _view_preview(view, points):
        """The board outline drawn over a secondary's picture, for candidate
        landmarks - the same check the primary offers before saving."""
        from .calibration import Calibration
        frame = view.latest()
        if frame is None or len(points) < 4:
            return None
        try:
            calib = Calibration.from_points(points, frame.shape[1::-1])
        except ValueError:
            return None
        return view._encode(render.draw_board_overlay(frame.copy(), calib))

    @staticmethod
    def _view_auto(view):
        from .calibration import ellipse_reference_guess, fit_board_ellipse
        frame = view.latest()
        if frame is None:
            return None
        ellipse = fit_board_ellipse(frame)
        return None if ellipse is None else ellipse_reference_guess(ellipse)

    # ------------------------------------------------------------------ #
    # responses
    # ------------------------------------------------------------------ #
    def _file(self, path: Path, extra_headers=None, content_type=None):
        path = path.resolve()
        try:
            inside_web = path.is_relative_to(WEB_ROOT.resolve())
        except AttributeError:                               # Python < 3.9
            inside_web = str(path).startswith(str(WEB_ROOT.resolve()))
        allowed = inside_web or path == Path(self.engine.config.log_path).resolve()
        if not allowed or not path.is_file():
            return self._error(HTTPStatus.NOT_FOUND, "not found")
        ctype = content_type or mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self._send(HTTPStatus.OK, path.read_bytes(), ctype, extra_headers)

    def _image(self, jpeg):
        if jpeg is None:
            return self._error(HTTPStatus.SERVICE_UNAVAILABLE, "no frame yet")
        self._send(HTTPStatus.OK, jpeg, "image/jpeg")

    def _stream(self, view=None):
        self.close_connection = True
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type",
                         f"multipart/x-mixed-replace; boundary={BOUNDARY}")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            frames = view.stream() if view is not None else self.engine.stream()
            for jpeg in frames:
                self.wfile.write(f"--{BOUNDARY}\r\n".encode())
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass                                             # viewer went away

    def _events(self, cursor: int):
        self.close_connection = True
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            self.wfile.write(b": connected\n\n")
            self.wfile.write(f"data: {json.dumps(self.engine.status())}\n\n"
                             .encode("utf-8"))
            last_ping = time.time()
            while True:
                events, cursor = self.engine.events.since(cursor, timeout=10.0)
                for event in events:
                    payload = json.dumps(event, default=str)
                    self.wfile.write(f"id: {event['seq']}\ndata: {payload}\n\n"
                                     .encode("utf-8"))
                if not events and time.time() - last_ping > 15:
                    self.wfile.write(b": ping\n\n")          # keep proxies awake
                    last_ping = time.time()
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass


def probe_cameras(limit: int = 6) -> list[int]:
    """Indices that actually open. Slow, so the UI asks for it explicitly."""
    import cv2

    found = []
    for index in range(limit):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            found.append(index)
        cap.release()
    return found


def _read(path) -> str:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return ""


def _stable_path(index: int) -> str:
    """The /dev/v4l/by-id/... name for a camera, if it has one.

    Worth preferring over the index: indices are handed out in probe order, so
    unplugging anything renumbers them, while the by-id name follows the camera.
    """
    byid = Path("/dev/v4l/by-id")
    if not byid.is_dir():
        return ""
    for link in sorted(byid.iterdir()):
        try:
            if link.resolve() == Path(f"/dev/video{index}").resolve():
                return str(link)
        except OSError:
            continue
    return ""


def _linux_cameras() -> list[dict] | None:
    """Cameras from sysfs, without opening any of them.

    Opening each index in turn - which is what probe_cameras does - cannot see
    a camera that something else already has open, so the running scorer's own
    camera would be missing from its own picker. Reading sysfs has no such
    problem, and it also tells us the make and model.
    """
    root = Path("/sys/class/video4linux")
    if not root.is_dir():
        return None
    found = []
    for node in sorted(root.iterdir(), key=lambda p: int(p.name[5:] or 0)):
        index = int(node.name[5:] or 0)
        name = _read(node / "name")
        # A capture-capable node reports index 0; the others are metadata or
        # output nodes of the same device, which nothing can stream from.
        if _read(node / "index") not in ("", "0"):
            continue
        # The Pi's own codec and ISP blocks are video4linux devices too.
        if any(k in name.lower() for k in ("bcm2835", "codec", "isp")):
            continue
        vendor = _read(node / "device/../idVendor")
        product = _read(node / "device/../idProduct")
        stable = _stable_path(index)
        found.append({
            "source": stable or str(index),
            "index": index,
            "label": f"{name} (/dev/video{index})" if name else f"/dev/video{index}",
            "kind": "camera",
            "usb": f"{vendor}:{product}" if vendor else "",
            "stable": bool(stable),
        })
    return found


def probe_devices(limit: int = 6) -> dict:
    """Everything the operator could pick as a source.

    A Kinect is not a video4linux device - libfreenect drives it over raw USB -
    so it never appears as an index no matter how many are scanned. It is found
    on the USB bus instead, and listed whether or not the driver to use it is
    installed, with the reason attached.
    """
    from .kinect import USB_CAMERA, USB_VENDOR, kinect_status, usb_devices

    cameras = _linux_cameras()
    if cameras is None:                       # not Linux: fall back to probing
        cameras = [{"source": str(i), "index": i, "label": f"camera {i}",
                    "kind": "camera", "usb": "", "stable": False}
                   for i in probe_cameras(limit)]

    kinect = kinect_status()
    plugged = usb_devices()
    if plugged and not kinect["driver"]:
        kinect["reason"] = ("a Kinect is plugged in, but it cannot be used yet: "
                            + kinect["reason"])

    for device in range(len(plugged)):
        cameras.append({
            "source": f"kinect:{device}", "index": None,
            "label": f"Kinect v1 #{device}"
                     + ("" if kinect["available"] else " - needs libfreenect"),
            "kind": "kinect", "usb": f"{USB_VENDOR}:{USB_CAMERA}",
            "stable": True, "usable": kinect["available"],
        })

    # Kept for anyone still reading the old shape.
    return {"cameras": [c["index"] for c in cameras if c.get("index") is not None],
            "devices": cameras, "kinect": kinect}


def serve(engine, host: str = "127.0.0.1", port: int = 8080,
          token: str | None = None, verbose: bool = False):
    Handler.engine = engine
    Handler.token = token
    Handler.verbose = verbose
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True
    return httpd
