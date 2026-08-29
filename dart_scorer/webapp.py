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
            if path == "/stream.mjpg":
                return self._stream()
            if path == "/snapshot.jpg":
                annotated = (query.get("annotated") or ["1"])[0] != "0"
                return self._image(self.engine.snapshot(annotated))
            if path == "/rectified.jpg":
                return self._image(self.engine.rectified())
            if path == "/api/status":
                return self._json(self.engine.status())
            if path == "/api/events":
                return self._events(int((query.get("cursor") or ["0"])[0]))
            if path == "/api/config":
                return self._json(self.engine.config.to_dict())
            if path == "/api/calibration":
                status = self.engine.status()
                return self._json({"calibrated": status["calibrated"],
                                   "points": status["calibration_points"],
                                   "frame_size": status["frame_size"]})
            if path == "/api/throws.csv":
                return self._file(Path(self.engine.config.log_path), None,
                                  "text/csv; charset=utf-8")
            if path == "/api/cameras":
                return self._json({"cameras": probe_cameras()})
        except BrokenPipeError:
            return
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

            if path == "/api/calibration":
                points = body.get("points") or []
                if len(points) < 4:
                    return self._error(HTTPStatus.BAD_REQUEST,
                                       "four landmarks are needed")
                result = self.engine.set_calibration(points, save=body.get("save", True))
                return self._json(result)

            if path == "/api/calibration/preview":
                jpeg = self.engine.preview(body.get("points") or [])
                if jpeg is None:
                    return self._error(HTTPStatus.BAD_REQUEST,
                                       "need a frame and four usable landmarks")
                return self._image(jpeg)

            if path == "/api/calibration/auto":
                points = self.engine.auto_points()
                if points is None:
                    return self._error(HTTPStatus.NOT_FOUND,
                                       "could not find the board - place the "
                                       "landmarks by hand")
                return self._json({"points": points})

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
            self.engine.clear_calibration()
            return self._json({"calibrated": False})
        return self._error(HTTPStatus.NOT_FOUND, f"no route for {parsed.path}")

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

    def _stream(self):
        self.close_connection = True
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type",
                         f"multipart/x-mixed-replace; boundary={BOUNDARY}")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            for jpeg in self.engine.stream():
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


def serve(engine, host: str = "127.0.0.1", port: int = 8080,
          token: str | None = None, verbose: bool = False):
    Handler.engine = engine
    Handler.token = token
    Handler.verbose = verbose
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True
    return httpd
