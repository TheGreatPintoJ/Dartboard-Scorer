"""Find out where a jittery picture is actually coming from.

A stuttering feed can be the camera, the capture settings, this machine being
too slow, or the browser being asked to decode more than it can. Those need
different fixes, so this measures each stage separately and says which one is
at fault.

The interesting measurement is the horizontal wrap check. A frame that has been
read from the buffer at the wrong offset comes out shifted sideways with the
content wrapping around the edge; comparing consecutive frames of a still scene
by circular cross-correlation finds it. If the camera and the served stream are
both clean and the picture on screen is not, the fault is past the socket -
browser compositing, GPU driver, or the screen recorder - not this program.
"""

from __future__ import annotations

import json
import time
import urllib.request

import cv2
import numpy as np

from . import render
from .camera import BACKENDS, default_backend, fourcc_name
from .detector import DartDetector
from .synthetic import synthetic_camera


def horizontal_wrap(a: np.ndarray, b: np.ndarray) -> int:
    """Circular horizontal offset between two frames, in pixels.

    Zero for a static scene from a healthy camera. Anything else means frames
    are arriving misaligned.
    """
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    a -= a.mean()
    b -= b.mean()
    spectrum = np.fft.rfft(a, axis=1) * np.conj(np.fft.rfft(b, axis=1))
    correlation = np.fft.irfft(spectrum, axis=1).sum(axis=0)
    peak = int(np.argmax(correlation))
    width = a.shape[1]
    return peak if peak < width // 2 else peak - width


def _grab(cap, count):
    frames = []
    start = time.time()
    for _ in range(count):
        ok, frame = cap.read()
        if ok:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    return frames, time.time() - start


def probe_camera(index: int, backend: str | None = None, frames: int = 40) -> list[dict]:
    """Try the usual formats and report what each actually delivers."""
    name = backend or default_backend()
    api = BACKENDS.get(name, cv2.CAP_ANY)
    results = []
    for width, height in ((640, 480), (1280, 720), (1920, 1080)):
        for code in ("", "MJPG"):
            cap = cv2.VideoCapture(index, api)
            if not cap.isOpened():
                cap.release()
                results.append({"requested": f"{width}x{height} {code or 'default'}",
                                "error": "camera did not open"})
                continue
            if code:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*code))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            for _ in range(8):                       # let exposure settle
                cap.read()
            grabbed, elapsed = _grab(cap, frames)
            row = {
                "requested": f"{width}x{height} {code or 'default'}",
                "got": f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
                       f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}",
                "fourcc": fourcc_name(cap.get(cv2.CAP_PROP_FOURCC)),
                "fps": round(len(grabbed) / elapsed, 1) if elapsed else 0.0,
            }
            cap.release()
            if len(grabbed) < 5:
                row["error"] = "no frames"
            else:
                shifts = [horizontal_wrap(a, b) for a, b in zip(grabbed, grabbed[1:])]
                row["wrapped"] = sum(1 for s in shifts if abs(s) > 2)
                row["of"] = len(shifts)
                row["max_shift"] = max(abs(s) for s in shifts)
            results.append(row)
    return results


def probe_pipeline(width=1280, height=720) -> dict:
    """How long this machine takes to detect, annotate and encode one frame."""
    board, view, calib = synthetic_camera(width, height)
    frame = view(board)
    detector = DartDetector(calib)
    for _ in range(30):
        detector.update(frame)

    def timed(fn, n=30):
        fn()
        start = time.perf_counter()
        for _ in range(n):
            fn()
        return (time.perf_counter() - start) / n * 1000

    detect = timed(lambda: detector.update(frame))
    overlay = timed(lambda: render.draw_board_overlay(frame.copy(), calib))
    encode = timed(lambda: cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75]))
    total = detect + overlay + encode
    return {"detect_ms": round(detect, 2), "overlay_ms": round(overlay, 2),
            "encode_ms": round(encode, 2), "total_ms": round(total, 2),
            "ceiling_fps": round(1000 / total, 1)}


def probe_stream(url: str, seconds: float = 8.0, token: str | None = None) -> dict:
    """Measure the MJPEG stream a running server is actually serving."""
    if not url.endswith(".mjpg"):
        url = url.rstrip("/") + "/stream.mjpg"
    if token:
        url += ("&" if "?" in url else "?") + "token=" + token
    response = urllib.request.urlopen(url, timeout=20)
    buffer = b""
    jpegs, times = [], []
    start = time.perf_counter()
    try:
        while time.perf_counter() - start < seconds:
            chunk = response.read(8192)
            if not chunk:
                break
            buffer += chunk
            while True:
                begin = buffer.find(b"\xff\xd8")
                end = buffer.find(b"\xff\xd9", begin + 2)
                if begin < 0 or end < 0:
                    break
                jpegs.append(buffer[begin:end + 2])
                times.append(time.perf_counter())
                buffer = buffer[end + 2:]
    finally:
        response.close()

    if len(jpegs) < 6:
        return {"error": f"only {len(jpegs)} frames arrived"}
    gaps = [(b - a) * 1000 for a, b in zip(times, times[1:])][2:]
    images = [cv2.cvtColor(cv2.imdecode(np.frombuffer(j, np.uint8), cv2.IMREAD_COLOR),
                           cv2.COLOR_BGR2GRAY) for j in jpegs[3:43]]
    shifts = [horizontal_wrap(a, b) for a, b in zip(images, images[1:])]
    return {
        "fps": round(len(gaps) / seconds, 1),
        "gap_mean_ms": round(sum(gaps) / len(gaps), 1),
        "gap_max_ms": round(max(gaps), 1),
        "jitter_ms": round(float(np.std(gaps)), 1),
        "wrapped": sum(1 for s in shifts if abs(s) > 2),
        "of": len(shifts),
        "max_shift_px": max(abs(s) for s in shifts) if shifts else 0,
        "kb_per_frame": round(sum(len(j) for j in jpegs) / len(jpegs) / 1024, 1),
    }


def report(index: int | None, url: str | None, token: str | None = None) -> int:
    """Run the checks and say, in words, which stage is at fault."""
    verdicts = []

    print("== this machine ==")
    pipeline = probe_pipeline()
    print(f"  detect {pipeline['detect_ms']}ms + overlay {pipeline['overlay_ms']}ms "
          f"+ encode {pipeline['encode_ms']}ms = {pipeline['total_ms']}ms per frame")
    print(f"  ceiling: {pipeline['ceiling_fps']} fps at 1280x720")
    if pipeline["ceiling_fps"] < 25:
        verdicts.append("This machine cannot process 25 fps at 720p. Lower the "
                        "resolution, or accept a lower rate.")

    if index is not None:
        print(f"\n== camera {index} ==")
        print(f"  {'requested':24s} {'got':11s} {'fourcc':7s} {'fps':>6s} "
              f"{'wrapped':>9s} {'max shift':>10s}")
        best = 0.0
        clean = True
        for row in probe_camera(index):
            if "error" in row:
                print(f"  {row['requested']:24s} {row['error']}")
                continue
            wrapped = f"{row['wrapped']}/{row['of']}"
            print(f"  {row['requested']:24s} {row['got']:11s} {row['fourcc']:7s} "
                  f"{row['fps']:6.1f} {wrapped:>9s} {row['max_shift']:10d}")
            best = max(best, row["fps"])
            clean = clean and row["wrapped"] == 0
        if best < 20:
            verdicts.append(f"The camera never exceeded {best:.0f} fps. Try the MJPG "
                            "row above, or a lower resolution.")
        if not clean:
            verdicts.append("The camera returned misaligned frames. Update the camera "
                            "driver, try another USB port, and avoid USB hubs.")

    if url:
        print(f"\n== stream from {url} ==")
        stream = probe_stream(url, token=token)
        if "error" in stream:
            print(f"  {stream['error']}")
        else:
            print(f"  {stream['fps']} fps, {stream['kb_per_frame']} KB/frame")
            print(f"  gap mean {stream['gap_mean_ms']}ms, jitter {stream['jitter_ms']}ms, "
                  f"worst {stream['gap_max_ms']}ms")
            print(f"  horizontal wrap: {stream['wrapped']}/{stream['of']} frames, "
                  f"max {stream['max_shift_px']} px")
            if stream["wrapped"] == 0 and stream["fps"] >= 15:
                verdicts.append(
                    "The served stream is clean and steady. If the picture on "
                    "screen is not, the fault is after the socket: the browser, "
                    "the GPU driver, or the screen recorder - not this program. "
                    "Try turning off browser hardware acceleration, or open "
                    "/stream.mjpg in another browser to compare.")
            if stream["fps"] > 30:
                verdicts.append(
                    f"Serving {stream['fps']} fps is more than a browser decodes "
                    "comfortably. Lower 'stream fps cap' in Settings to 20.")

    print("\n== verdict ==")
    if not verdicts:
        print("  Nothing obviously wrong.")
    for line in verdicts:
        print(f"  - {line}")
    return 0
