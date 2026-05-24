#!/usr/bin/env python3
"""
static2rtsp — Publish a periodically-refreshed static image as an RTSP live stream.

Overlays:
  • Current date/time in the bottom-left corner
  • Countdown ring in the bottom-right showing seconds until the next image fetch

Runtime requirements:
  • pip install opencv-python requests numpy
  • ffmpeg in PATH (H.264 encoding + RTSP push)
  • An RTSP server at the publish URL, e.g.:
      docker run -d --rm -p 8554:8554 bluenviron/mediamtx:latest
    or download mediamtx from https://github.com/bluenviron/mediamtx/releases

Example:
  python static2rtsp.py --url https://example.com/snapshot.jpg
  ffplay rtsp://localhost:8554/stream
"""

import argparse
import math
import subprocess
import sys
import threading
import time
from datetime import datetime
from typing import Optional, Tuple

import cv2
import numpy as np
import requests


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


# ──────────────────────────────────────────────────────────────────────────────
# Image fetcher (background thread)
# ──────────────────────────────────────────────────────────────────────────────

class ImageFetcher:
    """Downloads a remote image on a fixed interval, always keeping the latest."""

    def __init__(self, url: str, interval: float, session: Optional[requests.Session] = None):
        self.url = url
        self.interval = interval
        self._session = session or requests.Session()
        self._frame: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._next_fetch = 0.0          # monotonic time of next scheduled fetch
        self._stop = threading.Event()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        """Perform the first fetch synchronously, then schedule background updates."""
        self._do_fetch()
        t = threading.Thread(target=self._loop, daemon=True, name="image-fetcher")
        t.start()

    def stop(self):
        self._stop.set()

    @property
    def frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def countdown(self) -> Tuple[float, float]:
        """Return (seconds_remaining, total_interval)."""
        return max(0.0, self._next_fetch - time.monotonic()), self.interval

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _loop(self):
        while not self._stop.is_set():
            wait = max(0.0, self._next_fetch - time.monotonic())
            if self._stop.wait(wait):
                break
            try:
                self._do_fetch()
            except Exception as exc:
                # Last-resort guard: log and keep looping so the thread never dies
                print(f"[{_ts()}] ERROR: unexpected exception in fetcher — {exc!r}")
                self._next_fetch = time.monotonic() + self.interval

    def _do_fetch(self):
        try:
            real_url = self.url + "?" + str(int(time.time()))
            resp = self._session.get(real_url, timeout=10)
            resp.raise_for_status()
            if not resp.content:
                print(f"[{_ts()}] WARNING: empty response body from {real_url}")
                return
            arr = np.frombuffer(resp.content, np.uint8)
            if arr.size == 0:
                print(f"[{_ts()}] WARNING: zero-length buffer after frombuffer")
                return
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                with self._lock:
                    self._frame = img
                print(f"[{_ts()}] fetched {real_url}  ({img.shape[1]}×{img.shape[0]})")
            else:
                print(f"[{_ts()}] WARNING: could not decode image from response")
        except requests.RequestException as exc:
            print(f"[{_ts()}] WARNING: fetch failed — {exc}")
        finally:
            self._next_fetch = time.monotonic() + self.interval


# ──────────────────────────────────────────────────────────────────────────────
# Frame renderer
# ──────────────────────────────────────────────────────────────────────────────

class Renderer:
    """Scales the source image and composites date/time + countdown overlays."""

    _FONT = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(self, width: int, height: int):
        self.w = width
        self.h = height
        # Pre-built black placeholder shown before the first image arrives
        self._blank = np.zeros((height, width, 3), dtype=np.uint8)

    def render(self, source: Optional[np.ndarray], remaining: float, total: float) -> np.ndarray:
        if source is None:
            frame = self._blank.copy()
            self._draw_waiting(frame)
        else:
            frame = cv2.resize(source, (self.w, self.h), interpolation=cv2.INTER_LINEAR)

        self._draw_spinner(frame, remaining, total)
        return frame

    # ------------------------------------------------------------------

    def _draw_waiting(self, frame: np.ndarray):
        text = "Waiting for image..."
        scale, thickness = 0.8, 2
        (tw, th), _ = cv2.getTextSize(text, self._FONT, scale, thickness)
        x = (self.w - tw) // 2
        y = (self.h + th) // 2
        cv2.putText(frame, text, (x, y), self._FONT, scale, (120, 120, 120), thickness, cv2.LINE_AA)

    def _draw_spinner(self, frame: np.ndarray, remaining: float, total: float):
        if total <= 0:
            return

        pad = 55
        cx, cy = self.w - pad, self.h - pad
        radius = 32
        progress = remaining / total   # 1.0 (just fetched) → 0.0 (about to fetch)

        # Background track
        cv2.circle(frame, (cx, cy), radius, (55, 55, 55), 3, cv2.LINE_AA)

        # Foreground arc — shrinks clockwise from 12 o'clock as countdown ticks down
        if progress > 0.0:
            sweep = int(round(360 * progress))
            # cv2.ellipse: angle=0 means no rotation; startAngle/endAngle measured clockwise
            # from 3-o'clock in OpenCV's coordinate system (y-down → CW == visual CW)
            # Starting at -90° puts the origin at 12 o'clock.
            arc_color = (0, 210, 255) if progress > 0.25 else (0, 100, 255)
            cv2.ellipse(frame, (cx, cy), (radius, radius), 0, -90, -90 + sweep,
                        arc_color, 3, cv2.LINE_AA)

        # Dot at the current arc tip (visual polish)
        if 0.0 < progress < 1.0:
            tip_angle = math.radians(-90 + 360 * progress)
            tx = int(cx + radius * math.cos(tip_angle))
            ty = int(cy + radius * math.sin(tip_angle))
            cv2.circle(frame, (tx, ty), 4, (255, 255, 255), -1, cv2.LINE_AA)

        # Seconds label centred inside the ring
        label = str(int(math.ceil(remaining)))
        lscale, lthick = 0.55, 2
        (lw, lh), _ = cv2.getTextSize(label, self._FONT, lscale, lthick)
        cv2.putText(frame, label,
                    (cx - lw // 2, cy + lh // 2),
                    self._FONT, lscale, (255, 255, 255), lthick, cv2.LINE_AA)


# ──────────────────────────────────────────────────────────────────────────────
# FFmpeg RTSP publisher
# ──────────────────────────────────────────────────────────────────────────────

class FFmpegPublisher:
    """Pipes raw BGR frames to ffmpeg, which encodes and pushes them as RTSP."""

    def __init__(self, rtsp_url: str, width: int, height: int, fps: int):
        self.rtsp_url = rtsp_url
        self.width = width
        self.height = height
        self.fps = fps
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

    def start(self):
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{self.width}x{self.height}",
            "-pix_fmt", "bgr24",
            "-r", str(self.fps),
            "-i", "pipe:0",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-g", str(self.fps * 2),     # keyframe every 2 s for fast seek
            "-f", "rtsp",
            "-rtsp_transport", "tcp",
            self.rtsp_url,
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"[{_ts()}] ffmpeg publishing → {self.rtsp_url}")

    def write(self, frame: np.ndarray):
        with self._lock:
            if self._proc is None:
                return
            try:
                self._proc.stdin.write(frame.tobytes())
            except (BrokenPipeError, OSError):
                print(f"[{_ts()}] pipe broken — restarting ffmpeg in 2 s")
                self._restart_locked()

    def stop(self):
        with self._lock:
            self._shutdown_locked()

    # ------------------------------------------------------------------

    def _restart_locked(self):
        self._shutdown_locked()
        time.sleep(2)
        self.start()

    def _shutdown_locked(self):
        if self._proc:
            try:
                self._proc.stdin.close()
            except Exception:
                pass
            try:
                self._proc.terminate()
                self._proc.wait(timeout=3)
            except Exception:
                pass
            self._proc = None


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def _check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        sys.exit("ERROR: ffmpeg not found in PATH. Install it and try again.")


def main():
    ap = argparse.ArgumentParser(
        description="Serve a periodically-fetched static image as an RTSP live stream.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--url",      required=True,                    help="HTTP(S) URL of the source image")
    ap.add_argument("--interval", type=float,  default=60.0,        help="Image refresh interval in seconds", metavar="SEC")
    ap.add_argument("--rtsp",     default="rtsp://localhost:8554/stream", help="RTSP publish URL",            metavar="URL")
    ap.add_argument("--width",    type=int,    default=1280,         help="Output frame width")
    ap.add_argument("--height",   type=int,    default=720,          help="Output frame height")
    ap.add_argument("--fps",      type=int,    default=25,           help="Output frame rate")
    args = ap.parse_args()

    _check_ffmpeg()

    print("=" * 60)
    print(f"  static2rtsp")
    print(f"  source   : {args.url}")
    print(f"  interval : {args.interval} s")
    print(f"  output   : {args.rtsp}")
    print(f"  format   : {args.width}×{args.height} @ {args.fps} fps")
    print("=" * 60)
    print("  RTSP server required — e.g.:")
    print("  docker run -d --rm -p 8554:8554 bluenviron/mediamtx:latest")
    print("  Playback:  ffplay", args.rtsp)
    print("=" * 60)

    fetcher   = ImageFetcher(args.url, args.interval)
    renderer  = Renderer(args.width, args.height)
    publisher = FFmpegPublisher(args.rtsp, args.width, args.height, args.fps)

    fetcher.start()
    publisher.start()

    frame_budget = 1.0 / args.fps

    try:
        while True:
            t0 = time.monotonic()

            remaining, total = fetcher.countdown()
            frame = renderer.render(fetcher.frame, remaining, total)
            publisher.write(frame)

            slack = frame_budget - (time.monotonic() - t0)
            if slack > 0:
                time.sleep(slack)

    except KeyboardInterrupt:
        print(f"\n[{_ts()}] interrupted — shutting down")
    finally:
        fetcher.stop()
        publisher.stop()


if __name__ == "__main__":
    main()
