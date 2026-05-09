"""Stream simulator: serve local construction footage as looping RTSP streams.

Why FFmpeg + RTSP instead of just feeding files directly into the app:
    The end deployment uses RTSP cameras. By simulating that exact transport
    in dev, we exercise the StreamManager's reconnection logic, frame-skip
    timing, and error handling against a realistic source — not just a
    happy-path local file read.

Requirements:
    - ffmpeg installed and on PATH
    - mediamtx (formerly rtsp-simple-server) running on localhost:8554
      Install: https://github.com/bluenviron/mediamtx/releases

Usage:
    python -m simulation.stream_simulator \\
        --videos cam_01:/path/to/site_north.mp4 \\
                 cam_02:/path/to/site_east.mp4 \\
        --rtsp-host localhost --rtsp-port 8554

Then run the app with:
    LABOR_INTEL_STREAM_SOURCES=cam_01=rtsp://localhost:8554/cam_01,\\
                               cam_02=rtsp://localhost:8554/cam_02

Each ffmpeg child process loops its video file with -stream_loop -1, so the
simulator runs indefinitely until you Ctrl-C.
"""
from __future__ import annotations

import argparse
import logging
import shutil
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
logger = logging.getLogger("stream_simulator")


@dataclass
class VideoSource:
    camera_id: str
    path: Path

    @classmethod
    def parse(cls, spec: str) -> "VideoSource":
        if ":" not in spec:
            raise argparse.ArgumentTypeError(
                f"Bad video spec '{spec}'. Need 'cam_id:path/to/video.mp4'."
            )
        cam_id, path = spec.split(":", 1)
        p = Path(path)
        if not p.exists():
            raise argparse.ArgumentTypeError(f"Video file does not exist: {p}")
        return cls(camera_id=cam_id, path=p)


def _ffmpeg_cmd(src: VideoSource, rtsp_url: str) -> list[str]:
    # -re reads input at native frame rate (so RTSP timing matches video FPS)
    # -stream_loop -1 loops forever
    # -c:v libx264 re-encodes to ensure compatibility with downstream OpenCV
    # -tune zerolatency keeps encoding fast for live preview
    # -f rtsp pushes via RTSP TCP transport
    return [
        "ffmpeg",
        "-loglevel", "warning",
        "-re",
        "-stream_loop", "-1",
        "-i", str(src.path),
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-tune", "zerolatency",
        "-pix_fmt", "yuv420p",
        "-an",  # drop audio
        "-f", "rtsp",
        "-rtsp_transport", "tcp",
        rtsp_url,
    ]


def main() -> int:
    if shutil.which("ffmpeg") is None:
        logger.error("ffmpeg not found on PATH. Install ffmpeg first.")
        return 2

    parser = argparse.ArgumentParser(description="Loop local videos to RTSP")
    parser.add_argument(
        "--videos", nargs="+", required=True, type=VideoSource.parse,
        help="cam_id:path entries, e.g. cam_01:/data/north.mp4",
    )
    parser.add_argument("--rtsp-host", default="localhost")
    parser.add_argument("--rtsp-port", type=int, default=8554)
    args = parser.parse_args()

    procs: list[tuple[VideoSource, subprocess.Popen]] = []
    try:
        for src in args.videos:
            url = f"rtsp://{args.rtsp_host}:{args.rtsp_port}/{src.camera_id}"
            cmd = _ffmpeg_cmd(src, url)
            logger.info("Starting %s -> %s", src.path.name, url)
            p = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            procs.append((src, p))

        # Wait for any to die; print a diagnostic if so.
        def _shutdown(*_):
            logger.info("Shutting down ffmpeg children")
            for _, p in procs:
                if p.poll() is None:
                    p.terminate()

        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

        # Block until any child exits — RTSP server going down would do this.
        while True:
            for src, p in procs:
                rc = p.poll()
                if rc is not None:
                    err = p.stderr.read().decode(errors="replace") if p.stderr else ""
                    logger.error(
                        "ffmpeg for %s exited (rc=%d): %s",
                        src.camera_id, rc, err[-500:],
                    )
                    return 1
            try:
                signal.pause()
            except AttributeError:
                # Windows doesn't have signal.pause
                import time
                time.sleep(1)
    finally:
        for _, p in procs:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
