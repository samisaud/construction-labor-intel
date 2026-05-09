from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Iterator

import cv2                                

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class StreamSource:
    """Stream definition. RTSP URLs and local paths look the same to OpenCV."""
    camera_id: str
    uri: str                                                      

@dataclass(frozen=True)
class FramePacket:
    camera_id: str
    frame_idx: int                                                                      
    captured_monotonic: float
    frame_bgr: "cv2.Mat"                       

class StreamReader(threading.Thread):
    """Reads one stream, samples to target FPS, pushes to a shared queue."""

    def __init__(
        self,
        source: StreamSource,
        out_queue: queue.Queue[FramePacket],
        target_fps: int,
        shutdown: threading.Event,
        reconnect_max_seconds: int = 30,
    ) -> None:
        super().__init__(name=f"StreamReader[{source.camera_id}]", daemon=True)
        self.source = source
        self.out_queue = out_queue
        self.target_fps = target_fps
        self.shutdown = shutdown
        self.reconnect_max_seconds = reconnect_max_seconds

                                         
        self.frames_read = 0
        self.frames_dropped_full_queue = 0
        self.reconnect_count = 0
        self.last_frame_monotonic: float = 0.0

    def _open(self) -> tuple[cv2.VideoCapture, int]:
        """Open capture and compute frame-skip stride for target FPS."""
                                                                             
                                                        
        uri: str | int = self.source.uri
        if uri.isdigit():
            uri = int(uri)
        cap = cv2.VideoCapture(uri)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open stream: {self.source.uri}")

        src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
                                                                    
        if src_fps <= 0 or src_fps > 120:
            logger.warning(
                "Stream %s reports fps=%.2f, defaulting stride to 1",
                self.source.camera_id, src_fps,
            )
            stride = max(1, 25 // self.target_fps)
        else:
            stride = max(1, int(round(src_fps / self.target_fps)))

        logger.info(
            "Opened %s (src_fps=%.1f, target=%dfps, stride=%d)",
            self.source.camera_id, src_fps, self.target_fps, stride,
        )
        return cap, stride

    def _read_loop(self, cap: cv2.VideoCapture, stride: int) -> None:
        raw_idx = 0
        while not self.shutdown.is_set():
            ok, frame = cap.read()
            if not ok:
                                                            
                logger.warning("Read failed for %s at raw_idx=%d", self.source.camera_id, raw_idx)
                return

            raw_idx += 1
            if raw_idx % stride != 0:
                continue

            packet = FramePacket(
                camera_id=self.source.camera_id,
                frame_idx=self.frames_read,
                captured_monotonic=time.monotonic(),
                frame_bgr=frame,
            )
            self.last_frame_monotonic = packet.captured_monotonic
            self.frames_read += 1

                                                                   
            try:
                self.out_queue.put_nowait(packet)
            except queue.Full:
                try:
                    self.out_queue.get_nowait()
                    self.frames_dropped_full_queue += 1
                except queue.Empty:
                    pass
                try:
                    self.out_queue.put_nowait(packet)
                except queue.Full:
                                                                     
                    self.frames_dropped_full_queue += 1

    def run(self) -> None:
        backoff = 1.0
        while not self.shutdown.is_set():
            cap = None
            try:
                cap, stride = self._open()
                backoff = 1.0                            
                self._read_loop(cap, stride)
            except Exception:
                logger.exception("Stream %s reader crashed", self.source.camera_id)
            finally:
                if cap is not None:
                    cap.release()

            if self.shutdown.is_set():
                break

            self.reconnect_count += 1
            sleep_for = min(backoff, float(self.reconnect_max_seconds))
            logger.info(
                "Reconnecting %s in %.1fs (attempt %d)",
                self.source.camera_id, sleep_for, self.reconnect_count,
            )
                                           
            self.shutdown.wait(timeout=sleep_for)
            backoff = min(backoff * 2, float(self.reconnect_max_seconds))

class StreamManager:
    """Owns all stream readers and exposes a unified frame iterator.

    Usage:
        mgr = StreamManager(sources=[...], target_fps=3, queue_maxsize=10)
        mgr.start()
        for packet in mgr.frames():
            ...
        mgr.stop()
    """

    def __init__(
        self,
        sources: list[StreamSource],
        target_fps: int = 3,
        queue_maxsize: int = 10,
        reconnect_max_seconds: int = 30,
    ) -> None:
        if not sources:
            raise ValueError("At least one stream source required")
                                                                       
        self._queues: dict[str, queue.Queue[FramePacket]] = {
            s.camera_id: queue.Queue(maxsize=queue_maxsize) for s in sources
        }
        self._shutdown = threading.Event()
        self._readers: list[StreamReader] = [
            StreamReader(
                source=s,
                out_queue=self._queues[s.camera_id],
                target_fps=target_fps,
                shutdown=self._shutdown,
                reconnect_max_seconds=reconnect_max_seconds,
            )
            for s in sources
        ]
                                                              
        self._poll_order = deque(s.camera_id for s in sources)

    def start(self) -> None:
        for r in self._readers:
            r.start()
        logger.info("StreamManager started %d readers", len(self._readers))

    def stop(self, timeout_seconds: float = 5.0) -> None:
        self._shutdown.set()
        for r in self._readers:
            r.join(timeout=timeout_seconds)
        logger.info("StreamManager stopped")

    def frames(self, poll_interval: float = 0.01) -> Iterator[FramePacket]:
        """Yield frames in round-robin across cameras as they become available.

        Blocks briefly when no camera has a frame ready. Exits cleanly on shutdown.
        """
        while not self._shutdown.is_set():
            yielded_any = False
                                          
            for _ in range(len(self._poll_order)):
                cam_id = self._poll_order[0]
                self._poll_order.rotate(-1)
                try:
                    yield self._queues[cam_id].get_nowait()
                    yielded_any = True
                except queue.Empty:
                    continue
            if not yielded_any:
                                                                       
                time.sleep(poll_interval)

    def stats(self) -> dict[str, dict[str, float | int]]:
        """Per-camera stats for the /health endpoint."""
        now = time.monotonic()
        return {
            r.source.camera_id: {
                "frames_read": r.frames_read,
                "frames_dropped_full_queue": r.frames_dropped_full_queue,
                "reconnect_count": r.reconnect_count,
                "seconds_since_last_frame": (
                    now - r.last_frame_monotonic if r.last_frame_monotonic else -1
                ),
                "alive": r.is_alive(),
            }
            for r in self._readers
        }
