from __future__ import annotations

import asyncio
import logging
import os
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.analytics_engine import AnalyticsEngine
from app.anomaly_detector import AnomalyDetector
from app.config import settings
from app.inference_engine import InferenceEngine
from app.schemas import (
    HealthResponse,
    SnapshotPayload,
    TrackedWorker,
    WSEnvelope,
    WSMessageType,
    ZoneStats,
)
from app.stream_manager import StreamManager, StreamSource
from app.time_series_store import TimeSeriesStore

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
logger = logging.getLogger("labor_intel")

VERSION = "0.1.0"

                                                            

def _resolve_stream_sources() -> list[StreamSource]:
    """Sources come from env LABOR_INTEL_STREAM_SOURCES as comma-separated
    'cam_id=uri' pairs:
        LABOR_INTEL_STREAM_SOURCES=cam_01=rtsp://x,cam_02=/path/to/file.mp4

    If unset, return an empty list — startup will warn but won't crash, so the
    API can come up for development even without streams configured.
    """
    env = os.environ.get("LABOR_INTEL_STREAM_SOURCES", "").strip()
    if not env:
        return []
    sources = []
    for piece in env.split(","):
        if "=" not in piece:
            logger.warning("Bad stream source spec '%s' (need cam_id=uri)", piece)
            continue
        cam_id, uri = piece.split("=", 1)
        sources.append(StreamSource(camera_id=cam_id.strip(), uri=uri.strip()))
    return sources

                                                         

class WSBroadcaster:
    """Fan-out to connected WebSocket clients with per-client backpressure.

    Each client gets a small bounded queue; if it fills, we drop the oldest
    message. No client can stall the broadcaster.
    """

    def __init__(self, per_client_queue: int = 16) -> None:
        self._clients: dict[WebSocket, asyncio.Queue[str]] = {}
        self._per_client_queue = per_client_queue
        self._lock = asyncio.Lock()

    async def register(self, ws: WebSocket) -> asyncio.Queue[str]:
        q: asyncio.Queue[str] = asyncio.Queue(maxsize=self._per_client_queue)
        async with self._lock:
            self._clients[ws] = q
        return q

    async def unregister(self, ws: WebSocket) -> None:
        async with self._lock:
            self._clients.pop(ws, None)

    async def broadcast(self, envelope: WSEnvelope) -> None:
                                                   
        payload = envelope.model_dump_json()
        async with self._lock:
            clients = list(self._clients.items())
        for ws, q in clients:
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                                         
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    q.put_nowait(payload)
                except asyncio.QueueFull:
                    pass

    @property
    def client_count(self) -> int:
        return len(self._clients)

                                                       

class AppState:
    """All long-lived components live here. Stored on app.state."""

    def __init__(self) -> None:
        self.store: TimeSeriesStore | None = None
        self.inference: InferenceEngine | None = None
        self.analytics: AnalyticsEngine | None = None
        self.anomaly: AnomalyDetector | None = None
        self.streams: StreamManager | None = None
        self.broadcaster = WSBroadcaster()

                                                                               
                                         
        self.snapshot_lock = asyncio.Lock()
        self.latest_workers: list[TrackedWorker] = []
        self.latest_workers_ts: datetime = datetime.now(timezone.utc)

                                                     
        self._inference_task: asyncio.Task | None = None
        self._analytics_task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()

                                                    

async def _inference_loop(state: AppState) -> None:
    """Drain frames from StreamManager, run detection, update latest_workers.

    cv2.VideoCapture and Ultralytics inference are blocking. We pull frames in
    an executor to avoid blocking the event loop.
    """
    assert state.streams is not None and state.inference is not None
    loop = asyncio.get_running_loop()

    def _next_frame_blocking():
                                                                           
        for packet in state.streams.frames(poll_interval=0.02):
            return packet
        return None

                                                                                 
                                                              
    per_camera: dict[str, list[TrackedWorker]] = {}

    while not state._stop_event.is_set():
        try:
            packet = await loop.run_in_executor(None, _next_frame_blocking)
            if packet is None:
                continue
            workers = await loop.run_in_executor(
                None,
                state.inference.process_frame,
                packet.frame_bgr,
                packet.camera_id,
                packet.frame_idx,
            )
            per_camera[packet.camera_id] = workers

                                                     
            merged: list[TrackedWorker] = []
            for ws in per_camera.values():
                merged.extend(ws)
            async with state.snapshot_lock:
                state.latest_workers = merged
                state.latest_workers_ts = datetime.now(timezone.utc)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("inference_loop error")
            await asyncio.sleep(0.1)

async def _analytics_loop(state: AppState) -> None:
    """Every analytics_tick_seconds: compute zone stats, persist, broadcast."""
    assert (state.analytics is not None and state.store is not None
            and state.anomaly is not None)
    tick = settings.analytics_tick_seconds

    while not state._stop_event.is_set():
        await asyncio.sleep(tick)
        try:
            async with state.snapshot_lock:
                workers = list(state.latest_workers)

            now = datetime.now(timezone.utc)
            zone_stats, tagged_workers = state.analytics.compute_zone_stats(
                workers, now=now,
            )

                     
            await state.store.write_zone_stats(zone_stats)
            await state.store.write_worker_tracks(tagged_workers, ts=now)

                       
            anomalies = state.anomaly.evaluate(zone_stats)
            for a in anomalies:
                await state.store.write_anomaly(a)
                await state.broadcaster.broadcast(WSEnvelope(
                    type=WSMessageType.ANOMALY,
                    timestamp=now,
                    payload=a.model_dump(mode="json"),
                ))

                                
            payload = SnapshotPayload(
                zone_stats=zone_stats,
                tracked_workers=tagged_workers,
                active_camera_count=(
                    sum(1 for s in (state.streams.stats() if state.streams else {}).values()
                        if s.get("alive"))
                ),
            )
            await state.broadcaster.broadcast(WSEnvelope(
                type=WSMessageType.SNAPSHOT,
                timestamp=now,
                payload=payload.model_dump(mode="json"),
            ))
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("analytics_loop error")

                                            

@asynccontextmanager
async def lifespan(app: FastAPI):
    state = AppState()
    app.state.app_state = state

             
    state.store = TimeSeriesStore()
    await state.store.open()

                                                                     
    state.analytics = AnalyticsEngine()
    critical = {z.zone_id for z in state.analytics.zones if z.critical_path}
    state.anomaly = AnomalyDetector(critical_path_zones=critical)

                                                                         
                                                                        
                                                                      
    try:
        state.inference = InferenceEngine()
    except (FileNotFoundError, ModuleNotFoundError, ImportError) as e:
        logger.error("Inference engine init failed: %s", e)
        state.inference = None

                                                                                        
    sources = _resolve_stream_sources()
    if sources and state.inference is not None:
        state.streams = StreamManager(
            sources=sources,
            target_fps=settings.stream_target_fps,
            queue_maxsize=settings.stream_queue_maxsize,
            reconnect_max_seconds=settings.stream_reconnect_max_seconds,
        )
        state.streams.start()
        state._inference_task = asyncio.create_task(
            _inference_loop(state), name="inference_loop",
        )
    else:
        logger.warning(
            "Starting in degraded mode (sources=%d, inference=%s)",
            len(sources), "ok" if state.inference else "missing",
        )

    state._analytics_task = asyncio.create_task(
        _analytics_loop(state), name="analytics_loop",
    )

    try:
        yield
    finally:
        logger.info("Shutting down...")
        state._stop_event.set()
        for t in (state._inference_task, state._analytics_task):
            if t is not None:
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass
        if state.streams is not None:
            state.streams.stop(timeout_seconds=5.0)
        if state.store is not None:
            await state.store.close()
        logger.info("Shutdown complete")

app = FastAPI(title="Construction Labor Intel", version=VERSION, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _state(req_app: FastAPI) -> AppState:
    return req_app.state.app_state                               

                                                  

@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    s = _state(app)
    db_ok = await s.store.db_writable() if s.store else False
    streams_alive = (
        sum(1 for st in s.streams.stats().values() if st["alive"])
        if s.streams else 0
    )
    if s.inference is None or not db_ok:
        status = "degraded"
    elif s.streams and streams_alive == 0:
        status = "degraded"
    else:
        status = "ok"
    return HealthResponse(
        status=status,
        version=VERSION,
        active_streams=streams_alive,
        inference_provider=settings.inference_provider,
        db_writable=db_ok,
    )

@app.get("/zones/{zone_id}/trend")
async def zone_trend(zone_id: str, minutes: int = Query(60, ge=1, le=1440)):
    s = _state(app)
    if s.store is None:
        raise HTTPException(503, "store not ready")
    return await s.store.get_zone_trend(zone_id, minutes)

@app.get("/shift/summary")
async def shift_summary(hours: int = Query(8, ge=1, le=24)):
    s = _state(app)
    if s.store is None:
        raise HTTPException(503, "store not ready")
    return await s.store.get_shift_summary(hours)

@app.get("/anomalies")
async def anomalies(since_minutes: int = Query(60, ge=1, le=1440)):
    s = _state(app)
    if s.store is None:
        raise HTTPException(503, "store not ready")
    since = datetime.now(timezone.utc) - timedelta(minutes=since_minutes)
    return await s.store.get_anomalies_since(since)

@app.get("/preview/{camera_id}.jpg")
async def preview_jpeg(camera_id: str):
    s = _state(app)
    if s.inference is None:
        raise HTTPException(503, "inference engine not loaded")
    jpg = s.inference.get_latest_jpeg(camera_id)
    if jpg is None:
        raise HTTPException(404, f"no frame available for {camera_id}")
    return Response(content=jpg, media_type="image/jpeg",
                    headers={"Cache-Control": "no-store"})

                                             

@app.websocket("/ws/live")
async def ws_live(ws: WebSocket) -> None:
    s = _state(ws.app)
    await ws.accept()
    q = await s.broadcaster.register(ws)
    logger.info("WS client connected (total=%d)", s.broadcaster.client_count)
    try:
        while True:
            msg = await q.get()
            await ws.send_text(msg)
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("WS client error")
    finally:
        await s.broadcaster.unregister(ws)
        logger.info("WS client disconnected (total=%d)", s.broadcaster.client_count)
