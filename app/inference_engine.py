from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.config import settings
from app.schemas import (
    ActivityState,
    BoundingBox,
    PPEStatus,
    TrackedWorker,
)

logger = logging.getLogger(__name__)

                                                                                     
CLS_HARDHAT = 0
CLS_NO_HARDHAT = 2
CLS_NO_VEST = 4
CLS_PERSON = 5
CLS_VEST = 7

@dataclass
class _TrackHistory:
    """Per-track rolling state. Kept in inference engine memory."""
                                                                            
                                         
    centroids: deque[tuple[int, float, float]] = field(
        default_factory=lambda: deque(maxlen=10)
    )
    last_seen_monotonic: float = 0.0

                   
    state: ActivityState = ActivityState.TRANSITIONING
    last_active_signal_monotonic: float = 0.0
    idle_signal_started_monotonic: float | None = None

def _bbox_iou(a: tuple[float, float, float, float],
              b: tuple[float, float, float, float]) -> float:
    """IoU on (x1, y1, x2, y2). All coords same units (any units)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0

def _associate_ppe(
    person_box: tuple[float, float, float, float],
    hardhats: list[tuple[float, float, float, float]],
    vests: list[tuple[float, float, float, float]],
) -> PPEStatus:
    """Decide PPE status for one person via geometric overlap.

    Hardhat must overlap the top band of the person bbox (configurable).
    Vest must overlap the middle band. IoU threshold prevents tiny spurious matches.
    """
    px1, py1, px2, py2 = person_box
    ph = py2 - py1
    if ph <= 0:
        return PPEStatus.UNKNOWN

                                                  
    hh_top, hh_bot = settings.ppe_hardhat_zone
    vest_top, vest_bot = settings.ppe_vest_zone
    hardhat_zone = (px1, py1 + hh_top * ph, px2, py1 + hh_bot * ph)
    vest_zone = (px1, py1 + vest_top * ph, px2, py1 + vest_bot * ph)

    has_hardhat = any(
        _bbox_iou(hardhat_zone, hh) >= settings.ppe_min_iou for hh in hardhats
    )
    has_vest = any(
        _bbox_iou(vest_zone, v) >= settings.ppe_min_iou for v in vests
    )

    if has_hardhat and has_vest:
        return PPEStatus.COMPLIANT
    if has_hardhat and not has_vest:
        return PPEStatus.TORSO_VIOLATION
    if has_vest and not has_hardhat:
        return PPEStatus.HEAD_VIOLATION
    return PPEStatus.BOTH_VIOLATION

class InferenceEngine:
    """Wraps Ultralytics YOLO model, ByteTrack, and PPE/activity logic.

    One instance per process. Not thread-safe — caller serializes frame submission.
    """

    def __init__(self) -> None:
                                                                       
        from ultralytics import YOLO                                

        weights = settings.resolve(settings.weights_path)
        if not weights.exists():
            raise FileNotFoundError(
                f"Weights not found at {weights}. Run training and export ONNX, "
                "or set LABOR_INTEL_WEIGHTS_PATH."
            )

        logger.info("Loading model: %s", weights)
        self.model = YOLO(str(weights))
        self.imgsz = settings.inference_imgsz

                          
        self._tracks: dict[int, _TrackHistory] = defaultdict(_TrackHistory)

                                                              
        self._latest_annotated_jpeg: dict[str, bytes] = {}

                                   
        if self.model.names and len(self.model.names) != len(settings.class_names):
            logger.warning(
                "Model has %d classes, settings expects %d. Class indices may misalign.",
                len(self.model.names), len(settings.class_names),
            )

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        camera_id: str,
        frame_idx: int,
    ) -> list[TrackedWorker]:
        """Run full pipeline on one frame. Returns tracked workers (Persons only)."""
        h, w = frame_bgr.shape[:2]
        if h == 0 or w == 0:
            return []

                                                                         
                                                                                     
        results = self.model.track(
            source=frame_bgr,
            imgsz=self.imgsz,
            conf=settings.detection_conf_threshold,
            iou=settings.detection_iou_threshold,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )

        if not results or len(results) == 0:
            return []
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []

                                   
        xyxy = r.boxes.xyxy.cpu().numpy()                          
        cls = r.boxes.cls.cpu().numpy().astype(int)
        conf = r.boxes.conf.cpu().numpy()
                                                                           
        ids = r.boxes.id
        ids_np = ids.cpu().numpy().astype(int) if ids is not None else np.full(len(xyxy), -1)

                                   
        hardhats: list[tuple[float, float, float, float]] = []
        vests: list[tuple[float, float, float, float]] = []
        persons: list[tuple[int, float, tuple[float, float, float, float]]] = []
        for i in range(len(xyxy)):
            box = (float(xyxy[i, 0]), float(xyxy[i, 1]),
                   float(xyxy[i, 2]), float(xyxy[i, 3]))
            c = int(cls[i])
            if c == CLS_PERSON:
                tid = int(ids_np[i])
                if tid < 0:
                                                                                      
                    continue
                persons.append((tid, float(conf[i]), box))
            elif c == CLS_HARDHAT:
                hardhats.append(box)
            elif c == CLS_VEST:
                vests.append(box)
                                                                           
                                                                         

        now = time.monotonic()
        out: list[TrackedWorker] = []
        for tid, person_conf, pbox in persons:
            ppe = _associate_ppe(pbox, hardhats, vests)

                                                                       
            cx = (pbox[0] + pbox[2]) / 2
            cy = (pbox[1] + pbox[3]) / 2
            hist = self._tracks[tid]
            hist.centroids.append((frame_idx, cx, cy))
            hist.last_seen_monotonic = now
            velocity = self._compute_velocity(hist)

                                                                    
            new_state = self._update_state(hist, velocity, now)

                                             
            bbox_norm = BoundingBox(
                x1=max(0.0, min(1.0, pbox[0] / w)),
                y1=max(0.0, min(1.0, pbox[1] / h)),
                x2=max(0.0, min(1.0, pbox[2] / w)),
                y2=max(0.0, min(1.0, pbox[3] / h)),
            )

            out.append(TrackedWorker(
                track_id=tid,
                camera_id=camera_id,
                bbox=bbox_norm,
                confidence=person_conf,
                ppe_status=ppe,
                activity_state=new_state,
                velocity_px_per_frame=velocity,
            ))

                                               
        self._gc_tracks(now, max_age_seconds=30.0)

                                                                 
        self._cache_annotated(frame_bgr, camera_id, out)

        return out

    def _cache_annotated(self, frame_bgr, camera_id, workers):
        """Draw bboxes + labels and store as JPEG bytes."""
        import cv2 as _cv2
        h, w = frame_bgr.shape[:2]
        annotated = frame_bgr.copy()
        ppe_colors = {
            "compliant": (0, 200, 0),
            "head_violation": (0, 165, 255),
            "torso_violation": (0, 165, 255),
            "both_violation": (0, 0, 255),
            "unknown": (180, 180, 180),
        }
        for wo in workers:
            b = wo.bbox
            x1, y1 = int(b.x1 * w), int(b.y1 * h)
            x2, y2 = int(b.x2 * w), int(b.y2 * h)
            ppe = wo.ppe_status.value
            state = wo.activity_state.value
            color = ppe_colors.get(ppe, (180, 180, 180))
            _cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"#{wo.track_id} {state} {ppe}"
            _cv2.putText(annotated, label, (x1, max(20, y1 - 8)),
                         _cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, _cv2.LINE_AA)
        ok, buf = _cv2.imencode(".jpg", annotated, [_cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            self._latest_annotated_jpeg[camera_id] = buf.tobytes()

    def get_latest_jpeg(self, camera_id):
        return self._latest_annotated_jpeg.get(camera_id)

    def _compute_velocity(self, hist: _TrackHistory) -> float:
        """Mean per-frame displacement over the last N entries."""
        n = settings.activity_velocity_window_frames
        recent = list(hist.centroids)[-n:]
        if len(recent) < 2:
            return 0.0
        deltas = []
        for i in range(1, len(recent)):
            f0, x0, y0 = recent[i - 1]
            f1, x1, y1 = recent[i]
            frame_gap = max(1, f1 - f0)
            d = float(np.hypot(x1 - x0, y1 - y0)) / frame_gap
            deltas.append(d)
        return float(np.mean(deltas)) if deltas else 0.0

    def _update_state(
        self,
        hist: _TrackHistory,
        velocity: float,
        now_monotonic: float,
    ) -> ActivityState:
        """Apply state machine. Spec rules from project doc:

        ACTIVE -> TRANSITIONING after 5s of idle signal
        TRANSITIONING -> IDLE after 15s of idle signal
        IDLE -> ACTIVE immediately on active signal
        """
        is_active_signal = velocity >= settings.activity_velocity_threshold_px

        if is_active_signal:
            hist.last_active_signal_monotonic = now_monotonic
            hist.idle_signal_started_monotonic = None
            hist.state = ActivityState.ACTIVE
            return hist.state

                                 
        if hist.idle_signal_started_monotonic is None:
            hist.idle_signal_started_monotonic = now_monotonic

        idle_duration = now_monotonic - hist.idle_signal_started_monotonic

        if hist.state == ActivityState.ACTIVE:
            if idle_duration >= settings.state_active_to_transitioning_sec:
                hist.state = ActivityState.TRANSITIONING
        elif hist.state == ActivityState.TRANSITIONING:
            if idle_duration >= settings.state_transitioning_to_idle_sec:
                hist.state = ActivityState.IDLE
                                                     

        return hist.state

    def _gc_tracks(self, now: float, max_age_seconds: float) -> None:
        stale = [
            tid for tid, h in self._tracks.items()
            if now - h.last_seen_monotonic > max_age_seconds
        ]
        for tid in stale:
            del self._tracks[tid]

    @property
    def active_track_count(self) -> int:
        return len(self._tracks)
