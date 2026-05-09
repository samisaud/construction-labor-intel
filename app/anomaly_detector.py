from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable

from app.schemas import (
    Anomaly,
    AnomalySeverity,
    AnomalyType,
    ZoneStats,
)

logger = logging.getLogger(__name__)

                                                                     

UNDERSTAFFING_RATIO_THRESHOLD = 0.5
UNDERSTAFFING_PERSISTENCE_SEC = 5 * 60

IDLE_SPIKE_DROP_THRESHOLD = 0.30                        
IDLE_SPIKE_BASELINE_WINDOW_MIN = 10

ZPI_COLLAPSE_THRESHOLD = 0.5
ZPI_COLLAPSE_PERSISTENCE_SEC = 10 * 60

PPE_CLUSTER_THRESHOLD = 3                   

@dataclass
class _ZoneRollingState:
    """Per-zone state for persistence tracking and rolling baselines."""
                                                                           
    active_ratio_hist: deque[tuple[datetime, float]] = field(
        default_factory=lambda: deque(maxlen=300)                      
    )
                                                                     
    understaffed_since: datetime | None = None
    zpi_collapse_since: datetime | None = None

                                                                       
    understaffed_fired: bool = False
    zpi_collapse_fired: bool = False
    idle_spike_fired: bool = False
    ppe_cluster_fired: bool = False

                                                                
    critical_path: bool = False

def _trim_old(hist: deque[tuple[datetime, float]], cutoff: datetime) -> None:
    while hist and hist[0][0] < cutoff:
        hist.popleft()

def _baseline_active_ratio(
    hist: Iterable[tuple[datetime, float]], baseline_minutes: int, now: datetime
) -> float | None:
    cutoff = now.timestamp() - baseline_minutes * 60
    samples = [r for ts, r in hist if ts.timestamp() >= cutoff]
    if len(samples) < 5:                                                 
        return None
    return sum(samples) / len(samples)

class AnomalyDetector:
    def __init__(self, critical_path_zones: set[str] | None = None) -> None:
                                                                               
                                           
        self._critical: set[str] = critical_path_zones or set()
        self._state: dict[str, _ZoneRollingState] = defaultdict(_ZoneRollingState)

    def evaluate(self, zone_stats: Iterable[ZoneStats]) -> list[Anomaly]:
        """Run all four detectors against the latest tick. Returns new anomalies."""
        emitted: list[Anomaly] = []
        for s in zone_stats:
            st = self._state[s.zone_id]
            st.critical_path = s.zone_id in self._critical
            now = s.timestamp

                                          
            understaffed = (
                s.expected_workers > 0
                and s.total_workers < UNDERSTAFFING_RATIO_THRESHOLD * s.expected_workers
            )
            if understaffed:
                if st.understaffed_since is None:
                    st.understaffed_since = now
                duration = (now - st.understaffed_since).total_seconds()
                if duration >= UNDERSTAFFING_PERSISTENCE_SEC and not st.understaffed_fired:
                    severity = AnomalySeverity.HIGH if st.critical_path else AnomalySeverity.MEDIUM
                    emitted.append(Anomaly(
                        timestamp=now,
                        zone_id=s.zone_id,
                        camera_id=s.camera_id,
                        type=AnomalyType.ZONE_UNDERSTAFFED,
                        severity=severity,
                        current_value=float(s.total_workers),
                        threshold_value=UNDERSTAFFING_RATIO_THRESHOLD * s.expected_workers,
                        duration_seconds=int(duration),
                        message=(
                            f"Zone {s.zone_id}: {s.total_workers} workers vs "
                            f"expected {s.expected_workers} for {int(duration / 60)}m"
                        ),
                    ))
                    st.understaffed_fired = True
            else:
                st.understaffed_since = None
                st.understaffed_fired = False

                                   
                                                                                
            baseline = _baseline_active_ratio(
                st.active_ratio_hist, IDLE_SPIKE_BASELINE_WINDOW_MIN, now
            )
            spike = (
                baseline is not None
                and (baseline - s.active_ratio) >= IDLE_SPIKE_DROP_THRESHOLD
            )
            if spike and not st.idle_spike_fired:
                emitted.append(Anomaly(
                    timestamp=now,
                    zone_id=s.zone_id,
                    camera_id=s.camera_id,
                    type=AnomalyType.IDLE_SPIKE,
                    severity=AnomalySeverity.MEDIUM,
                    current_value=s.active_ratio,
                    threshold_value=baseline - IDLE_SPIKE_DROP_THRESHOLD,                          
                    duration_seconds=0,                           
                    message=(
                        f"Zone {s.zone_id}: active_ratio dropped from "
                        f"{baseline:.2f} to {s.active_ratio:.2f}"                            
                    ),
                ))
                st.idle_spike_fired = True
            elif not spike:
                st.idle_spike_fired = False

                                                                          
            st.active_ratio_hist.append((now, s.active_ratio))
            _trim_old(
                st.active_ratio_hist,
                cutoff=datetime.fromtimestamp(
                    now.timestamp() - IDLE_SPIKE_BASELINE_WINDOW_MIN * 60 - 60,
                    tz=timezone.utc,
                ),
            )

                                              
            collapse = s.zpi_band != "unknown" and s.zpi < ZPI_COLLAPSE_THRESHOLD
            if collapse:
                if st.zpi_collapse_since is None:
                    st.zpi_collapse_since = now
                duration = (now - st.zpi_collapse_since).total_seconds()
                if duration >= ZPI_COLLAPSE_PERSISTENCE_SEC and not st.zpi_collapse_fired:
                    emitted.append(Anomaly(
                        timestamp=now,
                        zone_id=s.zone_id,
                        camera_id=s.camera_id,
                        type=AnomalyType.PRODUCTIVITY_COLLAPSE,
                        severity=AnomalySeverity.HIGH,
                        current_value=s.zpi,
                        threshold_value=ZPI_COLLAPSE_THRESHOLD,
                        duration_seconds=int(duration),
                        message=(
                            f"Zone {s.zone_id}: ZPI {s.zpi:.2f} below 0.5 "
                            f"for {int(duration / 60)}m"
                        ),
                    ))
                    st.zpi_collapse_fired = True
            else:
                st.zpi_collapse_since = None
                st.zpi_collapse_fired = False

                                              
            cluster = s.ppe_violation_workers > PPE_CLUSTER_THRESHOLD
            if cluster and not st.ppe_cluster_fired:
                emitted.append(Anomaly(
                    timestamp=now,
                    zone_id=s.zone_id,
                    camera_id=s.camera_id,
                    type=AnomalyType.PPE_VIOLATION_CLUSTER,
                    severity=AnomalySeverity.HIGH,
                    current_value=float(s.ppe_violation_workers),
                    threshold_value=float(PPE_CLUSTER_THRESHOLD),
                    duration_seconds=0,
                    message=(
                        f"Zone {s.zone_id}: {s.ppe_violation_workers} PPE violations detected"
                    ),
                ))
                st.ppe_cluster_fired = True
            elif not cluster:
                st.ppe_cluster_fired = False

        return emitted
