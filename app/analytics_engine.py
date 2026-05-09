from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import yaml
from shapely.geometry import Point, Polygon                                

from app.config import settings
from app.schemas import ActivityState, PPEStatus, TrackedWorker, ZoneStats

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class ZoneDef:
    zone_id: str
    camera_id: str
    polygon: Polygon
    expected_workers: int
    expected_active: int
    critical_path: bool

def _load_zones(path: Path) -> list[ZoneDef]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    zones: list[ZoneDef] = []
    for cam_id, cam_cfg in (cfg.get("cameras") or {}).items():
        for z in cam_cfg.get("zones", []):
            poly_pts = z["polygon"]
            if len(poly_pts) < 3:
                logger.warning("Zone %s has < 3 vertices, skipping", z["id"])
                continue
            zones.append(ZoneDef(
                zone_id=z["id"],
                camera_id=cam_id,
                polygon=Polygon(poly_pts),
                expected_workers=int(z["expected"]["workers"]),
                expected_active=int(z["expected"]["active"]),
                critical_path=bool(z.get("critical_path", False)),
            ))
    if not zones:
        raise ValueError(f"No zones loaded from {path}")
    logger.info("Loaded %d zones from %s", len(zones), path)
    return zones

def _zpi_band(zpi: float, expected_active: int) -> str:
    if expected_active <= 0:
        return "unknown"
    if zpi >= 1.0:
        return "green"
    if zpi >= 0.7:
        return "amber"
    return "red"

class AnalyticsEngine:
    """Stateless w.r.t. workers — the inference engine owns track state.

    This class only maps workers to zones and aggregates per tick.
    """

    def __init__(self, zones_yaml_path: Path | None = None) -> None:
        path = zones_yaml_path or settings.resolve(settings.zones_config_path)
        self.zones = _load_zones(path)
                                                  
        self._zones_by_camera: dict[str, list[ZoneDef]] = {}
        for z in self.zones:
            self._zones_by_camera.setdefault(z.camera_id, []).append(z)

    def assign_zone(self, worker: TrackedWorker) -> str | None:
        """Find which zone this worker's bbox centroid sits in.

        Returns zone_id or None if no zone matches (out of frame, or worker on a
        camera with no defined zones).
        """
        zones = self._zones_by_camera.get(worker.camera_id, [])
        if not zones:
            return None
        cx, cy = worker.bbox.centroid
        pt = Point(cx, cy)
        for z in zones:
            if z.polygon.covers(pt):                                                  
                return z.zone_id
        return None

    def compute_zone_stats(
        self,
        workers: Iterable[TrackedWorker],
        now: datetime | None = None,
    ) -> tuple[list[ZoneStats], list[TrackedWorker]]:
        """Aggregate per-zone stats. Returns (stats, workers_with_zone_assigned).

        Workers whose centroid falls outside all zones get zone_id=None and are
        excluded from ZoneStats but still returned for the dashboard overlay.
        """
        ts = now or datetime.now(timezone.utc)

                                                                           
        bucket: dict[str, list[TrackedWorker]] = {z.zone_id: [] for z in self.zones}
        tagged: list[TrackedWorker] = []
        for w in workers:
            zid = self.assign_zone(w)
            tagged_worker = w.model_copy(update={"zone_id": zid})
            tagged.append(tagged_worker)
            if zid is not None:
                bucket[zid].append(tagged_worker)

        zone_index = {z.zone_id: z for z in self.zones}
        out: list[ZoneStats] = []
        for zid, zone_workers in bucket.items():
            zdef = zone_index[zid]
            total = len(zone_workers)
            active = sum(1 for w in zone_workers if w.activity_state == ActivityState.ACTIVE)
            idle = sum(1 for w in zone_workers if w.activity_state == ActivityState.IDLE)
            transitioning = sum(
                1 for w in zone_workers if w.activity_state == ActivityState.TRANSITIONING
            )
            compliant = sum(
                1 for w in zone_workers if w.ppe_status == PPEStatus.COMPLIANT
            )
            violations = sum(
                1 for w in zone_workers if w.ppe_status in (
                    PPEStatus.HEAD_VIOLATION,
                    PPEStatus.TORSO_VIOLATION,
                    PPEStatus.BOTH_VIOLATION,
                )
            )

            active_ratio = (active / total) if total > 0 else 0.0
            idle_ratio = (idle / total) if total > 0 else 0.0

            if zdef.expected_active > 0:
                zpi = active / zdef.expected_active
            else:
                zpi = 0.0

            out.append(ZoneStats(
                zone_id=zid,
                camera_id=zdef.camera_id,
                timestamp=ts,
                total_workers=total,
                active_workers=active,
                idle_workers=idle,
                transitioning_workers=transitioning,
                ppe_compliant_workers=compliant,
                ppe_violation_workers=violations,
                active_ratio=active_ratio,
                idle_ratio=idle_ratio,
                expected_workers=zdef.expected_workers,
                expected_active=zdef.expected_active,
                zpi=zpi,
                zpi_band=_zpi_band(zpi, zdef.expected_active),                          
                low_confidence=total < settings.analytics_low_confidence_min_workers,
            ))

        return out, tagged
