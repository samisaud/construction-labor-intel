from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

                                                                       

class ActivityState(str, Enum):
    ACTIVE = "active"
    IDLE = "idle"
    TRANSITIONING = "transitioning"

class PPEStatus(str, Enum):
    COMPLIANT = "compliant"                                        
    HEAD_VIOLATION = "head_violation"                    
    TORSO_VIOLATION = "torso_violation"               
    BOTH_VIOLATION = "both_violation"                 
    UNKNOWN = "unknown"                                        

class AnomalySeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class AnomalyType(str, Enum):
    ZONE_UNDERSTAFFED = "zone_understaffed"
    IDLE_SPIKE = "idle_spike"
    PRODUCTIVITY_COLLAPSE = "productivity_collapse"
    PPE_VIOLATION_CLUSTER = "ppe_violation_cluster"

                                

class BoundingBox(BaseModel):
    """Axis-aligned bounding box in NORMALIZED image coords (0.0-1.0).

    Normalized coords mean zone polygons (also normalized) compare directly
    without per-camera resolution bookkeeping.
    """
    model_config = ConfigDict(frozen=True)

    x1: float = Field(ge=0.0, le=1.0)
    y1: float = Field(ge=0.0, le=1.0)
    x2: float = Field(ge=0.0, le=1.0)
    y2: float = Field(ge=0.0, le=1.0)

    @property
    def centroid(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

                                                             

class TrackedWorker(BaseModel):
    """One tracked worker as seen in one analytics tick.

    Produced by inference_engine, consumed by analytics_engine.
    """
    track_id: int
    camera_id: str
    bbox: BoundingBox
    confidence: float = Field(ge=0.0, le=1.0)
    ppe_status: PPEStatus
    activity_state: ActivityState
    velocity_px_per_frame: float
    zone_id: str | None = None                                                  

                                         

class ZoneStats(BaseModel):
    """Per-zone analytics computed each tick."""
    zone_id: str
    camera_id: str
    timestamp: datetime

            
    total_workers: int
    active_workers: int
    idle_workers: int
    transitioning_workers: int
    ppe_compliant_workers: int
    ppe_violation_workers: int

            
    active_ratio: float = Field(ge=0.0, le=1.0)
    idle_ratio: float = Field(ge=0.0, le=1.0)

                  
    expected_workers: int
    expected_active: int
    zpi: float = Field(description="Zone Productivity Index = actual_active / expected_active")
    zpi_band: Literal["green", "amber", "red", "unknown"]

                                               
    low_confidence: bool

class Anomaly(BaseModel):
    """Anomaly event. Persisted to SQLite and pushed via WebSocket."""
    timestamp: datetime
    zone_id: str
    camera_id: str
    type: AnomalyType
    severity: AnomalySeverity
    current_value: float
    threshold_value: float
    duration_seconds: int
    message: str

                                          

class WSMessageType(str, Enum):
    SNAPSHOT = "snapshot"                                 
    ANOMALY = "anomaly"                                       
    HEALTH = "health"                                    

class WSEnvelope(BaseModel):
    """Outer envelope for every WebSocket message.

    The dashboard switches on `type` to route the payload.
    """
    model_config = ConfigDict(use_enum_values=True)

    type: WSMessageType
    timestamp: datetime
    payload: dict

class SnapshotPayload(BaseModel):
    """Sent every analytics_tick_seconds. Full picture across all cameras."""
    zone_stats: list[ZoneStats]
    tracked_workers: list[TrackedWorker]
    active_camera_count: int

                                            

class ZoneTrendPoint(BaseModel):
    timestamp: datetime
    total_workers: int
    active_workers: int
    zpi: float

class ShiftSummary(BaseModel):
    start: datetime
    end: datetime
    peak_workers_total: int
    trough_workers_total: int
    mean_active_ratio: float
    anomaly_count: int
    anomaly_count_high_severity: int

class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "down"]
    version: str
    active_streams: int
    inference_provider: str
    db_writable: bool
