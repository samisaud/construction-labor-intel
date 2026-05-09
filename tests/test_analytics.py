from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from app.analytics_engine import AnalyticsEngine
from app.schemas import (
    ActivityState,
    BoundingBox,
    PPEStatus,
    TrackedWorker,
)

@pytest.fixture
def zones_yaml(tmp_path: Path) -> Path:
    config = {
        "phase": "test",
        "cameras": {
            "cam_test": {
                "label": "test",
                "zones": [
                    {
                        "id": "left_half",
                        "polygon": [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]],
                        "expected": {"workers": 4, "active": 3},
                        "critical_path": True,
                    },
                    {
                        "id": "right_half",
                        "polygon": [[0.5, 0.0], [1.0, 0.0], [1.0, 1.0], [0.5, 1.0]],
                        "expected": {"workers": 2, "active": 2},
                        "critical_path": False,
                    },
                ],
            }
        },
    }
    p = tmp_path / "zones.yaml"
    p.write_text(yaml.safe_dump(config))
    return p

def _worker(track_id: int, x_center: float, state: ActivityState,
            ppe: PPEStatus = PPEStatus.COMPLIANT) -> TrackedWorker:
    """Build a worker with bbox centered at (x_center, 0.5)."""
    half = 0.05
    return TrackedWorker(
        track_id=track_id,
        camera_id="cam_test",
        bbox=BoundingBox(
            x1=max(0.0, x_center - half),
            y1=0.45,
            x2=min(1.0, x_center + half),
            y2=0.55,
        ),
        confidence=0.9,
        ppe_status=ppe,
        activity_state=state,
        velocity_px_per_frame=3.0 if state == ActivityState.ACTIVE else 0.5,
    )

class TestZoneAssignment:
    def test_left_workers_land_in_left_zone(self, zones_yaml: Path):
        eng = AnalyticsEngine(zones_yaml_path=zones_yaml)
        workers = [_worker(1, 0.2, ActivityState.ACTIVE)]
        stats, tagged = eng.compute_zone_stats(workers)
        assert tagged[0].zone_id == "left_half"

    def test_right_workers_land_in_right_zone(self, zones_yaml: Path):
        eng = AnalyticsEngine(zones_yaml_path=zones_yaml)
        workers = [_worker(1, 0.8, ActivityState.ACTIVE)]
        _, tagged = eng.compute_zone_stats(workers)
        assert tagged[0].zone_id == "right_half"

class TestZPI:
    def test_zpi_green_when_meeting_target(self, zones_yaml: Path):
        eng = AnalyticsEngine(zones_yaml_path=zones_yaml)
                                                
        workers = [
            _worker(i, 0.2, ActivityState.ACTIVE) for i in range(3)
        ]
        stats, _ = eng.compute_zone_stats(workers)
        left = next(s for s in stats if s.zone_id == "left_half")
        assert left.active_workers == 3
        assert left.zpi == pytest.approx(1.0)
        assert left.zpi_band == "green"

    def test_zpi_amber_at_70_percent(self, zones_yaml: Path):
        eng = AnalyticsEngine(zones_yaml_path=zones_yaml)
                                                                  
                                                                                  
        workers = [_worker(i, 0.2, ActivityState.ACTIVE) for i in range(2)]
        stats, _ = eng.compute_zone_stats(workers)
        left = next(s for s in stats if s.zone_id == "left_half")
                                    
        assert left.zpi_band == "red"

                                                                    
                                                              

    def test_zpi_red_when_below_half(self, zones_yaml: Path):
        eng = AnalyticsEngine(zones_yaml_path=zones_yaml)
        workers = [_worker(1, 0.2, ActivityState.IDLE)]                    
        stats, _ = eng.compute_zone_stats(workers)
        left = next(s for s in stats if s.zone_id == "left_half")
        assert left.zpi == 0.0
        assert left.zpi_band == "red"

    def test_low_confidence_flag_for_small_zone(self, zones_yaml: Path):
        eng = AnalyticsEngine(zones_yaml_path=zones_yaml)
        workers = [_worker(1, 0.2, ActivityState.ACTIVE)]
        stats, _ = eng.compute_zone_stats(workers)
        left = next(s for s in stats if s.zone_id == "left_half")
                                         
        assert left.low_confidence is True

class TestPPECounts:
    def test_violation_counts(self, zones_yaml: Path):
        eng = AnalyticsEngine(zones_yaml_path=zones_yaml)
        workers = [
            _worker(1, 0.2, ActivityState.ACTIVE, PPEStatus.COMPLIANT),
            _worker(2, 0.2, ActivityState.ACTIVE, PPEStatus.HEAD_VIOLATION),
            _worker(3, 0.2, ActivityState.ACTIVE, PPEStatus.BOTH_VIOLATION),
        ]
        stats, _ = eng.compute_zone_stats(workers)
        left = next(s for s in stats if s.zone_id == "left_half")
        assert left.ppe_compliant_workers == 1
        assert left.ppe_violation_workers == 2
