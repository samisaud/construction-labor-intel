from __future__ import annotations

from datetime import datetime, timedelta, timezone


from app.anomaly_detector import (
    AnomalyDetector,
    UNDERSTAFFING_PERSISTENCE_SEC,
    ZPI_COLLAPSE_PERSISTENCE_SEC,
)
from app.schemas import AnomalySeverity, AnomalyType, ZoneStats

def _stats(
    zone_id: str = "z1",
    ts: datetime | None = None,
    total: int = 4,
    active: int = 3,
    ppe_violations: int = 0,
    expected_workers: int = 4,
    expected_active: int = 3,
    zpi: float | None = None,
    active_ratio: float | None = None,
) -> ZoneStats:
    if ts is None:
        ts = datetime.now(timezone.utc)
    if zpi is None:
        zpi = active / expected_active if expected_active else 0.0
    if active_ratio is None:
        active_ratio = active / total if total else 0.0
    band = "green" if zpi >= 1.0 else ("amber" if zpi >= 0.7 else "red")
    if expected_active == 0:
        band = "unknown"
    return ZoneStats(
        zone_id=zone_id,
        camera_id="cam_test",
        timestamp=ts,
        total_workers=total,
        active_workers=active,
        idle_workers=total - active,
        transitioning_workers=0,
        ppe_compliant_workers=total - ppe_violations,
        ppe_violation_workers=ppe_violations,
        active_ratio=active_ratio,
        idle_ratio=1.0 - active_ratio,
        expected_workers=expected_workers,
        expected_active=expected_active,
        zpi=zpi,
        zpi_band=band,
        low_confidence=total < 3,
    )

class TestPPECluster:
    def test_fires_on_4_violations(self):
        det = AnomalyDetector()
        stats = _stats(total=10, active=8, ppe_violations=4)
        out = det.evaluate([stats])
        assert len(out) == 1
        assert out[0].type == AnomalyType.PPE_VIOLATION_CLUSTER
        assert out[0].severity == AnomalySeverity.HIGH

    def test_does_not_fire_on_3(self):
        det = AnomalyDetector()
        stats = _stats(total=10, active=8, ppe_violations=3)
        out = det.evaluate([stats])
        assert out == []

    def test_does_not_double_fire(self):
        det = AnomalyDetector()
                                                               
        s1 = _stats(total=10, active=8, ppe_violations=4)
        s2 = _stats(total=10, active=8, ppe_violations=4,
                    ts=s1.timestamp + timedelta(seconds=3))
        assert len(det.evaluate([s1])) == 1
        assert len(det.evaluate([s2])) == 0

    def test_re_fires_after_clearing(self):
        det = AnomalyDetector()
        s1 = _stats(total=10, active=8, ppe_violations=4)
        s2 = _stats(total=10, active=8, ppe_violations=2,
                    ts=s1.timestamp + timedelta(seconds=3))
        s3 = _stats(total=10, active=8, ppe_violations=4,
                    ts=s1.timestamp + timedelta(seconds=6))
        assert len(det.evaluate([s1])) == 1
        assert len(det.evaluate([s2])) == 0
        assert len(det.evaluate([s3])) == 1

class TestUnderstaffing:
    def test_no_fire_before_persistence(self):
        det = AnomalyDetector()
                                                                   
        out = det.evaluate([_stats(total=1, active=0, expected_workers=4)])
        assert out == []

    def test_fires_after_persistence(self):
        det = AnomalyDetector()
        t0 = datetime.now(timezone.utc)
                                   
        det.evaluate([_stats(total=1, active=0, expected_workers=4, ts=t0)])
                                                          
        out = det.evaluate([
            _stats(total=1, active=0, expected_workers=4,
                   ts=t0 + timedelta(seconds=UNDERSTAFFING_PERSISTENCE_SEC + 1))
        ])
        assert len(out) == 1
        assert out[0].type == AnomalyType.ZONE_UNDERSTAFFED

    def test_high_severity_for_critical_path(self):
        det = AnomalyDetector(critical_path_zones={"z1"})
        t0 = datetime.now(timezone.utc)
        det.evaluate([_stats(total=1, active=0, expected_workers=4, ts=t0)])
        out = det.evaluate([
            _stats(total=1, active=0, expected_workers=4,
                   ts=t0 + timedelta(seconds=UNDERSTAFFING_PERSISTENCE_SEC + 1))
        ])
        assert out[0].severity == AnomalySeverity.HIGH

    def test_medium_severity_for_non_critical(self):
        det = AnomalyDetector()                     
        t0 = datetime.now(timezone.utc)
        det.evaluate([_stats(total=1, active=0, expected_workers=4, ts=t0)])
        out = det.evaluate([
            _stats(total=1, active=0, expected_workers=4,
                   ts=t0 + timedelta(seconds=UNDERSTAFFING_PERSISTENCE_SEC + 1))
        ])
        assert out[0].severity == AnomalySeverity.MEDIUM

class TestProductivityCollapse:
    def test_fires_after_persistence(self):
        det = AnomalyDetector()
        t0 = datetime.now(timezone.utc)
                                 
        det.evaluate([_stats(total=10, active=2, expected_active=8, zpi=0.25, ts=t0)])
        out = det.evaluate([
            _stats(total=10, active=2, expected_active=8, zpi=0.25,
                   ts=t0 + timedelta(seconds=ZPI_COLLAPSE_PERSISTENCE_SEC + 1))
        ])
        assert len(out) == 1
        assert out[0].type == AnomalyType.PRODUCTIVITY_COLLAPSE
        assert out[0].severity == AnomalySeverity.HIGH

class TestIdleSpike:
    def test_fires_on_30pp_drop(self):
        det = AnomalyDetector()
        t0 = datetime.now(timezone.utc)
                                                           
                                                    
        baseline_samples = []
        for i in range(8):
            baseline_samples.append(_stats(
                total=10, active=9, active_ratio=0.9,
                ts=t0 + timedelta(seconds=i * 30),
            ))
            det.evaluate([baseline_samples[-1]])
                                        
        out = det.evaluate([_stats(
            total=10, active=5, active_ratio=0.5,
            ts=t0 + timedelta(seconds=300),
        )])
        assert any(a.type == AnomalyType.IDLE_SPIKE for a in out)

    def test_no_fire_without_baseline(self):
        det = AnomalyDetector()
        t0 = datetime.now(timezone.utc)
        out = det.evaluate([_stats(total=10, active=2, active_ratio=0.2, ts=t0)])
                                                                     
        assert all(a.type != AnomalyType.IDLE_SPIKE for a in out)
