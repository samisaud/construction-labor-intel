from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import pytest_asyncio                                                                                

from app.schemas import (
    ActivityState,
    Anomaly,
    AnomalySeverity,
    AnomalyType,
    BoundingBox,
    PPEStatus,
    TrackedWorker,
    ZoneStats,
)
from app.time_series_store import TimeSeriesStore

def _zone_stats(zone_id: str = "z1", ts: datetime | None = None) -> ZoneStats:
    return ZoneStats(
        zone_id=zone_id,
        camera_id="cam_test",
        timestamp=ts or datetime.now(timezone.utc),
        total_workers=5,
        active_workers=3,
        idle_workers=2,
        transitioning_workers=0,
        ppe_compliant_workers=4,
        ppe_violation_workers=1,
        active_ratio=0.6,
        idle_ratio=0.4,
        expected_workers=5,
        expected_active=4,
        zpi=0.75,
        zpi_band="amber",
        low_confidence=False,
    )

def _worker(track_id: int = 1, zone_id: str | None = "z1") -> TrackedWorker:
    return TrackedWorker(
        track_id=track_id,
        camera_id="cam_test",
        bbox=BoundingBox(x1=0.1, y1=0.1, x2=0.2, y2=0.5),
        confidence=0.9,
        ppe_status=PPEStatus.COMPLIANT,
        activity_state=ActivityState.ACTIVE,
        velocity_px_per_frame=3.5,
        zone_id=zone_id,
    )

@pytest.mark.asyncio
async def test_open_close_creates_schema(tmp_path: Path):
    db = tmp_path / "test.db"
    store = TimeSeriesStore(db_path=db)
    await store.open()
    assert db.exists()
    await store.close()

@pytest.mark.asyncio
async def test_write_and_query_zone_trend(tmp_path: Path):
    store = TimeSeriesStore(db_path=tmp_path / "test.db")
    await store.open()
    try:
        now = datetime.now(timezone.utc)
        stats = [
            _zone_stats(ts=now - timedelta(minutes=i))
            for i in range(5)
        ]
        await store.write_zone_stats(stats)

        trend = await store.get_zone_trend("z1", minutes=10)
        assert len(trend) == 5
                                        
        assert trend[0].timestamp < trend[-1].timestamp
        assert all(p.zpi == pytest.approx(0.75) for p in trend)
    finally:
        await store.close()

@pytest.mark.asyncio
async def test_retention_purges_old_rows(tmp_path: Path):
    store = TimeSeriesStore(db_path=tmp_path / "test.db")
    await store.open()
    try:
        old = datetime.now(timezone.utc) - timedelta(hours=48)
        new = datetime.now(timezone.utc)
        await store.write_zone_stats([_zone_stats(ts=old)])
        await store.write_zone_stats([_zone_stats(ts=new)])

                                                    
        deleted = await store._purge_old_rows(hours=24)
        assert deleted == 1

        trend = await store.get_zone_trend("z1", minutes=24 * 60)
        assert len(trend) == 1
    finally:
        await store.close()

@pytest.mark.asyncio
async def test_anomaly_round_trip(tmp_path: Path):
    store = TimeSeriesStore(db_path=tmp_path / "test.db")
    await store.open()
    try:
        a = Anomaly(
            timestamp=datetime.now(timezone.utc),
            zone_id="z1",
            camera_id="cam_test",
            type=AnomalyType.PPE_VIOLATION_CLUSTER,
            severity=AnomalySeverity.HIGH,
            current_value=4.0,
            threshold_value=3.0,
            duration_seconds=0,
            message="test",
        )
        await store.write_anomaly(a)
        rows = await store.get_anomalies_since(
            datetime.now(timezone.utc) - timedelta(minutes=5)
        )
        assert len(rows) == 1
        assert rows[0]["type"] == "ppe_violation_cluster"
        assert rows[0]["severity"] == "high"
    finally:
        await store.close()

@pytest.mark.asyncio
async def test_worker_tracks_persisted(tmp_path: Path):
    store = TimeSeriesStore(db_path=tmp_path / "test.db")
    await store.open()
    try:
        ts = datetime.now(timezone.utc)
        await store.write_worker_tracks([_worker(1), _worker(2)], ts=ts)
                                   
        cur = await store.conn.execute("SELECT COUNT(*) FROM worker_tracks")
        row = await cur.fetchone()
        assert row[0] == 2
    finally:
        await store.close()

@pytest.mark.asyncio
async def test_shift_summary(tmp_path: Path):
    store = TimeSeriesStore(db_path=tmp_path / "test.db")
    await store.open()
    try:
        now = datetime.now(timezone.utc)
        stats = []
        for i in range(10):
            s = _zone_stats(ts=now - timedelta(minutes=i))
                                                        
            stats.append(s.model_copy(update={"total_workers": 5 + i,
                                              "active_ratio": 0.5}))
        await store.write_zone_stats(stats)

        summary = await store.get_shift_summary(hours=1)
        assert summary.peak_workers_total == 14
        assert summary.trough_workers_total == 5
        assert summary.mean_active_ratio == pytest.approx(0.5)
    finally:
        await store.close()
