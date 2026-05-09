from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import aiosqlite

from app.config import settings
from app.schemas import (
    ActivityState,
    Anomaly,
    PPEStatus,
    ShiftSummary,
    TrackedWorker,
    ZoneStats,
    ZoneTrendPoint,
)

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS labor_counts (
    timestamp TEXT NOT NULL,
    zone_id TEXT NOT NULL,
    camera_id TEXT NOT NULL,
    total INTEGER NOT NULL,
    active INTEGER NOT NULL,
    idle INTEGER NOT NULL,
    transitioning INTEGER NOT NULL,
    ppe_compliant INTEGER NOT NULL,
    ppe_violations INTEGER NOT NULL,
    active_ratio REAL NOT NULL,
    zpi REAL NOT NULL,
    zpi_band TEXT NOT NULL,
    low_confidence INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_lc_zone_ts ON labor_counts(zone_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_lc_ts ON labor_counts(timestamp);

CREATE TABLE IF NOT EXISTS anomalies (
    timestamp TEXT NOT NULL,
    zone_id TEXT NOT NULL,
    camera_id TEXT NOT NULL,
    type TEXT NOT NULL,
    severity TEXT NOT NULL,
    current_value REAL NOT NULL,
    threshold_value REAL NOT NULL,
    duration_seconds INTEGER NOT NULL,
    message TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_an_ts ON anomalies(timestamp);
CREATE INDEX IF NOT EXISTS idx_an_zone_ts ON anomalies(zone_id, timestamp);

CREATE TABLE IF NOT EXISTS worker_tracks (
    timestamp TEXT NOT NULL,
    camera_id TEXT NOT NULL,
    zone_id TEXT,
    track_id INTEGER NOT NULL,
    activity_state TEXT NOT NULL,
    ppe_status TEXT NOT NULL,
    velocity REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_wt_ts ON worker_tracks(timestamp);
CREATE INDEX IF NOT EXISTS idx_wt_track ON worker_tracks(track_id, timestamp);
"""

def _iso(dt: datetime) -> str:
    """ISO-8601 UTC. Sortable as text — that's the whole point of using TEXT timestamps."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()

class TimeSeriesStore:
    """Async wrapper around aiosqlite. Single-writer model: one connection.

    Concurrent reads are fine because we use WAL mode, but writes are serialized
    through the same connection to avoid `database is locked` errors.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        self._path = db_path or settings.resolve(settings.sqlite_path)
        self._conn: aiosqlite.Connection | None = None

    async def open(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self._path)
                                                                    
        await self._conn.execute("PRAGMA journal_mode=WAL;")
        await self._conn.execute("PRAGMA synchronous=NORMAL;")
        await self._conn.executescript(SCHEMA)
        await self._conn.commit()
        logger.info("SQLite store opened at %s", self._path)

        if settings.sqlite_retention_check_on_startup:
            deleted = await self._purge_old_rows(settings.sqlite_retention_hours)
            logger.info("Retention sweep deleted %d rows older than %dh",
                        deleted, settings.sqlite_retention_hours)

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("TimeSeriesStore not opened. Call open() first.")
        return self._conn

    async def _purge_old_rows(self, hours: int) -> int:
        cutoff = _iso(datetime.now(timezone.utc) - timedelta(hours=hours))
        total = 0
        for table in ("labor_counts", "anomalies", "worker_tracks"):
            cur = await self.conn.execute(
                f"DELETE FROM {table} WHERE timestamp < ?", (cutoff,)
            )
            total += cur.rowcount or 0
        await self.conn.commit()
        return total

                                  

    async def write_zone_stats(self, stats: Iterable[ZoneStats]) -> None:
        rows = [
            (
                _iso(s.timestamp), s.zone_id, s.camera_id,
                s.total_workers, s.active_workers, s.idle_workers,
                s.transitioning_workers, s.ppe_compliant_workers,
                s.ppe_violation_workers, s.active_ratio, s.zpi, s.zpi_band,
                int(s.low_confidence),
            )
            for s in stats
        ]
        if not rows:
            return
        await self.conn.executemany(
            """INSERT INTO labor_counts
               (timestamp, zone_id, camera_id, total, active, idle, transitioning,
                ppe_compliant, ppe_violations, active_ratio, zpi, zpi_band, low_confidence)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            rows,
        )
        await self.conn.commit()

    async def write_worker_tracks(self, workers: Iterable[TrackedWorker],
                                  ts: datetime) -> None:
        ts_iso = _iso(ts)
        rows = [
            (ts_iso, w.camera_id, w.zone_id, w.track_id,
             w.activity_state.value if isinstance(w.activity_state, ActivityState)
                 else str(w.activity_state),
             w.ppe_status.value if isinstance(w.ppe_status, PPEStatus)
                 else str(w.ppe_status),
             w.velocity_px_per_frame)
            for w in workers
        ]
        if not rows:
            return
        await self.conn.executemany(
            """INSERT INTO worker_tracks
               (timestamp, camera_id, zone_id, track_id, activity_state, ppe_status, velocity)
               VALUES (?,?,?,?,?,?,?)""",
            rows,
        )
        await self.conn.commit()

    async def write_anomaly(self, a: Anomaly) -> None:
        await self.conn.execute(
            """INSERT INTO anomalies
               (timestamp, zone_id, camera_id, type, severity, current_value,
                threshold_value, duration_seconds, message)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (_iso(a.timestamp), a.zone_id, a.camera_id, a.type.value,
             a.severity.value, a.current_value, a.threshold_value,
             a.duration_seconds, a.message),
        )
        await self.conn.commit()

                         

    async def get_zone_trend(
        self, zone_id: str, minutes: int = 60
    ) -> list[ZoneTrendPoint]:
        cutoff = _iso(datetime.now(timezone.utc) - timedelta(minutes=minutes))
        cur = await self.conn.execute(
            """SELECT timestamp, total, active, zpi
               FROM labor_counts
               WHERE zone_id = ? AND timestamp >= ?
               ORDER BY timestamp ASC""",
            (zone_id, cutoff),
        )
        rows = await cur.fetchall()
        return [
            ZoneTrendPoint(
                timestamp=datetime.fromisoformat(r[0]),
                total_workers=r[1],
                active_workers=r[2],
                zpi=r[3],
            )
            for r in rows
        ]

    async def get_anomalies_since(self, since: datetime) -> list[dict]:
        cur = await self.conn.execute(
            """SELECT timestamp, zone_id, camera_id, type, severity,
                      current_value, threshold_value, duration_seconds, message
               FROM anomalies
               WHERE timestamp >= ?
               ORDER BY timestamp DESC""",
            (_iso(since),),
        )
        rows = await cur.fetchall()
        return [
            {
                "timestamp": r[0], "zone_id": r[1], "camera_id": r[2],
                "type": r[3], "severity": r[4], "current_value": r[5],
                "threshold_value": r[6], "duration_seconds": r[7],
                "message": r[8],
            }
            for r in rows
        ]

    async def get_shift_summary(self, hours: int = 8) -> ShiftSummary:
        end = datetime.now(timezone.utc)
        start = end - timedelta(hours=hours)

        cur = await self.conn.execute(
            """SELECT COALESCE(MAX(total), 0), COALESCE(MIN(total), 0),
                      COALESCE(AVG(active_ratio), 0.0)
               FROM labor_counts WHERE timestamp >= ?""",
            (_iso(start),),
        )
        row = await cur.fetchone()
        peak = int(row[0]) if row else 0
        trough = int(row[1]) if row else 0
        mean_active = float(row[2]) if row else 0.0

        cur = await self.conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(severity = 'high'), 0) "
            "FROM anomalies WHERE timestamp >= ?",
            (_iso(start),),
        )
        row = await cur.fetchone()
        anomaly_count = int(row[0]) if row else 0
        anomaly_high = int(row[1]) if row else 0

        return ShiftSummary(
            start=start, end=end,
            peak_workers_total=peak,
            trough_workers_total=trough,
            mean_active_ratio=mean_active,
            anomaly_count=anomaly_count,
            anomaly_count_high_severity=anomaly_high,
        )

    async def db_writable(self) -> bool:
        """Cheap probe for /health."""
        try:
            await self.conn.execute("SELECT 1")
            return True
        except Exception:
            logger.exception("DB writability check failed")
            return False
