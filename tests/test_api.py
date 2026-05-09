from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

@pytest.fixture(autouse=True)
def isolate_env(tmp_path, monkeypatch):
    """Each API test gets a fresh DB and no stream sources."""
    monkeypatch.setenv("LABOR_INTEL_SQLITE_PATH", str(tmp_path / "test.db"))
    monkeypatch.delenv("LABOR_INTEL_STREAM_SOURCES", raising=False)
                                                      
    import importlib
    import app.config
    importlib.reload(app.config)
    import app.main
    importlib.reload(app.main)
    yield

def test_app_boots_in_degraded_mode():
    from app.main import app
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
                                                                            
        assert body["status"] == "degraded"
        assert body["db_writable"] is True
        assert body["active_streams"] == 0

def test_anomalies_endpoint_empty():
    from app.main import app
    with TestClient(app) as client:
        r = client.get("/anomalies?since_minutes=10")
        assert r.status_code == 200
        assert r.json() == []

def test_shift_summary_empty():
    from app.main import app
    with TestClient(app) as client:
        r = client.get("/shift/summary?hours=1")
        assert r.status_code == 200
        body = r.json()
        assert body["peak_workers_total"] == 0
        assert body["anomaly_count"] == 0

def test_zone_trend_unknown_zone_returns_empty_list():
    from app.main import app
    with TestClient(app) as client:
        r = client.get("/zones/nonexistent_zone/trend?minutes=30")
        assert r.status_code == 200
        assert r.json() == []
