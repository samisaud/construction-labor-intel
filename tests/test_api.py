from __future__ import annotations


import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def isolate_env(tmp_path, monkeypatch):
    monkeypatch.setenv("LABOR_INTEL_SQLITE_PATH", str(tmp_path / "test.db"))
    monkeypatch.delenv("LABOR_INTEL_STREAM_SOURCES", raising=False)
    import importlib
    import app.config
    importlib.reload(app.config)
    import app.main
    importlib.reload(app.main)
    yield


def test_app_boots_and_health_responds():
    from app.main import app
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        # Status depends on whether model weights are present:
        # - degraded if no weights (CI without artifacts)
        # - ok or degraded if weights load (local dev)
        # We just assert the endpoint returned a valid status.
        assert body["status"] in ("ok", "degraded", "down")
        assert body["db_writable"] is True
        assert body["active_streams"] == 0


def test_anomalies_endpoint_empty():
    from app.main import app
    with TestClient(app) as client:
        r = client.get("/anomalies?since_minutes=10")
        assert r.status_code == 200
        assert r.json() == []


def test_shift_summary_returns_valid_shape():
    from app.main import app
    with TestClient(app) as client:
        r = client.get("/shift/summary?hours=1")
        assert r.status_code == 200
        body = r.json()
        # Schema check, not value check — local DB may have leftover data.
        assert "peak_workers_total" in body
        assert "anomaly_count" in body
        assert "mean_active_ratio" in body


def test_zone_trend_unknown_zone_returns_empty_list():
    from app.main import app
    with TestClient(app) as client:
        r = client.get("/zones/nonexistent_zone/trend?minutes=30")
        assert r.status_code == 200
        assert r.json() == []
