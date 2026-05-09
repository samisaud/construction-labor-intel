"""Streamlit dashboard.

Pulls live snapshots from the FastAPI backend over WebSocket and renders:
    - 4-camera grid with bbox overlay (placeholders if no live frames available)
    - Per-zone stat cards (count, active ratio, ZPI gauge)
    - Anomaly alert panel
    - 1-hour trend charts per zone

Why Streamlit + WebSocket: Streamlit doesn't natively do WebSockets, so we run
a tiny background thread that maintains the WS connection and stuffs messages
into st.session_state. The auto-refresh loop reads from session_state.

Limitation: Streamlit reruns top-to-bottom on every interaction. For a real
production dashboard you'd use a JS framework. This is fine for the v1 demo.
"""
from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from datetime import datetime
from typing import Any

import pandas as pd
import requests
import streamlit as st
import websocket  # websocket-client


# ---------- Config ----------

API_HOST = os.environ.get("LABOR_INTEL_API_HOST", "localhost")
API_PORT = int(os.environ.get("LABOR_INTEL_API_PORT", "8000"))
HTTP_BASE = f"http://{API_HOST}:{API_PORT}"
WS_URL = f"ws://{API_HOST}:{API_PORT}/ws/live"


st.set_page_config(
    page_title="Construction Labor Intel",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------- WebSocket background thread ----------

def _ensure_ws_thread():
    """Start one persistent WS reader thread per session."""
    if "ws_started" in st.session_state:
        return

    st.session_state.ws_started = True
    st.session_state.latest_snapshot = None
    st.session_state.recent_anomalies = deque(maxlen=50)
    st.session_state.ws_connected = False

    snapshot_box = st.session_state
    anomaly_box = st.session_state.recent_anomalies

    def _run():
        while True:
            try:
                snapshot_box.ws_connected = False
                ws = websocket.create_connection(WS_URL, timeout=10)
                snapshot_box.ws_connected = True
                while True:
                    msg = ws.recv()
                    if not msg:
                        break
                    env = json.loads(msg)
                    if env.get("type") == "snapshot":
                        snapshot_box.latest_snapshot = env
                    elif env.get("type") == "anomaly":
                        anomaly_box.appendleft(env)
            except Exception as e:
                import traceback
                print(f'WS thread error: {type(e).__name__}: {e}')
                traceback.print_exc()
                snapshot_box.ws_connected = False
                time.sleep(2)

    t = threading.Thread(target=_run, daemon=True, name="dashboard_ws")
    try:
        from streamlit.runtime.scriptrunner import add_script_run_ctx
        add_script_run_ctx(t)
    except Exception:
        pass
    t.start()


_ensure_ws_thread()


# ---------- Sidebar ----------

with st.sidebar:
    st.title("Labor Intel")
    st.caption(f"API: `{HTTP_BASE}`")

    try:
        h = requests.get(f"{HTTP_BASE}/health", timeout=2).json()
        status = h.get("status", "unknown")
        color = {"ok": "🟢", "degraded": "🟡", "down": "🔴"}.get(status, "⚪")
        st.markdown(f"**Backend:** {color} {status}")
        st.caption(f"Streams alive: {h.get('active_streams', 0)}")
        st.caption(f"DB writable: {h.get('db_writable', False)}")
    except Exception as e:
        st.markdown("**Backend:** 🔴 unreachable")
        st.caption(str(e))

    ws_color = "🟢" if st.session_state.get("ws_connected") else "🔴"
    st.markdown(f"**WebSocket:** {ws_color}")

    st.divider()
    st.caption("Auto-refresh every 3s")
    refresh_rate = st.slider("Refresh (s)", 1, 10, 3)

    st.divider()
    selected_zone = st.text_input("Trend zone_id", value="framing_north")
    trend_minutes = st.slider("Trend window (min)", 10, 120, 60)


# ---------- Main grid ----------

st.title("Site Labor Dashboard")

snap = st.session_state.get("latest_snapshot")
if not snap:
    st.info("Waiting for first snapshot from backend... (WS connected: " + 
            str(st.session_state.get("ws_connected", False)) + ")")
    time.sleep(1)
    st.rerun()

payload = snap.get("payload", {})
zone_stats: list[dict] = payload.get("zone_stats", [])
tracked: list[dict] = payload.get("tracked_workers", [])

# ---- Top KPI row ----

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
total = sum(z["total_workers"] for z in zone_stats)
active = sum(z["active_workers"] for z in zone_stats)
violations = sum(z["ppe_violation_workers"] for z in zone_stats)
red_zones = sum(1 for z in zone_stats if z["zpi_band"] == "red")

kpi1.metric("Workers on site", total)
kpi2.metric("Active now", active, f"{(active/total*100 if total else 0):.0f}%")
kpi3.metric("PPE violations", violations)
kpi4.metric("Zones below target", red_zones)


# ---- Live camera previews ----

st.subheader("Live Camera Feeds")
camera_ids = sorted({z["camera_id"] for z in zone_stats})
if camera_ids:
    n_cols = min(len(camera_ids), 2)
    cols = st.columns(n_cols)
    cache_buster = int(time.time())
    for i, cam_id in enumerate(camera_ids):
        with cols[i % n_cols]:
            st.caption(f"**{cam_id}**")
            img_url = f"{HTTP_BASE}/preview/{cam_id}.jpg?t={cache_buster}"
            try:
                st.image(img_url, use_container_width=True)
            except Exception:
                st.warning(f"No preview yet for {cam_id}")


# ---- Zone stats grid ----

st.subheader("Zones")
if not zone_stats:
    st.warning("No zone stats yet.")
else:
    cols = st.columns(min(len(zone_stats), 4))
    for i, z in enumerate(zone_stats):
        with cols[i % len(cols)]:
            band = z["zpi_band"]
            band_color = {"green": "🟢", "amber": "🟡", "red": "🔴",
                          "unknown": "⚪"}[band]
            st.markdown(f"### {band_color} {z['zone_id']}")
            st.caption(f"Camera: {z['camera_id']}")
            c1, c2 = st.columns(2)
            c1.metric("Workers", f"{z['total_workers']}/{z['expected_workers']}")
            c2.metric("Active", f"{z['active_workers']}/{z['expected_active']}")
            st.progress(min(1.0, z["zpi"]), text=f"ZPI {z['zpi']:.2f}")
            if z["low_confidence"]:
                st.caption("⚠️ low confidence (small sample)")
            if z["ppe_violation_workers"] > 0:
                st.error(f"PPE violations: {z['ppe_violation_workers']}")


# ---- Anomalies ----

st.subheader("Recent anomalies")
anomalies = list(st.session_state.recent_anomalies)
if not anomalies:
    st.caption("No anomalies in this session.")
else:
    rows = []
    for a in anomalies[:10]:
        p = a["payload"]
        rows.append({
            "time": p["timestamp"][:19],
            "severity": p["severity"],
            "zone": p["zone_id"],
            "type": p["type"],
            "message": p["message"],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ---- Trend chart for selected zone ----

st.subheader(f"Trend: {selected_zone}")
try:
    r = requests.get(
        f"{HTTP_BASE}/zones/{selected_zone}/trend",
        params={"minutes": trend_minutes},
        timeout=5,
    )
    if r.ok:
        trend = r.json()
        if trend:
            df = pd.DataFrame(trend)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")
            st.line_chart(df[["total_workers", "active_workers"]])
            st.line_chart(df[["zpi"]])
        else:
            st.caption("No data yet for this zone.")
    else:
        st.caption(f"API {r.status_code}: {r.text}")
except Exception as e:
    st.caption(f"Trend fetch failed: {e}")


# ---- Auto-rerun ----

time.sleep(refresh_rate)
st.rerun()
