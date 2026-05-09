# Construction Labor Intelligence

A computer vision system that turns construction-site CCTV into operational data for project managers — labor counts per zone, PPE compliance, productivity trends, and anomaly alerts. Runs on a single machine, on-prem, no imagery leaves the system.

![Dashboard](demo_assets/screenshots/dashboard_kpis.png)

## What's interesting about it

The hard part isn't detection — fine-tuning YOLOv8m on the [Roboflow Construction Site Safety dataset](https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety) gets you 0.79 test mAP@0.5 in an afternoon. The hard part is everything that turns those boxes into something a project manager can act on without staring at a video wall.

A few decisions that mattered:

**PPE compliance from the strong classes only.** The model is great at spotting hardhats (0.95 mAP) and vests (0.92 mAP) but wobbly on their negatives (NO-Hardhat: 0.56, missing 44% of bare heads). Instead of trusting the weak negative classes, the system derives compliance geometrically: a worker with no hardhat detection in the upper third of their bounding box is flagged as head-violation. Trades the model's strongest signal for the weakest, removes the 44% false-negative hole.

**ZPI as a single ratio.** First version multiplied `(actual_total / expected_total) × (actual_active / expected_active)` — which double-penalised understaffed zones (a separate alert already covers that). Switched to `actual_active / expected_active`. Measures productivity given who's actually there.

**Detection at 3 FPS, ByteTrack at display rate.** ONNX-RT on T4 caps at ~37 FPS aggregate across all streams. 4 cams × 3 FPS = 12 FPS, leaves room. The tracker fills in the gaps so the dashboard still updates smoothly.

## Architecture

```
RTSP cameras ──▶ StreamManager (threaded readers, drop-oldest backpressure)
                     │
                     ▼
                InferenceEngine  (YOLOv8m-ONNX → ByteTrack → geometric PPE assoc → state machine)
                     │
                     ▼ latest_workers snapshot (asyncio lock)
                     │
                AnalyticsEngine  (zone polygon match, ZPI, counts) ─┐
                     │                                              │
                     ▼                                              ▼
              SQLite TimeSeriesStore                          AnomalyDetector
                     │                                              │
                     └─────────────────┬────────────────────────────┘
                                       ▼
                              FastAPI WebSocket fan-out
                                       │
                                       ▼
                                Streamlit dashboard
```

Single process, single GPU, asyncio for I/O, threads for the OpenCV reads that won't release the GIL.

## What you see in the dashboard

Per-zone breakdown — workers present vs expected, active ratio, ZPI:

![Zones](demo_assets/screenshots/zones.png)

Anomaly log — four detector types with persistence requirements so a single flicker doesn't fire an alert:

![Anomalies](demo_assets/screenshots/anomalies.png)

## Try it

```bash
git clone https://github.com/samisaud/construction-labor-intel
cd construction-labor-intel
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Drop a fine-tuned ONNX into `training/artifacts/weights/best.onnx` (training notebook not included — Roboflow CSS dataset, YOLOv8m, 50 epochs gets you there). Then:

```bash
LABOR_INTEL_STREAM_SOURCES="cam_01=demo_assets/sample_footage.mp4" \
  uvicorn app.main:app --port 8000
```

In another terminal:

```bash
pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
```

The included sample footage is a 1.2 MB CC0 clip from Pexels. Replace with your own RTSP URLs (`cam_01=rtsp://...`) for real cameras.

## The numbers

| | |
|---|---|
| Test mAP@0.5 | 0.794 |
| Test mAP@0.5:0.95 | 0.445 |
| Person detection | 0.887 |
| Hardhat | 0.950 |
| Vest | 0.922 |
| NO-Hardhat (weak class) | 0.562 |
| Inference latency (T4, ONNX-RT) | 27 ms / image |
| Aggregate throughput ceiling | ~37 FPS |
| Anomaly detection cadence | 3s tick |

Full training and validation breakdown in [MODEL_CARD.md](MODEL_CARD.md). Operational diagnostics in [RUNBOOK.md](RUNBOOK.md).

## What I'd do next

The honest known issues:

- **Geometric PPE association produces noticeable false positives on workers facing away from the camera** where the hardhat bbox is small and doesn't overlap the upper-third zone enough to register. Lowering the IoU threshold helps but trades for more false negatives on distant workers. Real fix: retrain with more back-of-head samples, retire the geometric fallback.
- **Safety Cone class collapsed val→test (0.91 → 0.46)** — distribution shift, not noise. Fine for v1 since cones don't drive any logic, but blocks any future "approach to danger zone" alert.
- **MoViNet activity classifier deferred.** Velocity-based state machine works but is crude — workers leaning on rebar get flagged idle, workers walking past on break get flagged active. MoViNet on cropped tracks would fix it, gated on velocity to keep CPU cost in check.

## Stack

YOLOv8m (Ultralytics) → ONNX → ByteTrack → Shapely (polygon ops) → FastAPI → aiosqlite → Streamlit. Tests via pytest with 38 cases covering the analytics and storage layers. CI on GitHub Actions.

## Tests

```bash
pytest -v
```

Covers PPE geometric association, zone polygon assignment, ZPI math, anomaly persistence and re-fire logic, SQLite round-trips and retention, and end-to-end FastAPI lifespan boot. The model isn't loaded in tests — that path is exercised by the demo script on real footage.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness + stream/DB status |
| GET | `/zones/{zone_id}/trend?minutes=60` | Rolling labor count + ZPI history |
| GET | `/shift/summary?hours=8` | Peak / trough / mean active ratio + anomaly count |
| GET | `/anomalies?since_minutes=60` | Raw anomaly log |
| WS | `/ws/live` | Snapshot every 3s + immediate anomaly pushes |

Configuration via env vars prefixed `LABOR_INTEL_` — see [`.env.example`](.env.example). Zone polygons in [`training/configs/zones.yaml`](training/configs/zones.yaml), normalised 0–1 so they survive resolution changes.
