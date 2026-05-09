# Construction Labor Intelligence

Real-time multi-camera computer vision system for construction site labor monitoring. Runs entirely on-prem — no worker imagery leaves the deployment.

## What it does

- **Detects** workers and PPE (hardhats, vests) across 4 simultaneous camera feeds
- **Tracks** workers across frames with ByteTrack (persistent track IDs)
- **Classifies** worker activity as ACTIVE / IDLE / TRANSITIONING via a velocity-based state machine
- **Aggregates** per-zone labor counts, activity ratios, and a Zone Productivity Index (ZPI) on a 3-second tick
- **Flags** four anomaly types: zone understaffed, idle spike, productivity collapse, PPE violation cluster
- **Broadcasts** snapshots and alerts via WebSocket to a Streamlit dashboard

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

### Key design decisions

- **PPE compliance is derived geometrically from positive-class detectors** (Hardhat 0.95 mAP, Vest 0.92 mAP), not from the weak `NO-Hardhat` (0.56 mAP) class. A hardhat detection inside the upper third of a person's bbox = compliant head; absence = head violation. This avoids the 44% recall hole on the negative classes.
- **ZPI = `actual_active / expected_active`** — single ratio, not the product spec'd originally. The product form double-penalized understaffed zones, which the dedicated understaffing alert already covers.
- **3 FPS detection per camera** with ByteTrack interpolating between frames. T4 ONNX-RT ceiling is ~37 FPS aggregate; 4 cams × 3 FPS = 12 FPS leaves headroom for pose model and other costs.
- **Single-process, single uvicorn worker.** Inference engine holds GPU + ByteTrack state; multi-worker would break track continuity.
- **MoViNet activity classifier deferred to v1.1.** The velocity-based state machine is sufficient for the v1 demo; MoViNet adds CPU cost we want measured before committing.

## Setup

### Requirements

- Python 3.11+
- ONNX-RT GPU (CUDA) for production; CPU-only is fine for development
- ffmpeg (for the simulator) and a video file or two
- MediaMTX for RTSP simulation, or real RTSP cameras

### Install

```bash
git clone <repo> && cd construction-labor-intel
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Place model weights

```bash
mkdir -p training/artifacts/weights
cp /path/to/best.onnx training/artifacts/weights/
```

### Quickest path: run the demo on a single video

If you have model weights and a construction footage MP4, you can skip RTSP entirely and run end-to-end in 30 seconds:

```bash
make install
make demo VIDEO=./footage.mp4 CAM=cam_01
```

This runs detection + tracking + zone analytics + anomaly detection on the file and writes:
- `demo_output/annotated.mp4` — bbox overlay with PPE color and activity state
- `demo_output/zone_stats.jsonl` — per-tick zone statistics
- `demo_output/anomalies.jsonl` — any anomalies fired during the video
- `demo_output/summary.json` — single-line summary with throughput

This is the fastest way to validate the model and pipeline before standing up the full multi-camera stack.

### Run with simulated streams

Three terminals:

```bash
# 1. RTSP server
./mediamtx

# 2. Stream simulator
python -m simulation.stream_simulator \
  --videos cam_01:./videos/north.mp4 cam_02:./videos/east.mp4

# 3. Backend
export LABOR_INTEL_STREAM_SOURCES="cam_01=rtsp://localhost:8554/cam_01,cam_02=rtsp://localhost:8554/cam_02"
uvicorn app.main:app --reload
```

Then dashboard:

```bash
cd dashboard && pip install -r requirements.txt
streamlit run app.py
```

Visit http://localhost:8501.

### Run with Docker

```bash
cp .env.example .env  # edit stream sources
docker compose up --build
```

Backend at http://localhost:8000, dashboard at http://localhost:8501.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness + stream/DB status |
| GET | `/zones/{zone_id}/trend?minutes=60` | Rolling labor count + ZPI history |
| GET | `/shift/summary?hours=8` | Peak / trough / mean active ratio + anomaly count |
| GET | `/anomalies?since_minutes=60` | Raw anomaly log |
| WS | `/ws/live` | Snapshot every 3s + immediate anomaly pushes |

## Configuration

All knobs are env vars with prefix `LABOR_INTEL_`. See [.env.example](.env.example).

Zone definitions live in [training/configs/zones.yaml](training/configs/zones.yaml). Polygons are normalized 0–1 so they're resolution-independent.

## Tests

```bash
pip install pytest pytest-asyncio
pytest -v
```

Tests cover: PPE geometric association, zone assignment, ZPI math, anomaly persistence/firing logic, SQLite round-trips and retention. The inference model itself is not loaded in tests — that's covered by the validation runs in Colab.

## Repo layout

```
construction-labor-intel/
├── app/                  # FastAPI backend
│   ├── main.py           # lifespan, WS, HTTP routes
│   ├── stream_manager.py # threaded RTSP/file ingestion
│   ├── inference_engine.py
│   ├── analytics_engine.py
│   ├── anomaly_detector.py
│   ├── time_series_store.py
│   ├── schemas.py        # Pydantic v2 wire format
│   └── config.py         # pydantic-settings
├── dashboard/            # Streamlit UI
├── scripts/              # Standalone tools (demo_video.py, ...)
├── simulation/           # FFmpeg-based RTSP simulator
├── training/
│   ├── configs/zones.yaml
│   └── artifacts/weights/  # best.pt, best.onnx (gitignored)
├── tests/                # pytest, ~38 tests
├── Dockerfile
├── docker-compose.yml
├── Makefile              # make help to see all tasks
├── MODEL_CARD.md
├── RUNBOOK.md
└── requirements.txt
```

## Documentation

- [`MODEL_CARD.md`](MODEL_CARD.md) — training metrics, known limitations, intended use, fairness considerations
- [`RUNBOOK.md`](RUNBOOK.md) — operational diagnostics; what to check when something breaks
- [`simulation/README.md`](simulation/README.md) — RTSP simulator setup

## Roadmap

- **v1.1**: MoViNet-A0 activity classifier (gated on velocity, batched across tracks)
- **v1.2**: TensorRT export for 2-3× inference throughput
- **v1.3**: Multi-GPU support for >4 cameras
- **v2.0**: PostgreSQL + TimescaleDB backend for long-term retention; multi-site federation
