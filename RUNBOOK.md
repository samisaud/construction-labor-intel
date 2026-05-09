# Runbook

What to check when something is wrong, in order of "fastest to verify". Aimed
at whoever is on call after this is deployed — assumes familiarity with the
[architecture](README.md) but no other context.

## Quick health check

```bash
curl -s http://<host>:8000/health | jq
```

Healthy response:

```json
{
  "status": "ok",
  "version": "0.1.0",
  "active_streams": 4,
  "inference_provider": "auto",
  "db_writable": true
}
```

| Field | Meaning if wrong |
|---|---|
| `status: degraded` | At least one of: model failed to load, no streams alive, DB not writable |
| `status: down` | API process is up but a critical component crashed |
| `active_streams < expected` | One or more cameras are disconnected. See "RTSP issues" below. |
| `db_writable: false` | Disk full, permission error, or DB file locked by another process |

## Symptom: dashboard shows "Waiting for first snapshot..."

1. Is the backend reachable? `curl http://<host>:8000/health` — if connection refused, the API process is down.
2. Is the WebSocket connecting? Open browser dev tools on the dashboard — Network → WS — should see `/ws/live` with status 101.
3. Are there any active streams? If `active_streams == 0`, the snapshot loop produces empty payloads (which the dashboard treats as "still waiting"). Check stream sources.
4. Check backend logs: `docker compose logs -f backend` (or the equivalent for your deployment). The first log line you want to see is `StreamManager started N readers`. If you don't see it, no streams are configured.

## Symptom: zones all show 0 workers but cameras are showing footage

The detector ran but found no Persons. Likely causes:

1. **Wrong model loaded.** Check the startup log — `Loading model: ...` should point to your fine-tuned ONNX, not a stock COCO weight file. If the path is wrong, the COCO model still has class 0 = "person" and would actually detect, but the entire class index map is misaligned and `Hardhat` becomes `bicycle`. Stop and verify weights.
2. **Confidence threshold too high.** Check `LABOR_INTEL_DETECTION_CONF_THRESHOLD`. Default is 0.35. If someone bumped it to 0.7, low-confidence detections in poor lighting will silently disappear.
3. **Camera misaimed.** Look at the raw RTSP feed in VLC. If the camera shifted overnight (vibration, knocked, sun glare), the framing may no longer match the zone polygons.

## Symptom: zones show workers but they're "unassigned" (no zone_id)

Zone polygon configuration doesn't cover where workers actually appear in frame.

1. Run the demo script on a recent snapshot: `python -m scripts.demo_video --video <recent.mp4> --camera-id <cam> --output /tmp/diag --max-frames 10`
2. Look at `/tmp/diag/annotated.mp4`. If bboxes are inside what you expect to be a zone but `zone_stats.jsonl` shows the zone empty, the polygon is wrong.
3. Polygons in `training/configs/zones.yaml` are normalized 0–1. Sketch them on paper from a frame and re-edit the YAML. Edit, restart the backend (zone config is loaded once at startup).

## Symptom: ZPI permanently red on a zone

1. Check `expected_workers` and `expected_active` in `zones.yaml`. If construction phase has changed (e.g., framing → finishing) the expected counts need updating.
2. Pull the trend: `curl 'http://<host>:8000/zones/<zone_id>/trend?minutes=120' | jq`. If `total_workers` is consistently below half of `expected_workers`, this isn't a productivity issue — it's understaffing. Either reassign crew or update `expected_workers`.

## Symptom: anomalies firing constantly / dashboard alert spam

1. **PPE Violation Cluster fires repeatedly:** check the test-set mAP@0.5 of `Hardhat` (0.95) and `Safety Vest` (0.92). These are reliable. But if your site has unusual headwear (welding masks, soft beanies, religious head coverings, hooded sweatshirts), the Hardhat detector may not fire for those workers, and they'll be flagged as head-violation. Either the site needs to standardize hardhats or the model needs site-specific fine-tuning.
2. **Idle Spike fires every break:** the 10-minute baseline picks up activity drops. If lunch breaks fire it predictably, either schedule break windows where alerts are suppressed (not yet implemented; v1.1 backlog) or raise `IDLE_SPIKE_DROP_THRESHOLD` in `anomaly_detector.py`.
3. **Productivity Collapse fires after lunch return:** workers re-entering the zone gradually means low active count for several minutes. Persistence threshold is 10 minutes already, but if lunch breaks last >10 minutes in the zone, this will fire. Same mitigation as idle spike.

## Symptom: backend pegged at 100% CPU, can't keep up

1. `docker stats` (or `top`). If a single Python process is at 100% of one core and stream queue backlog is growing (`stats.frames_dropped_full_queue` rising), the inference loop is the bottleneck.
2. Most likely cause: ONNX-RT is running on **CPU** instead of GPU. Restart the container, watch the startup log for `inference_provider`. If it shows `cpu` you need to install `onnxruntime-gpu` (not `onnxruntime`) and make the GPU visible to the container.
3. If GPU is active and still saturated: drop `LABOR_INTEL_STREAM_TARGET_FPS` from 3 to 2. T4 ceiling is ~37 FPS aggregate; 4 cameras × 3 FPS = 12 FPS leaves a margin of 25 FPS. If you're saturating that, something else is competing for the GPU.

## Symptom: SQLite database growing larger than expected

1. Default retention is 24 hours, swept on startup. If the backend hasn't restarted in days, the DB grows.
2. Manual sweep:
   ```bash
   docker compose exec backend python -c "
   import asyncio, app.time_series_store as ts
   async def f():
     s = ts.TimeSeriesStore()
     await s.open()
     n = await s._purge_old_rows(hours=24)
     print(f'Purged {n} rows')
     await s.close()
   asyncio.run(f())
   "
   ```
3. For production, set `LABOR_INTEL_SQLITE_RETENTION_HOURS=12` and add a daily cron-restart of the backend.

## Symptom: RTSP stream keeps reconnecting

Look at the StreamReader stats via `/health` (or directly in logs):

| Pattern | Likely cause |
|---|---|
| `reconnect_count` increments every few seconds | Camera firmware issue or network instability |
| `reconnect_count` increments once per hour | Camera credential lease expiring — rotate the RTSP password and update sources |
| `seconds_since_last_frame > 30` despite "alive" | Stream connected but stalled (codec mismatch?). Try `ffprobe rtsp://...` from the backend container. |

If the simulator (FFmpeg → MediaMTX) keeps reconnecting in dev:
- Most common cause: the source MP4 has variable frame rate or unusual codec. Re-encode once: `ffmpeg -i in.mp4 -c:v libx264 -r 25 -fps_mode cfr -an out.mp4`.

## Symptom: dashboard shows different numbers than the API

The dashboard reads zone trends via HTTP and live state via WebSocket. If they disagree:
1. Browser caching: hard-reload the dashboard (Cmd-Shift-R).
2. Clock drift: SQLite stores UTC ISO timestamps. If the dashboard host clock is off by a minute, "last 60 minutes" queries will silently exclude recent data.
3. Multiple backend instances: dashboard hits one, WebSocket subscribed to another. Make sure there's exactly one backend reachable.

## Dumping diagnostic data

When opening a bug report, attach:

```bash
mkdir -p /tmp/diag && cd /tmp/diag
curl -s http://<host>:8000/health > health.json
curl -s 'http://<host>:8000/anomalies?since_minutes=120' > anomalies.json
curl -s 'http://<host>:8000/shift/summary?hours=4' > shift.json
docker compose logs --tail=500 backend > backend.log
docker compose logs --tail=200 rtsp > rtsp.log
tar czf diag.tar.gz *.json *.log
```

## Restart procedures

Ordered from least to most disruptive:

1. **Reload zones.yaml only** — not currently supported, requires restart.
2. **Restart backend** (loses ByteTrack track IDs, all currently-tracked workers get new IDs):
   ```bash
   docker compose restart backend
   ```
3. **Full stack restart** (also drops MediaMTX):
   ```bash
   docker compose down && docker compose up -d
   ```
4. **Reset DB** (loses all historical data — only do if corrupted):
   ```bash
   docker compose down
   rm data/labor_intel.db*
   docker compose up -d
   ```
