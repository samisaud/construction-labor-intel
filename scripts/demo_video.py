"""End-to-end smoke test against a single video file.

Runs the full pipeline (inference + tracking + PPE + zone stats + anomalies)
on a local MP4 — no RTSP server, no FastAPI, no dashboard. Lets you verify
the model and the analytics layer are doing the right thing before standing
up the multi-camera infrastructure.

Outputs:
  - annotated video at <output_dir>/annotated.mp4 with bbox + PPE + state overlay
  - per-tick stats at <output_dir>/zone_stats.jsonl
  - any anomalies fired at <output_dir>/anomalies.jsonl
  - one-line summary at <output_dir>/summary.json

Usage:
  python -m scripts.demo_video \\
    --video /path/to/footage.mp4 \\
    --camera-id cam_01 \\
    --output ./demo_output

  # Skip annotated video (faster):
  python -m scripts.demo_video --video x.mp4 --camera-id cam_01 \\
    --output ./out --no-annotate

This script is also the closest thing to an integration test that exercises
the inference engine end-to-end. CI can't run it (needs the model + a video)
but anyone with the artifacts can run it in 30 seconds.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2  # type: ignore[import-untyped]
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
logger = logging.getLogger("demo")


# ---------- Visualization ----------

_PPE_COLORS = {
    "compliant": (0, 200, 0),         # green
    "head_violation": (0, 165, 255),  # orange
    "torso_violation": (0, 165, 255),
    "both_violation": (0, 0, 255),    # red
    "unknown": (180, 180, 180),       # gray
}

_STATE_GLYPH = {
    "active": "▶",
    "idle": "■",
    "transitioning": "▷",
}


def _annotate(frame: np.ndarray, workers, zone_stats) -> np.ndarray:
    """Draw bboxes, PPE color, state glyph, and zone overlay on a frame."""
    h, w = frame.shape[:2]
    out = frame.copy()

    for w_obj in workers:
        bbox = w_obj.bbox
        x1, y1 = int(bbox.x1 * w), int(bbox.y1 * h)
        x2, y2 = int(bbox.x2 * w), int(bbox.y2 * h)
        ppe = (w_obj.ppe_status.value
               if hasattr(w_obj.ppe_status, "value") else str(w_obj.ppe_status))
        state = (w_obj.activity_state.value
                 if hasattr(w_obj.activity_state, "value") else str(w_obj.activity_state))
        color = _PPE_COLORS.get(ppe, (180, 180, 180))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        label = f"#{w_obj.track_id} {_STATE_GLYPH.get(state, '?')} {ppe[:3]}"
        cv2.putText(out, label, (x1, max(15, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    # Zone summary header
    y_off = 20
    for zs in zone_stats:
        band = zs.zpi_band
        band_color = {"green": (0, 200, 0), "amber": (0, 200, 200),
                      "red": (0, 0, 255), "unknown": (180, 180, 180)}.get(band, (200, 200, 200))
        text = (f"{zs.zone_id}: {zs.total_workers}w "
                f"({zs.active_workers}a/{zs.idle_workers}i) "
                f"ZPI={zs.zpi:.2f} {band}")
        cv2.putText(out, text, (10, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, band_color, 1, cv2.LINE_AA)
        y_off += 22

    return out


# ---------- Main ----------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path,
                        help="Path to local MP4 / AVI / RTSP URL")
    parser.add_argument("--camera-id", required=True,
                        help="Camera ID (must exist in zones.yaml)")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output directory (created if needed)")
    parser.add_argument("--target-fps", type=int, default=3,
                        help="Inference FPS (default: matches LABOR_INTEL_STREAM_TARGET_FPS)")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Stop after N inferred frames (0 = process whole video)")
    parser.add_argument("--no-annotate", action="store_true",
                        help="Skip writing the annotated video (faster)")
    args = parser.parse_args()

    # Defer heavy imports so --help is fast.
    from app.analytics_engine import AnalyticsEngine
    from app.anomaly_detector import AnomalyDetector
    from app.inference_engine import InferenceEngine

    args.output.mkdir(parents=True, exist_ok=True)

    if not args.video.exists() and not str(args.video).startswith("rtsp://"):
        logger.error("Video path does not exist: %s", args.video)
        return 2

    # ---- Set up pipeline ----
    logger.info("Loading inference engine...")
    inference = InferenceEngine()

    logger.info("Loading analytics engine...")
    analytics = AnalyticsEngine()
    if args.camera_id not in analytics._zones_by_camera:
        logger.error(
            "Camera '%s' not found in zones.yaml. Known cameras: %s",
            args.camera_id, list(analytics._zones_by_camera.keys()),
        )
        return 3
    critical = {z.zone_id for z in analytics.zones if z.critical_path}
    anomaly_det = AnomalyDetector(critical_path_zones=critical)

    # ---- Open video ----
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        logger.error("Could not open video: %s", args.video)
        return 4

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stride = max(1, int(round(src_fps / args.target_fps)))
    total_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    logger.info(
        "Video: %dx%d @ %.1f FPS, %d frames -> sampling every %d (target %d FPS)",
        src_w, src_h, src_fps, total_src, stride, args.target_fps,
    )

    writer = None
    if not args.no_annotate:
        out_path = args.output / "annotated.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, args.target_fps, (src_w, src_h))
        logger.info("Writing annotated video to %s", out_path)

    stats_fp = (args.output / "zone_stats.jsonl").open("w")
    anomaly_fp = (args.output / "anomalies.jsonl").open("w")

    # ---- Process loop ----
    raw_idx = 0
    inferred_idx = 0
    tick_seconds = 3
    last_tick = time.monotonic()

    # Buffer the latest workers between ticks (analytics fires at fixed cadence,
    # not every frame).
    latest_workers: list = []

    inference_t_total = 0.0
    inference_count = 0
    anomaly_total = 0
    summary_zone_stats: list = []

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            raw_idx += 1
            if raw_idx % stride != 0:
                continue

            t0 = time.perf_counter()
            workers = inference.process_frame(frame, args.camera_id, inferred_idx)
            inference_t_total += time.perf_counter() - t0
            inference_count += 1
            inferred_idx += 1
            latest_workers = workers

            # Tick analytics on a fixed cadence.
            now_mono = time.monotonic()
            if now_mono - last_tick >= tick_seconds:
                last_tick = now_mono
                now_dt = datetime.now(timezone.utc)
                zone_stats, tagged = analytics.compute_zone_stats(
                    latest_workers, now=now_dt,
                )
                anomalies = anomaly_det.evaluate(zone_stats)

                for zs in zone_stats:
                    stats_fp.write(zs.model_dump_json() + "\n")
                for a in anomalies:
                    anomaly_fp.write(a.model_dump_json() + "\n")
                anomaly_total += len(anomalies)
                summary_zone_stats = zone_stats
                if anomalies:
                    for a in anomalies:
                        logger.info("ANOMALY: %s", a.message)

            # Annotate. Use most recent zone stats for header even between ticks.
            if writer is not None:
                annotated = _annotate(frame, latest_workers, summary_zone_stats)
                writer.write(annotated)

            if inference_count % 30 == 0:
                logger.info(
                    "Processed %d inferred frames, %d active tracks, mean inference: %.1f ms",
                    inference_count, inference.active_track_count,
                    1000 * inference_t_total / inference_count,
                )

            if args.max_frames and inference_count >= args.max_frames:
                logger.info("Reached --max-frames=%d, stopping", args.max_frames)
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        stats_fp.close()
        anomaly_fp.close()

    # ---- Summary ----
    summary = {
        "video": str(args.video),
        "camera_id": args.camera_id,
        "frames_inferred": inference_count,
        "mean_inference_ms": (
            1000 * inference_t_total / inference_count if inference_count else None
        ),
        "throughput_fps": (
            inference_count / inference_t_total if inference_t_total else None
        ),
        "anomalies_total": anomaly_total,
        "final_zone_stats": [
            {
                "zone_id": zs.zone_id,
                "total_workers": zs.total_workers,
                "active_workers": zs.active_workers,
                "zpi": zs.zpi,
                "zpi_band": zs.zpi_band,
            }
            for zs in summary_zone_stats
        ],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Done. Summary: %s", json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
