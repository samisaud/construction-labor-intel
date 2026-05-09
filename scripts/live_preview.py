"""Live webcam preview with bbox overlay. Standalone — needs the backend OFF
so we can grab the camera. Press 'q' to quit.
"""
import cv2
import time
from app.inference_engine import InferenceEngine
from app.analytics_engine import AnalyticsEngine

PPE_COLORS = {
    "compliant": (0, 200, 0),
    "head_violation": (0, 165, 255),
    "torso_violation": (0, 165, 255),
    "both_violation": (0, 0, 255),
    "unknown": (180, 180, 180),
}

print("Loading model...")
inference = InferenceEngine()
analytics = AnalyticsEngine()

print("Opening camera...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit("Camera failed to open")

frame_idx = 0
last_inference = 0.0
inference_interval = 0.33  # 3 FPS

while True:
    ok, frame = cap.read()
    if not ok:
        break

    h, w = frame.shape[:2]
    now = time.time()
    if now - last_inference >= inference_interval:
        workers = inference.process_frame(frame, "cam_01", frame_idx)
        last_inference = now
        frame_idx += 1
    else:
                                                 s from this tick
    for w_obj in workers:
        b = w_obj.bbox
        x1, y1         x1 * w), int(b.        x1, y1         x1 * w), int(b.        x2 * h)
        ppe = w_obj.ppe_status.value
        state = w_obj.activity_state.value
        color = PPE_COLORS.get(ppe, (180, 180, 180))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"#{w_obj.track_id} {state}        label = f"#{putText(frame, label, (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5                    cv2.)

    cv2.imshow("Construction Labor Intel — Live (q to quit)", frame)
    if cv2.waitKey(1) & 0xFF == o    if cv2.waitKey(1) & 0xFF == ose(    if cv2royAllWindows()
