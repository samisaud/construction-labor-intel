# Model Card — Construction PPE Detector v1

| Field | Value |
|---|---|
| Model architecture | YOLOv8m (Ultralytics) |
| Task | Multi-class object detection (10 classes) |
| Input | 640×640 RGB image |
| Output | Bounding boxes + class probabilities |
| Training data | CSS (Construction Site Safety) — Roboflow Universe |
| Training set size | 2,605 images |
| Validation set size | 114 images |
| Test set size | 82 images |
| Training hardware | NVIDIA Tesla T4 (16 GB), Google Colab |
| Training duration | ~1.4 hours, 50 epochs |
| Optimizer | AdamW, lr0=0.001 |
| Export format | ONNX, opset 12, dynamic batch |
| ONNX file size | 99.4 MB |

## Classes

| Index | Class |
|---|---|
| 0 | Hardhat |
| 1 | Mask |
| 2 | NO-Hardhat |
| 3 | NO-Mask |
| 4 | NO-Safety Vest |
| 5 | Person |
| 6 | Safety Cone |
| 7 | Safety Vest |
| 8 | machinery |
| 9 | vehicle |

## Performance — Test Set (held out, never used for selection)

| Class | mAP@0.5 | Threshold | Status |
|---|---|---|---|
| Person | 0.887 | > 0.70 | ✅ |
| Hardhat | 0.950 | > 0.60 | ✅ |
| Safety Vest | 0.922 | > 0.50 | ✅ |
| NO-Safety Vest | 0.833 | > 0.50 | ✅ |
| **NO-Hardhat** | **0.562** | > 0.60 | ❌ **below threshold** |
| **Safety Cone** | **0.464** | n/a | ⚠️ collapsed vs val (0.911 → 0.464) |
| Mask | 0.783 | n/a | ✅ |
| NO-Mask | 0.840 | n/a | ✅ |
| machinery | 0.907 | n/a | ✅ |
| vehicle | 0.788 | n/a | ✅ |
| **Overall** | **0.794** mAP@0.5, **0.445** mAP@0.5:0.95 | | |

### Performance — Validation Set (used for early stopping)

Validation numbers run higher (overall mAP@0.5 = 0.833) but should be treated as
optimistic — the val set was used for model selection and is mildly contaminated
as a generalization estimator. Test-set numbers above are the ones to cite.

### Inference latency — T4 GPU, ONNX-RT CUDA

| Batch size | ms/image | Throughput |
|---|---|---|
| 1 | 27.2 | 37 img/s |
| 4 | 26.1 | 38 img/s |
| 8 | 27.5 | 36 img/s |

**No batch-size scaling.** YOLOv8m at 640×640 is memory-bandwidth-bound on T4.
For higher per-camera throughput, export to TensorRT (expected 2–3× speedup,
not yet done — flagged for v1.2).

## Intended use

- Deployment behind a closed-circuit network at a single construction site
- Inputs: 4 fixed RTSP cameras at 3 FPS, indoor + outdoor mix
- Outputs consumed by a PMO dashboard for labor productivity monitoring
- Inputs and outputs **never leave** the customer's network

### Out-of-scope uses

- Identifying individual workers (this is not a face recognition or biometric system)
- Disciplinary action against named individuals
- Replacing trained safety officers
- Crowd estimation in dense scenes (>30 people in one frame) — model not validated for this
- Aerial / drone imagery — training set is ground-level CCTV
- Night / IR cameras — training set is daylight only

## Known limitations

### Detection limitations

1. **NO-Hardhat recall is 0.561 on test** — the model misses ~44% of bare heads
   when relying on this class directly. The system **mitigates this** by
   deriving non-compliance geometrically: a Person bbox without a Hardhat
   detection in its upper third is treated as head-violation. This uses the
   strong positive-class detector (Hardhat 0.95) instead of the weak negative
   class. See `app/inference_engine.py::_associate_ppe`.

2. **Safety Cone test mAP collapsed** (0.911 val → 0.464 test). Likely a
   distribution shift — val cones were "easy" (foreground, isolated), test
   cones are clustered or distant. Cones are not used in v1 zone logic, but
   any future cone-based "danger zone proximity" alert will need retraining
   or stronger augmentation.

3. **Vehicle / machinery overlap.** Some confusion between classes 8 and 9
   (machinery / vehicle). Acceptable for v1 since neither feeds into
   compliance or productivity logic.

4. **Small objects.** mAP@0.5:0.95 of 0.445 indicates loose bounding boxes.
   For zone-edge logic ("is this person inside a 2m-wide danger polygon?"),
   loose boxes will cause flicker. Mitigation: increase `imgsz` to 832 or
   960 in v2 retraining.

5. **Training was slightly under-converged.** Loss was still trending down
   at epoch 50; another 20–30 epochs could yield ~2–3% additional mAP. Not
   blocking but flagged for v2.

### Population / fairness considerations

- Training images are skewed toward construction crews on visible commercial
  sites in daylight. Performance may degrade on:
  - Crews with non-standard PPE (e.g. soft caps for utility work)
  - Sites where workers wear weather gear that occludes vests
  - Workers significantly out of typical adult body proportions
- The system reports counts and ratios at the zone level. **It does not
  identify, name, or assign actions to individual workers.** Track IDs are
  ephemeral integers that reset across server restarts and are never linked
  to identity.

## Operational guardrails enforced by the surrounding system

These belong to the system, not the model — listed here so the model card is
honest about how the model is used:

- Imagery is never persisted to disk by the pipeline (only structured stats).
- Anomalies report at the **zone level**, not the worker level.
- PPE alerts have a persistence requirement (>3 violations in the same zone
  simultaneously) before firing — this damps single false positives.
- Zone Productivity Index (ZPI) measures throughput against an
  operator-defined `expected_active` baseline, not against other workers
  or other zones.

## Re-training & updates

- Re-evaluate quarterly against new site footage.
- If `NO-Hardhat` test mAP can be brought above 0.70 by adding negative
  examples, retire the geometric fallback in favor of direct class use.
- Expand training set with night / IR / aerial imagery before deploying to
  any of those modalities.

## Data and privacy

- Training data is publicly available CSS dataset under CC-BY-4.0.
- No customer imagery was used for the released model.
- Customer footage stays on customer infrastructure. Inference happens
  on-prem; only structured event data crosses the network boundary.

## Authors / contact

- Model trained by: Sami (project owner)
- Pipeline architecture: documented in `README.md` and inline comments
- File issues / questions in the repository tracker
