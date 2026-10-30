# models/

Put ONNX exports here (from `training/train.py`) and build TensorRT engines on the
Jetson itself with `scripts/build_engines.sh`:

| File | Needed for | Notes |
|---|---|---|
| `freshness_cls_fp16.engine` | optional, both modes | fresh / ripe / overripe / spoiled classifier, dynamic batch 1–32 |
| `produce_det_fp16.engine` | `runtime.mode: detection` | YOLO11 produce detector, 640×640 |

Engines are tied to the exact GPU + TensorRT version. Rebuild after every JetPack upgrade.
Without any engines the system still runs in `zones` mode using the CUDA colour-defect analysis.
