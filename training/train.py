"""Train and export the two models on a workstation GPU (not on the Jetson).

    pip install ultralytics onnx onnxsim
    python training/train.py detector   --data training/produce_det.yaml
    python training/train.py classifier --data datasets/cls

Outputs ONNX files with FP32 I/O to models/. Copy them to the Jetson and run
scripts/build_engines.sh there.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["detector", "classifier"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default=None, help="base weights (default yolo11s.pt / yolo11s-cls.pt)")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", type=Path, default=Path("models"))
    args = ap.parse_args()

    from ultralytics import YOLO

    if args.task == "detector":
        model = YOLO(args.model or "yolo11s.pt")
        imgsz = args.imgsz or 640
        model.train(data=args.data, epochs=args.epochs, imgsz=imgsz, batch=args.batch,
                    hsv_h=0.005, hsv_s=0.3, hsv_v=0.3,  # keep hue augmentation tiny: colour IS the signal
                    mosaic=1.0, close_mosaic=15, patience=30)
        onnx = model.export(format="onnx", imgsz=imgsz, opset=17, simplify=True, dynamic=False)
        target = args.out / "produce_det.onnx"
    else:
        model = YOLO(args.model or "yolo11s-cls.pt")
        imgsz = args.imgsz or 224
        model.train(data=args.data, epochs=args.epochs, imgsz=imgsz, batch=args.batch,
                    hsv_h=0.0, hsv_s=0.2, hsv_v=0.3, fliplr=0.5, erasing=0.0, patience=30)
        onnx = model.export(format="onnx", imgsz=imgsz, opset=17, simplify=True, dynamic=True)
        target = args.out / "freshness_cls.onnx"

    args.out.mkdir(parents=True, exist_ok=True)
    shutil.copy(onnx, target)
    print(f"exported {target}")


if __name__ == "__main__":
    main()
