"""Cut classifier training crops from YOLO-labelled store images.

Uses the exact same crop/resize maths as the on-device CUDA kernel
(segi_freshness.reference.crop_resize), so the classifier sees identical
inputs in training and production.

    python training/make_crops.py --images datasets/det/images/train \
        --labels datasets/det/labels/train --out datasets/cls_unsorted --size 224

Then have store staff sort datasets/cls_unsorted/<produce>/*.jpg into
datasets/cls/{train,val}/{fresh,ripe,overripe,spoiled}/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from segi_freshness.reference import crop_resize


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--names", nargs="*", default=None, help="class names in YOLO id order")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--min-px", type=int, default=24, help="skip boxes smaller than this")
    args = ap.parse_args()

    count = 0
    for img_path in sorted(p for p in args.images.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}):
        label_path = args.labels / f"{img_path.stem}.txt"
        if not label_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        rows = np.loadtxt(label_path, ndmin=2)
        for j, (cid, cx, cy, bw, bh) in enumerate(rows[:, :5]):
            box = np.array([[(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]], np.float32)
            if bw * w < args.min_px or bh * h < args.min_px:
                continue
            crop = crop_resize(img, box, args.size)[0]  # 3 x S x S RGB in [0, 1]
            bgr = (crop.transpose(1, 2, 0)[..., ::-1] * 255).round().astype(np.uint8)
            name = args.names[int(cid)] if args.names else str(int(cid))
            dst = args.out / name
            dst.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dst / f"{img_path.stem}_{j:03d}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            count += 1
    print(f"wrote {count} crops to {args.out}")


if __name__ == "__main__":
    main()
