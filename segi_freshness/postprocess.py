"""YOLO (Ultralytics v8 / 11 ONNX layout) decoding and NMS."""

from __future__ import annotations

import numpy as np

from .reference import LetterboxInfo

_CLASS_OFFSET = 8192.0  # > max input size, separates classes for batched NMS


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Greedy NMS over xyxy boxes. Returns kept indices sorted by score."""
    if len(boxes) == 0:
        return np.empty(0, dtype=np.int64)
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(xx2 - xx1, 0) * np.maximum(yy2 - yy1, 0)
        iou = inter / np.maximum(areas[i] + areas[rest] - inter, 1e-9)
        order = rest[iou <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


def decode_yolo(
    pred: np.ndarray,
    num_classes: int,
    conf_threshold: float,
    iou_threshold: float,
    max_detections: int,
    info: LetterboxInfo,
    src_w: int,
    src_h: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode one image's raw output, shaped (4+nc, anchors) or (anchors, 4+nc).

    Returns (boxes xyxy in source pixels, scores, class ids).
    """
    pred = np.asarray(pred, dtype=np.float32)
    if pred.ndim == 3:
        pred = pred[0]
    if pred.shape[0] == 4 + num_classes and pred.shape[1] != 4 + num_classes:
        pred = pred.T
    if pred.shape[1] != 4 + num_classes:
        raise ValueError(f"unexpected detector output shape {pred.shape} for {num_classes} classes")

    cls_scores = pred[:, 4:]
    class_ids = cls_scores.argmax(axis=1)
    scores = cls_scores[np.arange(len(pred)), class_ids]
    mask = scores >= conf_threshold
    if not mask.any():
        return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64)

    xywh = pred[mask, :4]
    scores = scores[mask]
    class_ids = class_ids[mask]
    boxes = np.empty_like(xywh)
    boxes[:, 0] = xywh[:, 0] - xywh[:, 2] / 2
    boxes[:, 1] = xywh[:, 1] - xywh[:, 3] / 2
    boxes[:, 2] = xywh[:, 0] + xywh[:, 2] / 2
    boxes[:, 3] = xywh[:, 1] + xywh[:, 3] / 2

    keep = nms(boxes + (class_ids[:, None] * _CLASS_OFFSET), scores, iou_threshold)[:max_detections]
    boxes = info.to_source(boxes[keep])
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, src_w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, src_h)
    valid = (boxes[:, 2] - boxes[:, 0] >= 2) & (boxes[:, 3] - boxes[:, 1] >= 2)
    return boxes[valid], scores[keep][valid], class_ids[keep][valid].astype(np.int64)
