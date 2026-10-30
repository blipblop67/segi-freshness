import numpy as np
import pytest

from segi_freshness.postprocess import decode_yolo, nms
from segi_freshness.reference import letterbox_params


def test_nms_suppresses_overlaps():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], np.float32)
    scores = np.array([0.9, 0.8, 0.7], np.float32)
    assert nms(boxes, scores, 0.5).tolist() == [0, 2]


def test_nms_empty():
    assert nms(np.zeros((0, 4)), np.zeros(0), 0.5).size == 0


def _pred(nc, anchors):
    p = np.zeros((4 + nc, anchors), np.float32)
    return p


def test_decode_yolo_maps_back_to_source():
    nc, info = 3, letterbox_params(1280, 720, 640)
    pred = _pred(nc, 10)
    # anchor 0: class 1 box centred in the letterboxed image
    pred[:4, 0] = [320, 320, 64, 36]
    pred[4 + 1, 0] = 0.9
    # anchor 1: same box, same class, lower score -> suppressed
    pred[:4, 1] = [321, 320, 64, 36]
    pred[4 + 1, 1] = 0.6
    # anchor 2: same box, different class -> kept (per-class NMS)
    pred[:4, 2] = [320, 320, 64, 36]
    pred[4 + 2, 2] = 0.5
    # anchor 3: below threshold
    pred[4 + 0, 3] = 0.1
    boxes, scores, cls = decode_yolo(pred[None], nc, 0.25, 0.5, 100, info, 1280, 720)
    assert cls.tolist() == [1, 2]
    assert scores.tolist() == pytest.approx([0.9, 0.5])
    assert boxes[0] == pytest.approx([576, 324, 704, 396], abs=0.5)


def test_decode_yolo_transposed_layout():
    nc, info = 2, letterbox_params(640, 640, 640)
    pred = _pred(nc, 5)
    pred[:4, 0] = [100, 100, 20, 20]
    pred[4, 0] = 0.8
    boxes, _, _ = decode_yolo(pred.T, nc, 0.25, 0.5, 10, info, 640, 640)
    assert boxes[0] == pytest.approx([90, 90, 110, 110])


def test_decode_yolo_bad_shape():
    with pytest.raises(ValueError):
        decode_yolo(np.zeros((7, 9)), 2, 0.25, 0.5, 10, letterbox_params(640, 640, 640), 640, 640)
