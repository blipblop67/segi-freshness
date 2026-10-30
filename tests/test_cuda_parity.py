"""CUDA kernels vs NumPy reference. Runs only where the extension is built (Jetson / CUDA dev box)."""

import numpy as np
import pytest

from segi_freshness import reference as ref
from segi_freshness.config import ColorThresholdsConfig

ops = pytest.importorskip("segi_freshness._cuda_ops")
gpu = pytest.importorskip("segi_freshness.gpu")


@pytest.fixture(scope="module")
def stream():
    s = gpu.Stream()
    yield s
    s.close()


def _upload(img, stream):
    img = np.ascontiguousarray(img)
    buf = gpu.DeviceBuffer(img.nbytes)
    gpu.h2d(buf.ptr, img.ctypes.data, img.nbytes, stream)
    return buf


def _download(buf, shape, stream):
    out = np.empty(shape, np.float32)
    gpu.d2h(out.ctypes.data, buf.ptr, out.nbytes, stream)
    stream.synchronize()
    return out


@pytest.fixture
def frame():
    rng = np.random.default_rng(1)
    return rng.integers(0, 256, (271, 433, 3), dtype=np.uint8)  # odd sizes on purpose


@pytest.mark.parametrize("channels", [3, 4])
def test_letterbox_parity(frame, stream, channels):
    if channels == 4:
        frame = np.dstack([frame, np.zeros(frame.shape[:2], np.uint8)])
    h, w = frame.shape[:2]
    expected, info = ref.letterbox(frame, 320)
    src = _upload(frame, stream)
    dst = gpu.DeviceBuffer(expected.nbytes)
    ops.letterbox(src.ptr, w, h, w * channels, channels, dst.ptr, 320, 320, info.scale,
                  info.pad_x, info.pad_y, info.new_w, info.new_h, ref.LETTERBOX_PAD, stream.ptr)
    got = _download(dst, expected.shape, stream)
    np.testing.assert_allclose(got, expected, atol=2e-4)


def test_crop_resize_parity(frame, stream):
    h, w = frame.shape[:2]
    boxes = np.array([[0, 0, 100, 80], [50.5, 20.25, 400, 260], [-10, -10, 30, 30]], np.float32)
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    expected = ref.crop_resize(frame, boxes, 64, mean, std)
    src, bdev = _upload(frame, stream), _upload(boxes, stream)
    dst = gpu.DeviceBuffer(expected.nbytes)
    ops.crop_resize(src.ptr, w, h, w * 3, 3, bdev.ptr, len(boxes), dst.ptr, 64, mean, std, stream.ptr)
    np.testing.assert_allclose(_download(dst, expected.shape, stream), expected, atol=1e-3)


def test_color_stats_parity(frame, stream):
    h, w = frame.shape[:2]
    t_cfg = ColorThresholdsConfig()
    t = ops.ColorThresholds()
    for k, v in t_cfg.model_dump().items():
        setattr(t, k, v)
    boxes = np.array([[0, 0, w, h], [10, 10, 200, 150], [300, 100, 500, 400]], np.float32)
    expected = ref.color_stats(frame, boxes, 48, t_cfg)
    src, bdev = _upload(frame, stream), _upload(boxes, stream)
    out = gpu.DeviceBuffer(expected.nbytes)
    ops.color_stats(src.ptr, w, h, w * 3, 3, bdev.ptr, len(boxes), 48, t, out.ptr, stream.ptr)
    # Ratios: allow a couple of samples to land differently on float rounding at bin edges.
    np.testing.assert_allclose(_download(out, expected.shape, stream), expected, atol=2.0 / (48 * 48) + 1e-5)
