"""NumPy reference implementations of the CUDA kernels in csrc/.

They serve three purposes:
  * the CPU backend (laptop demos, CI) runs on them,
  * tests/test_cuda_parity.py checks the CUDA kernels against them on device,
  * training/make_crops.py uses crop_resize so training crops match inference.

Every formula here must stay identical to the corresponding .cu code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ColorThresholdsConfig

DARK, BROWN, MOLD, YELLOW, MEAN_S, MEAN_V = range(6)
NUM_COLOR_STATS = 6
LETTERBOX_PAD = 114.0 / 255.0


@dataclass(frozen=True)
class LetterboxInfo:
    scale: float
    pad_x: int
    pad_y: int
    new_w: int
    new_h: int
    dst: int

    def to_source(self, boxes: np.ndarray) -> np.ndarray:
        """Map xyxy boxes from network-input pixels back to source pixels."""
        out = boxes.astype(np.float32, copy=True)
        out[:, [0, 2]] = (out[:, [0, 2]] - self.pad_x) / self.scale
        out[:, [1, 3]] = (out[:, [1, 3]] - self.pad_y) / self.scale
        return out


def letterbox_params(src_w: int, src_h: int, dst: int) -> LetterboxInfo:
    scale = min(dst / src_w, dst / src_h)
    new_w = int(round(src_w * scale))
    new_h = int(round(src_h * scale))
    return LetterboxInfo(
        scale=float(scale),
        pad_x=(dst - new_w) // 2,
        pad_y=(dst - new_h) // 2,
        new_w=new_w,
        new_h=new_h,
        dst=dst,
    )


def _bilinear(img: np.ndarray, fx: np.ndarray, fy: np.ndarray) -> np.ndarray:
    """Bilinear sample with edge clamping; returns float32 BGR in [0, 255]."""
    h, w = img.shape[:2]
    fx = np.clip(fx.astype(np.float32), 0.0, np.float32(w - 1))
    fy = np.clip(fy.astype(np.float32), 0.0, np.float32(h - 1))
    x0 = np.floor(fx).astype(np.int32)
    y0 = np.floor(fy).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    ax = (fx - x0)[..., None]
    ay = (fy - y0)[..., None]
    f = img[..., :3].astype(np.float32)
    top = f[y0, x0] * (1.0 - ax) + f[y0, x1] * ax
    bot = f[y1, x0] * (1.0 - ax) + f[y1, x1] * ax
    return top * (1.0 - ay) + bot * ay


def letterbox(img: np.ndarray, dst: int, pad_value: float = LETTERBOX_PAD) -> tuple[np.ndarray, LetterboxInfo]:
    """BGR(x) HWC uint8 -> RGB CHW float32 in [0, 1], letterboxed to dst x dst."""
    h, w = img.shape[:2]
    info = letterbox_params(w, h, dst)
    out = np.full((dst, dst, 3), pad_value, dtype=np.float32)
    ys, xs = np.mgrid[0 : info.new_h, 0 : info.new_w].astype(np.float32)
    fx = (xs + np.float32(0.5)) / np.float32(info.scale) - np.float32(0.5)
    fy = (ys + np.float32(0.5)) / np.float32(info.scale) - np.float32(0.5)
    bgr = _bilinear(img, fx, fy) / np.float32(255.0)
    out[info.pad_y : info.pad_y + info.new_h, info.pad_x : info.pad_x + info.new_w] = bgr[..., ::-1]
    return np.ascontiguousarray(out.transpose(2, 0, 1)), info


def crop_resize(
    img: np.ndarray,
    boxes: np.ndarray,
    size: int,
    mean: tuple[float, float, float] = (0.0, 0.0, 0.0),
    std: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Stretch-resize each xyxy box to size x size. Returns N x 3 x S x S RGB float32."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    out = np.empty((len(boxes), 3, size, size), dtype=np.float32)
    o = np.arange(size, dtype=np.float32) + np.float32(0.5)
    mean_a = np.asarray(mean, dtype=np.float32)
    inv_std = np.float32(1.0) / np.asarray(std, dtype=np.float32)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        bw = max(x2 - x1, np.float32(1.0))
        bh = max(y2 - y1, np.float32(1.0))
        fx = x1 + o * bw / np.float32(size) - np.float32(0.5)
        fy = y1 + o * bh / np.float32(size) - np.float32(0.5)
        gx, gy = np.meshgrid(fx, fy)
        rgb = _bilinear(img, gx, gy)[..., ::-1] / np.float32(255.0)
        out[i] = ((rgb - mean_a) * inv_std).transpose(2, 0, 1)
    return out


def _hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = np.maximum(r, np.maximum(g, b))
    mn = np.minimum(r, np.minimum(g, b))
    d = mx - mn
    v = mx
    s = np.where(mx > 0, d / np.where(mx > 0, mx, 1), 0).astype(np.float32)
    safe_d = np.where(d > 0, d, 1)
    h = np.where(
        mx == r,
        60.0 * ((g - b) / safe_d),
        np.where(mx == g, 60.0 * ((b - r) / safe_d + 2.0), 60.0 * ((r - g) / safe_d + 4.0)),
    )
    h = np.where(d > 0, h, 0.0)
    h = np.where(h < 0, h + 360.0, h).astype(np.float32)
    return h, s, v


def color_stats(
    img: np.ndarray, boxes: np.ndarray, grid: int, t: ColorThresholdsConfig
) -> np.ndarray:
    """Per-box colour-defect ratios. Returns N x 6 float32 (see DARK..MEAN_V)."""
    h_img, w_img = img.shape[:2]
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    out = np.zeros((len(boxes), NUM_COLOR_STATS), dtype=np.float32)
    g = np.arange(grid, dtype=np.float32) + np.float32(0.5)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        x1 = max(x1, np.float32(0.0))
        y1 = max(y1, np.float32(0.0))
        x2 = min(x2, np.float32(w_img))
        y2 = min(y2, np.float32(h_img))
        bw = max(x2 - x1, np.float32(1.0))
        bh = max(y2 - y1, np.float32(1.0))
        px = np.clip((x1 + g * bw / np.float32(grid)).astype(np.int32), 0, w_img - 1)
        py = np.clip((y1 + g * bh / np.float32(grid)).astype(np.int32), 0, h_img - 1)
        samples = img[py[:, None], px[None, :], :3].astype(np.float32) / np.float32(255.0)
        h, s, v = _hsv(samples[..., ::-1])

        dark = v < t.dark_v_max
        brown = (
            ~dark
            & (h >= t.brown_h_lo)
            & (h <= t.brown_h_hi)
            & (s >= t.brown_s_min)
            & (v <= t.brown_v_max)
        )
        mold = ~dark & (s <= t.mold_s_max) & (v >= t.mold_v_min)
        yellow = ~dark & ~brown & (h >= t.yellow_h_lo) & (h <= t.yellow_h_hi) & (s >= t.yellow_s_min)

        out[i, DARK] = dark.mean()
        out[i, BROWN] = brown.mean()
        out[i, MOLD] = mold.mean()
        out[i, YELLOW] = yellow.mean()
        out[i, MEAN_S] = s.mean()
        out[i, MEAN_V] = v.mean()
    return out


def shrink_boxes(boxes: np.ndarray, ratio: float) -> np.ndarray:
    """Shrink xyxy boxes towards their centre to avoid bin edges / background."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
    if ratio <= 0:
        return boxes
    dw = (boxes[:, 2] - boxes[:, 0]) * ratio
    dh = (boxes[:, 3] - boxes[:, 1]) * ratio
    boxes[:, 0] += dw
    boxes[:, 2] -= dw
    boxes[:, 1] += dh
    boxes[:, 3] -= dh
    return boxes


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)
