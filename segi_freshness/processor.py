"""Frame processors: the GPU path (CUDA kernels + TensorRT) and a NumPy CPU path.

Both produce the same FrameResult, so everything downstream (scoring, zones,
alerts, API) is backend-agnostic.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import reference
from .config import AppConfig
from .postprocess import decode_yolo

log = logging.getLogger(__name__)


@dataclass
class FrameResult:
    boxes: np.ndarray  # N x 4 xyxy, source pixels
    labels: list[str]  # produce type per box
    det_conf: np.ndarray  # N (1.0 for fixed zone tiles)
    cls_probs: Optional[np.ndarray]  # N x K, or None without a classifier
    color: np.ndarray  # N x 6, see reference.DARK..MEAN_V
    timings_ms: dict[str, float] = field(default_factory=dict)


def _probs(raw: np.ndarray) -> np.ndarray:
    """Accept either softmax outputs (Ultralytics export) or logits."""
    raw = raw.reshape(raw.shape[0], -1)
    looks_normalised = raw.min() >= 0 and np.allclose(raw.sum(axis=1), 1.0, atol=1e-3)
    return raw if looks_normalised else reference.softmax(raw)


class FrameProcessor(ABC):
    backend = "abstract"

    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg

    @property
    @abstractmethod
    def has_detector(self) -> bool: ...

    @property
    @abstractmethod
    def has_classifier(self) -> bool: ...

    @abstractmethod
    def process(self, frame: np.ndarray, rois: Optional[np.ndarray] = None, roi_labels: Optional[list[str]] = None) -> FrameResult:
        """Analyse a BGR/BGRx frame. With rois, skip detection and score those boxes."""

    def close(self) -> None:  # noqa: B027 - optional hook
        pass


class CpuFrameProcessor(FrameProcessor):
    """Reference path for laptops and CI. Colour analysis only, zones mode only."""

    backend = "cpu"

    @property
    def has_detector(self) -> bool:
        return False

    @property
    def has_classifier(self) -> bool:
        return False

    def process(self, frame, rois=None, roi_labels=None) -> FrameResult:
        if rois is None:
            raise RuntimeError("CPU backend supports zones mode only")
        t0 = time.perf_counter()
        color = reference.color_stats(
            frame, reference.shrink_boxes(rois, self.cfg.scoring.roi_shrink), self.cfg.scoring.color_grid, self.cfg.color
        )
        return FrameResult(
            boxes=np.asarray(rois, np.float32),
            labels=list(roi_labels or ["mixed"] * len(rois)),
            det_conf=np.ones(len(rois), np.float32),
            cls_probs=None,
            color=color,
            timings_ms={"color": (time.perf_counter() - t0) * 1e3},
        )


class GpuFrameProcessor(FrameProcessor):
    """One upload per frame; letterbox, crop and colour kernels read the same
    device frame and write straight into TensorRT input bindings."""

    backend = "cuda"

    def __init__(self, cfg: AppConfig) -> None:
        super().__init__(cfg)
        from . import _cuda_ops as ops  # compiled from csrc/ (scripts/build_ext.sh)
        from .gpu import DeviceBuffer, PinnedArray, Stream

        self._ops = ops
        self._DeviceBuffer, self._PinnedArray = DeviceBuffer, PinnedArray
        self.stream = Stream()
        m = cfg.models

        self.thresholds = ops.ColorThresholds()
        for k, v in cfg.color.model_dump().items():
            setattr(self.thresholds, k, float(v))

        self.detector = None
        self.classifier = None
        if cfg.runtime.mode == "detection":
            from .trt_engine import TrtEngine

            assert m.detector_engine is not None
            self.detector = TrtEngine(m.detector_engine, self.stream)
            self._require_fp32(self.detector)
        if m.classifier_engine is not None:
            from .trt_engine import TrtEngine

            if m.classifier_engine.is_file():
                self.classifier = TrtEngine(m.classifier_engine, self.stream, max_batch=m.classifier_max_batch)
                self._require_fp32(self.classifier)
            else:
                log.warning("classifier engine %s missing; using colour analysis only", m.classifier_engine)

        max_boxes = max(m.max_detections, 256)
        self.max_boxes = max_boxes
        self.frame_dev = DeviceBuffer()
        self.frame_host: Optional[PinnedArray] = None
        self.boxes_dev = DeviceBuffer(max_boxes * 4 * 4)
        self.shrunk_dev = DeviceBuffer(max_boxes * 4 * 4)
        self.boxes_host = PinnedArray((max_boxes, 4), np.float32)
        self.shrunk_host = PinnedArray((max_boxes, 4), np.float32)
        self.color_dev = DeviceBuffer(max_boxes * reference.NUM_COLOR_STATS * 4)
        self.color_host = PinnedArray((max_boxes, reference.NUM_COLOR_STATS), np.float32)

    @staticmethod
    def _require_fp32(engine) -> None:
        if engine.input_dtype != np.float32:
            raise ValueError(
                f"{engine.path.name}: input dtype {engine.input_dtype}; build engines with FP32 I/O "
                "(trtexec default) - FP16/INT8 precision inside the network is fine"
            )

    @property
    def has_detector(self) -> bool:
        return self.detector is not None

    @property
    def has_classifier(self) -> bool:
        return self.classifier is not None

    # ------------------------------------------------------------------
    def _upload(self, frame: np.ndarray) -> tuple[int, int, int, int]:
        from .gpu import h2d

        frame = np.ascontiguousarray(frame)
        h, w = frame.shape[:2]
        ch = frame.shape[2] if frame.ndim == 3 else 1
        if ch not in (3, 4):
            raise ValueError(f"expected BGR or BGRx frame, got {frame.shape}")
        if self.frame_host is None or self.frame_host.shape != frame.shape:
            if self.frame_host is not None:
                self.stream.synchronize()
                self.frame_host.free()
            self.frame_host = self._PinnedArray(frame.shape, np.uint8)
        np.copyto(self.frame_host.array, frame)
        self.frame_dev.ensure(frame.nbytes)
        h2d(self.frame_dev.ptr, self.frame_host.ptr, frame.nbytes, self.stream)
        return w, h, w * ch, ch

    def _upload_boxes(self, boxes: np.ndarray, host, dev) -> None:
        from .gpu import h2d

        n = len(boxes)
        host.array[:n] = boxes
        h2d(dev.ptr, host.ptr, n * 16, self.stream)

    def _detect(self, w: int, h: int, pitch: int, ch: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self.detector is not None
        m = self.cfg.models
        info = reference.letterbox_params(w, h, m.detector_input)
        self._ops.letterbox(
            self.frame_dev.ptr, w, h, pitch, ch, self.detector.input_ptr, info.dst, info.dst,
            info.scale, info.pad_x, info.pad_y, info.new_w, info.new_h, reference.LETTERBOX_PAD, self.stream.ptr,
        )
        out = self.detector.run(batch=1)[self.detector.output_names[0]]
        return decode_yolo(
            out, len(m.detector_labels), m.conf_threshold, m.iou_threshold, m.max_detections, info, w, h
        )

    def _classify(self, n: int, w: int, h: int, pitch: int, ch: int) -> np.ndarray:
        assert self.classifier is not None
        m = self.cfg.models
        size = m.classifier_input
        bs = self.classifier.max_batch
        chunks = []
        for start in range(0, n, bs):
            cnt = min(bs, n - start)
            self._ops.crop_resize(
                self.frame_dev.ptr, w, h, pitch, ch, self.boxes_dev.ptr + start * 16, cnt,
                self.classifier.input_ptr, size, tuple(m.classifier_mean), tuple(m.classifier_std), self.stream.ptr,
            )
            out = self.classifier.run(batch=cnt)[self.classifier.output_names[0]]
            chunks.append(_probs(out[:cnt]))
        return np.concatenate(chunks, axis=0)

    def _color(self, n: int, w: int, h: int, pitch: int, ch: int) -> np.ndarray:
        from .gpu import d2h

        self._ops.color_stats(
            self.frame_dev.ptr, w, h, pitch, ch, self.shrunk_dev.ptr, n,
            self.cfg.scoring.color_grid, self.thresholds, self.color_dev.ptr, self.stream.ptr,
        )
        d2h(self.color_host.ptr, self.color_dev.ptr, n * reference.NUM_COLOR_STATS * 4, self.stream)
        self.stream.synchronize()
        return self.color_host.array[:n].copy()

    def process(self, frame, rois=None, roi_labels=None) -> FrameResult:
        timings: dict[str, float] = {}
        t = time.perf_counter()

        def lap(name: str) -> None:
            nonlocal t
            now = time.perf_counter()
            timings[name] = (now - t) * 1e3
            t = now

        w, h, pitch, ch = self._upload(frame)
        lap("upload")

        if rois is None:
            if self.detector is None:
                raise RuntimeError("detection mode requested but no detector engine loaded")
            boxes, conf, cls_ids = self._detect(w, h, pitch, ch)
            labels = [self.cfg.models.detector_labels[i] for i in cls_ids]
            lap("detect")
        else:
            boxes = np.asarray(rois, np.float32).reshape(-1, 4)
            conf = np.ones(len(boxes), np.float32)
            labels = list(roi_labels or ["mixed"] * len(boxes))

        if len(boxes) > self.max_boxes:
            log.warning("truncating %d boxes to %d", len(boxes), self.max_boxes)
            boxes, conf, labels = boxes[: self.max_boxes], conf[: self.max_boxes], labels[: self.max_boxes]
        n = len(boxes)
        if n == 0:
            self.stream.synchronize()
            return FrameResult(np.zeros((0, 4), np.float32), [], np.zeros(0, np.float32), None,
                               np.zeros((0, reference.NUM_COLOR_STATS), np.float32), timings)

        self._upload_boxes(boxes, self.boxes_host, self.boxes_dev)
        self._upload_boxes(reference.shrink_boxes(boxes, self.cfg.scoring.roi_shrink), self.shrunk_host, self.shrunk_dev)

        probs = None
        if self.classifier is not None:
            probs = self._classify(n, w, h, pitch, ch)
            lap("classify")
        color = self._color(n, w, h, pitch, ch)
        lap("color")
        return FrameResult(boxes, labels, np.asarray(conf, np.float32), probs, color, timings)

    def close(self) -> None:
        self.stream.synchronize()
        for eng in (self.detector, self.classifier):
            if eng is not None:
                eng.close()
        for buf in (self.frame_dev, self.boxes_dev, self.shrunk_dev, self.color_dev):
            buf.free()
        for pin in (self.frame_host, self.boxes_host, self.shrunk_host, self.color_host):
            if pin is not None:
                pin.free()
        self.stream.close()


def cuda_available() -> tuple[bool, str]:
    try:
        from . import _cuda_ops  # noqa: F401
        from .gpu import device_name

        return True, device_name(0)
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def create_processor(cfg: AppConfig) -> FrameProcessor:
    backend = cfg.runtime.backend
    if backend in ("auto", "cuda"):
        ok, detail = cuda_available()
        if ok:
            log.info("using CUDA backend on %s", detail)
            return GpuFrameProcessor(cfg)
        if backend == "cuda" or cfg.runtime.mode == "detection":
            raise RuntimeError(f"CUDA backend unavailable: {detail}")
        log.warning("CUDA backend unavailable (%s); falling back to CPU reference path", detail)
    return CpuFrameProcessor(cfg)
