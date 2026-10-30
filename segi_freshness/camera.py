"""Camera capture thread with hardware-accelerated GStreamer pipelines and
automatic reconnection. Always holds only the most recent frame."""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .config import CameraConfig

log = logging.getLogger(__name__)

_APPSINK = "appsink drop=true max-buffers=1 sync=false"
_TO_BGR = "nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR"


def build_pipeline(cfg: CameraConfig) -> str:
    """GStreamer pipeline using the Jetson ISP/VIC (nvarguscamerasrc, nvvidconv)
    and NVDEC (nvv4l2decoder) so the CPU never decodes video."""
    if cfg.pipeline:
        return cfg.pipeline
    caps = f"width={cfg.width},height={cfg.height},framerate={cfg.fps}/1"
    if cfg.source == "csi":
        return (
            f"nvarguscamerasrc sensor-id={cfg.sensor_id} ! video/x-raw(memory:NVMM),{caps} ! "
            f"{_TO_BGR} ! {_APPSINK}"
        )
    if cfg.source == "usb":
        dev = cfg.uri or "/dev/video0"
        return (
            f"v4l2src device={dev} ! image/jpeg,{caps} ! jpegparse ! nvv4l2decoder mjpeg=1 ! "
            f"{_TO_BGR} ! {_APPSINK}"
        )
    if cfg.source == "rtsp":
        # CCTV / NVR stream. rtspsrc errors out after `timeout` us without data, which
        # surfaces as a failed read and triggers the reconnect loop.
        codec = cfg.rtsp_codec
        return (
            f"rtspsrc location={cfg.uri} latency={cfg.rtsp_latency_ms} protocols={cfg.rtsp_transport} "
            f"timeout=5000000 tcp-timeout=10000000 drop-on-latency=true ! "
            f"rtp{codec}depay ! {codec}parse ! nvv4l2decoder ! {_TO_BGR} ! {_APPSINK}"
        )
    raise ValueError(f"no GStreamer pipeline for source {cfg.source!r}")


def use_hw_decoder(cfg: CameraConfig) -> bool:
    """Jetson NVDEC via GStreamer, or OpenCV/FFmpeg software decode (laptops, CI)."""
    if cfg.decoder == "auto":
        return os.path.exists("/etc/nv_tegra_release")
    return cfg.decoder == "nvidia"


def redact_uri(uri: str) -> str:
    """rtsp://user:secret@host/... -> rtsp://user:***@host/... for logs and the API."""
    return re.sub(r"(//[^:/@]+):[^@/]*@", r"\1:***@", uri)


@dataclass
class CameraStatus:
    connected: bool
    last_frame_at: Optional[float]
    fps: float
    reconnects: int
    frames: int
    error: Optional[str]


class CameraStream(threading.Thread):
    def __init__(self, cfg: CameraConfig) -> None:
        super().__init__(name="camera", daemon=True)
        self.cfg = cfg
        self._halt = threading.Event()
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._seq = 0
        self._ts: Optional[float] = None
        self._connected = False
        self._reconnects = 0
        self._fps = 0.0
        self._error: Optional[str] = None

    def latest(self) -> tuple[Optional[np.ndarray], int, Optional[float]]:
        with self._lock:
            return self._frame, self._seq, self._ts

    def status(self) -> CameraStatus:
        with self._lock:
            return CameraStatus(self._connected, self._ts, round(self._fps, 1), self._reconnects, self._seq, self._error)

    def stop(self) -> None:
        self._halt.set()

    def _open(self) -> cv2.VideoCapture:
        if self.cfg.source == "file":
            cap = cv2.VideoCapture(self.cfg.uri)
        elif self.cfg.source == "rtsp" and not use_hw_decoder(self.cfg):
            os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", f"rtsp_transport;{self.cfg.rtsp_transport}")
            cap = cv2.VideoCapture(
                self.cfg.uri, cv2.CAP_FFMPEG,
                [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000],
            )
        else:
            cap = cv2.VideoCapture(build_pipeline(self.cfg), cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            cap.release()
            where = redact_uri(self.cfg.uri) if self.cfg.uri else f"sensor {self.cfg.sensor_id}"
            raise RuntimeError(f"cannot open camera source {self.cfg.source}: {where}")
        return cap

    def run(self) -> None:
        backoff = self.cfg.reconnect_initial_s
        while not self._halt.is_set():
            try:
                cap = self._open()
            except Exception as exc:  # noqa: BLE001
                self._set_error(str(exc))
                log.error("%s; retrying in %.0fs", exc, backoff)
                self._halt.wait(backoff)
                backoff = min(backoff * 2, self.cfg.reconnect_max_s)
                continue

            log.info("camera connected (%s)", self.cfg.source)
            backoff = self.cfg.reconnect_initial_s
            file_period = 0.0
            if self.cfg.source == "file":
                src_fps = cap.get(cv2.CAP_PROP_FPS) or self.cfg.fps
                file_period = 1.0 / max(src_fps, 1.0)
            with self._lock:
                self._connected, self._error = True, None
            try:
                self._read_loop(cap, file_period)
            finally:
                cap.release()
                with self._lock:
                    self._connected = False
            if not self._halt.is_set():
                self._reconnects += 1
                self._halt.wait(backoff)

    def _read_loop(self, cap: cv2.VideoCapture, file_period: float) -> None:
        last = time.monotonic()
        while not self._halt.is_set():
            ok, frame = cap.read()
            if not ok or frame is None:
                if self.cfg.source == "file" and self.cfg.loop_file:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                self._set_error("frame read failed")
                log.warning("camera read failed; reconnecting")
                return
            now = time.monotonic()
            dt = now - last
            last = now
            with self._lock:
                self._frame = frame
                self._seq += 1
                self._ts = time.time()
                if dt > 0:
                    self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt) if self._fps else 1.0 / dt
            if file_period:  # emulate a live camera when replaying a file
                self._halt.wait(max(0.0, file_period - (time.monotonic() - now)))

    def _set_error(self, msg: str) -> None:
        with self._lock:
            self._error = msg
