"""Prometheus metrics, Jetson SoC health, systemd watchdog, and JSON logging."""

from __future__ import annotations

import glob
import json
import logging
import os
import socket
import sys
import time
from typing import Optional

from prometheus_client import Counter, Gauge, Histogram

ZONE_SCORE = Gauge("segi_zone_freshness_score", "Smoothed freshness score 0-100", ["zone"])
ZONE_ITEMS = Gauge("segi_zone_items", "Items (or tiles) observed in zone", ["zone"])
ZONE_SPOILED = Gauge("segi_zone_spoiled_items", "Items below spoiled threshold", ["zone"])
FRAMES = Counter("segi_frames_processed_total", "Frames analysed")
ERRORS = Counter("segi_pipeline_errors_total", "Pipeline iteration errors")
STAGE_LATENCY = Histogram(
    "segi_stage_latency_seconds", "Per-stage processing latency", ["stage"],
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)
CAMERA_UP = Gauge("segi_camera_up", "1 if camera is delivering frames")
CAMERA_FPS = Gauge("segi_camera_fps", "Camera delivery rate")
ALERTS_ACTIVE = Gauge("segi_alerts_active", "Active alerts", ["severity"])
SOC_TEMP = Gauge("segi_soc_temperature_celsius", "Jetson thermal zones", ["zone"])
GPU_LOAD = Gauge("segi_gpu_load_ratio", "GPU utilisation 0-1")


def read_thermals() -> dict[str, float]:
    out: dict[str, float] = {}
    for zone in glob.glob("/sys/devices/virtual/thermal/thermal_zone*"):
        try:
            with open(f"{zone}/type") as fh:
                name = fh.read().strip()
            with open(f"{zone}/temp") as fh:
                out[name] = int(fh.read().strip()) / 1000.0
        except (OSError, ValueError):
            continue
    return out


def read_gpu_load() -> Optional[float]:
    # Orin exposes GPU load (0-1000) under the GPU platform device.
    for path in glob.glob("/sys/devices/platform/**/gpu.0/load", recursive=True) + glob.glob(
        "/sys/devices/platform/bus@0/*.gpu/load"
    ):
        try:
            with open(path) as fh:
                return int(fh.read().strip()) / 1000.0
        except (OSError, ValueError):
            continue
    return None


def update_soc_metrics() -> dict:
    temps = read_thermals()
    for k, v in temps.items():
        SOC_TEMP.labels(zone=k).set(v)
    load = read_gpu_load()
    if load is not None:
        GPU_LOAD.set(load)
    return {"temperatures_c": temps, "gpu_load": load}


class SystemdNotifier:
    """Minimal sd_notify(3) so systemd WatchdogSec= can restart a hung pipeline."""

    def __init__(self) -> None:
        addr = os.environ.get("NOTIFY_SOCKET")
        self._sock: Optional[socket.socket] = None
        self._addr: Optional[str] = None
        if addr and hasattr(socket, "AF_UNIX"):
            self._addr = "\0" + addr[1:] if addr.startswith("@") else addr
            self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self._last = 0.0

    def notify(self, msg: str) -> None:
        if self._sock is not None and self._addr is not None:
            try:
                self._sock.sendto(msg.encode(), self._addr)
            except OSError:
                pass

    def ready(self) -> None:
        self.notify("READY=1")

    def watchdog(self, min_interval_s: float = 5.0) -> None:
        now = time.monotonic()
        if now - self._last >= min_interval_s:
            self._last = now
            self.notify("WATCHDOG=1")

    def stopping(self) -> None:
        self.notify("STOPPING=1")


class JsonFormatter(logging.Formatter):
    _SKIP = set(vars(logging.makeLogRecord({})).keys()) | {"message", "asctime"}

    def format(self, record: logging.LogRecord) -> str:
        d = {
            "ts": round(record.created, 3),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for k, v in record.__dict__.items():
            if k not in self._SKIP:
                d[k] = v
        if record.exc_info:
            d["exc"] = self.formatException(record.exc_info)
        return json.dumps(d, default=str)


def setup_logging(level: str = "INFO", json_logs: bool = True) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if json_logs else logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
