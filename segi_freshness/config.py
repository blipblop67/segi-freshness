"""Typed configuration, loaded from YAML with secrets taken from the environment."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RuntimeConfig(_Model):
    # "zones": score fixed display bins (no detector needed).
    # "detection": detect every produce item with a TensorRT YOLO engine.
    mode: Literal["zones", "detection"] = "zones"
    backend: Literal["auto", "cuda", "cpu"] = "auto"
    inference_fps: float = Field(2.0, gt=0, le=30)
    snapshot_interval_s: float = Field(2.0, gt=0)
    max_consecutive_errors: int = Field(20, ge=1)
    # Demos only: calibrate all bins automatically this many seconds after the first
    # frame (skipped if baselines already exist). Leave unset in stores.
    auto_calibrate_after_s: Optional[float] = Field(None, ge=0)
    log_level: str = "INFO"
    log_json: bool = True


class CameraConfig(_Model):
    source: Literal["csi", "usb", "rtsp", "file"] = "file"
    uri: str = ""  # /dev/videoN, rtsp://..., or a video file path
    sensor_id: int = 0
    width: int = 1920
    height: int = 1080
    fps: int = 30
    loop_file: bool = True
    pipeline: Optional[str] = None  # raw GStreamer pipeline override
    # CCTV / NVR over RTSP. Most IP cameras send H.264 on the sub-stream and
    # H.264 or H.265 on the main stream; check the camera's encoding settings.
    rtsp_codec: Literal["h264", "h265"] = "h264"
    rtsp_transport: Literal["tcp", "udp"] = "tcp"  # tcp survives busy store Wi-Fi/LAN
    rtsp_latency_ms: int = Field(200, ge=0, le=5000)
    # auto: Jetson NVDEC when running on L4T, else OpenCV/FFmpeg software decoding.
    decoder: Literal["auto", "nvidia", "software"] = "auto"
    reconnect_initial_s: float = 1.0
    reconnect_max_s: float = 30.0

    @model_validator(mode="after")
    def _uri_required(self) -> CameraConfig:
        if self.source in ("rtsp", "file") and not self.uri and not self.pipeline:
            raise ValueError(f"camera.uri is required for source={self.source}")
        if self.source == "rtsp" and not self.pipeline and not self.uri.startswith(("rtsp://", "rtsps://")):
            raise ValueError("camera.uri must start with rtsp:// for source=rtsp")
        return self


class ModelConfig(_Model):
    detector_engine: Optional[Path] = None
    detector_labels: list[str] = Field(default_factory=list)
    detector_input: int = 640
    conf_threshold: float = Field(0.35, ge=0, le=1)
    iou_threshold: float = Field(0.5, ge=0, le=1)
    max_detections: int = Field(100, ge=1, le=1000)

    classifier_engine: Optional[Path] = None
    classifier_labels: list[str] = Field(
        default_factory=lambda: ["fresh", "ripe", "overripe", "spoiled"]
    )
    classifier_input: int = 224
    classifier_max_batch: int = Field(32, ge=1)
    classifier_mean: tuple[float, float, float] = (0.0, 0.0, 0.0)
    classifier_std: tuple[float, float, float] = (1.0, 1.0, 1.0)


class ColorThresholdsConfig(_Model):
    dark_v_max: float = 0.20
    brown_h_lo: float = 8.0
    brown_h_hi: float = 35.0
    brown_s_min: float = 0.25
    brown_v_max: float = 0.55
    mold_s_max: float = 0.18
    mold_v_min: float = 0.65
    yellow_h_lo: float = 35.0
    yellow_h_hi: float = 65.0
    yellow_s_min: float = 0.30


class ProduceProfile(_Model):
    """Penalty weights applied to each colour-defect ratio for a produce type."""

    dark: float = 1.5
    brown: float = 2.0
    mold: float = 1.0
    yellow: float = 0.0


class GradeBands(_Model):
    a: float = 80.0
    b: float = 60.0
    c: float = 40.0


class ScoringConfig(_Model):
    classifier_weight: float = Field(0.65, ge=0, le=1)
    class_scores: dict[str, float] = Field(
        default_factory=lambda: {"fresh": 100.0, "ripe": 80.0, "overripe": 45.0, "spoiled": 5.0}
    )
    ema_alpha: float = Field(0.2, gt=0, le=1)
    spoiled_item_score: float = 30.0
    roi_shrink: float = Field(0.08, ge=0, lt=0.5)
    color_grid: int = Field(48, ge=4, le=256)
    grades: GradeBands = Field(default_factory=GradeBands)
    profiles: dict[str, ProduceProfile] = Field(default_factory=lambda: {"default": ProduceProfile()})
    produce_profile_map: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_profiles(self) -> ScoringConfig:
        self.profiles.setdefault("default", ProduceProfile())
        missing = set(self.produce_profile_map.values()) - set(self.profiles)
        if missing:
            raise ValueError(f"produce_profile_map references unknown profiles: {sorted(missing)}")
        return self


class ZoneConfig(_Model):
    id: str = Field(pattern=r"^[A-Za-z0-9_\-]+$")
    name: str
    produce: str = "mixed"
    # Normalised [0, 1] image coordinates. Give either rect or polygon.
    rect: Optional[tuple[float, float, float, float]] = None
    polygon: Optional[list[tuple[float, float]]] = None
    # Zones mode only: split the bin into cols x rows tiles so one rotting item
    # shows up as a bad tile instead of being averaged away.
    grid: tuple[int, int] = (1, 1)
    low_threshold: Optional[float] = None

    @model_validator(mode="after")
    def _geometry(self) -> ZoneConfig:
        if self.polygon is None:
            if self.rect is None:
                raise ValueError(f"zone {self.id}: either rect or polygon is required")
            x1, y1, x2, y2 = self.rect
            if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                raise ValueError(f"zone {self.id}: rect must be normalised x1<x2, y1<y2 in [0,1]")
            self.polygon = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        if len(self.polygon) < 3:
            raise ValueError(f"zone {self.id}: polygon needs at least 3 points")
        if any(not (0 <= x <= 1 and 0 <= y <= 1) for x, y in self.polygon):
            raise ValueError(f"zone {self.id}: polygon points must be normalised to [0,1]")
        if self.grid[0] < 1 or self.grid[1] < 1:
            raise ValueError(f"zone {self.id}: grid must be >= 1x1")
        return self


class AlertConfig(_Model):
    low_threshold: float = 55.0
    critical_threshold: float = 35.0
    hysteresis: float = 5.0
    trigger_samples: int = Field(5, ge=1)
    clear_samples: int = Field(5, ge=1)
    cooldown_s: float = 900.0
    spoiled_trigger_samples: int = Field(5, ge=1)
    restock_trigger_samples: int = Field(20, ge=1)
    camera_offline_s: float = 30.0


class MqttConfig(_Model):
    enabled: bool = False
    host: str = "localhost"
    port: int = 1883
    username: Optional[str] = None
    password: Optional[str] = None
    tls: bool = False
    topic_prefix: str = "segi"
    qos: int = Field(1, ge=0, le=2)
    keepalive_s: int = 60
    publish_interval_s: float = 10.0


class WebhookConfig(_Model):
    enabled: bool = False
    url: str = ""
    timeout_s: float = 5.0


class ApiConfig(_Model):
    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 8080
    api_key: Optional[str] = None


class StorageConfig(_Model):
    path: Path = Path("data/freshness.db")
    persist_interval_s: float = 30.0
    retention_days: int = Field(30, ge=1)


class AppConfig(_Model):
    store_id: str = Field(pattern=r"^[A-Za-z0-9_\-]+$")
    store_name: str = ""
    camera_id: str = Field("cam01", pattern=r"^[A-Za-z0-9_\-]+$")
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    camera: CameraConfig = Field(default_factory=CameraConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    color: ColorThresholdsConfig = Field(default_factory=ColorThresholdsConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    zones: list[ZoneConfig]
    alerts: AlertConfig = Field(default_factory=AlertConfig)
    mqtt: MqttConfig = Field(default_factory=MqttConfig)
    webhook: WebhookConfig = Field(default_factory=WebhookConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)

    @field_validator("zones")
    @classmethod
    def _unique_zones(cls, zones: list[ZoneConfig]) -> list[ZoneConfig]:
        if not zones:
            raise ValueError("at least one zone is required")
        ids = [z.id for z in zones]
        if len(ids) != len(set(ids)):
            raise ValueError("zone ids must be unique")
        return zones

    @model_validator(mode="after")
    def _mode_requirements(self) -> AppConfig:
        if self.runtime.mode == "detection":
            if self.models.detector_engine is None:
                raise ValueError("runtime.mode=detection requires models.detector_engine")
            if not self.models.detector_labels:
                raise ValueError("runtime.mode=detection requires models.detector_labels")
            if self.runtime.backend == "cpu":
                raise ValueError("detection mode needs TensorRT; use backend auto or cuda")
        missing = set(self.models.classifier_labels) - set(self.scoring.class_scores)
        if missing:
            raise ValueError(f"scoring.class_scores missing classifier labels: {sorted(missing)}")
        return self


ENV_OVERRIDES = {
    "SEGI_API_KEY": ("api", "api_key"),
    "SEGI_MQTT_USERNAME": ("mqtt", "username"),
    "SEGI_MQTT_PASSWORD": ("mqtt", "password"),
    "SEGI_WEBHOOK_URL": ("webhook", "url"),
    "SEGI_CAMERA_URI": ("camera", "uri"),
}


def _merge(base: dict, override: dict) -> dict:
    """Recursive dict merge; lists (e.g. zones) and scalars are replaced."""
    out = dict(base)
    for k, v in override.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _read_yaml(path: Path, depth: int = 0) -> dict:
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    parent = data.pop("extends", None)
    if parent:
        if depth >= 5:
            raise ValueError("config 'extends' chain too deep")
        data = _merge(_read_yaml(path.parent / parent, depth + 1), data)
    return data


def load_config(path: str | Path) -> AppConfig:
    """Load YAML (optionally `extends: other.yaml`, relative to the file), then env secrets."""
    data = _read_yaml(Path(path))
    for env, (section, key) in ENV_OVERRIDES.items():
        value = os.environ.get(env)
        if value:
            data.setdefault(section, {})[key] = value
    return AppConfig.model_validate(data)
