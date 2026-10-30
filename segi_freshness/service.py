"""The monitoring service: wires camera -> processor -> scoring -> alerts ->
storage / MQTT / API, and runs the supervised main loop."""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import numpy as np

from . import __version__
from .alerts import Alert, AlertManager, CallbackSink, LogSink, WebhookSink
from .camera import CameraStream
from .config import AppConfig
from .mqtt import create_publisher
from .processor import FrameProcessor, FrameResult, create_processor
from .render import render_snapshot
from .scoring import FreshnessScorer, ZoneTracker
from .storage import Store
from .telemetry import (
    ALERTS_ACTIVE,
    CAMERA_FPS,
    CAMERA_UP,
    ERRORS,
    FRAMES,
    STAGE_LATENCY,
    ZONE_ITEMS,
    ZONE_SCORE,
    ZONE_SPOILED,
    SystemdNotifier,
    update_soc_metrics,
)
from .zones import ZoneMap

log = logging.getLogger(__name__)


class SharedState:
    """Thread-safe snapshot read by the API."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.zones: dict[str, dict] = {}
        self.snapshot_jpeg: Optional[bytes] = None
        self.snapshot_at: Optional[float] = None
        self.last_cycle_at: Optional[float] = None
        self.last_timings_ms: dict[str, float] = {}
        self.soc: dict = {}
        self.camera: dict = {}
        self.started_at = time.time()

    def update(self, **kw) -> None:
        with self._lock:
            for k, v in kw.items():
                setattr(self, k, v)

    def read(self, *names: str):
        with self._lock:
            return tuple(getattr(self, n) for n in names)


class FreshnessService:
    def __init__(self, cfg: AppConfig, processor: Optional[FrameProcessor] = None, store: Optional[Store] = None) -> None:
        self.cfg = cfg
        self.state = SharedState()
        self.store = store or Store(cfg.storage.path)
        self.processor = processor or create_processor(cfg)
        self.zone_map = ZoneMap(cfg.zones)
        self.scorer = FreshnessScorer(cfg.scoring, cfg.models.classifier_labels)
        self.tracker = ZoneTracker(cfg.scoring, [(z.id, z.name, z.produce) for z in cfg.zones])
        self.detection_mode = cfg.runtime.mode == "detection"
        self.mqtt = create_publisher(cfg.mqtt, cfg.store_id, cfg.camera_id)

        sinks: list = [LogSink(), CallbackSink(self._persist_alert)]
        if self.mqtt is not None:
            sinks.append(self.mqtt)
        if cfg.webhook.enabled and cfg.webhook.url:
            sinks.append(WebhookSink(cfg.webhook, cfg.store_id))
        self.alerts = AlertManager(cfg.alerts, cfg.zones, sinks, detection_mode=self.detection_mode)

        self.camera: Optional[CameraStream] = None
        self.notifier = SystemdNotifier()
        self._last_persist = 0.0
        self._last_snapshot = 0.0
        self._last_mqtt = 0.0
        self._last_purge = 0.0
        # Zones mode: per-tile colour stats captured from freshly stocked bins.
        # Only defects above this baseline are penalised, which cancels the bin
        # floor, shadows and lighting that are visible between items.
        self._calib_lock = threading.Lock()
        self.baselines: dict[str, np.ndarray] = self.store.load_baselines()
        self._last_color: dict[str, np.ndarray] = {}
        closed = self.store.close_dangling_alerts(time.time())
        if closed:
            log.info("closed %d alerts left open by previous run", closed)

    # ------------------------------------------------------------ per frame
    def _persist_alert(self, event: str, alert: Alert) -> None:
        self.store.upsert_alert(alert)

    def process_frame(self, frame: np.ndarray, now: Optional[float] = None) -> dict[str, dict]:
        """Analyse one frame and update all downstream state. Returns zone states."""
        now = time.time() if now is None else now
        h, w = frame.shape[:2]

        if self.detection_mode:
            result = self.processor.process(frame)
            owners = self.zone_map.assign(result.boxes, w, h)
        else:
            rois, owners = self.zone_map.tile_rois(w, h)
            produce = {z.id: z.produce for z in self.cfg.zones}
            result = self.processor.process(frame, rois, [produce[o] for o in owners])
            self._apply_baselines(result, owners)

        item_scores = self._score_items(result)
        per_zone: dict[str, tuple[list, list, list]] = {z.id: ([], [], []) for z in self.cfg.zones}
        for i, zid in enumerate(owners):
            if zid is None:
                continue
            x1, y1, x2, y2 = result.boxes[i]
            s = item_scores[i]
            per_zone[zid][0].append(s.score)
            per_zone[zid][1].append(float((x2 - x1) * (y2 - y1)))
            per_zone[zid][2].append({
                "label": result.labels[i],
                "box": [round(float(v), 1) for v in result.boxes[i]],
                "score": round(s.score, 1),
                "color_score": round(s.color_score, 1),
                "class": s.top_class,
                "class_prob": None if s.top_prob is None else round(s.top_prob, 3),
                "defects": {k: round(float(result.color[i, j]), 3) for j, k in enumerate(("dark", "brown", "mold", "yellow"))},
            })

        zones: dict[str, dict] = {}
        for zid, (scores, areas, items) in per_zone.items():
            st = self.tracker.update(zid, scores, areas, items, now)
            self.alerts.evaluate_zone(st, now)
            zones[zid] = st.to_dict()
            zones[zid]["calibrated"] = self.detection_mode or zid in self.baselines
            if st.ema is not None:
                ZONE_SCORE.labels(zone=zid).set(st.ema)
            ZONE_ITEMS.labels(zone=zid).set(st.n_items)
            ZONE_SPOILED.labels(zone=zid).set(st.n_spoiled)

        FRAMES.inc()
        for stage, ms in result.timings_ms.items():
            STAGE_LATENCY.labels(stage=stage).observe(ms / 1e3)
        self.state.update(zones=zones, last_cycle_at=now, last_timings_ms=result.timings_ms)

        if now - self._last_snapshot >= self.cfg.runtime.snapshot_interval_s:
            self._last_snapshot = now
            jpeg = render_snapshot(frame, self.zone_map, zones, result.boxes, [s.score for s in item_scores], result.labels)
            self.state.update(snapshot_jpeg=jpeg, snapshot_at=now)

        self._periodic(now, zones)
        return zones

    def _apply_baselines(self, result: FrameResult, owners: list) -> None:
        idx: dict[str, list[int]] = {}
        for i, zid in enumerate(owners):
            idx.setdefault(zid, []).append(i)
        with self._calib_lock:
            for zid, rows in idx.items():
                self._last_color[zid] = result.color[rows].copy()
                base = self.baselines.get(zid)
                if base is not None and base.shape == (len(rows), result.color.shape[1]):
                    result.color[rows, :4] = np.clip(result.color[rows, :4] - base[:, :4], 0.0, 1.0)

    def calibrate(self, zone_ids: Optional[list[str]] = None) -> list[str]:
        """Record the current frame's defect ratios as the 'fresh' baseline."""
        if self.detection_mode:
            raise ValueError("calibration applies to zones mode only")
        ids = zone_ids or [z.id for z in self.cfg.zones]
        with self._calib_lock:
            missing = [z for z in ids if z not in self._last_color]
            if missing:
                raise ValueError(f"no analysed frame yet for zones {missing}")
            for zid in ids:
                self.baselines[zid] = self._last_color[zid].copy()
                self.store.save_baseline(zid, self.baselines[zid])
        # Restart smoothing so the new baseline takes effect immediately.
        for zid in ids:
            self.tracker.states[zid].ema = None
        log.info("calibrated baselines for zones %s", ids)
        return ids

    def clear_calibration(self, zone_id: str) -> None:
        with self._calib_lock:
            self.baselines.pop(zone_id, None)
            self.store.delete_baseline(zone_id)

    def _score_items(self, result: FrameResult):
        out = []
        for i in range(len(result.boxes)):
            probs = None if result.cls_probs is None else result.cls_probs[i]
            out.append(self.scorer.score(result.labels[i], probs, result.color[i]))
        return out

    def _periodic(self, now: float, zones: dict[str, dict]) -> None:
        if now - self._last_persist >= self.cfg.storage.persist_interval_s:
            self._last_persist = now
            self.store.insert_readings(list(self.tracker.states.values()), now)
        if self.mqtt is not None and now - self._last_mqtt >= self.cfg.mqtt.publish_interval_s:
            self._last_mqtt = now
            for z in zones.values():
                self.mqtt.publish_zone(z)
        if now - self._last_purge >= 6 * 3600:
            self._last_purge = now
            removed = self.store.purge(now - self.cfg.storage.retention_days * 86400)
            if removed:
                log.info("retention purge removed %d rows", removed)
        active = self.alerts.active()
        for sev in ("info", "warning", "critical"):
            ALERTS_ACTIVE.labels(severity=sev).set(sum(a.severity == sev for a in active))

    # ------------------------------------------------------------ main loop
    def run(self, stop: threading.Event) -> None:
        self.camera = CameraStream(self.cfg.camera)
        self.camera.start()
        self.notifier.ready()
        log.info("segi-freshness %s running: store=%s mode=%s backend=%s", __version__,
                 self.cfg.store_id, self.cfg.runtime.mode, self.processor.backend)

        period = 1.0 / self.cfg.runtime.inference_fps
        last_seq = -1
        errors = 0
        last_soc = 0.0
        first_frame_at: Optional[float] = None
        # Demo convenience only; in stores, staff calibrate after restocking.
        auto_done = self.detection_mode or bool(self.baselines)
        while not stop.is_set():
            cycle_start = time.monotonic()
            now = time.time()
            frame, seq, ts = self.camera.latest()
            cam = self.camera.status()
            CAMERA_UP.set(1 if cam.connected and ts and now - ts < 5 else 0)
            CAMERA_FPS.set(cam.fps)
            # Seconds since the last good frame (or since start-up if none yet).
            self.alerts.evaluate_camera(now - (ts or self.state.started_at), now)

            if now - last_soc >= 5:
                last_soc = now
                self.state.update(soc=update_soc_metrics())
            self.state.update(camera=cam.__dict__)

            if frame is not None and seq != last_seq:
                last_seq = seq
                try:
                    self.process_frame(frame, now)
                    errors = 0
                    auto = self.cfg.runtime.auto_calibrate_after_s
                    if auto is not None and first_frame_at is None:
                        first_frame_at = now
                    if auto is not None and not auto_done and now - first_frame_at >= auto:
                        auto_done = True
                        self.calibrate()
                except Exception:
                    errors += 1
                    ERRORS.inc()
                    log.exception("pipeline iteration failed (%d consecutive)", errors)
                    if errors >= self.cfg.runtime.max_consecutive_errors:
                        raise RuntimeError("too many consecutive pipeline errors; exiting for supervisor restart") from None

            self.notifier.watchdog()
            stop.wait(max(0.0, period - (time.monotonic() - cycle_start)))

    def close(self) -> None:
        self.notifier.stopping()
        if self.camera is not None:
            self.camera.stop()
            self.camera.join(timeout=5)
        self.store.insert_readings(list(self.tracker.states.values()))
        if self.mqtt is not None:
            self.mqtt.stop()
        self.processor.close()
        self.store.close()
