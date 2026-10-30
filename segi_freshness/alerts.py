"""Alert state machine with debouncing, hysteresis, escalation and cooldown."""

from __future__ import annotations

import json
import logging
import queue
import threading
import urllib.request
import uuid
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Optional, Protocol

from .config import AlertConfig, WebhookConfig, ZoneConfig
from .scoring import ZoneState

log = logging.getLogger(__name__)

SEVERITY_RANK = {"info": 0, "warning": 1, "critical": 2}


@dataclass
class Alert:
    id: str
    zone_id: str
    type: str  # low_freshness | spoiled_items | restock | camera_offline
    severity: str
    message: str
    value: Optional[float]
    raised_at: float
    cleared_at: Optional[float] = None
    acked: bool = False
    acked_by: Optional[str] = None

    @property
    def active(self) -> bool:
        return self.cleared_at is None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["active"] = self.active
        return d


class AlertSink(Protocol):
    def emit(self, event: str, alert: Alert) -> None: ...


@dataclass
class _Condition:
    bad: int = 0
    good: int = 0
    active: Optional[Alert] = None
    last_cleared_at: float = float("-inf")


class AlertManager:
    def __init__(
        self,
        cfg: AlertConfig,
        zones: Sequence[ZoneConfig],
        sinks: Sequence[AlertSink] = (),
        detection_mode: bool = False,
    ) -> None:
        self.cfg = cfg
        self.zones = {z.id: z for z in zones}
        self.sinks = list(sinks)
        self.detection_mode = detection_mode
        self._conds: dict[tuple[str, str], _Condition] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ public
    def evaluate_zone(self, st: ZoneState, now: float) -> None:
        zone = self.zones[st.zone_id]
        low = zone.low_threshold if zone.low_threshold is not None else self.cfg.low_threshold
        crit = min(self.cfg.critical_threshold, low)

        if st.ema is not None and st.n_items > 0:
            severity = "critical" if st.ema < crit else "warning"
            self._step(
                (st.zone_id, "low_freshness"),
                bad=st.ema < low,
                good=st.ema >= low + self.cfg.hysteresis,
                trigger=self.cfg.trigger_samples,
                clear=self.cfg.clear_samples,
                now=now,
                severity=severity,
                value=st.ema,
                message=f"{zone.name}: freshness {st.ema:.0f}/100 below {low:.0f} - inspect and rotate stock",
            )

        self._step(
            (st.zone_id, "spoiled_items"),
            bad=st.n_spoiled > 0,
            good=st.n_spoiled == 0,
            trigger=self.cfg.spoiled_trigger_samples,
            clear=self.cfg.clear_samples,
            now=now,
            severity="warning",
            value=float(st.n_spoiled),
            message=f"{zone.name}: {st.n_spoiled} spoiled item(s)/area(s) detected - remove from display",
        )

        if self.detection_mode:
            self._step(
                (st.zone_id, "restock"),
                bad=st.n_items == 0,
                good=st.n_items > 0,
                trigger=self.cfg.restock_trigger_samples,
                clear=1,
                now=now,
                severity="info",
                value=0.0,
                message=f"{zone.name}: bin appears empty - restock",
            )

    def evaluate_camera(self, offline_for_s: Optional[float], now: float) -> None:
        offline = offline_for_s is not None and offline_for_s >= self.cfg.camera_offline_s
        self._step(
            ("_camera", "camera_offline"),
            bad=offline,
            good=not offline,
            trigger=1,
            clear=1,
            now=now,
            severity="critical",
            value=offline_for_s,
            message="Camera offline - freshness monitoring paused",
            cooldown=0.0,
        )

    def active(self) -> list[Alert]:
        with self._lock:
            return [c.active for c in self._conds.values() if c.active is not None]

    def ack(self, alert_id: str, by: str = "dashboard") -> Optional[Alert]:
        with self._lock:
            for c in self._conds.values():
                if c.active is not None and c.active.id == alert_id:
                    c.active.acked = True
                    c.active.acked_by = by
                    alert = c.active
                    break
            else:
                return None
        self._emit("acked", alert)
        return alert

    # ---------------------------------------------------------------- internal
    def _step(
        self,
        key: tuple[str, str],
        *,
        bad: bool,
        good: bool,
        trigger: int,
        clear: int,
        now: float,
        severity: str,
        value: Optional[float],
        message: str,
        cooldown: Optional[float] = None,
    ) -> None:
        cooldown = self.cfg.cooldown_s if cooldown is None else cooldown
        events: list[tuple[str, Alert]] = []
        with self._lock:
            c = self._conds.setdefault(key, _Condition())
            if bad:
                c.bad += 1
                c.good = 0
            elif good:
                c.good += 1
                c.bad = 0
            else:  # inside the hysteresis band: hold state
                c.bad = 0
                c.good = 0

            if c.active is None:
                if c.bad >= trigger and now - c.last_cleared_at >= cooldown:
                    c.active = Alert(
                        id=uuid.uuid4().hex,
                        zone_id=key[0],
                        type=key[1],
                        severity=severity,
                        message=message,
                        value=value,
                        raised_at=now,
                    )
                    events.append(("raised", c.active))
            else:
                if c.good >= clear:
                    c.active.cleared_at = now
                    c.last_cleared_at = now
                    events.append(("cleared", c.active))
                    c.active = None
                elif bad:
                    c.active.value = value
                    c.active.message = message
                    if SEVERITY_RANK[severity] > SEVERITY_RANK[c.active.severity]:
                        c.active.severity = severity
                        c.active.acked = False
                        events.append(("escalated", c.active))
        for event, alert in events:
            self._emit(event, alert)

    def _emit(self, event: str, alert: Alert) -> None:
        for sink in self.sinks:
            try:
                sink.emit(event, alert)
            except Exception:  # a broken sink must never stop monitoring
                log.exception("alert sink %s failed", type(sink).__name__)


class LogSink:
    def emit(self, event: str, alert: Alert) -> None:
        level = logging.WARNING if alert.severity != "info" else logging.INFO
        log.log(level, "alert %s", event, extra={"alert": alert.to_dict()})


class CallbackSink:
    def __init__(self, fn: Callable[[str, Alert], None]) -> None:
        self.fn = fn

    def emit(self, event: str, alert: Alert) -> None:
        self.fn(event, alert)


class WebhookSink:
    """POSTs alert events as JSON from a background thread (never blocks inference)."""

    def __init__(self, cfg: WebhookConfig, store_id: str) -> None:
        self.cfg = cfg
        self.store_id = store_id
        self._q: queue.Queue[dict] = queue.Queue(maxsize=1000)
        threading.Thread(target=self._worker, name="webhook", daemon=True).start()

    def emit(self, event: str, alert: Alert) -> None:
        try:
            self._q.put_nowait({"event": event, "store_id": self.store_id, "alert": alert.to_dict()})
        except queue.Full:
            log.error("webhook queue full; dropping alert %s", alert.id)

    def _worker(self) -> None:
        while True:
            payload = self._q.get()
            req = urllib.request.Request(
                self.cfg.url,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(req, timeout=self.cfg.timeout_s):
                        break
                except Exception as exc:  # noqa: BLE001
                    log.warning("webhook attempt %d failed: %s", attempt + 1, exc)
                    threading.Event().wait(2**attempt)
