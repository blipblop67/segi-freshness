"""MQTT uplink to the store / HQ broker.

Topics (prefix defaults to "segi"):
  {prefix}/{store}/{camera}/status            online|offline (retained, LWT)
  {prefix}/{store}/{camera}/zones/{zone_id}   zone state JSON (retained)
  {prefix}/{store}/{camera}/alerts            alert events JSON
"""

from __future__ import annotations

import json
import logging
import ssl
from typing import Optional

from .alerts import Alert
from .config import MqttConfig

log = logging.getLogger(__name__)


class MqttPublisher:
    def __init__(self, cfg: MqttConfig, store_id: str, camera_id: str) -> None:
        import paho.mqtt.client as mqtt

        self.cfg = cfg
        self.base = f"{cfg.topic_prefix}/{store_id}/{camera_id}"
        self.client = mqtt.Client(
            mqtt.CallbackAPIVersion.VERSION2, client_id=f"segi-fresh-{store_id}-{camera_id}", clean_session=True
        )
        if cfg.username:
            self.client.username_pw_set(cfg.username, cfg.password)
        if cfg.tls:
            self.client.tls_set(cert_reqs=ssl.CERT_REQUIRED)
        self.client.will_set(f"{self.base}/status", "offline", qos=1, retain=True)
        self.client.reconnect_delay_set(min_delay=1, max_delay=60)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect

    def start(self) -> None:
        self.client.connect_async(self.cfg.host, self.cfg.port, keepalive=self.cfg.keepalive_s)
        self.client.loop_start()

    def stop(self) -> None:
        try:
            self.client.publish(f"{self.base}/status", "offline", qos=1, retain=True).wait_for_publish(2)
        except Exception:  # noqa: BLE001
            pass
        self.client.disconnect()
        self.client.loop_stop()

    def _on_connect(self, client, userdata, flags, reason_code, properties) -> None:
        if reason_code.is_failure:
            log.error("MQTT connect failed: %s", reason_code)
            return
        log.info("MQTT connected to %s:%d", self.cfg.host, self.cfg.port)
        client.publish(f"{self.base}/status", "online", qos=1, retain=True)

    def _on_disconnect(self, client, userdata, flags, reason_code, properties) -> None:
        log.warning("MQTT disconnected: %s", reason_code)

    def publish_zone(self, zone: dict) -> None:
        payload = {k: v for k, v in zone.items() if k != "items"}
        self.client.publish(f"{self.base}/zones/{zone['zone_id']}", json.dumps(payload), qos=self.cfg.qos, retain=True)

    def emit(self, event: str, alert: Alert) -> None:  # AlertSink
        self.client.publish(f"{self.base}/alerts", json.dumps({"event": event, **alert.to_dict()}), qos=self.cfg.qos)


def create_publisher(cfg: MqttConfig, store_id: str, camera_id: str) -> Optional[MqttPublisher]:
    if not cfg.enabled:
        return None
    pub = MqttPublisher(cfg, store_id, camera_id)
    pub.start()
    return pub
