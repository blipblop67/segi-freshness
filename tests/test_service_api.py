"""End-to-end on the CPU backend: synthetic frame -> scores -> alerts -> storage -> API."""

import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from segi_freshness.api import create_app
from segi_freshness.config import load_config
from segi_freshness.service import FreshnessService

from .conftest import BROWN, GREEN, RED, ROOT

W, H = 640, 360


def _frame(rotten_tomatoes: bool) -> np.ndarray:
    img = np.full((H, W, 3), (60, 60, 60), np.uint8)
    img[: H // 2, : W // 2] = BROWN if rotten_tomatoes else RED  # tomato bin
    img[: H // 2, W // 2 :] = (40, 215, 240)                      # banana bin
    img[H // 2 :, : W // 2] = GREEN                                # sawi bin
    img[H // 2 :, W // 2 :] = (20, 110, 240)                      # carrot bin
    return img


@pytest.fixture
def svc(cfg):
    cfg.alerts.trigger_samples = 2
    cfg.alerts.spoiled_trigger_samples = 2
    cfg.storage.persist_interval_s = 0
    s = FreshnessService(cfg)
    yield s
    s.close()


def test_fresh_frame_scores_high(svc):
    zones = svc.process_frame(_frame(False), now=1000.0)
    assert set(zones) == {"bin-tomato", "bin-banana", "bin-sawi", "bin-carrot"}
    for z in zones.values():
        assert z["score"] > 90, z
        assert z["grade"] == "A"
        assert z["n_items"] == 6  # 3x2 tiles
    assert svc.alerts.active() == []


def test_rotten_bin_raises_alerts_and_api_reports(svc):
    t0 = time.time()
    for i in range(4):
        zones = svc.process_frame(_frame(True), now=t0 + i)
    assert zones["bin-tomato"]["raw_score"] == pytest.approx(0.0)
    assert zones["bin-tomato"]["n_spoiled"] == 6
    assert zones["bin-sawi"]["score"] > 90
    types = {(a.zone_id, a.type) for a in svc.alerts.active()}
    assert ("bin-tomato", "spoiled_items") in types

    client = TestClient(create_app(svc))
    status = client.get("/api/status").json()
    assert {z["zone_id"] for z in status["zones"]} == set(zones)
    assert status["backend"] == "cpu"
    assert client.get("/api/snapshot.jpg").headers["content-type"] == "image/jpeg"
    hist = client.get("/api/zones/bin-tomato/history", params={"hours": 1}).json()
    assert len(hist) == 4 and hist[-1]["n_spoiled"] == 6
    stored = client.get("/api/alerts", params={"active": True}).json()
    assert any(a["zone_id"] == "bin-tomato" and a["type"] == "spoiled_items" for a in stored)
    assert client.get("/api/zones/nope/history").status_code == 404
    assert "segi_zone_freshness_score" in client.get("/metrics").text


def test_ack_requires_api_key(svc):
    svc.cfg.api.api_key = "secret"
    for i in range(3):
        svc.process_frame(_frame(True), now=2000.0 + i)
    alert_id = svc.alerts.active()[0].id
    client = TestClient(create_app(svc))
    assert client.post(f"/api/alerts/{alert_id}/ack").status_code == 401
    r = client.post(f"/api/alerts/{alert_id}/ack", headers={"X-API-Key": "secret"})
    assert r.status_code == 200 and r.json()["acked"] is True


def test_calibration_cancels_background(svc):
    frame = _frame(False)
    frame[:, ::8] = (25, 30, 25)  # dark bin floor visible between items in every bin
    before = svc.process_frame(frame, now=4000.0)
    assert before["bin-sawi"]["raw_score"] < 90 and not before["bin-sawi"]["calibrated"]
    client = TestClient(create_app(svc))
    assert client.post("/api/calibrate", params={"zone": ["nope"]}).status_code == 404
    assert client.post("/api/calibrate").json()["calibrated"]
    after = svc.process_frame(frame, now=4001.0)
    assert after["bin-sawi"]["raw_score"] == pytest.approx(100.0) and after["bin-sawi"]["calibrated"]
    assert set(svc.store.load_baselines()) == set(after)  # survives restarts
    # Real rot on top of the baseline still counts.
    rotten = frame.copy()
    rotten[: H // 2, 1 : W // 2 : 4] = BROWN
    assert svc.process_frame(rotten, now=4002.0)["bin-tomato"]["raw_score"] < 80


def test_alerts_persisted(svc):
    for i in range(3):
        svc.process_frame(_frame(True), now=3000.0 + i)
    rows = svc.store.alerts(0)
    assert any(r["type"] == "spoiled_items" and r["active"] for r in rows)


def test_shipped_config_is_valid():
    c = load_config(ROOT / "configs" / "store.yaml")
    assert c.runtime.mode in ("zones", "detection")
