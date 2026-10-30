import pytest

from segi_freshness.alerts import AlertManager, CallbackSink
from segi_freshness.config import AlertConfig, ZoneConfig
from segi_freshness.scoring import ZoneState

ZONES = [ZoneConfig(id="z1", name="Tomato", rect=(0, 0, 1, 1))]


@pytest.fixture
def mgr():
    events = []
    cfg = AlertConfig(low_threshold=55, critical_threshold=35, hysteresis=5, trigger_samples=3,
                      clear_samples=2, cooldown_s=100, spoiled_trigger_samples=2)
    m = AlertManager(cfg, ZONES, [CallbackSink(lambda e, a: events.append((e, a.type, a.severity)))])
    return m, events


def _st(ema, n_items=4, n_spoiled=0):
    return ZoneState("z1", "Tomato", "tomato", ema=ema, n_items=n_items, n_spoiled=n_spoiled)


def test_debounce_then_raise_and_clear(mgr):
    m, ev = mgr
    for t in range(2):
        m.evaluate_zone(_st(50), t)
    assert ev == []
    m.evaluate_zone(_st(50), 2)
    assert ev == [("raised", "low_freshness", "warning")]
    m.evaluate_zone(_st(58), 3)  # inside hysteresis band: no clear
    m.evaluate_zone(_st(58), 4)
    assert len(m.active()) == 1
    m.evaluate_zone(_st(61), 5)
    m.evaluate_zone(_st(61), 6)
    assert ev[-1] == ("cleared", "low_freshness", "warning")
    assert m.active() == []


def test_escalation_to_critical(mgr):
    m, ev = mgr
    for t in range(3):
        m.evaluate_zone(_st(50), t)
    m.evaluate_zone(_st(30), 3)
    assert ev[-1] == ("escalated", "low_freshness", "critical")


def test_cooldown_blocks_flapping(mgr):
    m, ev = mgr
    for t in range(3):
        m.evaluate_zone(_st(50), t)
    for t in range(3, 5):
        m.evaluate_zone(_st(90), t)
    for t in range(5, 15):
        m.evaluate_zone(_st(50), t)
    assert [e[0] for e in ev] == ["raised", "cleared"]
    m.evaluate_zone(_st(50), 200)
    assert ev[-1][0] == "raised"


def test_spoiled_items_and_ack(mgr):
    m, ev = mgr
    m.evaluate_zone(_st(90, n_spoiled=1), 0)
    m.evaluate_zone(_st(90, n_spoiled=1), 1)
    assert ev[-1] == ("raised", "spoiled_items", "warning")
    alert = m.active()[0]
    assert m.ack(alert.id, "supervisor").acked
    assert ev[-1][0] == "acked"
    assert m.ack("nope") is None


def test_camera_offline(mgr):
    m, ev = mgr
    m.evaluate_camera(5, 0)
    assert ev == []
    m.evaluate_camera(31, 1)
    assert ev[-1] == ("raised", "camera_offline", "critical")
    m.evaluate_camera(0.5, 2)
    assert ev[-1][0] == "cleared"


def test_broken_sink_does_not_propagate():
    def boom(e, a):
        raise RuntimeError("sink down")

    m = AlertManager(AlertConfig(trigger_samples=1), ZONES, [CallbackSink(boom)])
    m.evaluate_zone(_st(10), 0)  # must not raise
    assert len(m.active()) == 1
