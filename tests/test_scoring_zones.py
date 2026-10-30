import numpy as np
import pytest

from segi_freshness.config import ScoringConfig, ZoneConfig
from segi_freshness.reference import BROWN, YELLOW
from segi_freshness.scoring import FreshnessScorer, ZoneTracker, grade
from segi_freshness.zones import ZoneMap, point_in_polygon


@pytest.fixture
def scfg(cfg) -> ScoringConfig:
    return cfg.scoring


def test_color_score_uses_profile(scfg):
    s = FreshnessScorer(scfg, ["fresh", "ripe", "overripe", "spoiled"])
    yellowing = np.zeros(6, np.float32)
    yellowing[YELLOW] = 0.3
    assert s.score("banana", None, yellowing).score == pytest.approx(100.0)  # yellow is normal
    assert s.score("sawi", None, yellowing).score == pytest.approx(55.0)     # 1 - 1.5*0.3


def test_penalty_saturates(scfg):
    s = FreshnessScorer(scfg, ["fresh", "ripe", "overripe", "spoiled"])
    c = np.zeros(6, np.float32)
    c[BROWN] = 0.9
    assert s.score("tomato", None, c).score == 0.0


def test_classifier_fusion(scfg):
    s = FreshnessScorer(scfg, ["fresh", "ripe", "overripe", "spoiled"])
    probs = np.array([0.0, 0.0, 0.0, 1.0], np.float32)
    r = s.score("tomato", probs, np.zeros(6, np.float32))
    assert r.top_class == "spoiled"
    assert r.score == pytest.approx(0.65 * 5 + 0.35 * 100)


def test_grades(scfg):
    assert [grade(v, scfg.grades) for v in (95, 70, 45, 10)] == ["A", "B", "C", "D"]


def test_tracker_ema_and_spoiled(scfg):
    t = ZoneTracker(scfg, [("z", "Zone", "tomato")])
    st = t.update("z", [100, 100], [1, 1], [], 0)
    assert st.ema == 100
    st = t.update("z", [20, 20], [1, 1], [], 1)
    assert st.ema == pytest.approx(0.2 * 20 + 0.8 * 100)
    assert st.n_spoiled == 2
    st = t.update("z", [], [], [], 2)
    assert st.n_items == 0 and st.empty_samples == 1 and st.ema == pytest.approx(84.0)


def test_tracker_area_weighting(scfg):
    t = ZoneTracker(scfg, [("z", "Zone", "x")])
    st = t.update("z", [100, 0], [3, 1], [], 0)
    assert st.raw == pytest.approx(75.0)
    assert st.worst == 0


def test_point_in_polygon():
    tri = [(0, 0), (1, 0), (0, 1)]
    assert point_in_polygon(0.2, 0.2, tri)
    assert not point_in_polygon(0.8, 0.8, tri)


def test_zone_tiles_and_assignment():
    zm = ZoneMap([
        ZoneConfig(id="a", name="A", rect=(0, 0, 0.5, 1), grid=(2, 3)),
        ZoneConfig(id="b", name="B", polygon=[(0.5, 0), (1, 0), (1, 1), (0.5, 1)]),
    ])
    rois, owners = zm.tile_rois(200, 100)
    assert owners == ["a"] * 6 + ["b"]
    assert rois[0] == pytest.approx([0, 0, 50, 100 / 3])
    assigned = zm.assign(np.array([[10, 10, 20, 20], [150, 40, 170, 60], [-50, -50, -40, -40]]), 200, 100)
    assert assigned == ["a", "b", None]


def test_zone_validation():
    with pytest.raises(ValueError):
        ZoneConfig(id="bad", name="x")
    with pytest.raises(ValueError):
        ZoneConfig(id="bad", name="x", rect=(0.5, 0, 0.2, 1))
