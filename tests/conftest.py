from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from segi_freshness.config import AppConfig, load_config

ROOT = Path(__file__).resolve().parents[1]

# BGR colours with known HSV classification under the default thresholds.
BROWN = (30, 60, 110)    # h~20 deg, s~0.73, v~0.43 -> brown
DARK = (15, 15, 20)      # v < 0.2 -> dark
MOLD = (215, 220, 225)   # s < 0.18, v > 0.65 -> mould
YELLOW = (40, 210, 220)  # h~57 deg, s~0.82 -> yellow
GREEN = (50, 160, 40)    # healthy leaf -> none
RED = (30, 30, 205)      # ripe tomato -> none


def solid(color, h=120, w=160) -> np.ndarray:
    img = np.empty((h, w, 3), np.uint8)
    img[:] = color
    return img


@pytest.fixture
def cfg(tmp_path) -> AppConfig:
    c = load_config(ROOT / "configs" / "store.yaml")
    c.storage.path = tmp_path / "test.db"
    c.runtime.backend = "cpu"
    c.models.classifier_engine = None
    return c
