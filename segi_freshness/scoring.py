"""Fuses classifier probabilities and colour-defect statistics into a 0-100
freshness score per item, and smooths scores per zone."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .config import GradeBands, ProduceProfile, ScoringConfig
from .reference import BROWN, DARK, MOLD, YELLOW


@dataclass(frozen=True)
class ItemScore:
    score: float
    color_score: float
    classifier_score: Optional[float]
    top_class: Optional[str]
    top_prob: Optional[float]


def grade(score: float, bands: GradeBands) -> str:
    if score >= bands.a:
        return "A"
    if score >= bands.b:
        return "B"
    if score >= bands.c:
        return "C"
    return "D"


class FreshnessScorer:
    def __init__(self, cfg: ScoringConfig, classifier_labels: Sequence[str]) -> None:
        self.cfg = cfg
        self.labels = list(classifier_labels)
        self._class_values = np.array([cfg.class_scores[label] for label in self.labels], dtype=np.float32)

    def profile_for(self, produce: str) -> ProduceProfile:
        name = self.cfg.produce_profile_map.get(produce, "default")
        return self.cfg.profiles.get(name, self.cfg.profiles["default"])

    @staticmethod
    def color_score(color: np.ndarray, profile: ProduceProfile) -> float:
        penalty = (
            profile.dark * float(color[DARK])
            + profile.brown * float(color[BROWN])
            + profile.mold * float(color[MOLD])
            + profile.yellow * float(color[YELLOW])
        )
        return 100.0 * (1.0 - min(1.0, max(0.0, penalty)))

    def classifier_score(self, probs: np.ndarray) -> float:
        return float(np.dot(probs, self._class_values))

    def score(self, produce: str, probs: Optional[np.ndarray], color: np.ndarray) -> ItemScore:
        c_score = self.color_score(color, self.profile_for(produce))
        if probs is None:
            return ItemScore(c_score, c_score, None, None, None)
        k_score = self.classifier_score(probs)
        w = self.cfg.classifier_weight
        top = int(np.argmax(probs))
        return ItemScore(
            score=w * k_score + (1.0 - w) * c_score,
            color_score=c_score,
            classifier_score=k_score,
            top_class=self.labels[top],
            top_prob=float(probs[top]),
        )


@dataclass
class ZoneState:
    zone_id: str
    name: str
    produce: str
    ema: Optional[float] = None
    raw: Optional[float] = None
    worst: Optional[float] = None
    n_items: int = 0
    n_spoiled: int = 0
    empty_samples: int = 0
    samples: int = 0
    updated_at: float = 0.0
    grade: str = "-"
    items: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "zone_id": self.zone_id,
            "name": self.name,
            "produce": self.produce,
            "score": None if self.ema is None else round(self.ema, 1),
            "raw_score": None if self.raw is None else round(self.raw, 1),
            "worst_item": None if self.worst is None else round(self.worst, 1),
            "grade": self.grade,
            "n_items": self.n_items,
            "n_spoiled": self.n_spoiled,
            "updated_at": self.updated_at,
            "items": self.items,
        }


class ZoneTracker:
    """Area-weighted per-zone scoring with exponential smoothing."""

    def __init__(self, cfg: ScoringConfig, zones: Sequence[tuple[str, str, str]]) -> None:
        self.cfg = cfg
        self.states = {zid: ZoneState(zid, name, produce) for zid, name, produce in zones}

    def update(self, zone_id: str, scores: Sequence[float], areas: Sequence[float], items: list[dict], now: float) -> ZoneState:
        st = self.states[zone_id]
        st.samples += 1
        st.updated_at = now
        st.items = items
        st.n_items = len(scores)
        if not scores:
            st.empty_samples += 1
            st.n_spoiled = 0
            st.raw = None
            st.worst = None
            return st
        st.empty_samples = 0
        s = np.asarray(scores, dtype=np.float64)
        a = np.maximum(np.asarray(areas, dtype=np.float64), 1.0)
        st.raw = float(np.sum(s * a) / np.sum(a))
        st.worst = float(s.min())
        st.n_spoiled = int(np.sum(s < self.cfg.spoiled_item_score))
        alpha = self.cfg.ema_alpha
        st.ema = st.raw if st.ema is None else alpha * st.raw + (1 - alpha) * st.ema
        st.grade = grade(st.ema, self.cfg.grades)
        return st
