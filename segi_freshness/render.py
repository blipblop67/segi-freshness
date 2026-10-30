"""Annotated JPEG snapshots for the dashboard."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from .zones import ZoneMap

_GRADE_BGR = {"A": (80, 175, 76), "B": (59, 200, 205), "C": (0, 140, 255), "D": (54, 67, 244), "-": (160, 160, 160)}


def _score_bgr(score: float) -> tuple[int, int, int]:
    if score >= 80:
        return _GRADE_BGR["A"]
    if score >= 60:
        return _GRADE_BGR["B"]
    if score >= 40:
        return _GRADE_BGR["C"]
    return _GRADE_BGR["D"]


def render_snapshot(
    frame: np.ndarray,
    zone_map: ZoneMap,
    zone_states: dict[str, dict],
    boxes: np.ndarray,
    scores: Sequence[float],
    labels: Sequence[str],
    max_width: int = 1280,
    quality: int = 80,
) -> bytes:
    img = frame[..., :3].copy()
    h, w = img.shape[:2]
    thick = max(1, w // 640)

    for (x1, y1, x2, y2), s, label in zip(np.asarray(boxes).reshape(-1, 4), scores, labels, strict=True):
        color = _score_bgr(s)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thick)
        cv2.putText(img, f"{label} {s:.0f}", (int(x1) + 3, int(y2) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45 * thick, color, thick, cv2.LINE_AA)

    for z in zone_map.zones:
        st = zone_states.get(z.id, {})
        color = _GRADE_BGR.get(st.get("grade", "-"), _GRADE_BGR["-"])
        pts = np.array([(int(x * w), int(y * h)) for x, y in z.polygon], np.int32)
        cv2.polylines(img, [pts], True, color, thick + 1)
        score = st.get("score")
        text = f"{z.cfg.name}: {'--' if score is None else f'{score:.0f}'} [{st.get('grade', '-')}]"
        org = (int(z.bounds[0] * w) + 6, int(z.bounds[1] * h) + 24 * thick)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * thick, thick)
        cv2.rectangle(img, (org[0] - 4, org[1] - th - 6), (org[0] + tw + 4, org[1] + 6), (0, 0, 0), -1)
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * thick, color, thick, cv2.LINE_AA)

    if w > max_width:
        img = cv2.resize(img, (max_width, int(h * max_width / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()
