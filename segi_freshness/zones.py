"""Display-bin geometry: point-in-zone assignment and zone tiling."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Optional

import numpy as np

from .config import ZoneConfig


def point_in_polygon(x: float, y: float, poly: Sequence[tuple[float, float]]) -> bool:
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


class Zone:
    def __init__(self, cfg: ZoneConfig) -> None:
        self.cfg = cfg
        assert cfg.polygon is not None
        self.polygon = list(cfg.polygon)
        xs = [p[0] for p in self.polygon]
        ys = [p[1] for p in self.polygon]
        self.bounds = (min(xs), min(ys), max(xs), max(ys))

    @property
    def id(self) -> str:
        return self.cfg.id

    def contains(self, nx: float, ny: float) -> bool:
        return point_in_polygon(nx, ny, self.polygon)

    def tiles(self, width: int, height: int) -> np.ndarray:
        """Bounding rect split into cols x rows tiles, xyxy in pixels."""
        x1, y1, x2, y2 = self.bounds
        cols, rows = self.cfg.grid
        xs = np.linspace(x1 * width, x2 * width, cols + 1, dtype=np.float32)
        ys = np.linspace(y1 * height, y2 * height, rows + 1, dtype=np.float32)
        out = [(xs[c], ys[r], xs[c + 1], ys[r + 1]) for r in range(rows) for c in range(cols)]
        tiles = np.asarray(out, dtype=np.float32).reshape(-1, 4)
        # Polygon zones: keep only tiles whose centre falls inside the polygon.
        centres = ((tiles[:, 0] + tiles[:, 2]) / 2 / width, (tiles[:, 1] + tiles[:, 3]) / 2 / height)
        keep = [self.contains(cx, cy) for cx, cy in zip(*centres, strict=True)]
        return tiles[np.asarray(keep, dtype=bool)] if any(keep) else tiles


class ZoneMap:
    def __init__(self, zones: Sequence[ZoneConfig]) -> None:
        self.zones = [Zone(z) for z in zones]
        self.by_id = {z.id: z for z in self.zones}

    def assign(self, boxes: np.ndarray, width: int, height: int) -> list[Optional[str]]:
        """Zone id for each xyxy box, by box centre (first match wins)."""
        out: list[Optional[str]] = []
        for x1, y1, x2, y2 in np.asarray(boxes).reshape(-1, 4):
            cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
            out.append(next((z.id for z in self.zones if z.contains(cx, cy)), None))
        return out

    def tile_rois(self, width: int, height: int) -> tuple[np.ndarray, list[str]]:
        boxes, owners = [], []
        for z in self.zones:
            t = z.tiles(width, height)
            boxes.append(t)
            owners.extend([z.id] * len(t))
        return np.concatenate(boxes, axis=0), owners
