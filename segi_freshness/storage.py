"""Local SQLite persistence (WAL) for readings and alerts. Survives network
outages; MQTT/webhook forward to central systems when available."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .alerts import Alert
from .scoring import ZoneState

_SCHEMA = """
CREATE TABLE IF NOT EXISTS readings (
    ts REAL NOT NULL,
    zone_id TEXT NOT NULL,
    score REAL,
    raw_score REAL,
    worst_item REAL,
    grade TEXT,
    n_items INTEGER,
    n_spoiled INTEGER
);
CREATE INDEX IF NOT EXISTS idx_readings_zone_ts ON readings (zone_id, ts);
CREATE TABLE IF NOT EXISTS alerts (
    id TEXT PRIMARY KEY,
    zone_id TEXT NOT NULL,
    type TEXT NOT NULL,
    severity TEXT NOT NULL,
    message TEXT NOT NULL,
    value REAL,
    raised_at REAL NOT NULL,
    cleared_at REAL,
    acked INTEGER NOT NULL DEFAULT 0,
    acked_by TEXT
);
CREATE INDEX IF NOT EXISTS idx_alerts_raised ON alerts (raised_at);
CREATE TABLE IF NOT EXISTS baselines (
    zone_id TEXT PRIMARY KEY,
    stats TEXT NOT NULL,
    ts REAL NOT NULL
);
"""


class Store:
    def __init__(self, path: Path | str) -> None:
        path = Path(path)
        if str(path) != ":memory:":
            path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(_SCHEMA)

    def insert_readings(self, states: list[ZoneState], ts: Optional[float] = None) -> None:
        ts = time.time() if ts is None else ts
        rows = [
            (ts, s.zone_id, s.ema, s.raw, s.worst, s.grade, s.n_items, s.n_spoiled)
            for s in states
            if s.samples > 0
        ]
        if not rows:
            return
        with self._lock:
            self._conn.executemany("INSERT INTO readings VALUES (?,?,?,?,?,?,?,?)", rows)

    def upsert_alert(self, a: Alert) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT INTO alerts VALUES (?,?,?,?,?,?,?,?,?,?)
                   ON CONFLICT(id) DO UPDATE SET severity=excluded.severity, message=excluded.message,
                   value=excluded.value, cleared_at=excluded.cleared_at, acked=excluded.acked,
                   acked_by=excluded.acked_by""",
                (a.id, a.zone_id, a.type, a.severity, a.message, a.value, a.raised_at, a.cleared_at,
                 int(a.acked), a.acked_by),
            )

    def history(self, zone_id: str, since: float, limit: int = 5000) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT ts, score, worst_item, grade, n_items, n_spoiled FROM readings "
                "WHERE zone_id=? AND ts>=? ORDER BY ts LIMIT ?",
                (zone_id, since, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def alerts(self, since: float = 0.0, active_only: bool = False, limit: int = 200) -> list[dict]:
        q = "SELECT * FROM alerts WHERE raised_at>=?"
        if active_only:
            q += " AND cleared_at IS NULL"
        q += " ORDER BY raised_at DESC LIMIT ?"
        with self._lock:
            rows = self._conn.execute(q, (since, limit)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["acked"] = bool(d["acked"])
            d["active"] = d["cleared_at"] is None
            out.append(d)
        return out

    def close_dangling_alerts(self, now: float) -> int:
        """Alerts left active by a crash/restart are closed; live state re-raises them."""
        with self._lock:
            cur = self._conn.execute("UPDATE alerts SET cleared_at=? WHERE cleared_at IS NULL", (now,))
            return cur.rowcount

    def purge(self, older_than: float) -> int:
        with self._lock:
            n = self._conn.execute("DELETE FROM readings WHERE ts<?", (older_than,)).rowcount
            n += self._conn.execute("DELETE FROM alerts WHERE raised_at<? AND cleared_at IS NOT NULL", (older_than,)).rowcount
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        return n

    def save_baseline(self, zone_id: str, stats: np.ndarray) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO baselines VALUES (?,?,?)",
                (zone_id, json.dumps(np.asarray(stats, np.float32).tolist()), time.time()),
            )

    def delete_baseline(self, zone_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM baselines WHERE zone_id=?", (zone_id,))

    def load_baselines(self) -> dict[str, np.ndarray]:
        with self._lock:
            rows = self._conn.execute("SELECT zone_id, stats FROM baselines").fetchall()
        return {r["zone_id"]: np.asarray(json.loads(r["stats"]), np.float32) for r in rows}

    def close(self) -> None:
        with self._lock:
            self._conn.close()
