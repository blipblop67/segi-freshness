"""REST API + dashboard (FastAPI)."""

from __future__ import annotations

import hmac
import time
from importlib import resources
from typing import TYPE_CHECKING, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Response
from fastapi.responses import HTMLResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from . import __version__

if TYPE_CHECKING:
    from .service import FreshnessService


def create_app(svc: FreshnessService) -> FastAPI:
    cfg = svc.cfg
    app = FastAPI(title="Segi Fresh Monitor", version=__version__, docs_url="/api/docs", openapi_url="/api/openapi.json")

    def require_key(x_api_key: Optional[str] = Header(default=None)) -> None:
        expected = cfg.api.api_key
        if expected and not (x_api_key and hmac.compare_digest(x_api_key, expected)):
            raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def dashboard() -> str:
        return resources.files("segi_freshness").joinpath("static/index.html").read_text(encoding="utf-8")

    @app.get("/api/health")
    def health(response: Response) -> dict:
        last_cycle, camera, started = svc.state.read("last_cycle_at", "camera", "started_at")
        now = time.time()
        stale_after = max(10.0, 5.0 / cfg.runtime.inference_fps)
        healthy = last_cycle is not None and now - last_cycle < stale_after
        if not healthy and now - started > 60:
            response.status_code = 503
        return {
            "status": "ok" if healthy else "degraded",
            "version": __version__,
            "store_id": cfg.store_id,
            "camera_id": cfg.camera_id,
            "mode": cfg.runtime.mode,
            "backend": svc.processor.backend,
            "detector": svc.processor.has_detector,
            "classifier": svc.processor.has_classifier,
            "last_cycle_age_s": None if last_cycle is None else round(now - last_cycle, 2),
            "uptime_s": round(now - started, 1),
            "camera": camera,
        }

    @app.get("/api/status")
    def status() -> dict:
        zones, timings, soc, snap_at = svc.state.read("zones", "last_timings_ms", "soc", "snapshot_at")
        return {
            "store_id": cfg.store_id,
            "store_name": cfg.store_name,
            "zones": list(zones.values()),
            "alerts": [a.to_dict() for a in svc.alerts.active()],
            "timings_ms": {k: round(v, 2) for k, v in timings.items()},
            "soc": soc,
            "snapshot_at": snap_at,
            "backend": svc.processor.backend,
            "mode": cfg.runtime.mode,
        }

    @app.get("/api/zones")
    def zones() -> list[dict]:
        (z,) = svc.state.read("zones")
        return list(z.values())

    @app.get("/api/zones/{zone_id}/history")
    def history(zone_id: str, hours: float = Query(24, gt=0, le=24 * 30)) -> list[dict]:
        if zone_id not in svc.zone_map.by_id:
            raise HTTPException(404, "unknown zone")
        return svc.store.history(zone_id, time.time() - hours * 3600)

    @app.get("/api/alerts")
    def alerts(active: bool = False, hours: float = Query(24, gt=0, le=24 * 30), limit: int = Query(200, le=1000)) -> list[dict]:
        return svc.store.alerts(time.time() - hours * 3600, active_only=active, limit=limit)

    @app.post("/api/alerts/{alert_id}/ack", dependencies=[Depends(require_key)])
    def ack(alert_id: str, by: str = Query("dashboard", max_length=64)) -> dict:
        alert = svc.alerts.ack(alert_id, by)
        if alert is None:
            raise HTTPException(404, "no active alert with that id")
        return alert.to_dict()

    @app.post("/api/calibrate", dependencies=[Depends(require_key)])
    def calibrate(zone: Optional[list[str]] = Query(None)) -> dict:
        """Call right after bins are restocked with fresh produce."""
        unknown = [z for z in zone or [] if z not in svc.zone_map.by_id]
        if unknown:
            raise HTTPException(404, f"unknown zones {unknown}")
        try:
            return {"calibrated": svc.calibrate(zone)}
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.delete("/api/calibrate/{zone_id}", dependencies=[Depends(require_key)])
    def clear_calibration(zone_id: str) -> dict:
        if zone_id not in svc.zone_map.by_id:
            raise HTTPException(404, "unknown zone")
        svc.clear_calibration(zone_id)
        return {"cleared": zone_id}

    @app.get("/api/snapshot.jpg")
    def snapshot() -> Response:
        (jpeg,) = svc.state.read("snapshot_jpeg")
        if jpeg is None:
            raise HTTPException(503, "no snapshot yet")
        return Response(jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


class ApiServer:
    """Runs uvicorn in a background thread alongside the inference loop."""

    def __init__(self, svc: FreshnessService) -> None:
        import threading

        import uvicorn

        config = uvicorn.Config(create_app(svc), host=svc.cfg.api.host, port=svc.cfg.api.port,
                                log_level="warning", access_log=False)
        self.server = uvicorn.Server(config)
        self.thread = threading.Thread(target=self.server.run, name="api", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.server.should_exit = True
        self.thread.join(timeout=5)
