"""Command line entry point: `segi-freshness {run,check,benchmark}`."""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import signal
import subprocess
import sys
import threading
import time

import numpy as np

from . import __version__
from .config import load_config

log = logging.getLogger("segi_freshness")

# Target device: Jetson Orin Nano 8GB, JetPack 6.2 (L4T r36.4.3, CUDA 12.6, TensorRT 10.3).
TARGET_L4T = "36.4"


def parse_l4t_release(text: str) -> str | None:
    """'# R36 (release), REVISION: 4.3, ...' -> '36.4.3'."""
    m = re.search(r"R(\d+) \(release\), REVISION: ([\d.]+)", text)
    return f"{m.group(1)}.{m.group(2)}" if m else None


def jetson_platform() -> tuple[str, str] | None:
    """(board model, L4T version) on a Jetson, else None."""
    try:
        with open("/etc/nv_tegra_release", encoding="utf-8") as fh:
            l4t = parse_l4t_release(fh.read()) or "unknown"
    except OSError:
        return None
    try:
        with open("/proc/device-tree/model", "rb") as fh:
            model = fh.read().rstrip(b"\0").decode(errors="replace")
    except OSError:
        model = "unknown Jetson"
    return model, l4t


def cmd_run(args: argparse.Namespace) -> int:
    from .api import ApiServer
    from .service import FreshnessService
    from .telemetry import setup_logging

    cfg = load_config(args.config)
    setup_logging(cfg.runtime.log_level, cfg.runtime.log_json)
    stop = threading.Event()

    def _signal(signum, _frame):
        log.info("received signal %s, shutting down", signum)
        stop.set()

    signal.signal(signal.SIGINT, _signal)
    signal.signal(signal.SIGTERM, _signal)

    svc = FreshnessService(cfg)
    api = ApiServer(svc) if cfg.api.enabled else None
    if api:
        api.start()
        log.info("dashboard on http://%s:%d", cfg.api.host, cfg.api.port)
    code = 0
    try:
        svc.run(stop)
    except Exception:
        log.exception("fatal error")
        code = 1
    finally:
        if api:
            api.stop()
        svc.close()
    return code


def cmd_check(args: argparse.Namespace) -> int:
    """Validate config and report what this device can run."""
    from .processor import cuda_available

    ok = True

    def line(name: str, good: bool, detail: str = "", required: bool = True) -> None:
        nonlocal ok
        ok &= good or not required
        tag = "OK" if good else ("FAIL" if required else "WARN")
        print(f"[{tag:>4}] {name}{': ' + detail if detail else ''}")

    try:
        cfg = load_config(args.config)
        line("config", True, f"{args.config} ({len(cfg.zones)} zones, mode={cfg.runtime.mode})")
    except Exception as exc:  # noqa: BLE001
        line("config", False, str(exc))
        return 1

    platform = jetson_platform()
    if platform is not None:
        model, l4t = platform
        line("device", "Orin Nano" in model, model, required=False)
        line("JetPack", l4t.startswith(TARGET_L4T), f"L4T r{l4t} (target r36.4.3 = JetPack 6.2)", required=False)

    cuda_ok, detail = cuda_available()
    need_cuda = cfg.runtime.backend == "cuda" or cfg.runtime.mode == "detection"
    line("CUDA kernels (_cuda_ops)", cuda_ok, detail, required=need_cuda)
    try:
        import tensorrt as trt  # type: ignore[import-not-found]

        line("TensorRT", True, trt.__version__)
    except ImportError as exc:
        line("TensorRT", False, str(exc), required=need_cuda)
    for name, path in (("detector engine", cfg.models.detector_engine), ("classifier engine", cfg.models.classifier_engine)):
        if path is not None:
            required = name.startswith("detector") and cfg.runtime.mode == "detection"
            line(name, path.is_file(), f"{path} {'found' if path.is_file() else 'missing'}", required=required)

    import cv2

    from .camera import CameraStream, redact_uri, use_hw_decoder

    hw = cfg.camera.source in ("csi", "usb") or (cfg.camera.source == "rtsp" and use_hw_decoder(cfg.camera))
    gst = re.search(r"GStreamer:\s+YES", cv2.getBuildInformation()) is not None
    line("OpenCV GStreamer", gst, f"OpenCV {cv2.__version__}", required=hw)
    if hw and shutil.which("gst-inspect-1.0"):
        plugins = ["nvvidconv", "nvv4l2decoder"]
        if cfg.camera.source == "csi":
            plugins.append("nvarguscamerasrc")
        if cfg.camera.source == "rtsp":
            plugins += ["rtspsrc", f"rtp{cfg.camera.rtsp_codec}depay", f"{cfg.camera.rtsp_codec}parse"]
        for plugin in plugins:
            found = subprocess.run(["gst-inspect-1.0", plugin], capture_output=True).returncode == 0
            line(f"gst plugin {plugin}", found)

    # Open the real source and grab one frame (catches wrong RTSP URL / password / codec).
    where = redact_uri(cfg.camera.uri) if cfg.camera.uri else f"sensor {cfg.camera.sensor_id}"
    try:
        cap = CameraStream(cfg.camera)._open()
        got, frame = cap.read()
        cap.release()
        detail = f"{where} -> {frame.shape[1]}x{frame.shape[0]}" if got else f"{where} opened but no frame"
        line(f"camera ({cfg.camera.source}, {'NVDEC' if hw else 'software decode'})", got, detail)
    except Exception as exc:  # noqa: BLE001
        line(f"camera ({cfg.camera.source})", False, str(exc))
    if shutil.which("nvpmodel"):
        q = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True)
        print(f"[INFO] power mode: {q.stdout.strip().splitlines()[0] if q.stdout else 'unknown'}")
    return 0 if ok else 1


def cmd_benchmark(args: argparse.Namespace) -> int:
    from .processor import create_processor
    from .telemetry import setup_logging
    from .zones import ZoneMap

    cfg = load_config(args.config)
    setup_logging("WARNING", False)
    proc = create_processor(cfg)
    rng = np.random.default_rng(0)
    frame = rng.integers(0, 255, (args.height, args.width, 3), dtype=np.uint8)
    rois, owners = ZoneMap(cfg.zones).tile_rois(args.width, args.height)
    use_rois = None if cfg.runtime.mode == "detection" else rois

    for _ in range(args.warmup):
        proc.process(frame, use_rois, ["mixed"] * len(owners) if use_rois is not None else None)
    totals, stages = [], {}
    for _ in range(args.iterations):
        t0 = time.perf_counter()
        r = proc.process(frame, use_rois, ["mixed"] * len(owners) if use_rois is not None else None)
        totals.append((time.perf_counter() - t0) * 1e3)
        for k, v in r.timings_ms.items():
            stages.setdefault(k, []).append(v)
    proc.close()

    def pct(xs, p):
        return float(np.percentile(xs, p))

    print(f"backend={proc.backend} mode={cfg.runtime.mode} frame={args.width}x{args.height} rois={len(rois)}")
    for k, xs in [("total", totals)] + sorted(stages.items()):
        print(f"  {k:<10} p50={pct(xs, 50):7.2f} ms  p95={pct(xs, 95):7.2f} ms")
    print(f"  max sustainable rate ~ {1000 / pct(totals, 50):.1f} fps")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="segi-freshness", description="Segi Fresh Monitor - produce freshness AI on Jetson")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn, help_ in (("run", cmd_run, "run the monitoring service"),
                            ("check", cmd_check, "validate config and device readiness"),
                            ("benchmark", cmd_benchmark, "measure pipeline latency")):
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("-c", "--config", default="configs/store.yaml")
        sp.set_defaults(fn=fn)
        if name == "benchmark":
            sp.add_argument("--iterations", type=int, default=200)
            sp.add_argument("--warmup", type=int, default=20)
            sp.add_argument("--width", type=int, default=1920)
            sp.add_argument("--height", type=int, default=1080)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
