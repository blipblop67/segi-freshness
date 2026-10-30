# Segi Fresh Monitor

Edge AI freshness monitoring for wet-market fruit and vegetables at Segi Cash & Carry,
running on an **NVIDIA Jetson Orin Nano** with custom **CUDA** kernels and **TensorRT**.

The store CCTV camera looks down at the display bins. Every 0.5 s the device scores each bin
(0–100, grade A–D), spots rotting, mouldy, bruised or yellowing produce, and alerts staff
on the dashboard, over MQTT, or through a webhook *before* customers see bad stock.

```mermaid
flowchart LR
  CAM[Store CCTV / NVR<br/>RTSP H.264/H.265] -->|NVDEC + VIC<br/>GStreamer| CAP[Capture thread<br/>latest frame only]
  CAP -->|1 upload / cycle| GPU
  subgraph GPU [Jetson GPU - one CUDA stream]
    LB[letterbox kernel] --> DET[TensorRT YOLO11<br/>produce detector]
    CR[batched crop-resize kernel] --> CLS[TensorRT freshness<br/>classifier]
    CS[HSV defect-analysis kernel<br/>dark / brown / mould / yellow]
  end
  DET -->|boxes| CR
  DET -->|boxes| CS
  ZONES[Fixed bin tiles<br/>zones mode] --> CR & CS
  CLS & CS --> SC[Score fusion<br/>+ produce profiles<br/>+ baseline calibration]
  SC --> TR[Per-bin EMA + grade] --> AL[Alert state machine<br/>debounce / hysteresis / cooldown]
  AL --> MQ[MQTT] & WH[Webhook] & DB[(SQLite WAL)]
  TR --> API[FastAPI + dashboard<br/>Prometheus /metrics]
```

## Why this design

| Decision | Reason |
|---|---|
| **Custom CUDA kernels** (`csrc/`) | The frame is uploaded once; letterboxing, crop+resize for every item, and colour analysis all read that buffer and write directly into TensorRT input bindings. No CPU resizing, no extra copies. |
| **Two scoring signals** | A learned classifier (fresh/ripe/overripe/spoiled) is fused with an explainable colour-defect measurement. Staff can see *why* a bin scored low (e.g. "brown 18%"), and the system works on day one before a classifier is trained. |
| **Two modes** | `zones`: fixed bins split into tiles, no detector needed, so you can deploy fast. `detection`: YOLO finds each item, which adds per-item scoring and *restock* alerts. |
| **Produce profiles** | Yellow is normal for bananas but a staleness sign for sawi or kangkung; brown sugar spots on bananas are fine. Weights are set per produce type. |
| **Baseline calibration** | Bin floors, shadows and store lighting differ per bin. After restocking, staff press *Calibrate*, and only defects *above* the fresh baseline count. |
| **2 Hz inference** | Produce doesn't move. Running slowly keeps the Orin Nano cool and leaves headroom for more cameras. |
| **Edge-first storage** | SQLite keeps working through network outages. MQTT and webhooks forward data upstream when the network is available. |

## Repository layout

```
csrc/                 CUDA kernels + pybind11 bindings (letterbox, crop_resize, color_stats)
segi_freshness/
  reference.py        NumPy twins of every kernel (CPU path, tests, training crops)
  gpu.py              cuda-python device/pinned memory + streams
  trt_engine.py       TensorRT 10 runner (preallocated I/O, dynamic batch)
  processor.py        GPU and CPU frame processors -> FrameResult
  postprocess.py      YOLO decode + NMS
  scoring.py          score fusion, grades, per-bin EMA
  zones.py            bin geometry, tiling, point-in-polygon
  alerts.py           alert state machine + log/webhook sinks
  service.py          main loop, calibration, supervision
  api.py / static/    REST API + offline dashboard
  camera.py           GStreamer capture with reconnect/backoff
  storage.py mqtt.py telemetry.py render.py
configs/
  store.yaml          production template: store CCTV over RTSP
  demo.yaml           demo: replays the sample clip (extends store.yaml)
  demo-rtsp.yaml      demo: pulls the simulated CCTV RTSP stream
samples/              CCTV-style sample clip for demos (synthetic)
docs/CCTV.md          camera placement, settings, RTSP URLs, zone drawing
scripts/              build_ext.sh, build_engines.sh, setup_jetson.sh, make_demo_video.py
training/             train.py (YOLO11 det + cls), make_crops.py, dataset yaml
docker/ deploy/       container, compose, CCTV simulator, systemd unit (watchdog), env template
tests/                CPU tests + CUDA-vs-NumPy parity (runs on device)
```

## Quick start

### 1. Demo with the sample CCTV clip (no Jetson, no camera)

```bash
pip install -e ".[dev]"
segi-freshness run -c configs/demo.yaml
```

Open http://localhost:8080. The bundled clip ([samples/](samples/README.md)) imitates a store
CCTV camera. Over 2 minutes the tomatoes rot and the sawi yellows, so scores drop and alerts fire.
Then a staff member restocks, the scores recover, and the clip loops. Without a GPU this uses
the NumPy reference path, which is identical in maths to the CUDA kernels.

To demo the **real CCTV path** (RTSP), serve the clip as a simulated IP camera:

```bash
docker compose -f docker/cctv-simulator.yml up -d     # rtsp://localhost:8554/wetmarket-cam01
segi-freshness run -c configs/demo-rtsp.yaml
```

### 2. Store CCTV

The monitor reads any IP camera or NVR channel over RTSP (H.264 or H.265), decoded by the
Jetson's hardware decoder. See **[docs/CCTV.md](docs/CCTV.md)** for camera placement, the colour
settings that matter (lock the white balance!), RTSP URLs for Hikvision/Dahua/Uniview, and how
to draw the bin zones. In short:

```bash
echo 'SEGI_CAMERA_URI=rtsp://segi-viewer:PASSWORD@192.168.1.64:554/Streaming/Channels/101' | sudo tee -a /etc/segi-freshness/env
segi-freshness check -c configs/store.yaml   # verifies the stream and grabs a frame
```

### 3. Jetson Orin Nano (native, recommended for production)

Requires JetPack 6.x (L4T r36, CUDA 12.6, TensorRT 10).

```bash
git clone <repo> && cd segi-freshness
./scripts/setup_jetson.sh          # deps, venv, builds CUDA kernels, installs systemd unit
sudo -u segi /opt/segi-freshness/.venv/bin/segi-freshness check -c /opt/segi-freshness/configs/store.yaml
sudo systemctl enable --now segi-freshness
journalctl -u segi-freshness -f
```

### 4. Jetson with Docker

```bash
docker compose -f docker/docker-compose.yml up -d --build
```

Set the `BASE` build-arg to match `/etc/nv_tegra_release` (default `l4t-jetpack:r36.4.0`).

## Models

The system runs without trained models (colour analysis only). For production accuracy, train both models:

1. **Collect** 1–2 weeks of images from the real store camera at different times of day.
2. **Detector:** label produce types in YOLO format (Roboflow or CVAT), then
   `python training/train.py detector --data training/produce_det.yaml`
3. **Classifier:** `python training/make_crops.py ...` cuts crops using the *exact* on-device
   crop maths. Store staff sort them into `fresh/ripe/overripe/spoiled`, then
   `python training/train.py classifier --data datasets/cls`
4. Copy the ONNX files to the Jetson, then run `scripts/build_engines.sh` (FP16 TensorRT).
5. `segi-freshness benchmark -c configs/store.yaml` measures the latency.

Hue augmentation is deliberately small during training, because colour *is* the freshness signal.

INT8: the Orin Nano's FP16 tensor cores already run YOLO11s fast enough at 2 Hz. INT8 is
only worth it for multi-camera setups, and needs a calibration cache built from real store frames
(`trtexec --int8 --calib=...`).

## Operations

| Task | How |
|---|---|
| Restocked a bin | Dashboard, then **Calibrate bins** (or `POST /api/calibrate?zone=bin-tomato`) |
| Acknowledge alert | Dashboard **Ack** (needs `SEGI_API_KEY`) |
| Health / liveness | `GET /api/health` (returns 503 when the pipeline stalls) |
| Metrics | `GET /metrics`: zone scores, stage latency, SoC temperature, GPU load, camera fps |
| Max performance | `sudo nvpmodel -m <MAXN SUPER id> && sudo jetson_clocks` |
| Hung pipeline | systemd `WatchdogSec=60` restarts it; after 20 consecutive errors the process exits so the supervisor restarts it |
| Data retention | `storage.retention_days` (default 30), purged every 6 h |

### API

| Method | Path | |
|---|---|---|
| GET | `/api/status` | zones, active alerts, timings, SoC |
| GET | `/api/zones/{id}/history?hours=24` | stored readings |
| GET | `/api/alerts?active=true` | alert log |
| POST | `/api/alerts/{id}/ack` | 🔑 |
| POST | `/api/calibrate[?zone=...]` | 🔑 |
| DELETE | `/api/calibrate/{zone_id}` | 🔑 |
| GET | `/api/snapshot.jpg` | annotated live view |
| GET | `/api/docs` | OpenAPI UI |

MQTT topics: `segi/{store}/{camera}/status` (LWT), `.../zones/{zone}` (retained), `.../alerts`.

## Testing

```bash
pytest -q                     # anywhere; CUDA parity tests auto-skip without the extension
pytest tests/test_cuda_parity.py -v   # on the Jetson after scripts/build_ext.sh
```

CI runs lint and tests, and compiles the CUDA kernels for SM 8.7 in an `nvidia/cuda` container.

## Known limitations and next steps

- Colour analysis can't see internal rot or firmness. The classifier and staff spot-checks cover this.
- In `zones` mode, a bin that empties shows more floor. Use `detection` mode for restock alerts.
- Strong colour-cast lighting (e.g. red "meat" lamps) shifts hues. Calibrate per bin, and prefer neutral 4000–5000 K lighting over the produce.
- Next: per-SKU shelf-life models, POS markdown integration (auto-discount grade C), multi-camera on one Orin Nano with DeepStream.
