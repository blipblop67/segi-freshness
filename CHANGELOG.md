# Changelog

## Unreleased

- Released under the MIT license

- Target platform pinned to the Jetson Orin Nano 8GB on JetPack 6.2 (L4T r36.4.3)
- `setup_jetson.sh` checks the L4T version and board, and selects MAXN SUPER from `/etc/nvpmodel.conf`
- `segi-freshness check` reports the board model and L4T version

## 1.0.0 - 2026-10-30

First release for the Segi Cash & Carry wet-market pilot.

- CUDA kernels for letterbox, batched crop-resize and HSV colour-defect analysis (SM 8.7)
- TensorRT 10 runner for the YOLO11 produce detector and the freshness classifier
- `zones` mode (fixed bins, no models required) and `detection` mode
- Per-produce scoring profiles and per-bin baseline calibration
- Alerts with debounce, hysteresis, escalation and cooldown, sent to the dashboard, MQTT and a webhook
- CCTV over RTSP (H.264/H.265) with Jetson NVDEC, software-decode fallback, and passwords masked in logs
- CCTV-style sample clip, plus an RTSP CCTV simulator for demos
- Docker image, systemd unit with watchdog, Prometheus metrics, SQLite storage
