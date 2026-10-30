#!/usr/bin/env bash
# One-time native setup on a Jetson Orin Nano running JetPack 6.x (L4T r36).
# Installs to /opt/segi-freshness with a venv that can see the system
# TensorRT and GStreamer-enabled OpenCV.
set -euo pipefail

PREFIX="${PREFIX:-/opt/segi-freshness}"
SRC="$(cd "$(dirname "$0")/.." && pwd)"

if [[ ! -f /etc/nv_tegra_release ]]; then
  echo "This script targets NVIDIA Jetson (L4T). Aborting." >&2
  exit 1
fi
echo "L4T: $(head -n1 /etc/nv_tegra_release)"

sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  python3-venv python3-dev python3-pip cmake build-essential \
  python3-opencv gstreamer1.0-tools gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
  python3-libnvinfer

# Performance: max clocks for the configured power mode. On Orin Nano with
# JetPack 6.2, mode 2 is "MAXN SUPER" (25 W). Check `nvpmodel -q --verbose`.
sudo nvpmodel -m "${NVP_MODE:-0}" || true
sudo jetson_clocks || true

sudo useradd --system --home "$PREFIX" --shell /usr/sbin/nologin segi 2>/dev/null || true
sudo usermod -aG video segi
sudo mkdir -p "$PREFIX"
sudo rsync -a --delete --exclude build --exclude .git --exclude data "$SRC/" "$PREFIX/"
sudo mkdir -p "$PREFIX/data" "$PREFIX/models"
sudo chown -R segi:segi "$PREFIX"

sudo -u segi python3 -m venv --system-site-packages "$PREFIX/.venv"
sudo -u segi "$PREFIX/.venv/bin/pip" install --upgrade pip
sudo -u segi "$PREFIX/.venv/bin/pip" install -e "$PREFIX[jetson]" pybind11
sudo -u segi PYTHON="$PREFIX/.venv/bin/python" "$PREFIX/scripts/build_ext.sh"

sudo install -m 0644 "$PREFIX/deploy/segi-freshness.service" /etc/systemd/system/
sudo install -d -m 0750 /etc/segi-freshness
[[ -f /etc/segi-freshness/env ]] || sudo install -m 0640 "$PREFIX/deploy/env.example" /etc/segi-freshness/env
sudo systemctl daemon-reload
echo
echo "Next:"
echo "  1. edit $PREFIX/configs/store.yaml (camera + zones) and /etc/segi-freshness/env (secrets)"
echo "  2. copy ONNX models to $PREFIX/models and run scripts/build_engines.sh"
echo "  3. sudo -u segi $PREFIX/.venv/bin/segi-freshness check -c $PREFIX/configs/store.yaml"
echo "  4. sudo systemctl enable --now segi-freshness"
