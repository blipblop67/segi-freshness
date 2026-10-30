#!/usr/bin/env bash
# One-time native setup on the target device:
#   Jetson Orin Nano 8GB developer kit, JetPack 6.2 (L4T r36.4.3, CUDA 12.6, TensorRT 10.3).
# Installs to /opt/segi-freshness with a venv that can see the system
# TensorRT and GStreamer-enabled OpenCV.
#
# Environment overrides:
#   NVP_MODE=<id>    power mode id (default: MAXN SUPER if available, else unchanged)
#   SKIP_CLOCKS=1    do not run jetson_clocks
set -euo pipefail

PREFIX="${PREFIX:-/opt/segi-freshness}"
SRC="$(cd "$(dirname "$0")/.." && pwd)"
TARGET_L4T="36.4"   # JetPack 6.2 = r36.4.3

if [[ ! -f /etc/nv_tegra_release ]]; then
  echo "This script targets NVIDIA Jetson (L4T). Aborting." >&2
  exit 1
fi
# e.g. "# R36 (release), REVISION: 4.3, GCID: ..., BOARD: generic, ..."
L4T="$(sed -nE 's/^# R([0-9]+) \(release\), REVISION: ([0-9.]+).*/\1.\2/p' /etc/nv_tegra_release)"
MODEL="$(tr -d '\0' </proc/device-tree/model 2>/dev/null || echo unknown)"
echo "Device: $MODEL"
echo "L4T:    r${L4T:-unknown} (target r36.4.3 / JetPack 6.2)"
if [[ "$L4T" != ${TARGET_L4T}* ]]; then
  echo "WARNING: this project is built and tested for JetPack 6.2 (L4T r36.4.x)." >&2
  echo "         Other releases need a matching docker BASE image and rebuilt TensorRT engines." >&2
  read -r -t 30 -p "Continue anyway? [y/N] " ans || true
  [[ "${ans:-n}" =~ ^[Yy]$ ]] || exit 1
fi
if [[ "$MODEL" != *"Orin Nano"* ]]; then
  echo "WARNING: expected a Jetson Orin Nano, found '$MODEL'. Performance figures will differ." >&2
fi

sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  python3-venv python3-dev python3-pip cmake build-essential \
  python3-opencv gstreamer1.0-tools gstreamer1.0-plugins-good gstreamer1.0-plugins-bad \
  python3-libnvinfer

# Power mode. JetPack 6.2 adds "MAXN SUPER" for the Orin Nano (needs the dev kit's
# 19 V supply and its fan). Mode ids differ between boards and releases, so read the
# id from /etc/nvpmodel.conf instead of hardcoding it.
MODE_ID="${NVP_MODE:-}"
if [[ -z "$MODE_ID" && -f /etc/nvpmodel.conf ]]; then
  MODE_ID="$(sed -nE 's/.*POWER_MODEL ID=([0-9]+) NAME=MAXN_SUPER.*/\1/p' /etc/nvpmodel.conf | head -n1)"
fi
if [[ -n "$MODE_ID" ]]; then
  echo "Setting power mode $MODE_ID ($(sed -nE "s/.*POWER_MODEL ID=${MODE_ID} NAME=([A-Za-z0-9_]+).*/\1/p" /etc/nvpmodel.conf | head -n1))"
  # Answer "no" to nvpmodel's reboot prompt; the summary at the end says to reboot.
  yes no | sudo nvpmodel -m "$MODE_ID" || echo "WARNING: nvpmodel -m $MODE_ID failed; keeping current mode" >&2
else
  echo "MAXN SUPER not found in /etc/nvpmodel.conf; keeping current power mode"
fi
sudo nvpmodel -q || true
[[ -n "${SKIP_CLOCKS:-}" ]] || sudo jetson_clocks || true

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
echo
echo "If nvpmodel reported that the power-mode change needs a reboot, reboot before"
echo "running the benchmark: sudo reboot"
