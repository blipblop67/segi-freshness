#!/usr/bin/env bash
# Build the CUDA kernels (csrc/) into segi_freshness/_cuda_ops*.so.
# Usage: scripts/build_ext.sh [cuda_arch]   (default 87 = Orin)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ARCH="${1:-87}"
NVCC="${CUDACXX:-/usr/local/cuda/bin/nvcc}"
PY="${PYTHON:-python3}"

[[ -x "$NVCC" ]] || { echo "nvcc not found at $NVCC (install JetPack / set CUDACXX)" >&2; exit 1; }
"$PY" -c "import pybind11" 2>/dev/null || "$PY" -m pip install "pybind11>=2.12"

cmake -S "$ROOT/csrc" -B "$ROOT/build/csrc" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$NVCC" \
  -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
  -DPython3_EXECUTABLE="$(command -v "$PY")"
cmake --build "$ROOT/build/csrc" -j"$(nproc)"
cp "$ROOT"/build/csrc/_cuda_ops*.so "$ROOT/segi_freshness/"
echo "built: $(ls "$ROOT"/segi_freshness/_cuda_ops*.so)"
