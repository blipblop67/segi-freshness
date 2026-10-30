#!/usr/bin/env bash
# Convert ONNX models into TensorRT engines ON THE TARGET JETSON.
# Engines are specific to the GPU + TensorRT version; rebuild after every JetPack upgrade.
#
#   scripts/build_engines.sh models/produce_det.onnx models/freshness_cls.onnx
#
# FP16 is the right default on Orin Nano (Ampere tensor cores). For INT8, see README.
set -euo pipefail

DET_ONNX="${1:-models/produce_det.onnx}"
CLS_ONNX="${2:-models/freshness_cls.onnx}"
CLS_SIZE="${CLS_SIZE:-224}"
CLS_MAX_BATCH="${CLS_MAX_BATCH:-32}"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
WORKSPACE_MB="${WORKSPACE_MB:-2048}"

[[ -x "$TRTEXEC" ]] || { echo "trtexec not found at $TRTEXEC" >&2; exit 1; }

if [[ -f "$DET_ONNX" ]]; then
  echo ">> building detector engine from $DET_ONNX"
  "$TRTEXEC" --onnx="$DET_ONNX" --saveEngine="${DET_ONNX%.onnx}_fp16.engine" \
    --fp16 --memPoolSize=workspace:"$WORKSPACE_MB" --skipInference
else
  echo "!! $DET_ONNX not found, skipping detector (only needed for runtime.mode=detection)"
fi

if [[ -f "$CLS_ONNX" ]]; then
  echo ">> building classifier engine from $CLS_ONNX (dynamic batch 1..$CLS_MAX_BATCH)"
  INPUT_NAME="$(python3 -c "import onnx,sys; print(onnx.load(sys.argv[1]).graph.input[0].name)" "$CLS_ONNX" 2>/dev/null || echo images)"
  "$TRTEXEC" --onnx="$CLS_ONNX" --saveEngine="${CLS_ONNX%.onnx}_fp16.engine" \
    --fp16 --memPoolSize=workspace:"$WORKSPACE_MB" --skipInference \
    --minShapes="${INPUT_NAME}:1x3x${CLS_SIZE}x${CLS_SIZE}" \
    --optShapes="${INPUT_NAME}:8x3x${CLS_SIZE}x${CLS_SIZE}" \
    --maxShapes="${INPUT_NAME}:${CLS_MAX_BATCH}x3x${CLS_SIZE}x${CLS_SIZE}"
else
  echo "!! $CLS_ONNX not found, skipping classifier (colour analysis will be used alone)"
fi
ls -lh models/*.engine 2>/dev/null || true
