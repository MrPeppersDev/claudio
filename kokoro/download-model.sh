#!/bin/bash
# Downloads Kokoro v1.0 model weights + voices from the upstream GitHub release.
# Idempotent — skips if files already exist.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
BASE="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"

if [ ! -f "$HERE/kokoro-v1.0.onnx" ]; then
  echo "downloading kokoro-v1.0.onnx (~310 MB)..."
  curl -L --fail --progress-bar -o "$HERE/kokoro-v1.0.onnx" "$BASE/kokoro-v1.0.onnx"
else
  echo "kokoro-v1.0.onnx already present, skipping"
fi

if [ ! -f "$HERE/voices-v1.0.bin" ]; then
  echo "downloading voices-v1.0.bin (~27 MB)..."
  curl -L --fail --progress-bar -o "$HERE/voices-v1.0.bin" "$BASE/voices-v1.0.bin"
else
  echo "voices-v1.0.bin already present, skipping"
fi

echo "done. files in $HERE:"
ls -lh "$HERE"/kokoro-v1.0.onnx "$HERE"/voices-v1.0.bin
