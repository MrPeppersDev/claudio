#!/bin/bash
# Downloads Kokoro v1.0 model weights + voices from the upstream GitHub release
# and verifies SHA-256 before use. Idempotent — skips if the expected file is
# already present and verifies clean.
#
# Why we check: the ONNX file is loaded by onnxruntime, and malicious models
# have historically been able to abuse runtime bugs. Pinning a checksum means
# forks that change BASE or a tampered upstream release get caught here rather
# than silently propagated to every installer downstream.
#
# If upstream ever re-publishes these files under the same release URL, the
# hash below has to be updated in the same PR that retests synthesis.

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
BASE="https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"

# SHA-256 of the upstream files at release tag "model-files-v1.0".
# Bump both together if the pinned release ever changes.
ONNX_SHA256="7d5df8ecf7d4b1878015a32686053fd0eebe2bc377234608764cc0ef3636a6c5"
VOICES_SHA256="bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d"

verify() {
  local path="$1" expected="$2"
  local actual
  actual=$(shasum -a 256 "$path" | awk '{print $1}')
  if [ "$actual" != "$expected" ]; then
    echo "ERROR: checksum mismatch for $path"
    echo "  expected: $expected"
    echo "  actual:   $actual"
    echo "  (file removed — re-run to retry, or update the pinned hash if"
    echo "   upstream has legitimately republished the release)"
    rm -f "$path"
    return 1
  fi
}

fetch_verified() {
  local name="$1"
  local expected="$2"
  local path="$HERE/$name"

  if [ -f "$path" ] && verify "$path" "$expected" 2>/dev/null; then
    echo "$name already present and verified, skipping"
    return 0
  fi

  echo "downloading $name..."
  curl -L --fail --progress-bar -o "$path" "$BASE/$name"
  verify "$path" "$expected"
  echo "$name: sha256 verified"
}

fetch_verified "kokoro-v1.0.onnx" "$ONNX_SHA256"
fetch_verified "voices-v1.0.bin" "$VOICES_SHA256"

echo "done. files in $HERE:"
ls -lh "$HERE"/kokoro-v1.0.onnx "$HERE"/voices-v1.0.bin
