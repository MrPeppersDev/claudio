#!/bin/bash
# kokoro/patch-model.sh — Idempotent ONNX model patcher.
#
# Produces kokoro/kokoro-v1.0-durations.onnx from kokoro/kokoro-v1.0.onnx by
# adding three additional graph outputs from the duration predictor encoder:
#
#   /encoder/Cast_output_0    INT64  — phoneme durations (cast output)
#   /encoder/Gather_output_0  INT64  — per-phoneme frame counts (Gather)
#   /encoder/CumSum_output_0  INT64  — cumulative frame counts (CumSum)
#
# These three outputs let timing.py convert per-source-word phoneme-index spans
# to exact sample offsets:  start_sample = sum(Gather[0:start_phoneme]) * 600
# at 24 kHz.  sum(all Gather) * 600 = len(audio) exactly (zero rounding error).
#
# INT64 is mandatory — UNDEFINED dtype causes onnxruntime to refuse the model
# at load time with a type-inference error.
#
# Idempotency: if the output file exists and is newer than the source, no-op.
# This makes the script safe to call on every server start.
#
# Usage:
#   ./kokoro/patch-model.sh
#   KOKORO_DIR=/path/to/kokoro ./kokoro/patch-model.sh  (non-default location)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-$SCRIPT_DIR/venv/bin/python}"
SRC="${KOKORO_MODEL_PATH:-$SCRIPT_DIR/kokoro-v1.0.onnx}"
DST="${KOKORO_DURATIONS_MODEL_PATH:-$SCRIPT_DIR/kokoro-v1.0-durations.onnx}"

if [ ! -r "$SRC" ]; then
    echo "[patch-model] ERROR: source model not found: $SRC" >&2
    exit 1
fi

# Idempotency check: skip if output exists and is newer than source.
if [ -f "$DST" ] && [ "$DST" -nt "$SRC" ]; then
    echo "[patch-model] already up to date: $DST" >&2
    exit 0
fi

echo "[patch-model] patching $SRC -> $DST" >&2

"$PYTHON" - "$SRC" "$DST" <<'PYTHON'
import sys
import onnx
from onnx import TensorProto, helper

src_path, dst_path = sys.argv[1], sys.argv[2]

model = onnx.load(src_path)
graph = model.graph

# Names of the three duration-predictor nodes we want to expose.
# All three are INT64 (confirmed by feasibility study: sum(Gather)*600 = len(audio)
# with zero rounding error at 24 kHz).
new_output_names = [
    "/encoder/Cast_output_0",
    "/encoder/Gather_output_0",
    "/encoder/CumSum_output_0",
]

# Build a set of names already in graph outputs so we don't double-add.
existing_output_names = {o.name for o in graph.output}

added = []
for name in new_output_names:
    if name in existing_output_names:
        continue
    # Verify the node actually exists in the graph (fail loudly if model
    # structure changed so the output name is no longer valid).
    node_found = any(
        name in node.output
        for node in graph.node
    )
    if not node_found:
        print(f"[patch-model] ERROR: node output '{name}' not found in graph", file=sys.stderr)
        sys.exit(1)
    # None shape = dynamic/unknown — onnxruntime will infer at runtime.
    vi = helper.make_tensor_value_info(name, TensorProto.INT64, None)
    graph.output.append(vi)
    added.append(name)

onnx.save(model, dst_path)
print(f"[patch-model] added outputs: {added}", file=sys.stderr)
print(f"[patch-model] saved: {dst_path}", file=sys.stderr)
PYTHON
