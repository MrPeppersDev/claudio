#!/usr/bin/env python3
"""
tests/test_head_trim.py — pytest cases for trim_sacrificial_head().

Mirrors the stub setup in test_server_trim.py so kokoro/server.py imports
without a real model. See that file for why the stubs are needed.

Run with:
    python -m pytest tests/test_head_trim.py -v
"""
import sys
import os
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Additive stub: re-use any scipy.signal module already in sys.modules so
# multiple test files can install their own attributes (lfilter for server
# tests, resample_poly for play-stream tests) without clobbering each other
# via setdefault no-ops.
_scipy_signal = sys.modules.setdefault("scipy.signal", types.ModuleType("scipy.signal"))
_scipy_signal.lfilter = lambda b, a, x: x
_scipy = sys.modules.setdefault("scipy", types.ModuleType("scipy"))
_scipy.signal = _scipy_signal

_kokoro_onnx = types.ModuleType("kokoro_onnx")


class _FakeKokoro:
    def __init__(self, *a, **kw):
        pass

    def get_voices(self):
        return ["af_bella"]

    def get_voice_style(self, name):
        import numpy as np
        return np.zeros(64, dtype="float32")


_kokoro_onnx.Kokoro = _FakeKokoro
sys.modules.setdefault("kokoro_onnx", _kokoro_onnx)

import numpy as np  # noqa: E402
from kokoro.server import (  # noqa: E402
    trim_sacrificial_head,
    HEAD_TRIM_DWELL_FRAMES,
    HEAD_TRIM_SEARCH_MS,
    TAIL_RMS_FRAME_SAMPLES,
)

SR = 24_000
FRAME = TAIL_RMS_FRAME_SAMPLES
DWELL = HEAD_TRIM_DWELL_FRAMES


def _silence(ms: float) -> np.ndarray:
    return np.zeros(int(SR * ms / 1000), dtype=np.float32)


def _tone(ms: float, amplitude: float = 0.3, freq: float = 440.0) -> np.ndarray:
    n = int(SR * ms / 1000)
    t = np.arange(n, dtype=np.float32) / SR
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class TestFailOpen:
    def test_empty_input(self):
        samples = np.array([], dtype=np.float32)
        out = trim_sacrificial_head(samples, SR)
        assert out.size == 0

    def test_no_gap_returns_unchanged(self):
        # Continuous tone through the whole search window — no silence to cut at.
        samples = _tone(HEAD_TRIM_SEARCH_MS + 200)
        out = trim_sacrificial_head(samples, SR)
        assert np.array_equal(out, samples)

    def test_all_silence_returns_unchanged(self):
        # No audible frame ever, so there's nothing to anchor the cut to.
        samples = _silence(HEAD_TRIM_SEARCH_MS + 200)
        out = trim_sacrificial_head(samples, SR)
        assert np.array_equal(out, samples)

    def test_audible_never_followed_by_dwell(self):
        # Short audible burst then a too-short silence before the window ends —
        # detector should not commit a trim.
        prefix = _tone(120)
        short_silence = _silence(30)  # 30ms < 60ms dwell
        tail_tone = _tone(50)
        samples = np.concatenate([prefix, short_silence, tail_tone])
        out = trim_sacrificial_head(samples, SR)
        assert np.array_equal(out, samples)


class TestTrims:
    def test_trims_prefix_at_gap(self):
        # Prefix must be ≥ HEAD_TRIM_MIN_CUT_AT_1X_MS (430ms) so the cut
        # offset clears the min-cut guardrail. Real "banana, " prefixes at
        # 1x speed land in the 430–700ms range; 500ms is a realistic value.
        prefix = _tone(500)          # "banana, "
        gap = _silence(100)          # post-comma pause, ≥20ms dwell
        body = _tone(500)            # "recap..."
        samples = np.concatenate([prefix, gap, body])

        out = trim_sacrificial_head(samples, SR)

        # The cut lands at the end of the silent dwell. Body should remain
        # essentially intact; prefix should be gone.
        assert out.size < samples.size
        # Body tone (500ms) should still be present — allow a 1-frame slop
        # for the frame-aligned cut.
        body_ms = out.size * 1000 / SR
        assert body_ms >= (500 - 20), f"body truncated: only {body_ms:.1f}ms remains"
        # The cut lands at the end of the silent DWELL (20ms into a 100ms
        # gap), so the first 80ms of the output are still silence — the
        # body tone starts right after that. Check a 200ms window so we
        # straddle the leftover silence and land in the body tone.
        early = out[: int(SR * 200 / 1000)]
        rms_early = float(np.sqrt(np.mean(early ** 2)))
        assert rms_early > 0.05, f"body tone missing after cut (rms={rms_early})"

    def test_idempotent_on_already_trimmed(self):
        # Apply once, then apply again — second pass has no gap to find
        # (body is pure tone with no silence), so it should fail open.
        prefix = _tone(300)
        gap = _silence(100)
        body = _tone(500)
        once = trim_sacrificial_head(
            np.concatenate([prefix, gap, body]), SR
        )
        twice = trim_sacrificial_head(once, SR)
        assert np.array_equal(once, twice)

    def test_short_prefix_then_gap(self):
        # Prefix at the lower edge of the realistic banana-prefix range —
        # just past HEAD_TRIM_MIN_CUT_AT_1X_MS (430ms). Verifies the
        # detector still cuts when the offset clears min_cut by a small
        # margin (different from the comfortable 500ms in the previous
        # test).
        prefix = _tone(440)
        gap = _silence(80)
        body = _tone(400)
        samples = np.concatenate([prefix, gap, body])
        out = trim_sacrificial_head(samples, SR)
        assert out.size < samples.size

    def test_input_shorter_than_one_frame(self):
        samples = np.zeros(FRAME // 2, dtype=np.float32)
        out = trim_sacrificial_head(samples, SR)
        assert np.array_equal(out, samples)
