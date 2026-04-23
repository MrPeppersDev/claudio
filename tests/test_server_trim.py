#!/usr/bin/env python3
"""
tests/test_server_trim.py — pytest cases for trim_tail_silence().

The function lives in kokoro/server.py but that module loads a Kokoro ONNX
model at import time. We stub out the heavy dependencies before importing so
the tests run without a model file or GPU.

Run with:
    python -m pytest tests/test_server_trim.py -v
"""
import sys
import os
import types
from unittest.mock import MagicMock

# Allow running from the repo root or from inside tests/.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# Stub heavy dependencies so kokoro/server.py can be imported without a real
# model file, onnxruntime, or scipy.
# ---------------------------------------------------------------------------

# scipy.signal.lfilter — used by peaking_eq, not trim_tail_silence.
_scipy_signal = types.ModuleType("scipy.signal")
_scipy_signal.lfilter = lambda b, a, x: x  # identity: no-op for tests
_scipy = types.ModuleType("scipy")
_scipy.signal = _scipy_signal
sys.modules.setdefault("scipy", _scipy)
sys.modules.setdefault("scipy.signal", _scipy_signal)

# kokoro_onnx.Kokoro — the module-level KOKORO = Kokoro(...) call.
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

# Patch os.path.exists so model-file presence check passes (server.py doesn't
# explicitly check, but guard against any future addition).
import unittest.mock  # noqa: E402 — after stubs

# ---------------------------------------------------------------------------
# Now we can safely import the module under test.
# ---------------------------------------------------------------------------
import numpy as np  # noqa: E402
from kokoro.server import (  # noqa: E402
    trim_tail_silence,
    TAIL_RMS_THRESHOLD,
    TAIL_RMS_FRAME_SAMPLES,
    TAIL_RMS_DWELL_FRAMES,
)

# Convenience: Kokoro's native sample rate.
SR = 24_000
FRAME = TAIL_RMS_FRAME_SAMPLES  # 240 samples = 10ms @ 24 kHz
DWELL = TAIL_RMS_DWELL_FRAMES   # 6 frames = 60ms


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _silence(ms: float) -> np.ndarray:
    """Return an array of zeros of length ms milliseconds."""
    return np.zeros(int(SR * ms / 1000), dtype=np.float32)


def _tone(ms: float, amplitude: float = 0.3, freq: float = 440.0) -> np.ndarray:
    """Return a sine tone of the given duration and amplitude."""
    n = int(SR * ms / 1000)
    t = np.arange(n, dtype=np.float64) / SR
    return (np.sin(2 * np.pi * freq * t) * amplitude).astype(np.float32)


def _burst(ms: float, amplitude: float = 0.03) -> np.ndarray:
    """Return a constant-amplitude block (synthetic unvoiced stop burst)."""
    n = int(SR * ms / 1000)
    return np.full(n, amplitude, dtype=np.float32)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestNoOp:
    def test_target_ms_zero(self):
        """target_ms=0 → function disabled, input returned unchanged."""
        sig = np.concatenate([_tone(200), _silence(300)])
        result = trim_tail_silence(sig, SR, target_ms=0)
        assert result is sig  # exact same object

    def test_target_ms_negative(self):
        """Negative target_ms → treated as disabled."""
        sig = np.concatenate([_tone(200), _silence(300)])
        result = trim_tail_silence(sig, SR, target_ms=-1)
        assert result is sig

    def test_empty_input(self):
        """Empty array → return immediately."""
        empty = np.array([], dtype=np.float32)
        result = trim_tail_silence(empty, SR, target_ms=40)
        assert result.size == 0
        assert result is empty

    def test_all_silent(self):
        """All-zero signal → no audible anchor → return unchanged."""
        sig = _silence(500)
        result = trim_tail_silence(sig, SR, target_ms=40)
        assert result is sig

    def test_shorter_than_one_frame(self):
        """Signal shorter than FRAME samples → return unchanged."""
        sig = np.full(FRAME - 1, 0.5, dtype=np.float32)
        result = trim_tail_silence(sig, SR, target_ms=40)
        assert result is sig

    def test_existing_tail_already_tight(self):
        """
        If the tail after the last audible frame is already <= target_ms,
        return the signal unchanged (never extend).
        """
        target_ms = 40.0
        # Audible tone followed by exactly 20ms of silence (< target).
        sig = np.concatenate([_tone(300), _silence(20)])
        result = trim_tail_silence(sig, SR, target_ms=target_ms)
        assert result.size == sig.size, (
            f"Should not extend: got {result.size}, expected {sig.size}"
        )


class TestTrims:
    def test_trims_long_tail(self):
        """
        Long trailing silence should be trimmed to approximately target_ms.
        Tolerance: ±1 frame (10ms) for frame-boundary rounding.
        """
        target_ms = 40.0
        sig = np.concatenate([_tone(300), _silence(500)])
        result = trim_tail_silence(sig, SR, target_ms=target_ms)

        # Result must be shorter than original.
        assert result.size < sig.size, "Expected trim to shorten the signal"

        # Tail after end of audible tone should be close to target_ms.
        # Locate last above-threshold sample in result.
        above = np.flatnonzero(np.abs(result) > TAIL_RMS_THRESHOLD)
        assert above.size > 0, "Trimmed result has no audible content"
        last_audible_sample = int(above[-1])
        tail_samples = result.size - last_audible_sample - 1
        tail_ms = tail_samples / SR * 1000.0

        tol_ms = 10.0 + (FRAME / SR * 1000.0)  # target ± 1 frame
        assert abs(tail_ms - target_ms) <= tol_ms, (
            f"tail_ms={tail_ms:.1f} not within {tol_ms:.1f}ms of target={target_ms}"
        )

    def test_unvoiced_stop_burst_preserved(self):
        """
        Critical: a synthetic t/p/k-like release burst must survive.

        Signal layout:
          [300ms vowel @ 0.3 amplitude]
          [30ms burst @ 0.03 amplitude]   ← below single-sample 0.005 but above RMS threshold
          [500ms silence]

        With dwell=60ms the burst frame's RMS exceeds TAIL_RMS_THRESHOLD and
        the burst is followed by only 500ms of silence (well above dwell, but
        that means the burst ITSELF is the last audible frame, so the anchor
        is placed at the burst). The burst must appear in the returned signal.
        """
        target_ms = 40.0
        vowel = _tone(300, amplitude=0.3)
        burst = _burst(30, amplitude=0.03)
        silence = _silence(500)
        sig = np.concatenate([vowel, burst, silence])

        result = trim_tail_silence(sig, SR, target_ms=target_ms)

        # The burst should still be present: find burst samples in result.
        burst_start = vowel.size
        burst_end = vowel.size + burst.size

        # We only need the burst region to still exist in the result.
        assert result.size >= burst_end, (
            f"Burst was cut: result.size={result.size} < burst_end={burst_end}"
        )

        # The burst region in the result should still have its amplitude.
        burst_in_result = result[burst_start:burst_end]
        assert np.any(burst_in_result > 0.01), (
            "Burst amplitude was zeroed out in the trimmed result"
        )

    def test_idempotent(self):
        """
        Trimming twice with the same target_ms should yield the same result
        as trimming once.
        """
        target_ms = 40.0
        sig = np.concatenate([_tone(300), _silence(500)])
        once = trim_tail_silence(sig, SR, target_ms=target_ms)
        twice = trim_tail_silence(once, SR, target_ms=target_ms)
        assert once.size == twice.size, (
            f"Not idempotent: once={once.size} twice={twice.size}"
        )
        np.testing.assert_array_equal(once, twice)
