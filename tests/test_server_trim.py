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
    trim_sacrificial_head,
    TAIL_RMS_THRESHOLD,
    TAIL_RMS_FRAME_SAMPLES,
    TAIL_RMS_DWELL_FRAMES,
    HEAD_TRIM_DWELL_FRAMES,
    HEAD_TRIM_MIN_CUT_AT_1X_MS,
    HEAD_TRIM_MAX_CUT_AT_1X_MS,
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


# ---------------------------------------------------------------------------
# trim_sacrificial_head tests
#
# The detector's job is to find the post-prefix comma gap and cut there. It
# runs under three guardrails: relative RMS threshold (% of peak), min/max
# cut-position bounds (scaled by synth speed), and a dwell requirement. The
# tests below exercise the combinations that actually come up in production:
# detecting a compressed comma gap, ignoring intra-prefix dips, and failing
# open when the first qualifying gap is past the real first word.
# ---------------------------------------------------------------------------

class TestTrimSacrificialHead:
    # Speed=1.0 keeps the arithmetic readable — min_cut = 430ms, max_cut =
    # 800ms, so a "gap at 480ms" is unambiguously post-prefix and a "gap at
    # 850ms" is unambiguously past-the-real-word.
    DEFAULT_SPEED = 1.0

    def test_empty_input(self):
        """Empty array → return immediately."""
        empty = np.array([], dtype=np.float32)
        result = trim_sacrificial_head(empty, SR, speed=self.DEFAULT_SPEED)
        assert result.size == 0
        assert result is empty

    def test_signal_shorter_than_dwell(self):
        """A signal shorter than the dwell window → return unchanged."""
        sig = _tone(10)  # 10ms, well under dwell=30ms
        result = trim_sacrificial_head(sig, SR, speed=self.DEFAULT_SPEED)
        assert result is sig

    def test_basic_gap_detected(self):
        """
        Classic shape: prefix audible, comma gap, real word audible. The cut
        should land inside the comma gap so the returned array starts with
        (at most) a little leading silence and then the real word.
        """
        prefix = _tone(500)      # 500ms of "banana"
        gap = _silence(100)      # 100ms comma gap
        word = _tone(400, freq=660)
        sig = np.concatenate([prefix, gap, word])

        result = trim_sacrificial_head(sig, SR, speed=self.DEFAULT_SPEED)

        # Result must be shorter than input (we cut something).
        assert result.size < sig.size
        # The cut should happen somewhere inside the comma gap, i.e. the
        # returned prefix should be << the original 500ms prefix.
        #   sig.size - result.size ≈ samples cut from the head.
        cut_ms = (sig.size - result.size) * 1000.0 / SR
        assert 430 <= cut_ms <= 600, (
            f"Cut at {cut_ms:.0f}ms — expected between min_cut (430ms) "
            f"and end of comma gap (~600ms)."
        )

    def test_compressed_gap_still_detected(self):
        """
        At fast synth the comma gap can compress to 40-50ms — narrower than
        the old 60ms dwell but still above the 30ms dwell. The detector
        should still fire.
        """
        prefix = _tone(500)
        gap = _silence(40)    # compressed — above 30ms dwell, below old 60ms
        word = _tone(400, freq=660)
        sig = np.concatenate([prefix, gap, word])

        result = trim_sacrificial_head(sig, SR, speed=self.DEFAULT_SPEED)
        assert result.size < sig.size, (
            "Compressed 40ms comma gap should still trigger a trim"
        )

    def test_intra_prefix_dip_ignored(self):
        """
        A silent dip *inside* the prefix word (before min_cut_frame) must
        not cause a cut. The real comma gap further in should be what
        triggers the trim.

        Layout: 150ms loud, 40ms dip, 250ms loud (total 440ms "banana"),
        then 100ms real comma gap, then the real word. Without the
        min-cut guardrail, the 40ms dip at 150-190ms would have fired a
        cut inside the prefix.
        """
        loud_a = _tone(150)
        dip = _silence(40)
        loud_b = _tone(250)
        gap = _silence(100)
        word = _tone(400, freq=660)
        sig = np.concatenate([loud_a, dip, loud_b, gap, word])

        result = trim_sacrificial_head(sig, SR, speed=self.DEFAULT_SPEED)

        cut_ms = (sig.size - result.size) * 1000.0 / SR
        # Cut must be beyond min_cut_at_1x (430ms), not around the dip (~190ms).
        assert cut_ms >= HEAD_TRIM_MIN_CUT_AT_1X_MS, (
            f"Cut at {cut_ms:.0f}ms fell inside the prefix — the intra-word "
            f"dip at ~190ms should not have triggered a cut."
        )

    def test_no_gap_fails_open(self):
        """All audible through the search window → samples returned unchanged."""
        sig = _tone(1200)
        result = trim_sacrificial_head(sig, SR, speed=self.DEFAULT_SPEED)
        assert result is sig

    def test_late_gap_fails_open(self):
        """
        First qualifying silent run lives past max_cut. Cutting there would
        chop the real first word out (this is the "pros/cons got skipped"
        failure mode when the comma gap was missed). Detector must fail
        open instead.

        Layout: long audible stretch (prefix + real first word run together
        — comma gap below threshold), then a pause. The pause has to start
        past HEAD_TRIM_MAX_CUT_AT_1X_MS for the safety net to trip.
        """
        audible = _tone(HEAD_TRIM_MAX_CUT_AT_1X_MS + 100)  # past max_cut
        late_gap = _silence(200)
        trailing = _tone(300, freq=660)
        sig = np.concatenate([audible, late_gap, trailing])

        result = trim_sacrificial_head(sig, SR, speed=self.DEFAULT_SPEED)
        assert result is sig, (
            "Late gap (past max_cut) should fail open, not cut — cutting "
            "there would eat the real first word."
        )

    def test_speed_scales_cut_bounds(self):
        """
        At higher synth speed the prefix takes less real time. min_cut and
        max_cut both scale inversely. A gap that would fail open at 1.0x
        (because it's past max_cut=800ms) should still fail open at 2.0x
        (max_cut=400ms) — just for a different reason — but a gap at 500ms
        that fails open at 1.0x should *not* fail open at 1.0x where it's
        clearly inside the expected window. This test specifically checks
        that doubling speed moves min_cut down: a gap ending at ~260ms
        that's *ignored* at 1.0x (inside min_cut=430ms) gets *cut* at 2.0x
        (min_cut=215ms).
        """
        prefix = _tone(200)   # 200ms — half the 1x prefix duration
        gap = _silence(80)    # comma gap ending at 280ms
        word = _tone(400, freq=660)
        sig = np.concatenate([prefix, gap, word])

        # At 1.0x the gap at 200-280ms is well inside min_cut=430ms — no cut.
        at_1x = trim_sacrificial_head(sig, SR, speed=1.0)
        assert at_1x is sig, (
            "Gap inside 1x min_cut region should not be cut at 1.0x"
        )
        # At 2.0x min_cut = 215ms. The gap starts at 200ms; dwell (30ms) is
        # met at ~230ms, past min_cut. Should cut.
        at_2x = trim_sacrificial_head(sig, SR, speed=2.0)
        assert at_2x.size < sig.size, (
            "At 2.0x speed the scaled min_cut should allow cutting the "
            "~280ms gap."
        )
