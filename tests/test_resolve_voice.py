#!/usr/bin/env python3
"""tests/test_resolve_voice.py — pytest cases for resolve_voice().

Covers the blend path (the only place resolve_voice calls
KOKORO.get_voice_style), the normalization invariant, and rejection
paths: unknown voice, zero-total weight, blend cap.

The blend math previously had zero coverage despite being the only path
through KOKORO.get_voice_style. A regression there silently produces
a malformed style vector that downstream synthesis tries to use.

Run with:
    python -m pytest tests/test_resolve_voice.py -v
"""
import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Additive stub: re-use any scipy.signal module already in sys.modules so
# multiple test files can install their own attributes without clobbering.
_scipy_signal = sys.modules.setdefault("scipy.signal", types.ModuleType("scipy.signal"))
_scipy_signal.lfilter = lambda b, a, x: x
_scipy = sys.modules.setdefault("scipy", types.ModuleType("scipy"))
_scipy.signal = _scipy_signal


# Multi-voice fake so resolve_voice can verify blends across known voices.
# Each get_voice_style returns a deterministic 64-dim vector keyed by voice
# name so the test can verify weighted-sum math, not just type/shape.
class _FakeKokoro:
    def __init__(self, *a, **kw):
        pass

    def get_voices(self):
        return ["af_bella", "am_michael", "bf_emma"]

    def get_voice_style(self, name):
        import numpy as np
        # Distinct constant per voice so weighted-sum produces a known value.
        voice_value = {"af_bella": 1.0, "am_michael": 2.0, "bf_emma": 3.0}[name]
        return np.full(64, voice_value, dtype="float32")


_kokoro_onnx = types.ModuleType("kokoro_onnx")
_kokoro_onnx.Kokoro = _FakeKokoro
sys.modules.setdefault("kokoro_onnx", _kokoro_onnx)

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import kokoro.server as server_mod  # noqa: E402
from kokoro.server import resolve_voice, MAX_BLEND_ENTRIES  # noqa: E402


# Whichever test file imported `kokoro.server` first wins the import cache,
# so server_mod.VOICES and server_mod.KOKORO depend on collection order. We
# monkeypatch both per-test so the blend math is testable regardless of who
# ran first. The original values are restored after each test.
@pytest.fixture(autouse=True)
def _patch_voices_and_kokoro(monkeypatch):
    class _BlendFakeKokoro:
        def get_voices(self):
            return ["af_bella", "am_michael", "bf_emma"]

        def get_voice_style(self, name):
            voice_value = {"af_bella": 1.0, "am_michael": 2.0, "bf_emma": 3.0}[name]
            return np.full(64, voice_value, dtype="float32")

    monkeypatch.setattr(server_mod, "VOICES", {"af_bella", "am_michael", "bf_emma"})
    monkeypatch.setattr(server_mod, "KOKORO", _BlendFakeKokoro())


class TestResolveVoicePassthrough:
    def test_plain_name_returns_string(self):
        out = resolve_voice("af_bella")
        # Plain name path bypasses validation so the synth step does it.
        assert out == "af_bella"

    def test_plain_name_unknown_still_passes_through(self):
        # Unknown voice in plain-name path returns the string unchanged —
        # downstream KOKORO.create() will reject it. This is by design;
        # don't break that contract by adding a check here.
        out = resolve_voice("nope_not_real")
        assert out == "nope_not_real"


class TestResolveVoiceBlend:
    def test_two_way_equal_weight_returns_ndarray(self):
        out = resolve_voice("af_bella:1,am_michael:1")
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.float32
        assert out.shape == (64,)

    def test_normalization_two_way(self):
        # 1.0 * 0.5 + 2.0 * 0.5 = 1.5
        out = resolve_voice("af_bella:1,am_michael:1")
        assert np.allclose(out, 1.5)

    def test_normalization_unequal_weights(self):
        # 1.0 * (70/100) + 2.0 * (30/100) = 0.7 + 0.6 = 1.3
        out = resolve_voice("af_bella:70,am_michael:30")
        assert np.allclose(out, 1.3)

    def test_three_way_equal(self):
        # (1.0 + 2.0 + 3.0) / 3 = 2.0
        out = resolve_voice("af_bella:1,am_michael:1,bf_emma:1")
        assert np.allclose(out, 2.0)

    def test_weights_neednt_sum_to_one(self):
        # Should normalize: 1*(2/4) + 2*(2/4) = 1.5 — same as ":1,:1".
        out = resolve_voice("af_bella:2,am_michael:2")
        assert np.allclose(out, 1.5)


class TestResolveVoiceRejection:
    def test_unknown_voice_in_blend(self):
        with pytest.raises(ValueError, match="unknown voice"):
            resolve_voice("af_bella:1,nope:1")

    def test_zero_total_weight(self):
        with pytest.raises(ValueError, match="positive number"):
            resolve_voice("af_bella:0,am_michael:0")

    def test_negative_weights_summing_to_zero(self):
        with pytest.raises(ValueError, match="positive number"):
            resolve_voice("af_bella:1,am_michael:-1")

    def test_too_many_blend_entries(self):
        spec = ",".join(["af_bella:1"] * (MAX_BLEND_ENTRIES + 1))
        with pytest.raises(ValueError, match="too many blend entries"):
            resolve_voice(spec)

    def test_at_cap_accepted(self):
        # Exactly at the cap should still work — boundary condition.
        spec = ",".join(["af_bella:1"] * MAX_BLEND_ENTRIES)
        out = resolve_voice(spec)
        assert isinstance(out, np.ndarray)
