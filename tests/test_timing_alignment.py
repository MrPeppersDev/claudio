#!/usr/bin/env python3
"""
tests/test_timing_alignment.py — pytest cases for kokoro/timing.py.

Tests cover:
  - Simple phrase (monotonic ms, coverage)
  - Number expansion ("2024" → multiple phoneme groups, still 8 source words)
  - Contraction ("It's", "isn't") word count
  - Abbreviation ("Dr.", "Ave.") word count
  - Sacrificial head-word excluded from sidecar, words[0].start_ms == 0
  - Trim/pad invariant: words[-1].end_ms <= audio_samples / sample_rate * 1000

The heavy dependencies (phonemizer, onnxruntime) are stubbed so tests run
without a model or GPU, using the same pattern as tests/test_server_trim.py.

Because phonemizer is stubbed, all "phonemization" returns a fixed predictable
mapping — enough to exercise the alignment and timing conversion logic without
needing espeak installed.
"""

import sys
import os
import types
import math
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

# Allow running from repo root or from inside tests/.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# Stub heavy server.py dependencies before importing anything from kokoro/.
# ---------------------------------------------------------------------------

_scipy_signal = types.ModuleType("scipy.signal")
_scipy_signal.lfilter = lambda b, a, x: x
_scipy = types.ModuleType("scipy")
_scipy.signal = _scipy_signal
sys.modules.setdefault("scipy", _scipy)
sys.modules.setdefault("scipy.signal", _scipy_signal)

_kokoro_onnx = types.ModuleType("kokoro_onnx")


class _FakeKokoro:
    def __init__(self, *a, **kw):
        pass

    def get_voices(self):
        return ["af_bella"]

    def get_voice_style(self, name):
        return np.zeros(64, dtype="float32")


_kokoro_onnx.Kokoro = _FakeKokoro
sys.modules.setdefault("kokoro_onnx", _kokoro_onnx)

# ---------------------------------------------------------------------------
# Stub phonemizer so tests run without espeak.
#
# Strategy: the stub maps each word to a small fixed phoneme string using a
# deterministic rule (first char + "ə" per remaining char).  The whole-phrase
# phonemization is the concatenation of per-word results.  This keeps the
# alignment trivial (each per-word token run appears verbatim in the phrase
# token list) so we can verify word counts and monotonicity without needing
# real IPA output.
#
# Special cases for the tests that need specific word counts:
#   "2024"   → "tuːθaʊzəndtwentifɔː"  (20 chars, simulating expansion)
#   "CO2"    → "siːoʊtuː"  (9 chars, simulating expansion)
#   "It's"   → "ɪts"  (contraction → 1 group)
#   "isn't"  → "ɪznt"  (contraction → 1 group)
#   "Dr."    → "dɒktə"
#   "Ave."   → "ævɪnjuː"
# ---------------------------------------------------------------------------

_WORD_PHONEME_MAP = {
    "2024": "tuːθaʊzəndtwentifɔː",
    "CO2": "siːoʊtuː",
    "It's": "ɪts",
    "isn't": "ɪznt",
    "Dr.": "dɒktə",
    "Ave.": "ævɪnjuː",
}


def _fake_word_phonemes(word: str) -> str:
    """Return a deterministic IPA-like string for a word."""
    if word in _WORD_PHONEME_MAP:
        return _WORD_PHONEME_MAP[word]
    # Simple rule: first char + "ə" for each subsequent char.
    if not word:
        return "ə"
    result = word[0]
    for _ in word[1:]:
        result += "ə"
    return result


def _fake_phonemize(texts, language="en-us", backend="espeak",
                    preserve_punctuation=True, with_stress=True,
                    language_switch="remove-flags"):
    """Stub phonemize(): accepts list or str, returns same type."""
    if isinstance(texts, str):
        # Whole-phrase mode: phonemize each whitespace-separated word and join.
        words = texts.split()
        return "".join(_fake_word_phonemes(w) for w in words)
    else:
        # List mode: return one phoneme string per item.
        return [_fake_word_phonemes(w) for w in texts]


# Patch the phonemizer module before importing timing.py.
_phonemizer_mod = types.ModuleType("phonemizer")
_phonemizer_mod.phonemize = _fake_phonemize
_phonemizer_backend_mod = types.ModuleType("phonemizer.backend")


class _FakeEspeakBackend:
    pass


_phonemizer_backend_mod.EspeakBackend = _FakeEspeakBackend
_phonemizer_mod.backend = _phonemizer_backend_mod
sys.modules["phonemizer"] = _phonemizer_mod
sys.modules["phonemizer.backend"] = _phonemizer_backend_mod

# Patch _PHONEMIZER_AVAILABLE so timing.py doesn't bail early.
# Import timing.py now that stubs are in place.
from kokoro.timing import (  # noqa: E402
    compute_word_timings,
    _align_words_to_phrase,
    _phonemize_list,
    _phonemize_phrase,
    SAMPLES_PER_FRAME,
)

# Override the availability flag so the real code path runs.
import kokoro.timing as _timing_module
_timing_module._PHONEMIZER_AVAILABLE = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SR = 24_000


def _make_gather(words: list[str], frames_per_phoneme: int = 3) -> np.ndarray:
    """
    Build a fake gather (per-phoneme frame count) array for a list of words.

    Each word contributes len(phonemes_for_word) phoneme tokens, and each
    token gets `frames_per_phoneme` frames.  The result is a 1-D INT64 array
    of length == total phoneme tokens.
    """
    tokens: list[int] = []
    for w in words:
        ph = _fake_word_phonemes(w)
        tokens.extend([frames_per_phoneme] * len(ph))
    return np.array(tokens, dtype=np.int64)


def _audio_samples_for_words(words: list[str], frames_per_phoneme: int = 3) -> int:
    gather = _make_gather(words, frames_per_phoneme)
    return int(np.sum(gather) * SAMPLES_PER_FRAME)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestSimplePhrase:
    """Simple 9-word phrase: monotonic ms, last end_ms <= audio length."""

    WORDS = "The quick brown fox jumps over the lazy dog.".split()

    def test_word_count(self):
        gather = _make_gather(self.WORDS)
        result = compute_word_timings(" ".join(self.WORDS), gather, lang="en-us")
        assert len(result) == 9, f"Expected 9 words, got {len(result)}"

    def test_monotonic_start_ms(self):
        gather = _make_gather(self.WORDS)
        result = compute_word_timings(" ".join(self.WORDS), gather, lang="en-us")
        starts = [w["start_ms"] for w in result]
        for i in range(1, len(starts)):
            assert starts[i] >= starts[i - 1], (
                f"start_ms not monotonic at index {i}: {starts}"
            )

    def test_monotonic_end_ms(self):
        gather = _make_gather(self.WORDS)
        result = compute_word_timings(" ".join(self.WORDS), gather, lang="en-us")
        ends = [w["end_ms"] for w in result]
        for i in range(1, len(ends)):
            assert ends[i] >= ends[i - 1], (
                f"end_ms not monotonic at index {i}: {ends}"
            )

    def test_last_end_within_audio(self):
        gather = _make_gather(self.WORDS)
        audio_samples = _audio_samples_for_words(self.WORDS)
        result = compute_word_timings(" ".join(self.WORDS), gather, lang="en-us")
        audio_ms = audio_samples / SR * 1000
        last_end = result[-1]["end_ms"]
        assert last_end <= audio_ms + 1, (  # +1ms tolerance for rounding
            f"last end_ms={last_end} > audio_ms={audio_ms}"
        )


class TestNumberExpansion:
    """
    "2024 was a big year for CO2 emissions." → 8 source words.
    Verifies alignment handles number-expansion (2024→multi-group) and
    word-fusion ("was a"→fused) without dropping or merging source words.
    """

    TEXT = "2024 was a big year for CO2 emissions."
    WORDS = TEXT.split()  # ['2024', 'was', 'a', 'big', 'year', 'for', 'CO2', 'emissions.']

    def test_source_word_count_is_8(self):
        assert len(self.WORDS) == 8, f"Expected 8 source words, got {len(self.WORDS)}"

    def test_result_has_8_entries(self):
        gather = _make_gather(self.WORDS)
        result = compute_word_timings(self.TEXT, gather, lang="en-us")
        assert len(result) == 8, (
            f"Expected 8 entries, got {len(result)}: {[w['text'] for w in result]}"
        )

    def test_word_texts_preserved(self):
        gather = _make_gather(self.WORDS)
        result = compute_word_timings(self.TEXT, gather, lang="en-us")
        result_texts = [w["text"] for w in result]
        assert result_texts == self.WORDS, (
            f"Text mismatch: {result_texts} != {self.WORDS}"
        )

    def test_no_negative_durations(self):
        gather = _make_gather(self.WORDS)
        result = compute_word_timings(self.TEXT, gather, lang="en-us")
        for w in result:
            assert w["end_ms"] >= w["start_ms"], (
                f"Negative duration for word {w['text']!r}: "
                f"start={w['start_ms']} end={w['end_ms']}"
            )


class TestContraction:
    """
    "It's a well-known fact, isn't it?" → 6 source words.
    """

    TEXT = "It's a well-known fact, isn't it?"
    WORDS = TEXT.split()  # ["It's", "a", "well-known", "fact,", "isn't", "it?"]

    def test_source_word_count_is_6(self):
        assert len(self.WORDS) == 6

    def test_result_has_6_entries(self):
        gather = _make_gather(self.WORDS)
        result = compute_word_timings(self.TEXT, gather, lang="en-us")
        assert len(result) == 6, (
            f"Expected 6 entries, got {len(result)}: {[w['text'] for w in result]}"
        )

    def test_word_texts_preserved(self):
        gather = _make_gather(self.WORDS)
        result = compute_word_timings(self.TEXT, gather, lang="en-us")
        assert [w["text"] for w in result] == self.WORDS


class TestAbbreviation:
    """
    "Dr. Smith lives on 5th Ave." → 6 source words.
    """

    TEXT = "Dr. Smith lives on 5th Ave."
    WORDS = TEXT.split()  # ["Dr.", "Smith", "lives", "on", "5th", "Ave."]

    def test_source_word_count_is_6(self):
        assert len(self.WORDS) == 6

    def test_result_has_6_entries(self):
        gather = _make_gather(self.WORDS)
        result = compute_word_timings(self.TEXT, gather, lang="en-us")
        assert len(result) == 6, (
            f"Expected 6 entries, got {len(result)}: {[w['text'] for w in result]}"
        )

    def test_word_texts_preserved(self):
        gather = _make_gather(self.WORDS)
        result = compute_word_timings(self.TEXT, gather, lang="en-us")
        assert [w["text"] for w in result] == self.WORDS


class TestSacrificialBanana:
    """
    compute_word_timings is called on the real text (no banana).
    Verify: "banana" not in result, words[0].start_ms == 0 (no pad).
    """

    REAL_TEXT = "Recap the highlights of last week."
    REAL_WORDS = REAL_TEXT.split()

    def test_no_banana_in_result(self):
        gather = _make_gather(self.REAL_WORDS)
        result = compute_word_timings(self.REAL_TEXT, gather, lang="en-us")
        texts = [w["text"] for w in result]
        assert "banana" not in texts, f"banana leaked into sidecar: {texts}"

    def test_first_word_starts_at_zero_no_pad(self):
        gather = _make_gather(self.REAL_WORDS)
        result = compute_word_timings(
            self.REAL_TEXT, gather, lang="en-us", pad_start_ms=0.0
        )
        assert result[0]["start_ms"] == 0, (
            f"First word should start at 0ms, got {result[0]['start_ms']}"
        )

    def test_first_word_starts_at_pad_offset(self):
        """If pad_start_ms=150, first word start_ms >= 150."""
        gather = _make_gather(self.REAL_WORDS)
        result = compute_word_timings(
            self.REAL_TEXT, gather, lang="en-us", pad_start_ms=150.0
        )
        assert result[0]["start_ms"] >= 150, (
            f"First word should start at >= 150ms (pad), got {result[0]['start_ms']}"
        )


class TestTrimPadInvariant:
    """
    words[-1].end_ms <= audio_samples / sample_rate * 1000.
    """

    WORDS = "Hello world this is a test sentence here.".split()

    def test_last_end_within_audio_length(self):
        gather = _make_gather(self.WORDS)
        audio_samples = _audio_samples_for_words(self.WORDS)
        result = compute_word_timings(
            " ".join(self.WORDS), gather,
            lang="en-us", sample_rate=SR, pad_start_ms=0.0
        )
        audio_ms = audio_samples / SR * 1000.0
        last_end = result[-1]["end_ms"]
        assert last_end <= audio_ms + 1.0, (
            f"last end_ms={last_end} > audio_ms={audio_ms:.1f}"
        )

    def test_with_pad_still_within_padded_audio(self):
        """With pad_start_ms, timing shifts but must stay within padded audio."""
        pad_ms = 150.0
        gather = _make_gather(self.WORDS)
        audio_samples = (
            _audio_samples_for_words(self.WORDS)
            + int(round(SR * pad_ms / 1000.0))
        )
        result = compute_word_timings(
            " ".join(self.WORDS), gather,
            lang="en-us", sample_rate=SR, pad_start_ms=pad_ms
        )
        audio_ms = audio_samples / SR * 1000.0
        last_end = result[-1]["end_ms"]
        assert last_end <= audio_ms + 1.0, (
            f"last end_ms={last_end} > padded audio_ms={audio_ms:.1f}"
        )


class TestAlignmentEdgeCases:
    """
    Unit tests for the alignment module internals.
    """

    def test_empty_text_returns_empty(self):
        gather = np.array([3, 3, 3], dtype=np.int64)
        result = compute_word_timings("", gather, lang="en-us")
        assert result == []

    def test_single_word(self):
        word = "hello"
        gather = _make_gather([word])
        result = compute_word_timings(word, gather, lang="en-us")
        assert len(result) == 1
        assert result[0]["text"] == word
        assert result[0]["start_ms"] == 0

    def test_all_entries_have_required_keys(self):
        words = "Testing one two three.".split()
        gather = _make_gather(words)
        result = compute_word_timings(" ".join(words), gather, lang="en-us")
        for entry in result:
            assert "text" in entry
            assert "start_ms" in entry
            assert "end_ms" in entry

    def test_start_always_le_end(self):
        """No entry should have start_ms > end_ms."""
        words = "The quick brown fox jumps.".split()
        gather = _make_gather(words)
        result = compute_word_timings(" ".join(words), gather, lang="en-us")
        for w in result:
            assert w["start_ms"] <= w["end_ms"], (
                f"start > end for word {w['text']!r}: {w}"
            )
