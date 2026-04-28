#!/usr/bin/env python3
"""tests/test_chunk_phonemes.py — pytest cases for chunk_phonemes() and the
helpers it composes (_split_keeping_delims, _chunk_on_clause,
_chunk_on_whitespace, _greedy_pack).

These functions implement the ≤500-phoneme safety net for Kokoro-82M's
510-phoneme IndexError on long unpunctuated spans. A bug here either
crashes the model on long inputs or silently corrupts audio at chunk
seams. Production has zero coverage of the chunking path otherwise.

Run with:
    python -m pytest tests/test_chunk_phonemes.py -v
"""
import os
import sys
import types
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Additive stub: re-use any scipy.signal module already in sys.modules so
# multiple test files can install their own attributes without clobbering
# each other via setdefault no-ops.
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

import pytest  # noqa: E402
from kokoro.server import (  # noqa: E402
    chunk_phonemes,
    _split_keeping_delims,
    _chunk_on_clause,
    _chunk_on_whitespace,
    _greedy_pack,
    MAX_PHONEMES_PER_CALL,
)


class TestChunkPhonemesPassthrough:
    def test_short_input_single_chunk(self):
        text = "hello world"
        out = chunk_phonemes(text, limit=20)
        assert out == ["hello world"]

    def test_exact_limit_single_chunk(self):
        text = "x" * 50
        out = chunk_phonemes(text, limit=50)
        assert out == [text]

    def test_default_limit_constant(self):
        """Default arg uses MAX_PHONEMES_PER_CALL (the model-derived 500)."""
        text = "x" * (MAX_PHONEMES_PER_CALL - 1)
        out = chunk_phonemes(text)
        assert out == [text]


class TestChunkPhonemesSentenceSplit:
    def test_splits_on_period(self):
        # Two sentences, each ~30 chars, total > limit
        text = ("a" * 30) + ". " + ("b" * 30) + ". " + ("c" * 30)
        out = chunk_phonemes(text, limit=40)
        # Each chunk must respect the limit
        for chunk in out:
            assert len(chunk) <= 40
        # Non-whitespace content survives chunking. _greedy_pack strips
        # boundary whitespace between chunks (each chunk gets synthesized
        # independently; inter-chunk whitespace doesn't matter), so the
        # join can lose spaces — alphanumerics and punctuation must not.
        joined = "".join(out)
        assert joined.replace(" ", "") == text.replace(" ", "")

    def test_splits_on_question(self):
        text = "question one? " * 10
        text = text.strip()
        out = chunk_phonemes(text, limit=40)
        for chunk in out:
            assert len(chunk) <= 40

    def test_splits_on_exclamation(self):
        text = "hey there! " * 10
        text = text.strip()
        out = chunk_phonemes(text, limit=40)
        for chunk in out:
            assert len(chunk) <= 40

    def test_chunks_packed_greedily(self):
        """Many short sentences should pack into FEW chunks, not many."""
        text = ("a." * 50)  # 100 chars total, lots of split points
        out = chunk_phonemes(text, limit=20)
        # 100 chars / 20 limit = 5 chunks minimum; greedy pack should hit that
        assert len(out) <= 6


class TestChunkPhonemesClauseFallback:
    def test_long_unpunctuated_falls_through_to_clause(self):
        # No sentence punct, but commas — the clause splitter should kick in
        text = ("aaa," * 30).rstrip(",")
        out = chunk_phonemes(text, limit=30)
        for chunk in out:
            assert len(chunk) <= 30
        # Boundary whitespace can be stripped between chunks; non-whitespace
        # content must round-trip.
        joined = "".join(out)
        assert joined.replace(" ", "") == text.replace(" ", "")


class TestChunkPhonemesWhitespaceFallback:
    def test_long_unpunctuated_with_spaces(self):
        # No punct, just spaces — words greedy-packed up to the limit
        words = ["wordA", "wordB", "wordC", "wordD", "wordE", "wordF"]
        text = " ".join(words)
        out = chunk_phonemes(text, limit=15)
        for chunk in out:
            assert len(chunk) <= 15

    def test_word_longer_than_limit_hard_cut(self):
        # Single token that exceeds the limit — only the hard-cut path
        # can rescue this (a phonemized URL, all-numeric ID, etc.).
        text = "x" * 100
        out = chunk_phonemes(text, limit=30)
        for chunk in out:
            assert len(chunk) <= 30
        # All chars survive the cut.
        assert "".join(out) == text


class TestSplitKeepingDelims:
    def test_basic_sentence_split(self):
        out = _split_keeping_delims("abc. def! ghi", r"[.!?]")
        assert out == ["abc.", " def!", " ghi"]

    def test_no_delims(self):
        out = _split_keeping_delims("no punctuation here", r"[.!?]")
        assert out == ["no punctuation here"]

    def test_empty_input(self):
        out = _split_keeping_delims("", r"[.!?]")
        assert out == []

    def test_only_delim(self):
        out = _split_keeping_delims(".", r"[.!?]")
        assert out == ["."]

    def test_clause_class(self):
        out = _split_keeping_delims("a, b; c: d", r"[,;:—–]")
        assert out == ["a,", " b;", " c:", " d"]


class TestChunkOnClause:
    def test_splits_on_commas_when_under_limit(self):
        # Long clause-bounded piece — comma split lets each fragment fit
        text = ("hello there, " * 5).rstrip(", ")
        out = _chunk_on_clause(text, limit=20)
        for chunk in out:
            assert len(chunk) <= 20

    def test_falls_through_to_whitespace_for_long_clause(self):
        # No clause punct, just words — must use whitespace split
        text = "one two three four five six seven eight"
        out = _chunk_on_clause(text, limit=15)
        for chunk in out:
            assert len(chunk) <= 15


class TestChunkOnWhitespace:
    def test_packs_words_under_limit(self):
        out = _chunk_on_whitespace("aa bb cc dd ee", limit=8)
        for chunk in out:
            assert len(chunk) <= 8
        assert " ".join(out).replace("  ", " ").strip() != ""

    def test_hard_cuts_oversized_word(self):
        # Single word longer than limit — hard-cut into chunks of `limit`
        out = _chunk_on_whitespace("x" * 50, limit=10)
        # All chunks except possibly the last are exactly `limit` chars
        for c in out[:-1]:
            assert len(c) == 10
        assert len(out[-1]) <= 10
        assert "".join(out) == "x" * 50

    def test_empty_string(self):
        out = _chunk_on_whitespace("", limit=10)
        assert out == [""]


class TestGreedyPack:
    def test_concatenates_under_limit(self):
        # Three short pieces all under limit — should combine into one
        out = _greedy_pack(["abc", "def", "ghi"], limit=20)
        assert len(out) == 1
        assert out[0] == "abcdefghi"

    def test_breaks_at_limit(self):
        out = _greedy_pack(["aaaa", "bbbb", "cccc"], limit=8)
        # 4 + 4 = 8 fits, +4 = 12 doesn't
        assert out == ["aaaabbbb", "cccc"]

    def test_oversized_piece_kept_intact(self):
        """A piece already over the limit is passed through whole — splitting
        should have happened upstream in _chunk_on_clause/_whitespace."""
        out = _greedy_pack(["short", "x" * 100, "another"], limit=20)
        assert "x" * 100 in out

    def test_empty_list(self):
        assert _greedy_pack([], limit=20) == []
