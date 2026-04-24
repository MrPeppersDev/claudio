"""
kokoro/timing.py — per-source-word timing from Kokoro's duration predictor.

Public API
----------
compute_word_timings(
    source_text,        # the real text (no sacrificial head word)
    gather_output,      # np.ndarray INT64 from /encoder/Gather_output_0 (1-D)
    lang,               # phonemizer language, e.g. "en-us"
    sample_rate,        # usually 24000
    pad_start_ms,       # ms of silence prepended by the server
) -> list[dict]

Returns a list of dicts:
    [{"text": "word", "start_ms": 0, "end_ms": 800}, ...]

One entry per source word (whitespace-split), in order.  Punctuation attached
to the word is preserved ("Recap.", "year,").  The sacrificial head-word must
NOT be present in source_text — caller is responsible.

Design
------
Kokoro phonemizes the entire synth_text in one shot and the ONNX model emits
one frame-duration per phoneme token.  We need to map each source word's
phonemes back to that flat token stream to read off its start/end sample.

Problem: phonemizer in list-mode (one string per word) does not produce the
same tokens as phonemizing the whole phrase.  Connected-speech rules fuse
adjacent phonemes across word boundaries (e.g. "was a" → "wʌzɐ" in phrase
mode, but "wʌz" + "ˈeɪ" in per-word mode).  A simple token-count doesn't
survive this.

Solution: Needham–Wunsch (global sequence alignment) between:
  - flat_tokens   : list of IPA characters from phonemizing the whole phrase
  - per_word_tokens: list-of-lists of IPA characters from phonemizing each
                    word separately

Alignment scoring: match=+2, mismatch=-1, gap=-1.  This puts a high enough
premium on matching that vowel substitutions (ʌ→ɐ in "was a" fusion) score
lower than gaps, so the fused phoneme gets assigned to one of the two
adjacent words rather than silently dropped.

After alignment we convert aligned spans to sample offsets using:
    start_sample = sum(gather[0:start_token]) * 600
    end_sample   = sum(gather[0:end_token])   * 600

600 samples/frame is the hard-coded hop in Kokoro's duration predictor at
24 kHz.
"""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np

try:
    from phonemizer import phonemize as _phonemize
    from phonemizer.backend import EspeakBackend
    _PHONEMIZER_AVAILABLE = True
except ImportError:
    _PHONEMIZER_AVAILABLE = False

# Kokoro's duration predictor emits one duration per phoneme frame.
# Each frame = 600 samples at 24 kHz.
SAMPLES_PER_FRAME = 600

# Alignment scoring constants.
MATCH_SCORE = 2
MISMATCH_SCORE = -1
GAP_SCORE = -1


# ---------------------------------------------------------------------------
# Phonemization helpers
# ---------------------------------------------------------------------------

def _phonemize_list(words: list[str], lang: str) -> list[list[str]]:
    """
    Phonemize each word independently. Returns a list of lists, one inner
    list per source word, containing the IPA characters for that word.

    Preserves punctuation via preserve_punctuation=True so that
    "Recap." phonemizes as a complete token (the period may survive or be
    dropped by espeak; we just keep whatever espeak returns).
    """
    if not _PHONEMIZER_AVAILABLE:
        raise RuntimeError("phonemizer package is required for timing.py")

    results = _phonemize(
        words,
        language=lang,
        backend="espeak",
        preserve_punctuation=True,
        with_stress=True,
        language_switch="remove-flags",
    )
    # phonemize() with a list input returns a list of strings.
    # Split each string into individual IPA characters (not graphemes).
    return [list(s) for s in results]


def _phonemize_phrase(text: str, lang: str) -> list[str]:
    """
    Phonemize the whole phrase as one string. Returns a flat list of IPA
    characters exactly as the model tokenizer will see them.
    """
    if not _PHONEMIZER_AVAILABLE:
        raise RuntimeError("phonemizer package is required for timing.py")

    result = _phonemize(
        text,
        language=lang,
        backend="espeak",
        preserve_punctuation=True,
        with_stress=True,
        language_switch="remove-flags",
    )
    return list(result)


# ---------------------------------------------------------------------------
# Needham–Wunsch global sequence alignment
# ---------------------------------------------------------------------------

def _nw_align(seq_a: list, seq_b: list) -> tuple[list, list]:
    """
    Global sequence alignment (Needham–Wunsch) between seq_a and seq_b.

    Returns (aligned_a, aligned_b) where each element is either the original
    token or None (representing a gap).  len(aligned_a) == len(aligned_b).

    Scoring: MATCH_SCORE / MISMATCH_SCORE / GAP_SCORE (module-level constants).
    """
    n, m = len(seq_a), len(seq_b)

    # Build score matrix.
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + GAP_SCORE
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + GAP_SCORE

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match = dp[i - 1][j - 1] + (
                MATCH_SCORE if seq_a[i - 1] == seq_b[j - 1] else MISMATCH_SCORE
            )
            delete = dp[i - 1][j] + GAP_SCORE
            insert = dp[i][j - 1] + GAP_SCORE
            dp[i][j] = max(match, delete, insert)

    # Traceback.
    aligned_a: list = []
    aligned_b: list = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            score_diag = dp[i - 1][j - 1] + (
                MATCH_SCORE if seq_a[i - 1] == seq_b[j - 1] else MISMATCH_SCORE
            )
            score_up = dp[i - 1][j] + GAP_SCORE
            score_left = dp[i][j - 1] + GAP_SCORE
            if dp[i][j] == score_diag:
                aligned_a.append(seq_a[i - 1])
                aligned_b.append(seq_b[j - 1])
                i -= 1
                j -= 1
            elif dp[i][j] == score_up:
                aligned_a.append(seq_a[i - 1])
                aligned_b.append(None)
                i -= 1
            else:
                aligned_a.append(None)
                aligned_b.append(seq_b[j - 1])
                j -= 1
        elif i > 0:
            aligned_a.append(seq_a[i - 1])
            aligned_b.append(None)
            i -= 1
        else:
            aligned_a.append(None)
            aligned_b.append(seq_b[j - 1])
            j -= 1

    aligned_a.reverse()
    aligned_b.reverse()
    return aligned_a, aligned_b


# ---------------------------------------------------------------------------
# Core: map per-word phonemes onto the flat phrase token stream
# ---------------------------------------------------------------------------

def _align_words_to_phrase(
    per_word_tokens: list[list[str]],
    flat_tokens: list[str],
) -> list[tuple[int, int]]:
    """
    Return per-source-word [start, end) spans into flat_tokens.

    Strategy:
    1. Flatten per_word_tokens to a single list, keeping track of where each
       word boundary falls in the flat list.
    2. Run NW alignment between the per-word flat list and flat_tokens.
    3. Map the per-word boundary positions through the alignment to find
       corresponding positions in flat_tokens.
    4. For each source word, derive a contiguous [start, end) span in
       flat_tokens.

    The "contiguous per source word" requirement: after mapping word boundaries
    through the alignment, we may end up with a word that aligns to zero or
    overlapping flat_tokens positions (due to fusion).  In those cases the word
    inherits the span of its neighbor (fusion case: share the fused phoneme's
    frame duration with the adjacent word, giving the smaller half to the
    word that contributed fewer phonemes).
    """
    # Flatten per-word tokens, tracking word boundaries.
    flat_per_word: list[str] = []
    # word_ends[i] = index in flat_per_word where word i ends (exclusive).
    word_ends: list[int] = []
    for word_toks in per_word_tokens:
        flat_per_word.extend(word_toks)
        word_ends.append(len(flat_per_word))

    n_words = len(per_word_tokens)

    if not flat_per_word or not flat_tokens:
        # Degenerate: split time evenly.
        total = len(flat_tokens) if flat_tokens else 0
        step = max(1, total // max(1, n_words))
        spans = []
        cur = 0
        for i in range(n_words):
            end = cur + step if i < n_words - 1 else total
            spans.append((cur, end))
            cur = end
        return spans

    # NW alignment: flat_per_word as seq_a, flat_tokens as seq_b.
    aligned_pw, aligned_ft = _nw_align(flat_per_word, flat_tokens)

    # Build mapping: index in flat_per_word → index in flat_tokens (or None).
    # aligned columns where aligned_pw[k] is not None tell us the pw index;
    # aligned_ft[k] (possibly None) gives the ft index.
    ft_idx = 0      # current position in flat_tokens
    pw_idx = 0      # current position in flat_per_word
    # pw_to_ft[i] = first flat_tokens index aligned with pw position i,
    # or None if that pw position was aligned to a gap.
    pw_to_ft = [None] * len(flat_per_word)  # Optional[int] per pw position
    for aw, af in zip(aligned_pw, aligned_ft):
        if aw is not None and af is not None:
            pw_to_ft[pw_idx] = ft_idx
            pw_idx += 1
            ft_idx += 1
        elif aw is not None:
            # pw token aligned to gap in ft.
            pw_to_ft[pw_idx] = None
            pw_idx += 1
        else:
            # ft token aligned to gap in pw.
            ft_idx += 1

    # For each source word compute [start, end) in flat_tokens.
    # word i occupies pw positions [word_starts[i], word_ends[i]).
    word_starts_pw = [0] + word_ends[:-1]

    spans: list[tuple[int, int]] = []
    total_ft = len(flat_tokens)

    for i in range(n_words):
        ps = word_starts_pw[i]
        pe = word_ends[i]
        # Find all non-None flat_tokens positions for this word's pw range.
        ft_positions = [pw_to_ft[j] for j in range(ps, pe) if pw_to_ft[j] is not None]

        if ft_positions:
            start = ft_positions[0]
            end = ft_positions[-1] + 1  # exclusive
        else:
            # Fusion / total gap: inherit position from previous span end.
            if spans:
                prev_end = spans[-1][1]
            else:
                prev_end = 0
            start = prev_end
            end = prev_end  # zero-width; will be absorbed by next word or clamped

        spans.append((start, end))

    # Post-process: snap every word's start to the previous word's end so
    # orphaned phoneme frames (NW gaps between aligned spans) get absorbed
    # by the following word rather than silently dropped. Without this,
    # e.g. "for CO2" leaves ~3 frames of the leading `sˌi` unassigned.
    fixed: list[tuple[int, int]] = []
    cursor = 0
    for i, (s, e) in enumerate(spans):
        s = cursor
        e = max(e, s)
        fixed.append((s, e))
        cursor = e

    # Stretch final word to cover remaining frames.
    if fixed:
        ls, le = fixed[-1]
        fixed[-1] = (ls, total_ft)

    return fixed


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_word_timings(
    source_text: str,
    gather_output: "np.ndarray",
    lang: str = "en-us",
    sample_rate: int = 24000,
    pad_start_ms: float = 0.0,
) -> list[dict]:
    """
    Compute per-source-word start/end timings in milliseconds.

    Parameters
    ----------
    source_text : str
        The real text (without sacrificial head-word).  Whitespace-split into
        words; punctuation attached to words is preserved in the output.
    gather_output : np.ndarray
        1-D INT64 array from /encoder/Gather_output_0.  Length == number of
        phoneme tokens in the synthesized phrase.  Each element is the number
        of audio frames (600 samples each) assigned to that phoneme.
    lang : str
        phonemizer language code, e.g. "en-us".
    sample_rate : int
        Audio sample rate (default 24000).
    pad_start_ms : float
        Leading silence in ms that was prepended to the WAV.  Word timings are
        shifted forward by this amount so they are correct in the WAV's time
        frame.

    Returns
    -------
    list[dict]
        One dict per source word:
            {"text": str, "start_ms": float, "end_ms": float}
        Times are in ms relative to the start of the WAV (0 = first sample),
        after applying pad_start_ms offset.
    """
    gather = np.asarray(gather_output, dtype=np.int64).ravel()
    # Cumulative sample counts for each frame boundary.
    cum_samples = np.concatenate([[0], np.cumsum(gather * SAMPLES_PER_FRAME)])

    words = source_text.split()
    if not words:
        return []

    # Phonemize per-word and as a phrase.
    per_word_tokens = _phonemize_list(words, lang)
    flat_tokens = _phonemize_phrase(source_text, lang)

    # Align per-word phonemes onto the flat phrase token stream.
    spans = _align_words_to_phrase(per_word_tokens, flat_tokens)

    ms_per_sample = 1000.0 / sample_rate
    pad_samples = int(round(sample_rate * pad_start_ms / 1000.0))

    result = []
    for word, (start_tok, end_tok) in zip(words, spans):
        start_tok = max(0, min(start_tok, len(gather)))
        end_tok = max(start_tok, min(end_tok, len(gather)))
        start_sample = int(cum_samples[start_tok]) + pad_samples
        end_sample = int(cum_samples[end_tok]) + pad_samples
        start_ms = round(start_sample * ms_per_sample)
        end_ms = round(end_sample * ms_per_sample)
        result.append({"text": word, "start_ms": start_ms, "end_ms": end_ms})

    return result
