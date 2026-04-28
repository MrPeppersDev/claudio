#!/usr/bin/env python3
"""
tests/test_preprocess_sentences.py — parametric tests for split_sentences().

Run with:
    /usr/bin/python3 tests/test_preprocess_sentences.py

Tests run against whichever code path is active: pysbd if installed, regex
fallback otherwise. Both paths must satisfy all assertions.
"""
import sys
import os

# Make sure we can import from repo root regardless of cwd.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from preprocess import (
    split_sentences,
    SENTENCE_SENTINEL,
    PARAGRAPH_SENTINEL,
    EARCON_SENTINEL,
    _HAVE_PYSBD,
)

SEP = SENTENCE_SENTINEL  # \x1d
PARA = PARAGRAPH_SENTINEL  # \x1c
EARCON = EARCON_SENTINEL  # \x1e

_passed = 0
_failed = 0


def check(label: str, result: str, expected_count: int = None,
          contains: list = None, not_contains: list = None,
          exact: str = None):
    global _passed, _failed
    ok = True
    reasons = []

    if expected_count is not None:
        parts = result.split(SEP)
        # Filter out empty parts from edge cases
        nonempty = [p for p in parts if p.strip().lstrip(PARA).strip()]
        actual = len(nonempty)
        if actual != expected_count:
            ok = False
            reasons.append(f"expected {expected_count} sentences, got {actual}: {nonempty!r}")

    if contains:
        for substr in contains:
            if substr not in result:
                ok = False
                reasons.append(f"expected {substr!r} in result {result!r}")

    if not_contains:
        for substr in not_contains:
            if substr in result:
                ok = False
                reasons.append(f"expected {substr!r} NOT in result {result!r}")

    if exact is not None:
        if result != exact:
            ok = False
            reasons.append(f"expected exact {exact!r}, got {result!r}")

    if ok:
        print(f"  PASS  {label}")
        _passed += 1
    else:
        print(f"  FAIL  {label}")
        for r in reasons:
            print(f"        {r}")
        _failed += 1


print(f"\nRunning with pysbd={'present' if _HAVE_PYSBD else 'absent (regex fallback)'}\n")

# --- Abbreviation handling (main motivation for pysbd) --------------------

check(
    "Dr. Smith: 2 sentences",
    split_sentences("Dr. Smith said hi. She left."),
    expected_count=2,
)

check(
    "U.S. economy: 2 sentences",
    split_sentences("The U.S. economy grew. Jobs are up."),
    expected_count=2,
)

check(
    "e.g. abbreviation: 2 sentences",
    split_sentences("e.g. like this. Next sentence."),
    expected_count=2,
)

# --- Paragraph boundary ---------------------------------------------------

para_result = split_sentences("Para one.\n\nPara two begins.")
para_parts = para_result.split(SEP)
nonempty_parts = [p for p in para_parts if p.strip()]

check(
    "paragraph case: 2 sentences total",
    para_result,
    expected_count=2,
)

check(
    "paragraph case: second sentence has PARAGRAPH_SENTINEL prefix",
    para_result,
    contains=[PARA],
)

# Verify the paragraph sentinel is on the second sentence (not the first)
second_sentence = [p for p in nonempty_parts if p][-1] if nonempty_parts else ""
if PARA in para_result:
    para_sentence = [p for p in para_result.split(SEP) if p.startswith(PARA)]
    check(
        "paragraph case: PARAGRAPH_SENTINEL is prefix, not standalone",
        para_result,
        contains=[PARA + "Para two"],
    )

# --- Sentinel preservation ------------------------------------------------

earcon_input = "Sentence one.\x1eAnd after earcon."
earcon_result = split_sentences(earcon_input)

check(
    "earcon sentinel preserved in output",
    earcon_result,
    contains=[EARCON],
)

# The earcon and surrounding text should be kept (not eaten by the splitter)
check(
    "earcon: content after earcon is present",
    earcon_result,
    contains=["And after earcon."],
)

check(
    "earcon: content before earcon is present",
    earcon_result,
    contains=["Sentence one."],
)

# --- Edge cases -----------------------------------------------------------

check(
    "empty input -> empty output",
    split_sentences(""),
    exact="",
)

check(
    "single sentence no terminator -> 1 sentence",
    split_sentences("Hello there"),
    expected_count=1,
)

check(
    "single sentence with period -> 1 sentence",
    split_sentences("Hello there."),
    expected_count=1,
)

# --- Leading paragraph sentinel stripped ---------------------------------

leading_para = split_sentences("\n\nStarts with blank line.")
check(
    "leading paragraph sentinel is stripped",
    leading_para,
    not_contains=[PARA],  # \x1c should be stripped from the very start
)

# --- Multi-sentence paragraph ---------------------------------------------

check(
    "three clear sentences",
    split_sentences("First sentence. Second sentence. Third sentence."),
    expected_count=3,
)

# --- Summary --------------------------------------------------------------
print(f"\nResults: {_passed} passed, {_failed} failed")
if _failed:
    sys.exit(1)


# Pytest-visible wrapper so `pytest tests/` collects these assertions as a
# single test instead of reporting "0 collected" for the file. The
# module-level check() calls above already executed at import time and
# populated _failed; this just exposes the pass/fail state to pytest.
def test_split_sentences_runner_passes():
    assert _failed == 0, (
        f"{_failed} sub-test(s) failed; see stdout for details "
        f"(run `python3 tests/test_preprocess_sentences.py` for per-case output)"
    )
