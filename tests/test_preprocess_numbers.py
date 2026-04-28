#!/usr/bin/env python3
"""
Tests for normalize_numbers() in preprocess.py.

Run with:  /usr/bin/python3 tests/test_preprocess_numbers.py
Requires:  num2words installed (pip install --user num2words)
"""
import sys
import os
from typing import Optional

# Allow running from the repo root or from inside tests/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from preprocess import normalize_numbers, _NUMBERS_AVAILABLE

# Skip gracefully if num2words isn't available (stdlib-only environment).
if not _NUMBERS_AVAILABLE:
    print("SKIP: num2words not installed — skipping normalize_numbers tests")
    sys.exit(0)


def check(description: str, input_text: str, expected_substr: Optional[str] = None,
          full_expected: Optional[str] = None, must_not_contain: Optional[str] = None) -> bool:
    result = normalize_numbers(input_text)
    ok = True
    if expected_substr is not None and expected_substr not in result:
        print(f"FAIL [{description}]")
        print(f"  input:    {input_text!r}")
        print(f"  expected to contain: {expected_substr!r}")
        print(f"  got:      {result!r}")
        ok = False
    if full_expected is not None and result != full_expected:
        print(f"FAIL [{description}]")
        print(f"  input:    {input_text!r}")
        print(f"  expected: {full_expected!r}")
        print(f"  got:      {result!r}")
        ok = False
    if must_not_contain is not None and must_not_contain in result:
        print(f"FAIL [{description}]")
        print(f"  input:    {input_text!r}")
        print(f"  must NOT contain: {must_not_contain!r}")
        print(f"  got:      {result!r}")
        ok = False
    if ok:
        print(f"PASS [{description}]")
    return ok


tests_passed = 0
tests_failed = 0


def run(description: str, *args, **kwargs) -> None:
    global tests_passed, tests_failed
    if check(description, *args, **kwargs):
        tests_passed += 1
    else:
        tests_failed += 1


# --- Currency ---
run(
    "currency: $1,234.56 dollars part",
    "Paid $1,234.56 today.",
    expected_substr="one thousand, two hundred and thirty-four dollars",
)
run(
    "currency: $1,234.56 cents part",
    "Paid $1,234.56 today.",
    expected_substr="fifty-six cents",
)
run(
    "currency: whole dollars",
    "It cost $42.",
    expected_substr="forty-two dollars",
)
run(
    "currency: cents only",
    "Only $0.99 left.",
    expected_substr="ninety-nine cents",
)

# --- Years ---
run(
    "year: in 2026 (trigger word 'in')",
    "In 2026, things change.",
    expected_substr="twenty twenty-six",
)
run(
    "year: since 1999",
    "Since 1999 we have grown.",
    expected_substr="nineteen ninety-nine",
)
run(
    "year: 2026 followed by comma (context trigger)",
    "The year 2026, a turning point.",
    expected_substr="twenty twenty-six",
)
run(
    "year: bare 4-digit without context is NOT converted",
    "1234",
    must_not_contain="twelve",  # no year trigger → passthrough
)

# --- Ordinals ---
run(
    "ordinal: 3rd",
    "The 3rd item.",
    full_expected="The third item.",
)
run(
    "ordinal: 21st",
    "Her 21st birthday.",
    expected_substr="twenty-first",
)
run(
    "ordinal: 42nd",
    "The 42nd floor.",
    expected_substr="forty-second",
)

# --- Percentages ---
run(
    "percent: 42%",
    "42% of users prefer this.",
    full_expected="forty-two percent of users prefer this.",
)
run(
    "percent: 100%",
    "100% complete.",
    full_expected="one hundred percent complete.",
)

# --- Integers with thousands separators ---
run(
    "thousands: 1,234",
    "1,234 items were sold.",
    full_expected="one thousand, two hundred and thirty-four items were sold.",
)
run(
    "thousands: 10,000",
    "We need 10,000 units.",
    expected_substr="ten thousand",
)

# --- Negatives / should-not-change ---
run(
    "negative: version string v1.2.3 unchanged",
    "v1.2.3",
    full_expected="v1.2.3",
)
run(
    "negative: route with hyphen unchanged",
    "Take route 1-95 north.",
    must_not_contain="one hyphen",
)
run(
    "negative: bare 4-digit number without year context",
    "Port 8080 is the default.",
    must_not_contain="eighty",  # 8080 is outside our year range entirely
)

# --- Summary ---
print()
print(f"Results: {tests_passed} passed, {tests_failed} failed")
# sys.exit only on failure so pytest collection (which executes module-level
# code) doesn't trip on a successful run. Matches test_preprocess_sentences.py.
if tests_failed:
    sys.exit(1)


# Pytest-visible wrapper so `pytest tests/` collects these assertions as a
# single test instead of reporting "0 collected" for the file. The
# module-level run() calls above already executed at import time and
# populated tests_failed; this just exposes the pass/fail state to pytest.
def test_normalize_numbers_runner_passes():
    assert tests_failed == 0, (
        f"{tests_failed} sub-test(s) failed; see stdout for details "
        f"(run `python3 tests/test_preprocess_numbers.py` for per-case output)"
    )
