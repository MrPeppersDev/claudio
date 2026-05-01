#!/usr/bin/env python3
"""
Tests for table-row preservation through preprocess.py.

Regression for the bug where table data rows like "Q1, 100" / "Q2, 150"
were dropped by the all-caps micro-line filter — single capital letter
plus digits with no lowercase characters tripped the heuristic, leaving
listeners hearing only the table's header. Fix: skip the all-caps check
for any line containing sentence punctuation (matching the existing
behavior of _is_short_chrome_line).

Run with: /usr/bin/python3 tests/test_preprocess_tables.py
"""
import sys
import os

# Allow importing preprocess.py from the repo root regardless of cwd.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from preprocess import (  # noqa: E402
    _is_all_caps_micro_line,
    _is_short_chrome_line,
    clean_web_artifacts,
    strip_markdown,
)


def _assert(condition: bool, msg: str) -> None:
    if not condition:
        print(f"FAIL: {msg}")
        sys.exit(1)
    print(f"PASS: {msg}")


# ── Unit tests on the all-caps detector ──────────────────────────────────

def test_short_numeric_row_not_all_caps() -> None:
    """Q1, 100 — table data row, has comma, must not be flagged."""
    _assert(not _is_all_caps_micro_line("Q1, 100"),
            "Q1, 100 (comma) is not all-caps micro-line")
    _assert(not _is_all_caps_micro_line("Q2, 150"),
            "Q2, 150 (comma) is not all-caps micro-line")


def test_real_nav_token_still_caught() -> None:
    """MENU / FAQ / ABOUT — actual nav, no punctuation, must be flagged."""
    _assert(_is_all_caps_micro_line("MENU"),
            "MENU is all-caps micro-line")
    _assert(_is_all_caps_micro_line("FAQ"),
            "FAQ is all-caps micro-line")
    _assert(_is_all_caps_micro_line("ABOUT"),
            "ABOUT is all-caps micro-line")


def test_punctuated_line_skips_filter() -> None:
    """Any line with sentence punctuation is content, not nav."""
    _assert(not _is_all_caps_micro_line("Q1."),
            "Q1. (period) is not all-caps")
    _assert(not _is_all_caps_micro_line("API; HTTP"),
            "API; HTTP (semicolon) is not all-caps")
    _assert(not _is_all_caps_micro_line("X: 5"),
            "X: 5 (colon) is not all-caps")


def test_lowercase_disqualifies() -> None:
    """Any lowercase letter — even one — means the line isn't all-caps."""
    _assert(not _is_all_caps_micro_line("MENu"),
            "MENu (one lowercase) is not all-caps")


# ── Full-pipeline tests through clean_web_artifacts ──────────────────────

def test_table_rows_survive_clean_web_artifacts() -> None:
    """Markdown table → strip_markdown linearizes rows → clean_web_artifacts
    must NOT eat the data rows."""
    md = (
        "| Quarter | Sales |\n"
        "| --- | --- |\n"
        "| Q1 | 100 |\n"
        "| Q2 | 150 |\n"
        "\n"
        "Next paragraph.\n"
    )
    after_md = strip_markdown(md)
    # After Phase 1 of #163 PR 4 follow-up, table rows are marker-wrapped
    # rather than linearized inline. The cell *contents* must still be
    # present (audio side translates the markers back to commas in Phase 2).
    for cell in ("Quarter", "Sales", "Q1", "100", "Q2", "150"):
        _assert(cell in after_md, f"{cell!r} present after strip_markdown")

    after_web = clean_web_artifacts(after_md)
    for cell in ("Quarter", "Sales", "Q1", "100", "Q2", "150"):
        _assert(cell in after_web, f"{cell!r} survives clean_web_artifacts")


def test_actual_nav_still_filtered() -> None:
    """A 5-line run of unpunctuated all-caps tokens is still treated as nav
    chrome and dropped — the fix shouldn't weaken legitimate nav filtering.
    """
    nav = (
        "Real opening sentence here.\n"
        "\n"
        "MENU\n"
        "ABOUT\n"
        "CONTACT\n"
        "FAQ\n"
        "BLOG\n"
        "\n"
        "Real body text continues normally.\n"
    )
    out = clean_web_artifacts(nav)
    _assert("MENU" not in out, "MENU dropped as nav")
    _assert("ABOUT" not in out, "ABOUT dropped as nav")
    _assert("Real opening sentence here" in out, "real prose preserved")


# ── Sanity: chrome line detector unchanged ──────────────────────────────

def test_short_chrome_unaffected() -> None:
    """The fix touched _is_all_caps_micro_line only; _is_short_chrome_line
    semantics must be unchanged."""
    _assert(_is_short_chrome_line("nav"),
            "nav (short, no punct) still chrome-flagged")
    _assert(not _is_short_chrome_line("Q1, 100"),
            "Q1, 100 (comma) not short chrome (existing rule)")


def main() -> None:
    test_short_numeric_row_not_all_caps()
    test_real_nav_token_still_caught()
    test_punctuated_line_skips_filter()
    test_lowercase_disqualifies()
    test_table_rows_survive_clean_web_artifacts()
    test_actual_nav_still_filtered()
    test_short_chrome_unaffected()
    print()
    print("All preprocess-table tests passed.")


# pytest-visible wrappers (matches the test_preprocess_numbers / sentences
# pattern so `pytest tests/` collects these automatically).
def test_short_numeric_row_not_all_caps_pytest() -> None:
    test_short_numeric_row_not_all_caps()


def test_real_nav_token_still_caught_pytest() -> None:
    test_real_nav_token_still_caught()


def test_punctuated_line_skips_filter_pytest() -> None:
    test_punctuated_line_skips_filter()


def test_lowercase_disqualifies_pytest() -> None:
    test_lowercase_disqualifies()


def test_table_rows_survive_clean_web_artifacts_pytest() -> None:
    test_table_rows_survive_clean_web_artifacts()


def test_actual_nav_still_filtered_pytest() -> None:
    test_actual_nav_still_filtered()


def test_short_chrome_unaffected_pytest() -> None:
    test_short_chrome_unaffected()


if __name__ == "__main__":
    main()
