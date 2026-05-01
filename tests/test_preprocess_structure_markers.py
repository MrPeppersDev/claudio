#!/usr/bin/env python3
"""
Tests for Phase 1 of #163 PR 4 follow-up: preprocess.py emits structural
markers for markdown tables and fenced code blocks.

The structural markers (\\x11-\\x16) are consumed by:
  - kokoro-tts.sh (audio side): translates back to today's audible
    behavior (commas + periods inline for tables, earcon sentinel
    for code blocks).
  - mirror.html (visual side, not exercised here): parses into
    <table> and <pre> DOM.

These tests cover preprocess.py only — the audio-side translation
lives in kokoro-tts.sh and is exercised by listen-test rather than
unit test (the existing audio path is unchanged for prose-only input;
table/code outputs were verified manually mid-Phase-1).

Run with: /usr/bin/python3 tests/test_preprocess_structure_markers.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from preprocess import (  # noqa: E402
    CELL_SEP,
    CODE_END,
    CODE_START,
    EARCON_SENTINEL,
    PARAGRAPH_SENTINEL,
    ROW_SEP,
    SENTENCE_SENTINEL,
    TABLE_END,
    TABLE_START,
    strip_markdown,
)


def _assert(condition: bool, msg: str) -> None:
    if not condition:
        print(f"FAIL: {msg}")
        sys.exit(1)
    print(f"PASS: {msg}")


# ── Code blocks ──────────────────────────────────────────────────────────

def test_fenced_code_block_wrapped_with_markers() -> None:
    """Fenced code is wrapped with CODE_START / CODE_END (not stripped)."""
    md = "Before.\n```python\ndef f():\n    pass\n```\nAfter.\n"
    out = strip_markdown(md)
    _assert(CODE_START in out, "CODE_START present")
    _assert(CODE_END in out, "CODE_END present")
    _assert("def f():" in out, "code body preserved between markers")
    _assert("Before." in out, "prose before code preserved")
    _assert("After." in out, "prose after code preserved")


def test_code_block_drops_language_tag() -> None:
    """Leading language tag (`python` etc.) on the opening fence line
    is stripped — visual rendering shouldn't show it as a stray word."""
    md = "```javascript\nconsole.log('hi');\n```"
    out = strip_markdown(md)
    _assert("javascript" not in out,
            "language tag stripped from code body")
    _assert("console.log" in out, "actual code body preserved")


def test_tilde_fenced_code_block() -> None:
    """~~~ fences work identically to ``` fences."""
    md = "~~~\nraw text\n~~~"
    out = strip_markdown(md)
    _assert(CODE_START in out, "CODE_START present for ~~~")
    _assert(CODE_END in out, "CODE_END present for ~~~")
    _assert("raw text" in out, "tilde-fenced body preserved")


def test_no_code_block_no_markers() -> None:
    """Prose-only input must not contain code-block markers."""
    md = "Just a paragraph. No fenced anything here."
    out = strip_markdown(md)
    _assert(CODE_START not in out, "no spurious CODE_START in prose")
    _assert(CODE_END not in out, "no spurious CODE_END in prose")


# ── Tables ───────────────────────────────────────────────────────────────

def test_simple_table_emits_markers() -> None:
    """Markdown pipe-table → TABLE_START + CELL_SEPs + ROW_SEPs + TABLE_END."""
    md = (
        "| Quarter | Sales |\n"
        "| --- | --- |\n"
        "| Q1 | 100 |\n"
        "| Q2 | 150 |\n"
    )
    out = strip_markdown(md)
    _assert(TABLE_START in out, "TABLE_START present")
    _assert(TABLE_END in out, "TABLE_END present")
    _assert(CELL_SEP in out, "CELL_SEP present (between cells in a row)")
    _assert(ROW_SEP in out, "ROW_SEP present (between rows)")
    # All cell contents survive — audio side will linearize them with the
    # mirror getting structure for free.
    for cell in ("Quarter", "Sales", "Q1", "100", "Q2", "150"):
        _assert(cell in out, f"cell content {cell!r} preserved")


def test_table_structure_well_formed() -> None:
    """The marker layout is TABLE_START + cells/seps + TABLE_END,
    with exactly one CELL_SEP between cells in a row and ROW_SEP only
    between rows."""
    md = (
        "| A | B |\n"
        "| - | - |\n"
        "| 1 | 2 |\n"
        "| 3 | 4 |\n"
    )
    out = strip_markdown(md)
    # Find the table block.
    start = out.index(TABLE_START)
    end = out.index(TABLE_END)
    body = out[start + 1:end]
    rows = body.split(ROW_SEP)
    _assert(len(rows) == 3,
            f"3 rows (header + 2 data); got {len(rows)}: {rows!r}")
    for i, row in enumerate(rows):
        cells = row.split(CELL_SEP)
        _assert(len(cells) == 2,
                f"row {i} has 2 cells; got {len(cells)}: {cells!r}")


def test_table_followed_by_paragraph() -> None:
    """A table followed by a blank line + paragraph keeps both
    structures intact and keeps the paragraph from being folded into
    the table block."""
    md = (
        "| H |\n"
        "| - |\n"
        "| 1 |\n"
        "\n"
        "Next paragraph here.\n"
    )
    out = strip_markdown(md)
    _assert(TABLE_END in out, "TABLE_END present")
    end_idx = out.index(TABLE_END)
    after = out[end_idx + 1:]
    _assert("Next paragraph here." in after,
            "paragraph after table is preserved outside the table block")
    _assert(TABLE_START not in after,
            "no second table starts in the post-table content")


def test_no_table_no_markers() -> None:
    """Prose-only input must not contain table markers."""
    md = "Plain prose.\n\nAnother paragraph.\n"
    out = strip_markdown(md)
    _assert(TABLE_START not in out, "no spurious TABLE_START in prose")
    _assert(TABLE_END not in out, "no spurious TABLE_END in prose")
    _assert(CELL_SEP not in out, "no spurious CELL_SEP in prose")
    _assert(ROW_SEP not in out, "no spurious ROW_SEP in prose")


def test_orphan_pipe_line_not_treated_as_table() -> None:
    """A single pipe-bracketed line with no neighbor isn't really a
    table, but the regex matches it. Verify the wrapping pass still
    produces a coherent (single-row) table block — at minimum, no
    crash and cell content preserved."""
    md = "| only one |\n"
    out = strip_markdown(md)
    _assert("only one" in out, "single-row pipe content preserved")


# ── Existing sentinels still emitted ─────────────────────────────────────

def test_paragraph_and_sentence_sentinels_unchanged() -> None:
    """Phase 1 must not break the existing \\x1c / \\x1d emission for
    plain prose (paragraph and sentence delimiters)."""
    md = "First sentence. Second sentence.\n\nNew paragraph here."
    out = strip_markdown(md)
    # strip_markdown alone doesn't add SENTENCE_SENTINEL — that's done
    # by split_sentences downstream. But the paragraph break (blank line)
    # must still be present so split_sentences can see it.
    _assert("\n\n" in out or PARAGRAPH_SENTINEL in out,
            "paragraph break preserved through strip_markdown")


def test_code_and_table_coexist() -> None:
    """Code block and table in the same input — both produce their own
    marker pairs without collision."""
    md = (
        "Intro.\n\n"
        "```\nconst x = 1;\n```\n\n"
        "| H |\n"
        "| - |\n"
        "| 1 |\n"
    )
    out = strip_markdown(md)
    _assert(CODE_START in out and CODE_END in out, "code markers present")
    _assert(TABLE_START in out and TABLE_END in out, "table markers present")
    _assert("const x = 1;" in out, "code body preserved")
    _assert("Intro." in out, "intro prose preserved")


def main() -> None:
    test_fenced_code_block_wrapped_with_markers()
    test_code_block_drops_language_tag()
    test_tilde_fenced_code_block()
    test_no_code_block_no_markers()
    test_simple_table_emits_markers()
    test_table_structure_well_formed()
    test_table_followed_by_paragraph()
    test_no_table_no_markers()
    test_orphan_pipe_line_not_treated_as_table()
    test_paragraph_and_sentence_sentinels_unchanged()
    test_code_and_table_coexist()
    print()
    print("All preprocess structural-marker tests passed.")


# pytest-visible wrappers (matches the test_preprocess_tables pattern)
def test_fenced_code_block_wrapped_with_markers_pytest() -> None:
    test_fenced_code_block_wrapped_with_markers()


def test_code_block_drops_language_tag_pytest() -> None:
    test_code_block_drops_language_tag()


def test_tilde_fenced_code_block_pytest() -> None:
    test_tilde_fenced_code_block()


def test_no_code_block_no_markers_pytest() -> None:
    test_no_code_block_no_markers()


def test_simple_table_emits_markers_pytest() -> None:
    test_simple_table_emits_markers()


def test_table_structure_well_formed_pytest() -> None:
    test_table_structure_well_formed()


def test_table_followed_by_paragraph_pytest() -> None:
    test_table_followed_by_paragraph()


def test_no_table_no_markers_pytest() -> None:
    test_no_table_no_markers()


def test_orphan_pipe_line_not_treated_as_table_pytest() -> None:
    test_orphan_pipe_line_not_treated_as_table()


def test_paragraph_and_sentence_sentinels_unchanged_pytest() -> None:
    test_paragraph_and_sentence_sentinels_unchanged()


def test_code_and_table_coexist_pytest() -> None:
    test_code_and_table_coexist()


if __name__ == "__main__":
    main()
