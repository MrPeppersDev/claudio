#!/usr/bin/env python3
"""Tests for pdf_extract.py — run with: /usr/bin/python3 tests/test_pdf_extract.py"""

import importlib.util
import os
import subprocess
import sys
import tempfile

# ---------------------------------------------------------------------------
# Resolve the script path relative to this test file, regardless of cwd.
# ---------------------------------------------------------------------------
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(TESTS_DIR)
SCRIPT = os.path.join(REPO_DIR, "pdf_extract.py")
PYTHON = sys.executable


def run(args, *, check=False):
    """Run pdf_extract.py with the given args list. Returns CompletedProcess."""
    return subprocess.run(
        [PYTHON, SCRIPT] + args,
        capture_output=True,
        text=True,
    )


# ---------------------------------------------------------------------------
# PDF helpers (create synthetic PDFs with fitz)
# ---------------------------------------------------------------------------

def make_text_pdf(text, path):
    """Create a single-page PDF containing `text` at a standard position."""
    import fitz
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 100), text, fontsize=12)
    doc.save(path)
    doc.close()


def make_empty_pdf(path):
    """Create a single-page PDF with no text content (blank page)."""
    import fitz
    doc = fitz.open()
    doc.new_page()  # blank — no text inserted
    doc.save(path)
    doc.close()


def make_two_column_pdf(path):
    """Create a PDF with two text columns (left column A, right column B).

    Left column starts at x≈72 (in the first 200pt column bucket).
    Right column starts at x≈330 (in the second 200pt column bucket).
    Both columns share the same y range so naive y-sort would interleave them.
    """
    import fitz
    doc = fitz.open()
    page = doc.new_page()
    # Left column: lines A1..A5
    for i, label in enumerate(["A1", "A2", "A3", "A4", "A5"]):
        page.insert_text((72, 100 + i * 20), label, fontsize=12)
    # Right column: lines B1..B5 — same y range as left column
    for i, label in enumerate(["B1", "B2", "B3", "B4", "B5"]):
        page.insert_text((330, 100 + i * 20), label, fontsize=12)
    doc.save(path)
    doc.close()


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

PASS = 0
FAIL = 0


def ok(name):
    global PASS
    PASS += 1
    print(f"  PASS  {name}")


def fail(name, detail):
    global FAIL
    FAIL += 1
    print(f"  FAIL  {name}: {detail}")


def test_text_only_pdf():
    """Text-only PDF → exit 0 and the expected text appears in stdout."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        path = f.name
    try:
        make_text_pdf("Hello Claudio", path)
        result = run(["--path", path])
        if result.returncode != 0:
            fail("text_only_pdf:exit_code", f"expected 0, got {result.returncode}\nstderr: {result.stderr}")
        elif "Hello Claudio" not in result.stdout:
            fail("text_only_pdf:content", f"expected 'Hello Claudio' in stdout, got: {result.stdout!r}")
        else:
            ok("text_only_pdf")
    finally:
        os.unlink(path)


def test_empty_page_exit_2():
    """Blank page (no text layer) → exit code 2."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        path = f.name
    try:
        make_empty_pdf(path)
        result = run(["--path", path])
        if result.returncode == 2:
            ok("empty_page_exit_2")
        else:
            fail("empty_page_exit_2", f"expected exit 2, got {result.returncode}\nstdout: {result.stdout!r}\nstderr: {result.stderr!r}")
    finally:
        os.unlink(path)


def test_missing_file_exit_1():
    """Non-existent path → exit code 1."""
    result = run(["--path", "/tmp/does_not_exist_claudio_test.pdf"])
    if result.returncode == 1:
        ok("missing_file_exit_1")
    else:
        fail("missing_file_exit_1", f"expected exit 1, got {result.returncode}")


def test_two_column_reading_order():
    """Two-column PDF → left column (A1–A5) appears before right column (B1–B5)."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        path = f.name
    try:
        make_two_column_pdf(path)
        result = run(["--path", path])
        if result.returncode != 0:
            fail("two_column_order:exit_code", f"expected 0, got {result.returncode}\nstderr: {result.stderr}")
            return
        stdout = result.stdout
        a1_pos = stdout.find("A1")
        b1_pos = stdout.find("B1")
        a5_pos = stdout.find("A5")
        if a1_pos == -1 or b1_pos == -1:
            fail("two_column_order:content", f"A1 or B1 not found in output:\n{stdout!r}")
        elif a1_pos < b1_pos and a5_pos < b1_pos:
            ok("two_column_order")
        else:
            fail(
                "two_column_order:ordering",
                f"expected all A-lines before B1; a1={a1_pos}, a5={a5_pos}, b1={b1_pos}\n{stdout!r}",
            )
    finally:
        os.unlink(path)


def test_single_page_flag():
    """--page 1 on a text PDF works the same as no --page flag."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        path = f.name
    try:
        make_text_pdf("Page one text", path)
        result = run(["--path", path, "--page", "1"])
        if result.returncode != 0:
            fail("single_page_flag:exit_code", f"expected 0, got {result.returncode}")
        elif "Page one text" not in result.stdout:
            fail("single_page_flag:content", f"text not found: {result.stdout!r}")
        else:
            ok("single_page_flag")
    finally:
        os.unlink(path)


def test_out_of_range_page():
    """--page 99 on a 1-page PDF → exit code 1."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        path = f.name
    try:
        make_text_pdf("Only one page", path)
        result = run(["--path", path, "--page", "99"])
        if result.returncode == 1:
            ok("out_of_range_page")
        else:
            fail("out_of_range_page", f"expected exit 1, got {result.returncode}")
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Verify PyMuPDF is available before trying to generate fixtures.
    try:
        import fitz  # noqa: F401
    except ImportError:
        print("ERROR: PyMuPDF (fitz) not installed.")
        print("Install: /usr/bin/python3 -m pip install --user pymupdf")
        sys.exit(1)

    print(f"Running tests for {SCRIPT}\n")
    test_text_only_pdf()
    test_empty_page_exit_2()
    test_missing_file_exit_1()
    test_two_column_reading_order()
    test_single_page_flag()
    test_out_of_range_page()

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(0 if FAIL == 0 else 1)
