#!/usr/bin/env python3
"""pdf_extract.py — extract plain text from a PDF in reading order using PyMuPDF.

Usage:
    python3 pdf_extract.py --path <pdf-file> [--page <N>]

Arguments:
    --path <pdf-file>   Path to the PDF file (required).
    --page <N>          1-based page number to extract. Omit to extract all pages.
    --help              Show this help and exit.

Exit codes:
    0  Success — text written to stdout.
    1  General error (file not found, corrupt PDF, etc.).
    2  No extractable text — PDF is likely scanned (no text layer).

Text is extracted in reading order: columns are sorted left-to-right (x-midpoint
bucketed to 200px columns), then top-to-bottom within each column. This handles
two-column academic papers and news layouts better than raw block order.
"""

import sys
import argparse


def reading_order_key(block, column_width=200):
    """Sort key: bucket blocks by left edge into columns, then by y0 within each column.

    column_width=200 works for typical letter/A4 pages (612–595 pt wide).
    Each 200-pt bucket represents one visual column. Within a column, blocks
    sort top-to-bottom by their y0 coordinate.
    """
    x0 = block["bbox"][0]
    y0 = block["bbox"][1]
    column_bucket = round(x0 / column_width)
    return (column_bucket, y0)


def extract_text_from_page(page):
    """Extract text from a single fitz Page, preserving reading order.

    Returns a string with paragraphs separated by blank lines, or an empty
    string if the page has no text blocks.
    """
    data = page.get_text("dict")
    blocks = [b for b in data["blocks"] if b["type"] == 0]  # type 0 = text

    if not blocks:
        return ""

    blocks.sort(key=reading_order_key)

    paragraphs = []
    for block in blocks:
        lines = []
        for line in block.get("lines", []):
            # Concatenate all spans in the line into a single string.
            span_texts = [s["text"] for s in line.get("spans", []) if s.get("text")]
            line_text = "".join(span_texts).strip()
            if line_text:
                lines.append(line_text)
        if lines:
            paragraphs.append("\n".join(lines))

    return "\n\n".join(paragraphs)


def main():
    parser = argparse.ArgumentParser(
        description="Extract text from a PDF in reading order.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--path", required=True, help="Path to PDF file")
    parser.add_argument(
        "--page",
        type=int,
        default=None,
        help="1-based page number to extract (default: all pages)",
    )
    args = parser.parse_args()

    try:
        import fitz  # PyMuPDF
    except ImportError:
        print(
            "ERROR: PyMuPDF (fitz) is not installed.\n"
            "Install with: /usr/bin/python3 -m pip install --user pymupdf",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        doc = fitz.open(args.path)
    except FileNotFoundError:
        print(f"ERROR: file not found: {args.path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: could not open PDF: {e}", file=sys.stderr)
        sys.exit(1)

    if args.page is not None:
        page_num = args.page
        if page_num < 1 or page_num > len(doc):
            print(
                f"ERROR: page {page_num} out of range (PDF has {len(doc)} page(s))",
                file=sys.stderr,
            )
            sys.exit(1)
        pages = [doc[page_num - 1]]
    else:
        pages = list(doc)

    page_texts = []
    for page in pages:
        text = extract_text_from_page(page)
        if text:
            page_texts.append(text)

    if not page_texts:
        # No text layer found — PDF is likely scanned or image-only.
        print(
            "No extractable text found. PDF may be scanned; use Live Text instead.",
            file=sys.stderr,
        )
        sys.exit(2)

    print("\n\n".join(page_texts))
    sys.exit(0)


if __name__ == "__main__":
    main()
