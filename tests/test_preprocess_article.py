#!/usr/bin/env python3
"""
Tests for maybe_extract_article() in preprocess.py.

Run with: /usr/bin/python3 tests/test_preprocess_article.py
"""
import sys
import os

# Allow importing preprocess.py from the repo root regardless of cwd.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from preprocess import maybe_extract_article, _looks_like_html  # noqa: E402


def _assert(condition: bool, msg: str) -> None:
    if not condition:
        print(f"FAIL: {msg}")
        sys.exit(1)
    print(f"PASS: {msg}")


def test_html_article_extraction() -> None:
    """HTML with article tag: prose extracted, footer cookie banner dropped."""
    html = (
        "<html><body>"
        "<article><p>Real prose here. Another sentence.</p></article>"
        "<footer>Cookie banner text that should not appear.</footer>"
        "</body></html>"
    )
    result = maybe_extract_article(html)
    # trafilatura should extract the article prose
    _assert("Real prose here" in result, "article prose present in extraction")
    _assert("Cookie banner" not in result, "footer chrome dropped by extraction")


def test_plain_text_unchanged() -> None:
    """Plain text must pass through unchanged."""
    text = "Just a normal message. Nothing HTML about it."
    result = maybe_extract_article(text)
    _assert(result == text, "plain text returned unchanged")


def test_short_html_fallback() -> None:
    """Short HTML snippet — too little content for extraction to beat input."""
    html = "<div>hi</div>"
    result = maybe_extract_article(html)
    # Either trafilatura returns something equally short, or we get original back.
    # Either way the result should not be empty or radically different.
    _assert(len(result) > 0, "short HTML fallback returns non-empty result")
    # The snippet is < 500 chars so the short-extraction guard won't trigger;
    # trafilatura may return 'hi' or None. In either case we should not crash.


def test_non_html_angle_brackets() -> None:
    """Comparison operators should not trigger HTML detection."""
    text = "if x < 5 and y > 3"
    _assert(not _looks_like_html(text), "comparison operators not detected as HTML")
    result = maybe_extract_article(text)
    _assert(result == text, "non-HTML with angle brackets returned unchanged")


if __name__ == "__main__":
    test_html_article_extraction()
    test_plain_text_unchanged()
    test_short_html_fallback()
    test_non_html_angle_brackets()
    print("\nAll tests passed.")
