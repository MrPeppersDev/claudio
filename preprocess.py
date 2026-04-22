#!/usr/bin/env python3
"""
preprocess.py — text transformations applied before TTS synthesis.

Reads stdin, writes transformed text to stdout. Applied by play-last.sh
feeding kokoro-tts.sh.

Transformations, in order:
  1. Markdown stripping (code blocks, links, emphasis, etc.)
  2. Pronunciation dictionary (from pronunciations.txt)

Deliberately stdlib-only so we can run under /usr/bin/python3 (no venv dep).
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


# --- Markdown -------------------------------------------------------------

# Each entry: (compiled pattern, replacement or callable). Order matters —
# e.g. fenced code must go before anything that might see inside it, and
# images must go before links (image syntax is a superset of link syntax).

_MD_TRANSFORMS = [
    # Fenced code blocks — dropped entirely. Silence is fine; users who
    # actually want code read aloud can pass it through selection.
    (re.compile(r"```.*?```", re.DOTALL), ""),
    (re.compile(r"~~~.*?~~~", re.DOTALL), ""),

    # Images — drop (alt text is usually redundant with surrounding prose).
    (re.compile(r"!\[[^\]]*\]\([^)]*\)"), ""),
    # Links — keep the visible text, drop the URL.
    (re.compile(r"\[([^\]]+)\]\([^)]+\)"), r"\1"),
    # Reference-style link definitions: `[label]: url` on their own line.
    (re.compile(r"(?m)^\s*\[[^\]]+\]:\s+\S.*$"), ""),

    # Inline code — keep the content, drop the backticks. This lets the
    # pronunciation dictionary still act on identifiers like `kubectl`.
    (re.compile(r"`([^`\n]+)`"), r"\1"),

    # Bold (**x** and __x__) — strip markers, keep text. Must go before italic.
    (re.compile(r"\*\*([^*]+)\*\*"), r"\1"),
    (re.compile(r"__([^_]+)__"), r"\1"),
    # Italic (*x* and _x_) — strip markers. The simple pattern is good
    # enough for TTS; stray asterisks in prose are rare.
    (re.compile(r"(?<!\*)\*([^*\n]+)\*(?!\*)"), r"\1"),
    (re.compile(r"(?<![_\w])_([^_\n]+)_(?![_\w])"), r"\1"),
    # Strikethrough — strip entirely (nothing meaningful to speak).
    (re.compile(r"~~([^~]+)~~"), r"\1"),

    # Blockquote markers at line start — strip.
    (re.compile(r"(?m)^\s{0,3}>\s?"), ""),

    # Horizontal rules — strip. `[ \t]*` (not `\s*`) so we don't eat the
    # newline and merge adjacent lines.
    (re.compile(r"(?m)^[ \t]{0,3}(?:-{3,}|\*{3,}|_{3,})[ \t]*$"), ""),

    # Table separator rows (|---|---|) — strip.
    (re.compile(r"(?m)^[ \t]*\|?[ \t]*:?-+:?[ \t]*(?:\|[ \t]*:?-+:?[ \t]*)+\|?[ \t]*$"), ""),
    # Table cell rows — join cells with ", " so the row reads naturally.
    (
        re.compile(r"(?m)^[ \t]*\|([^\n]+)\|[ \t]*$"),
        lambda m: ", ".join(c.strip() for c in m.group(1).split("|") if c.strip()),
    ),

    # Headings (# … ######) — strip the marker, append period so TTS
    # pauses between the heading and the body. Trailing `#`s (closed-atx
    # style) are also removed.
    (
        re.compile(r"(?m)^[ \t]{0,3}#{1,6}[ \t]+([^\n]+?)[ \t]*#*[ \t]*$"),
        lambda m: m.group(1).rstrip(".") + ".",
    ),

    # Unordered bullets — strip the marker; the newline already gives a pause.
    (re.compile(r"(?m)^(\s*)[-*+]\s+"), r"\1"),
    # Ordered list markers — strip "N." prefix.
    (re.compile(r"(?m)^(\s*)\d+\.\s+"), r"\1"),

    # Collapse runs of 3+ blank lines to 2 (avoids long dead-air after
    # large code-block removals).
    (re.compile(r"\n{3,}"), "\n\n"),
]


def strip_markdown(text: str) -> str:
    for pattern, repl in _MD_TRANSFORMS:
        text = pattern.sub(repl, text)
    return text


# --- Pronunciation dictionary --------------------------------------------

def load_pronunciations(path: Path) -> list[tuple[str, str]]:
    """
    Returns list of (pattern_src, replacement) in file order.
    Order matters — list longer/more-specific entries first in the file.
    """
    rules: list[tuple[str, str]] = []
    if not path.exists():
        return rules
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or "|" not in line:
            continue
        src, dst = line.split("|", 1)
        src, dst = src.strip(), dst.strip()
        if src:
            rules.append((src, dst))
    return rules


def apply_pronunciations(text: str, rules: list[tuple[str, str]]) -> str:
    for src, dst in rules:
        # \b doesn't consider '.' a word char, so rules ending in '.' still
        # match correctly (e.g. "e.g." with \b acts as expected because the
        # surrounding space/punct is a non-word position).
        if src.endswith("."):
            pattern = re.compile(r"\b" + re.escape(src))
        else:
            pattern = re.compile(r"\b" + re.escape(src) + r"\b")
        text = pattern.sub(dst, text)
    return text


def main() -> int:
    text = sys.stdin.read()
    text = strip_markdown(text)
    rules = load_pronunciations(HERE / "pronunciations.txt")
    text = apply_pronunciations(text, rules)
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
