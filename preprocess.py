#!/usr/bin/env python3
"""
preprocess.py — text transformations applied before TTS synthesis.

Reads stdin, writes transformed text to stdout. Applied by play-last.sh
feeding kokoro-tts.sh.

Transformations, in order:
  1. Unicode whitespace normalization (NBSP → space, drop zero-width)
  2. Markdown stripping (code blocks, links, emphasis, etc.)
  3. Web-paste chrome cleanup (bare URL lines, short-line-run nav, all-caps
     micro-lines) — conservative heuristics for pasted web pages
  4. Pronunciation dictionary (from pronunciations.txt)
  5. Sentence / paragraph sentinel injection (for downstream split+cache)

Deliberately stdlib-only so we can run under /usr/bin/python3 (no venv dep).
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


# --- Unicode whitespace --------------------------------------------------

# Paste from a browser brings along characters that aren't in `[ \t]` but
# visually behave as spaces. If we don't normalize them, the explicit
# `[ \t]` classes in the markdown regexes (heading, bullet, hr) silently
# miss matches at the leading/trailing edges of lines. Zero-width chars
# (U+200B, U+200C, U+200D, U+FEFF) carry no acoustic meaning at all — drop.
_WS_NORMALIZE = str.maketrans({
    "\u00a0": " ",  # no-break space
    "\u202f": " ",  # narrow no-break space
    "\u2007": " ",  # figure space
    "\u2009": " ",  # thin space
    "\u200b": "",   # zero-width space
    "\u200c": "",   # zero-width non-joiner
    "\u200d": "",   # zero-width joiner
    "\ufeff": "",   # BOM / zero-width no-break space
})


def normalize_whitespace(text: str) -> str:
    return text.translate(_WS_NORMALIZE)


# --- Markdown -------------------------------------------------------------

# Each entry: (compiled pattern, replacement or callable). Order matters —
# e.g. fenced code must go before anything that might see inside it, and
# images must go before links (image syntax is a superset of link syntax).

# Sentinel for structural elisions. kokoro-tts.sh splits on this and plays
# an earcon at each boundary, so the listener hears "...prose, THOK, prose..."
# where a code block used to be. ASCII 0x1E (record separator) — guaranteed
# not to appear in normal text; survives shell pipes and JSON encoding fine.
EARCON_SENTINEL = "\x1e"

_MD_TRANSFORMS = [
    # Fenced code blocks — dropped with an earcon sentinel. Users who
    # actually want code read aloud can pass it through selection.
    (re.compile(r"```.*?```", re.DOTALL), EARCON_SENTINEL),
    (re.compile(r"~~~.*?~~~", re.DOTALL), EARCON_SENTINEL),

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

    # Collapse adjacent sentinels (two code blocks with only whitespace
    # between them) to one. kokoro-tts.sh would play them as one earcon
    # anyway — but keeping the input clean means per-segment synth logs
    # stay readable. Note: \s in Python matches \x1e/\x1d, so we use an
    # explicit whitespace class to avoid eating other sentinels.
    (re.compile(r"\x1e([ \t\n\r\f\v]*\x1e)+"), EARCON_SENTINEL),
]


def strip_markdown(text: str) -> str:
    for pattern, repl in _MD_TRANSFORMS:
        text = pattern.sub(repl, text)
    return text


# --- Web paste chrome ----------------------------------------------------
#
# Heuristics for stripping nav/header/footer residue that comes along when a
# user copies text out of a web page. Runs AFTER strip_markdown so the code
# fences are already collapsed into EARCON_SENTINEL (and we don't misread
# their short lines as chrome). Runs BEFORE pronunciation and sentence split
# so the dict/sentence logic only sees prose.
#
# Conservative by design:
#   * bare-URL lines — drop outright, they read as alphabet soup.
#   * short-line runs — drop a run of ≥5 consecutive lines that are each
#     ≤20 chars with no sentence punctuation. Threshold chosen so shopping
#     lists / tight bullet lists (usually 3-4 items) pass through; nav
#     sidebars (typically 7+ links) get caught.
#   * all-caps micro-lines — drop standalone lines that are ≤25 chars and
#     contain no lowercase letters ("MENU", "FOLLOW US"). Lines that mix
#     case (including acronyms in prose) are unaffected.

_BARE_URL_LINE = re.compile(r"(?m)^[ \t]*https?://\S+[ \t]*$")
_SENTENCE_PUNCT = re.compile(r"[.!?,:;]")

# Run length at which a window of short lines is interpreted as chrome.
_SHORT_RUN_MIN = 5
_SHORT_LINE_MAX_CHARS = 20
_ALLCAPS_LINE_MAX_CHARS = 25


def _is_short_chrome_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > _SHORT_LINE_MAX_CHARS:
        return False
    if _SENTENCE_PUNCT.search(stripped):
        return False
    # Structural sentinels from strip_markdown aren't chrome — leave alone.
    if EARCON_SENTINEL in stripped:
        return False
    return True


def _is_all_caps_micro_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > _ALLCAPS_LINE_MAX_CHARS:
        return False
    if EARCON_SENTINEL in stripped:
        return False
    saw_letter = False
    for ch in stripped:
        if ch.islower():
            return False
        if ch.isalpha():
            saw_letter = True
    return saw_letter


def clean_web_artifacts(text: str) -> str:
    # Drop bare URLs first so they don't pad out the short-line-run count
    # with lines that are "short" but not really nav.
    text = _BARE_URL_LINE.sub("", text)

    lines = text.split("\n")
    runs_to_drop: list[tuple[int, int]] = []
    run_start: int | None = None
    for i, line in enumerate(lines):
        if _is_short_chrome_line(line):
            if run_start is None:
                run_start = i
        else:
            if run_start is not None and i - run_start >= _SHORT_RUN_MIN:
                runs_to_drop.append((run_start, i))
            run_start = None
    if run_start is not None and len(lines) - run_start >= _SHORT_RUN_MIN:
        runs_to_drop.append((run_start, len(lines)))

    # Splice from the tail so earlier indices stay valid.
    for start, end in reversed(runs_to_drop):
        del lines[start:end]

    lines = [l for l in lines if not _is_all_caps_micro_line(l)]
    return "\n".join(lines)


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


# --- Sentence boundaries -------------------------------------------------

# ASCII 0x1D (group separator). kokoro-tts.sh splits on this to cache each
# sentence separately under sha256(voice|speed|lang|sentence). Cache hits
# across messages are common for stock phrasing ("Let me check that.",
# "Here's what I found.").
SENTENCE_SENTINEL = "\x1d"

# ASCII 0x1C (file separator). Carried as a *prefix* on the first sentence of
# each new paragraph, not as an independent separator. kokoro-tts.sh strips
# it when reading the sentence and inserts a longer pause before playback.
# Prefix (rather than separator) keeps the sentence-level split on \x1d
# working unchanged, so the sentence cache keying stays stable.
PARAGRAPH_SENTINEL = "\x1c"

# Sentence boundary: terminal punct + whitespace + uppercase-letter/digit.
# Uppercase lookahead keeps "e.g. foo" from splitting — at the cost of
# missing "Dr. Smith" style splits, which hurts nothing beyond cache
# granularity. Explicit whitespace class (not \s): Python's \s matches our
# sentinel bytes \x1e/\x1d/\x1c, which would otherwise let the match step
# over a structural boundary.
_WS = r"[ \t\n\r\f\v]"
_SENTENCE_BOUNDARY = re.compile(rf"(?<=[.!?]){_WS}+(?=[A-Z0-9])")
# Paragraph boundary: two or more consecutive newlines, possibly with
# horizontal whitespace between. Handled first so the paragraph marker
# survives into the downstream sentence-level split.
_PARAGRAPH_BOUNDARY = re.compile(r"\n[ \t]*\n[\s]*")


def split_sentences(text: str) -> str:
    # Paragraph first: replace the blank-line run with a sentence sentinel
    # (so sentence-level splitting still sees the boundary) plus a paragraph
    # prefix (so the next sentence carries "longer pause before me").
    text = _PARAGRAPH_BOUNDARY.sub(SENTENCE_SENTINEL + PARAGRAPH_SENTINEL, text)
    text = _SENTENCE_BOUNDARY.sub(SENTENCE_SENTINEL, text)
    # Collapse runs of sentence sentinels with whitespace between them.
    # \x1c deliberately not in the whitespace class — it survives as the
    # paragraph prefix on the next surviving sentinel.
    text = re.sub(rf"\x1d({_WS}*\x1d)+", SENTENCE_SENTINEL, text)
    # If a paragraph marker ends up at the very start of the output (the
    # selection began with a blank line), there is no "before" to pause
    # against — drop leading markers.
    text = text.lstrip(PARAGRAPH_SENTINEL)
    return text


def main() -> int:
    text = sys.stdin.read()
    text = normalize_whitespace(text)
    text = strip_markdown(text)
    text = clean_web_artifacts(text)
    rules = load_pronunciations(HERE / "pronunciations.txt")
    text = apply_pronunciations(text, rules)
    text = split_sentences(text)
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
