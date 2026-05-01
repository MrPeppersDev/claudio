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
  4. Dash normalization (phrasal dashes → commas for prosody)
  5. Pronunciation dictionary (from pronunciations.txt)
  6. Sentence / paragraph sentinel injection (for downstream split+cache)

Deliberately stdlib-only so we can run under /usr/bin/python3 (no venv dep).
pysbd is used for sentence segmentation when available (pip install pysbd),
with an automatic regex fallback for stdlib-only environments.
"""
import re
import sys
from pathlib import Path

try:
    import pysbd as _pysbd
    _PYSBD_SEGMENTER = _pysbd.Segmenter(language="en", clean=False)
    _HAVE_PYSBD = True
except ImportError:  # pragma: no cover — fallback path for stdlib-only envs
    _HAVE_PYSBD = False

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


# --- HTML article extraction ---------------------------------------------

# trafilatura is an optional user-site dep. When absent, extraction is a
# pass-through; the regex web-chrome cleanup downstream still covers simple
# cases.
try:
    import trafilatura as _trafilatura
    _HAVE_TRAFILATURA = True
except ImportError:  # pragma: no cover — optional
    _HAVE_TRAFILATURA = False

# Cheap HTML sniff. Full-document tags are the strong signal; closing-tag
# density catches fragments that omit <html>/<body>. "if x < 5 and y > 3"
# has zero `</`, so comparison operators never trigger.
_HTML_SIGNALS = re.compile(r"<html|<article|<body", re.IGNORECASE)
_CLOSE_TAG = re.compile(r"</")


def _looks_like_html(text: str) -> bool:
    if _HTML_SIGNALS.search(text):
        return True
    return len(_CLOSE_TAG.findall(text)) >= 5


def maybe_extract_article(text: str) -> str:
    """Run trafilatura if the input looks like HTML; otherwise pass through.

    Returns Markdown on success — downstream strip_markdown() handles both
    Markdown and plain text, so callers don't need to know which ran.
    """
    if not _HAVE_TRAFILATURA or not _looks_like_html(text):
        return text
    extracted = _trafilatura.extract(
        text,
        output_format="markdown",
        include_comments=False,
        include_tables=False,
    )
    if extracted is None:
        return text
    # Guard against trafilatura returning a fragment for malformed snippets.
    if len(text) > 500 and len(extracted) < 100:
        return text
    return extracted


# --- Markdown -------------------------------------------------------------

# Each entry: (compiled pattern, replacement or callable). Order matters —
# e.g. fenced code must go before anything that might see inside it, and
# images must go before links (image syntax is a superset of link syntax).

# Sentinel codepoints. ASCII control bytes (0x11-0x1E) — guaranteed
# not to appear in normal text; survive shell pipes and JSON encoding fine.
#
# Existing sentinels (kept):
#   \x1c — paragraph-start prefix on the first sentence of a new paragraph
#   \x1d — sentence separator (kokoro-tts.sh splits on this for cache granularity)
#   \x1e — segment / earcon-position marker (kokoro-tts.sh splits on this for
#          inter-segment EARCON; also written by code-block-strip and adjacent-
#          sentinel collapse paths historically)
#
# New in Phase 1 of #163 PR 4 follow-up (structural rendering for the mirror
# webview). Audio side translates these back to today's behavior; mirror.html
# parses them into <pre> / <table> DOM (Phases 2 + 3).
#
#   \x11 — TABLE_START   (open `<table>`, audio: dropped)
#   \x12 — CELL_SEP      (next `<td>` in the same row, audio: ", ")
#   \x13 — ROW_SEP       (next `<tr>`, audio: ". " — sentence-end pause between rows)
#   \x14 — TABLE_END     (close `</table>`, audio: dropped)
#   \x15 — CODE_BLOCK_START (open `<pre>`, audio: drop content + emit earcon)
#   \x16 — CODE_BLOCK_END   (close `</pre>`, audio: marker that earcon already played)
EARCON_SENTINEL = "\x1e"
TABLE_START   = "\x11"
CELL_SEP      = "\x12"
ROW_SEP       = "\x13"
TABLE_END     = "\x14"
CODE_START    = "\x15"
CODE_END      = "\x16"


def _wrap_code_block(match: "re.Match[str]") -> str:
    """Wrap fenced-code content with CODE_START / CODE_END markers, preserving
    the inner text (with trailing newline trimmed). Audio side will drop the
    content + emit an earcon; mirror.html will render as <pre>.
    """
    raw = match.group(0)
    # Strip the leading and trailing fence (```...``` or ~~~...~~~). The
    # exact length of the fence isn't fixed (>= 3 chars) but for our regex
    # it's always exactly 3 since we matched ``` or ~~~. Slice and trim.
    if raw.startswith("```") and raw.endswith("```"):
        inner = raw[3:-3]
    elif raw.startswith("~~~") and raw.endswith("~~~"):
        inner = raw[3:-3]
    else:
        inner = raw
    # Drop a leading language tag line ("```python\n…") so the rendered
    # <pre> doesn't start with "python" on its own line.
    if "\n" in inner:
        first_line, rest = inner.split("\n", 1)
        # Heuristic: a language tag is a single short alpha token on the
        # opening line. Anything with whitespace or > 20 chars is treated
        # as actual content.
        if first_line.strip() and " " not in first_line.strip() \
           and len(first_line.strip()) <= 20:
            inner = rest
    return CODE_START + inner.strip("\n") + CODE_END


# Pre-compiled markdown table-row pattern: a line that starts and ends with
# "|" and isn't a separator row (which already gets stripped above). Matches
# a single row; consecutive rows are stitched into a TABLE_START..TABLE_END
# block by _wrap_table after the per-row substitution.
_TABLE_ROW_RE = re.compile(
    r"(?m)^[ \t]*\|([^\n]+)\|[ \t]*$"
)


def _row_to_cells(match: "re.Match[str]") -> str:
    """Replace a markdown table row with CELL_SEP-separated cells,
    surrounded by ROW_SEP at both ends. _wrap_table_blocks then collapses
    runs of these into TABLE_START / TABLE_END boundaries.
    """
    raw = match.group(1)
    cells = [c.strip() for c in raw.split("|") if c.strip()]
    if not cells:
        return ""
    return ROW_SEP + CELL_SEP.join(cells) + ROW_SEP


def _wrap_table_blocks(text: str) -> str:
    """After _row_to_cells has substituted each row, runs of consecutive
    rows look like "ROW_SEP cells ROW_SEP ROW_SEP cells ROW_SEP …".
    Collapse the doubled ROW_SEPs and add TABLE_START / TABLE_END at the
    ends. The result is a single sentinel-bracketed structure that
    downstream consumers can split on cleanly.
    """
    if ROW_SEP not in text:
        return text
    # Collapse adjacent ROW_SEPs (interior boundaries between rows) so we
    # have ROW_SEP only between rows, not at the ends of each row.
    text = re.sub(r"\x13[ \t\n]*\x13", ROW_SEP, text)
    # Find runs that start and end with ROW_SEP and contain no blank line
    # (a blank line breaks a table). Wrap each run in TABLE_START..TABLE_END.
    # Use a callback so adjacent unrelated text (e.g. the row before a
    # paragraph break) doesn't get folded in.
    def wrap(m: "re.Match[str]") -> str:
        body = m.group(0).strip("\x13").strip()
        if not body:
            return ""
        # body looks like: cells ROW_SEP cells ROW_SEP cells
        return TABLE_START + body + TABLE_END
    # The full run: ROW_SEP, then anything that doesn't include two
    # consecutive newlines (paragraph break), ending with ROW_SEP.
    return re.sub(
        r"\x13[^\n]*(?:\n[^\n]+)*\x13",
        wrap,
        text,
    )


_MD_TRANSFORMS = [
    # Fenced code blocks — wrap content with CODE_START / CODE_END markers.
    # Audio side strips the wrapper + content and emits an earcon
    # (preserving today's "code block → THOK" audible behavior). The
    # mirror webview parses these markers into <pre> blocks.
    (re.compile(r"```.*?```", re.DOTALL), _wrap_code_block),
    (re.compile(r"~~~.*?~~~", re.DOTALL), _wrap_code_block),

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
    # Table cell rows — substitute each row with CELL_SEP-joined cells
    # surrounded by ROW_SEP markers. _wrap_table_blocks (run after the
    # _MD_TRANSFORMS loop in strip_markdown) folds consecutive rows into
    # a TABLE_START / TABLE_END block. Audio side replaces CELL_SEP with
    # ", " and ROW_SEP with ". " (matching today's prose rendering);
    # mirror.html parses the markers into a <table> with <tr>/<td>.
    (_TABLE_ROW_RE, _row_to_cells),

    # Headings (# … ######) — strip the marker, append period so TTS
    # pauses between the heading and the body. Trailing `#`s (closed-atx
    # style) are also removed.
    (
        re.compile(r"(?m)^[ \t]{0,3}#{1,6}[ \t]+([^\n]+?)[ \t]*#*[ \t]*$"),
        lambda m: m.group(1).rstrip(".") + ".",
    ),

    # Unordered bullets — strip the marker; the newline already gives a pause.
    (re.compile(r"(?m)^(\s*)[-*+]\s+"), r"\1"),
    # Ordered list markers — keep the number so the listener hears "one, two,
    # three" as they go by, but turn the period into a comma. The period
    # would otherwise register as an end-of-sentence and (a) drop the number
    # from the following phrase's prosody and (b) get eaten as a bare
    # sentence fragment by pysbd in some pathological cases.
    (re.compile(r"(?m)^(\s*)(\d+)\.\s+"), r"\1\2, "),

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
    # After per-row CELL_SEP/ROW_SEP substitution, fold consecutive rows
    # into a single TABLE_START / TABLE_END block so downstream consumers
    # see one structural unit per markdown table.
    text = _wrap_table_blocks(text)
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


# Lines that contain ANY structural sentinel (earcon, table, code block)
# are content — chrome heuristics must not eat them. A markered line
# typically has only short cell tokens between sentinel bytes (e.g.
# "\x11A\x12B\x14") which would otherwise look like nav-chrome AND
# all-caps to the heuristics that follow.
_STRUCTURAL_SENTINELS = (
    EARCON_SENTINEL,    # \x1e
    TABLE_START, CELL_SEP, ROW_SEP, TABLE_END,  # \x11-\x14
    CODE_START, CODE_END,                        # \x15-\x16
)


def _has_structural_sentinel(s: str) -> bool:
    for sentinel in _STRUCTURAL_SENTINELS:
        if sentinel in s:
            return True
    return False


def _is_short_chrome_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > _SHORT_LINE_MAX_CHARS:
        return False
    if _SENTENCE_PUNCT.search(stripped):
        return False
    # Structural sentinels from strip_markdown aren't chrome — leave alone.
    if _has_structural_sentinel(stripped):
        return False
    return True


def _is_all_caps_micro_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > _ALLCAPS_LINE_MAX_CHARS:
        return False
    if _has_structural_sentinel(stripped):
        return False
    # Sentence punctuation (comma/period/semicolon/etc.) signals content,
    # not a navigation tag. Without this check, table data rows like
    # "Q1, 100" — one capital letter plus digits — were getting flagged
    # as all-caps micro-lines and dropped, leaving listeners hearing only
    # the table's header. Mirrors the same check in _is_short_chrome_line.
    if _SENTENCE_PUNCT.search(stripped):
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


# --- Dashes --------------------------------------------------------------

# Phrasal dashes (em/en/double-hyphen/space-flanked hyphen) become commas
# so Kokoro applies its brief comma pause. Compound-word hyphens
# ("first-audio"), number ranges ("1-10"), negatives ("-5"), and CLI flag
# fragments ("ls -la") all stay untouched because they lack horizontal
# whitespace on both sides.
#
# Horizontal whitespace only ([ \t]+, not \s+): a trailing dash at a line
# break is a stylistic artifact, not a phrase boundary; eating the newline
# around it would collapse two lines into one.
_DASH_TRANSFORMS = [
    # Em-dash (U+2014) and en-dash (U+2013), space-flanked.
    (re.compile(r"[ \t]+[—–][ \t]+"), ", "),
    # Double-hyphen em-dash substitute, space-flanked. `---` horizontal
    # rules are stripped earlier by _MD_TRANSFORMS; a prose `---` (three
    # dashes) won't match because no internal pair is whitespace-flanked.
    (re.compile(r"[ \t]+--[ \t]+"), ", "),
    # Single hyphen, space-flanked — the "phrasal" case. Order-independent
    # of the double-hyphen rule above since `foo -- bar` has no single
    # `-` that's whitespace-flanked (the other dash blocks it).
    (re.compile(r"[ \t]+-[ \t]+"), ", "),
]


def normalize_dashes(text: str) -> str:
    for pattern, repl in _DASH_TRANSFORMS:
        text = pattern.sub(repl, text)
    return text


# --- Numbers -------------------------------------------------------------
#
# Spell out numerals, currency, ordinals, percentages, and integers with
# thousands separators so Kokoro receives words rather than glyphs. Runs
# AFTER dash normalization (dashes → commas) and BEFORE the pronunciation
# dictionary so dict entries can still act on the spelled-out output.
#
# Optional: if num2words is not installed (e.g. on a fresh /usr/bin/python3
# without user-site packages), the function degrades to a no-op passthrough
# so the rest of the pipeline keeps working.

try:
    import num2words as _num2words
    _NUMBERS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _NUMBERS_AVAILABLE = False


def _year_words(y: int) -> str:
    """Return the spoken form of a 4-digit year.

    1100-1999 and 2010-2099: split hi/lo (e.g. 2026 -> "twenty twenty-six").
    1000-1099 and 2000-2009: full cardinal (e.g. 2001 -> "two thousand and one").
    Exact centuries: "nineteen hundred" etc. (except 2000 -> "two thousand").
    """
    hi = y // 100
    lo = y % 100
    if y <= 1099 or (2000 <= y <= 2009):
        return _num2words.num2words(y)
    elif lo == 0:
        return _num2words.num2words(hi) + " hundred"
    else:
        return _num2words.num2words(hi) + " " + _num2words.num2words(lo)


# Pattern to find 4-digit years in the range 1100-2029.
_YEAR_BARE = re.compile(r"\b(1[1-9]\d{2}|20[0-2]\d)\b")
# Preceding words that signal a year is being used as a year.
_YEAR_TRIGGER_BEFORE = re.compile(
    r"(?:in|since|until|before|after|around|by|from|through|during|circa|year|ca\.?|c\.)\s+$",
    re.IGNORECASE,
)
# Following text that signals a year reading is correct.
_YEAR_TRIGGER_AFTER = re.compile(
    r"^\s*(?:AD|BC|CE|BCE\b|[,\.]\s|\s+(?:was|is|will|are|saw|marks|marked|began|ended|started|witnessed))",
    re.IGNORECASE,
)

# Currency: optional space after $, comma-grouped digits, optional cents.
_CURRENCY_RE = re.compile(r"\$\s?[\d,]+(?:\.\d{1,2})?")
# Ordinals: digits followed immediately by st/nd/rd/th.
_ORDINAL_RE = re.compile(r"\b(\d+)(st|nd|rd|th)\b", re.IGNORECASE)
# Percentages: digits (with optional decimal) immediately before %.
_PERCENT_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s?%")
# Integers with comma thousands-separators (at least one comma group).
_THOUSANDS_RE = re.compile(r"\b\d{1,3}(?:,\d{3})+\b")


def _replace_currency(m):
    # type: (re.Match) -> str
    raw = m.group(0).lstrip("$").replace(",", "").strip()
    if "." in raw:
        dollar_str, cent_str = raw.split(".", 1)
        dollars = int(dollar_str) if dollar_str else 0
        cents = int(cent_str.ljust(2, "0")[:2])
    else:
        dollars = int(raw)
        cents = 0
    dw = _num2words.num2words(dollars)
    dollar_label = "dollar" if dollars == 1 else "dollars"
    if cents:
        cw = _num2words.num2words(cents)
        cent_label = "cent" if cents == 1 else "cents"
        return "{} {} and {} {}".format(dw, dollar_label, cw, cent_label)
    return "{} {}".format(dw, dollar_label)


def _replace_ordinal(m):
    # type: (re.Match) -> str
    return _num2words.num2words(int(m.group(1)), to="ordinal")


def _replace_percent(m):
    # type: (re.Match) -> str
    n = m.group(1)
    val = float(n) if "." in n else int(n)
    return _num2words.num2words(val) + " percent"


def _replace_thousands(m):
    # type: (re.Match) -> str
    return _num2words.num2words(int(m.group(0).replace(",", "")))


def normalize_numbers(text):
    # type: (str) -> str
    """Spell out numbers, currency, ordinals, and percentages in *text*.

    Order of substitution:
      1. Currency (``$1,234.56``) -- must go first so the comma-grouped
         digits don't get consumed by the thousands rule.
      2. Ordinals (``3rd``, ``21st``) -- must go before bare integers so
         ``3rd`` isn't turned into ``3 rd``.
      3. Percentages (``42%``).
      4. Years (``2026``) -- only when context suggests a calendar year.
      5. Comma-grouped integers (``1,234``).

    If ``num2words`` is not installed, returns *text* unchanged.
    """
    if not _NUMBERS_AVAILABLE:
        return text

    # 1. Currency
    text = _CURRENCY_RE.sub(_replace_currency, text)

    # 2. Ordinals
    text = _ORDINAL_RE.sub(_replace_ordinal, text)

    # 3. Percentages
    text = _PERCENT_RE.sub(_replace_percent, text)

    # 4. Years -- context-sensitive substitution. We snapshot the string
    #    before substitution so before/after slices stay stable throughout
    #    the re.sub pass (re.sub passes original match positions to the
    #    callable, but the closure must refer to the *same* string the
    #    regex was run against).
    _year_source = text

    def _replace_year(m):
        before = _year_source[: m.start()]
        after = _year_source[m.end() :]
        if _YEAR_TRIGGER_BEFORE.search(before) or _YEAR_TRIGGER_AFTER.match(after):
            return _year_words(int(m.group(1)))
        return m.group(0)

    text = _YEAR_BARE.sub(_replace_year, _year_source)

    # 5. Comma-grouped integers (run after years so "1,900" isn't eaten first)
    text = _THOUSANDS_RE.sub(_replace_thousands, text)

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


# Below this visible-text length, Kokoro prosody goes flat/clipped ("OK.",
# "Got it.") because the duration model lacks phoneme context. We fuse such
# sentences into a neighbor so the synth sees enough to inflect naturally.
# 15 chars captures "That's fine." (12c) and shorter; "I agree with you."
# (17c) stays independent.
_SHORT_SENTENCE_CHARS = 15

# Matches all structural sentinels plus whitespace — used to measure how much
# actual text a sentence-segment carries (as opposed to punctuation or pure
# structural markers).
_SENTINEL_OR_WS = re.compile(r"[\x1c\x1d\x1e\s]")


def _visible_text_length(segment: str) -> int:
    return len(_SENTINEL_OR_WS.sub("", segment))


def merge_short_sentences(text: str) -> str:
    """
    Post-processing over the sentinel-tagged text from split_sentences.
    Merges sentences whose visible text is shorter than _SHORT_SENTENCE_CHARS
    with an adjacent sentence so Kokoro has enough phoneme context for
    natural prosody. Never crosses a paragraph boundary (\\x1c prefix).
    """
    if SENTENCE_SENTINEL not in text:
        return text
    parts = text.split(SENTENCE_SENTINEL)
    starts_para = [p.startswith(PARAGRAPH_SENTINEL) for p in parts]

    def has_text(i: int) -> bool:
        return 0 <= i < len(parts) and _visible_text_length(parts[i]) > 0

    i = 0
    while i < len(parts):
        vis = _visible_text_length(parts[i])
        # Segments with no visible text at all (pure earcon between
        # paragraphs, say) aren't sentences — skip them rather than trying
        # to fuse them into a neighbor and swallowing the earcon.
        if vis == 0 or vis >= _SHORT_SENTENCE_CHARS:
            i += 1
            continue

        # Forward merge: pull the next sentence in if it's in the same
        # paragraph and has real text.
        if i + 1 < len(parts) and not starts_para[i + 1] and has_text(i + 1):
            prefix = PARAGRAPH_SENTINEL if starts_para[i] else ""
            cur = parts[i].lstrip(PARAGRAPH_SENTINEL)
            parts[i + 1] = prefix + cur.rstrip() + " " + parts[i + 1]
            starts_para[i + 1] = starts_para[i]
            del parts[i]
            del starts_para[i]
            continue  # recheck: the merged segment may still be short

        # Backward merge: stick onto the previous sentence, but only if the
        # current sentence isn't itself a paragraph-start (don't blur the
        # pause between two paragraphs).
        if i > 0 and not starts_para[i] and has_text(i - 1):
            parts[i - 1] = parts[i - 1].rstrip() + " " + parts[i]
            del parts[i]
            del starts_para[i]
            # The growing parts[i-1] was already over threshold (we never
            # leave a short segment behind on a forward pass), so no recheck
            # needed — advance.
            continue

        # Singleton short paragraph with no valid merge candidate. Leave it;
        # the intentional paragraph break matters more than the prosody hit.
        i += 1

    return SENTENCE_SENTINEL.join(parts)


def _split_sentences_pysbd(text: str) -> str:
    """Split sentences using pysbd, preserving paragraph boundaries and sentinels."""
    # Split on paragraph boundaries first, preserving inter-paragraph separators.
    # We use re.split with a capturing group so we know where boundaries were.
    parts = _PARAGRAPH_BOUNDARY.split(text)

    result_sentences: list[str] = []
    for para_idx, para in enumerate(parts):
        if not para:
            continue
        # pysbd can treat terminal punctuation followed by a control character
        # (e.g. EARCON_SENTINEL \x1e) as a sentence boundary, which would
        # split "foo.\x1ebar" into ["foo.", "bar"] and lose the earcon from
        # its sentence.  Protect earcons with a temporary ASCII placeholder
        # before segmentation, then restore them in each returned segment.
        _EARCON_PH = "\x02EARCON\x03"
        para_safe = para.replace(EARCON_SENTINEL, _EARCON_PH)

        # Segment this paragraph into sentences.
        raw_sentences = _PYSBD_SEGMENTER.segment(para_safe)
        # Strip trailing whitespace pysbd may leave; drop empties.
        # Restore earcon sentinels that were temporarily replaced.
        sentences = [s.strip().replace(_EARCON_PH, EARCON_SENTINEL) for s in raw_sentences]
        sentences = [s for s in sentences if s]
        if not sentences:
            continue
        if para_idx > 0:
            # Prefix first sentence of each non-first paragraph with the
            # paragraph sentinel so downstream sees the longer-pause marker.
            sentences[0] = PARAGRAPH_SENTINEL + sentences[0]
        result_sentences.extend(sentences)

    return SENTENCE_SENTINEL.join(result_sentences)


def _split_sentences_regex(text: str) -> str:
    """Regex-based sentence splitter — stdlib fallback when pysbd is absent."""
    # Paragraph first: replace the blank-line run with a sentence sentinel
    # (so sentence-level splitting still sees the boundary) plus a paragraph
    # prefix (so the next sentence carries "longer pause before me").
    text = _PARAGRAPH_BOUNDARY.sub(SENTENCE_SENTINEL + PARAGRAPH_SENTINEL, text)
    text = _SENTENCE_BOUNDARY.sub(SENTENCE_SENTINEL, text)
    # Collapse runs of sentence sentinels with whitespace between them.
    # \x1c deliberately not in the whitespace class — it survives as the
    # paragraph prefix on the next surviving sentinel.
    text = re.sub(rf"\x1d({_WS}*\x1d)+", SENTENCE_SENTINEL, text)
    return text


def split_sentences(text: str) -> str:
    if _HAVE_PYSBD:
        text = _split_sentences_pysbd(text)
    else:
        text = _split_sentences_regex(text)
    # If a paragraph marker ends up at the very start of the output (the
    # selection began with a blank line), there is no "before" to pause
    # against — drop leading markers.
    text = text.lstrip(PARAGRAPH_SENTINEL)
    return text


def main() -> int:
    text = sys.stdin.read()
    text = normalize_whitespace(text)
    text = maybe_extract_article(text)
    text = strip_markdown(text)
    text = clean_web_artifacts(text)
    text = normalize_dashes(text)
    text = normalize_numbers(text)
    rules = load_pronunciations(HERE / "pronunciations.txt")
    text = apply_pronunciations(text, rules)
    text = split_sentences(text)
    text = merge_short_sentences(text)
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
