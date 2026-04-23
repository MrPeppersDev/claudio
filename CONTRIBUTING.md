# Contributing to Claudio

Small project, small process. A few notes to make PRs smooth.

## What's in scope

Claudio is a thin wrapper around upstream [Kokoro-82M][kokoro] for Claude
Code on macOS. Good PRs:

- Bug fixes in the wrapper scripts (`play-last.sh`, `kokoro-tts.sh`, etc.)
- Preprocessing improvements — markdown handling, pronunciation
  dictionary entries
- Menu bar / Hammerspoon UX tweaks
- Performance work (cache behavior, pipelining)
- Documentation / README clarity

Out of scope:

- Changes to the upstream Kokoro model or ONNX runtime (file those with
  [kokoro-onnx][kokoro])
- Cross-platform support (Linux / Windows) — macOS-only for now, Apple
  system frameworks (`afplay`, Hammerspoon) are load-bearing
- API-based TTS backends — the point is offline and local

## Testing locally

No test suite — this is a shell + Lua wrapper around a local HTTP server.
Smoke test instead:

```bash
# With an existing install
echo "Quick brown fox." | ./kokoro-tts.sh   # you should hear it
./kokoro-server.sh status                   # should show healthy
./warm-cache.sh                             # should exit 0
```

For Hammerspoon changes, reload the config (`hs -c 'hs.reload()'`) and hit
F13 with text selected.

For a full install test on a fresh machine, see `install.sh` — it's
idempotent, so running it twice should be harmless.

## Style

- Shell: `set -euo pipefail` at the top, double-quote variables,
  `shellcheck`-clean if possible
- Lua: match what's in `hammerspoon/claudio.lua` — no frameworks, plain
  Hammerspoon API
- Python: `server.py` + `preprocess.py` use stdlib + optional user-site deps;
  avoid adding further heavy deps without a graceful fallback. Optional deps:
  - `pysbd` (MIT) — sentence boundary detection; falls back to regex splitter.
    Install: `pip install --user pysbd`
  - `num2words` (LGPL-2.1) + `inflect` (MIT) — number/currency/ordinal
    spell-out; skipped when unavailable. Install: `pip install --user num2words inflect`
  - `trafilatura` (Apache-2.0) — article extraction from HTML pastes;
    skipped when unavailable. Install: `pip install --user trafilatura lxml_html_clean`
  - `pymupdf` (AGPL-3.0, acceptable for local use) — `pdf_extract.py` reads
    PDFs directly so PDF viewers don't need to fall back to the Live Text
    nudge. Install: `pip install --user pymupdf`

Comments explain *why*, not *what*. If a block does something surprising
(e.g., why earcons are synchronous), say so.

## PRs

- One logical change per PR
- Rebase cleanly on `main`
- Reference the issue number in the title or body if one exists
- Draft PRs welcome while you iterate

## Reporting bugs

Open an issue with steps to reproduce and the contents of
`~/.claude/claudio/play.log` and `~/.claude/claudio/tts.log` if relevant.

[kokoro]: https://github.com/thewh1teagle/kokoro-onnx
