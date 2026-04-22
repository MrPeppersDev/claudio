# Claudio

Local, offline text-to-speech for Claude Code on macOS. F13 reads either
a text selection or the last assistant message from the focused iTerm
session, synthesized by a local Kokoro-82M model.

- **Fully offline** — no API keys, no network. Once installed, everything
  runs against weights on disk.
- **~660 MB RAM** when the server is running; manually stoppable from the
  menu bar to release.
- **Synth in 0.7–1.6 s** warm, 0.4 s cold-start for the model.

## What's in this repo

This repo is a *wrapper* around upstream Kokoro. It ships ~25 KB of
augmentation code — the model, the ONNX runtime, and the inference library
are installed separately from their upstream sources:

- `pip install kokoro-onnx` → upstream inference library (Apache 2.0)
- `download-model.sh` → fetches the ONNX weights + voice embeddings from
  [thewh1teagle/kokoro-onnx releases][upstream]

What Claudio adds on top:

- **Shared text preprocessing** — markdown-to-speech (strip code blocks,
  flatten links, normalize headings/bullets) + a pronunciation dictionary
  (`API` → `ay pee eye`, `kubectl` → `kube control`, etc.)
- **Split-speed synthesis** — model runs at 1.5× and afplay's phase vocoder
  covers the remaining 1.33× to hit effective 2.0× without muddied phonemes
- **Menu bar UI + F13 hotkey** (Hammerspoon)
- **Selection-or-last-message** capture — picks up highlighted text if any,
  otherwise extracts the last assistant message from the focused iTerm pane
- **Session-scoped transcripts** — the claudetop statusline plugin writes
  `iterm_session_id → transcript_path` so F13 reads from the *focused* pane,
  not the globally-most-recent one

## Layout

```
.
├── play-last.sh               # top-level orchestrator; owns the job lock
├── preprocess.py              # markdown + pronunciation + sentence split
├── pronunciations.txt         # FROM|TO dictionary for tech jargon
├── common-phrases.txt         # stock utterances for cache pre-warm
├── kokoro-tts.sh              # backend wrapper (HTTP → server → afplay)
├── kokoro-server.sh           # start/stop/status for the local Python server
├── warm-cache.sh              # pre-synthesize common-phrases.txt into cache
├── kokoro/
│   ├── server.py              # stdlib http.server wrapping kokoro-onnx
│   ├── requirements.txt
│   ├── download-model.sh      # fetches .onnx + voices .bin from upstream
│   ├── kokoro-v1.0.onnx       # (gitignored, 310 MB, pulled at install)
│   └── voices-v1.0.bin        # (gitignored, 27 MB, pulled at install)
├── hammerspoon/
│   └── claudio.lua            # menubar dropdown, F13 hotkey, selection capture
├── claudetop.d/
│   └── claudio-session-map    # statusline plugin: iTerm session → transcript
└── install.sh                 # bootstrap for a new machine
```

## Install

```bash
git clone <repo> ~/.claude/claudio
cd ~/.claude/claudio
./install.sh
```

Install does:
1. Symlinks `hammerspoon/claudio.lua` and the claudetop plugin to their
   runtime locations.
2. Creates a Python venv (prefers 3.12; accepts anything ≥ 3.10) and
   installs `kokoro-onnx`, `soundfile`, `scipy`. Override the interpreter
   with `CLAUDIO_PYTHON=/path/to/python3` if auto-detection picks wrong.
3. Downloads Kokoro v1.0 model weights + voices (~340 MB one-time).

Then add to your Hammerspoon `init.lua`:

```lua
local claudio = require("claudio")
claudio.start()
hs.hotkey.bind({}, "F13", function() claudio.toggle() end)
```

## Usage

- **F13** — highlight text in any app, then press F13 to read it aloud.
  Press F13 again to stop.
- **Menu bar icon** — dropdown with Play/Stop and manual server stop.
  `▸` idle, `⟳` synth, `▶` playing.

> **Note:** F13 with no selection currently shows a reminder rather than
> reading the latest assistant message. Full-message playback will come
> back once we can summarize server-side — raw multi-paragraph responses
> are rarely what you actually want to hear.

## Env vars

- `KOKORO_VOICE` — default `af_bella`. See [VOICES.md][voices] for the full list.
  Accepts a weighted blend too: `af_bella:70,am_michael:30`. Weights are
  normalized — `1,1` is a 50/50 mix. Embeddings are averaged server-side.
- `KOKORO_SPEED` — target effective speed. Default `2.0`.
- `KOKORO_SYNTH_CAP` — split point between synth and afplay. Default `1.5`
  (synth at 1.5×, afplay covers the remaining multiplier). Lower values push
  more work to afplay's phase vocoder; higher push it to the model.
- `KOKORO_LANG` — override auto-detected phoneme language (`en-us` / `en-gb`).
- `KOKORO_URL` — server base URL. Default `http://127.0.0.1:8880`.
- `KOKORO_EARCON` — WAV/AIFF played at structural boundaries (where a code
  block was stripped). Default `/System/Library/Sounds/Pop.aiff`.
- `KOKORO_CACHE_DIR` — per-sentence WAV cache. Default `~/.claude/claudio/cache`.
- `KOKORO_CACHE_MAX_MB` — cache cap, mtime-LRU pruned. Default `200`.
- `KOKORO_NO_PLAY` — synth-to-cache and skip playback (used by `warm-cache.sh`).
- `KOKORO_EQ_GAIN_DB` / `KOKORO_EQ_FREQ` / `KOKORO_EQ_Q` — server-side peaking EQ.
  Default `3.0` / `2500` / `1.0`. A small presence bump helps consonant intelligibility
  survive the 1.33× afplay time-stretch. Set `KOKORO_EQ_GAIN_DB=0` to disable.
  All three participate in the cache key, so toggling them auto-invalidates stale entries.

## Pronunciation dictionary

Edit `pronunciations.txt`. Format: `FROM|TO`, one per line, `#` for comments.
Matching is case-sensitive and word-boundary respecting. Longer/more-specific
entries should come first (plurals before singulars). Changes take effect on
the next invocation — no reload needed.

## Cache pre-warming

`./warm-cache.sh` synthesizes every line in `common-phrases.txt` into the
sentence cache with playback disabled. Stock utterances ("Let me check
that.", "Done.") then hit on first real use instead of paying a cold
synth. Run it after changing `KOKORO_VOICE` to warm the new voice's cache
(cache keys include voice, so each voice needs its own warm). `install.sh`
runs it automatically at the end of a fresh install.

## Resource footprint

- RAM: ~660 MB RSS (model held resident in the server process)
- Disk: ~310 MB model + 27 MB voices + ~270 MB venv
- CPU: brief single-core spike during synth, idle otherwise
- Latency: 0.4 s cold start (first synth after boot), 0.7–1.6 s warm synth

## Architecture notes

- **Lock model:** `play-last.sh` owns `job.pid` for its entire run (synth +
  playback). A second invocation while mid-synth kills the whole process
  tree — that's how the menu bar toggle-stop works.
- **Lazy server start:** `play-last.sh` calls `kokoro-server.sh start` before
  each synth. Idempotent: ~0.05 s no-op when already healthy, up to ~8 s on
  cold boot.
- **Selection capture** (`claudio.lua`): Accessibility API → simulated
  Cmd+C → iTerm copy-on-select fallback. Three paths in order of preference.
- **Session-scoped transcripts:** the `claudetop.d` statusline plugin writes
  `iterm_session_id → transcript_path` on each render; `play-last.sh` uses
  that map so F13 reads the Claude message from the *focused* pane rather
  than the globally-most-recent one.

[upstream]: https://github.com/thewh1teagle/kokoro-onnx/releases
[voices]: https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md
