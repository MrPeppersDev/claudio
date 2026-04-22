# Speechify

Local + cloud text-to-speech for Claude Code on macOS, with a Hammerspoon
menu-bar dropdown and an F13 hotkey that reads either a text selection or
the last assistant message from the focused iTerm session.

Two backends you can hot-swap from the menu:

- **Speechify** — cloud API (`george` voice, SSML `+200%` prosody ≈ 2.5×)
- **Kokoro-82M** — fully local, offline, Apache 2.0. Python HTTP server
  wraps `kokoro-onnx`; default voice `af_bella` at effective 2.0× (synth
  1.5× + afplay 1.33×, pitch-preserving). No API key, no network.

Only one backend is live at a time — switching to Speechify stops the
Kokoro server (frees ~600 MB); switching to Kokoro spins it up.

## Layout

```
.
├── play-last.sh              # top-level orchestrator; owns the job lock
├── speechify-tts.sh          # Speechify cloud backend wrapper
├── kokoro-tts.sh             # Kokoro local backend wrapper
├── kokoro-server.sh          # start/stop/status for the local Python server
├── kokoro/
│   ├── server.py             # stdlib http.server wrapping kokoro-onnx
│   ├── requirements.txt
│   ├── download-model.sh     # fetches .onnx + voices .bin from GitHub release
│   ├── kokoro-v1.0.onnx      # (gitignored, 310 MB)
│   └── voices-v1.0.bin       # (gitignored, 27 MB)
├── hammerspoon/
│   └── speechify.lua         # menubar dropdown, F13 hotkey, selection capture
├── claudetop.d/
│   └── speechify-session-map # statusline plugin: iTerm session → transcript
└── install.sh                # bootstrap for a new machine
```

## Install

```bash
# Clone to the expected path (scripts hardcode ~/.claude/speechify)
git clone <repo> ~/.claude/speechify
cd ~/.claude/speechify
./install.sh
```

Install does:
1. Symlinks `hammerspoon/speechify.lua` and the claudetop plugin to their
   runtime locations.
2. Creates a Python 3.12 venv and installs `kokoro-onnx`, `soundfile`, `scipy`.
3. Downloads Kokoro v1.0 model weights + voices (~340 MB one-time).

## Usage

- **F13** — toggle playback of selection (if any) or last Claude message.
- **Menu bar icon** — dropdown with Play/Stop, backend picker, live status.
  Title is `▸·s` idle/Speechify, `▸·k` idle/Kokoro, with `⟳` and `▶` for
  synth and play respectively.

## Env vars (backend tuning)

**Kokoro:**
- `KOKORO_VOICE` — default `af_bella`. See [VOICES.md][voices] for the full list.
- `KOKORO_SPEED` — target effective speed. Default `2.0`.
- `KOKORO_SYNTH_CAP` — split point between synth and afplay. Default `1.5`
  (synth at 1.5×, afplay covers the remaining multiplier). Lower values push
  more work to afplay's phase vocoder; higher push it to the model.
- `KOKORO_LANG` — override auto-detected phoneme language (`en-us` / `en-gb`).
- `KOKORO_URL` — server base URL. Default `http://127.0.0.1:8880`.

**Speechify:**
- `SPEECHIFY_VOICE` — default `george`.
- `SPEECHIFY_MODEL` — default `simba-english`.
- `SPEECHIFY_SPEED` — SSML prosody rate. Default `+200%` (empirically ≈ 2.5×).
- `SPEECHIFY_KEY_FILE` — default `~/.config/fletch/speechify-api-key`.

## Backend switching

From the menu bar, or manually:

```bash
echo kokoro    > ~/.claude/speechify/backend    # local
echo speechify > ~/.claude/speechify/backend    # cloud
rm             ~/.claude/speechify/backend      # default (speechify)
```

The Hammerspoon dropdown starts/stops the Kokoro server automatically as
you switch, so whichever backend is selected is the only one using RAM.

## Resource footprint

**Kokoro server (when selected):**
- RAM: ~660 MB RSS (model held resident)
- Disk: ~310 MB model + 27 MB voices + ~270 MB venv
- CPU: brief single-core spike during synth, idle otherwise
- Latency: 0.4 s cold start, 0.7–1.6 s warm synth depending on length

**Speechify (when selected):**
- RAM: 0 (no daemon)
- Network: HTTPS to api.speechify.ai per request
- Latency: ~1.5–2.5 s time-to-first-audio (network + cloud synth)

## Architecture notes

- **Lock model:** `play-last.sh` owns `job.pid` for its entire run (synth +
  playback). A second invocation while mid-synth kills the whole process
  tree — that's how the menu bar toggle-stop works.
- **Selection capture** (`speechify.lua`): Accessibility API → simulated
  Cmd+C → iTerm copy-on-select fallback. Three paths in order of preference.
- **Session-scoped transcripts:** the `claudetop.d` statusline plugin writes
  `iterm_session_id → transcript_path` on each render; `play-last.sh` uses
  that map so F13 reads the Claude message from the *focused* pane rather
  than the globally-most-recent one.

[voices]: https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md
