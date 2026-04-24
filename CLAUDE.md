# Claudio — guide for Claude Code sessions

Everything in this file is aimed at a Claude instance that's just walked into the repo. The `README.md` is the human-facing install/usage doc; this file captures the *conventions, gotchas, and load-bearing design decisions* that aren't obvious from reading the code.

For live session state (current branch, in-flight work, recent PRs) see `.claude/notes/SESSION.md` — that file is gitignored.

## Pipeline in one pass

Text comes in on stdin → `preprocess.py` (markdown strip, pronunciation dict, sentence split with 0x1D/0x1E/0x1C delimiter bytes) → `kokoro-tts.sh` (per-sentence cache lookup, HTTP POST to local server, spawns `play-stream.py` child) → `kokoro/server.py` (phonemize, chunk to ≤500 phonemes, synth, EQ, head/tail trim, pad, WAV encode) → `kokoro/play-stream.py` (one long-lived `sounddevice.OutputStream`, reads `PLAY`/`PAUSE`/`EARCON`/`RATE` commands from fd 9, stretches with `sox tempo -s`).

The top-level entry point is `play-last.sh`, which owns the job lock (`job.pid`) and routes to `kokoro-tts.sh`. Hammerspoon (`hammerspoon/claudio.lua`) binds F13 → `play-last.sh`.

## Files worth knowing about

| File | Role |
|---|---|
| `play-last.sh` | Entry point. Job lock, state file, routes into `kokoro-tts.sh`. |
| `kokoro-tts.sh` | Core pipeline. Sentence loop, cache, curl, play-stream spawn. **Env var catalog lives in this file's header** — don't duplicate it here. |
| `preprocess.py` | Markdown/HTML/PDF → pronounceable text. Delimits with 0x1D (sentence), 0x1E (segment), 0x1C (paragraph-start marker). |
| `kokoro/server.py` | Local HTTP synth server. Single-threaded on purpose (one ORT session). Binds loopback only unless `KOKORO_ALLOW_REMOTE=1`. |
| `kokoro/play-stream.py` | Persistent `OutputStream`. One process per `kokoro-tts.sh` invocation. |
| `kokoro-server.sh` | Start/stop/restart/status for the Python server. Keeps it resident across F13 presses. |
| `warm-cache.sh` | Pre-synth common phrases so first-use hits cache. |
| `scripts/try-voices.sh` | A/B shortlist for picking a voice. |
| `scripts/bench_stretch.py` | Benchmark rig for comparing pitch-preserving time stretchers. |
| `tests/` | pytest; imports `kokoro/server.py` with stubbed `kokoro_onnx`/`scipy`. |

## Load-bearing design decisions (non-obvious)

These are the kinds of things that cost hours to rediscover if you don't know they're deliberate.

- **Split-speed synthesis**. Target effective rate = `KOKORO_SPEED` (default 2.0). Kokoro synthesizes up to `KOKORO_SYNTH_CAP` (default 1.3), and `play-stream.py` covers the rest with `sox tempo -s`. Kokoro itself degrades above ~1.3x (enunciation softens, first consonants slur). Pitch-preserving stretch picks up the slack. Don't raise SYNTH_CAP back to 1.5+ without listening.
- **Sacrificial head-word** (`KOKORO_SACRIFICIAL_WORD`, default `banana`). Kokoro drops the first consonant of each sentence — the acoustic model's warm-up eats it. We prepend `"banana, {text}"`, synth, then the server cuts audio up to the first post-comma silence gap using a forward-walking windowed-RMS detector. Fails open (audible "banana, …") rather than silent garbling.
- **400ms silent pre-roll** in `play-stream.py`. CoreAudio device warm-up + sox-tempo cold start + ring-buffer drain all overlap the first real samples if the pre-roll is too short. 200ms wasn't enough on Bluetooth/virtual outputs.
- **3ms linear edge fade** on every PLAY buffer. Kokoro WAVs can end/begin at non-zero amplitude; direct concatenation makes an audible DC-step click (woodblock-ish). 3ms fade-in + fade-out is inaudible as a pause but eliminates the discontinuity.
- **Windowed-RMS tail trim with 60ms dwell**. Single-sample threshold trims ate unvoiced stop releases (t/p/k sit at 0.01–0.04 RMS but their release bursts are brief). The dwell requirement preserves them while still trimming dead air.
- **Coalesce short paragraphs** (`KOKORO_COALESCE_MAX_CHARS`, default 400). Per-sentence synth loses Kokoro's paragraph-level prosody — short paragraphs synth as one unit to keep natural breath patterns.
- **Cache key includes every synthesis-affecting knob**. Voice, synth speed, lang, EQ, pad, tail-trim, sacrificial word, and text. Toggling any env var auto-invalidates stale cache entries — never "why is this sentence the old voice?" again.
- **Server binds loopback only** by default; `KOKORO_ALLOW_REMOTE=1` is the explicit opt-in. No auth.

## Env vars

The canonical catalog is in the header comment of `kokoro-tts.sh`. Go read it there — duplicating it here guarantees it'll drift. Server-side validation (ranges, types) lives in `kokoro/server.py`.

## Workflow conventions

- **Never commit directly to `main`**. Always: issue → branch → draft PR → work → merge. Squash-merge + delete branch on merge.
- **Commit co-author line** is `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>`.
- **Never use `--no-verify`, `--amend`, or force-push** without explicit approval. If a pre-commit hook fails, fix the issue and make a new commit.
- When a cherry-pick onto post-squash-merge main conflicts, use `git rebase --onto main <squashed-upstream-tip> <branch>` to replay only the unique commits. I've gotten burned by this — if you branched from a branch that was squash-merged, a plain rebase will try to re-apply every ancestor commit.

## Running things

```bash
# Start/restart the synth server (kept resident):
./kokoro-server.sh start
./kokoro-server.sh restart   # after code changes to kokoro/server.py

# One-off synth + play:
echo "Your text here." | ./kokoro-tts.sh

# Warm the cache for common phrases:
./warm-cache.sh

# Pick a voice:
./scripts/try-voices.sh

# Run tests (pytest, stubs heavy deps so no model needed):
python3 -m pytest tests/ -v
```

## Testing & verification

- Python test suite imports `kokoro/server.py` with `kokoro_onnx` and `scipy` stubbed (see the top of `tests/test_server_trim.py` for the pattern). Add new tests in the same file or a peer file using the same stub block.
- For audio-quality changes: `./kokoro-server.sh restart` (so the server picks up code), then listen. Don't rely on tests alone for "does it sound right" questions — the ear catches what RMS doesn't.
- `kokoro-server.log` has the server's per-request telemetry (phonemes, chunks, head_trim, synth time). Check here when a change should have had an effect and seemingly didn't.

## Things I've tried that didn't work

Worth recording so we don't re-walk these paths:
- `scipy.signal.resample_poly` for time stretch → chipmunk voice (it's sample-rate conversion, not phase vocoder). Use sox.
- `sleep` between `PAUSE` commands in shell instead of in-stream PAUSE → breaks the persistent-stream invariant. Always PAUSE through the fd-9 pipe so gaps happen on-stream.
- 200ms pre-roll → not enough on Bluetooth/AirPlay. 400ms is the floor.
- Period after sacrificial word (`"banana. recap"`) → Kokoro sometimes treats it as two independent utterances, which defeats the warm-up trick. Comma is correct.

## Interacting with the user

- Terse responses, no trailing summaries.
- Don't write exploratory planning docs or markdown deliverables unless asked.
- For UI/audio changes, playtest before claiming done — type checks don't verify sound.
- When a design choice has tradeoffs (e.g. fail-open vs fail-quiet), flag them before committing.
