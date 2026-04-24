#!/bin/bash
# kokoro-tts.sh — local Kokoro-82M TTS via our own Python HTTP server.
#
# Reads text on stdin, synthesizes locally (no external network), plays via
# a persistent play-stream.py child (sounddevice OutputStream). The server
# is managed by kokoro-server.sh, which play-last.sh starts lazily on the
# first F13 after a reboot.
#
# play-stream.py command protocol (fd 9, one line per command):
#   PLAY <id> <wav>  — play WAV with buffer id (integer, increments per sentence)
#   EARCON <path>    — play earcon at 1.0x
#   PAUSE            — soft-pause (stream stays alive, position frozen)
#   PAUSE <ms>       — backward-compat: insert <ms> of silence
#   RESUME           — resume after soft-pause
#   SEEK <ms>        — relative seek in source-domain ms (±)
#   STOP             — clear queue, halt playback
#   RATE <float>     — set sox-tempo stretch factor
#
# POS output: play-stream.py writes "POS <ms> <id>" lines to stderr at ~20ms
# cadence while playing. <ms> is in the source WAV's time frame (pre-stretch)
# so it aligns with the PR-1 timing sidecar word offsets. Filter stderr lines
# starting with "POS " to consume the playhead position.
#
# Env vars:
#   KOKORO_VOICE       voice name, or a weighted blend (default: af_sarah).
#                      Blend syntax: "af_bella:70,am_michael:30" — weights
#                      are normalized, so "1,1" is a 50/50 mix.
#   KOKORO_SPEED       target effective speed (default: 2.0)
#   KOKORO_SYNTH_CAP   max synth speed before play-stream picks up the rest
#                      (default: 1.5; see split-speed note below)
#   KOKORO_URL         server base URL (default: http://127.0.0.1:8880)
#   KOKORO_LANG        phoneme lang (auto: bm_/bf_ → en-gb, else en-us)
#   KOKORO_EARCON      WAV/AIFF played at structural boundaries
#                      (default: /System/Library/Sounds/Pop.aiff)
#   KOKORO_CACHE_DIR   per-sentence WAV cache (default: ~/.claude/claudio/cache)
#   KOKORO_CACHE_MAX_MB cache size cap, mtime-LRU pruned (default: 200)
#   KOKORO_NO_PLAY     if "1", synth+cache but skip playback (warm-cache.sh)
#   KOKORO_EQ_GAIN_DB  peaking-EQ gain (default: 3.0; 0 disables)
#   KOKORO_EQ_FREQ     EQ center frequency, Hz (default: 2500)
#   KOKORO_EQ_Q        EQ Q factor (default: 1.0)
#   KOKORO_PAD_START_MS leading silence prepended to each synth WAV so the
#                      first phoneme isn't smushed (default: 150; 0 disables).
#                      Only applied to the FIRST sentence of an utterance —
#                      later sentences get pad=0 so per-sentence seams don't
#                      stack into an audible stutter. play-stream.py adds a
#                      pre-roll at stream open that covers CoreAudio warm-up,
#                      and by the time sentence 2 starts, the stream is
#                      already live so there's no cold start to hide.
#   KOKORO_COALESCE_MAX_CHARS  if a segment is this many chars or fewer, the
#                      whole segment is synthesized as one Kokoro call instead
#                      of per-sentence. Bigger single synth = smoother prosody
#                      across sentence boundaries (Kokoro picks its own breath
#                      pattern) AND fewer sacrificial-word prepends per
#                      utterance, at the cost of losing sentence-level cache
#                      granularity for short messages (default: 700; 0 disables).
#   KOKORO_TAIL_TRIM_MS trailing silence kept after the last audible frame.
#                      Uses windowed-RMS energy detection with a 60ms dwell
#                      so unvoiced closing consonants (t/p/k) survive intact
#                      — the previous single-sample threshold ate release
#                      bursts from t/p/k whose amplitude sits below 0.005.
#                      (default: 40; 0 disables — keep Kokoro's native tail)
#   KOKORO_SACRIFICIAL_WORD  throwaway prefix word. Kokoro's first-token
#                      warm-up eats the initial consonant of every sentence,
#                      so we synth "banana, {real_text}" and the server cuts
#                      the prefix off. Must be a single alphabetic word ≤32
#                      chars. Set to empty string to disable.
#                      (default: banana)
#
# Owned by play-last.sh via job.pid; writes state={synth,play} to the state
# file; play-stream.py is a long-lived child — the parent's pkill -P tears
# everything down on stop.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/kokoro/venv/bin/python3"
PLAY_STREAM="$SCRIPT_DIR/kokoro/play-stream.py"

VOICE="${KOKORO_VOICE:-af_sarah}"
SPEED="${KOKORO_SPEED:-2.0}"
URL="${KOKORO_URL:-http://127.0.0.1:8880}"
# Default lang follows voice family prefix: bm_/bf_ → en-gb, otherwise en-us
if [ -z "${KOKORO_LANG:-}" ]; then
  case "$VOICE" in
    bm_*|bf_*) LANG_CODE="en-gb" ;;
    *)         LANG_CODE="en-us" ;;
  esac
else
  LANG_CODE="$KOKORO_LANG"
fi

STATE_DIR="${CLAUDIO_STATE_DIR:-$HOME/.claude/claudio}"
STATE_FILE="$STATE_DIR/state"
LOG_FILE="$STATE_DIR/tts.log"

mkdir -p "$STATE_DIR"

EARCON="${KOKORO_EARCON:-/System/Library/Sounds/Pop.aiff}"
CACHE_DIR="${KOKORO_CACHE_DIR:-$STATE_DIR/cache}"
CACHE_MAX_MB="${KOKORO_CACHE_MAX_MB:-200}"
EQ_GAIN_DB="${KOKORO_EQ_GAIN_DB:-3.0}"
EQ_FREQ="${KOKORO_EQ_FREQ:-2500}"
EQ_Q="${KOKORO_EQ_Q:-1.0}"
PAD_START_MS="${KOKORO_PAD_START_MS:-150}"
TAIL_TRIM_MS="${KOKORO_TAIL_TRIM_MS:-40}"
COALESCE_MAX_CHARS="${KOKORO_COALESCE_MAX_CHARS:-700}"
SACRIFICIAL_WORD="${KOKORO_SACRIFICIAL_WORD-banana}"
mkdir -p "$CACHE_DIR"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >> "$LOG_FILE"; }

# Hash key: voice|synth_speed|lang|eq_*|pad_start_ms|tail_trim_ms|
# sacrificial_word|synth_version|text. Any change in those produces different
# audio, so they all go into the key. Null bytes between fields prevent "ab|c"
# colliding with "a|bc". Every synthesis-affecting knob is mixed in, so
# toggling any env var (EQ, pad, trim, sacrificial word) auto-invalidates
# stale cache instead of silently mixing pre/post audio.
#
# SYNTH_VERSION bump history:
#   1 (implicit) — original
#   2 — RSVP timing sidecar: patched ONNX model + .words.json written beside
#       every cached WAV. Bumping here forces re-synth so all cache entries
#       gain sidecars.
#   3 — Sacrificial-head trim rework: relative RMS threshold + 20ms dwell
#       + min/max cut-position guardrails. Old WAVs may have "banana,"
#       leakage baked in from the prior detector — bump to force re-synth.
SYNTH_VERSION=3
sentence_hash() {
  local text="$1"
  local pad="$2"
  printf '%s\0%s\0%s\0%s\0%s\0%s\0%s\0%s\0%s\0%s\0%s' \
    "$VOICE" "$SYNTH_SPEED" "$LANG_CODE" \
    "$EQ_GAIN_DB" "$EQ_FREQ" "$EQ_Q" \
    "$pad" "$TAIL_TRIM_MS" "$SACRIFICIAL_WORD" "$SYNTH_VERSION" "$text" \
    | shasum -a 256 | awk '{print $1}'
}

# Evict cache entries by mtime (oldest first) until total size is under cap.
# macOS stat: -f '%m %z %N' → mtime size path. Only called when misses>0
# (see end of synth loop): a pure cache-hit invocation can't have grown the
# cache, so we skip the stat-every-file scan — the hot path that matters
# once the cache has thousands of entries.
prune_cache() {
  local max_bytes=$((CACHE_MAX_MB * 1024 * 1024))
  local total
  total=$(find "$CACHE_DIR" -name '*.wav' -type f -print0 2>/dev/null \
    | xargs -0 stat -f '%z' 2>/dev/null | awk '{s+=$1} END{print s+0}')
  [ "$total" -le "$max_bytes" ] && return
  local freed_before="$total"
  while IFS=' ' read -r mtime size path; do
    [ "$total" -le "$max_bytes" ] && break
    [ -z "$path" ] && continue
    rm -f "$path"
    total=$((total - size))
  done < <(find "$CACHE_DIR" -name '*.wav' -type f -print0 \
    | xargs -0 stat -f '%m %z %N' 2>/dev/null | sort -n)
  log "cache prune: ${freed_before} -> ${total} bytes (cap ${max_bytes})"
}

current_preview() {
  grep -E '^preview=' "$STATE_FILE" 2>/dev/null | head -1 | sed 's/^preview=//'
}
write_state() {
  local state="$1"
  local preview
  preview=$(current_preview)
  printf 'state=%s\npreview=%s\nts=%s\n' "$state" "$preview" "$(date +%s)" > "$STATE_FILE"
}

TEXT=$(cat)
if [ -z "$TEXT" ]; then
  log "ERROR: no input text on stdin"
  exit 3
fi

# Split the target speed between the model and play-stream. Pushing Kokoro
# to its 2.0x cap forces the model to compress phonemes and syllables muddle;
# a gentler synth rate keeps prosody clean, and play-stream's pitch-preserving
# sox tempo covers the rest without blurring consonants.
#   effective_speed = SYNTH_SPEED * PLAYBACK_RATE = SPEED
# Lower KOKORO_SYNTH_CAP → cleaner phonemes, more play-stream stretch.
# Higher KOKORO_SYNTH_CAP → less play-stream work, more model compression.
# 1.3 picked after listening at 1.5 — Sarah (and most voices) slur words
# like "Governance" above synth=1.4x. 1.3x synth + 1.54x sox stretch is
# well within sox's clean-quality range and keeps Kokoro's phonemes crisp.
SYNTH_CAP="${KOKORO_SYNTH_CAP:-1.3}"
SYNTH_SPEED=$(awk -v s="$SPEED" -v c="$SYNTH_CAP" 'BEGIN{print (s>c?c:s)}')
PLAYBACK_RATE=$(awk -v s="$SPEED" -v ss="$SYNTH_SPEED" 'BEGIN{print s/ss}')

# Two-level split:
#   outer \x1e = earcon sentinel (from preprocess.py; ran over code blocks)
#   inner \x1d = sentence sentinel (cache granularity)
# A single-paragraph message with no code produces one segment with N
# sentences; a short reply with no code and one sentence produces a single
# segment/single sentence and behaves like the original single-shot path.
WORK_DIR=$(mktemp -d "$STATE_DIR/segments.XXXXXX")

# FIFO + fixed fd (9) for communicating with play-stream.py. Using a named
# pipe (mkfifo) instead of process substitution so this works with macOS's
# bash 3.2 (which doesn't support {VAR}> automatic fd allocation).
STREAM_FIFO=""
STREAM_PID=""
# Track whether fd 9 is currently open so cleanup never tries to close it twice.
STREAM_FD_OPEN=0

cleanup() {
  # Close fd 9 so play-stream.py hits EOF and begins draining.
  if [ "$STREAM_FD_OPEN" -eq 1 ]; then
    exec 9>&- 2>/dev/null || true
    STREAM_FD_OPEN=0
  fi
  # Wait for play-stream.py to drain and exit (it blocks on writes so EOF
  # wakes it; it then stops the OutputStream and exits).
  if [ -n "$STREAM_PID" ]; then
    wait "$STREAM_PID" 2>/dev/null || true
    STREAM_PID=""
  fi
  [ -n "$STREAM_FIFO" ] && rm -f "$STREAM_FIFO"
  rm -rf "$WORK_DIR"
}
trap cleanup EXIT

printf '%s' "$TEXT" | awk -v RS=$'\x1e' -v dir="$WORK_DIR" '
  { out = sprintf("%s/seg-%04d.txt", dir, NR); printf "%s", $0 > out; close(out) }
'

SEG_COUNT=$(find "$WORK_DIR" -maxdepth 1 -name 'seg-*.txt' | wc -l | tr -d ' ')
log "kokoro synthesize: voice=$VOICE target=${SPEED}x synth=${SYNTH_SPEED}x playback=${PLAYBACK_RATE}x lang=$LANG_CODE chars=${#TEXT} segments=$SEG_COUNT"
write_state synth

have_earcon=0
[ -r "$EARCON" ] && have_earcon=1

# Launch play-stream.py as a persistent child. We connect its stdin to a
# named pipe (FIFO). Opening the reader first then the writer avoids the
# blocking open that would occur if we opened the write end first.
if [ "${KOKORO_NO_PLAY:-0}" != "1" ]; then
  STREAM_FIFO=$(mktemp -u "$STATE_DIR/stream.XXXXXX")
  mkfifo "$STREAM_FIFO"
  # Start the reader (play-stream.py) in the background; it opens the read
  # end and blocks waiting for data.
  "$VENV_PYTHON" "$PLAY_STREAM" < "$STREAM_FIFO" &
  STREAM_PID=$!
  # Open the write end on fd 9. Safe to remove the FIFO name at this point
  # because both ends are now open — the pipe lives in the kernel until
  # both ends close.
  exec 9>"$STREAM_FIFO"
  rm -f "$STREAM_FIFO"
  STREAM_FIFO=""
  STREAM_FD_OPEN=1
  # Send initial playback rate.
  printf 'RATE %s\n' "$PLAYBACK_RATE" >&9
fi

# Interleaved synth + playback. Each iteration resolves its sentence's WAV
# (cache hit = instant; miss = one curl to the local server) then sends the
# path to play-stream.py via fd 9. The next iteration's synth runs while
# play-stream is consuming the current WAV, so server round-trip cost is
# masked behind the previous sentence's audible playback.
hits=0
misses=0
playing=0
first_seg=1
# Tracks whether we've dispatched any sentence yet in this utterance. Drives
# the PAD_START_MS decision: only the very first sentence gets leading
# silence; otherwise per-sentence pads stack up as audible seams.
first_sentence=1
# Buffer id counter: incremented for each PLAY command so play-stream.py and
# downstream consumers (RSVP UI / timing sidecar) can identify which sentence
# is currently playing. Echoed in every POS <ms> <id> message.
play_id=0
for seg_txt in "$WORK_DIR"/seg-*.txt; do
  seg_id=$(basename "$seg_txt" .txt)
  sent_dir="$WORK_DIR/$seg_id.sent"
  mkdir -p "$sent_dir"

  # B: if the segment is short enough, collapse all its sentences into a
  # single synth call so Kokoro handles sentence-boundary prosody natively.
  # \x1c (paragraph-start marker) is stripped — a short segment is unlikely
  # to span paragraphs, and Kokoro infers breath from the text punctuation.
  # \x1d (sentence separator) becomes a space; sentence-ending periods are
  # already in the text, so "Foo.\x1dBar." → "Foo. Bar.".
  seg_chars=$(wc -c < "$seg_txt" | tr -d ' ')
  if [ "$COALESCE_MAX_CHARS" -gt 0 ] && [ "$seg_chars" -le "$COALESCE_MAX_CHARS" ]; then
    tr -d $'\x1c' < "$seg_txt" \
      | awk -v RS=$'\x1d' 'NF { if (seen) printf " "; printf "%s", $0; seen=1 }' \
      > "$sent_dir/0001.txt"
  else
    awk -v RS=$'\x1d' -v dir="$sent_dir" '
      { out = sprintf("%s/%04d.txt", dir, NR); printf "%s", $0 > out; close(out) }
    ' "$seg_txt"
  fi

  seg_has_audio=0
  for sent_txt in "$sent_dir"/*.txt; do
    sent_text=$(cat "$sent_txt")
    # \x1c at the start = preprocess.py marked this sentence as the first
    # of a new paragraph. Strip the marker so it doesn't poison the cache
    # key (identical sentences must hash the same regardless of position)
    # and remember the flag so we insert a longer pause before playback.
    paragraph_before=0
    if [ "${sent_text:0:1}" = $'\x1c' ]; then
      paragraph_before=1
      sent_text="${sent_text:1}"
    fi
    if [ -z "${sent_text//[[:space:]]/}" ]; then
      continue
    fi
    # A: only the first sentence of the utterance gets PAD_START_MS. Later
    # sentences use pad=0 — the stream's already running so there's no
    # CoreAudio warm-up to hide, and extra pads between sentences read as
    # stutter. Different pads = different cache keys, which is fine: common
    # phrases end up with both a pad=150 "first" variant and a pad=0
    # "middle" variant and either hits correctly depending on position.
    if [ "$first_sentence" -eq 1 ]; then
      this_pad="$PAD_START_MS"
    else
      this_pad=0
    fi
    hash=$(sentence_hash "$sent_text" "$this_pad")
    cache_wav="$CACHE_DIR/$hash.wav"
    if [ -f "$cache_wav" ]; then
      touch "$cache_wav"
      hits=$((hits + 1))
    else
      REQ_BODY=$(jq -cn \
        --arg text "$sent_text" \
        --arg voice "$VOICE" \
        --arg lang "$LANG_CODE" \
        --arg sacrificial_head_word "$SACRIFICIAL_WORD" \
        --arg cache_path "$cache_wav" \
        --argjson speed "$SYNTH_SPEED" \
        --argjson eq_gain_db "$EQ_GAIN_DB" \
        --argjson eq_freq "$EQ_FREQ" \
        --argjson eq_q "$EQ_Q" \
        --argjson pad_start_ms "$this_pad" \
        --argjson tail_trim_ms "$TAIL_TRIM_MS" \
        '{text: $text, voice: $voice, speed: $speed, lang: $lang,
          eq_gain_db: $eq_gain_db, eq_freq: $eq_freq, eq_q: $eq_q,
          pad_start_ms: $pad_start_ms, tail_trim_ms: $tail_trim_ms,
          sacrificial_head_word: $sacrificial_head_word,
          cache_path: $cache_path}')
      HTTP_CODE=$(curl -sS -o "$cache_wav.tmp" -w '%{http_code}' \
        --max-time 120 \
        -X POST "$URL/speak" \
        -H "Content-Type: application/json" \
        -d "$REQ_BODY") || {
        log "ERROR: curl failed — is kokoro-server running at $URL?"
        rm -f "$cache_wav.tmp"
        exit 4
      }
      if [ "$HTTP_CODE" != "200" ]; then
        ERR=$(cat "$cache_wav.tmp" 2>/dev/null || true)
        log "HTTP $HTTP_CODE: $ERR"
        rm -f "$cache_wav.tmp"
        exit 4
      fi
      mv "$cache_wav.tmp" "$cache_wav"
      misses=$((misses + 1))
    fi

    # KOKORO_NO_PLAY: used by warm-cache.sh to fill the cache during idle
    # time. Everything up to this point still runs (split, hash, synth-on-
    # miss) so the cache ends up in the exact state a real playback would
    # leave it in. We just skip the play-stream handoff.
    if [ "${KOKORO_NO_PLAY:-0}" = "1" ]; then
      seg_has_audio=1
      continue
    fi

    # Earcon fires at segment boundaries. Send it to play-stream.py before
    # the first sentence of each non-first segment. play-stream processes
    # commands serially (blocking writes), so the earcon finishes before
    # the next PLAY starts — same synchronous semantics as the old afplay call.
    if [ "$seg_has_audio" -eq 0 ] && [ "$first_seg" -eq 0 ] && [ "$have_earcon" -eq 1 ]; then
      printf 'EARCON %s\n' "$EARCON" >&9
    fi

    # Paragraph-level pause via play-stream silence insert. With windowed-
    # RMS tail trim active (~40 ms kept), this 150 ms gap plus the trimmed
    # tail on the previous sentence ≈ paragraph-y without dragging. Gated
    # on "playing" so we never pause before the very first utterance, and
    # routed through play-stream PAUSE (not shell sleep) because the bash
    # loop is ahead of what the stream has actually played — a shell sleep
    # here would pause the producer, not the listener.
    if [ "$paragraph_before" -eq 1 ] && [ "$playing" -eq 1 ]; then
      printf 'PAUSE 150\n' >&9
    fi

    if [ "$playing" -eq 0 ]; then
      write_state play
      playing=1
    fi

    play_id=$((play_id + 1))
    printf 'PLAY %s %s\n' "$play_id" "$cache_wav" >&9
    seg_has_audio=1
    first_sentence=0
  done
  [ "$seg_has_audio" -eq 1 ] && first_seg=0
done

# Close fd 9 so play-stream.py sees EOF, drains, and exits cleanly.
if [ "$STREAM_FD_OPEN" -eq 1 ]; then
  exec 9>&-
  STREAM_FD_OPEN=0
fi
# Wait for the last audio to finish playing before returning to the caller.
if [ -n "$STREAM_PID" ]; then
  wait "$STREAM_PID" 2>/dev/null || true
  STREAM_PID=""
fi

# Prune only when we actually added bytes. On a pure cache-hit run (the
# common case for stock phrases once warm) the cache can't have grown, so
# we skip the stat-every-file scan entirely. At thousands of cached WAVs
# that's the difference between F13 feeling instant and F13 pausing.
if [ "$misses" -gt 0 ]; then
  prune_cache
fi
log "cache: hits=$hits misses=$misses"
log "done"
