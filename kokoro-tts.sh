#!/bin/bash
# kokoro-tts.sh — local Kokoro-82M TTS via our own Python HTTP server.
#
# Reads text on stdin, synthesizes locally (no external network), plays via
# afplay. The server is managed by kokoro-server.sh, which play-last.sh
# starts lazily on the first F13 after a reboot.
#
# Env vars:
#   KOKORO_VOICE       voice name, or a weighted blend (default: af_bella).
#                      Blend syntax: "af_bella:70,am_michael:30" — weights
#                      are normalized, so "1,1" is a 50/50 mix.
#   KOKORO_SPEED       target effective speed (default: 2.0)
#   KOKORO_SYNTH_CAP   max synth speed before afplay picks up the rest
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
#                      first phoneme isn't smushed and so CoreAudio ramp-up
#                      on the first afplay doesn't eat the opening word
#                      (default: 150; 0 disables)
#   KOKORO_TAIL_TRIM_MS trailing silence kept after the last audible frame.
#                      Uses windowed-RMS energy detection with a 60ms dwell
#                      so unvoiced closing consonants (t/p/k) survive intact.
#                      (default: 40; 0 disables — keep Kokoro's native tail)
#
# Owned by play-last.sh via job.pid; writes state={synth,play} to the state
# file; afplay runs in foreground so the parent's pkill -P tears everything
# down on stop.

set -euo pipefail

VOICE="${KOKORO_VOICE:-af_bella}"
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
mkdir -p "$CACHE_DIR"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >> "$LOG_FILE"; }

# Hash key for a sentence: voice|synth_speed|lang|eq_*|pad_start_ms|text.
# Any change in those produces different audio, so they all go into the key.
# Null bytes between fields prevent "ab|c" colliding with "a|bc". Every
# synthesis-affecting knob is mixed in, so toggling any env var (EQ, pad)
# auto-invalidates stale cache instead of silently mixing pre/post audio.
sentence_hash() {
  printf '%s\0%s\0%s\0%s\0%s\0%s\0%s\0%s\0%s' \
    "$VOICE" "$SYNTH_SPEED" "$LANG_CODE" \
    "$EQ_GAIN_DB" "$EQ_FREQ" "$EQ_Q" "$PAD_START_MS" "$TAIL_TRIM_MS" "$1" \
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

# Split the target speed between the model and afplay. Pushing Kokoro to its
# 2.0x cap forces the model to compress phonemes and syllables muddle; a
# gentler synth rate keeps prosody clean, and afplay's pitch-preserving phase
# vocoder (-q 1) covers the rest without blurring consonants.
#   effective_speed = SYNTH_SPEED * PLAYBACK_RATE = SPEED
# Lower KOKORO_SYNTH_CAP → cleaner phonemes, more afplay stretch.
# Higher KOKORO_SYNTH_CAP → less afplay work, more model compression.
SYNTH_CAP="${KOKORO_SYNTH_CAP:-1.5}"
SYNTH_SPEED=$(awk -v s="$SPEED" -v c="$SYNTH_CAP" 'BEGIN{print (s>c?c:s)}')
PLAYBACK_RATE=$(awk -v s="$SPEED" -v ss="$SYNTH_SPEED" 'BEGIN{print s/ss}')

# Two-level split:
#   outer \x1e = earcon sentinel (from preprocess.py; ran over code blocks)
#   inner \x1d = sentence sentinel (cache granularity)
# A single-paragraph message with no code produces one segment with N
# sentences; a short reply with no code and one sentence produces a single
# segment/single sentence and behaves like the original single-shot path.
WORK_DIR=$(mktemp -d "$STATE_DIR/segments.XXXXXX")
prev_pid=""
cleanup() {
  # Kill dangling background afplay on any exit path (error, signal, done).
  # Parent (play-last.sh) also reaps us via pkill -P for explicit stops;
  # this covers the synth-failure path where set -e trips.
  [ -n "$prev_pid" ] && kill "$prev_pid" 2>/dev/null || true
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

# Interleaved synth + playback. Each iteration resolves its sentence's WAV
# (cache hit = instant; miss = one curl to the local server) then hands off
# to afplay in the background. The next iteration's synth runs concurrently
# with the current afplay, so server round-trip cost is masked behind the
# previous sentence's audible playback. Cold first sentence still pays the
# full synth+start cost; everything after is pipelined.
hits=0
misses=0
playing=0
first_seg=1
for seg_txt in "$WORK_DIR"/seg-*.txt; do
  seg_id=$(basename "$seg_txt" .txt)
  sent_dir="$WORK_DIR/$seg_id.sent"
  mkdir -p "$sent_dir"
  awk -v RS=$'\x1d' -v dir="$sent_dir" '
    { out = sprintf("%s/%04d.txt", dir, NR); printf "%s", $0 > out; close(out) }
  ' "$seg_txt"

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
    hash=$(sentence_hash "$sent_text")
    cache_wav="$CACHE_DIR/$hash.wav"
    if [ -f "$cache_wav" ]; then
      touch "$cache_wav"
      hits=$((hits + 1))
    else
      REQ_BODY=$(jq -cn \
        --arg text "$sent_text" \
        --arg voice "$VOICE" \
        --arg lang "$LANG_CODE" \
        --argjson speed "$SYNTH_SPEED" \
        --argjson eq_gain_db "$EQ_GAIN_DB" \
        --argjson eq_freq "$EQ_FREQ" \
        --argjson eq_q "$EQ_Q" \
        --argjson pad_start_ms "$PAD_START_MS" \
        --argjson tail_trim_ms "$TAIL_TRIM_MS" \
        '{text: $text, voice: $voice, speed: $speed, lang: $lang,
          eq_gain_db: $eq_gain_db, eq_freq: $eq_freq, eq_q: $eq_q,
          pad_start_ms: $pad_start_ms, tail_trim_ms: $tail_trim_ms}')
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
    # leave it in. We just skip the afplay handoff.
    if [ "${KOKORO_NO_PLAY:-0}" = "1" ]; then
      seg_has_audio=1
      continue
    fi

    # Gate on previous afplay before starting this one. The waiting happens
    # *after* we resolved the current WAV, so synth cost overlaps with prior
    # playback rather than adding to the gap between sentences.
    if [ -n "$prev_pid" ]; then
      wait "$prev_pid" 2>/dev/null || true
      prev_pid=""
    fi

    # Earcon fires synchronously at segment boundaries. Keep it blocking so
    # the listener hears the "clunk" distinctly before the next sentence.
    if [ "$seg_has_audio" -eq 0 ] && [ "$first_seg" -eq 0 ] && [ "$have_earcon" -eq 1 ]; then
      afplay "$EARCON"
    fi

    # Paragraph-level pause. With windowed-RMS tail trim active (~40ms
    # kept), 150ms here + 40ms trimmed tail + 150ms leading pad on the
    # next sentence ≈ 340ms of break — paragraph-y without dragging.
    # Gated on "playing" so we never pause before the very first utterance.
    if [ "$paragraph_before" -eq 1 ] && [ "$playing" -eq 1 ]; then
      sleep 0.15
    fi

    if [ "$playing" -eq 0 ]; then
      write_state play
      playing=1
    fi

    afplay -q 1 -r "$PLAYBACK_RATE" "$cache_wav" &
    prev_pid=$!
    seg_has_audio=1
  done
  [ "$seg_has_audio" -eq 1 ] && first_seg=0
done

if [ -n "$prev_pid" ]; then
  wait "$prev_pid" 2>/dev/null || true
  prev_pid=""
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
