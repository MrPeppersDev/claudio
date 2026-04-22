#!/bin/bash
# kokoro-tts.sh — local Kokoro-82M TTS via our own Python HTTP server.
#
# Reads text on stdin, synthesizes locally (no external network), plays via
# afplay. The server is managed by kokoro-server.sh, which play-last.sh
# starts lazily on the first F13 after a reboot.
#
# Env vars:
#   KOKORO_VOICE       voice name (default: af_bella)
#   KOKORO_SPEED       target effective speed (default: 2.0)
#   KOKORO_SYNTH_CAP   max synth speed before afplay picks up the rest
#                      (default: 1.5; see split-speed note below)
#   KOKORO_URL         server base URL (default: http://127.0.0.1:8880)
#   KOKORO_LANG        phoneme lang (auto: bm_/bf_ → en-gb, else en-us)
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

STATE_DIR="$HOME/.claude/claudio"
STATE_FILE="$STATE_DIR/state"
AUDIO_FILE="$STATE_DIR/last.wav"
LOG_FILE="$STATE_DIR/tts.log"

mkdir -p "$STATE_DIR"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >> "$LOG_FILE"; }

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

REQ_BODY=$(jq -cn \
  --arg text "$TEXT" \
  --arg voice "$VOICE" \
  --arg lang "$LANG_CODE" \
  --argjson speed "$SYNTH_SPEED" \
  '{text: $text, voice: $voice, speed: $speed, lang: $lang}')

log "kokoro synthesize: voice=$VOICE target=${SPEED}x synth=${SYNTH_SPEED}x playback=${PLAYBACK_RATE}x lang=$LANG_CODE chars=${#TEXT}"
write_state synth

HTTP_CODE=$(curl -sS -o "$AUDIO_FILE" -w '%{http_code}' \
  --max-time 120 \
  -X POST "$URL/speak" \
  -H "Content-Type: application/json" \
  -d "$REQ_BODY") || {
  log "ERROR: curl failed — is kokoro-server running at $URL?"
  rm -f "$AUDIO_FILE"
  exit 4
}

if [ "$HTTP_CODE" != "200" ]; then
  ERR=$(cat "$AUDIO_FILE" 2>/dev/null || true)
  log "HTTP $HTTP_CODE: $ERR"
  rm -f "$AUDIO_FILE"
  exit 4
fi

log "playing $(wc -c < "$AUDIO_FILE" | tr -d ' ') bytes"
write_state play

afplay -q 1 -r "$PLAYBACK_RATE" "$AUDIO_FILE"
log "done"
