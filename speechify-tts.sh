#!/bin/bash
# speechify-tts.sh — read text on stdin, synthesize via Speechify, play via afplay.
#
# Env vars:
#   SPEECHIFY_VOICE     voice_id (default: george)
#   SPEECHIFY_MODEL     model (default: simba-english)
#   SPEECHIFY_KEY_FILE  path to API key file (default: ~/.config/fletch/speechify-api-key)
#
# Lifecycle: this script is normally invoked by play-last.sh, which owns the
# job lock. When play-last.sh is killed, it uses pkill -P to take down this
# wrapper and its children (curl, afplay). We only manage the state file to
# communicate phase transitions to the Hammerspoon menu bar.

set -euo pipefail

VOICE="${SPEECHIFY_VOICE:-george}"
MODEL="${SPEECHIFY_MODEL:-simba-english}"
KEY_FILE="${SPEECHIFY_KEY_FILE:-$HOME/.config/fletch/speechify-api-key}"
# SSML prosody rate. "+200%" measured at ~2.5x on simba-english (non-linear).
# Other calibration points (empirical): +150% ≈ 1.64x, +300% ≈ 2.7x, +500% ≈ 4.1x.
# Accepts named values too: x-slow, slow, medium, fast, x-fast.
SPEED="${SPEECHIFY_SPEED:-+200%}"

STATE_DIR="$HOME/.claude/speechify"
STATE_FILE="$STATE_DIR/state"
AUDIO_FILE="$STATE_DIR/last.mp3"
LOG_FILE="$STATE_DIR/tts.log"

mkdir -p "$STATE_DIR"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >> "$LOG_FILE"; }

# Preserve preview line from play-last.sh's state write when we rewrite it.
current_preview() {
  grep -E '^preview=' "$STATE_FILE" 2>/dev/null | head -1 | sed 's/^preview=//'
}
write_state() {
  local state="$1"
  local preview
  preview=$(current_preview)
  printf 'state=%s\npreview=%s\nts=%s\n' "$state" "$preview" "$(date +%s)" > "$STATE_FILE"
}

if [ ! -r "$KEY_FILE" ]; then
  log "ERROR: API key not readable at $KEY_FILE"
  exit 2
fi
API_KEY=$(tr -d '[:space:]' < "$KEY_FILE")

TEXT=$(cat)
if [ -z "$TEXT" ]; then
  log "ERROR: no input text on stdin"
  exit 3
fi

# XML-escape the text for safe SSML embedding. Order matters: & first.
ESCAPED=$(printf '%s' "$TEXT" \
  | sed -e 's/&/\&amp;/g' -e 's/</\&lt;/g' -e 's/>/\&gt;/g' \
        -e 's/"/\&quot;/g' -e "s/'/\&apos;/g")
SSML="<speak><prosody rate=\"$SPEED\">$ESCAPED</prosody></speak>"

REQ_BODY=$(jq -cn \
  --arg input "$SSML" \
  --arg voice "$VOICE" \
  --arg model "$MODEL" \
  '{input: $input, voice_id: $voice, model: $model, audio_format: "mp3"}')

log "synthesize: voice=$VOICE model=$MODEL chars=${#TEXT}"
write_state synth

HTTP_CODE=$(curl -sS -o "$AUDIO_FILE" -w '%{http_code}' \
  -X POST https://api.speechify.ai/v1/audio/stream \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d "$REQ_BODY")

if [ "$HTTP_CODE" != "200" ]; then
  ERR=$(cat "$AUDIO_FILE" 2>/dev/null || true)
  log "HTTP $HTTP_CODE: $ERR"
  rm -f "$AUDIO_FILE"
  exit 4
fi

log "playing $(wc -c < "$AUDIO_FILE" | tr -d ' ') bytes"
write_state play

# Run afplay in foreground so our exit status reflects playback completion.
# play-last.sh's cleanup uses pkill -P to take us (and afplay) down on stop.
afplay "$AUDIO_FILE"
log "done"
