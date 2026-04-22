#!/bin/bash
# warm-cache.sh — pre-synth common phrases into the Kokoro cache so the
# first time F13 trips over one of them, it's an instant cache hit
# instead of a ~400 ms synth cold start.
#
# Reads $CLAUDIO_DIR/common-phrases.txt (or $1) and feeds each non-blank,
# non-comment line through the regular pipeline with KOKORO_NO_PLAY=1.
# That means cache keys here are the *exact same* keys a real playback
# would use — phrase text, voice, synth speed, lang — so runtime hits
# line up perfectly.
#
# Safe to re-run; already-cached phrases just touch the file (LRU bump).
# Intended usage:
#   - install.sh calls this at the end of a fresh install
#   - user runs it manually after changing KOKORO_VOICE to warm the new voice
#   - could be hooked into the menu bar ("Warm cache") — not wired yet
set -euo pipefail

CLAUDIO_DIR="$HOME/.claude/claudio"
PHRASES="${1:-$CLAUDIO_DIR/common-phrases.txt}"
PREPROCESS="$CLAUDIO_DIR/preprocess.py"
TTS_SCRIPT="$CLAUDIO_DIR/kokoro-tts.sh"
SERVER_SCRIPT="$CLAUDIO_DIR/kokoro-server.sh"

if [ ! -f "$PHRASES" ]; then
  echo "warm-cache: no phrase file at $PHRASES" >&2
  exit 1
fi

# Ensure the server's up — without it every line would fail, loudly.
"$SERVER_SCRIPT" start >/dev/null

voice="${KOKORO_VOICE:-af_bella}"
cache_dir="${KOKORO_CACHE_DIR:-$CLAUDIO_DIR/cache}"
mkdir -p "$cache_dir"
before=$({ find "$cache_dir" -name '*.wav' 2>/dev/null || true; } | wc -l | tr -d ' ')

total=0
while IFS= read -r line || [ -n "$line" ]; do
  phrase="${line%%#*}"                              # drop inline comments
  phrase="${phrase#"${phrase%%[![:space:]]*}"}"     # ltrim
  phrase="${phrase%"${phrase##*[![:space:]]}"}"     # rtrim
  [ -z "$phrase" ] && continue
  total=$((total + 1))
  # Run through preprocess + kokoro-tts with playback off. We don't care
  # about stdout; we only want the cache/<hash>.wav side effect.
  printf '%s' "$phrase" \
    | /usr/bin/python3 "$PREPROCESS" \
    | KOKORO_NO_PLAY=1 "$TTS_SCRIPT" >/dev/null
done < "$PHRASES"

after=$({ find "$cache_dir" -name '*.wav' 2>/dev/null || true; } | wc -l | tr -d ' ')
new=$((after - before))
echo "warm-cache: voice=$voice phrases=$total new_cache_entries=$new total_cache=$after"
