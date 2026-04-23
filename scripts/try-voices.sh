#!/bin/bash
# try-voices.sh — play the same sentence in a curated shortlist of Kokoro
# voices so you can A/B listen and pick one. Each voice is announced via
# macOS `say` (fast, low-pitch, labels each clip without collision with the
# Kokoro output).
#
# Tweak SAMPLE_TEXT below if you want to listen to content closer to what
# you actually hear day-to-day.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SAMPLE_TEXT="The commit replaces scipy's resample_poly with sox tempo. Playback runs at two-x without pitch distortion. Show the dependent entries line only when there is, in fact, a dependent count greater than zero."

# Curated shortlist — mix of American/British, female/male, different timbres.
# Grades per community VOICES.md: af_heart=A, af_bella=A-, af_sarah=B+, others B.
VOICES=(
  af_heart
  af_bella
  af_nicole
  af_sarah
  am_michael
  am_adam
  am_onyx
  bm_george
)

for voice in "${VOICES[@]}"; do
  echo "--- $voice ---"
  say -r 220 "$voice"
  sleep 0.3
  KOKORO_VOICE="$voice" bash "$SCRIPT_DIR/kokoro-tts.sh" <<<"$SAMPLE_TEXT"
  sleep 0.5
done

echo ""
echo "Done. Pick one and set it as your default:"
echo "  export KOKORO_VOICE=<name>    # for shell"
echo "  # or edit hammerspoon/claudio.lua if you want it system-wide"
