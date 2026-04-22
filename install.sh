#!/bin/bash
# install.sh — bootstrap Claudio (Kokoro TTS for Claude Code) on a new machine.
#
# Assumes this repo is checked out at ~/.claude/claudio. If you cloned it
# elsewhere, move or symlink it there before running.
#
# What this does:
#   1. Verifies ~/.claude/claudio is the repo path (scripts hardcode it).
#   2. Symlinks hammerspoon/claudio.lua  → ~/.hammerspoon/claudio.lua
#      and  claudetop.d/claudio-session-map → ~/.claude/claudetop.d/…
#   3. Creates the Python 3.12 venv and installs kokoro dependencies.
#   4. Downloads the Kokoro model weights + voices (one-time, ~340 MB).
#   5. Prints next steps (Hammerspoon reload).

set -euo pipefail

REPO="$HOME/.claude/claudio"
if [ ! -d "$REPO/.git" ]; then
  echo "ERROR: expected repo at $REPO — got: $(ls -ld "$REPO" 2>&1)"
  exit 1
fi

# --- 1. Symlinks for integrations ---
mkdir -p "$HOME/.hammerspoon" "$HOME/.claude/claudetop.d"

link() {
  local src="$1" dest="$2"
  if [ -e "$dest" ] && [ ! -L "$dest" ]; then
    echo "WARN: $dest exists and is not a symlink — leaving alone"
    return
  fi
  ln -sfn "$src" "$dest"
  echo "linked $dest -> $src"
}
link "$REPO/hammerspoon/claudio.lua"             "$HOME/.hammerspoon/claudio.lua"
link "$REPO/claudetop.d/claudio-session-map"     "$HOME/.claude/claudetop.d/claudio-session-map"

# --- 2. Python venv + Kokoro deps ---
PY="/opt/homebrew/bin/python3.12"
if [ ! -x "$PY" ]; then
  echo "ERROR: python3.12 not found at $PY — install via: brew install python@3.12"
  exit 1
fi
if [ ! -d "$REPO/kokoro/venv" ]; then
  echo "creating venv..."
  "$PY" -m venv "$REPO/kokoro/venv"
fi
echo "installing deps into venv..."
"$REPO/kokoro/venv/bin/pip" install --upgrade pip >/dev/null
"$REPO/kokoro/venv/bin/pip" install -r "$REPO/kokoro/requirements.txt"

# --- 3. Model weights ---
bash "$REPO/kokoro/download-model.sh"

# --- 4. Next steps ---
echo ""
echo "Install complete."
echo ""
echo "Next steps:"
echo "  1. Reload Hammerspoon:  hs -c 'hs.reload()'"
echo "     (or click the Hammerspoon icon → Reload Config)"
echo "  2. Ensure your init.lua contains:"
echo "       local claudio = require(\"claudio\"); claudio.start()"
echo "       hs.hotkey.bind({}, \"F13\", function() claudio.toggle() end)"
echo "  3. Hit F13 once to trigger the first synth (cold start ~8s while the"
echo "     Kokoro server boots; subsequent synth is ~1s)."
