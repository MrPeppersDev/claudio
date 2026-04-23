#!/bin/bash
# install.sh — bootstrap Claudio (Kokoro TTS for Claude Code) on a new machine.
#
# The repo can be cloned anywhere; this script derives its location from its
# own path. State (cache, logs, pid files) defaults to ~/.claude/claudio and
# can be moved by exporting CLAUDIO_STATE_DIR before running the scripts.
#
# What this does:
#   1. Symlinks hammerspoon/claudio.lua  → ~/.hammerspoon/claudio.lua
#   2. Creates a Python >= 3.10 venv and installs kokoro dependencies.
#      (Prefers 3.12. Override with CLAUDIO_PYTHON=/path/to/python if needed.)
#   3. Downloads the Kokoro model weights + voices (one-time, ~340 MB).
#   4. Warms the cache with common phrases so the first F13 isn't cold.
#   5. Prints next steps (Hammerspoon reload).

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- 1. Symlinks for integrations ---
mkdir -p "$HOME/.hammerspoon"

# Clean up a legacy symlink from earlier versions that wired a claudetop
# statusline plugin for iTerm transcript mapping. That whole code path is
# gone — selection-only playback doesn't need it. Dangling symlinks are
# skipped by claudetop but still visible in `ls`, so remove it properly.
LEGACY_CLAUDETOP_LINK="$HOME/.claude/claudetop.d/claudio-session-map"
if [ -L "$LEGACY_CLAUDETOP_LINK" ]; then
  rm -f "$LEGACY_CLAUDETOP_LINK"
  echo "removed legacy symlink: $LEGACY_CLAUDETOP_LINK"
fi

link() {
  local src="$1" dest="$2"

  # Existing symlink. Two sub-cases: already points where we want (no-op),
  # or points somewhere stale (fine to overwrite — it's cheap and by-design
  # reversible since we haven't touched the real file).
  if [ -L "$dest" ]; then
    local current
    current=$(readlink "$dest")
    if [ "$current" = "$src" ]; then
      echo "ok $dest -> $src (already linked)"
      return 0
    fi
    ln -sfn "$src" "$dest"
    echo "relinked $dest: $current -> $src"
    return 0
  fi

  # Existing regular file. The previous behavior printed WARN and silently
  # skipped, so installs appeared to succeed while Hammerspoon kept loading
  # a stale copy. Back up and link, so the user keeps their file but gets
  # a working install.
  if [ -f "$dest" ]; then
    local bak="$dest.bak.$(date +%s)"
    mv "$dest" "$bak"
    ln -s "$src" "$dest"
    echo "linked $dest -> $src (old file backed up to $bak)"
    return 0
  fi

  # Anything else at the path — directory, socket, FIFO — is surprising and
  # we shouldn't auto-clobber. Abort with enough context for the user to
  # decide what to do.
  if [ -e "$dest" ]; then
    echo "ERROR: $dest exists and is neither a file nor a symlink:"
    ls -ld "$dest"
    echo "       resolve manually, then re-run install.sh"
    exit 1
  fi

  # Fresh install — nothing at the destination.
  ln -s "$src" "$dest"
  echo "linked $dest -> $src"
}
link "$REPO/hammerspoon/claudio.lua"             "$HOME/.hammerspoon/claudio.lua"

# --- 2. Python venv + Kokoro deps ---
# Find a Python >= 3.10 — kokoro-onnx supports 3.10+. We prefer 3.12 because
# that's what the maintainer tests against, but accept anything 3.10+ so
# Intel Macs, pyenv/asdf users, and non-Homebrew installs work out of the box.
find_python() {
  # 1. Explicit override — CLAUDIO_PYTHON=/path/to/python
  if [ -n "${CLAUDIO_PYTHON:-}" ] && [ -x "$CLAUDIO_PYTHON" ]; then
    echo "$CLAUDIO_PYTHON"; return 0
  fi
  # 2. Prefer 3.12, then 3.13, 3.11, 3.10, finally generic python3.
  local candidate
  for candidate in python3.12 python3.13 python3.11 python3.10 python3; do
    local resolved
    resolved=$(command -v "$candidate" 2>/dev/null || true)
    [ -n "$resolved" ] || continue
    # Verify the interpreter is actually >= 3.10.
    if "$resolved" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; then
      echo "$resolved"; return 0
    fi
  done
  return 1
}

PY=$(find_python) || {
  cat >&2 <<'MSG'
ERROR: no Python >= 3.10 found on PATH.

Install one of:
  - macOS (Homebrew):  brew install python@3.12
  - pyenv:             pyenv install 3.12 && pyenv shell 3.12
  - asdf:              asdf install python 3.12.0

Or set CLAUDIO_PYTHON=/absolute/path/to/python3 and re-run.
MSG
  exit 1
}
echo "using $PY ($("$PY" --version 2>&1))"

if [ ! -d "$REPO/kokoro/venv" ]; then
  echo "creating venv..."
  "$PY" -m venv "$REPO/kokoro/venv"
fi
echo "installing deps into venv..."
"$REPO/kokoro/venv/bin/pip" install --upgrade pip >/dev/null
"$REPO/kokoro/venv/bin/pip" install -r "$REPO/kokoro/requirements.txt"

# preprocess.py runs under /usr/bin/python3 (not the venv above) and uses
# optional user-site deps. Each has a graceful fallback so failures are non-fatal.
/usr/bin/python3 -m pip install --user --quiet pysbd \
  || echo "WARN: pysbd install failed — preprocess.py will use regex fallback"
/usr/bin/python3 -m pip install --user --quiet num2words inflect \
  || echo "WARN: num2words/inflect install failed — number spell-out will be skipped (non-fatal)"
/usr/bin/python3 -m pip install --user --quiet trafilatura lxml_html_clean \
  || echo "WARN: trafilatura install failed — HTML article extraction will be skipped (non-fatal)"

# --- 3. Model weights ---
bash "$REPO/kokoro/download-model.sh"

# --- 4. Pre-warm cache with common phrases ---
# Boots the server (lazy start), synthesizes ~40 stock phrases into the
# cache. ~15s one-time cost; the payoff is that "Let me check that.",
# "Done.", etc. are instant cache hits the first time they come up.
echo "warming phrase cache..."
bash "$REPO/warm-cache.sh" || echo "WARN: cache warm failed (non-fatal)"

# --- 5. Next steps ---
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
