#!/bin/bash
# play-last.sh — synthesize and play a file of text, or toggle-stop.
#
# Behavior:
#   * If a job is already running, kill its process tree (synth or playback)
#     and exit. This is the "toggle stop" path.
#   * Otherwise: read text from --text-file, preprocess it, and hand off to
#     kokoro-tts.sh for synth + playback.
#
# Scope: selection-only. The caller (hammerspoon/claudio.lua) captures the
# user's highlighted text and writes it to a temp file. We never scan
# transcripts or produce speech from anything other than what was passed in.
#
# Lock model: play-last.sh owns job.pid for its entire run — synthesis *and*
# playback. A second invocation while the first is mid-synth finds the lock
# and kills the whole tree, not just afplay.
#
# State file: writes state={idle,synth,play} + optional text preview, consumed
# by the Hammerspoon menu bar indicator.

set -euo pipefail

# SCRIPT_DIR is where *this* script lives — the Claudio repo checkout. Used
# to find sibling scripts (preprocess.py, kokoro-tts.sh, kokoro-server.sh)
# so the repo can live anywhere, not just ~/.claude/claudio.
# STATE_DIR is where runtime files (lock, state, log, cache) live. It
# defaults to ~/.claude/claudio and is user-overridable via
# CLAUDIO_STATE_DIR so the repo can be checked out elsewhere (e.g. a test
# worktree) without state leaking into it.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_DIR="${CLAUDIO_STATE_DIR:-$HOME/.claude/claudio}"
LOCK_FILE="$STATE_DIR/job.pid"
STATE_FILE="$STATE_DIR/state"
LOG_FILE="$STATE_DIR/play.log"
PREPROCESS="$SCRIPT_DIR/preprocess.py"
TTS_SCRIPT="$SCRIPT_DIR/kokoro-tts.sh"
SERVER_SCRIPT="$SCRIPT_DIR/kokoro-server.sh"

# Ensure Kokoro server is up. Idempotent: ~0.05s when healthy, up to ~8s on
# cold start. Called only when we're about to synth (not on toggle-stop).
ensure_server() {
  "$SERVER_SCRIPT" start >/dev/null 2>&1 || {
    log "ERROR: kokoro-server failed to start"
    exit 1
  }
}

# Usage: play-last.sh --text-file <path>
#   --text-file   read playback text from this file
# No-args form is reserved for the toggle-stop path (kills an in-flight job).
TEXT_FILE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --text-file) TEXT_FILE="${2:-}"; shift 2 ;;
    *)           shift ;;  # ignore unknowns for forward-compat
  esac
done

mkdir -p "$STATE_DIR"
log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >> "$LOG_FILE"; }

# Menu-bar settings: let the Hammerspoon UI persist a playback speed
# without users having to touch env vars or restart anything. Env var
# wins if explicitly set (CLI override path); otherwise we adopt what
# the menu wrote. Values that don't parse as a plausible number are
# silently ignored — kokoro-tts.sh falls back to its own default.
SPEED_FILE="$STATE_DIR/speed"
if [ -z "${KOKORO_SPEED:-}" ] && [ -r "$SPEED_FILE" ]; then
  raw_speed=$(head -1 "$SPEED_FILE" 2>/dev/null | tr -d '[:space:]')
  if [[ "$raw_speed" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    export KOKORO_SPEED="$raw_speed"
  fi
fi
write_state() {
  # state=<one of: idle|synth|play>   preview=<truncated text, one line>
  local state="$1" preview="${2:-}"
  printf 'state=%s\npreview=%s\nts=%s\n' "$state" "$preview" "$(date +%s)" > "$STATE_FILE"
}

# Returns 0 if $1 is a live play-last.sh process. Guards against PID recycling:
# between one invocation writing its PID to the lock file and the next reading
# it, the kernel may have reassigned that PID to an unrelated program. `ps -o
# command=` gives us the full argv; we look for our own script name.
is_our_process() {
  local pid="$1" cmd
  cmd=$(ps -o command= -p "$pid" 2>/dev/null || true)
  [ -n "$cmd" ] && [[ "$cmd" == *play-last.sh* ]]
}

# Recursively collect descendants of $1 (inclusive of $1) into $DESCENDANTS.
# Needed because SIGTERM on a bash parent does not reliably propagate to
# grandchildren like afplay, and pkill -P only walks one level.
collect_descendants() {
  local parent="$1" kid
  DESCENDANTS="$DESCENDANTS $parent"
  for kid in $(pgrep -P "$parent" 2>/dev/null || true); do
    collect_descendants "$kid"
  done
}

# --- Toggle: stop any in-flight job ---
if [ -f "$LOCK_FILE" ]; then
  OLD_PID=$(cat "$LOCK_FILE" 2>/dev/null || true)
  if [ -n "${OLD_PID:-}" ] && kill -0 "$OLD_PID" 2>/dev/null && is_our_process "$OLD_PID"; then
    # Walk the whole descendant tree first so we can target every level
    # (bash wrapper → curl/python → afplay) without pkill -9'ing every
    # afplay on the system.
    DESCENDANTS=""
    collect_descendants "$OLD_PID"
    # shellcheck disable=SC2086 # word-split on purpose
    kill -TERM $DESCENDANTS 2>/dev/null || true
    sleep 0.1
    # shellcheck disable=SC2086
    kill -KILL $DESCENDANTS 2>/dev/null || true
    rm -f "$LOCK_FILE"
    write_state idle
    log "stopped pid=$OLD_PID tree=$(printf '%s' "$DESCENDANTS" | wc -w | tr -d ' ')"
    exit 0
  fi
  # Stale lock: PID dead or reassigned to an unrelated process. Clear it and
  # fall through to start a new job rather than exiting.
  [ -n "${OLD_PID:-}" ] && log "stale lock pid=$OLD_PID (not ours) — clearing"
  rm -f "$LOCK_FILE"
fi

# --- Acquire lock for this run ---
echo $$ > "$LOCK_FILE"
cleanup() {
  # Kill any still-running children and clear state.
  pkill -TERM -P $$ 2>/dev/null || true
  rm -f "$LOCK_FILE"
  write_state idle
}
trap cleanup EXIT INT TERM

# --- Play path: --text-file is required ---
if [ -z "$TEXT_FILE" ]; then
  log "ERROR: no --text-file given; nothing to play"
  osascript -e 'display notification "No text to play — highlight text and press F13" with title "Claudio"'
  exit 1
fi

if [ ! -r "$TEXT_FILE" ]; then
  log "ERROR: text file not readable: $TEXT_FILE"
  exit 1
fi
TEXT=$(cat "$TEXT_FILE")
# Unlink immediately — claudio.lua writes selections here, and selection
# text can contain secrets or PII. We've already copied it into $TEXT; the
# file has no further purpose and shouldn't persist between runs.
rm -f "$TEXT_FILE"
if [ -z "${TEXT:-}" ]; then
  log "ERROR: text file is empty: $TEXT_FILE"
  exit 1
fi
# PREVIEW goes to the state file (used by the menu bar) but NOT to the
# persistent log — user content can include tokens, PII, or chat
# fragments that shouldn't accumulate on disk indefinitely. The log
# line records only the length.
PREVIEW=$(printf '%s' "$TEXT" | head -1 | cut -c1-80)
log "playing ${#TEXT} chars from selection"
write_state synth "$PREVIEW"
ensure_server
printf '%s' "$TEXT" | /usr/bin/python3 "$PREPROCESS" | "$TTS_SCRIPT"
