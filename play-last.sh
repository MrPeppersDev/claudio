#!/bin/bash
# play-last.sh — toggle playback of the last assistant message.
#
# Behavior:
#   * If a job is already running, kill its process tree (synth or playback)
#     and exit. This is the "toggle stop" path.
#   * Otherwise: find the most recently modified transcript under
#     ~/.claude/projects/**, extract the last assistant message that contains
#     text content, pass it through preprocess.py, and pipe to kokoro-tts.sh.
#
# Lock model: play-last.sh owns job.pid for its entire run — synthesis *and*
# playback. A second invocation while the first is mid-synth finds the lock
# and kills the whole tree, not just afplay.
#
# State file: writes state={idle,synth,play} + optional text preview, consumed
# by the Hammerspoon menu bar indicator.

set -euo pipefail

STATE_DIR="$HOME/.claude/claudio"
LOCK_FILE="$STATE_DIR/job.pid"
STATE_FILE="$STATE_DIR/state"
LOG_FILE="$STATE_DIR/play.log"
SESSION_MAP_DIR="$STATE_DIR/sessions"
PROJECTS_DIR="$HOME/.claude/projects"
PREPROCESS="$STATE_DIR/preprocess.py"
TTS_SCRIPT="$STATE_DIR/kokoro-tts.sh"
SERVER_SCRIPT="$STATE_DIR/kokoro-server.sh"

# Ensure Kokoro server is up. Idempotent: ~0.05s when healthy, up to ~8s on
# cold start. Called only when we're about to synth (not on toggle-stop).
ensure_server() {
  "$SERVER_SCRIPT" start >/dev/null 2>&1 || {
    log "ERROR: kokoro-server failed to start"
    exit 1
  }
}

# Usage: play-last.sh [--session <id>] [--text-file <path>]
#   --session     iTerm session id; used to scope transcript lookup
#   --text-file   read playback text from this file and skip transcript lookup
# With no args, plays last message from the globally-most-recent transcript.
REQUESTED_SID=""
TEXT_FILE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --session)   REQUESTED_SID="${2:-}"; shift 2 ;;
    --text-file) TEXT_FILE="${2:-}"; shift 2 ;;
    *)           shift ;;  # ignore unknowns for forward-compat
  esac
done

mkdir -p "$STATE_DIR"
log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >> "$LOG_FILE"; }
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

# --- Text-file path: skip transcript lookup entirely ---
if [ -n "$TEXT_FILE" ]; then
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
  exit 0
fi

# --- Resolve transcript for the requested iTerm session, if any ---
LATEST=""
if [ -n "$REQUESTED_SID" ]; then
  # Try both raw id and UUID-after-colon (claudetop plugin writes both keys).
  for candidate in "$REQUESTED_SID" "${REQUESTED_SID##*:}"; do
    safe=$(printf '%s' "$candidate" | tr ':/ ' '___')
    map_file="$SESSION_MAP_DIR/$safe"
    if [ -f "$map_file" ]; then
      mapped=$(grep -E '^transcript_path=' "$map_file" | head -1 | sed 's/^transcript_path=//')
      if [ -n "$mapped" ] && [ -r "$mapped" ]; then
        LATEST="$mapped"
        log "session-mapped: sid=$REQUESTED_SID -> $LATEST"
        break
      fi
    fi
  done
fi

# --- Fallback: most recently modified transcript anywhere ---
# Subshell disables pipefail because `head -1` closing early makes sort exit
# with SIGPIPE, which pipefail would promote to a script-killing error.
if [ -z "$LATEST" ]; then
  log "fallback: no session mapping (sid='${REQUESTED_SID:-none}'), using global latest"
  LATEST=$(set +o pipefail; \
    find "$PROJECTS_DIR" -type f -name '*.jsonl' -print0 2>/dev/null \
    | xargs -0 stat -f '%m %N' 2>/dev/null \
    | sort -rn | head -1 | cut -d' ' -f2-)
fi

if [ -z "${LATEST:-}" ] || [ ! -r "$LATEST" ]; then
  log "ERROR: no transcript found under $PROJECTS_DIR"
  osascript -e 'display notification "No Claude transcript found" with title "Claudio"'
  exit 1
fi

log "transcript: $LATEST"

# --- Extract last assistant message that has text content ---
# Assistant messages can contain mixed content (thinking, text, tool_use).
# Take the last JSONL line where .type=="assistant" AND at least one content
# block has type=="text", then join all text blocks with blank lines.
TEXT=$(jq -rs '
  [.[] | select(.type=="assistant" and (.message.content | any(.type=="text")))]
  | last
  | [.message.content[] | select(.type=="text") | .text]
  | join("\n\n")
' "$LATEST" 2>/dev/null || true)

if [ -z "${TEXT:-}" ] || [ "$TEXT" = "null" ]; then
  log "ERROR: no assistant text found in $LATEST"
  osascript -e 'display notification "No assistant text to play" with title "Claudio"'
  exit 1
fi

# Preview = first 80 chars of first line, for the menu bar dropdown.
# Kept out of the persistent log for the same reason as the --text-file
# branch above: assistant text can contain secrets, PII, or chat fragments
# we don't want accumulating on disk. The log records only the length.
PREVIEW=$(printf '%s' "$TEXT" | head -1 | cut -c1-80)
log "playing ${#TEXT} chars"

write_state synth "$PREVIEW"
ensure_server
printf '%s' "$TEXT" | /usr/bin/python3 "$PREPROCESS" | "$TTS_SCRIPT"
