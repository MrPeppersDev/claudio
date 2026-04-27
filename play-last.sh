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
# Queue mode: items in QUEUE_DIR, drain coordinator's pid + atomic lock
# sit as siblings so wiping items doesn't disturb drain ownership.
QUEUE_DIR="$STATE_DIR/queue"
QUEUE_PID_FILE="$STATE_DIR/queue.pid"
QUEUE_LOCK_DIR="$STATE_DIR/queue.lock"

# Ensure Kokoro server is up. Idempotent: ~0.05s when healthy, up to ~8s on
# cold start. Called only when we're about to synth (not on toggle-stop).
ensure_server() {
  "$SERVER_SCRIPT" start >/dev/null 2>&1 || {
    log "ERROR: kokoro-server failed to start"
    exit 1
  }
}

# Usage:
#   play-last.sh                              toggle-stop any in-flight job + queue
#   play-last.sh --text-file <path>           play one selection (old behavior)
#   play-last.sh --queue-add --text-file <p>  append selection to queue; drain if idle
#   play-last.sh --drain                      internal: coordinator loop for queue mode
MODE=play
TEXT_FILE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --text-file) TEXT_FILE="${2:-}"; shift 2 ;;
    --queue-add) MODE=queue-add; shift ;;
    --drain)     MODE=drain; shift ;;
    *)           shift ;;  # ignore unknowns for forward-compat
  esac
done

mkdir -p "$STATE_DIR"
log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >> "$LOG_FILE"; }

# Menu-bar settings: let the Hammerspoon UI persist a playback speed
# and voice without users having to touch env vars or restart anything.
# Env vars win if explicitly set (CLI override path); otherwise we adopt
# what the menu wrote. Values that don't parse are silently ignored —
# kokoro-tts.sh falls back to its own default.
SPEED_FILE="$STATE_DIR/speed"
if [ -z "${KOKORO_SPEED:-}" ] && [ -r "$SPEED_FILE" ]; then
  raw_speed=$(head -1 "$SPEED_FILE" 2>/dev/null | tr -d '[:space:]')
  if [[ "$raw_speed" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    export KOKORO_SPEED="$raw_speed"
  fi
fi
VOICE_FILE="$STATE_DIR/voice"
if [ -z "${KOKORO_VOICE:-}" ] && [ -r "$VOICE_FILE" ]; then
  raw_voice=$(head -1 "$VOICE_FILE" 2>/dev/null | tr -d '[:space:]')
  # Allow blend specs (colon + comma) plus plain names. Anything with
  # shell-dangerous characters is refused outright — the value gets
  # passed to downstream tools as an env var, not eval'd, so this is
  # belt-and-braces rather than a real escape hatch, but cheap.
  if [[ "$raw_voice" =~ ^[A-Za-z0-9_:,.]+$ ]]; then
    export KOKORO_VOICE="$raw_voice"
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

# Returns 0 if the drain coordinator is running.
drain_is_alive() {
  [ -r "$QUEUE_PID_FILE" ] || return 1
  local pid
  pid=$(cat "$QUEUE_PID_FILE" 2>/dev/null || true)
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null
}

# --- Queue-add: append this selection to the queue; start drain if needed ---
if [ "$MODE" = "queue-add" ]; then
  if [ -z "$TEXT_FILE" ] || [ ! -r "$TEXT_FILE" ]; then
    log "queue-add: missing/unreadable text file: ${TEXT_FILE:-<unset>}"
    exit 1
  fi
  mkdir -p "$QUEUE_DIR"
  # mktemp gives a collision-safe name; mtime is what we sort on for
  # playback order, so two items added the same second still stay distinct.
  dest=$(mktemp "$QUEUE_DIR/item.XXXXXXXX")
  mv "$TEXT_FILE" "$dest" || { log "queue-add: move failed"; exit 1; }
  queued_count=$(ls -1 "$QUEUE_DIR" 2>/dev/null | wc -l | tr -d ' ')
  log "queue-add: enqueued $(basename "$dest") (queue=$queued_count)"
  # Advisory check — the atomic lock inside --drain is what actually
  # prevents two drainers, so races here are benign.
  if drain_is_alive; then
    exit 0
  fi
  log "queue-add: starting drain"
  nohup "$0" --drain >/dev/null 2>&1 &
  disown 2>/dev/null || true
  exit 0
fi

# --- Drain: coordinator loop for queued items ---
if [ "$MODE" = "drain" ]; then
  # Atomic start — mkdir is the lock primitive. A stale lock dir from a
  # hard-killed drain would otherwise block all future drains, so check the
  # pid inside it and steal the lock if that process is gone.
  if ! mkdir "$QUEUE_LOCK_DIR" 2>/dev/null; then
    old_drain=""
    [ -r "$QUEUE_PID_FILE" ] && old_drain=$(cat "$QUEUE_PID_FILE" 2>/dev/null || true)
    if [ -n "$old_drain" ] && kill -0 "$old_drain" 2>/dev/null; then
      log "drain: another drain already running (pid=$old_drain), exiting"
      exit 0
    fi
    log "drain: clearing stale lock (old pid=$old_drain)"
    rm -rf "$QUEUE_LOCK_DIR"
    rm -f "$QUEUE_PID_FILE"
    mkdir "$QUEUE_LOCK_DIR" || { log "drain: lock acquire failed"; exit 1; }
  fi
  echo $$ > "$QUEUE_PID_FILE"
  log "drain: started pid=$$"

  current_child=""
  drain_cleanup() {
    # Kill whatever child is still running (if we got SIGTERMed mid-item)
    if [ -n "$current_child" ] && kill -0 "$current_child" 2>/dev/null; then
      kill -TERM "$current_child" 2>/dev/null || true
    fi
    rm -rf "$QUEUE_DIR" "$QUEUE_LOCK_DIR"
    rm -f "$QUEUE_PID_FILE"
    write_state idle
    log "drain: cleanup done"
  }
  trap drain_cleanup EXIT INT TERM

  while :; do
    # Oldest first. `ls -tr` sorts by mtime ascending on both BSD + GNU ls.
    next=$(ls -tr "$QUEUE_DIR" 2>/dev/null | head -1 || true)
    if [ -z "$next" ]; then
      log "drain: queue empty"
      break
    fi
    next_path="$QUEUE_DIR/$next"
    log "drain: playing $next"
    # Run as a background child so we can capture its pid for our trap.
    # `wait` returns the child's exit status.
    "$0" --text-file "$next_path" &
    current_child=$!
    set +e
    wait "$current_child"
    ec=$?
    set -e
    current_child=""
    # 143 = SIGTERM (toggle-stop killed us). Any nonzero exit → stop draining;
    # the user either asked to stop or the play path errored and further
    # items probably won't fare better.
    if [ "$ec" -ne 0 ]; then
      log "drain: child exited $ec — stopping"
      break
    fi
  done
  exit 0
fi

# --- Toggle: stop any in-flight job or queue ---
# Queue drain first: a running drain would otherwise spawn the next item
# seconds after we killed the current one, defeating the stop. Drain's own
# trap sweeps its child + state; the rm lines are belt-and-braces in case
# the drain got SIGKILLed (trap wouldn't fire) or died uncleanly earlier.
if [ -z "$TEXT_FILE" ] && [ "$MODE" = "play" ] && drain_is_alive; then
  drain_pid=$(cat "$QUEUE_PID_FILE" 2>/dev/null || true)
  kill -TERM "$drain_pid" 2>/dev/null || true
  sleep 0.2
  if [ -n "$drain_pid" ] && kill -0 "$drain_pid" 2>/dev/null; then
    kill -KILL "$drain_pid" 2>/dev/null || true
  fi
  rm -rf "$QUEUE_DIR" "$QUEUE_LOCK_DIR"
  rm -f "$QUEUE_PID_FILE"
  log "stopped drain pid=$drain_pid"
  # If there's no single-item lock left to handle, we're done; otherwise
  # fall through so the LOCK_FILE block below kills the remaining playback.
  if [ ! -f "$LOCK_FILE" ]; then
    write_state idle
    exit 0
  fi
fi

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
    # 100ms used to be plenty, but cooperative shutdown in play-stream.py joins
    # the player thread with a 500ms timeout, plus kokoro-tts.sh's cleanup
    # trap (FIFO unlink, forwarder kill) takes a few tens of ms. 600ms covers
    # both with margin so SIGKILL is rarely needed; user-perceived stop is
    # already instant because play-stream.py's SIGTERM handler aborts the
    # OutputStream on the first signal.
    sleep 0.6
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
