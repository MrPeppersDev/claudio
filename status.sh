#!/bin/bash
# status.sh — paste-friendly diagnostic snapshot.
#
# Prints the state of the server, the playback job, the cache, and a tail
# of both log files. No arguments, no colors — output is meant to be
# pasted directly into issues or chat when something isn't behaving.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_DIR="${CLAUDIO_STATE_DIR:-$HOME/.claude/claudio}"
CACHE_DIR="${KOKORO_CACHE_DIR:-$STATE_DIR/cache}"
PLAY_LOG="$STATE_DIR/play.log"
SERVER_LOG="$STATE_DIR/kokoro-server.log"
STATE_FILE="$STATE_DIR/state"
LOCK_FILE="$STATE_DIR/job.pid"

section() { printf '\n== %s ==\n' "$1"; }

section "claudio status  —  $(date '+%Y-%m-%d %H:%M:%S')"
printf 'repo:  %s\n' "$SCRIPT_DIR"
printf 'state: %s\n' "$STATE_DIR"
printf 'cache: %s\n' "$CACHE_DIR"

section "kokoro server"
# kokoro-server.sh status prints three lines: state=, pid=, port=. Quoting
# it verbatim keeps the output machine-diffable across invocations.
"$SCRIPT_DIR/kokoro-server.sh" status 2>/dev/null || echo "(kokoro-server.sh not found or failed)"

section "playback state"
if [ -r "$STATE_FILE" ]; then
  cat "$STATE_FILE"
else
  echo "(no state file)"
fi

section "job lock"
if [ -r "$LOCK_FILE" ]; then
  pid=$(cat "$LOCK_FILE" 2>/dev/null || echo '')
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    printf 'lock=%s (alive)\n' "$pid"
    # ps line for the locked PID is useful when a runaway job needs a
    # manual kill — the user pastes this and sees exactly what's running.
    ps -o pid=,command= -p "$pid" 2>/dev/null | sed 's/^/  /'
  else
    printf 'lock=%s (stale)\n' "${pid:-empty}"
  fi
else
  echo 'no lock file (idle)'
fi

section "cache"
if [ -d "$CACHE_DIR" ]; then
  count=$(find "$CACHE_DIR" -maxdepth 1 -type f -name '*.wav' 2>/dev/null | wc -l | tr -d ' ')
  # du -sk keeps this portable (macOS du lacks -b); convert to MB for human reading.
  size_kb=$(du -sk "$CACHE_DIR" 2>/dev/null | awk '{print $1}')
  size_mb=$(awk -v k="${size_kb:-0}" 'BEGIN{printf "%.1f", k/1024}')
  cap_mb="${KOKORO_CACHE_MAX_MB:-200}"
  printf 'entries=%s  size=%sMB  cap=%sMB\n' "$count" "$size_mb" "$cap_mb"
else
  echo "(no cache dir)"
fi

section "env overrides (set only)"
# Only print vars that are actually set — a full "VAR= (unset)" dump of 20
# optional env vars is just noise in a paste.
any=0
for var in \
  KOKORO_VOICE KOKORO_SPEED KOKORO_SYNTH_CAP KOKORO_LANG \
  KOKORO_URL KOKORO_HOST KOKORO_PORT KOKORO_ALLOW_REMOTE \
  KOKORO_EARCON KOKORO_CACHE_DIR KOKORO_CACHE_MAX_MB KOKORO_NO_PLAY \
  KOKORO_EQ_GAIN_DB KOKORO_EQ_FREQ KOKORO_EQ_Q KOKORO_PAD_START_MS \
  CLAUDIO_DIR CLAUDIO_STATE_DIR CLAUDIO_PYTHON; do
  if [ -n "${!var:-}" ]; then
    printf '%s=%s\n' "$var" "${!var}"
    any=1
  fi
done
[ "$any" = 0 ] && echo "(none)"

section "recent play.log (last 10)"
if [ -r "$PLAY_LOG" ]; then
  tail -n 10 "$PLAY_LOG"
else
  echo "(no play.log)"
fi

section "recent kokoro-server.log (last 10)"
if [ -r "$SERVER_LOG" ]; then
  tail -n 10 "$SERVER_LOG"
else
  echo "(no kokoro-server.log)"
fi

echo
