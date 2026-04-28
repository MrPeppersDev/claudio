#!/bin/bash
# kokoro-server.sh — start / stop / status controller for the local Kokoro
# TTS HTTP server. play-last.sh starts it lazily on the first synth; the
# Hammerspoon menu offers a manual stop to release RAM.
#
# Usage: kokoro-server.sh {start|stop|restart|status}
#
# Start is idempotent: no-op if a healthy server is already listening on :8880.
# Stop kills the recorded PID (and any stragglers on :8880) and clears the
# state file.

set -euo pipefail

# Split state (pid, log) from repo (venv, server.py). The repo can live
# anywhere — we derive KOKORO_DIR from this script's own location — while
# state defaults to ~/.claude/claudio and can be moved via CLAUDIO_STATE_DIR.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_DIR="${CLAUDIO_STATE_DIR:-$HOME/.claude/claudio}"
KOKORO_DIR="$SCRIPT_DIR/kokoro"
PID_FILE="$STATE_DIR/kokoro-server.pid"
LOG_FILE="$STATE_DIR/kokoro-server.log"
PYTHON="$KOKORO_DIR/venv/bin/python"
SERVER_PY="$KOKORO_DIR/server.py"
PORT="${KOKORO_PORT:-8880}"

mkdir -p "$STATE_DIR"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >> "$LOG_FILE"; }

is_healthy() {
  curl -fsS -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1
}

recorded_pid() {
  [ -r "$PID_FILE" ] || return 1
  local pid
  pid=$(cat "$PID_FILE" 2>/dev/null || true)
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && printf '%s' "$pid"
}

port_holders() {
  # Print PIDs currently listening on our port (may be empty)
  lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t 2>/dev/null || true
}

cmd_status() {
  local pid state model log_tail
  if pid=$(recorded_pid); then
    if is_healthy; then state="running (healthy)"; else state="running (unhealthy)"; fi
  else
    pid=""
    if is_healthy; then state="running (unmanaged)"; else state="stopped"; fi
  fi
  # Best-effort context for bug reports: which model file is loaded and
  # the last few log lines. Both come from $LOG_FILE which is always
  # under STATE_DIR, so no path traversal concern. `claudio.lua` only
  # parses the `state=` line, so additional fields are safe to add.
  model=""
  if [ -r "$LOG_FILE" ]; then
    model=$(grep 'loading model from' "$LOG_FILE" 2>/dev/null | tail -1 \
      | sed 's/.*loading model from //')
  fi
  log_tail=""
  if [ -r "$LOG_FILE" ]; then
    log_tail=$(tail -n 3 "$LOG_FILE" 2>/dev/null | sed 's/^/  /')
  fi
  printf 'state=%s\npid=%s\nport=%s\nlog_file=%s\nmodel=%s\nlog_tail=\n%s\n' \
    "$state" "$pid" "$PORT" "$LOG_FILE" "$model" "$log_tail"
}

cmd_start() {
  if is_healthy; then
    log "start: already healthy on :$PORT"
    cmd_status
    return 0
  fi

  if [ ! -x "$PYTHON" ] || [ ! -r "$SERVER_PY" ]; then
    log "start: python or server.py missing ($PYTHON / $SERVER_PY)"
    echo "kokoro not installed" >&2
    return 1
  fi

  # Ensure the patched ONNX model (with duration predictor outputs) exists.
  # patch-model.sh is idempotent: no-op if the patched file is already newer
  # than the source model.
  local patch_script="$KOKORO_DIR/patch-model.sh"
  if [ -x "$patch_script" ]; then
    log "start: running model patcher"
    if ! "$patch_script" >> "$LOG_FILE" 2>&1; then
      log "start: patch-model.sh failed — check $LOG_FILE for details"
      echo "[kokoro] ERROR: patch-model.sh failed. Check $LOG_FILE." >&2
      return 1
    fi
  else
    log "start: patch-model.sh not found at $patch_script — skipping (old-style model)"
  fi

  # Self-heal: /health failed, but something may still hold the port — an old
  # server whose event loop is wedged, a crashed worker that leaked the socket,
  # or a stale PID pointing at a live-but-hung process. If we don't clear them,
  # the fresh `nohup` below fails to bind and the whole 8s wait times out with
  # a confusing error. Since /health just said nobody's serving us anyway,
  # killing port holders is safe; log loudly so post-mortems can see it fired.
  local stragglers
  stragglers=$(port_holders)
  if [ -n "$stragglers" ]; then
    log "start: /health unhealthy but port $PORT held by: $stragglers — sweeping"
    echo "$stragglers" | xargs -I{} kill -KILL {} 2>/dev/null || true
    sleep 0.2
  fi
  # Clear a stale PID file if the recorded process is dead, so a later
  # recorded_pid() doesn't return a ghost.
  if [ -r "$PID_FILE" ]; then
    local old_pid
    old_pid=$(cat "$PID_FILE" 2>/dev/null || true)
    if [ -n "$old_pid" ] && ! kill -0 "$old_pid" 2>/dev/null; then
      log "start: clearing stale pid file (pid=$old_pid not alive)"
      rm -f "$PID_FILE"
    fi
  fi

  log "start: launching server"
  # PYTHONPATH=repo root so server.py can `from kokoro.timing import ...`.
  # Without this, running `python kokoro/server.py` only puts kokoro/ on
  # sys.path, and the timing-sidecar import fails at request time.
  PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}" \
    nohup "$PYTHON" "$SERVER_PY" >> "$LOG_FILE" 2>&1 &
  local pid=$!
  echo "$pid" > "$PID_FILE"

  # Wait up to 8s for /health to succeed
  local i
  for i in 1 2 3 4 5 6 7 8; do
    sleep 1
    if is_healthy; then
      log "start: healthy after ${i}s (pid=$pid)"
      cmd_status
      return 0
    fi
    # If the process already died, bail early.
    if ! kill -0 "$pid" 2>/dev/null; then
      log "start: process exited during boot (pid=$pid) — see log"
      rm -f "$PID_FILE"
      return 1
    fi
  done

  log "start: timed out waiting for health (pid=$pid)"
  return 1
}

cmd_stop() {
  local pid=""
  pid=$(recorded_pid || true)
  if [ -n "$pid" ]; then
    log "stop: sending TERM to pid=$pid"
    kill -TERM "$pid" 2>/dev/null || true
    for i in 1 2 3 4 5; do
      if ! kill -0 "$pid" 2>/dev/null; then break; fi
      sleep 1
    done
    if kill -0 "$pid" 2>/dev/null; then
      log "stop: escalating to KILL pid=$pid"
      kill -KILL "$pid" 2>/dev/null || true
    fi
  fi
  # Sweep any stragglers still holding the port
  local stragglers
  stragglers=$(port_holders)
  if [ -n "$stragglers" ]; then
    log "stop: killing port stragglers: $stragglers"
    echo "$stragglers" | xargs -I{} kill -KILL {} 2>/dev/null || true
  fi
  rm -f "$PID_FILE"
  log "stop: complete"
  cmd_status
}

cmd_restart() {
  cmd_stop || true
  cmd_start
}

case "${1:-status}" in
  start)   cmd_start ;;
  stop)    cmd_stop ;;
  restart) cmd_restart ;;
  status)  cmd_status ;;
  *)       echo "usage: $0 {start|stop|restart|status}" >&2; exit 2 ;;
esac
