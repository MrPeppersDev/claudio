-- claudio.lua — menu bar dropdown for Kokoro TTS.
--
-- State source: ~/.claude/claudio/state (written by play-last.sh / kokoro-tts.sh).
--   state=<idle|synth|play>
--   preview=<first 80 chars of the message being read>
--   ts=<unix seconds>
--
-- Icon: ▸ idle, ⟳ synth, ▶ play.
--
-- Menu: Play/Stop toggle, manual Stop server (to free ~600 MB RAM when idle),
-- plus live status. F13 triggers the same toggle as the top menu item.

local M = {}

local rsvp = require("rsvp")

-- RSVP wiring state. Reset on every runPlayScript spawn.
--   controlFifoPath: path to the command FIFO advertised by kokoro-tts.sh via
--     its "CTRL <path>" stderr line. Lives only for the duration of one play
--     task; cleanup() in kokoro-tts.sh unlinks it when the task exits.
--   stderrBuffer: partial-line accumulator. hs.task stream callbacks deliver
--     arbitrarily-chunked stderr; we split on \n and flush line-by-line.
--   userInitiatedStop: set by M.toggle when the user is stopping in-flight
--     playback. The dying task's exit callback consumes the flag so a
--     SIGKILL fallback (exit 9) is treated like a clean SIGTERM rather
--     than firing an alert. The grace period in play-last.sh's stop path
--     should make SIGKILL rare, but cleanup work that runs over budget on
--     a busy system shouldn't surface as a "Claudio error".
local controlFifoPath = nil
local stderrBuffer = ""
local userInitiatedStop = false

local HOME = os.getenv("HOME")
-- CLAUDIO_DIR = repo (scripts, venv, model files). Defaults to the install
-- location that install.sh created, but overridable so the repo can live
-- anywhere. STATE_DIR = runtime files (state, job.pid). Users who set
-- either env var in a LaunchAgent or before launching Hammerspoon get the
-- override; plain Dock launches pick up the defaults.
local CLAUDIO_DIR = os.getenv("CLAUDIO_DIR") or (HOME .. "/.claude/claudio")
local STATE_DIR = os.getenv("CLAUDIO_STATE_DIR") or CLAUDIO_DIR
local STATE_FILE = STATE_DIR .. "/state"
local SPEED_FILE = STATE_DIR .. "/speed"
local VOICE_FILE = STATE_DIR .. "/voice"
local PLAY_LOG = STATE_DIR .. "/play.log"
local SERVER_LOG = STATE_DIR .. "/kokoro-server.log"
local DIAG_LOG = STATE_DIR .. "/hs.log"
local QUEUE_DIR = STATE_DIR .. "/queue"
-- CACHE_DIR mirrors kokoro-tts.sh's KOKORO_CACHE_DIR resolution. Used to
-- bound paths read from the play task's stderr stream (SIDECAR <path>),
-- which would otherwise be a stderr-injection arbitrary-file-read primitive.
local CACHE_DIR = os.getenv("KOKORO_CACHE_DIR") or (STATE_DIR .. "/cache")

-- Diagnostic logger: append-only `hs.log` capturing claudio + rsvp lifecycle
-- state transitions. Lets us reconstruct a bug repro after the fact (toggle
-- decisions, exit codes, userInitiatedStop flag, balloon hide/linger calls,
-- SIDECAR/CTRL accept/reject) without depending on the Hammerspoon console
-- buffer being captured live. play-stream.py and rsvp.lua append to the
-- same file so the timeline reads in order.
local function diagLog(msg)
  local t = hs.timer.secondsSinceEpoch()
  local secs = math.floor(t)
  local ms = math.floor((t - secs) * 1000)
  local f = io.open(DIAG_LOG, "a")
  if not f then return end
  f:write(string.format("[%s.%03d] [claudio] %s\n",
    os.date("%H:%M:%S", secs), ms, msg))
  f:close()
end
local PLAY_SCRIPT = CLAUDIO_DIR .. "/play-last.sh"
local SERVER_SCRIPT = CLAUDIO_DIR .. "/kokoro-server.sh"
local KOKORO_URL = os.getenv("KOKORO_URL") or "http://127.0.0.1:8880"
local DEFAULT_VOICE = "af_bella"

-- Speed options shown in the menu. 2.0 is the shell default; keeping it
-- here in one list means the checkmarked value and the persisted value
-- always agree. Chosen spacing: dense near 1.5–2.0 (where most users
-- live), stepped out at the extremes.
local SPEED_OPTIONS = { 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0 }
local DEFAULT_SPEED = 2.0

local menubar = nil
local watcher = nil
local audioWatcher = nil
-- eventtap that intercepts system media keys (play/pause, next, previous).
-- Stored here so M.stop() can clean it up on reload. nil when disabled via
-- CLAUDIO_MEDIA_KEYS=0 env var.
local mediaKeyTap = nil
-- Tracks the pasteboard changeCount at the time of the last toggle. Used by
-- the copy-on-select terminal fallback to distinguish "user just selected
-- new text" (changeCount advanced since last toggle) from "stale clipboard
-- from minutes ago" (no change since last toggle).
local lastSeenChangeCount = nil
-- Snapshot of the frontmost app at the moment the dropdown was built. When
-- the user clicks the menu icon, focus eventually shifts to Hammerspoon —
-- but buildMenu fires just before that transfer completes, so grabbing it
-- there captures the *real* user app (Safari, Notes, etc.). Used to re-
-- activate that app before running the toggle, so selection capture (AX /
-- simulated Cmd+C) targets the right window.
local lastFrontApp = nil

-- Terminal apps that copy-on-select (selection is already on the pasteboard
-- without Cmd+C). Gating the stale-pasteboard fallback to these apps avoids
-- replaying whatever's been sitting on the clipboard in a non-terminal app.
-- Membership keyed by hs.application.frontmostApplication():name().
local COPY_ON_SELECT_TERMINALS = {
  ["iTerm2"] = true,
  ["iTerm"] = true,
  ["Terminal"] = true,
  ["Alacritty"] = true,
  ["kitty"] = true,
  ["WezTerm"] = true,
  ["Ghostty"] = true,
}

-- PDF / image viewers where an empty-capture is often "user didn't know
-- they could select" rather than "user selected nothing." For these we
-- surface a targeted hint instead of the generic "highlight text to play"
-- alert — Cmd+C already works for selected text, and Live Text bridges
-- the scanned-PDF case without us needing a Vision/OCR helper.
local PDF_VIEWERS = {
  ["Preview"] = true,
  ["Books"] = true,
  ["Skim"] = true,
  ["Adobe Acrobat Reader"] = true,
  ["Acrobat Reader"] = true,
  ["PDF Expert"] = true,
}
local PDF_EMPTY_HINT = "Select text first; for scanned PDFs use Live Text"

-- ============================================================
-- State helpers
-- ============================================================

local function readState()
  local f = io.open(STATE_FILE, "r")
  if not f then return { state = "idle", preview = "", ts = 0 } end
  local s = { state = "idle", preview = "", ts = 0 }
  for line in f:lines() do
    local k, v = line:match("^([^=]+)=(.*)$")
    if k == "state" then s.state = v
    elseif k == "preview" then s.preview = v
    elseif k == "ts" then s.ts = tonumber(v) or 0
    end
  end
  f:close()
  return s
end

local function readSpeed()
  local f = io.open(SPEED_FILE, "r")
  if not f then return DEFAULT_SPEED end
  local raw = (f:read("*a") or ""):match("^%s*([%d%.]+)")
  f:close()
  local n = tonumber(raw)
  return n or DEFAULT_SPEED
end

local function writeSpeed(s)
  local f = io.open(SPEED_FILE, "w")
  if not f then return end
  f:write(string.format("%g\n", s))
  f:close()
end

local function readVoice()
  local f = io.open(VOICE_FILE, "r")
  if not f then return DEFAULT_VOICE end
  local raw = (f:read("*a") or ""):match("^%s*([%w_%.,:]+)")
  f:close()
  return raw or DEFAULT_VOICE
end

local function writeVoice(v)
  local f = io.open(VOICE_FILE, "w")
  if not f then return end
  f:write(v .. "\n")
  f:close()
end

-- Voice list comes from the server's /voices endpoint so we don't have to
-- ship a separate manifest that drifts from the actual shipped model. The
-- server doesn't change voices between runs, so cache the response for
-- the life of the Hammerspoon process rather than re-fetching every menu
-- open. A cold fetch costs ~5-20 ms; stale results would only matter if
-- someone swaps model files mid-session, which they can fix by reloading
-- Hammerspoon.
local cachedVoices = nil
local function fetchVoices()
  if cachedVoices then return cachedVoices end
  local out, ok_ = hs.execute(
    "curl -fsS -m 2 '" .. KOKORO_URL .. "/voices' 2>/dev/null", true)
  if not ok_ or not out or out == "" then return nil end
  local voices = {}
  -- Minimal JSON pluck — the response is small and we don't want to pull
  -- in a JSON library just for this. Match bare-word voice names inside
  -- the array literal.
  for name in out:gmatch('"([%w_]+)"') do
    if name ~= "voices" then
      table.insert(voices, name)
    end
  end
  if #voices == 0 then return nil end
  table.sort(voices)
  cachedVoices = voices
  return voices
end

-- ============================================================
-- Queue helpers
-- ============================================================

-- Returns a list of queued items oldest-first: { {name, preview}, ... }.
-- Sorting by mtime matches play-last.sh's drain order (`ls -tr`), so what
-- the menu shows is exactly what will play. Preview is the first non-blank
-- line of the file truncated for menu display.
local function readQueue()
  -- hs.fs.dir throws on a missing directory (not just returns nil), so
  -- stat first. Queue dir is created by play-last.sh on first enqueue.
  if not hs.fs.attributes(QUEUE_DIR) then return {} end
  local ok_dir, dir = pcall(hs.fs.dir, QUEUE_DIR)
  if not ok_dir or not dir then return {} end
  local items = {}
  for name in dir do
    if name ~= "." and name ~= ".." then
      local path = QUEUE_DIR .. "/" .. name
      local attrs = hs.fs.attributes(path)
      if attrs and attrs.mode == "file" then
        local f = io.open(path, "r")
        local first = ""
        if f then
          -- Read enough to get past leading blank lines without slurping a
          -- 100 KB paste. 2 KB is plenty for a menu preview.
          local head = f:read(2048) or ""
          f:close()
          for line in head:gmatch("[^\n]+") do
            local t = line:match("^%s*(.-)%s*$")
            if t ~= "" then first = t; break end
          end
        end
        if #first > 60 then first = first:sub(1, 57) .. "…" end
        table.insert(items, {
          name = name,
          preview = first,
          mtime = attrs.modification or 0,
        })
      end
    end
  end
  table.sort(items, function(a, b) return a.mtime < b.mtime end)
  return items
end

-- Clear all queue files. Leaves the drain coordinator alive: it will see
-- the empty directory on its next `ls` iteration and exit through its own
-- cleanup trap, which is safer than trying to signal it from here.
local function clearQueueFiles()
  hs.execute("rm -f '" .. QUEUE_DIR .. "'/item.* 2>/dev/null", true)
end

-- ============================================================
-- Process helpers (Lua 5.3 vs 5.1 exit code shape)
-- ============================================================

local function ok(ec) return ec == 0 or ec == true end

local function jobIsRunning()
  local f = io.open(STATE_DIR .. "/job.pid", "r")
  if not f then return false end
  local pid = tonumber((f:read("*a") or ""):match("%d+"))
  f:close()
  if not pid then return false end
  local _, _, ec = os.execute("kill -0 " .. pid .. " 2>/dev/null")
  return ok(ec)
end

local function kokoroServerRunning()
  -- kokoro-server.sh status returns state=running (healthy|...|unmanaged) or stopped
  local out = hs.execute(SERVER_SCRIPT .. " status 2>/dev/null", true)
  return out ~= nil and out:find("state=running") ~= nil
end

local function kokoroServerStop()
  hs.execute(SERVER_SCRIPT .. " stop >/dev/null 2>&1", true)
end

-- POSIX shell single-quote escape: wraps `s` in single quotes and replaces any
-- embedded single quote with the standard `'\''` sequence. Used to interpolate
-- paths into shell commands (notably the tail-F invocation passed to Terminal
-- via AppleScript) so a path containing a single quote can't break out of the
-- argument and inject metacharacters. Hostile paths in normal use are unlikely
-- but CLAUDIO_STATE_DIR is read from the environment.
local function shellQuote(s)
  return "'" .. s:gsub("'", "'\\''") .. "'"
end

-- Opens Terminal.app running `tail -F` on both Claudio log files. Terminal is
-- shipped with macOS so this works on a vanilla install without requiring the
-- user to have iTerm / kitty / etc. `touch` first so `tail -F` doesn't spin
-- on a fresh install where no log exists yet; `-F` (capital) keeps following
-- across rotate/recreate, which `kokoro-server.sh stop; start` will cause.
local function tailLogs()
  local touchCmd = string.format(
    "touch %s %s 2>/dev/null", shellQuote(PLAY_LOG), shellQuote(SERVER_LOG)
  )
  hs.execute(touchCmd, true)
  local cmd = string.format(
    "tail -F %s %s", shellQuote(PLAY_LOG), shellQuote(SERVER_LOG)
  )
  -- %q quotes and escapes for Lua, but AppleScript double-quoted literals use
  -- the same `\"` / `\\` / `\n` conventions, so it round-trips cleanly for
  -- the plain-ASCII command we're passing.
  local script = string.format(
    'tell application "Terminal"\nactivate\ndo script %q\nend tell',
    cmd
  )
  hs.osascript.applescript(script)
end

-- ============================================================
-- Script runner (async — don't block Hammerspoon on synthesis)
-- ============================================================

-- Confine a path read from the play task's stderr to a known directory.
-- The SIDECAR/CTRL stderr protocol carries paths from the child process;
-- without these checks, any line a child writes to stderr matching the
-- protocol regex (e.g. a Python traceback containing such a string) can
-- redirect file reads or control writes to attacker-chosen locations.
-- Trailing-slash check on the prefix prevents `/foo/bar` matching
-- `/foo/barbaz`. Reject literal `..` segments rather than try to resolve
-- them — Lua has no realpath and the protocol never legitimately uses them.
local function pathInsideDir(path, dir)
  if type(path) ~= "string" or path == "" then return false end
  if path:find("/%.%./") or path:sub(-3) == "/.." or path:sub(1, 3) == "../" then
    return false
  end
  local prefix = dir
  if prefix:sub(-1) ~= "/" then prefix = prefix .. "/" end
  return path:sub(1, #prefix) == prefix
end

-- Read a sidecar JSON file produced by the Kokoro server. Returns the raw
-- JSON text (rsvp.feedSidecar does its own decode) or nil if unreadable.
-- Missing files are expected when the server ran without a patched model —
-- not an error, just "no word timings available for this buffer."
local function loadSidecar(path)
  local f = io.open(path, "r")
  if not f then return nil end
  local content = f:read("*a")
  f:close()
  return content
end

-- Append a line (PAUSE / RESUME / SEEK <ms>) to the control FIFO so the
-- shell subshell forwards it to play-stream.py's fd 9. Uses hs.task so the
-- write happens off the main thread — if the FIFO has no reader (stale job,
-- dead subshell), a direct io.open(fifo, "w") would block HS's main loop
-- indefinitely. The shell redirect still blocks in its own process but that
-- process is detached from HS, and the hs.timer cleans it up after 500 ms.
-- Whitelist is enforced on the shell side (PAUSE|RESUME|"SEEK "*) so
-- line contents can't drive arbitrary commands even with shell escaping.
local function writeControl(line)
  if not controlFifoPath then return end
  -- Race guard: kokoro-tts.sh's cleanup() unlinks the FIFO when its task
  -- exits, but its EXIT trap, the play-stream DRAIN_DONE handshake, and a
  -- late RSVP click all happen on independent timelines. If we shell out
  -- after the FIFO is gone, `printf > path` creates a *regular* file —
  -- which then accumulates as control.* turds in $STATE_DIR. Stat first;
  -- if the path no longer resolves to a named pipe, drop the line on the
  -- floor. The completion callback in runPlayScript will null out
  -- controlFifoPath once the task finishes, so subsequent calls bail at
  -- the nil check above.
  local attrs = hs.fs.attributes(controlFifoPath)
  if not attrs or attrs.mode ~= "named pipe" then return end
  local t = hs.task.new("/bin/sh", nil, {
    "-c", [[printf '%s\n' "$1" > "$2" 2>/dev/null]],
    "--", line, controlFifoPath,
  })
  if not t then return end
  t:start()
  hs.timer.doAfter(0.5, function()
    if t and t:isRunning() then t:terminate() end
  end)
end

-- Parse one stderr line from the play task. Protocol:
--   POS <ms> <buffer_id>      → drive the RSVP word highlight
--   SIDECAR <buffer_id> <path> → load sidecar JSON, feed to rsvp
--   CTRL <fifo_path>          → remember the command FIFO path
-- Any other line (e.g. shell diagnostics) is ignored — play-stream already
-- logs its own state to play.log, so nothing else on this channel matters.
local function processStderrLine(line)
  local prefix = line:sub(1, 4)
  if prefix == "POS " then
    local ms_s, id_s = line:match("^POS (%d+) (%d+)$")
    if ms_s and id_s then
      rsvp.onPos(tonumber(ms_s), tonumber(id_s))
    end
    return
  end
  if line:sub(1, 8) == "SIDECAR " then
    local id_s, path = line:match("^SIDECAR (%d+) (.+)$")
    if id_s and path then
      if not pathInsideDir(path, CACHE_DIR) or not path:match("%.words%.json$") then
        diagLog(string.format("SIDECAR rejected (outside CACHE_DIR=%s): %s",
                              CACHE_DIR, path))
        hs.printf("[claudio] rejecting SIDECAR path outside CACHE_DIR: %s", path)
        return
      end
      local json = loadSidecar(path)
      if json then
        diagLog(string.format("SIDECAR accepted: id=%s bytes=%d path=%s",
                              id_s, #json, path))
        rsvp.feedSidecar(tonumber(id_s), json)
      else
        -- Sidecar was announced but not readable. Common causes: server
        -- ran without a patched model (no timings produced), permission
        -- error, or a write race. Surface in the HS console so "balloon
        -- stays empty" has a diagnostic trail instead of looking like a
        -- silent UI bug.
        diagLog(string.format("SIDECAR unreadable: id=%s path=%s", id_s, path))
        hs.printf("[claudio] SIDECAR missing or unreadable: id=%d path=%s",
                  tonumber(id_s), path)
      end
    end
    return
  end
  if line:sub(1, 5) == "CTRL " then
    local path = line:match("^CTRL (.+)$")
    if path then
      local basename = path:match("([^/]+)$") or ""
      if not pathInsideDir(path, STATE_DIR) or basename:sub(1, 8) ~= "control." then
        diagLog(string.format("CTRL rejected (outside STATE_DIR=%s, basename=%s): %s",
                              STATE_DIR, basename, path))
        hs.printf("[claudio] rejecting CTRL path outside STATE_DIR: %s", path)
        return
      end
      diagLog("CTRL accepted: " .. path)
      controlFifoPath = path
    end
    return
  end
  -- DRAIN_DONE: play-stream emits this on stderr after the last sample of
  -- the last buffer is delivered to the audio device. We trigger the linger-
  -- fade now rather than waiting on the bash task to exit, because the bash
  -- shutdown chain (play-stream → kokoro-tts.sh → play-last.sh) can lag the
  -- final word by several seconds; without this hook the balloon sat on
  -- screen until the 8s hang watchdog force-hid it. endOfStream is
  -- idempotent (scheduleLingerFadeOut returns early if a timer is armed),
  -- so a later task-exit endOfStream is a no-op.
  if line == "DRAIN_DONE" then
    diagLog("DRAIN_DONE → endOfStream (linger-fade)")
    rsvp.endOfStream()
    return
  end
  -- Anything else on stderr is unstructured output (e.g. bash error messages,
  -- python tracebacks). Surface it so we don't lose visibility into spawn or
  -- script failures — without this, a `set -u` death or mkdir error in the
  -- spawned shell would be invisible.
  diagLog("stderr: " .. line)
end

local function runPlayScript(args)
  diagLog(string.format("runPlayScript: args=[%s]", table.concat(args, " ")))
  -- Per-run state reset. Leftover controlFifoPath from a prior run would
  -- point at an already-unlinked FIFO; writing to it is a silent no-op but
  -- clearer to just drop the reference. Stderr partial-line buffer is also
  -- per-run: we don't want a half-line from a crashed prior task bleeding
  -- into the next task's first line.
  controlFifoPath = nil
  stderrBuffer = ""
  -- Sweep stray control.* files left by the writeControl FIFO race —
  -- between hs.fs.attributes (named-pipe check) and the spawned shell's
  -- `>` redirect, kokoro-tts.sh's cleanup can unlink the FIFO, and the
  -- shell's `>` then creates a regular file at the same path. Those
  -- accumulate as turds in $STATE_DIR over rapid stop/play cycles.
  --
  -- Implemented in pure Lua via hs.fs.dir / os.remove rather than
  -- shelling out via hs.execute. The shell-out version called
  -- hs.execute(..., true) which spawns a LOGIN shell on every
  -- runPlayScript (every play AND every stop), running the user's
  -- full bash startup. That's both slow and a strong suspect for the
  -- "F13 stops responding after a stop sequence" symptom — login
  -- shell startup was modifying focus/keyboard state in some way
  -- that disrupted Hammerspoon's Carbon hotkey registration. Pure-
  -- Lua sweep is fast and side-effect-free.
  if STATE_DIR and STATE_DIR ~= "" then
    -- Wrap the whole loop in pcall, not just hs.fs.dir's call: hs.fs.dir
    -- returns (iterFn, dirObj), and Lua's `for x in expr do` only feeds
    -- expr's first return to the iterator unless the multi-return is
    -- preserved at the for-loop site. Splitting `local _, iter = pcall(...)`
    -- discards dirObj, so the first iteration calls iterFn(nil, nil) and
    -- throws "directory metatable expected, got nil" — which is exactly
    -- the silent runPlayScript abort that wedged every F13 after #155.
    pcall(function()
      for entry in hs.fs.dir(STATE_DIR) do
        if type(entry) == "string" and entry:sub(1, 8) == "control." then
          os.remove(STATE_DIR .. "/" .. entry)
        end
      end
    end)
  end

  hs.task.new("/bin/bash",
    function(code)
      -- Flush any trailing unterminated line (normally there isn't one —
      -- kokoro-tts.sh ends with newline-terminated output — but if the
      -- task died mid-write we'd miss the last event without this).
      if stderrBuffer ~= "" then
        processStderrLine(stderrBuffer)
        stderrBuffer = ""
      end
      controlFifoPath = nil
      -- Two task callbacks share userInitiatedStop when M.toggle's stop
      -- branch runs: the stopper task (which just kills the original and
      -- exits 0) and the original playback task (143 or 9). Snapshot the
      -- flag first, then clear only on signal-shaped exits (9, 143) — the
      -- code-0 stopper must not consume what the original is about to
      -- read. Clearing on 143 too prevents a stranded flag from masking a
      -- future real code-9 crash.
      local wasUserStop = userInitiatedStop
      if wasUserStop and (code == 9 or code == 143) then
        userInitiatedStop = false
      end
      local userStop = wasUserStop and code == 9
      -- Clean playback finish: trigger the balloon's linger-fade. SIGTERM
      -- (143, from user Stop) is handled separately in M.toggle where
      -- rsvp.hide() fires immediately. Non-zero non-143 is a real error;
      -- we still want the balloon cleared so a crashed task doesn't leave
      -- it dangling, so hide it directly.
      if code == 0 or code == 143 or userStop then
        -- Stale-callback guard: when the user does F13-stop then
        -- F13-play in quick succession, the OLD stopper task exits 0
        -- a few seconds later — well after the NEW playback's balloon
        -- is on screen. Without this guard, that stale exit calls
        -- rsvp.endOfStream() on the live balloon and arms a 1.25s
        -- linger fade, making the new balloon vanish mid-document.
        -- jobIsRunning() reading the new lock file is the simplest
        -- "is there a successor in progress" signal we have.
        if jobIsRunning() then
          diagLog(string.format(
            "task exit: code=%s — newer job is running; skipping endOfStream",
            tostring(code)))
        else
          diagLog(string.format(
            "task exit: code=%s wasUserStop=%s userStop=%s → endOfStream",
            tostring(code), tostring(wasUserStop), tostring(userStop)))
          rsvp.endOfStream()
        end
      else
        diagLog(string.format(
          "task exit: code=%s wasUserStop=%s userStop=%s → hide+alert",
          tostring(code), tostring(wasUserStop), tostring(userStop)))
        rsvp.hide()
        hs.alert.show("Claudio error (exit " .. code .. ")")
      end
    end,
    function(_, _, stderr)
      -- Stream callback. Buffer stderr chunks and emit complete lines.
      if stderr and stderr ~= "" then
        stderrBuffer = stderrBuffer .. stderr
        while true do
          local nl = stderrBuffer:find("\n", 1, true)
          if not nl then break end
          local line = stderrBuffer:sub(1, nl - 1)
          stderrBuffer = stderrBuffer:sub(nl + 1)
          if line ~= "" then
            processStderrLine(line)
          end
        end
      end
      return true  -- keep streaming
    end,
    args
  ):start()
end

-- ============================================================
-- Selection capture (AX → Cmd+C → copy-on-select terminal fallback)
-- ============================================================

-- Pull the selected text via Accessibility without touching the keyboard
-- or clipboard. Avoiding Cmd+C matters when other apps (e.g. Wispr Flow)
-- watch clipboard changes or intercept keyDown events, because that round-
-- trip introduces cross-app contention that can make F13 feel frozen.
--
-- Tries, in order:
--   1. AXSelectedText on the focused element — the obvious path, works
--      in native Cocoa apps (TextEdit, Notes, Safari native views).
--   2. AXStringForRange(AXSelectedTextRange) on the focused element —
--      Chromium/Electron renderers expose this where AXSelectedText is
--      empty. VS Code, Chrome, Arc, Cursor, Slack all fall here.
--   3. Walk up the parent chain once in case the focus landed on a
--      wrapper element rather than the text view that owns the
--      selection (common in custom text widgets and some Electron apps).
--
-- Returns the selected string, or nil if every path came up dry.
local function axSelection()
  local okCall, sys = pcall(hs.axuielement.systemWideElement)
  if not okCall or not sys then return nil end
  local focused = sys:attributeValue("AXFocusedUIElement")
  if not focused then return nil end

  local function tryElement(el)
    if not el then return nil end
    local ok, text = pcall(function() return el:attributeValue("AXSelectedText") end)
    if ok and type(text) == "string" and #text > 0 then return text end
    local okr, range = pcall(function() return el:attributeValue("AXSelectedTextRange") end)
    if okr and range then
      local okq, str = pcall(function()
        return el:parameterizedAttributeValue("AXStringForRange", range)
      end)
      if okq and type(str) == "string" and #str > 0 then return str end
    end
    return nil
  end

  local text = tryElement(focused)
  if text then return text end

  -- Second chance: focus landed on a wrapper; try the parent once.
  local okP, parent = pcall(function() return focused:attributeValue("AXParent") end)
  if okP and parent then
    text = tryElement(parent)
    if text then return text end
  end

  return nil
end

-- Compute a screen-space anchor rect for "where the user's attention is."
-- Preference order:
--   1. AX bounds for the focused element's selected text range — precise rect
--      around the actual highlighted glyphs. Works in TextEdit, Safari, Notes,
--      VS Code, most native apps. Falls through silently when the focused
--      element doesn't implement AXBoundsForRange (some Electron, Chromium
--      renderers).
--   2. Mouse cursor position — users who drag to highlight end with the mouse
--      hovering at the end of the selection, so this is a good proxy.
--   3. nil — rsvp.lua falls back to menubar-anchored positioning.
-- Returns a rect { x, y, w, h } in global screen coordinates, or nil.
local function selectionAnchorRect()
  local okCall, sys = pcall(hs.axuielement.systemWideElement)
  if okCall and sys then
    local focused = sys:attributeValue("AXFocusedUIElement")
    if focused then
      local range = focused:attributeValue("AXSelectedTextRange")
      if range then
        local ok_bounds, bounds = pcall(function()
          return focused:parameterizedAttributeValue("AXBoundsForRange", range)
        end)
        if ok_bounds and type(bounds) == "table"
           and bounds.x and bounds.y and bounds.w and bounds.h
           and bounds.w > 0 and bounds.h > 0 then
          return { x = bounds.x, y = bounds.y, w = bounds.w, h = bounds.h }
        end
      end
    end
  end
  -- Mouse-cursor fallback removed: in multi-monitor setups the cursor
  -- often ends up on a different screen than the focused window (e.g.,
  -- user reaches over to a portrait DELL while reading on the laptop),
  -- and using its position sends the balloon to that wrong monitor.
  -- Returning nil lets rsvp.lua's menubarFrame fallback position the
  -- balloon on the focused window's screen instead, which is much
  -- closer to where the user is actually reading.
  return nil
end

local function captureSelection(cb)
  local frontApp = hs.application.frontmostApplication()
  local frontName = frontApp and frontApp:name() or "?"

  local ax = axSelection()
  if ax then
    diagLog(string.format(
      "captureSelection: AX hit, %d chars (front=%s)", #ax, frontName))
    print(string.format("[claudio] AX selection: %d chars (front=%s)", #ax, frontName))
    cb(ax)
    return
  end

  diagLog(string.format(
    "captureSelection: AX empty, falling to Cmd+C (front=%s)", frontName))

  local prevCount = hs.pasteboard.changeCount()
  local prevContents = hs.pasteboard.getContents()
  -- 10 ms delay between keydown and keyup. Zero works in most apps but
  -- browsers (Chromium/WebKit) sometimes miss Cmd+C because the modifier
  -- up-event races the character through a different handler path, so
  -- the renderer never sees "cmd still down when c arrived." 10 ms is
  -- imperceptible on F13 press but reliable across Chrome/Safari/Firefox.
  hs.eventtap.keyStroke({ "cmd" }, "c", 10000)

  -- Poll the pasteboard for the copy to land. Cmd+C dispatch is async and
  -- per-app latency varies: AX-capable text fields respond in ~20-50 ms,
  -- iTerm up to ~150 ms under load. A fixed wait either gives up too early
  -- on a slow app or makes F13 feel laggy on a fast one. Polling every
  -- ~20 ms lets us return the moment changeCount advances.
  local MAX_WAIT_MS = 500
  local POLL_MS = 20

  local function finish()
    local newCount = hs.pasteboard.changeCount()
    local newContents = hs.pasteboard.getContents()

    if newCount > prevCount and newContents and #newContents > 0 then
      diagLog(string.format(
        "captureSelection: Cmd+C advanced count %d→%d, %d chars (front=%s)",
        prevCount, newCount, #newContents, frontName))
      -- Restore pasteboard. When the pasteboard was empty before we ran
      -- Cmd+C, prevContents is nil — setContents(nil) is a no-op, so the
      -- captured selection would linger until the user's next copy.
      -- setContents("") overwrites with empty and clears the residue.
      hs.pasteboard.setContents(prevContents or "")
      cb(newContents)
      return
    end

    local isTerminal = COPY_ON_SELECT_TERMINALS[frontName] == true
    local baseline = lastSeenChangeCount or prevCount
    if isTerminal and newContents and #newContents > 0
        and newCount > baseline then
      diagLog(string.format(
        "captureSelection: terminal copy-on-select hit, %d chars (front=%s)",
        #newContents, frontName))
      lastSeenChangeCount = newCount
      cb(newContents)
      return
    end

    lastSeenChangeCount = newCount
    diagLog(string.format(
      "captureSelection: empty (count %d→%d, front=%s)",
      prevCount, newCount, frontName))
    -- Empty capture. For known PDF viewers, try to resolve the open
    -- document path via AX so the caller can attempt direct extraction
    -- with pdf_extract.py. If AX yields a path we return a table hint
    -- {pdfPath=..., text=...} so the caller can choose; if not, fall back
    -- to the plain text hint that nudges toward Live Text.
    if PDF_VIEWERS[frontName] then
      local pdfPath = nil
      local ok, frontWin = pcall(function()
        return hs.application.frontmostApplication():focusedWindow()
      end)
      if ok and frontWin then
        local axOk, winEl = pcall(hs.axuielement.windowElement, frontWin)
        if axOk and winEl then
          -- AXDocument is a file:// URL on the main window in Preview and Skim.
          local docAttr = winEl:attributeValue("AXDocument")
          if type(docAttr) == "string" and docAttr:sub(1, 7) == "file://" then
            -- URL-decode %20 etc. via a simple gsub for the common case;
            -- full percent-decoding is not needed for typical file paths.
            pdfPath = docAttr:sub(8):gsub("%%(%x%x)", function(h)
              return string.char(tonumber(h, 16))
            end)
          end
        end
      end
      if pdfPath then
        cb(nil, { pdfPath = pdfPath, text = frontName .. ": " .. PDF_EMPTY_HINT })
      else
        cb(nil, { text = frontName .. ": " .. PDF_EMPTY_HINT })
      end
    else
      cb(nil, nil)
    end
  end

  local elapsed = 0
  local function poll()
    if hs.pasteboard.changeCount() > prevCount then
      finish()
      return
    end
    elapsed = elapsed + POLL_MS
    if elapsed >= MAX_WAIT_MS then
      finish()
      return
    end
    hs.timer.doAfter(POLL_MS / 1000, poll)
  end
  hs.timer.doAfter(POLL_MS / 1000, poll)
end

-- ============================================================
-- Toggle (same entry point from menu item and F13)
-- ============================================================

-- Common: grab selection → write to a mode-0600 tempfile → invoke play-last.sh.
-- Returns the tmp path, or nil on failure (already alerted).
local function selectionToTempfile(selection)
  -- mktemp creates the file mode-0600 on macOS, so the selection —
  -- which may contain tokens, private chat text, or PII — isn't
  -- world-readable in ~/.claude/claudio/. play-last.sh unlinks the
  -- file immediately after reading it, so the window of exposure is
  -- only the synth lifetime rather than until-next-F13.
  local tmp = hs.execute("mktemp '" .. STATE_DIR .. "/selection.XXXXXX'", true)
  tmp = tmp and tmp:gsub("%s+$", "") or nil
  if not tmp or tmp == "" then
    hs.alert.show("Claudio: mktemp failed")
    return nil
  end
  local f = io.open(tmp, "w")
  if f then f:write(selection); f:close() end
  return tmp
end

-- Stop the running TTS job (if any) and clear the balloon. Used by:
--   * M.toggle's stop branch (F13 during playback)
--   * rsvp.on_close_request (× button in balloon)
--   * audioWatcher (output device change)
-- Sets userInitiatedStop=true so the dying task's exit callback (which
-- typically arrives with code 9 from SIGKILL or 143 from SIGTERM) is
-- treated as a user stop and the spurious "Claudio error (exit 9)" alert
-- is suppressed. Previously only M.toggle set the flag — the × button and
-- audioWatcher paths invoked the stopper without it, so every device
-- switch or balloon close fired an alert.
local function stopPlayback(reason)
  if not jobIsRunning() then
    diagLog("stopPlayback: no job running (reason=" .. (reason or "?") .. ")")
    return false
  end
  userInitiatedStop = true
  diagLog(string.format(
    "stopPlayback: reason=%s, userInitiatedStop ← true, runPlayScript spawn",
    reason or "?"))
  runPlayScript({ PLAY_SCRIPT })
  -- Stop kills play-stream abruptly — POS events halt and rsvp would sit
  -- on its last word forever waiting for the linger timer that never
  -- arms (the linger trigger only fires when POS crosses last_word.end_ms).
  diagLog("stopPlayback: calling rsvp.hide()")
  rsvp.hide()
  return true
end

function M.toggle()
  local running = jobIsRunning()
  diagLog(string.format("toggle entry: jobIsRunning=%s userInitiatedStop=%s",
    tostring(running), tostring(userInitiatedStop)))
  if running then
    stopPlayback("M.toggle")
    return
  end
  diagLog("toggle: play branch — capturing selection")

  -- Snapshot the anchor rect *before* captureSelection goes async. The AX
  -- and mouse state is most reliable in the instant the user presses F13;
  -- after the Cmd+C fallback path runs, focus and selection can shift.
  local anchor = selectionAnchorRect()

  captureSelection(function(selection, hint)
    if selection then
      local tmp = selectionToTempfile(selection)
      if tmp then
        -- Show the balloon empty; the first SIDECAR event will populate it.
        -- Delaying this until after the first SIDECAR would leave F13 with
        -- no visible response for ~500ms-2s of synth time.
        rsvp.show(nil, { anchor = anchor })
        runPlayScript({ PLAY_SCRIPT, "--text-file", tmp })
      end
    elseif hint and hint.pdfPath then
      -- PDF viewer with a resolved document path: try direct extraction via
      -- pdf_extract.py. On success (exit 0) pipe the extracted text into
      -- play-last.sh the same way a selection would go. On exit 2 (scanned /
      -- no text layer) or exit 1 (other error) fall back to the Live Text
      -- nudge so the user still gets a useful prompt.
      local extractScript = CLAUDIO_DIR .. "/pdf_extract.py"
      local tmp = hs.execute("mktemp '" .. STATE_DIR .. "/pdf_extract.XXXXXX'", true)
      tmp = tmp and tmp:gsub("%s+$", "") or nil
      if not tmp or tmp == "" then
        hs.alert.show("Claudio: mktemp failed")
        return
      end
      hs.task.new("/usr/bin/python3",
        function(code, stdout, stderr)
          if code == 0 and stdout and #stdout > 0 then
            -- Write extracted text to the tempfile and play it.
            local f = io.open(tmp, "w")
            if f then
              f:write(stdout)
              f:close()
              rsvp.show(nil, { anchor = anchor })
              runPlayScript({ PLAY_SCRIPT, "--text-file", tmp })
            else
              os.remove(tmp)
              hs.alert.show("Claudio: could not write PDF text")
            end
          else
            -- Exit 2 = scanned PDF; exit 1 = other error. Either way, nudge
            -- toward Live Text so the user isn't left with nothing.
            os.remove(tmp)
            hs.alert.show("Claudio: " .. (hint.text or PDF_EMPTY_HINT), 3)
          end
        end,
        { extractScript, "--path", hint.pdfPath }
      ):start()
    elseif hint and hint.text then
      -- PDF viewer but no resolved path (AX gave us nothing): show the hint.
      hs.alert.show("Claudio: " .. hint.text, 3)
    else
      hs.alert.show("Claudio: highlight text to play")
    end
  end)
end

-- Append current selection to the queue. If a job is already running, this
-- lets the user stack "what's next" without interrupting what's playing.
-- play-last.sh handles both the mv-into-queue and (if needed) spawning the
-- drain coordinator, so from here it's just capture → write → invoke.
function M.queueAdd()
  captureSelection(function(selection)
    if not selection then
      hs.alert.show("Claudio: highlight text to queue")
      return
    end
    local tmp = selectionToTempfile(selection)
    if tmp then
      runPlayScript({ PLAY_SCRIPT, "--queue-add", "--text-file", tmp })
      hs.alert.show("Queued", 0.8)
    end
  end)
end

-- ============================================================
-- Menu rendering
-- ============================================================

local function iconFor(state)
  if state == "play"  then return "▶"
  elseif state == "synth" then return "⟳"
  else                     return "▸" end
end

local function buildMenu()
  -- Snapshot the user's app before the menu takes focus, so the Play item
  -- can refocus it and capture selection correctly.
  local app = hs.application.frontmostApplication()
  if app and app:name() ~= "Hammerspoon" then
    lastFrontApp = app
  end

  local s = readState()
  local busy = (s.state == "play" or s.state == "synth")
  local kokoroUp = kokoroServerRunning()
  local queue = readQueue()

  local statusLabel
  if s.state == "play" then statusLabel = "Playing"
  elseif s.state == "synth" then statusLabel = "Synthesizing"
  else statusLabel = "Idle" end

  local items = {}

  -- Play / Stop toggle. From the menu we refocus the previously-frontmost
  -- app before running the toggle, because AX and Cmd+C need the real user
  -- app as frontmost — not Hammerspoon, which the menu briefly focuses.
  -- Stop doesn't need any of that, so it can just fire.
  table.insert(items, {
    title = busy and ("■ Stop") or ("▶ Speak selection"),
    fn = function()
      if busy then
        M.toggle()
        return
      end
      if lastFrontApp then
        lastFrontApp:activate()
        -- 80 ms is enough for focus to settle on all apps I tested; AX and
        -- pasteboard reads come back clean after this delay.
        hs.timer.doAfter(0.08, function() M.toggle() end)
      else
        M.toggle()
      end
    end,
  })

  -- Queue selection. Same refocus dance as Speak — the capture path is
  -- identical, we just hand off to play-last.sh with --queue-add. Title
  -- shows the pending count when non-empty so the user doesn't have to
  -- open the submenu to see how deep the queue is.
  local queueTitle = "＋ Queue selection"
  if #queue > 0 then
    queueTitle = queueTitle .. " (" .. #queue .. " pending)"
  end
  table.insert(items, {
    title = queueTitle,
    fn = function()
      if lastFrontApp then
        lastFrontApp:activate()
        hs.timer.doAfter(0.08, function() M.queueAdd() end)
      else
        M.queueAdd()
      end
    end,
  })

  -- Queue inspection + clear. Submenu is only shown when there's something
  -- to inspect; an empty submenu is just noise. Items are disabled (display-
  -- only) — clicking them should not replay or reorder; the only action is
  -- "Clear queue".
  if #queue > 0 then
    local queueSubmenu = {}
    for i, item in ipairs(queue) do
      local preview = item.preview ~= "" and item.preview or "(empty)"
      table.insert(queueSubmenu, {
        title = string.format("%d. %s", i, preview),
        disabled = true,
      })
    end
    table.insert(queueSubmenu, { title = "-" })
    table.insert(queueSubmenu, {
      title = "Clear queue",
      fn = function() clearQueueFiles() end,
    })
    table.insert(items, {
      title = "Queue: " .. #queue .. " pending",
      menu = queueSubmenu,
    })
  end

  table.insert(items, { title = "-" })

  -- Speed submenu. Takes effect on the next press, not mid-playback —
  -- afplay's rate is fixed at process start. Check-mark follows the
  -- persisted value so the menu and the synth loop never disagree.
  local function fmtSpeed(s)
    if s == math.floor(s) then
      return string.format("%d×", s)
    end
    return string.format("%g×", s)
  end
  local currentSpeed = readSpeed()
  local speedSubmenu = {}
  for _, opt in ipairs(SPEED_OPTIONS) do
    local isCurrent = math.abs(opt - currentSpeed) < 1e-6
    table.insert(speedSubmenu, {
      title = fmtSpeed(opt),
      checked = isCurrent,
      fn = function() writeSpeed(opt) end,
    })
  end
  table.insert(items, {
    title = "Speed: " .. fmtSpeed(currentSpeed),
    menu = speedSubmenu,
  })

  -- Voice submenu. If the server isn't up, the submenu is disabled with
  -- the current voice as the title — we don't auto-start the server just
  -- to populate a menu. If the server is up but /voices fails for any
  -- reason, fall back to a single-item "<current> (server not responding)"
  -- so users see something instead of a silent failure.
  local currentVoice = readVoice()
  local voiceSubmenu = {}
  if kokoroUp then
    local voices = fetchVoices()
    if voices then
      for _, v in ipairs(voices) do
        local name = v  -- capture for closure
        table.insert(voiceSubmenu, {
          title = name,
          checked = (name == currentVoice),
          fn = function()
            writeVoice(name)
            hs.alert.show("Claudio voice: " .. name, 1.2)
          end,
        })
      end
    else
      table.insert(voiceSubmenu, {
        title = currentVoice .. " (voice list unavailable)",
        disabled = true,
      })
    end
  else
    table.insert(voiceSubmenu, {
      title = currentVoice .. " (server stopped — starts on next play)",
      disabled = true,
    })
  end
  table.insert(items, {
    title = "Voice: " .. currentVoice,
    menu = voiceSubmenu,
  })

  table.insert(items, { title = "-" })

  -- Server lifecycle (manual RAM release). Always clickable — the common
  -- reason you want to hit Stop is that synth is wedged, which is exactly
  -- when `busy` would otherwise be true. An accidental click during a
  -- real synth just kills that job, same as F13.
  if kokoroUp then
    table.insert(items, {
      title = "Stop server (free ~600 MB)",
      fn = function() kokoroServerStop() end,
    })
  else
    table.insert(items, { title = "Server: stopped (starts on next play)", disabled = true })
  end

  -- Live log tail. Opens Terminal.app following play.log and kokoro-server.log
  -- together — useful when debugging synth start-up, playback drop-outs, or
  -- the server's self-heal sweep. Separate window per click; closing the
  -- window kills its tail.
  table.insert(items, {
    title = "Tail logs…",
    fn = function() tailLogs() end,
  })

  table.insert(items, { title = "-" })

  -- Status block
  local statusText = "Status: " .. statusLabel
  if s.preview and s.preview ~= "" then
    statusText = statusText .. '  —  "' .. s.preview .. '"'
  end
  table.insert(items, { title = statusText, disabled = true })

  return items
end

local function render()
  if not menubar then return end
  local s = readState()
  menubar:setTitle(iconFor(s.state))

  local tip = "Claudio — " .. (
    s.state == "play" and "Playing"
      or s.state == "synth" and "Synthesizing"
      or "Idle")
  if s.preview and s.preview ~= "" then
    tip = tip .. '\n"' .. s.preview .. '"'
  end
  tip = tip .. "\nF13 or click to open menu"
  menubar:setTooltip(tip)
end

-- ============================================================
-- Media key interception (Option A: hs.eventtap on systemDefined events)
-- ============================================================
--
-- This intercepts the NX media-key scancodes that AirPods double-tap, the
-- keyboard F8 (play/pause), F9 (next), and F7 (previous) keys generate.
-- Handling them here lets claudio respond even when it is not the frontmost
-- app or registered as a Now Playing source.
--
-- Limitation: without a proper MPNowPlayingInfoCenter registration (Option B),
-- claudio will NOT appear in the Control Center Now Playing widget or on the
-- lock screen. Option B requires a compiled Swift/ObjC helper to call the
-- MediaPlayer framework from a background process. Track that work separately
-- (see the follow-up issue filed alongside this PR).
--
-- Disable by setting CLAUDIO_MEDIA_KEYS=0 in the environment before launching
-- Hammerspoon (e.g. in your shell profile or LaunchAgent plist). When
-- disabled, media keys pass through unmodified to whichever app macOS has
-- chosen as the active media session (Music.app, Spotify, etc.).
local function registerMediaKeys()
  local enabled = os.getenv("CLAUDIO_MEDIA_KEYS")
  if enabled == "0" then
    hs.printf("claudio: media key interception disabled (CLAUDIO_MEDIA_KEYS=0)")
    return nil
  end

  local tap = hs.eventtap.new(
    { hs.eventtap.event.types.systemDefined },
    function(event)
      local data = event:systemKey()
      if not data or not data.down then
        -- Ignore key-up events to prevent double-fires on press+release.
        return false
      end

      if data.key == "PLAY" then
        -- AirPods double-tap and the keyboard play/pause key both map here.
        M.toggle()
        return true  -- swallow: don't pass to Music.app / Spotify
      end
      -- NEXT / PREVIOUS: not intercepted. The system routes them to whichever
      -- app is currently registered as a Now Playing source (Music.app,
      -- Spotify, etc.). If/when claudio grows queue-advance or rewind, add
      -- branches here; until then, returning false lets the keys behave
      -- normally for other apps.
      return false
    end
  )

  tap:start()
  hs.printf("claudio: media key interception active (set CLAUDIO_MEDIA_KEYS=0 to disable)")
  return tap
end

-- ============================================================
-- Lifecycle
-- ============================================================

function M.start()
  if menubar then return end
  hs.fs.mkdir(STATE_DIR)
  menubar = hs.menubar.new()
  -- Expose for rsvp.lua to find the icon's screen position.
  M._menubar = menubar
  menubar:setTitle("▸")
  -- Dropdown menu. Using a function makes the menu content dynamic each open.
  menubar:setMenu(buildMenu)

  -- RSVP → play-stream command hookup. Set once; the callbacks read the
  -- current controlFifoPath, which runPlayScript resets per task. No-op if
  -- no task is running (writeControl bails when controlFifoPath is nil).
  rsvp.on_pause_request  = function() writeControl("PAUSE") end
  rsvp.on_resume_request = function() writeControl("RESUME") end
  rsvp.on_seek_request   = function(delta_ms)
    writeControl("SEEK " .. tostring(delta_ms))
  end
  -- × button in the balloon: same semantics as F13-during-playback — stop
  -- the TTS job (if any) and clear the balloon. Goes through stopPlayback
  -- so userInitiatedStop is set and the dying task's exit-9/143 doesn't
  -- fire a spurious alert.
  rsvp.on_close_request  = function()
    if not stopPlayback("rsvp.on_close_request") then
      -- No job running — just dismiss the balloon (it can be on screen
      -- during the post-playback linger window).
      rsvp.hide()
    end
  end

  watcher = hs.pathwatcher.new(STATE_DIR, function() render() end)
  watcher:start()

  -- Stop playback when the system's default output device changes. This
  -- catches headphones-unplug (macOS reroutes to the laptop speakers),
  -- AirPods disconnect, and the user manually switching output — any of
  -- which usually means the *previous* output was intentional and the
  -- new one probably isn't welcome to hear a chunk of text suddenly read
  -- aloud. Only fires when a job is actually running, so this never
  -- interferes with casual device switching while idle.
  audioWatcher = hs.audiodevice.watcher.setCallback(function(event)
    if event == "dOut" and jobIsRunning() then
      -- Same suppression rule as the × button and F13: this is a
      -- user-initiated-equivalent stop (the system is changing audio
      -- routing, which we treat as the user not wanting Claudio audio
      -- on the new output), so the dying task's exit-9 should not fire
      -- the "Claudio error" alert.
      stopPlayback("audioWatcher.dOut")
      hs.alert.show("Claudio paused — output device changed")
    end
  end)
  hs.audiodevice.watcher.start()

  mediaKeyTap = registerMediaKeys()

  render()
end

function M.stop()
  if menubar then menubar:delete(); menubar = nil; M._menubar = nil end
  if watcher then watcher:stop(); watcher = nil end
  if audioWatcher then
    hs.audiodevice.watcher.stop()
    audioWatcher = nil
  end
  if mediaKeyTap then
    mediaKeyTap:stop()
    mediaKeyTap = nil
  end
end

return M
