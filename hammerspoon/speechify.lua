-- speechify.lua — menu bar dropdown for Speechify / Kokoro TTS.
--
-- State source: ~/.claude/speechify/state (written by play-last.sh / *-tts.sh).
--   state=<idle|synth|play>
--   preview=<first 80 chars of the message being read>
--   ts=<unix seconds>
--
-- Backend: ~/.claude/speechify/backend (contains "speechify" or "kokoro").
--   Missing file = default "speechify". Changed via the dropdown menu.
--
-- Icon: ▸ idle, ⟳ synth, ▶ play. Suffix shows backend: "▸·s" (speechify)
-- or "▸·k" (kokoro) so you know at a glance which one F13 will hit.
--
-- Menu: Play/Stop toggle + backend radio + live status. Selecting a backend
-- stops any in-flight job AND starts/stops the Kokoro server so only the
-- active backend is consuming resources.
--
-- F13 still triggers the same toggle as the top menu item.

local M = {}

local HOME = os.getenv("HOME")
local STATE_DIR = HOME .. "/.claude/speechify"
local STATE_FILE = STATE_DIR .. "/state"
local BACKEND_FILE = STATE_DIR .. "/backend"
local PLAY_SCRIPT = STATE_DIR .. "/play-last.sh"
local SERVER_SCRIPT = STATE_DIR .. "/kokoro-server.sh"

local menubar = nil
local watcher = nil
-- Tracks the pasteboard changeCount at the time of the last toggle. Used by
-- the iTerm copy-on-select fallback to distinguish "user just selected new
-- text" (changeCount advanced since last toggle) from "stale clipboard from
-- minutes ago" (no change since last toggle).
local lastSeenChangeCount = nil

-- ============================================================
-- State / backend file helpers
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

local function readBackend()
  local f = io.open(BACKEND_FILE, "r")
  if not f then return "speechify" end
  local v = (f:read("*a") or ""):gsub("%s+", "")
  f:close()
  if v == "kokoro" or v == "speechify" then return v end
  return "speechify"
end

local function writeBackend(name)
  local f = io.open(BACKEND_FILE, "w")
  if not f then return false end
  f:write(name)
  f:close()
  return true
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

-- Start/stop run synchronously (via hs.execute) so menu-driven state changes
-- settle before the user can trigger F13. Each command is ~1s; Hammerspoon's
-- brief pause is a fair trade for not racing the server boot.
local function kokoroServerStart()
  hs.execute(SERVER_SCRIPT .. " start >/dev/null 2>&1", true)
end
local function kokoroServerStop()
  hs.execute(SERVER_SCRIPT .. " stop >/dev/null 2>&1", true)
end

-- stopActiveJob: used before switching backends. Uses play-last.sh's toggle
-- semantics — if a job is running, re-running it kills the tree.
local function stopActiveJob()
  if jobIsRunning() then
    hs.execute(PLAY_SCRIPT .. " >/dev/null 2>&1 &", true)
    -- give the kill a moment to propagate
    hs.timer.usleep(200000)
  end
end

-- ============================================================
-- Script runner (async — don't block Hammerspoon on synthesis)
-- ============================================================

local function runPlayScript(args)
  hs.task.new("/bin/bash",
    function(code)
      if code ~= 0 and code ~= 143 then -- 143 = SIGTERM on stop; not an error
        hs.alert.show("Speechify error (exit " .. code .. ")")
      end
    end,
    args
  ):start()
end

-- ============================================================
-- Selection capture (AX → Cmd+C → iTerm copy-on-select)
-- ============================================================

local function axSelection()
  local okCall, sys = pcall(hs.axuielement.systemWideElement)
  if not okCall or not sys then return nil end
  local focused = sys:attributeValue("AXFocusedUIElement")
  if not focused then return nil end
  local text = focused:attributeValue("AXSelectedText")
  if type(text) == "string" and #text > 0 then return text end
  return nil
end

local function captureSelection(cb)
  local frontApp = hs.application.frontmostApplication()
  local frontName = frontApp and frontApp:name() or "?"

  local ax = axSelection()
  if ax then
    print(string.format("[speechify] AX selection: %d chars (front=%s)", #ax, frontName))
    cb(ax)
    return
  end

  local prevCount = hs.pasteboard.changeCount()
  local prevContents = hs.pasteboard.getContents()
  hs.eventtap.keyStroke({ "cmd" }, "c", 0)
  hs.timer.doAfter(0.18, function()
    local newCount = hs.pasteboard.changeCount()
    local newContents = hs.pasteboard.getContents()

    if newCount > prevCount and newContents and #newContents > 0 then
      if prevContents ~= nil then hs.pasteboard.setContents(prevContents) end
      cb(newContents)
      return
    end

    local isTerminal = (frontName == "iTerm2" or frontName == "iTerm"
                        or frontName == "Terminal")
    local baseline = lastSeenChangeCount or prevCount
    if isTerminal and newContents and #newContents > 0
        and newCount > baseline then
      lastSeenChangeCount = newCount
      cb(newContents)
      return
    end

    lastSeenChangeCount = newCount
    cb(nil)
  end)
end

local function focusedItermSessionId()
  local okCall, id = hs.osascript.applescript([[
    tell application "System Events"
      if not (exists (processes where name is "iTerm2")) then return ""
    end tell
    tell application "iTerm2"
      try
        return unique id of current session of current window
      on error
        return ""
      end try
    end tell
  ]])
  if okCall and type(id) == "string" and id ~= "" then return id end
  return nil
end

-- ============================================================
-- Toggle (same entry point from menu item and F13)
-- ============================================================

function M.toggle()
  if jobIsRunning() then
    runPlayScript({ PLAY_SCRIPT })
    return
  end

  captureSelection(function(selection)
    if selection then
      local tmp = STATE_DIR .. "/selection.txt"
      local f = io.open(tmp, "w")
      if f then f:write(selection); f:close() end
      runPlayScript({ PLAY_SCRIPT, "--text-file", tmp })
    else
      local args = { PLAY_SCRIPT }
      local sid = focusedItermSessionId()
      if sid then table.insert(args, "--session"); table.insert(args, sid) end
      runPlayScript(args)
    end
  end)
end

-- ============================================================
-- Backend hot-swap
-- ============================================================
-- Contract: after switching, only the newly-selected backend has resources
-- in use. Order matters: stop the current job BEFORE changing the backend
-- file (so the stop hits the right backend), then start/stop the server
-- synchronously so the next F13 can't race.

local function setBackend(name)
  local current = readBackend()
  if current == name then return end

  stopActiveJob()
  writeBackend(name)

  if name == "kokoro" then
    kokoroServerStart()
  else
    kokoroServerStop()
  end
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
  local s = readState()
  local backend = readBackend()
  local busy = (s.state == "play" or s.state == "synth")
  local kokoroUp = kokoroServerRunning()

  local statusLabel
  if s.state == "play" then statusLabel = "Playing"
  elseif s.state == "synth" then statusLabel = "Synthesizing"
  else statusLabel = "Idle" end

  local items = {}

  -- Play / Stop toggle
  table.insert(items, {
    title = busy and ("■ Stop") or ("▶ Play last message"),
    fn = function() M.toggle() end,
  })

  table.insert(items, { title = "-" })

  -- Backend radio
  table.insert(items, { title = "Backend", disabled = true })
  table.insert(items, {
    title = "Speechify (cloud)",
    checked = (backend == "speechify"),
    fn = function() setBackend("speechify") end,
  })
  table.insert(items, {
    title = "Kokoro (local, offline)",
    checked = (backend == "kokoro"),
    fn = function() setBackend("kokoro") end,
  })

  table.insert(items, { title = "-" })

  -- Status block
  local statusText = "Status: " .. statusLabel
  if s.preview and s.preview ~= "" then
    statusText = statusText .. '  —  "' .. s.preview .. '"'
  end
  table.insert(items, { title = statusText, disabled = true })

  local serverLine
  if backend == "kokoro" then
    serverLine = "Kokoro server: " .. (kokoroUp and "running" or "stopped")
  else
    serverLine = "Kokoro server: " .. (kokoroUp and "running (switching off)" or "stopped")
  end
  table.insert(items, { title = serverLine, disabled = true })

  return items
end

local function render()
  if not menubar then return end
  local s = readState()
  local backend = readBackend()

  local title = iconFor(s.state) .. "·" .. (backend == "kokoro" and "k" or "s")
  menubar:setTitle(title)

  local tip = string.format("Speechify — %s (backend: %s)",
    s.state == "play" and "Playing"
      or s.state == "synth" and "Synthesizing"
      or "Idle",
    backend)
  if s.preview and s.preview ~= "" then
    tip = tip .. '\n"' .. s.preview .. '"'
  end
  tip = tip .. "\nF13 or click to open menu"
  menubar:setTooltip(tip)
end

-- ============================================================
-- Lifecycle
-- ============================================================

function M.start()
  if menubar then return end
  hs.fs.mkdir(STATE_DIR)
  menubar = hs.menubar.new()
  menubar:setTitle("▸·s")
  -- Dropdown menu. Using a function makes the menu content dynamic each open.
  menubar:setMenu(buildMenu)

  watcher = hs.pathwatcher.new(STATE_DIR, function() render() end)
  watcher:start()

  render()
end

function M.stop()
  if menubar then menubar:delete(); menubar = nil end
  if watcher then watcher:stop(); watcher = nil end
end

return M
