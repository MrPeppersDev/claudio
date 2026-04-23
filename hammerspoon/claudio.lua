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

local HOME = os.getenv("HOME")
-- CLAUDIO_DIR = repo (scripts, venv, model files). Defaults to the install
-- location that install.sh created, but overridable so the repo can live
-- anywhere. STATE_DIR = runtime files (state, job.pid). Users who set
-- either env var in a LaunchAgent or before launching Hammerspoon get the
-- override; plain Dock launches pick up the defaults.
local CLAUDIO_DIR = os.getenv("CLAUDIO_DIR") or (HOME .. "/.claude/claudio")
local STATE_DIR = os.getenv("CLAUDIO_STATE_DIR") or CLAUDIO_DIR
local STATE_FILE = STATE_DIR .. "/state"
local PLAY_SCRIPT = CLAUDIO_DIR .. "/play-last.sh"
local SERVER_SCRIPT = CLAUDIO_DIR .. "/kokoro-server.sh"

local menubar = nil
local watcher = nil
-- Tracks the pasteboard changeCount at the time of the last toggle. Used by
-- the copy-on-select terminal fallback to distinguish "user just selected
-- new text" (changeCount advanced since last toggle) from "stale clipboard
-- from minutes ago" (no change since last toggle).
local lastSeenChangeCount = nil

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

-- ============================================================
-- Script runner (async — don't block Hammerspoon on synthesis)
-- ============================================================

local function runPlayScript(args)
  hs.task.new("/bin/bash",
    function(code)
      if code ~= 0 and code ~= 143 then -- 143 = SIGTERM on stop; not an error
        hs.alert.show("Claudio error (exit " .. code .. ")")
      end
    end,
    args
  ):start()
end

-- ============================================================
-- Selection capture (AX → Cmd+C → copy-on-select terminal fallback)
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
    print(string.format("[claudio] AX selection: %d chars (front=%s)", #ax, frontName))
    cb(ax)
    return
  end

  local prevCount = hs.pasteboard.changeCount()
  local prevContents = hs.pasteboard.getContents()
  hs.eventtap.keyStroke({ "cmd" }, "c", 0)

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
      lastSeenChangeCount = newCount
      cb(newContents)
      return
    end

    lastSeenChangeCount = newCount
    cb(nil)
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

function M.toggle()
  if jobIsRunning() then
    runPlayScript({ PLAY_SCRIPT })
    return
  end

  captureSelection(function(selection)
    if selection then
      -- mktemp creates the file mode-0600 on macOS, so the selection —
      -- which may contain tokens, private chat text, or PII — isn't
      -- world-readable in ~/.claude/claudio/. play-last.sh unlinks the
      -- file immediately after reading it, so the window of exposure is
      -- only the synth lifetime rather than until-next-F13.
      local tmp = hs.execute("mktemp '" .. STATE_DIR .. "/selection.XXXXXX'", true)
      tmp = tmp and tmp:gsub("%s+$", "") or nil
      if not tmp or tmp == "" then
        hs.alert.show("Claudio: mktemp failed")
        return
      end
      local f = io.open(tmp, "w")
      if f then f:write(selection); f:close() end
      runPlayScript({ PLAY_SCRIPT, "--text-file", tmp })
    else
      hs.alert.show("Claudio: highlight text to play")
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
  local s = readState()
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

  -- Server lifecycle (manual RAM release). Disabled when busy so you can't
  -- kill the server mid-synth.
  if kokoroUp then
    table.insert(items, {
      title = "Stop server (free ~600 MB)",
      disabled = busy,
      fn = function() kokoroServerStop() end,
    })
  else
    table.insert(items, { title = "Server: stopped (starts on next play)", disabled = true })
  end

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
-- Lifecycle
-- ============================================================

function M.start()
  if menubar then return end
  hs.fs.mkdir(STATE_DIR)
  menubar = hs.menubar.new()
  menubar:setTitle("▸")
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
