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
local SPEED_FILE = STATE_DIR .. "/speed"
local VOICE_FILE = STATE_DIR .. "/voice"
local PLAY_LOG = STATE_DIR .. "/play.log"
local SERVER_LOG = STATE_DIR .. "/kokoro-server.log"
local QUEUE_DIR = STATE_DIR .. "/queue"
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

-- Opens Terminal.app running `tail -F` on both Claudio log files. Terminal is
-- shipped with macOS so this works on a vanilla install without requiring the
-- user to have iTerm / kitty / etc. `touch` first so `tail -F` doesn't spin
-- on a fresh install where no log exists yet; `-F` (capital) keeps following
-- across rotate/recreate, which `kokoro-server.sh stop; start` will cause.
local function tailLogs()
  hs.execute(string.format("touch '%s' '%s' 2>/dev/null", PLAY_LOG, SERVER_LOG), true)
  local cmd = string.format("tail -F '%s' '%s'", PLAY_LOG, SERVER_LOG)
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

function M.toggle()
  if jobIsRunning() then
    runPlayScript({ PLAY_SCRIPT })
    return
  end

  captureSelection(function(selection, hint)
    if selection then
      local tmp = selectionToTempfile(selection)
      if tmp then
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

      elseif data.key == "NEXT" then
        -- Future: skip to next queued item. Guard: only act when a job is
        -- running so the key still works for other apps when claudio is idle.
        if jobIsRunning() then
          -- Placeholder — queue-advance not yet implemented.
          -- TODO: call M.queueNext() when that function exists.
          return false
        end
        return false

      elseif data.key == "PREVIOUS" then
        -- Same guard: only intercept during active playback.
        if jobIsRunning() then
          -- Placeholder — rewind/restart not yet implemented.
          return false
        end
        return false
      end

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
      runPlayScript({ PLAY_SCRIPT })
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
