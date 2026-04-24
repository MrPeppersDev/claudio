-- rsvp.lua — RSVP balloon UI for Claudio
--
-- Renders an animated balloon beneath the Claudio menubar icon that flashes
-- one word at a time in sync with Kokoro TTS playback (Rapid Serial Visual
-- Presentation). Audio is authoritative: the balloon follows the audio
-- playhead via POS events, never drives it.
--
-- ── Public API ───────────────────────────────────────────────────────────────
--
--   rsvp.show(sidecar_path)
--       Load sidecar JSON from `sidecar_path`, show the balloon, and begin
--       listening for rsvp.onPos() calls.
--
--   rsvp.hide()
--       Immediately hide and destroy the balloon (no animation).
--
--   rsvp.pause()
--       Stop advancing words. Does NOT hide the balloon. Dims the word and
--       shows a "paused" indicator. Also fires rsvp.on_pause_request if set.
--
--   rsvp.resume()
--       Resume advancing words. Fires rsvp.on_resume_request if set.
--
--   rsvp.onPos(ms, buffer_id)
--       Called by PR 4 on each `POS <ms> <buffer_id>` event from play-stream.
--       Looks up the word at `ms` in the buffer's sidecar and renders it.
--       After `ms` advances past the last word's end_ms, a 2-second linger
--       timer starts; on expiry the balloon fades out and is destroyed.
--
--   rsvp.feedSidecar(buffer_id, sidecar_json)
--       Register the sidecar (as a JSON string) for a given buffer id.
--       Must be called before rsvp.onPos() references that buffer.
--
-- ── Callback hooks (filled in by PR 4) ──────────────────────────────────────
--
--   rsvp.on_pause_request   = function() ... end
--   rsvp.on_resume_request  = function() ... end
--   rsvp.on_seek_request    = function(delta_ms) ... end
--
-- ── How PR 4 wires this in ───────────────────────────────────────────────────
--
--   1. When F13 fires and a job starts, PR 4 calls rsvp.show(sidecar_path)
--      with the first sentence's sidecar (or nil to start empty).
--   2. As play-stream.py emits POS lines, PR 4 parses them and calls
--      rsvp.onPos(ms, buffer_id).
--   3. When a new sentence's buffer starts, PR 4 calls
--      rsvp.feedSidecar(buffer_id, json_string).
--   4. rsvp.on_pause_request / on_resume_request are set to functions that
--      write PAUSE/RESUME to play-stream's fd-9 pipe.
--   5. rsvp.on_seek_request is set to a function that writes SEEK <delta_ms>
--      to fd-9. (PR 4 defines SEEK handling in play-stream.py.)
--
-- ── Dev harness ──────────────────────────────────────────────────────────────
--
--   See rsvp-test.lua for a self-contained test harness.
--   Load it in init.lua alongside claudio.lua, then press Cmd+Alt+R to run.
--   The harness fires a canned 15-word sidecar and drives a simulated POS
--   stream at ~1.54x tempo to exercise the full show → word-flash → fade-out
--   lifecycle without needing the real Kokoro pipeline.

local M = {}

-- ── Tunables ─────────────────────────────────────────────────────────────────

-- Balloon dimensions (pixels)
local BALLOON_WIDTH  = 500
local BALLOON_HEIGHT = 200

-- Animate-in duration (seconds). Ease-out from menubar icon height to full rect.
local ANIMATE_IN_MS   = 200    -- milliseconds, converted below
local ANIMATE_OUT_MS  = 300    -- fade-out duration (CSS handles this)

-- Gap between the bottom of the menubar icon and the top of the balloon (px).
local ICON_GAP = 4

-- How long (seconds) to linger after the last word before fade-out.
local LINGER_SECS = 2.0

-- Keyboard scrub increment (milliseconds per ← / → keypress).
local SEEK_DELTA_MS = 2000

-- ── State ─────────────────────────────────────────────────────────────────────

local _webview      = nil    -- hs.webview instance (nil when hidden)
local _hotkeys      = {}     -- list of hs.hotkey bindings active while balloon is visible
local _lingerTimer  = nil    -- hs.timer: fires after LINGER_SECS to begin fade-out
local _animTimer    = nil    -- hs.timer: used during animate-in
local _htmlPath     = nil    -- absolute path to rsvp.html

-- Per-buffer sidecar cache: buffer_id (number) → parsed sidecar table
local _bufferSidecars = {}

-- Active sidecar (the most recently seen buffer, or the one from show())
local _activeSidecar = nil
local _activeBufferId = nil

-- Pause state
local _isPaused = false

-- Last POS seen (for linger detection)
local _lastPosMs = nil

-- ── Callbacks (filled by PR 4) ────────────────────────────────────────────────

M.on_pause_request  = nil    -- function()
M.on_resume_request = nil    -- function()
M.on_seek_request   = nil    -- function(delta_ms)

-- ── Helpers ───────────────────────────────────────────────────────────────────

-- Resolve the path to rsvp.html relative to this script.
local function htmlPath()
  if _htmlPath then return _htmlPath end
  -- __FILE__ isn't available in all Hammerspoon versions; use hs.configdir.
  local dir = hs.configdir or (os.getenv("HOME") .. "/.hammerspoon")
  _htmlPath = dir .. "/rsvp.html"
  return _htmlPath
end

-- Return the menubar item's screen rect from claudio.lua's `menubar` object.
-- We reach into the module to find the exported menubar. If claudio hasn't
-- started yet, fall back to a safe top-center position on the main screen.
local function menubarFrame()
  -- Try to get the frame from claudio module's menubar field.
  local ok, claudio = pcall(require, "claudio")
  if ok and claudio and claudio._menubar then
    local f = claudio._menubar:frame()
    if f then return f end
  end
  -- Fallback: center of main screen's menu bar area.
  local screen = hs.screen.mainScreen()
  local sf = screen:fullFrame()
  return {
    x = sf.x + sf.w / 2 - 12,
    y = sf.y,
    w = 24,
    h = 22,
  }
end

-- Compute the target rect for the fully-expanded balloon, anchored below
-- the menubar icon on the same screen.
local function balloonTargetRect(iconFrame)
  -- Center the balloon horizontally on the icon.
  local bx = iconFrame.x + iconFrame.w / 2 - BALLOON_WIDTH / 2
  local by = iconFrame.y + iconFrame.h + ICON_GAP

  -- Clamp to the screen containing the icon.
  -- hs.screen.find() accepts a point.
  local screen = hs.screen.find({ x = iconFrame.x + iconFrame.w/2, y = iconFrame.y })
               or hs.screen.mainScreen()
  local sf = screen:frame()  -- usable area (below menu bar)

  -- Clamp horizontal position to screen bounds.
  bx = math.max(sf.x, math.min(bx, sf.x + sf.w - BALLOON_WIDTH))

  return {
    x = bx,
    y = by,
    w = BALLOON_WIDTH,
    h = BALLOON_HEIGHT,
  }
end

-- Compute the starting rect for the animate-in (collapsed at icon height).
local function balloonStartRect(iconFrame, targetRect)
  return {
    x = targetRect.x,
    y = iconFrame.y + iconFrame.h + ICON_GAP,
    w = BALLOON_WIDTH,
    h = 0,
  }
end

-- Call a JS function on the webview (fire-and-forget; errors logged to console).
local function jsCall(fn, ...)
  if not _webview then return end
  local args = {...}
  local parts = {}
  for _, v in ipairs(args) do
    if type(v) == "string" then
      -- Escape for a JS string literal: backslash, then double-quote, then
      -- newlines and carriage returns. This is safe for arbitrary UTF-8 text
      -- and for JSON strings (which only use printable ASCII + UTF-8).
      local escaped = v
        :gsub("\\", "\\\\")
        :gsub('"',  '\\"')
        :gsub("\n", "\\n")
        :gsub("\r", "\\r")
      table.insert(parts, '"' .. escaped .. '"')
    elseif type(v) == "number" then
      table.insert(parts, tostring(v))
    elseif v == nil then
      table.insert(parts, "null")
    else
      table.insert(parts, tostring(v))
    end
  end
  local js = fn .. "(" .. table.concat(parts, ", ") .. ")"
  _webview:evaluateJavaScript(js, function(result, err)
    if err then
      hs.printf("rsvp: JS error in %s: %s", fn, tostring(err))
    end
  end)
end

-- Cancel any pending linger timer.
local function cancelLinger()
  if _lingerTimer then
    _lingerTimer:stop()
    _lingerTimer = nil
  end
end

-- Destroy the webview and clean up all hotkeys/timers.
local function destroyWebview()
  cancelLinger()
  if _animTimer then
    _animTimer:stop()
    _animTimer = nil
  end
  for _, hk in ipairs(_hotkeys) do
    hk:delete()
  end
  _hotkeys = {}
  if _webview then
    _webview:delete()
    _webview = nil
  end
  _isPaused = false
  _lastPosMs = nil
  _activeSidecar = nil
  _activeBufferId = nil
  _bufferSidecars = {}
end

-- Begin the fade-out animation then destroy.
local function beginFadeOut()
  cancelLinger()
  if not _webview then return end
  jsCall("window.rsvpFadeOut")
  -- Wait for CSS transition (~300ms) then destroy.
  hs.timer.doAfter((ANIMATE_OUT_MS + 50) / 1000, function()
    destroyWebview()
  end)
end

-- Register hotkeys that are only active while the balloon is visible.
local function registerHotkeys()
  -- Esc → pause
  table.insert(_hotkeys, hs.hotkey.new({}, "escape", function()
    M.pause()
  end))

  -- Space → resume
  table.insert(_hotkeys, hs.hotkey.new({}, "space", function()
    M.resume()
  end))

  -- ← → scrub
  table.insert(_hotkeys, hs.hotkey.new({}, "left", function()
    if M.on_seek_request then
      M.on_seek_request(-SEEK_DELTA_MS)
    end
  end))

  table.insert(_hotkeys, hs.hotkey.new({}, "right", function()
    if M.on_seek_request then
      M.on_seek_request(SEEK_DELTA_MS)
    end
  end))

  for _, hk in ipairs(_hotkeys) do
    hk:enable()
  end
end

-- ── Linger + last-word detection ─────────────────────────────────────────────

-- Called from onPos when we detect that playback has passed the last word.
local function scheduleLingerFadeOut()
  if _lingerTimer then return end   -- already scheduled
  _lingerTimer = hs.timer.doAfter(LINGER_SECS, function()
    _lingerTimer = nil
    beginFadeOut()
  end)
end

-- ── Public API ────────────────────────────────────────────────────────────────

--- show(sidecar_path)
--- Load sidecar JSON, create the webview, animate it open.
--- sidecar_path may be nil (start empty; PR 4 will call feedSidecar before
--- the first POS events arrive).
function M.show(sidecar_path)
  -- If already visible, just reload with new sidecar.
  if _webview then
    M.hide()
  end

  -- Reset state.
  _bufferSidecars  = {}
  _activeSidecar   = nil
  _activeBufferId  = nil
  _isPaused        = false
  _lastPosMs       = nil

  -- Load sidecar if provided.
  local sidecarJson = nil
  if sidecar_path then
    local f = io.open(sidecar_path, "r")
    if f then
      sidecarJson = f:read("*a")
      f:close()
    else
      hs.printf("rsvp: could not open sidecar: %s", sidecar_path)
    end
  end

  -- Determine icon position.
  local iconFrame  = menubarFrame()
  local targetRect = balloonTargetRect(iconFrame)
  local startRect  = balloonStartRect(iconFrame, targetRect)

  -- Create webview starting at collapsed height.
  local wm = hs.webview.windowMasks
  _webview = hs.webview.new(startRect, {
    developerExtrasEnabled = false,
  })
  if not _webview then
    hs.printf("rsvp: failed to create webview")
    return
  end

  _webview:windowStyle(wm.borderless)
  _webview:allowTextEntry(false)
  _webview:transparent(true)
  _webview:bringToFront(true)
  _webview:allowNewWindows(false)
  _webview:allowNavigationFrom(nil)

  -- Load the HTML file.
  local url = "file://" .. htmlPath()
  _webview:url(url)

  _webview:show()

  -- Once the page is loaded, inject sidecar and fade in.
  -- hs.webview doesn't expose a reliable onload callback in all versions,
  -- so we use a short timer to let the page settle before injecting JS.
  hs.timer.doAfter(0.15, function()
    if not _webview then return end

    -- Feed the sidecar into the page.
    if sidecarJson then
      jsCall("window.rsvpLoad", sidecarJson)
    end

    -- Animate open: grow from startRect to targetRect over ANIMATE_IN_MS.
    -- hs.webview:setFrameWithAnimation is not available in all builds;
    -- use hs.animate via hs.timer steps for compatibility.
    local steps     = 10
    local stepSecs  = (ANIMATE_IN_MS / 1000) / steps
    local step      = 0

    -- Ease-out cubic: t → 1 - (1-t)^3
    local function easeOut(t) return 1 - (1 - t)^3 end

    local function doStep()
      if not _webview then return end
      step = step + 1
      local t = easeOut(step / steps)
      local r = {
        x = startRect.x,
        y = startRect.y,
        w = targetRect.w,
        h = startRect.h + (targetRect.h - startRect.h) * t,
      }
      _webview:frame(r)
      if step < steps then
        _animTimer = hs.timer.doAfter(stepSecs, doStep)
      else
        -- Snap to final rect and fade in the content.
        _webview:frame(targetRect)
        _animTimer = nil
        jsCall("window.rsvpFadeIn")
        -- Register hotkeys now that balloon is fully visible.
        registerHotkeys()
      end
    end

    _animTimer = hs.timer.doAfter(stepSecs, doStep)
  end)
end

--- hide()
--- Immediately destroy the balloon without animation.
function M.hide()
  destroyWebview()
end

--- pause()
--- Stop advancing words; show the pause indicator.
function M.pause()
  if not _webview then return end
  _isPaused = true
  jsCall("window.rsvpPause")
  if M.on_pause_request then
    M.on_pause_request()
  end
end

--- resume()
--- Resume advancing words; remove the pause indicator.
function M.resume()
  if not _webview then return end
  _isPaused = false
  jsCall("window.rsvpResume")
  if M.on_resume_request then
    M.on_resume_request()
  end
end

--- feedSidecar(buffer_id, sidecar_json)
--- Register a sidecar JSON string for a given buffer id.
--- PR 4 calls this as each sentence's sidecar becomes available.
function M.feedSidecar(buffer_id, sidecar_json)
  if not sidecar_json then return end
  -- Cache the parsed table locally for linger detection.
  local ok_parse, parsed = pcall(function()
    -- Minimal parse: we just need to know the last word's end_ms.
    -- A full JSON parser isn't available in vanilla Hammerspoon Lua;
    -- use hs.json which is available in Hammerspoon >= 0.9.57.
    return hs.json.decode(sidecar_json)
  end)
  if ok_parse and parsed then
    _bufferSidecars[buffer_id] = parsed
  end

  -- Also push to the webview JS side.
  if _webview then
    -- We pass the raw JSON string; rsvpLoadBuffer does its own JSON.parse.
    jsCall("window.rsvpLoadBuffer", tostring(buffer_id), sidecar_json)
  end
end

--- onPos(ms, buffer_id)
--- Called by PR 4 on each POS event from play-stream.py.
--- Advances the word display.
function M.onPos(ms, buffer_id)
  if not _webview then return end
  if _isPaused then return end

  _lastPosMs = ms

  -- Determine the active sidecar table for linger detection.
  local sidecar = nil
  if buffer_id ~= nil and _bufferSidecars[buffer_id] then
    sidecar = _bufferSidecars[buffer_id]
    _activeBufferId = buffer_id
  elseif _activeSidecar then
    sidecar = _activeSidecar
  end

  -- Check if we've passed the end of all words (linger trigger).
  if sidecar and sidecar.words and #sidecar.words > 0 then
    local lastWord = sidecar.words[#sidecar.words]
    if ms >= lastWord.end_ms then
      scheduleLingerFadeOut()
      return
    end
    -- Cancel any pending linger if the pos moved back (e.g., a seek).
    cancelLinger()
  end

  -- Tell the webview to advance.
  if buffer_id ~= nil then
    jsCall("window.rsvpShowAt", ms, buffer_id)
  else
    jsCall("window.rsvpShowAt", ms)
  end
end

-- Expose the module for test harness and for PR 4 wiring.
-- PR 4 can also set M.on_pause_request, M.on_resume_request, M.on_seek_request.

-- Internal accessor for claudio.lua to expose menubar frame.
-- claudio.lua should export:  M._menubar = menubar  (the hs.menubar object)
-- This lets rsvp.lua find the icon position without tight coupling.

return M
