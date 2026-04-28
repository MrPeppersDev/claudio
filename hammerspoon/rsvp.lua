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

-- Gap between the anchor (selected text rect) and the balloon edge when
-- positioning in "above the selection" / "below the selection" mode.
local ANCHOR_GAP = 12

-- Screen-edge safety margin when clamping balloon position.
local SCREEN_PAD = 12

-- How long (seconds) to linger after the last word before fade-out.
local LINGER_SECS = 1.25

-- Keyboard scrub increment (milliseconds per ← / → keypress).
local SEEK_DELTA_MS = 2000

-- ── State ─────────────────────────────────────────────────────────────────────

local _webview      = nil    -- hs.webview instance (nil when hidden)
local _lingerTimer  = nil    -- hs.timer: fires after LINGER_SECS to begin fade-out
local _animTimer    = nil    -- hs.timer: used during animate-in
local _pollTimer    = nil    -- hs.timer: polls window.__claudio_q for button clicks
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
M.on_close_request  = nil    -- function()  -- user clicked × button

-- ── Helpers ───────────────────────────────────────────────────────────────────

-- Diagnostic logger: append-only `hs.log` capturing rsvp lifecycle state
-- (show/hide/fadeout/linger arming/POS threshold crosses). Same file as
-- claudio.lua's diagLog so the timeline reads in order. Uses the same
-- STATE_DIR resolution claudio.lua does so the env var contract matches.
local _diagLogPath = (os.getenv("CLAUDIO_STATE_DIR")
                       or os.getenv("CLAUDIO_DIR")
                       or (os.getenv("HOME") .. "/.claude/claudio")) .. "/hs.log"
local function diagLog(msg)
  local t = hs.timer.secondsSinceEpoch()
  local secs = math.floor(t)
  local ms = math.floor((t - secs) * 1000)
  local f = io.open(_diagLogPath, "a")
  if not f then return end
  f:write(string.format("[%s.%03d] [rsvp] %s\n",
    os.date("%H:%M:%S", secs), ms, msg))
  f:close()
end

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

  -- Find the screen that contains the icon's center point.
  -- hs.screen.find() with a geometry rect/point works in Hammerspoon >= 0.9.79.
  -- Wrap in pcall; if it fails (older build) fall through to mainScreen().
  local iconCenterX = iconFrame.x + iconFrame.w / 2
  local screen
  local ok_find, found = pcall(hs.screen.find, hs.geometry.point(iconCenterX, iconFrame.y))
  if ok_find and found then
    screen = found
  else
    -- Fallback: iterate all screens and find the one whose frame contains the point.
    for _, s in ipairs(hs.screen.allScreens()) do
      local r = s:fullFrame()
      if iconCenterX >= r.x and iconCenterX <= r.x + r.w
         and iconFrame.y >= r.y and iconFrame.y <= r.y + r.h then
        screen = s
        break
      end
    end
  end
  screen = screen or hs.screen.mainScreen()
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
-- Use h=2 (not 0) so WKWebView creates a valid backing surface.
local function balloonStartRect(iconFrame, targetRect)
  return {
    x = targetRect.x,
    y = iconFrame.y + iconFrame.h + ICON_GAP,
    w = BALLOON_WIDTH,
    h = 2,
  }
end

-- Pick the screen that best contains a given rect's center point. Falls back
-- to mainScreen() if hs.screen.find isn't available or nothing matches.
local function screenForRect(r)
  local cx = r.x + r.w / 2
  local cy = r.y + r.h / 2
  local ok, found = pcall(hs.screen.find, hs.geometry.point(cx, cy))
  if ok and found then return found end
  for _, s in ipairs(hs.screen.allScreens()) do
    local sf = s:fullFrame()
    if cx >= sf.x and cx <= sf.x + sf.w
       and cy >= sf.y and cy <= sf.y + sf.h then
      return s
    end
  end
  return hs.screen.mainScreen()
end

-- Compute the target balloon rect anchored relative to a screen-space
-- selection bounding rect. Prefers "above the anchor" so the balloon
-- doesn't cover the text the user is still reading; falls through to
-- "below" if above would clip off the top of the screen.
local function balloonTargetRectForAnchor(anchorRect)
  local screen = screenForRect(anchorRect)
  local sf = screen:frame()   -- usable area (excludes menu bar + Dock)

  local bx = anchorRect.x + anchorRect.w / 2 - BALLOON_WIDTH / 2
  local by_above = anchorRect.y - BALLOON_HEIGHT - ANCHOR_GAP
  local by_below = anchorRect.y + anchorRect.h + ANCHOR_GAP

  local by
  if by_above >= sf.y + SCREEN_PAD then
    by = by_above
  else
    by = by_below
  end

  bx = math.max(sf.x + SCREEN_PAD,
                math.min(bx, sf.x + sf.w - BALLOON_WIDTH - SCREEN_PAD))
  by = math.max(sf.y + SCREEN_PAD,
                math.min(by, sf.y + sf.h - BALLOON_HEIGHT - SCREEN_PAD))

  return { x = bx, y = by, w = BALLOON_WIDTH, h = BALLOON_HEIGHT }
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

-- ── Hover-button bridge ─────────────────────────────────────────────────────
--
-- rsvp.html buttons push command strings onto window.__claudio_q. We poll
-- that queue every ~80 ms, drain it, and dispatch. This was the fallback
-- after the rsvp:// URL-scheme approach failed — WKWebView either swallows
-- unknown schemes or fires the navigationCallback too late to intercept.

-- Polling cadence for window.__claudio_q. 200 ms gives "imperceptible"
-- button latency (well under the 300 ms reaction-time threshold) while
-- halving main-thread contention vs. 80 ms — important because POS events
-- from play-stream.py parse on the same main thread, and any contention
-- shows up as audio/RSVP drift.
local POLL_INTERVAL = 0.20

local function dispatchCmd(cmd)
  if cmd == "toggle_pause" then
    if _isPaused then M.resume() else M.pause() end
  elseif cmd == "pause_only" then
    if not _isPaused then M.pause() end
  elseif cmd == "seek_back" then
    if M.on_seek_request then M.on_seek_request(-SEEK_DELTA_MS) end
  elseif cmd == "seek_fwd" then
    if M.on_seek_request then M.on_seek_request(SEEK_DELTA_MS) end
  elseif cmd == "close" then
    if M.on_close_request then
      M.on_close_request()
    else
      M.hide()
    end
  end
end

local function startPolling()
  if _pollTimer then return end
  -- Keep the JS a single expression — WKWebView evaluates in script-tag
  -- context, so a top-level `return` is a syntax error. The trailing
  -- expression is what gets returned to Lua.
  local js = "var q=window.__claudio_q||[];window.__claudio_q=[];JSON.stringify(q)"
  _pollTimer = hs.timer.doEvery(POLL_INTERVAL, function()
    if not _webview then return end
    _webview:evaluateJavaScript(js, function(result, err)
      -- Some HS/WebKit versions hand back a sentinel err table with code=0
      -- on success. Treat as a real error only when we also have no result.
      if err and not result then return end
      if not result then return end
      local s = tostring(result)
      if s == "" or s == "[]" then return end
      local ok, cmds = pcall(hs.json.decode, s)
      if not ok or type(cmds) ~= "table" then return end
      for _, cmd in ipairs(cmds) do
        if type(cmd) == "string" then dispatchCmd(cmd) end
      end
    end)
  end)
end

local function stopPolling()
  if _pollTimer then
    _pollTimer:stop()
    _pollTimer = nil
  end
end

-- Cancel any pending linger timer.
local function cancelLinger()
  if _lingerTimer then
    _lingerTimer:stop()
    _lingerTimer = nil
  end
end

-- Destroy the webview and clean up the eventtap/timers.
local function destroyWebview()
  diagLog(string.format("destroyWebview: webview=%s linger=%s anim=%s",
    tostring(_webview ~= nil), tostring(_lingerTimer ~= nil),
    tostring(_animTimer ~= nil)))
  cancelLinger()
  stopPolling()
  if _animTimer then
    _animTimer:stop()
    _animTimer = nil
  end
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
  diagLog("beginFadeOut: webview=" .. tostring(_webview ~= nil))
  cancelLinger()
  if not _webview then return end
  jsCall("window.rsvpFadeOut")
  -- Wait for CSS transition (~300ms) then destroy.
  hs.timer.doAfter((ANIMATE_OUT_MS + 50) / 1000, function()
    diagLog("beginFadeOut: timer fired → destroyWebview")
    destroyWebview()
  end)
end

-- Space/Esc/←/→ are handled via JS keydown listeners in rsvp.html that push
-- commands into the same __claudio_q queue the hover buttons use. The Lua
-- side picks them up through startPolling()/dispatchCmd. Scope is identical
-- to the buttons — only fires while the webview content has focus, so keys
-- pass through to other apps unchanged when the balloon isn't focused.

-- ── Linger + last-word detection ─────────────────────────────────────────────

-- Called from onPos when we detect that playback has passed the last word.
local function scheduleLingerFadeOut()
  if _lingerTimer then
    diagLog("scheduleLingerFadeOut: already scheduled — no-op")
    return
  end
  diagLog(string.format("scheduleLingerFadeOut: arming for %.2fs", LINGER_SECS))
  _lingerTimer = hs.timer.doAfter(LINGER_SECS, function()
    _lingerTimer = nil
    diagLog("linger timer fired → beginFadeOut")
    beginFadeOut()
  end)
end

-- ── Public API ────────────────────────────────────────────────────────────────

--- show(sidecar_path, opts)
--- Load sidecar JSON, create the webview, animate it open.
--- sidecar_path may be nil (start empty; PR 4 will call feedSidecar before
--- the first POS events arrive).
--- opts is an optional table:
---   opts.anchor = { x, y, w, h }   -- screen-space rect around the user's
---     focused text (selection bounds or mouse cursor). When present, the
---     balloon positions itself just above (or below, if above clips) that
---     rect instead of under the menubar icon.
function M.show(sidecar_path, opts)
  diagLog(string.format(
    "M.show: sidecar_path=%s prior_webview=%s",
    tostring(sidecar_path), tostring(_webview ~= nil)))
  -- If already visible, just reload with new sidecar.
  if _webview then
    M.hide()
  end

  -- Reset per-session state. Note: _bufferSidecars is NOT reset here so that
  -- feedSidecar() can be called before show() (the test harness does this).
  -- destroyWebview() resets _bufferSidecars on hide/cleanup.
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

  -- Decide where to place the balloon. Anchor mode (selection/mouse rect)
  -- pops in at final size with a fade; menubar mode grows from the icon.
  local anchor     = opts and opts.anchor
  local targetRect
  local startRect
  local anchorMode = anchor ~= nil
  if anchorMode then
    targetRect = balloonTargetRectForAnchor(anchor)
    startRect  = targetRect   -- pop-in; CSS handles the fade
  else
    local iconFrame = menubarFrame()
    targetRect = balloonTargetRect(iconFrame)
    startRect  = balloonStartRect(iconFrame, targetRect)
  end

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
  _webview:transparent(true)
  _webview:bringToFront(true)
  -- Required so clicks make the webview content firstResponder; without it
  -- the JS keydown listener for Space/Esc/←/→ never fires.
  if _webview.allowTextEntry then _webview:allowTextEntry(true) end

  -- Fires once when the HTML has finished loading (triggered by the
  -- navigationCallback didFinishNavigation event, or by the safety-net
  -- timer if that event doesn't reach us). Idempotent via _pageReadyFired
  -- so double-fire is a no-op.
  local pageReadyFired = false
  local function onPageReady()
    if pageReadyFired then return end
    pageReadyFired = true
    if not _webview then return end

    -- Feed the top-level sidecar into the page.
    if sidecarJson then
      jsCall("window.rsvpLoad", sidecarJson)
    end

    -- Push any per-buffer sidecars that were fed via feedSidecar() before
    -- show() was called. The test harness does this: feedSidecar then show.
    for bid, sc in pairs(_bufferSidecars) do
      local ok_enc, json_str = pcall(hs.json.encode, sc)
      if ok_enc and json_str then
        jsCall("window.rsvpLoadBuffer", tostring(bid), json_str)
      end
    end

    if anchorMode then
      -- Anchor mode: balloon is already at final rect. Just fade in the
      -- content (CSS transition handles it) and start polling. No grow
      -- animation — a pop-in at cursor/selection position reads as more
      -- deliberate than a bubble unfurling from somewhere else.
      _webview:frame(targetRect)
      jsCall("window.rsvpFadeIn")
      startPolling()
      return
    end

    -- Menubar mode: grow from iconFrame collapsed rect to targetRect over
    -- ANIMATE_IN_MS. hs.webview:setFrameWithAnimation is not available in
    -- all builds; use hs.timer steps for compatibility.
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
        startPolling()
      end
    end

    _animTimer = hs.timer.doAfter(stepSecs, doStep)
  end

  -- Block link clicks / form navigation and watch for the page-load
  -- completion that tells us it's safe to inject JS. Our page is a single
  -- inline-script file:// load, so by the time didFinishNavigation fires
  -- the window.rsvp* functions are defined.
  _webview:navigationCallback(function(action, wv, navID, url)
    if action == "didFinishNavigation" then
      onPageReady()
      return true
    end
    if action == "didStartProvisionalNavigation" then
      return true
    end
    if action == "navigating" then
      -- Our own file:// load is always permitted.
      if url and url:sub(1, 7) == "file://" then return true end
      return false
    end
    return true
  end)

  -- Load the HTML file.
  local url = "file://" .. htmlPath()
  _webview:url(url)

  _webview:show()

  -- Safety net: some Hammerspoon builds don't deliver didFinishNavigation
  -- for file:// URLs. If that happens, fire the ready path ourselves after
  -- a bounded wait. Bigger than the old 150 ms so the navigation callback
  -- is the winning path in the common case; still imperceptible on F13.
  hs.timer.doAfter(0.5, function()
    if not pageReadyFired then onPageReady() end
  end)
end

--- hide()
--- Immediately destroy the balloon without animation.
function M.hide()
  diagLog("M.hide called")
  destroyWebview()
end

--- endOfStream()
--- Called by PR 4 when the play task exits cleanly. The in-stream linger
--- trigger fires only when onPos sees ms >= last_word.end_ms, but
--- play-stream can stop emitting POS slightly before that (tail silence,
--- 400ms pre-roll offset, etc.), so the timer never arms and the balloon
--- sits on-screen forever. This forces the linger-fade path regardless.
function M.endOfStream()
  diagLog(string.format(
    "M.endOfStream: webview=%s last_pos_ms=%s active_buf=%s",
    tostring(_webview ~= nil), tostring(_lastPosMs),
    tostring(_activeBufferId)))
  if not _webview then return end
  scheduleLingerFadeOut()
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
    local n = (parsed.words and #parsed.words) or 0
    local last = (n > 0 and parsed.words[n].end_ms) or -1
    diagLog(string.format(
      "feedSidecar: buf=%s words=%d last_end_ms=%s",
      tostring(buffer_id), n, tostring(last)))
  else
    diagLog(string.format("feedSidecar: parse FAILED for buf=%s",
      tostring(buffer_id)))
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
  -- No early-return on _isPaused. play-stream only emits POS during pause
  -- when a seek has just been applied (the regular per-chunk POS emit is
  -- skipped by the paused branch in _play_buffer), so any POS arriving
  -- here while paused is a seek-induced visual update — exactly what the
  -- user wants forwarded to the JS balloon.

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
      diagLog(string.format(
        "onPos linger trigger: ms=%d >= last_end_ms=%d (buf=%s)",
        ms, lastWord.end_ms, tostring(buffer_id)))
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
