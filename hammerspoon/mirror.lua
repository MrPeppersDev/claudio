-- mirror.lua — structure-mirror webview for #163 PR 4 (two-bubble
-- architecture, fallback surface for non-AX-supported apps).
--
-- The mirror runs ALONGSIDE the RSVP balloon (rsvp.lua), not instead of.
-- RSVP gives speed-flash; mirror gives spatial context — the words
-- being spoken accumulate in a scrollable transcript with karaoke
-- highlight on the current word. Audio is authoritative: the mirror
-- follows POS events the same way RSVP does, never drives playback.
--
-- ── v1 scope (this file) ────────────────────────────────────────────
--
-- Streaming-words display: each arriving sidecar appends its words to
-- the transcript; POS events highlight the matching word; auto-scroll
-- keeps the current word in the upper third. Plain text only — no
-- markdown structure (paragraphs / code blocks / tables) yet. Those
-- are follow-up PRs that build on this plumbing.
--
-- ── Public API ──────────────────────────────────────────────────────
--
--   mirror.show(opts)
--     Create the webview. opts.anchor = {x, y, w, h} optional anchor
--     rect for positioning; otherwise the mirror appears in a sensible
--     default location near the menubar.
--
--   mirror.hide()
--     Destroy immediately (no fade). Used by F13 stop / SIGTERM paths.
--
--   mirror.feedSidecar(buffer_id, sidecar_json)
--     Register a sidecar (parsed table or JSON string). Words append
--     to the transcript with data-buf / data-idx tags so highlight()
--     can target them.
--
--   mirror.onPos(ms, buffer_id)
--     POS event from play-stream. Walks the buffer's sidecar to find
--     the current word; calls window.mirror.highlight in the webview.
--
--   mirror.endOfStream()
--     Trigger the linger-fade. Idempotent (matches rsvp.endOfStream's
--     contract).
--
-- ── Module gating ───────────────────────────────────────────────────
--
-- KOKORO_MIRROR=1 enables the mirror on top of RSVP. Default OFF for
-- v1 — opt-in until validated against real selections, then flip to
-- on-by-default in a follow-up. The opt-in gate lives in claudio.lua
-- (the only caller); this module itself is always loadable.

local M = {}

-- ── Tunables ─────────────────────────────────────────────────────────

local BUBBLE_WIDTH  = 540
local BUBBLE_HEIGHT = 360

-- How long (seconds) to linger after endOfStream before fading. Matches
-- rsvp.lua's LINGER_SECS so both bubbles disappear in sync.
local LINGER_SECS = 1.25
local FADE_OUT_MS = 300

-- 3px movement threshold for drag (matches rsvp.lua).
local DRAG_THRESHOLD_PX = 3

-- ── State ────────────────────────────────────────────────────────────

local _webview      = nil    -- hs.webview instance (nil when hidden)
local _animTimer    = nil    -- hs.timer for any animations
local _lingerTimer  = nil    -- hs.timer scheduled by endOfStream
local _fadeTimer    = nil    -- hs.timer follow-up after fadeOut JS call

-- Per-buffer sidecar cache: buffer_id (number) → parsed sidecar table
-- (so re-rendering a word's highlight doesn't require re-parsing JSON).
local _bufferSidecars = {}

-- Cached structured text from feedText. Held at module scope so that if
-- feedText fires BEFORE the webview's onPageReady (rare but possible
-- when the file IO race goes the wrong way), the page-ready callback
-- can replay it. Cleared in destroyWebview.
local _structuredText = nil

-- The buffers we've already pushed into the webview, so feedSidecar
-- doesn't duplicate them on a second call.
local _pushed = {}

-- Last (buf_id, word_idx) we asked the webview to highlight. Avoids
-- redundant evaluateJavaScript round-trips at the 4Hz POS rate.
local _lastHighlightBuf = nil
local _lastHighlightIdx = nil

-- Drag tap state (mirrors rsvp.lua's pattern).
local _dragTap = nil
local _dragState = nil

-- ── Diagnostic logger ────────────────────────────────────────────────

local _diagLogPath = (os.getenv("CLAUDIO_STATE_DIR")
                       or os.getenv("CLAUDIO_DIR")
                       or (os.getenv("HOME") .. "/.claude/claudio")) .. "/hs.log"
local function diagLog(msg)
  local t = hs.timer.secondsSinceEpoch()
  local secs = math.floor(t)
  local ms = math.floor((t - secs) * 1000)
  local f = io.open(_diagLogPath, "a")
  if not f then return end
  f:write(string.format("[%s.%03d] [mirror] %s\n",
    os.date("%H:%M:%S", secs), ms, msg))
  f:close()
end

-- ── Helpers ──────────────────────────────────────────────────────────

local function htmlPath()
  local dir = hs.configdir or (os.getenv("HOME") .. "/.hammerspoon")
  return dir .. "/mirror.html"
end

-- Default frame: lower-right of the focused screen, with margins.
-- Anchor mode (caller passes opts.anchor) positions to the right of
-- the anchor rect when there's room, or under it as a fallback.
local function defaultFrame(anchor)
  local screen = hs.screen.mainScreen()
  if hs.window.focusedWindow() then
    local fw = hs.window.focusedWindow()
    if fw and fw:screen() then screen = fw:screen() end
  end
  local sf = screen:frame()

  if anchor and type(anchor) == "table"
     and anchor.x and anchor.y and anchor.w and anchor.h then
    -- Anchor mode: place to the right of the anchor with a 12px gap.
    -- Fall back to under-anchor or screen-default if right doesn't fit.
    local gap = 12
    local rightX = anchor.x + anchor.w + gap
    if rightX + BUBBLE_WIDTH < sf.x + sf.w then
      return {
        x = rightX,
        y = math.max(sf.y + 12,
                     math.min(anchor.y, sf.y + sf.h - BUBBLE_HEIGHT - 12)),
        w = BUBBLE_WIDTH,
        h = BUBBLE_HEIGHT,
      }
    end
    -- Right doesn't fit; try below.
    local belowY = anchor.y + anchor.h + gap
    if belowY + BUBBLE_HEIGHT < sf.y + sf.h then
      return {
        x = math.max(sf.x + 12,
                     math.min(anchor.x, sf.x + sf.w - BUBBLE_WIDTH - 12)),
        y = belowY,
        w = BUBBLE_WIDTH,
        h = BUBBLE_HEIGHT,
      }
    end
  end

  -- Default: lower-right corner of the focused screen.
  return {
    x = sf.x + sf.w - BUBBLE_WIDTH - 24,
    y = sf.y + sf.h - BUBBLE_HEIGHT - 24,
    w = BUBBLE_WIDTH,
    h = BUBBLE_HEIGHT,
  }
end

-- JS call wrapper — returns nil silently on errors. Mirrors rsvp.lua's
-- jsCall pattern (we don't need the result, just fire-and-forget).
local function jsCall(funcName, ...)
  if not _webview then return end
  local args = {...}
  local jsArgs = {}
  for _, a in ipairs(args) do
    if type(a) == "string" then
      table.insert(jsArgs, string.format("%q", a))
    else
      table.insert(jsArgs, tostring(a))
    end
  end
  local js = string.format("%s(%s)", funcName, table.concat(jsArgs, ","))
  _webview:evaluateJavaScript(js)
end

-- ── Drag tap (lifted-and-adapted from rsvp.lua) ──────────────────────

local function startDragTap()
  if _dragTap then return end
  _dragTap = hs.eventtap.new({
    hs.eventtap.event.types.leftMouseDown,
    hs.eventtap.event.types.leftMouseDragged,
    hs.eventtap.event.types.leftMouseUp,
  }, function(event)
    if not _webview then return false end
    local etype = event:getType()
    local types = hs.eventtap.event.types

    if etype == types.leftMouseDown then
      local pt = event:location()
      local frame = _webview:frame()
      local inside = pt.x >= frame.x and pt.x < frame.x + frame.w
                     and pt.y >= frame.y and pt.y < frame.y + frame.h
      if inside then
        _dragState = {
          mx0 = pt.x, my0 = pt.y,
          fx0 = frame.x, fy0 = frame.y,
          dragging = false,
        }
      else
        _dragState = nil
      end
      return false
    end

    if etype == types.leftMouseDragged then
      if not _dragState then return false end
      local pt = event:location()
      local dx = pt.x - _dragState.mx0
      local dy = pt.y - _dragState.my0
      if not _dragState.dragging then
        if math.abs(dx) < DRAG_THRESHOLD_PX
           and math.abs(dy) < DRAG_THRESHOLD_PX then
          return false
        end
        _dragState.dragging = true
      end
      _webview:topLeft({
        x = _dragState.fx0 + dx,
        y = _dragState.fy0 + dy,
      })
      return true
    end

    if etype == types.leftMouseUp then
      _dragState = nil
      return false
    end

    return false
  end)
  _dragTap:start()
end

local function stopDragTap()
  if _dragTap then
    _dragTap:stop()
    _dragTap = nil
  end
  _dragState = nil
end

-- ── Lifecycle ────────────────────────────────────────────────────────

local function destroyWebview()
  diagLog(string.format(
    "destroyWebview: webview=%s linger=%s anim=%s fade=%s",
    tostring(_webview ~= nil),
    tostring(_lingerTimer ~= nil),
    tostring(_animTimer ~= nil),
    tostring(_fadeTimer ~= nil)))
  if _lingerTimer then _lingerTimer:stop(); _lingerTimer = nil end
  if _animTimer   then _animTimer:stop();   _animTimer   = nil end
  if _fadeTimer   then _fadeTimer:stop();   _fadeTimer   = nil end
  stopDragTap()
  if _webview then
    _webview:delete()
    _webview = nil
  end
  _bufferSidecars = {}
  _pushed = {}
  _lastHighlightBuf = nil
  _lastHighlightIdx = nil
  _structuredText = nil
end

local function beginFadeOut()
  diagLog("beginFadeOut: webview=" .. tostring(_webview ~= nil))
  if _lingerTimer then _lingerTimer:stop(); _lingerTimer = nil end
  if not _webview then return end
  jsCall("window.mirror.fadeOut")
  _fadeTimer = hs.timer.doAfter((FADE_OUT_MS + 50) / 1000, function()
    _fadeTimer = nil
    diagLog("beginFadeOut: timer fired → destroyWebview")
    destroyWebview()
  end)
end

local function scheduleLinger()
  if _lingerTimer then
    diagLog("scheduleLinger: already scheduled — no-op")
    return
  end
  diagLog(string.format("scheduleLinger: arming for %.2fs", LINGER_SECS))
  _lingerTimer = hs.timer.doAfter(LINGER_SECS, function()
    _lingerTimer = nil
    diagLog("linger timer fired → beginFadeOut")
    beginFadeOut()
  end)
end

-- Push a sidecar's words into the webview. Idempotent — _pushed tracks
-- which buffers we've already announced.
local function pushSidecarToWebview(buffer_id, parsed)
  if _pushed[buffer_id] then return end
  if not parsed or not parsed.words or #parsed.words == 0 then return end
  -- Encode words as JSON (just the {text, start_ms, end_ms} fields).
  local trimmed = {}
  for _, w in ipairs(parsed.words) do
    table.insert(trimmed, {
      text = w.text,
      start_ms = w.start_ms,
      end_ms = w.end_ms,
    })
  end
  local ok_enc, json_str = pcall(hs.json.encode, trimmed)
  if not ok_enc or not json_str then
    diagLog(string.format("pushSidecarToWebview: json encode FAILED for buf=%s",
                          tostring(buffer_id)))
    return
  end
  jsCall("window.mirror.appendWords", tostring(buffer_id), json_str)
  _pushed[buffer_id] = true
end

-- ── Public API ───────────────────────────────────────────────────────

function M.show(opts)
  opts = opts or {}
  diagLog(string.format(
    "M.show: prior_webview=%s anchor=%s",
    tostring(_webview ~= nil), tostring(opts.anchor ~= nil)))

  if _webview then
    -- Already showing — replace.
    destroyWebview()
  end

  local frame = defaultFrame(opts.anchor)

  local wm = hs.webview.windowMasks
  _webview = hs.webview.new(frame, {
    developerExtrasEnabled = false,
  })
  if not _webview then
    diagLog("M.show: failed to create webview")
    return
  end
  _webview:windowStyle(wm.borderless)
  _webview:transparent(true)
  _webview:bringToFront(true)
  -- Required so clicks make the webview firstResponder for any future
  -- keyboard interactions inside it.
  if _webview.allowTextEntry then _webview:allowTextEntry(true) end

  -- Page-ready: navigationCallback fires didFinishNavigation when the
  -- HTML has fully loaded. Push any sidecars that arrived BEFORE show
  -- (uncommon but possible if Lua is fast enough).
  local pageReadyFired = false
  local function onPageReady()
    if pageReadyFired then return end
    pageReadyFired = true
    if not _webview then return end

    -- Replay cached structured text first (so words exist BEFORE we
    -- backfill sidecars — sidecars in v2 mode just tag pre-rendered
    -- spans; if no text was cached, the v1 streaming-append fallback
    -- kicks in inside the JS).
    if _structuredText then
      local ok_enc, json_str = pcall(hs.json.encode, _structuredText)
      if ok_enc and json_str then
        _webview:evaluateJavaScript(
          "window.mirror.renderText(" .. json_str .. ")")
      end
    end

    -- Backfill any sidecars that arrived before page-ready.
    for bid, sc in pairs(_bufferSidecars) do
      pushSidecarToWebview(bid, sc)
    end

    jsCall("window.mirror.fadeIn")
    startDragTap()
  end

  _webview:navigationCallback(function(action, _wv, _navID, url)
    if action == "didFinishNavigation" then
      onPageReady()
      return true
    end
    if action == "didStartProvisionalNavigation" then return true end
    if action == "navigating" then
      if url and url:sub(1, 7) == "file://" then return true end
      return false
    end
    return true
  end)

  _webview:url("file://" .. htmlPath())
  _webview:show()

  -- Safety net: some HS builds don't deliver didFinishNavigation reliably.
  hs.timer.doAfter(0.5, onPageReady)
end

function M.hide()
  diagLog("M.hide called")
  destroyWebview()
end

-- Pre-render structured text (post-preprocess, with \x1c paragraph and
-- \x1d sentence delimiters intact). The mirror parses the delimiters
-- into <p> blocks and per-word <span>s. Subsequent feedSidecar calls
-- TAG those spans with data-buf / data-idx; the streaming-append v1
-- behavior is the fallback when feedText was never called.
function M.feedText(text)
  if not text or text == "" then return end
  _structuredText = text
  diagLog(string.format("feedText: %d bytes (paragraphs cached for render)",
                        #text))
  if not _webview then return end
  local ok_enc, json_str = pcall(hs.json.encode, text)
  if not ok_enc or not json_str then
    diagLog("feedText: json encode FAILED")
    return
  end
  local js = "window.mirror.renderText(" .. json_str .. ")"
  _webview:evaluateJavaScript(js)
end

function M.feedSidecar(buffer_id, sidecar_json)
  if not sidecar_json then return end
  -- Accept either a parsed table (callers that already decoded) or a
  -- JSON string (the typical path from claudio.lua's processStderrLine).
  local parsed
  if type(sidecar_json) == "table" then
    parsed = sidecar_json
  else
    local ok_parse, p = pcall(hs.json.decode, sidecar_json)
    if not ok_parse or not p then
      diagLog(string.format("feedSidecar: parse FAILED for buf=%s",
                            tostring(buffer_id)))
      return
    end
    parsed = p
  end
  _bufferSidecars[buffer_id] = parsed
  diagLog(string.format(
    "feedSidecar: buf=%s words=%d",
    tostring(buffer_id),
    parsed.words and #parsed.words or 0))
  -- Push to webview if it's already up; otherwise onPageReady will
  -- backfill from the cache.
  if _webview then
    pushSidecarToWebview(buffer_id, parsed)
  end
end

function M.onPos(ms, buffer_id)
  if not _webview then return end
  if buffer_id == nil then return end
  local sidecar = _bufferSidecars[buffer_id]
  if not sidecar or not sidecar.words then return end
  -- Find the word covering ms. Linear scan is fine — sidecars are
  -- typically <50 words and the lookup runs at 4 Hz.
  local idx = nil
  for i, w in ipairs(sidecar.words) do
    if ms < w.end_ms then
      idx = i - 1   -- 0-indexed for JS-side lookup
      break
    end
  end
  if idx == nil then
    -- Past the last word — highlight the last one.
    idx = #sidecar.words - 1
  end
  if idx < 0 then return end
  if buffer_id == _lastHighlightBuf and idx == _lastHighlightIdx then
    return
  end
  _lastHighlightBuf = buffer_id
  _lastHighlightIdx = idx
  jsCall("window.mirror.highlight", tostring(buffer_id), tostring(idx))
end

function M.endOfStream()
  diagLog(string.format(
    "M.endOfStream: webview=%s",
    tostring(_webview ~= nil)))
  if not _webview then return end
  scheduleLinger()
end

return M
