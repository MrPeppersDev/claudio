-- rsvp-test.lua — dev harness for the RSVP balloon UI
--
-- HOW TO USE
-- ──────────
-- 1. Add to ~/.hammerspoon/init.lua (or your config loader):
--        local rsvpTest = require("rsvp-test")
--        rsvpTest.start()
--
-- 2. Reload Hammerspoon config (Cmd+Alt+Ctrl+R or menu → Reload Config).
--
-- 3. Press Cmd+Alt+R to launch the balloon with a canned 15-word sidecar.
--    A simulated POS stream fires every 50ms at tempo ×1.54, advancing
--    the word display as if real audio were playing.
--
-- 4. While the balloon is visible:
--      Esc   → pause   (word dims, "paused" indicator appears)
--      Space → resume
--      ←  /  → scrub by ±2 seconds (seeks within the canned sidecar)
--
-- 5. The balloon lingers 2 s after the last word, then fades out.
--    Press Cmd+Alt+R again to replay at any time; a running session is
--    torn down first.
--
-- This harness does NOT require the Kokoro server, play-stream.py, or any
-- real audio. It is safe to run on a machine with no Claudio services active.

local M = {}

local rsvp = require("rsvp")

-- ── Canned sidecar (15 words, realistic per-word timings at ~150–400 ms/word)
--    Sample rate and audio_samples are included for spec compliance but are not
--    used by rsvp.lua — only the `words` array matters for the balloon.

local CANNED_SIDECAR = {
  version       = 1,
  sample_rate   = 24000,
  audio_samples = 102600,
  words = {
    { text = "The",           start_ms = 0,    end_ms = 180  },
    { text = "quick",         start_ms = 190,  end_ms = 430  },
    { text = "brown",         start_ms = 440,  end_ms = 680  },
    { text = "fox",           start_ms = 700,  end_ms = 880  },
    { text = "jumps",         start_ms = 900,  end_ms = 1150 },
    { text = "over",          start_ms = 1170, end_ms = 1390 },
    { text = "the",           start_ms = 1400, end_ms = 1520 },
    { text = "lazy",          start_ms = 1540, end_ms = 1800 },
    { text = "dog.",          start_ms = 1820, end_ms = 2100 },
    { text = "Extraordinarily", start_ms = 2200, end_ms = 2700 },
    { text = "sophisticated", start_ms = 2720, end_ms = 3200 },
    { text = "language",      start_ms = 3220, end_ms = 3580 },
    { text = "processing",    start_ms = 3600, end_ms = 3980 },
    { text = "capabilities",  start_ms = 4000, end_ms = 4450 },
    { text = "exist.",        start_ms = 4470, end_ms = 4800 },
  }
}

-- Serialise the canned sidecar to a JSON string.
-- We use hs.json.encode (available since Hammerspoon 0.9.57).
local function sidecarJson()
  return hs.json.encode(CANNED_SIDECAR)
end

-- ── Simulated POS stream ─────────────────────────────────────────────────────

-- Tempo multiplier. Matches the real pipeline's default playback rate.
local SIMULATED_TEMPO = 1.54

-- POS tick interval (ms). play-stream.py emits POS roughly every 50 ms.
local TICK_MS = 50

-- Current simulated playhead position (ms).
local _posMs   = 0
local _posTimer = nil
local _buffer_id = 0   -- single buffer in the test harness

-- Write the seek callbacks so the harness responds to ← / →.
local function setupCallbacks()
  rsvp.on_pause_request = function()
    hs.printf("rsvp-test: PAUSE requested (would write PAUSE to fd-9)")
  end
  rsvp.on_resume_request = function()
    hs.printf("rsvp-test: RESUME requested (would write RESUME to fd-9)")
  end
  rsvp.on_seek_request = function(delta_ms)
    _posMs = math.max(0, _posMs + delta_ms)
    hs.printf("rsvp-test: SEEK %+d ms → new pos %d ms", delta_ms, _posMs)
  end
end

-- Advance the simulated playhead by one tick and call rsvp.onPos.
local function tick()
  if not _posTimer then return end
  _posMs = _posMs + TICK_MS * SIMULATED_TEMPO
  rsvp.onPos(_posMs, _buffer_id)

  -- Stop the timer when we've passed the end of the sidecar.
  local lastWord = CANNED_SIDECAR.words[#CANNED_SIDECAR.words]
  if _posMs > lastWord.end_ms + 500 then
    -- Let rsvp's linger logic take over; stop the POS flood.
    if _posTimer then
      _posTimer:stop()
      _posTimer = nil
    end
    return
  end

  _posTimer = hs.timer.doAfter(TICK_MS / 1000, tick)
end

-- Start the simulated POS stream.
local function startPosStream()
  if _posTimer then
    _posTimer:stop()
    _posTimer = nil
  end
  _posMs = 0
  _posTimer = hs.timer.doAfter(TICK_MS / 1000, tick)
end

-- Stop the simulated POS stream.
local function stopPosStream()
  if _posTimer then
    _posTimer:stop()
    _posTimer = nil
  end
end

-- ── Trigger hotkey ────────────────────────────────────────────────────────────

local _triggerKey = nil

--- start()
--- Register Cmd+Alt+R as the harness trigger.
function M.start()
  if _triggerKey then return end

  _triggerKey = hs.hotkey.bind({"cmd", "alt"}, "r", function()
    hs.printf("rsvp-test: launching harness")

    -- Tear down any running session.
    stopPosStream()
    rsvp.hide()

    -- Register callbacks before show() so they're ready.
    setupCallbacks()

    -- Inject the sidecar into the module (buffer 0).
    rsvp.feedSidecar(_buffer_id, sidecarJson())

    -- Show the balloon (no sidecar_path; we've already fed via feedSidecar).
    rsvp.show(nil)

    -- Give the balloon 300 ms to open and the page to settle, then start POS.
    hs.timer.doAfter(0.3, function()
      startPosStream()
    end)
  end)

  hs.printf("rsvp-test: registered — press Cmd+Alt+R to launch harness")
end

--- stop()
--- Remove the trigger hotkey and tear down any running session.
function M.stop()
  stopPosStream()
  rsvp.hide()
  if _triggerKey then
    _triggerKey:delete()
    _triggerKey = nil
  end
end

return M
