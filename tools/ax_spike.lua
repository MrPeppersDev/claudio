-- ax_spike.lua — AX coverage + latency spike for the #163 design step.
--
-- PR 1 of the structural-reading bundle is "1-day investigation: does
-- AXBoundsForRange(AXSelectedTextRange) return per-character rects fast
-- enough across the apps Claudio is going to be used in?" This script is
-- the rig that gathers that data.
--
-- ── Use ────────────────────────────────────────────────────────────────────
--
-- 1. In ~/.hammerspoon/init.lua, add:
--      local ax_spike = require("ax_spike")
--      ax_spike.start()
--
--    The require path resolves via hs.configdir; symlink or copy this file
--    into ~/.hammerspoon/ax_spike.lua.
--
-- 2. Open each target app, paste the 200-word reference text from
--    tools/ax_spike_text.txt (or any selection of similar size), and
--    select the entire pasted block.
--
-- 3. Press ⌃⌥F13. The script measures AXBoundsForRange latency over the
--    full selection and writes one entry to ~/.claude/claudio/ax_spike.log.
--    A summary alert pops in Hammerspoon for instant feedback.
--
-- 4. Repeat for each of the 13 target apps (Safari, TextEdit, Notes, Mail,
--    Preview, Books, Chrome, Firefox, VS Code, Cursor, iTerm2, Outlook,
--    Slack). Stop early if the first 5 all return nil — the answer is
--    "AX-follow is unsupported on most apps" and the spike's done.
--
-- 5. Aggregate the log into the #163 coverage matrix.
--
-- ── What it measures ──────────────────────────────────────────────────────
--
-- For each character index in the selection range, queries
-- AXBoundsForRange(AXUIElement, {location=i, length=1}) and times the
-- round-trip. Logs:
--   - app name + bundle id
--   - selection length (chars)
--   - rects-returned / rects-attempted
--   - per-query latency p50, p95, max
--   - first 3 rect samples (sanity check that values look like screen rects)
--
-- Ship gate for AX-follow viability is p95 < 50 ms at 4 Hz POS rate, per
-- the umbrella plan addendum A4. The script flags VIABLE / MARGINAL /
-- UNVIABLE based on that threshold.

local M = {}

local SPIKE_LOG_PATH = (os.getenv("CLAUDIO_STATE_DIR")
                         or os.getenv("CLAUDIO_DIR")
                         or (os.getenv("HOME") .. "/.claude/claudio"))
                         .. "/ax_spike.log"

-- p95 latency threshold for "AX-follow viable"; matches the design step
-- addendum A4 (umbrella issue #163).
local LATENCY_VIABLE_P95_MS = 50
local LATENCY_MARGINAL_P95_MS = 100

-- Cap per-app run time. A 5000-char selection at 5 ms/query is 25 s of
-- spike time; that's fine for an investigation but unfriendly if the user
-- accidentally selects a whole document. Cap iteration at this many chars.
local MAX_QUERIES = 500

local hotkey = nil

local function logLine(s)
  local f = io.open(SPIKE_LOG_PATH, "a")
  if not f then return end
  f:write(s)
  f:write("\n")
  f:close()
end

-- p-th percentile of a sorted array of numbers. Linear interpolation
-- between the two surrounding samples — cheap and good enough for an
-- investigation rig (we're not publishing this in a paper).
local function percentile(sorted, p)
  local n = #sorted
  if n == 0 then return 0 end
  if n == 1 then return sorted[1] end
  local rank = (p / 100) * (n - 1) + 1
  local lo = math.floor(rank)
  local hi = math.ceil(rank)
  if lo == hi then return sorted[lo] end
  local frac = rank - lo
  return sorted[lo] * (1 - frac) + sorted[hi] * frac
end

-- Rect-formatter used in the log preview block. AXBoundsForRange returns
-- a hs.geometry-style point/size pair on macOS.
local function fmtRect(r)
  if type(r) ~= "table" then return tostring(r) end
  local x = r.x or (r._w or {}).x or "?"
  local y = r.y or (r._w or {}).y or "?"
  local w = r.w or "?"
  local h = r.h or "?"
  return string.format("(%s,%s,%s×%s)",
    tostring(x), tostring(y), tostring(w), tostring(h))
end

local function classifyViability(p95)
  if p95 < LATENCY_VIABLE_P95_MS then return "VIABLE" end
  if p95 < LATENCY_MARGINAL_P95_MS then return "MARGINAL" end
  return "UNVIABLE"
end

function M.runSpike()
  local app = hs.application.frontmostApplication()
  local appName = app and app:name() or "unknown"
  local bundleId = app and app:bundleID() or "unknown"

  local stamp = os.date("%Y-%m-%d %H:%M:%S")

  -- Get the focused AX element. systemWideElement → AXFocusedUIElement is
  -- the standard path; pcall everything because some apps (Electron-based
  -- mainly) raise on the systemWide query under certain focus states.
  local okSys, sys = pcall(hs.axuielement.systemWideElement)
  if not okSys or not sys then
    local msg = string.format(
      "[%s] %s (%s) — systemWideElement FAILED (no AX root)",
      stamp, appName, bundleId)
    logLine(msg)
    hs.alert.show("AX spike: " .. appName .. " — no AX root", 3)
    return
  end

  local focused = sys:attributeValue("AXFocusedUIElement")
  if not focused then
    local msg = string.format(
      "[%s] %s (%s) — AXFocusedUIElement nil (no focus / app doesn't expose AX)",
      stamp, appName, bundleId)
    logLine(msg)
    hs.alert.show("AX spike: " .. appName .. " — no focused element", 3)
    return
  end

  -- Read the selection range. AXSelectedTextRange is a CFRange wrapped in
  -- an AXValue. Hammerspoon unwraps it to {location=N, length=N}.
  local range = focused:attributeValue("AXSelectedTextRange")
  if type(range) ~= "table" or type(range.location) ~= "number"
     or type(range.length) ~= "number" then
    local msg = string.format(
      "[%s] %s (%s) — AXSelectedTextRange unavailable or malformed (got %s)",
      stamp, appName, bundleId, hs.inspect(range))
    logLine(msg)
    hs.alert.show("AX spike: " .. appName .. " — no selected range", 3)
    return
  end

  if range.length == 0 then
    local msg = string.format(
      "[%s] %s (%s) — selection length is 0; select text first",
      stamp, appName, bundleId)
    logLine(msg)
    hs.alert.show("AX spike: " .. appName .. " — empty selection", 3)
    return
  end

  -- Determine how many per-character queries to run. Cap at MAX_QUERIES
  -- so very large selections don't lock the spike for minutes.
  local queryCount = math.min(range.length, MAX_QUERIES)

  local latencies = {}     -- per-query elapsed in ms
  local rectsValid = 0
  local sampleRects = {}   -- first 3 rects for the log preview

  for i = 0, queryCount - 1 do
    local subrange = { location = range.location + i, length = 1 }
    local t0 = hs.timer.absoluteTime()
    local okCall, rect = pcall(focused.parameterizedAttributeValue,
                                focused, "AXBoundsForRange", subrange)
    local t1 = hs.timer.absoluteTime()

    -- absoluteTime is in nanoseconds.
    local elapsed_ms = (t1 - t0) / 1e6
    table.insert(latencies, elapsed_ms)

    if okCall and type(rect) == "table" and rect.w and rect.h
       and rect.w > 0 and rect.h > 0 then
      rectsValid = rectsValid + 1
      if #sampleRects < 3 then
        table.insert(sampleRects, fmtRect(rect))
      end
    end
  end

  -- Sort copy for percentiles; keep latencies untouched in case a future
  -- variant of this script wants insertion order.
  local sortedLat = {}
  for _, v in ipairs(latencies) do table.insert(sortedLat, v) end
  table.sort(sortedLat)

  local p50 = percentile(sortedLat, 50)
  local p95 = percentile(sortedLat, 95)
  local maxLat = sortedLat[#sortedLat] or 0

  local rectFraction = string.format("%d/%d", rectsValid, queryCount)
  local viability = classifyViability(p95)

  -- Per-line entry that's easy to grep / aggregate.
  local entry = string.format(
    "[%s] app=%q bundle=%q sel_chars=%d queried=%d rects_valid=%s "
    .. "p50_ms=%.2f p95_ms=%.2f max_ms=%.2f viability=%s sample_rects=[%s]",
    stamp, appName, bundleId, range.length, queryCount, rectFraction,
    p50, p95, maxLat, viability, table.concat(sampleRects, " "))
  logLine(entry)

  hs.alert.show(string.format(
    "AX spike: %s\n%s rects, p50=%.1f ms, p95=%.1f ms\n%s",
    appName, rectFraction, p50, p95, viability), 4)
end

function M.start()
  if hotkey then return end
  hotkey = hs.hotkey.bind({ "ctrl", "alt" }, "F13", function()
    M.runSpike()
  end)
  hs.alert.show("AX spike armed: ⌃⌥F13 to capture", 2)
end

function M.stop()
  if hotkey then
    hotkey:delete()
    hotkey = nil
  end
end

return M
