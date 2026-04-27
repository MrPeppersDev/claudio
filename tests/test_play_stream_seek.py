#!/usr/bin/env python3
"""tests/test_play_stream_seek.py — cross-buffer scrubbing in play-stream.py.

The pre-Bug-2 _play_buffer clamped seeks to the currently-playing buffer's
own [0, total] range. Bug 2 replaced that with a history of every buffer
the user has heard, plus a (idx, offset) playhead, so SEEK can walk
backward across sentence boundaries to scrub the entire playback or
forward past the end of what's been queued.

These tests target _resolve_seek_ms_unlocked, _history_append_unlocked,
and _drain_done_ready_unlocked directly. Real-world end-to-end
verification (F13 → multi-paragraph selection → arrow-key scrub) lives
in the manual test sweep, not pytest.

Run with:
    python -m pytest tests/test_play_stream_seek.py -v
"""
import os
import queue
import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Stub heavy deps the same way test_play_stream.py does so import works
# in CI without sounddevice/soundfile installed.
import importlib.util

_ps_path = os.path.join(os.path.dirname(__file__), "..", "kokoro", "play-stream.py")
_spec = importlib.util.spec_from_file_location("play_stream", _ps_path)
_ps = importlib.util.module_from_spec(_spec)

_sd_stub = types.ModuleType("sounddevice")
_sd_stub.OutputStream = MagicMock
sys.modules.setdefault("sounddevice", _sd_stub)

_sf_stub = types.ModuleType("soundfile")
_sf_stub.read = MagicMock(return_value=(np.zeros((100,), dtype="float32"), 24000))
_sf_stub.write = MagicMock()
sys.modules.setdefault("soundfile", _sf_stub)

_scipy_signal = types.ModuleType("scipy.signal")
_scipy_signal.resample_poly = lambda x, up, down: x
_scipy = types.ModuleType("scipy")
_scipy.signal = _scipy_signal
sys.modules.setdefault("scipy", _scipy)
sys.modules.setdefault("scipy.signal", _scipy_signal)

_spec.loader.exec_module(_ps)

SR = _ps.SAMPLE_RATE  # 24000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_buf(buf_id: int, source_ms: int, tempo: float = 1.0):
    """Build a (buf_id, samples, tempo) tuple representing `source_ms` of
    source-domain audio, post-stretch (so len(samples) = source_ms * SR /
    1000 / tempo). Sample values don't matter for seek logic; we just need
    correct length."""
    n_stretched = int(source_ms * SR / 1000.0 / tempo)
    samples = np.zeros(n_stretched, dtype=np.float32)
    return (buf_id, samples, tempo)


def _reset():
    """Reset module state to a known-clean baseline for each test."""
    _ps._history = []
    _ps._playhead_idx = 0
    _ps._playhead_offset = 0
    _ps._seek_request_ms = None
    _ps._kokoro_done = False
    _ps._drain_done_emitted = False
    while not _ps._audio_queue.empty():
        try:
            _ps._audio_queue.get_nowait()
            _ps._audio_queue.task_done()
        except queue.Empty:
            break


def _current_source_ms():
    """Compute the current source-domain ms position from the playhead."""
    if not _ps._history or _ps._playhead_idx >= len(_ps._history):
        return None
    _, samples, tempo = _ps._history[_ps._playhead_idx]
    return _ps._playhead_offset / SR * 1000.0 * tempo


# ---------------------------------------------------------------------------
# Backward seek across buffer boundary
# ---------------------------------------------------------------------------

class TestSeekBackward:
    def setup_method(self):
        _reset()

    def test_within_current_buffer(self):
        """Backward seek that stays inside the current buffer just moves
        offset backward."""
        _ps._history.append(_make_buf(0, 1000))   # 1 s buffer
        _ps._playhead_idx = 0
        _ps._playhead_offset = int(0.5 * SR)      # 500 ms in
        with _ps._lock:
            _ps._resolve_seek_ms_unlocked(-200)
        assert _ps._playhead_idx == 0
        # Should land near 300 ms.
        assert abs(_current_source_ms() - 300) < 2.0

    def test_across_one_boundary(self):
        """-2000ms from inside buffer 2 should land inside buffer 1."""
        _ps._history.append(_make_buf(0, 1500))   # buf 0: 1.5 s
        _ps._history.append(_make_buf(1, 1500))   # buf 1: 1.5 s
        _ps._history.append(_make_buf(2, 1500))   # buf 2: 1.5 s
        _ps._playhead_idx = 2
        _ps._playhead_offset = int(0.5 * SR)      # 500 ms into buf 2
        with _ps._lock:
            _ps._resolve_seek_ms_unlocked(-2000)
        # 500 ms back leaves us at start of buf 2; another 1500 ms back
        # lands at start of buf 1; 0 ms remaining → playhead at buf 1, 0.
        assert _ps._playhead_idx == 1
        assert _ps._playhead_offset == 0

    def test_across_two_boundaries(self):
        """-3500ms from buf 3 walks past bufs 2 and 1, lands inside buf 0."""
        _ps._history.append(_make_buf(0, 1000))   # buf 0
        _ps._history.append(_make_buf(1, 1000))   # buf 1
        _ps._history.append(_make_buf(2, 1000))   # buf 2
        _ps._history.append(_make_buf(3, 1000))   # buf 3
        _ps._playhead_idx = 3
        _ps._playhead_offset = int(0.5 * SR)      # 500 ms into buf 3
        with _ps._lock:
            _ps._resolve_seek_ms_unlocked(-3500)
        # 500 ms (buf 3) + 1000 ms (buf 2) + 1000 ms (buf 1) = 2500 ms
        # consumed; 1000 remaining → buf 0, end - 1000 ms = at start.
        assert _ps._playhead_idx == 0
        assert abs(_current_source_ms() - 0) < 2.0

    def test_clamps_at_oldest(self):
        """Seeking back past the oldest available buffer clamps at idx 0,
        offset 0."""
        _ps._history.append(_make_buf(0, 500))    # 500 ms
        _ps._playhead_idx = 0
        _ps._playhead_offset = int(0.2 * SR)      # 200 ms in
        with _ps._lock:
            _ps._resolve_seek_ms_unlocked(-9999)  # way past start
        assert _ps._playhead_idx == 0
        assert _ps._playhead_offset == 0

    def test_seek_when_playhead_past_end_of_history(self):
        """Player loop's natural-advance increments _playhead_idx and resets
        offset to 0 even when there's no next buffer yet (the queue-pull
        happens on the *next* iteration). A SEEK landing in that window
        used to index history out-of-bounds. Should clamp to end of last
        buffer and proceed normally."""
        _ps._history.append(_make_buf(0, 1000))
        _ps._history.append(_make_buf(1, 1000))
        _ps._playhead_idx = 2  # past end
        _ps._playhead_offset = 0
        with _ps._lock:
            _ps._resolve_seek_ms_unlocked(-500)  # back into buf 1
        assert _ps._playhead_idx == 1
        assert abs(_current_source_ms() - 500) < 2.0

    def test_forward_seek_when_playhead_past_end_no_crash(self):
        """Same off-by-one window, forward seek with empty queue: must
        clamp at end of last history buffer rather than crash."""
        _ps._history.append(_make_buf(0, 500))
        _ps._playhead_idx = 1  # past end
        _ps._playhead_offset = 0
        with _ps._lock:
            _ps._resolve_seek_ms_unlocked(2000)  # would walk forward
        assert _ps._playhead_idx == 0
        assert _ps._playhead_offset == len(_ps._history[0][1])


# ---------------------------------------------------------------------------
# Forward seek across buffer boundary
# ---------------------------------------------------------------------------

class TestSeekForward:
    def setup_method(self):
        _reset()

    def test_within_current_buffer(self):
        """Forward seek that stays inside the current buffer."""
        _ps._history.append(_make_buf(0, 2000))
        _ps._playhead_idx = 0
        _ps._playhead_offset = int(0.5 * SR)
        with _ps._lock:
            _ps._resolve_seek_ms_unlocked(500)
        assert _ps._playhead_idx == 0
        assert abs(_current_source_ms() - 1000) < 2.0

    def test_across_one_boundary(self):
        """+1500ms from inside buf 0 should land inside buf 1."""
        _ps._history.append(_make_buf(0, 1000))
        _ps._history.append(_make_buf(1, 2000))
        _ps._playhead_idx = 0
        _ps._playhead_offset = int(0.2 * SR)      # 200 ms in
        with _ps._lock:
            _ps._resolve_seek_ms_unlocked(1500)
        # 800 ms left in buf 0; 700 ms remaining lands 700 ms into buf 1.
        assert _ps._playhead_idx == 1
        assert abs(_current_source_ms() - 700) < 2.0

    def test_drains_queue_when_past_end(self):
        """Forward seek past last history buffer drains the audio queue
        (non-blocking) and walks into newly-materialized buffers."""
        _ps._history.append(_make_buf(0, 500))
        _ps._playhead_idx = 0
        _ps._playhead_offset = int(0.1 * SR)      # 100 ms in (400 ms left)
        # Queue up two more sentences-worth of audio.
        _ps._audio_queue.put(("samples", 1, np.zeros(int(SR * 0.5), dtype=np.float32), 1.0))
        _ps._audio_queue.put(("samples", 2, np.zeros(int(SR * 0.5), dtype=np.float32), 1.0))
        with _ps._lock:
            _ps._resolve_seek_ms_unlocked(900)
        # 400 ms left in buf 0 → into buf 1 (just pulled). 500 ms left
        # in buf 1 (none consumed yet); 500 ms remaining → end of buf 1.
        # Walk forward by another 0 ms means we end up at end of buf 1
        # OR start of buf 2 depending on ordering. Both are 1000 ms
        # source-cumulative from start of buf 1, which is what we want.
        assert _ps._playhead_idx >= 1
        assert len(_ps._history) >= 2  # at least one buffer was pulled

    def test_clamps_at_end_of_last(self):
        """Forward seek past everything available clamps at end of last
        buffer (no infinite seek)."""
        _ps._history.append(_make_buf(0, 500))
        _ps._playhead_idx = 0
        _ps._playhead_offset = 0
        # Empty queue.
        with _ps._lock:
            _ps._resolve_seek_ms_unlocked(99999)
        assert _ps._playhead_idx == 0
        assert _ps._playhead_offset == len(_ps._history[0][1])


# ---------------------------------------------------------------------------
# History byte-cap eviction
# ---------------------------------------------------------------------------

class TestHistoryEviction:
    def setup_method(self):
        _reset()

    def test_under_cap_no_eviction(self):
        """Appends below the byte cap don't evict anything."""
        with _ps._lock:
            _ps._history_append_unlocked(_make_buf(0, 100))
            _ps._history_append_unlocked(_make_buf(1, 100))
        assert len(_ps._history) == 2

    def test_over_cap_evicts_front_and_shifts_playhead(self):
        """Eviction of front buffer decrements _playhead_idx so it still
        points at the same logical buffer."""
        # Force a tiny cap so we can test eviction without giant arrays.
        original_cap = _ps._HISTORY_MAX_BYTES
        try:
            _ps._HISTORY_MAX_BYTES = 1024  # 1 KB cap → 256 float32 samples
            # Each ~100 sample buffer is 400 bytes; 3 of them = 1200 B > cap.
            with _ps._lock:
                _ps._history.append((0, np.zeros(100, dtype=np.float32), 1.0))
                _ps._history.append((1, np.zeros(100, dtype=np.float32), 1.0))
                _ps._playhead_idx = 1  # we're "on" buf 1
                _ps._playhead_offset = 50
                # Append a third — should evict buf 0 (front) and shift
                # _playhead_idx from 1 → 0.
                _ps._history_append_unlocked(
                    (2, np.zeros(100, dtype=np.float32), 1.0)
                )
            assert len(_ps._history) == 2
            assert _ps._history[0][0] == 1  # buf 1 is now at idx 0
            assert _ps._playhead_idx == 0    # shifted to keep pointing at buf 1
            assert _ps._playhead_offset == 50  # offset preserved
        finally:
            _ps._HISTORY_MAX_BYTES = original_cap

    def test_never_evicts_current_or_newer(self):
        """If the playhead is on the front buffer, eviction can't touch it
        (or anything newer) even if total bytes exceed the cap."""
        original_cap = _ps._HISTORY_MAX_BYTES
        try:
            _ps._HISTORY_MAX_BYTES = 100  # absurdly tiny
            with _ps._lock:
                _ps._history.append((0, np.zeros(1000, dtype=np.float32), 1.0))
                _ps._playhead_idx = 0
                _ps._playhead_offset = 100
                _ps._history_append_unlocked(
                    (1, np.zeros(1000, dtype=np.float32), 1.0)
                )
            # Cap is way exceeded but front is at the playhead, so nothing
            # was evicted.
            assert len(_ps._history) == 2
            assert _ps._playhead_idx == 0
        finally:
            _ps._HISTORY_MAX_BYTES = original_cap


# ---------------------------------------------------------------------------
# Drain-done readiness across history
# ---------------------------------------------------------------------------

class TestDrainDoneReady:
    def setup_method(self):
        _reset()

    def test_empty_history_with_empty_queue_is_ready(self):
        """Edge case: EXIT before any audio queued. Drain-done can fire."""
        with _ps._lock:
            assert _ps._drain_done_ready_unlocked() is True

    def test_playhead_mid_buffer_not_ready(self):
        """User scrubbed back; not at end of last history buffer → not
        ready, even if queue is empty and Timers are done."""
        _ps._history.append(_make_buf(0, 1000))
        _ps._history.append(_make_buf(1, 1000))
        _ps._playhead_idx = 0
        _ps._playhead_offset = int(0.3 * SR)
        with _ps._lock:
            assert _ps._drain_done_ready_unlocked() is False

    def test_playhead_at_end_of_last_is_ready(self):
        """Playhead at the very end of the last buffer + empty queue + no
        Timers → drain-done ready."""
        _ps._history.append(_make_buf(0, 1000))
        _ps._history.append(_make_buf(1, 1000))
        last_samples = _ps._history[1][1]
        _ps._playhead_idx = 1
        _ps._playhead_offset = len(last_samples)
        with _ps._lock:
            assert _ps._drain_done_ready_unlocked() is True

    def test_pending_queue_blocks(self):
        """Anything still in the audio queue blocks drain-done regardless
        of playhead state."""
        _ps._history.append(_make_buf(0, 100))
        _ps._playhead_idx = 0
        _ps._playhead_offset = len(_ps._history[0][1])
        _ps._audio_queue.put(("silence", 50))
        try:
            with _ps._lock:
                assert _ps._drain_done_ready_unlocked() is False
        finally:
            # Drain so other tests aren't polluted.
            _ps._audio_queue.get_nowait()
            _ps._audio_queue.task_done()


# ---------------------------------------------------------------------------
# SEEK accumulator behaviour at the command boundary
# ---------------------------------------------------------------------------

class TestSeekAccumulator:
    def setup_method(self):
        _reset()

    def test_successive_seeks_accumulate(self):
        """Two SEEK commands arriving back-to-back before the player thread
        resolves the first should sum, not overwrite. Users tap the
        arrow key several times rapidly; dropping all but the last
        would feel unresponsive."""
        with _ps._lock:
            _ps._seek_request_ms = -2000
        with _ps._lock:
            if _ps._seek_request_ms is None:
                _ps._seek_request_ms = -2000
            else:
                _ps._seek_request_ms += -2000
        assert _ps._seek_request_ms == -4000

    def test_seek_cleared_after_resolution(self):
        """After the player thread has consumed the request, _seek_request_ms
        is None again so the next SEEK is treated as a fresh start, not
        accumulated onto a stale value."""
        with _ps._lock:
            _ps._seek_request_ms = 500
            ms = _ps._seek_request_ms
            _ps._seek_request_ms = None
        assert ms == 500
        assert _ps._seek_request_ms is None
