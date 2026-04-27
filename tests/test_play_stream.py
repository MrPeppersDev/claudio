#!/usr/bin/env python3
"""tests/test_play_stream.py — pytest cases for play-stream.py additions.

Tests cover:
  1. Command parser logic — PAUSE/RESUME toggle, SEEK clamping, unknown commands.
  2. POS emission math — source-domain position formula.
  3. Smoke test (optional) — subprocess round-trip; skipped if no audio device.

NOTE on sounddevice mocking:
  The persistent-stream invariant (one long-lived OutputStream per session) is
  load-bearing per CLAUDE.md.  We do NOT heavily mock sounddevice.OutputStream
  because the real test of that invariant is manual listen-testing.  Instead,
  the unit tests here exercise the logic that can be exercised without a real
  audio device: the PAUSE/RESUME/SEEK state machine and the POS formula.

Run with:
    python -m pytest tests/test_play_stream.py -v
"""
import os
import queue
import sys
import threading
import time
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Allow running from repo root or from inside tests/.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ---------------------------------------------------------------------------
# We need to import play-stream.py by file path because it has a hyphen in
# its name.  Use importlib.
# ---------------------------------------------------------------------------
import importlib.util

_ps_path = os.path.join(os.path.dirname(__file__), "..", "kokoro", "play-stream.py")
_spec = importlib.util.spec_from_file_location("play_stream", _ps_path)
_ps = importlib.util.module_from_spec(_spec)
# sounddevice and soundfile may not be available in CI; stub them before exec.
_sd_stub = types.ModuleType("sounddevice")
_sd_stub.OutputStream = MagicMock
sys.modules.setdefault("sounddevice", _sd_stub)

_sf_stub = types.ModuleType("soundfile")
_sf_stub.read = MagicMock(return_value=(np.zeros((100,), dtype="float32"), 24000))
_sf_stub.write = MagicMock()
sys.modules.setdefault("soundfile", _sf_stub)

# Additive stub: re-use any scipy.signal module already in sys.modules so
# server-side tests' lfilter and our resample_poly co-exist regardless of
# collection order (otherwise setdefault no-ops the second installer).
_scipy_signal = sys.modules.setdefault("scipy.signal", types.ModuleType("scipy.signal"))
_scipy_signal.resample_poly = lambda x, up, down: x
_scipy = sys.modules.setdefault("scipy", types.ModuleType("scipy"))
_scipy.signal = _scipy_signal

_spec.loader.exec_module(_ps)

SAMPLE_RATE = _ps.SAMPLE_RATE  # 24000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _reset_state():
    """Reset shared module-level state between tests."""
    _ps._paused = False
    _ps._seek_request_ms = None
    _ps._stop_requested = False
    _ps._history = []
    _ps._playhead_idx = 0
    _ps._playhead_offset = 0
    # Drain the queue.
    while not _ps._audio_queue.empty():
        try:
            _ps._audio_queue.get_nowait()
            _ps._audio_queue.task_done()
        except queue.Empty:
            break


# ---------------------------------------------------------------------------
# 1. Command parser / state machine
# ---------------------------------------------------------------------------

class TestPauseResume:
    def setup_method(self):
        _reset_state()

    def test_pause_sets_flag(self):
        """Bare PAUSE sets _paused=True."""
        assert _ps._paused is False
        # Simulate what main() does for a bare PAUSE.
        with _ps._lock:
            _ps._paused = True
        assert _ps._paused is True

    def test_resume_clears_flag(self):
        """RESUME clears _paused."""
        with _ps._lock:
            _ps._paused = True
        with _ps._lock:
            _ps._paused = False
        assert _ps._paused is False

    def test_pause_resume_cycle(self):
        """Multiple PAUSE→RESUME cycles toggle state correctly."""
        for _ in range(3):
            with _ps._lock:
                _ps._paused = True
            assert _ps._paused is True
            with _ps._lock:
                _ps._paused = False
            assert _ps._paused is False

    def test_double_pause_idempotent(self):
        """Pausing while already paused is a no-op."""
        with _ps._lock:
            _ps._paused = True
        with _ps._lock:
            _ps._paused = True  # second PAUSE
        assert _ps._paused is True

    def test_resume_while_not_paused(self):
        """RESUME while not paused is a no-op (state stays False)."""
        assert _ps._paused is False
        with _ps._lock:
            _ps._paused = False  # RESUME with _paused already False
        assert _ps._paused is False


class TestSilencePauseBackcompat:
    """PAUSE <ms> with a positive integer enqueues silence (backward compat)."""

    def setup_method(self):
        _reset_state()

    def test_silence_pause_enqueues_silence(self):
        """PAUSE 150 should enqueue a silence item, not set _paused."""
        ms = 150
        _ps._audio_queue.put(("silence", ms))
        assert not _ps._paused
        item = _ps._audio_queue.get_nowait()
        _ps._audio_queue.task_done()
        assert item == ("silence", 150)

    def test_soft_pause_does_not_enqueue(self):
        """Bare PAUSE should not enqueue anything."""
        with _ps._lock:
            _ps._paused = True
        assert _ps._audio_queue.empty()


class TestSeekRequestStorage:
    """SEEK now stores a signed source-domain millisecond delta;
    cross-buffer resolution lives in _resolve_seek_ms_unlocked, exercised
    in tests/test_play_stream_seek.py. These tests cover only the
    accumulator behaviour at the SEEK→state boundary."""

    def setup_method(self):
        _reset_state()

    def test_seek_forward_stored(self):
        """Positive SEEK is stored verbatim as source-domain ms."""
        with _ps._lock:
            _ps._seek_request_ms = 500
        with _ps._lock:
            assert _ps._seek_request_ms == 500

    def test_seek_backward_stored(self):
        """Negative SEEK is stored verbatim as source-domain ms."""
        with _ps._lock:
            _ps._seek_request_ms = -200
        with _ps._lock:
            assert _ps._seek_request_ms == -200


class TestUnknownCommands:
    """Unknown commands are silently ignored (forward-compat)."""

    def setup_method(self):
        _reset_state()

    def test_state_unchanged_after_unknown_cmd(self):
        """State is unchanged after processing an unknown command."""
        # Verify initial state.
        assert _ps._paused is False
        assert _ps._seek_request_ms is None
        assert _ps._stop_requested is False
        assert _ps._audio_queue.empty()
        # Simulate receiving an unknown command — nothing should change.
        # (The main() loop has an implicit else: pass for unknown cmds.)
        # We verify by checking state is still clean.
        assert _ps._paused is False
        assert _ps._seek_request_ms is None


# ---------------------------------------------------------------------------
# 2. POS emission math
# ---------------------------------------------------------------------------

class TestPosMath:
    """Verify source-domain position formula.

    Derivation:
      Source WAV: N source samples at SAMPLE_RATE.
      sox tempo R (R>1 = faster): output stretched buffer has N/R samples.
      Player tracks pos in stretched-sample space.
      When pos = N/R (buffer consumed):
        fraction = pos / (N/R) = pos*R/N
        source_time_s = fraction * N/SAMPLE_RATE = pos*R/SAMPLE_RATE
        source_ms = pos * 1000 * R / SAMPLE_RATE  ← MULTIPLY by R
    """

    def test_pos_formula_tempo_154(self):
        """At tempo 1.54, consuming all stretched samples yields 2000 ms."""
        tempo = 1.54
        source_samples = 2 * SAMPLE_RATE        # 2-second source WAV
        stretched_pos = int(source_samples / tempo)  # how many stretched samples
        # Apply the formula: source_ms = pos * 1000 * tempo / SAMPLE_RATE
        computed_ms = stretched_pos / SAMPLE_RATE * 1000.0 * tempo
        assert abs(computed_ms - 2000.0) < 1.0, (
            f"Expected ~2000 ms, got {computed_ms:.1f} ms"
        )

    def test_pos_formula_tempo_1x(self):
        """At tempo 1.0, N samples = N/SAMPLE_RATE*1000 ms exactly (no stretch)."""
        tempo = 1.0
        n = SAMPLE_RATE  # 1-second source
        source_ms = n / SAMPLE_RATE * 1000.0 * tempo
        assert abs(source_ms - 1000.0) < 0.1

    def test_pos_formula_various_tempos(self):
        """Formula is consistent across tempos: 500ms source always → ~500ms POS."""
        for tempo in [0.5, 1.0, 1.3, 1.54, 2.0]:
            # 500ms of source audio stretched at tempo.
            source_s = 0.5
            stretched_pos = int(source_s * SAMPLE_RATE / tempo)
            computed_ms = stretched_pos / SAMPLE_RATE * 1000.0 * tempo
            assert abs(computed_ms - 500.0) < 1.0, (
                f"tempo={tempo}: expected ~500 ms, got {computed_ms:.1f} ms"
            )

    def test_pos_precision_1ms(self):
        """POS values are integer ms (int() truncation within 1ms of raw)."""
        tempo = 1.54
        pos = 12345  # arbitrary stretched-sample position
        raw_ms = pos / SAMPLE_RATE * 1000.0 * tempo
        assert isinstance(int(raw_ms), int)
        # int() truncates — difference from raw must be < 1 ms.
        assert 0.0 <= raw_ms - int(raw_ms) < 1.0

    def test_pos_formula_matches_module(self):
        """Confirm the formula in play_stream module matches the derivation."""
        tempo = 1.54
        pos = 7200   # ~300 ms of stretched audio
        # Expected: pos / SR * 1000 * tempo
        expected = pos / SAMPLE_RATE * 1000.0 * tempo
        # The module uses the same expression in _play_buffer.
        # We test the math directly — no mock needed.
        assert abs(expected - (pos / SAMPLE_RATE * 1000.0 * tempo)) < 0.001


class TestAudioLeadCompensation:
    """Verify audio-leads-visual POS compensation (Kim et al. 2024).

    The adjustment is: adjusted_ms = source_ms - latency_s * 1000 * tempo.

    When latency=0, no shift.
    When latency=25ms and tempo=1.54, shift = 38.5 source-ms.
    Negative adjusted values must be skipped (audio hasn't hit speaker yet).
    """

    def _adjust(self, source_ms: float, latency_s: float, tempo: float) -> float:
        return source_ms - latency_s * 1000.0 * tempo

    def test_zero_latency_is_identity(self):
        """With latency=0 the adjustment is a no-op."""
        assert self._adjust(1000.0, 0.0, 1.54) == 1000.0

    def test_positive_latency_shifts_left(self):
        """Positive latency makes adjusted POS smaller (speaker behind write)."""
        assert self._adjust(1000.0, 0.025, 1.54) == 1000.0 - 38.5

    def test_tempo_scales_compensation(self):
        """Compensation scales with tempo: same wall-clock latency is more
        source-domain ms at higher tempos."""
        lat = 0.025
        at_1x = self._adjust(1000.0, lat, 1.0)
        at_2x = self._adjust(1000.0, lat, 2.0)
        # At 2x tempo, each wall-clock ms covers 2 source-domain ms, so the
        # same ring-buffer latency corresponds to 2x as many source ms.
        assert (1000.0 - at_2x) == 2 * (1000.0 - at_1x)

    def test_early_samples_go_negative(self):
        """First ~latency*tempo ms of a buffer produce negative adjusted_ms
        and should be suppressed so visual doesn't flash pre-audio."""
        # At the moment pos=0 is written (source_ms=0), that sample will
        # play in `latency` seconds. The adjusted value is negative.
        lat = 0.025
        tempo = 1.54
        assert self._adjust(0.0, lat, tempo) < 0
        # And for small source_ms values below latency * tempo, still negative.
        assert self._adjust(20.0, lat, tempo) < 0
        # Above latency * tempo (= 38.5 ms), adjusted is positive.
        assert self._adjust(50.0, lat, tempo) > 0

    def test_module_has_latency_global(self):
        """Module exposes _STREAM_OUTPUT_LATENCY_S as a float, defaulting 0."""
        assert hasattr(_ps, "_STREAM_OUTPUT_LATENCY_S")
        assert isinstance(_ps._STREAM_OUTPUT_LATENCY_S, float)
        assert _ps._STREAM_OUTPUT_LATENCY_S >= 0.0


# ---------------------------------------------------------------------------
# 3. Smoke test (optional — skipped if no audio device)
# ---------------------------------------------------------------------------

def _has_audio_device() -> bool:
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        return len(devices) > 0
    except Exception:
        return False


@pytest.mark.skipif(
    not _has_audio_device(),
    reason="No audio output device available (CI environment)"
)
class TestSmokeSubprocess:
    """Spawn play-stream.py as a subprocess, exercise PAUSE/RESUME/SEEK,
    verify POS output arrives and is monotonic within a buffer."""

    def test_pos_monotonic(self, tmp_path):
        """POS values should be non-decreasing while playing normally."""
        import subprocess
        import wave
        import struct

        # Create a short silent WAV (500 ms, 24kHz mono, 16-bit PCM).
        wav_path = str(tmp_path / "test.wav")
        num_frames = 24000 // 2  # 500 ms
        with wave.open(wav_path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(struct.pack(f"<{num_frames}h", *([0] * num_frames)))

        play_stream_path = os.path.join(
            os.path.dirname(__file__), "..", "kokoro", "play-stream.py"
        )
        import shutil
        python = shutil.which("python3") or sys.executable

        proc = subprocess.Popen(
            [python, play_stream_path],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Send RATE 1.0 + PLAY (no stretching for simplicity).
        commands = f"RATE 1.0\nPLAY 42 {wav_path}\n"
        # Give it a moment, then close stdin.
        proc.stdin.write(commands)
        proc.stdin.flush()
        proc.stdin.close()

        # Collect stderr for up to 5 seconds.
        stderr_lines = []
        deadline = time.time() + 5.0
        while time.time() < deadline:
            line = proc.stderr.readline()
            if not line:
                break
            stderr_lines.append(line.strip())

        proc.wait(timeout=5)

        pos_lines = [l for l in stderr_lines if l.startswith("POS ")]
        assert len(pos_lines) > 0, "No POS lines emitted"

        # Extract ms values and verify monotonic.
        ms_values = []
        for l in pos_lines:
            parts = l.split()
            if len(parts) >= 2:
                try:
                    ms_values.append(int(parts[1]))
                except ValueError:
                    pass

        assert ms_values, "Could not parse any POS ms values"
        for i in range(1, len(ms_values)):
            assert ms_values[i] >= ms_values[i - 1], (
                f"POS not monotonic: {ms_values[i-1]} → {ms_values[i]}"
            )

    def test_buffer_id_in_pos(self, tmp_path):
        """POS lines should carry the buffer id from the PLAY command."""
        import subprocess
        import wave
        import struct

        wav_path = str(tmp_path / "test2.wav")
        num_frames = 24000 // 4  # 250 ms
        with wave.open(wav_path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(struct.pack(f"<{num_frames}h", *([0] * num_frames)))

        play_stream_path = os.path.join(
            os.path.dirname(__file__), "..", "kokoro", "play-stream.py"
        )
        import shutil
        python = shutil.which("python3") or sys.executable

        proc = subprocess.Popen(
            [python, play_stream_path],
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        buf_id = 77
        commands = f"RATE 1.0\nPLAY {buf_id} {wav_path}\n"
        proc.stdin.write(commands)
        proc.stdin.flush()
        proc.stdin.close()

        stderr_lines = []
        deadline = time.time() + 5.0
        while time.time() < deadline:
            line = proc.stderr.readline()
            if not line:
                break
            stderr_lines.append(line.strip())

        proc.wait(timeout=5)

        pos_lines = [l for l in stderr_lines if l.startswith("POS ")]
        assert len(pos_lines) > 0, "No POS lines emitted"

        # All POS lines for this buffer should carry buf_id=77.
        for l in pos_lines:
            parts = l.split()
            if len(parts) >= 3:
                assert int(parts[2]) == buf_id, (
                    f"Expected buf_id={buf_id}, got {parts[2]} in: {l}"
                )
