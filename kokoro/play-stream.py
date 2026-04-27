#!/usr/bin/env python3
"""play-stream.py — persistent sounddevice OutputStream for Kokoro TTS playback.

Reads commands from stdin (one per line) and writes audio samples into a
single long-lived OutputStream.  Keeping the stream open across sentences
means CoreAudio initialises once (during the 400ms silent pre-roll) so the
very first word is never clipped.

Protocol (each stdin line is one command):

  PLAY <id> <absolute-path>  — time-stretch and play the WAV at the current
                               RATE; <id> is an integer buffer identifier
                               (incremented by kokoro-tts.sh per sentence) and
                               is echoed in every POS message for this buffer.
                               Backward-compat: PLAY <absolute-path> (no id)
                               is accepted and treated as id=0.
  EARCON <absolute-path>     — play the file at 1.0x (no stretching); resamples
                               to 24 kHz if needed (handles .aiff earcons).
  PAUSE                      — soft pause: halt buffer consumption without
                               closing the stream. The stream stays alive and
                               silent; POS emission pauses (position doesn't
                               advance). No-op if already paused.
                               NOTE: PAUSE <ms> (old form) now inserts silence
                               as before for backward compatibility when arg is
                               a positive integer; bare PAUSE is the soft-pause.
  RESUME                     — resume consumption after PAUSE. No-op if not
                               paused.
  SEEK <ms>                  — jump by ±ms relative to current playhead within
                               the currently-playing buffer. Clamped to
                               [0, buffer_duration]. Cross-buffer seeks are
                               out of scope and silently ignored (clamp only).
  STOP                       — clear the audio queue and halt playback. The
                               stream stays alive (PLAY can restart it).
  RATE <float>               — set playback rate for subsequent PLAY commands.

POS output (stderr, distinctive prefix so kokoro-tts.sh can tee/filter it):

  POS <ms> <id>

  Emitted at ~20 ms cadence while playing.  <ms> is the playhead position
  in the currently-playing WAV's own time frame (pre-sox-tempo stretch), so
  it matches the word-offset timestamps written by the PR-1 timing sidecar.

  Formula: ms = pos_stretched * 1000 * tempo_factor / SAMPLE_RATE
  where pos_stretched = stretched samples consumed so far and
  tempo_factor = current RATE (the sox tempo argument).
  Derivation: sox tempo R compresses source duration by R, so each stretched
  sample corresponds to R source samples; multiply (not divide) to recover
  source time.

  POS emission drops to 0 Hz during PAUSE.
  When a new PLAY starts, POS resets to 0 for that buffer's time frame.

Design note — stderr vs fd-10:
  Using stderr with a "POS " prefix is the least-invasive channel: no new fd
  plumbing needed in kokoro-tts.sh, and the shell caller can tee stderr to a
  pipe while keeping the prefix as a disambiguator.  PR 3 (RSVP UI) should
  read from the stderr pipe and filter lines that start with "POS ".

Exits cleanly on stdin EOF or SIGTERM.
"""

import io
import os
import queue
import signal
import subprocess
import sys
import threading
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.signal import resample_poly

# ---- constants ----------------------------------------------------------------

SAMPLE_RATE = 24000   # Kokoro native output rate; stream stays at this rate
CHANNELS = 1
DTYPE = "float32"
PRE_ROLL_MS = 400     # silent pre-roll so CoreAudio ramp-up doesn't eat word 1.
                      # Need to cover: CoreAudio device warm-up + the time the
                      # first sox-tempo subprocess takes to cold-start + any
                      # sounddevice ring-buffer drain between pre-roll write
                      # and first real samples arriving. Bumped from 200ms
                      # because 200 + 135ms (stretched 150ms WAV pad) wasn't
                      # covering the first-syllable cold start on Bluetooth/
                      # virtual audio outputs.
EDGE_FADE_MS = 3      # linear fade-in/out applied to every PLAY/EARCON buffer
                      # before streaming. Kokoro WAVs often end and begin mid-
                      # sample (non-zero amplitude), so direct concatenation
                      # produces an audible DC click at every sentence seam.
                      # 3 ms on each side is inaudible as a pause but
                      # eliminates the discontinuity.
POS_CADENCE_S = 0.020  # ~20 ms between POS emissions while playing

# Audio-leads-visual compensation (Kim et al. 2024, AJSLP).
# When the player loop writes a chunk, POS is naturally emitted against the
# write offset — but those samples are sitting in the CoreAudio ring
# buffer and haven't reached the speaker yet. Meanwhile the pipe→Lua→JS
# render pipeline adds its own latency. If ring > pipe, visual flashes
# BEFORE audio plays (the worst desync case per Kim 2024 — comprehension
# drops measurably). Compensating POS by the stream's output latency
# makes POS reflect speaker time, guaranteeing visual always lags audio.
#
# sounddevice exposes OutputStream.latency (seconds) once the stream is
# open. On macOS CoreAudio with default blocksize this is typically
# 20-30 ms. We cache it at startup.
_STREAM_OUTPUT_LATENCY_S: float = 0.0

# ---- shared state (protected by _lock) ----------------------------------------

_lock = threading.Lock()
_paused = False          # soft-pause flag; player thread checks this
# Pending relative seek in *source-domain* milliseconds. Resolution (which
# may walk across multiple history buffers and pull from _audio_queue) is
# done by the player thread under _lock — see _resolve_seek_ms_unlocked.
# Source-domain rather than stretched-samples so cross-buffer math doesn't
# need to know each buffer's tempo_factor at SEEK enqueue time.
_seek_request_ms: Optional[int] = None
_stop_requested = False  # STOP command sets this; player thread clears it
_shutdown_requested = False  # set by SIGTERM / EOF cleanup; player thread exits
_pending_timers: list = []   # threading.Timer instances to cancel on shutdown
_pending_timers_lock = threading.Lock()
# End-of-playback handshake. Set by EXIT command (the shell signalling
# "no more PLAYs coming"). The main thread keeps reading stdin so user
# controls (PAUSE/RESUME/SEEK) still work through the drain. The player
# thread, once it observes (queue empty AND kokoro_done AND pending
# Timers finished AND playhead at end of last history buffer), emits
# DRAIN_DONE on stderr (trackability breadcrumb) and closes stdin — that
# signals the main thread to fall through to the existing shutdown path.
# The shell's `wait $STREAM_PID` then returns and it tears down control
# plumbing. No os._exit, no protocol on the shell side beyond what
# already exists.
_kokoro_done = False
_drain_done_emitted = False

# ---- audio queue --------------------------------------------------------------
# Each item is one of:
#   ("samples", buffer_id: int, samples: np.ndarray, tempo_factor: float)
#   ("silence", ms: int)
# The player thread pulls from this and appends materialized buffers to
# _history (silence items become zero-amplitude buffers with buffer_id=-2
# so the player loop only has to handle one kind).

_audio_queue: queue.Queue = queue.Queue()

# ---- playback history (for cross-sentence scrubbing) --------------------------
#
# Every buffer the user has heard (or is about to hear, once pulled from
# _audio_queue) is kept here so SEEK can walk backward and forward across
# sentence boundaries — not just within the currently-playing buffer.
#
# Items are tuples: (buffer_id: int, samples: np.ndarray, tempo_factor: float)
#   buffer_id >=  0  → real synthesized speech, POS emits to RSVP
#   buffer_id == -1  → earcon (no POS)
#   buffer_id == -2  → materialized silence (no POS)
#
# _playhead_idx, _playhead_offset together identify the current playback
# position: idx into _history, then sample offset (in stretched-sample
# space) within that buffer. The player loop advances offset chunk by
# chunk; on natural end-of-buffer it bumps idx and resets offset to 0.
#
# Eviction (200 MB byte cap): when an append would push total bytes over
# _HISTORY_MAX_BYTES, drop buffers from the front — but only if the
# playhead is past them (_playhead_idx > 0). Never evict the buffer
# the user is currently scrubbing through; if they've scrubbed deep
# into history, the cap is a soft cap that recovers when they catch up.
# 200 MB ≈ 35 minutes of mono float32 24 kHz audio — well past any
# plausible single-selection size.
_history: list = []
_playhead_idx: int = 0
_playhead_offset: int = 0
_HISTORY_MAX_BYTES = 200 * 1024 * 1024

# ---- globals ------------------------------------------------------------------

_stream: Optional[sd.OutputStream] = None
_rate: float = 1.0   # updated by RATE commands


# ---- helpers ------------------------------------------------------------------

def _read_audio_as_float32(path: str) -> np.ndarray:
    """Read any audio file soundfile supports (WAV, AIFF, FLAC, …) and return
    float32 mono samples resampled to SAMPLE_RATE."""
    data, sr = sf.read(path, dtype="float32", always_2d=True)

    # Mix down to mono
    if data.shape[1] > 1:
        data = data.mean(axis=1)
    else:
        data = data[:, 0]

    # Resample to 24 kHz if the file is at a different rate (e.g. earcons
    # from /System/Library/Sounds/ are 48 kHz stereo AIFF).
    if sr != SAMPLE_RATE:
        # resample_poly needs integer ratio: find gcd(sr, SAMPLE_RATE)
        from math import gcd
        g = gcd(sr, SAMPLE_RATE)
        data = resample_poly(data, up=SAMPLE_RATE // g, down=sr // g)

    return data.astype(np.float32)


def _stretch(samples: np.ndarray, rate: float) -> np.ndarray:
    """Pitch-preserving time stretch via sox tempo (WSOLA phase vocoder).

    scipy.signal.resample_poly is NOT pitch-preserving — it's sample-rate
    conversion, so 1.33x faster becomes ~a third octave higher (chipmunk).
    Sox's `tempo -s` is a real time-domain phase vocoder that holds pitch
    steady while compressing duration; bench #101 confirmed its consonant
    preservation is within 0.25 dB of rubberband R3 at 1.33x. Cheaper
    than pulling pyrubberband / librosa as a Python dep.
    """
    if abs(rate - 1.0) < 1e-6:
        return samples
    buf = io.BytesIO()
    sf.write(buf, samples, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    try:
        proc = subprocess.run(
            ["sox", "-t", "wav", "-", "-t", "wav", "-", "tempo", "-s", f"{rate:.6f}"],
            input=buf.getvalue(),
            capture_output=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        # Sox missing or failed — play at 1.0x rather than corrupt pitch.
        print(f"play-stream: sox tempo unavailable, skipping stretch: {exc}",
              file=sys.stderr)
        return samples
    stretched, _ = sf.read(io.BytesIO(proc.stdout), dtype="float32", always_2d=False)
    return stretched.astype(np.float32)


def _silence_samples(ms: int) -> np.ndarray:
    n = int(SAMPLE_RATE * ms / 1000)
    return np.zeros(n, dtype=np.float32)


def _edge_fade(samples: np.ndarray, ms: int = EDGE_FADE_MS) -> np.ndarray:
    """Apply a linear fade-in at the head and fade-out at the tail. Mutates
    and returns the same array for callers that want to chain."""
    n = int(SAMPLE_RATE * ms / 1000)
    if len(samples) < 2 * n:
        return samples
    ramp_in = np.linspace(0.0, 1.0, n, dtype=np.float32)
    ramp_out = np.linspace(1.0, 0.0, n, dtype=np.float32)
    samples[:n] *= ramp_in
    samples[-n:] *= ramp_out
    return samples


def _stream_write(samples: np.ndarray) -> None:
    """Block-write samples into the stream (called from player thread only).

    Swallows exceptions because the stream may be aborted/closed underneath
    us during shutdown (cooperative-shutdown sequence aborts the stream to
    unblock a blocking write so the player thread can join). Without this,
    the abort raises a PortAudio error that would propagate out of the
    daemon thread mid-shutdown — sometimes after CFFI's underlying dylib
    is already torn down — and crash with EXC_BAD_ACCESS.
    """
    global _stream
    if _stream is None or len(samples) == 0:
        return
    try:
        _stream.write(samples)
    except Exception:
        # Either the stream was aborted (shutdown path) or PortAudio is
        # unhappy. Either way, not the player thread's job to recover —
        # main() owns lifecycle. Set shutdown so the loop exits.
        global _shutdown_requested
        with _lock:
            _shutdown_requested = True


# ---- POS emission -------------------------------------------------------------

def _emit_pos(ms_float: float, buffer_id: int) -> None:
    """Write a POS line to stderr. Using stderr with a 'POS ' prefix so the
    shell caller can tee stderr to a pipe and filter on the prefix."""
    print(f"POS {int(ms_float)} {buffer_id}", file=sys.stderr, flush=True)


# ---- player thread ------------------------------------------------------------

_CHUNK_SAMPLES = int(SAMPLE_RATE * POS_CADENCE_S)  # ~480 samples @ 24kHz → 20ms


def _history_byte_size_unlocked() -> int:
    total = 0
    for _, samples, _ in _history:
        total += samples.nbytes
    return total


def _history_append_unlocked(item: tuple) -> None:
    """Append (buffer_id, samples, tempo_factor) to history; evict from the
    front to honour _HISTORY_MAX_BYTES. Eviction adjusts _playhead_idx so
    the playhead still references the same buffer post-shift. We never
    evict the buffer the playhead is currently sitting on or any newer —
    that would yank audio out from under an active scrub. Caller holds
    _lock."""
    global _playhead_idx
    _history.append(item)
    # Only attempt eviction while front is strictly older than playhead.
    while (_history_byte_size_unlocked() > _HISTORY_MAX_BYTES
           and _playhead_idx > 0):
        _history.pop(0)
        _playhead_idx -= 1


def _materialize_queue_item(item: tuple) -> Optional[tuple]:
    """Convert a queue item to a (buffer_id, samples, tempo_factor) tuple
    suitable for _history. Silence items become zero-amplitude buffers
    with buffer_id=-2; samples items pass through unchanged."""
    kind = item[0]
    if kind == "silence":
        _, ms = item
        return (-2, _silence_samples(ms), 1.0)
    if kind == "samples":
        _, buf_id, samples, tempo = item
        return (buf_id, samples, tempo)
    return None


def _resolve_seek_ms_unlocked(delta_ms: int) -> None:
    """Walk _history forward/backward by `delta_ms` source-domain ms and
    update _playhead_idx, _playhead_offset. Pulls from _audio_queue
    non-blocking when a forward seek runs past the end of history;
    clamps at start of oldest available buffer / end of last queued
    buffer. Caller holds _lock."""
    global _playhead_idx, _playhead_offset

    if not _history:
        return  # pre-roll silence is enqueued before main starts the player,
                # but if somehow we got a SEEK before any item materialised,
                # there's nothing to seek into.

    if delta_ms > 0:
        remaining = delta_ms
        while remaining > 0:
            buf_id, samples, tempo = _history[_playhead_idx]
            buf_total_ms = len(samples) / SAMPLE_RATE * 1000.0 * tempo
            cur_ms = _playhead_offset / SAMPLE_RATE * 1000.0 * tempo
            ms_left_in_buf = buf_total_ms - cur_ms
            if remaining <= ms_left_in_buf:
                new_ms = cur_ms + remaining
                _playhead_offset = min(
                    len(samples),
                    int(new_ms * SAMPLE_RATE / (1000.0 * tempo)),
                )
                return
            # Past end of this buffer — try to advance to the next.
            remaining -= ms_left_in_buf
            if _playhead_idx + 1 < len(_history):
                _playhead_idx += 1
                _playhead_offset = 0
                continue
            # End of history. Pull from queue if anything's there;
            # otherwise clamp at end of last buffer.
            try:
                item = _audio_queue.get_nowait()
                _audio_queue.task_done()
            except queue.Empty:
                _playhead_offset = len(samples)
                return
            mat = _materialize_queue_item(item)
            if mat is None:
                _playhead_offset = len(samples)
                return
            _history_append_unlocked(mat)
            _playhead_idx += 1
            _playhead_offset = 0
        return

    if delta_ms < 0:
        remaining = delta_ms  # negative
        buf_id, samples, tempo = _history[_playhead_idx]
        cur_ms = _playhead_offset / SAMPLE_RATE * 1000.0 * tempo
        if cur_ms + remaining >= 0:
            new_ms = cur_ms + remaining
            _playhead_offset = max(
                0,
                int(new_ms * SAMPLE_RATE / (1000.0 * tempo)),
            )
            return
        # Walk backward through earlier buffers.
        remaining += cur_ms  # consume the way back to start of current buf
        _playhead_offset = 0
        while remaining < 0 and _playhead_idx > 0:
            _playhead_idx -= 1
            buf_id, samples, tempo = _history[_playhead_idx]
            buf_total_ms = len(samples) / SAMPLE_RATE * 1000.0 * tempo
            if -remaining <= buf_total_ms:
                new_ms = buf_total_ms + remaining  # remaining is negative
                _playhead_offset = max(
                    0,
                    int(new_ms * SAMPLE_RATE / (1000.0 * tempo)),
                )
                return
            remaining += buf_total_ms
        # Walked past the oldest available — clamp.
        _playhead_idx = 0
        _playhead_offset = 0
        return


def _drain_done_ready_unlocked() -> bool:
    """True iff the queue is empty AND every previously scheduled end-of-
    buffer Timer has finished firing AND the playhead has reached the end
    of the last history buffer. The end-of-history check matters during
    cross-buffer scrubbing: if the user has scrubbed back, we mustn't
    fire DRAIN_DONE just because EXIT was sent — they're still listening."""
    if not _audio_queue.empty():
        return False
    with _pending_timers_lock:
        for t in _pending_timers:
            if t.is_alive():
                return False
    if not _history:
        return True
    last_idx = len(_history) - 1
    if _playhead_idx < last_idx:
        return False
    if _playhead_idx == last_idx:
        if _playhead_offset < len(_history[last_idx][1]):
            return False
    return True


def _schedule_end_of_buffer_pos(buffer_id: int, samples: np.ndarray,
                                tempo_factor: float) -> None:
    """When a buffer plays through to its end (natural advance, not a seek-
    induced jump), schedule the final uncompensated POS to fire at the
    moment those last samples actually reach the speaker. See the long
    note on audio-leads-visual compensation for why this exists.

    No-op for non-speech buffers (silence/earcon, buffer_id < 0)."""
    if buffer_id < 0:
        return
    final_source_ms = len(samples) / SAMPLE_RATE * 1000.0 * tempo_factor
    final_delay_s = _STREAM_OUTPUT_LATENCY_S * tempo_factor
    if final_delay_s > 0.0:
        t = threading.Timer(
            final_delay_s,
            lambda sm=final_source_ms, bid=buffer_id: _emit_pos(sm, bid),
        )
        with _pending_timers_lock:
            _pending_timers.append(t)
        t.daemon = True
        t.start()
    else:
        _emit_pos(final_source_ms, buffer_id)


def _player_loop() -> None:
    """Drain _audio_queue → _history and write samples to the stream.

    Runs in a dedicated daemon thread so the main thread can keep reading
    stdin commands. Soft-pause is implemented by writing tiny silence
    chunks without advancing the playhead. The outer loop iterates on
    every tick — pause, drain-done check, seek resolution, and chunk
    write all happen at this level so none can starve the others. (The
    pre-Bug-2 code had an inner per-buffer loop that could spin on
    pause-write-silence forever, leaking the process when the user
    paused mid-last-buffer after EXIT.)
    """
    global _stream, _stop_requested, _drain_done_emitted, _seek_request_ms
    global _playhead_idx, _playhead_offset

    last_emit_buf_idx = -1   # detect buffer transitions for POS reset
    last_emit_pos = -1       # last-emitted POS in int source-ms

    while True:
        # Shutdown / STOP.
        with _lock:
            if _shutdown_requested:
                return
            if _stop_requested:
                _drain_queue()
                _stop_requested = False
                # Park playhead at end of whatever's in history so subsequent
                # PLAYs append cleanly and the loop waits for them.
                if _history:
                    _playhead_idx = len(_history) - 1
                    _playhead_offset = len(_history[_playhead_idx][1])
                continue
            paused = _paused
            seek_ms = _seek_request_ms
            _seek_request_ms = None

        # Apply any pending seek BEFORE the pause check so seeks issued while
        # paused take effect immediately. POS is emitted uncompensated so the
        # visual lands exactly where the user seeked to (no audio in flight
        # to lead with).
        if seek_ms is not None and seek_ms != 0:
            with _lock:
                _resolve_seek_ms_unlocked(seek_ms)
                if _playhead_idx < len(_history):
                    buf_id, samples, tempo = _history[_playhead_idx]
                    seek_source_ms = (
                        _playhead_offset / SAMPLE_RATE * 1000.0 * tempo
                    )
                else:
                    buf_id = -1
                    seek_source_ms = 0.0
            if buf_id >= 0:
                _emit_pos(seek_source_ms, buf_id)
                last_emit_pos = int(seek_source_ms)
            last_emit_buf_idx = _playhead_idx

        # End-of-playback handshake. Fires only when EXIT was sent AND
        # the queue is empty AND pending Timers are done AND the playhead
        # is at the end of the last history buffer. The last condition
        # protects mid-history scrub-backs and pause-mid-last-buffer:
        # in both states the user is still consuming, and shutting the
        # stream out from under them would cut audio.
        with _lock:
            kokoro_done = _kokoro_done
            already_emitted = _drain_done_emitted
            ready = _drain_done_ready_unlocked() if kokoro_done and not already_emitted else False
        if ready:
            print("DRAIN_DONE", file=sys.stderr, flush=True)
            with _lock:
                _drain_done_emitted = True
            try:
                sys.stdin.close()
            except Exception:
                pass

        # Pause: keep the stream alive with a tiny silence write, no playhead
        # advance. Drain-done check above still gets a chance every tick.
        if paused:
            _stream_write(_silence_samples(int(POS_CADENCE_S * 1000)))
            continue

        # Ensure the playhead points at a valid history slot. If it's at
        # len(_history), pull from the queue (briefly blocking so we don't
        # busy-loop while nothing's available).
        with _lock:
            need_pull = _playhead_idx >= len(_history)
        if need_pull:
            try:
                item = _audio_queue.get(timeout=POS_CADENCE_S)
            except queue.Empty:
                continue
            _audio_queue.task_done()
            mat = _materialize_queue_item(item)
            if mat is None:
                continue
            with _lock:
                _history_append_unlocked(mat)

        # Snapshot the current history slot.
        with _lock:
            if _playhead_idx >= len(_history):
                # Race: a STOP between the pull and here could reset the
                # playhead past the end. Loop and retry.
                continue
            buf_id, samples, tempo = _history[_playhead_idx]
            offset = _playhead_offset

        # New buffer? Reset emit tracking so the first POS for it isn't
        # suppressed by stale last_emit_pos from the previous buffer.
        if _playhead_idx != last_emit_buf_idx:
            last_emit_pos = -1
            last_emit_buf_idx = _playhead_idx

        # End of this buffer (natural advance)? Schedule end-of-buffer
        # final-POS Timer and step to the next history slot.
        if offset >= len(samples):
            _schedule_end_of_buffer_pos(buf_id, samples, tempo)
            with _lock:
                _playhead_idx += 1
                _playhead_offset = 0
            continue

        # Write one chunk and advance the playhead.
        end = min(offset + _CHUNK_SAMPLES, len(samples))
        chunk = samples[offset:end]
        _stream_write(chunk)
        with _lock:
            # Only advance if no concurrent SEEK landed elsewhere mid-write.
            if (_playhead_idx == last_emit_buf_idx
                    and _playhead_offset == offset):
                _playhead_offset = end

        # Emit POS in source-domain time, compensated for audio-leads-visual.
        # See the original derivation comment retained on the SEEK command.
        # source_ms = pos * 1000 * tempo / SAMPLE_RATE  (multiply by tempo).
        # Skip non-speech buffers (silence/earcon).
        if buf_id < 0:
            continue
        source_ms = end / SAMPLE_RATE * 1000.0 * tempo
        adjusted_ms = source_ms - _STREAM_OUTPUT_LATENCY_S * 1000.0 * tempo
        if adjusted_ms < 0:
            continue
        if int(adjusted_ms) != last_emit_pos:
            _emit_pos(adjusted_ms, buf_id)
            last_emit_pos = int(adjusted_ms)


def _drain_queue() -> None:
    """Empty the audio queue (called under _lock from player thread)."""
    while not _audio_queue.empty():
        try:
            _audio_queue.get_nowait()
            _audio_queue.task_done()
        except queue.Empty:
            break


# ---- signal handling ----------------------------------------------------------

_player_thread: Optional[threading.Thread] = None


def _shutdown() -> None:
    """Cooperative shutdown: stop the player thread, abort+close stream,
    cancel pending Timers. Safe to call from a signal handler or main().

    The order matters:
      1. Cancel pending Timers (so they don't fire after stderr/sys is gone).
      2. Set _shutdown_requested so the player loop returns next iteration.
      3. Abort the stream so any in-flight blocking _stream.write() returns
         immediately (otherwise the player thread can sit in CFFI for the
         duration of the buffer, racing with interpreter finalization).
      4. Join the player thread with a short timeout. If it's stuck in C,
         we let it die as a daemon — but the abort should unblock it well
         within the timeout in practice.
      5. Close the stream from the main thread, where lifecycle ownership
         lives.
    """
    global _stream, _shutdown_requested
    # 1) Cancel any pending POS Timers.
    with _pending_timers_lock:
        for t in _pending_timers:
            try:
                t.cancel()
            except Exception:
                pass
        _pending_timers.clear()
    # 2) Signal the player thread.
    with _lock:
        _shutdown_requested = True
    # 3) Abort to unblock a blocking write, if any.
    if _stream is not None:
        try:
            _stream.abort()
        except Exception:
            pass
    # 4) Join the player.
    if _player_thread is not None:
        try:
            _player_thread.join(timeout=0.5)
        except Exception:
            pass
    # 5) Close.
    if _stream is not None:
        try:
            _stream.close()
        except Exception:
            pass
        _stream = None


def _handle_sigterm(signum, frame):  # noqa: ARG001
    _shutdown()
    sys.exit(0)


signal.signal(signal.SIGTERM, _handle_sigterm)


# ---- main ---------------------------------------------------------------------

def main() -> None:
    global _stream, _rate, _paused, _stop_requested, _seek_request_ms
    global _player_thread, _kokoro_done

    global _STREAM_OUTPUT_LATENCY_S

    # Open the stream once for the entire session.
    _stream = sd.OutputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
        blocksize=0,   # let sounddevice choose an appropriate block size
    )
    _stream.start()

    # Cache the output latency for POS compensation. Post-start() this is a
    # float (seconds). Belt-and-braces: env var override (tunable without a
    # code change) and a floor of 0 / ceiling of 0.2 s (200 ms) against bad
    # backends reporting nonsense.
    lat_env = os.environ.get("KOKORO_AUDIO_LEAD_MS")
    if lat_env:
        try:
            _STREAM_OUTPUT_LATENCY_S = max(0.0, min(0.2, float(lat_env) / 1000.0))
        except ValueError:
            _STREAM_OUTPUT_LATENCY_S = 0.0
    else:
        raw = getattr(_stream, "latency", 0.0) or 0.0
        try:
            _STREAM_OUTPUT_LATENCY_S = max(0.0, min(0.2, float(raw)))
        except (TypeError, ValueError):
            _STREAM_OUTPUT_LATENCY_S = 0.0

    # Start the player thread as daemon so it exits when main() returns.
    # Also expose at module scope so _shutdown() can join it.
    _player_thread = threading.Thread(target=_player_loop, daemon=True)
    _player_thread.start()

    # 400 ms silent pre-roll: CoreAudio device ramp-up happens here, not
    # during the first real word.  Enqueue as a silence item so it's consumed
    # by the player thread in the same serialised order as everything else.
    _audio_queue.put(("silence", PRE_ROLL_MS))

    for raw_line in sys.stdin:
        line = raw_line.rstrip("\n")
        if not line:
            continue

        parts = line.split(" ", 2)
        cmd = parts[0].upper()

        if cmd == "RATE":
            try:
                _rate = float(parts[1]) if len(parts) > 1 else _rate
            except ValueError:
                pass  # ignore malformed RATE — keep previous value

        elif cmd == "PLAY":
            # Accept both:
            #   PLAY <id> <path>   (new form; id is integer)
            #   PLAY <path>        (backward-compat; id=0)
            if len(parts) >= 3:
                # Three-part: cmd id path
                try:
                    buf_id = int(parts[1])
                    path = parts[2].strip()
                except ValueError:
                    # parts[1] is not an integer — treat as old form
                    buf_id = 0
                    path = (parts[1] + (" " + parts[2] if len(parts) > 2 else "")).strip()
            elif len(parts) == 2:
                # Two-part: PLAY <path>
                buf_id = 0
                path = parts[1].strip()
            else:
                continue

            if not os.path.isfile(path):
                continue
            try:
                raw = _read_audio_as_float32(path)
                stretched = _stretch(raw, _rate)
                faded = _edge_fade(stretched)
                _audio_queue.put(("samples", buf_id, faded, _rate))
            except Exception as exc:
                print(f"play-stream: PLAY error {path}: {exc}", file=sys.stderr)

        elif cmd == "EARCON":
            path = parts[1].strip() if len(parts) > 1 else ""
            if not os.path.isfile(path):
                continue
            try:
                raw = _read_audio_as_float32(path)
                faded = _edge_fade(raw)  # no stretching for earcons
                # Earcons use tempo_factor=1.0 since they're not stretched.
                _audio_queue.put(("samples", -1, faded, 1.0))
            except Exception as exc:
                print(f"play-stream: EARCON error {path}: {exc}", file=sys.stderr)

        elif cmd == "PAUSE":
            # PAUSE with a positive integer arg → backward-compat silence insert.
            # Bare PAUSE (or non-integer arg) → soft-pause.
            arg = parts[1].strip() if len(parts) > 1 else ""
            try:
                ms = int(arg)
                if ms > 0:
                    _audio_queue.put(("silence", ms))
                # else: 0 or negative → treat as soft-pause
                    continue
            except ValueError:
                pass
            # Soft-pause.
            with _lock:
                _paused = True

        elif cmd == "RESUME":
            with _lock:
                _paused = False

        elif cmd == "SEEK":
            # SEEK <ms> — relative seek in source-domain milliseconds.
            # Positive = forward; negative = backward. Resolution (which
            # may walk across multiple history buffers and pull from the
            # audio queue) is deferred to the player thread, which holds
            # the lock while traversing _history. Successive SEEKs that
            # arrive before the player thread has resolved the previous
            # one accumulate (seek_request_ms += ms) so users can tap
            # the arrow key several times rapidly without dropping
            # any of the requested deltas.
            try:
                ms = int(parts[1]) if len(parts) > 1 else 0
            except ValueError:
                ms = 0
            if ms != 0:
                with _lock:
                    if _seek_request_ms is None:
                        _seek_request_ms = ms
                    else:
                        _seek_request_ms += ms

        elif cmd == "STOP":
            with _lock:
                _stop_requested = True

        elif cmd == "EXIT":
            # Cooperative end-of-playback signal from kokoro-tts.sh. The
            # shell has finished queueing audio but is leaving fd 9 open so
            # the control-fifo subshell keeps working — that means
            # PAUSE/RESUME/SEEK from Hammerspoon stay live through the
            # drain. We set _kokoro_done and KEEP READING stdin; the player
            # thread observes drain completion (queue empty + Timers done),
            # emits DRAIN_DONE on stderr, and closes our stdin to wind us
            # down. Process exit then unblocks the shell's `wait` and it
            # tears down the control plumbing.
            with _lock:
                _kokoro_done = True

        # else: unknown command — silently ignore for forward-compat

    # stdin EOF: wait for queue to drain, then run cooperative shutdown.
    # Same path as SIGTERM so the player thread gets a clean exit and any
    # pending end-of-buffer Timers are cancelled before the interpreter
    # tears down — preventing the post-finalization CFFI segfault.
    _audio_queue.join()
    _shutdown()


if __name__ == "__main__":
    main()
