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
import time
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
# When _play_buffer writes a chunk, POS is naturally emitted against the
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
_seek_delta_samples: Optional[int] = None  # pending relative seek; set by SEEK cmd
_stop_requested = False  # STOP command sets this; player thread clears it
_shutdown_requested = False  # set by SIGTERM / EOF cleanup; player thread exits
_pending_timers: list = []   # threading.Timer instances to cancel on shutdown
_pending_timers_lock = threading.Lock()

# ---- audio queue --------------------------------------------------------------
# Each item is one of:
#   ("samples", buffer_id: int, samples: np.ndarray, tempo_factor: float)
#   ("silence", ms: int)
# The player thread drains this.

_audio_queue: queue.Queue = queue.Queue()

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


def _player_loop() -> None:
    """Drain _audio_queue and write samples to the stream.

    Runs in a dedicated daemon thread so the main thread can keep reading
    stdin commands.  Soft-pause is implemented by spinning without consuming
    queue items (the stream stays alive and silent via zero-writes).
    """
    global _stream, _stop_requested

    while True:
        # Check for shutdown — exit the loop entirely so main() can join.
        with _lock:
            if _shutdown_requested:
                return

        # Check for STOP first — clears queue and continues.
        with _lock:
            if _stop_requested:
                _drain_queue()
                _stop_requested = False
                continue

        # Check pause — spin-wait with tiny sleep to avoid busy-loop.
        with _lock:
            paused = _paused
        if paused:
            # Write a small silence chunk to keep the stream alive.
            _stream_write(_silence_samples(int(POS_CADENCE_S * 1000)))
            continue

        # Try to get the next item.
        try:
            item = _audio_queue.get(timeout=POS_CADENCE_S)
        except queue.Empty:
            continue

        kind = item[0]

        if kind == "silence":
            _, ms = item
            _stream_write(_silence_samples(ms))
            _audio_queue.task_done()

        elif kind == "samples":
            _, buffer_id, samples, tempo_factor = item
            _play_buffer(buffer_id, samples, tempo_factor)
            _audio_queue.task_done()


def _drain_queue() -> None:
    """Empty the audio queue (called under _lock from player thread)."""
    while not _audio_queue.empty():
        try:
            _audio_queue.get_nowait()
            _audio_queue.task_done()
        except queue.Empty:
            break


def _play_buffer(buffer_id: int, samples: np.ndarray, tempo_factor: float) -> None:
    """Play a buffer of (already-stretched) samples in chunks, emitting POS.

    The samples are already time-stretched; the tempo_factor is the sox rate
    that was applied so we can recover source-domain position.

    Formula derivation:
      Source WAV: N samples at SAMPLE_RATE → duration = N/SAMPLE_RATE seconds.
      sox tempo R (R>1 = faster): output has N/R stretched samples.
      When player has consumed `pos` stretched samples:
        fraction_complete = pos / (N/R) = pos*R/N
        source_time_s     = fraction_complete * (N/SAMPLE_RATE)
                          = pos * R / SAMPLE_RATE
        source_ms         = pos / SAMPLE_RATE * 1000 * R
    So: source_ms = pos * 1000 * tempo_factor / SAMPLE_RATE  (MULTIPLY by R).

    SEEK delta: the fd-9 SEEK command is in source-domain ms. Converting to
    the stretched-sample domain:
        source_samples = ms * SAMPLE_RATE / 1000
        stretched_delta = source_samples / tempo_factor  (DIVIDE by R)
    because each source sample corresponds to 1/R stretched samples.
    """
    global _stop_requested, _seek_delta_samples

    pos = 0  # current position in samples array
    total = len(samples)
    last_pos_emit = -1  # track when we last emitted a POS

    while pos < total:
        # Check STOP.
        with _lock:
            if _stop_requested:
                return

        # Apply any pending seek BEFORE the pause check so seeks issued while
        # paused take effect immediately (otherwise they'd queue in
        # _seek_delta_samples and only apply on resume — and successive seeks
        # would overwrite, so only the last one would survive).
        with _lock:
            delta = _seek_delta_samples
            _seek_delta_samples = None
        if delta is not None:
            # delta is in source-domain samples; convert to stretched domain
            # (source_samples → stretched: divide by tempo_factor)
            stretched_delta = int(delta / tempo_factor)
            pos = max(0, min(total, pos + stretched_delta))

        # Check pause — spin-wait, write silence to keep stream alive.
        with _lock:
            paused = _paused
        if paused:
            _stream_write(_silence_samples(int(POS_CADENCE_S * 1000)))
            continue

        # Write one chunk.
        end = min(pos + _CHUNK_SAMPLES, total)
        chunk = samples[pos:end]
        _stream_write(chunk)
        pos = end

        # Emit POS in source-domain time, compensated for audio-leads-visual.
        #
        # Raw: source_ms = pos * 1000 * tempo_factor / SAMPLE_RATE.
        #      This is the position of the just-WRITTEN sample, which is
        #      sitting in CoreAudio's ring buffer and hasn't played yet.
        #
        # Compensated: subtract the ring-buffer latency converted to
        #      source-domain ms. Result = source position currently AT THE
        #      SPEAKER. Downstream visual-pipeline latency then lands
        #      visual onset slightly AFTER audio onset — the direction
        #      Kim et al. 2024 found protects comprehension.
        #
        # During the first ~latency*tempo ms of a buffer the adjusted value
        # is negative — that audio isn't at the speaker yet. Skip those so
        # rsvp doesn't flash a word before its sound plays.
        source_ms = pos / SAMPLE_RATE * 1000.0 * tempo_factor
        adjusted_ms = source_ms - _STREAM_OUTPUT_LATENCY_S * 1000.0 * tempo_factor
        if adjusted_ms < 0:
            continue
        if int(adjusted_ms) != last_pos_emit:
            _emit_pos(adjusted_ms, buffer_id)
            last_pos_emit = int(adjusted_ms)

    # End-of-buffer catch-up. The compensated POS stream stops
    # `latency*tempo` source-ms short of the buffer's true end — otherwise the
    # last word's visual would fire before its audio. That leaves the RSVP
    # balloon frozen on whichever word contains that final POS while audio
    # plays through the remaining ~latency*tempo ms of already-buffered
    # samples. Schedule one final uncompensated POS at source_ms_end, delayed
    # by the same compensation, so it lands at the speaker simultaneously
    # with audio's actual end — advancing the balloon's final word without
    # violating audio-leads-visual.
    final_source_ms = total / SAMPLE_RATE * 1000.0 * tempo_factor
    final_delay_s = _STREAM_OUTPUT_LATENCY_S * tempo_factor
    if final_delay_s > 0.0:
        t = threading.Timer(
            final_delay_s,
            lambda sm=final_source_ms, bid=buffer_id: _emit_pos(sm, bid),
        )
        # Track so cooperative shutdown can cancel before the interpreter
        # tears down — otherwise a Timer firing post-finalization can crash
        # via stderr/print after sys.stderr is closed.
        with _pending_timers_lock:
            _pending_timers.append(t)
        t.daemon = True
        t.start()
    else:
        _emit_pos(final_source_ms, buffer_id)


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
    global _stream, _rate, _paused, _stop_requested, _seek_delta_samples
    global _player_thread

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
            # Positive = forward; negative = backward.
            # Clamping to [0, buffer_duration] is done in the player thread
            # where we have the actual buffer length.
            try:
                ms = int(parts[1]) if len(parts) > 1 else 0
            except ValueError:
                ms = 0
            if ms != 0:
                # Convert source-domain ms to source-domain samples.
                delta_source_samples = int(ms * SAMPLE_RATE / 1000)
                with _lock:
                    _seek_delta_samples = delta_source_samples

        elif cmd == "STOP":
            with _lock:
                _stop_requested = True

        elif cmd == "EXIT":
            # Cooperative exit from kokoro-tts.sh teardown. Break out of the
            # stdin loop so we drain the queue normally and shut down — vs.
            # SIGTERM, which is fast/abort. The shell sends EXIT *instead of*
            # closing fd 9 so the control-fifo subshell (which holds a dup of
            # fd 9) can stay alive through the drain, keeping PAUSE/RESUME/
            # SEEK working during end-of-playback.
            break

        # else: unknown command — silently ignore for forward-compat

    # stdin EOF: wait for queue to drain, then run cooperative shutdown.
    # Same path as SIGTERM so the player thread gets a clean exit and any
    # pending end-of-buffer Timers are cancelled before the interpreter
    # tears down — preventing the post-finalization CFFI segfault.
    _audio_queue.join()
    _shutdown()


if __name__ == "__main__":
    main()
