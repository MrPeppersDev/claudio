#!/usr/bin/env python3
"""play-stream.py — persistent sounddevice OutputStream for Kokoro TTS playback.

Reads commands from stdin (one per line) and writes audio samples into a
single long-lived OutputStream.  Keeping the stream open across sentences
means CoreAudio initialises once (during the 200ms silent pre-roll) so the
very first word is never clipped.

Protocol (each stdin line is one command):
  PLAY <absolute-path>   — time-stretch and play the WAV at the current RATE
  EARCON <absolute-path> — play the file at 1.0x (no stretching); resamples
                           to 24 kHz if needed (handles .aiff earcons)
  PAUSE <ms>             — insert <ms> of silence into the stream
  RATE <float>           — set playback rate for subsequent PLAY commands

Exits cleanly on stdin EOF or SIGTERM.
"""

import os
import signal
import sys

import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.signal import resample_poly

# ---- constants ----------------------------------------------------------------

SAMPLE_RATE = 24000   # Kokoro native output rate; stream stays at this rate
CHANNELS = 1
DTYPE = "float32"
PRE_ROLL_MS = 200     # silent pre-roll so CoreAudio ramp-up doesn't eat word 1

# ---- globals ------------------------------------------------------------------

_stream: sd.OutputStream | None = None
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
    """Pitch-preserving time stretch via resample_poly (primary) with
    fallback to scipy.signal.resample if resample_poly fails."""
    if abs(rate - 1.0) < 1e-6:
        return samples
    # resample_poly(samples, up, down) outputs a signal of length
    #   input_len * up / down
    # To shorten by rate > 1.0: output_len = input_len / rate
    #   → up=10000, down=int(10000 * rate)
    up = 10000
    down = int(round(10000.0 * rate))
    if down <= 0:
        down = 1
    try:
        return resample_poly(samples, up=up, down=down).astype(np.float32)
    except Exception:
        # Fallback: not pitch-preserving but handles edge cases
        target_len = max(1, int(len(samples) / rate))
        from scipy.signal import resample as _resample
        return _resample(samples, target_len).astype(np.float32)


def _silence(ms: int) -> np.ndarray:
    n = int(SAMPLE_RATE * ms / 1000)
    return np.zeros(n, dtype=np.float32)


def _write(samples: np.ndarray) -> None:
    """Block-write samples into the stream."""
    global _stream
    if _stream is None or len(samples) == 0:
        return
    _stream.write(samples)


# ---- signal handling ----------------------------------------------------------

def _handle_sigterm(signum, frame):  # noqa: ARG001
    global _stream
    if _stream is not None:
        try:
            _stream.abort()
        except Exception:
            pass
    sys.exit(0)


signal.signal(signal.SIGTERM, _handle_sigterm)


# ---- main ---------------------------------------------------------------------

def main() -> None:
    global _stream, _rate

    # Open the stream once for the entire session.
    _stream = sd.OutputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
        blocksize=0,   # let sounddevice choose an appropriate block size
    )
    _stream.start()

    # 200 ms silent pre-roll: CoreAudio device ramp-up happens here, not
    # during the first real word.
    _write(_silence(PRE_ROLL_MS))

    for raw_line in sys.stdin:
        line = raw_line.rstrip("\n")
        if not line:
            continue

        parts = line.split(" ", 1)
        cmd = parts[0].upper()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd == "RATE":
            try:
                _rate = float(arg)
            except ValueError:
                pass  # ignore malformed RATE — keep previous value

        elif cmd == "PLAY":
            path = arg.strip()
            if not os.path.isfile(path):
                continue
            try:
                samples = _read_audio_as_float32(path)
                stretched = _stretch(samples, _rate)
                _write(stretched)
            except Exception as exc:
                print(f"play-stream: PLAY error {path}: {exc}", file=sys.stderr)

        elif cmd == "EARCON":
            path = arg.strip()
            if not os.path.isfile(path):
                continue
            try:
                samples = _read_audio_as_float32(path)
                _write(samples)  # no stretching for earcons
            except Exception as exc:
                print(f"play-stream: EARCON error {path}: {exc}", file=sys.stderr)

        elif cmd == "PAUSE":
            try:
                ms = int(arg)
            except ValueError:
                ms = 0
            if ms > 0:
                _write(_silence(ms))

        # else: unknown command — silently ignore for forward-compat

    # stdin EOF: drain the stream then exit cleanly.
    try:
        _stream.stop()
        _stream.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
