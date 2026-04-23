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

import io
import os
import signal
import subprocess
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
