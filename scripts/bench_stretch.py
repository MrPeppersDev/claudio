#!/usr/bin/env python3
"""
bench_stretch.py — Benchmark rubberband R3 vs sox-tempo (afplay analog) for
the 1.33x time-stretch we apply on top of Kokoro's 1.5x synth (effective 2.0x).

Usage:
    python3 scripts/bench_stretch.py

Outputs:
    bench/<n>-raw.wav         Kokoro synthesis at speed=1.5
    bench/<n>-afplay.wav      sox-tempo stretched at 1.333
    bench/<n>-rubberband.wav  rubberband R3 formant-preserving at 1.333
    bench/<n>-compare.png     spectrogram comparison grid

Prints a markdown metrics table at the end.
"""

import io
import json
import os
import pathlib
import subprocess
import sys
import wave

import numpy as np
import requests
import scipy.signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BENCH_DIR = pathlib.Path("bench")
SERVER_URL = "http://127.0.0.1:8880/speak"
STRETCH_RATIO = 1.333        # time-compression (faster → shorter duration)
FS = 24000                   # Kokoro output sample rate
CONSONANT_LO = 1500          # Hz — consonant-band low
CONSONANT_HI = 4000          # Hz — consonant-band high
CENTROID_WINDOW_MS = 50      # ms — spectral centroid slide window

SENTENCES = [
    (
        "dense-technical",
        "Gate now matches content. Show the 'N dependent entries' line only when "
        "there is, in fact, a dependent count greater than zero or an explicit "
        "dependents list.",
    ),
    (
        "short-punchy",
        "Done. Nice work. Let's ship it.",
    ),
    (
        "number-heavy",
        "Revenue grew 37.8 percent to 4.2 million dollars in Q4 2024, up from "
        "3.05 million the prior year.",
    ),
    (
        "long-prose",
        "The quick brown fox jumps over the lazy dog, and as it lands on the "
        "mossy stone, it realizes the forest has grown quieter than before, as "
        "though every creature is holding its breath in anticipation.",
    ),
    (
        "code-adjacent",
        "Call getUserById with a numeric identifier. The function returns null "
        "when the user does not exist.",
    ),
]

SPEAK_BODY_TEMPLATE = {
    "voice": "af_bella",
    "speed": 1.5,
    "lang": "en-us",
    "eq_gain_db": 3.0,
    "eq_freq": 2500,
    "eq_q": 1.0,
    "pad_start_ms": 0,
    "tail_trim_ms": 0,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def synthesize(text: str, out_path: pathlib.Path) -> None:
    """POST to Kokoro server and write WAV."""
    body = dict(SPEAK_BODY_TEMPLATE, text=text)
    resp = requests.post(SERVER_URL, json=body, timeout=60)
    resp.raise_for_status()
    out_path.write_bytes(resp.content)
    print(f"  synthesized → {out_path} ({len(resp.content)//1024} KB)")


def run(cmd: list[str]) -> None:
    """Run a subprocess, raising on failure."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(cmd)}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )


def read_wav_samples(path: pathlib.Path) -> tuple[np.ndarray, int]:
    """Read a WAV file and return (float32 samples [-1,1], sample_rate)."""
    with wave.open(str(path), "rb") as wf:
        n_frames = wf.getnframes()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        rate = wf.getframerate()
        raw = wf.readframes(n_frames)
    if sampwidth == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2**31
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)
    return samples, rate


def consonant_band_energy_db(samples: np.ndarray, fs: int) -> float:
    """Integrate Welch PSD in the consonant band, return dB."""
    nperseg = min(2048, len(samples) // 4)
    if nperseg < 64:
        return float("nan")
    freqs, psd = scipy.signal.welch(samples, fs=fs, nperseg=nperseg)
    freq_res = freqs[1] - freqs[0]
    mask = (freqs >= CONSONANT_LO) & (freqs <= CONSONANT_HI)
    band_power = float(np.sum(psd[mask]) * freq_res)
    if band_power <= 0:
        return float("-inf")
    return 10.0 * np.log10(band_power)


def spectral_centroid_cv(samples: np.ndarray, fs: int) -> float:
    """
    Slide a 50ms window, compute spectral centroid per window,
    return coefficient of variation (std/mean).
    """
    win_len = int(fs * CENTROID_WINDOW_MS / 1000)
    hop = win_len // 2
    centroids = []
    for start in range(0, len(samples) - win_len, hop):
        chunk = samples[start : start + win_len]
        spectrum = np.abs(np.fft.rfft(chunk))
        freqs = np.fft.rfftfreq(win_len, d=1.0 / fs)
        total = float(np.sum(spectrum))
        if total > 1e-10:
            centroid = float(np.dot(freqs, spectrum) / total)
            centroids.append(centroid)
    if len(centroids) < 2:
        return float("nan")
    arr = np.array(centroids)
    return float(np.std(arr) / np.mean(arr)) if np.mean(arr) > 0 else float("nan")


def duration_seconds(path: pathlib.Path) -> float:
    with wave.open(str(path), "rb") as wf:
        return wf.getnframes() / wf.getframerate()


def save_spectrogram_comparison(
    n: int,
    label: str,
    raw_path: pathlib.Path,
    afplay_path: pathlib.Path,
    rb_path: pathlib.Path,
) -> None:
    """Save a 3-panel spectrogram comparison figure."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    fig.suptitle(f"Sentence {n}: {label}", fontsize=11)

    for ax, path, title in zip(
        axes,
        [raw_path, afplay_path, rb_path],
        ["raw (speed=1.5)", "sox-tempo 1.333×", "rubberband R3 1.333×"],
    ):
        samples, _ = read_wav_samples(path)
        ax.specgram(samples, NFFT=2048, Fs=FS, noverlap=1536, cmap="inferno")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("freq (Hz)")
        ax.set_ylim(0, 8000)

    plt.tight_layout()
    out = BENCH_DIR / f"{n}-compare.png"
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print(f"  spectrogram → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    BENCH_DIR.mkdir(parents=True, exist_ok=True)

    rows = []

    for idx, (label, text) in enumerate(SENTENCES, start=1):
        print(f"\n[{idx}/5] {label}")

        raw_path    = BENCH_DIR / f"{idx}-raw.wav"
        afplay_path = BENCH_DIR / f"{idx}-afplay.wav"
        rb_path     = BENCH_DIR / f"{idx}-rubberband.wav"

        # 1. Synthesize
        synthesize(text, raw_path)

        # 2. sox-tempo stretch (afplay analog)
        run(["sox", str(raw_path), str(afplay_path), "tempo", "-s", "1.333"])
        print(f"  sox-tempo   → {afplay_path}")

        # 3. rubberband R3 formant-preserving stretch
        # -T = tempo ratio (>1 = faster, shorter output), matching sox tempo behaviour.
        # Equivalent to -t 0.750 (time ratio), but mirrors the sox "tempo 1.333" intent.
        run(["rubberband", "-3", "-F", "-T", "1.333", str(raw_path), str(rb_path)])
        print(f"  rubberband  → {rb_path}")

        # 4. Metrics
        raw_dur    = duration_seconds(raw_path)
        aff_dur    = duration_seconds(afplay_path)
        rb_dur     = duration_seconds(rb_path)

        aff_samples, aff_fs = read_wav_samples(afplay_path)
        rb_samples,  rb_fs  = read_wav_samples(rb_path)

        aff_cb  = consonant_band_energy_db(aff_samples, aff_fs)
        rb_cb   = consonant_band_energy_db(rb_samples,  rb_fs)
        aff_cv  = spectral_centroid_cv(aff_samples, aff_fs)
        rb_cv   = spectral_centroid_cv(rb_samples,  rb_fs)

        aff_ratio = raw_dur / aff_dur if aff_dur > 0 else float("nan")
        rb_ratio  = raw_dur / rb_dur  if rb_dur  > 0 else float("nan")

        rows.append({
            "n": idx,
            "label": label,
            "aff_ratio": aff_ratio,
            "rb_ratio": rb_ratio,
            "aff_cb_db": aff_cb,
            "rb_cb_db": rb_cb,
            "cb_delta": rb_cb - aff_cb,
            "aff_cv": aff_cv,
            "rb_cv": rb_cv,
        })

        # 5. Spectrograms
        save_spectrogram_comparison(idx, label, raw_path, afplay_path, rb_path)

    # ---------------------------------------------------------------------------
    # Print markdown table
    # ---------------------------------------------------------------------------
    print("\n\n## Metrics\n")
    header = (
        "| # | label | aff dur-ratio | rb dur-ratio "
        "| aff cband dB | rb cband dB | Δ cband dB "
        "| aff centroid CV | rb centroid CV |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|"
    print(header)
    print(sep)
    for r in rows:
        print(
            f"| {r['n']} | {r['label']} "
            f"| {r['aff_ratio']:.3f} | {r['rb_ratio']:.3f} "
            f"| {r['aff_cb_db']:.2f} | {r['rb_cb_db']:.2f} "
            f"| {r['cb_delta']:+.2f} "
            f"| {r['aff_cv']:.4f} | {r['rb_cv']:.4f} |"
        )

    mean_delta = float(np.mean([r["cb_delta"] for r in rows]))
    print(f"\n**Mean Δ consonant-band energy (rubberband − afplay): {mean_delta:+.2f} dB**")

    # Export JSON for the comment script
    metrics_path = BENCH_DIR / "metrics.json"
    metrics_path.write_text(json.dumps({"rows": rows, "mean_delta_db": mean_delta}, indent=2))
    print(f"\nMetrics JSON → {metrics_path}")


if __name__ == "__main__":
    main()
