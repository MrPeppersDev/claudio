#!/usr/bin/env python3
"""
Minimal local HTTP server wrapping kokoro-onnx.

Endpoints:
  GET  /health        -> {"ok": true, "voices": N, "loaded": bool}
  GET  /voices        -> {"voices": ["af_bella", ...]}  (sorted)
  POST /speak         -> audio/wav bytes
       body: {"text": "...", "voice": "bm_george", "speed": 1.3}

Why stdlib http.server (single-threaded): we serialize synthesis on purpose.
kokoro-onnx holds one ORT session; concurrent calls would just contend for
the same CPU/GPU path. The parent shell lock (job.pid in play-last.sh) already
guarantees only one synthesis request in flight, so serial is correct.

Design: model loads once at startup and stays resident. Subsequent /speak
calls skip the ~500ms load cost entirely.
"""
import io
import json
import math
import os
import re
import sys
import time
import wave
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
from kokoro_onnx import Kokoro
from scipy.signal import lfilter

HERE = os.path.dirname(os.path.abspath(__file__))
# Use the patched model (exposes duration-predictor outputs) when available.
# patch-model.sh produces kokoro-v1.0-durations.onnx from kokoro-v1.0.onnx;
# kokoro-server.sh calls patch-model.sh before launching this process, so
# the durations model should always be present at startup.
_MODEL_DURATIONS = os.path.join(HERE, "kokoro-v1.0-durations.onnx")
_MODEL_ORIGINAL = os.path.join(HERE, "kokoro-v1.0.onnx")
MODEL_PATH = _MODEL_DURATIONS if os.path.exists(_MODEL_DURATIONS) else _MODEL_ORIGINAL
VOICES_PATH = os.path.join(HERE, "voices-v1.0.bin")

HOST = os.environ.get("KOKORO_HOST", "127.0.0.1")
PORT = int(os.environ.get("KOKORO_PORT", "8880"))

# Bind-safety guard: the server has no auth. Binding to anything other than
# loopback would expose an unauthenticated TTS endpoint to the LAN (CPU burn,
# voice enumeration, at worst an OS-level vuln in onnxruntime). Refuse by
# default; force a deliberate opt-in via KOKORO_ALLOW_REMOTE=1 for anyone
# who really wants that (e.g., docker-bridge testing).
_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost", ""}
if HOST not in _LOOPBACK_HOSTS and os.environ.get("KOKORO_ALLOW_REMOTE") != "1":
    sys.stderr.write(
        f"[kokoro] refusing to bind {HOST!r}: no auth on this server.\n"
        f"[kokoro] loopback hosts: {sorted(_LOOPBACK_HOSTS)}.\n"
        f"[kokoro] set KOKORO_ALLOW_REMOTE=1 to acknowledge and proceed anyway.\n"
    )
    sys.exit(2)
if HOST not in _LOOPBACK_HOSTS:
    sys.stderr.write(
        f"[kokoro] WARNING: bound to {HOST} (non-loopback). "
        f"No auth — anyone reachable on this interface can call /speak.\n"
    )

# Kokoro-82M's hard per-call cap is 510 phonemes — the voice style array has
# shape (510, ...), and _create_audio indexes voice[len(tokens)] after
# truncating phonemes to 510 chars, so a 510-token call hits IndexError.
# Upstream _split_phonemes only splits between punctuated parts, so a single
# long unpunctuated span still blows up. We phonemize once, chunk to ≤500
# (leaves a 9-phoneme margin), and concatenate the per-chunk audio ourselves.
MAX_PHONEMES_PER_CALL = 500

# Hard cap on request body size. The only real payload is a /speak text blob
# (a few KB typically, tens of KB at most). Anything much larger is either
# a bug or a client trying to force a multi-GB allocation.
MAX_REQUEST_BYTES = 1_000_000  # 1 MB

# Supported phonemizer languages. Kokoro itself accepts more, but we ship with
# just the English variants — widening this set requires testing each addition
# for pronunciation quality.
SUPPORTED_LANGS = frozenset({"en-us", "en-gb"})

print(f"[kokoro] loading model from {MODEL_PATH}", flush=True)
t0 = time.time()
KOKORO = Kokoro(MODEL_PATH, VOICES_PATH)
VOICES = set(KOKORO.get_voices())
print(f"[kokoro] loaded in {time.time()-t0:.2f}s ({len(VOICES)} voices)", flush=True)

# Determine if the loaded model has duration outputs.  We check the session's
# output names rather than guessing from the filename so this works even if
# someone points MODEL_PATH at a custom export.
try:
    _sess_output_names = [o.name for o in KOKORO.sess.get_outputs()]
except AttributeError:
    # Stub/test environment where KOKORO.sess doesn't exist.
    _sess_output_names = []
_HAS_DURATION_OUTPUTS = "/encoder/Gather_output_0" in _sess_output_names
if _HAS_DURATION_OUTPUTS:
    print("[kokoro] duration predictor outputs detected — timing sidecars enabled", flush=True)
else:
    print("[kokoro] WARNING: duration outputs NOT found; timing sidecars disabled. "
          "Run kokoro/patch-model.sh to enable.", flush=True)


# Input bounds for /speak. Picked to cover realistic TTS use without inviting
# "oops, I DoS'd my own machine" floats. Chosen values are intentionally
# generous — anything outside them is almost certainly a client bug.
MAX_TEXT_CHARS = 50_000
SPEED_RANGE = (0.25, 4.0)
EQ_GAIN_DB_RANGE = (-24.0, 24.0)
EQ_FREQ_RANGE = (20.0, 20_000.0)
EQ_Q_RANGE = (0.1, 20.0)
# Leading silence (zero samples) prepended to the output WAV. Kokoro
# synthesizes each sentence as an independent utterance with no pre-roll,
# so the first phoneme's attack can sound smushed. A short pad (~80ms)
# gives the ear an onset cue without adding noticeable dead air.
PAD_START_MS_RANGE = (0.0, 1000.0)
# Trailing silence target. Kokoro bakes ~100-200ms of tail silence into
# every utterance; back-to-back sentences compound that with the next
# sentence's leading pad, so inter-sentence gaps drift to 250-350ms. We
# detect where audible signal ends (windowed RMS) and keep only this much.
TAIL_TRIM_MS_RANGE = (0.0, 1000.0)
# RMS energy threshold for "is this frame audible" when locating the tail.
# -45 dBFS  →  10 ** (-45/20) ≈ 0.0056. Chosen well above the noise floor
# but below any phoneme you'd want to keep — including unvoiced stops like
# t/p/k whose release bursts sit at 0.01-0.04 RMS.
TAIL_RMS_THRESHOLD = 10 ** (-45.0 / 20.0)  # ≈ 0.0056
# Frame length for RMS analysis: 10ms at Kokoro's native 24 kHz.
TAIL_RMS_FRAME_SAMPLES = 240  # 10ms @ 24000 Hz
# Dwell: how many consecutive below-threshold frames we require after the
# last audible frame before committing the trim point. 6 frames = 60ms.
# Unvoiced stop release bursts last 20-80ms (≤8 frames), so they survive
# as long as the silence after them hasn't accumulated 60ms yet.
TAIL_RMS_DWELL_FRAMES = 6  # 60ms

# Sacrificial head-word trim. Kokoro drops the first consonant/syllable of
# a sentence (the acoustic model "warms up" on the first token). The client
# can send a throwaway prefix word plus comma — "banana, recap." — and the
# server trims the prefix off before returning audio, so the playback starts
# mid-utterance at the real first word with full attack intact. The caller
# owns the word (typically "banana"); we just find the comma gap and cut.
#
# Search window: how far into the audio we'll look for the post-prefix gap.
HEAD_TRIM_SEARCH_MS = 1500
# Dwell required to recognize a silent run as the post-prefix gap. 20ms is
# tight enough to catch comma gaps that get compressed at fast synth speeds
# (e.g. 1.3x+) where the gap is often only 30-50ms.
HEAD_TRIM_DWELL_FRAMES = 2  # 20ms
# Relative silence threshold: a frame counts as silent if its RMS is below
# max(TAIL_RMS_THRESHOLD, peak_rms * HEAD_TRIM_REL_THRESHOLD). The peak is
# over the search window. Relative-to-peak adapts to voice loudness so fast
# or quiet synth still has a detectable trough at the comma; the absolute
# floor prevents all-silent input from picking a nonsense "peak".
HEAD_TRIM_REL_THRESHOLD = 0.15
# Min/max cut position bounds (at 1x synth speed — scaled by `speed` at call
# time). Min protects the body of the prefix word from being chopped if the
# detector fires on an intra-word dip. Max prevents the detector from walking
# past a missed comma gap into a natural intra-utterance pause (after e.g.
# "Pros:" or "Cons:") and chopping the real first word. Numbers sized for
# the default prefix "banana": ~430ms at 1x, +150ms comma = gap ends ~580ms,
# with a generous tail to accommodate voice/speed variance.
HEAD_TRIM_MIN_CUT_AT_1X_MS = 430
HEAD_TRIM_MAX_CUT_AT_1X_MS = 950


class ValidationError(ValueError):
    """Raised when a /speak payload field is out of range or malformed."""


def _parse_float(value, *, field: str, default: float) -> float:
    """
    Return ``value`` as a finite float, falling back to ``default`` when it
    comes in as None/empty. Rejects NaN, +/-Inf, and unparseable strings with
    a clear ValidationError so the handler can 400.
    """
    if value is None or value == "":
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        raise ValidationError(f"{field}: not a number ({value!r})")
    if not math.isfinite(parsed):
        raise ValidationError(f"{field}: must be finite (got {value!r})")
    return parsed


def _check_range(value: float, *, field: str, lo: float, hi: float) -> float:
    if not (lo <= value <= hi):
        raise ValidationError(f"{field}: {value} out of range [{lo}, {hi}]")
    return value


def resolve_voice(spec: str):
    """
    Parse a voice spec and return either a voice name (str) or a blended
    embedding (np.ndarray). Blend syntax: "name1:weight1,name2:weight2".
    Weights are normalized so they needn't sum to 1.

    Examples:
      "af_bella"                            -> "af_bella"           (str)
      "af_bella:70,am_michael:30"           -> blended ndarray
      "af_bella:1,am_michael:1,bf_emma:1"   -> equal 3-way blend

    Raises ValueError on unknown voice or zero total weight.
    """
    if "," not in spec and ":" not in spec:
        # Plain name — let KOKORO.create() validate by name.
        return spec
    entries = []
    for part in spec.split(","):
        if ":" in part:
            name, raw_w = part.split(":", 1)
            weight = float(raw_w.strip())
        else:
            name, weight = part, 1.0
        name = name.strip()
        if name not in VOICES:
            raise ValueError(f"unknown voice {name!r}")
        entries.append((name, weight))
    total = sum(w for _, w in entries)
    if total <= 0:
        raise ValueError("blend weights must sum to a positive number")
    blend = None
    for name, w in entries:
        contribution = KOKORO.get_voice_style(name) * (w / total)
        blend = contribution if blend is None else blend + contribution
    return blend.astype(np.float32)


def chunk_phonemes(phonemes: str, limit: int = MAX_PHONEMES_PER_CALL) -> list[str]:
    """
    Split a phoneme string into pieces each of length <= limit. Produces the
    fewest pieces possible given the length cap — every concat point adds an
    audible seam, so packing is worth the complexity.

    Strategy: split on the *strongest* boundary (sentence punct `.!?`), then
    greedy-pack the pieces back together up to the limit. For any single
    piece that still exceeds the limit, fall through to clause punct
    (`,;:—–`), then whitespace, then hard-cut.
    """
    if len(phonemes) <= limit:
        return [phonemes]

    sentence_pieces = _split_keeping_delims(phonemes, r"[.!?]")
    # If a piece is already over the limit (e.g. a 2000-char bulleted list
    # with no terminal punct), recurse into clause-level splitting first so
    # the greedy packer sees chunks it can actually fit.
    expanded: list[str] = []
    for p in sentence_pieces:
        if len(p) <= limit:
            expanded.append(p)
        else:
            expanded.extend(_chunk_on_clause(p, limit))
    return _greedy_pack(expanded, limit)


def _split_keeping_delims(text: str, delim_class: str) -> list[str]:
    """
    Split `text` such that each piece ends with its terminating delimiter
    (if one was present). e.g. "abc. def! ghi" + r"[.!?]" →
    ["abc.", " def!", " ghi"].
    """
    pattern = re.compile(rf"([^{delim_class[1:-1]}]*{delim_class})")
    pieces = pattern.findall(text)
    consumed = sum(len(p) for p in pieces)
    if consumed < len(text):
        pieces.append(text[consumed:])
    return [p for p in pieces if p]


def _chunk_on_clause(piece: str, limit: int) -> list[str]:
    """Second-tier split: `,;:—–` clause boundaries, then whitespace."""
    clause_pieces = _split_keeping_delims(piece, r"[,;:—–]")
    out: list[str] = []
    for cp in clause_pieces:
        if len(cp) <= limit:
            out.append(cp)
        else:
            out.extend(_chunk_on_whitespace(cp, limit))
    return out


def _chunk_on_whitespace(piece: str, limit: int) -> list[str]:
    """Third-tier: greedy-pack words. Hard-cut only if a single word > limit."""
    words = piece.split()
    if not words:
        return [piece]
    out: list[str] = []
    current = ""
    for w in words:
        if len(w) > limit:
            # Rare for phoneme strings, but possible for e.g. URLs that got
            # phonemized char-by-char. Flush and hard-cut the word.
            if current:
                out.append(current)
                current = ""
            for i in range(0, len(w), limit):
                out.append(w[i:i + limit])
            continue
        candidate = (current + " " + w) if current else w
        if len(candidate) <= limit:
            current = candidate
        else:
            out.append(current)
            current = w
    if current:
        out.append(current)
    return out


def _greedy_pack(pieces: list[str], limit: int) -> list[str]:
    """
    Concatenate `pieces` in order into chunks whose length never exceeds
    `limit`. Pieces are kept atomic — a piece that already fits the limit is
    never sliced further by the packer. Joining uses no separator; pieces
    are expected to retain their own leading/trailing whitespace.
    """
    out: list[str] = []
    current = ""
    for p in pieces:
        if not p:
            continue
        if len(current) + len(p) <= limit:
            current += p
        else:
            if current:
                out.append(current.strip())
            current = p
    if current:
        out.append(current.strip())
    return [c for c in out if c]


def peaking_eq(samples: np.ndarray, fs: int, f0: float, gain_db: float, q: float) -> np.ndarray:
    """
    Apply a peaking (parametric) EQ biquad at f0 with the given gain and Q.
    Formulas from the Audio EQ Cookbook (Robert Bristow-Johnson). A small
    presence bump (~3 dB at 2.5 kHz, Q ~ 1) helps consonants survive the
    1.33x afplay time-stretch that happens downstream.

    gain_db == 0 is treated as a no-op to avoid the numerical round-trip.
    """
    if gain_db == 0.0:
        return samples
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * math.pi * f0 / fs
    cos_w0 = math.cos(w0)
    sin_w0 = math.sin(w0)
    alpha = sin_w0 / (2.0 * q)

    b0 = 1.0 + alpha * A
    b1 = -2.0 * cos_w0
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha / A

    b = np.array([b0 / a0, b1 / a0, b2 / a0], dtype=np.float64)
    a = np.array([1.0, a1 / a0, a2 / a0], dtype=np.float64)
    return lfilter(b, a, samples).astype(samples.dtype)


def trim_tail_silence(samples: np.ndarray, sample_rate: int, target_ms: float) -> np.ndarray:
    """
    Trim trailing silence so only ``target_ms`` of tail remains after the
    last audible frame.

    Algorithm:
    1. Compute per-frame RMS energy in non-overlapping 10ms windows.
    2. Walk backward to find the last frame whose RMS exceeds
       TAIL_RMS_THRESHOLD.
    3. Require TAIL_RMS_DWELL_FRAMES (60ms) of continuous below-threshold
       frames after that anchor before committing the trim. This keeps
       unvoiced stop release bursts (t/p/k, 20-80ms) intact because their
       energy spike is audible and they haven't yet accumulated 60ms of
       silence behind them.
    4. Keep ``target_ms`` of silence after the anchor frame.

    Edge cases:
    - target_ms <= 0 → return unchanged (disable)
    - samples.size == 0 → return unchanged
    - all-silent input → return unchanged (no anchor to trim to)
    - existing tail already shorter than target_ms → return unchanged
    - signal shorter than one frame → return unchanged
    """
    if target_ms <= 0 or samples.size == 0:
        return samples

    frame = TAIL_RMS_FRAME_SAMPLES
    n_frames = samples.size // frame
    if n_frames == 0:
        return samples

    # Compute RMS per frame using reshape (avoids per-frame Python loop).
    frames = samples[:n_frames * frame].reshape(n_frames, frame).astype(np.float64)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))

    # Walk backward: find last frame above threshold followed by at least
    # TAIL_RMS_DWELL_FRAMES consecutive below-threshold frames.
    last_audible_frame = -1
    for i in range(n_frames - 1, -1, -1):
        if rms[i] > TAIL_RMS_THRESHOLD:
            # Check dwell: need TAIL_RMS_DWELL_FRAMES silent frames after i.
            dwell_end = i + 1 + TAIL_RMS_DWELL_FRAMES
            if dwell_end > n_frames:
                # Not enough frames left after i for the dwell requirement —
                # this could be a burst very close to the end of the signal.
                # Treat it as the anchor anyway (conservative: don't trim).
                last_audible_frame = i
                break
            # All frames in [i+1, i+1+DWELL) must be below threshold.
            if np.all(rms[i + 1:dwell_end] <= TAIL_RMS_THRESHOLD):
                last_audible_frame = i
                break

    if last_audible_frame < 0:
        # All silent — nothing to anchor the trim to.
        return samples

    keep_samples = int(round(sample_rate * target_ms / 1000.0))
    # Anchor at end of last audible frame.
    anchor_sample = (last_audible_frame + 1) * frame
    end = min(samples.size, anchor_sample + keep_samples)
    if end >= samples.size:
        # Existing tail is already no longer than target_ms — never extend.
        return samples
    return samples[:end]


def trim_sacrificial_head(
    samples: np.ndarray,
    sample_rate: int,
    speed: float = 1.0,
) -> np.ndarray:
    """
    Drop the sacrificial prefix word from the head of ``samples``.

    The client has asked Kokoro to synthesize ``"banana, {real_text}"``.
    The prefix warms Kokoro's acoustic model so the first consonant of
    ``real_text`` is fully formed. This function finds the comma-gap at
    the end of the prefix and cuts there.

    RMS-based detection with three guardrails:

    * **Relative threshold** — a frame is "silent" if its RMS is below
      ``max(TAIL_RMS_THRESHOLD, peak_rms * HEAD_TRIM_REL_THRESHOLD)``.
      Adapts to voice loudness: a compressed or quiet comma gap still
      reads as a clear trough.
    * **Min cut position** — ignore triggers before
      ``HEAD_TRIM_MIN_CUT_AT_1X_MS / speed`` ms. Protects the body of
      the prefix word from being chopped if an intra-word nasal/schwa
      dip crosses the threshold.
    * **Max cut position** — once we pass ``HEAD_TRIM_MAX_CUT_AT_1X_MS /
      speed`` ms without triggering, fail open. A gap found past that
      point is almost certainly an intra-utterance pause after the real
      first word (e.g. after "Pros:") — cutting there would chop the
      word out. Better to leak "banana," than to silently eat "Pros".

    Fails open on ambiguity: return ``samples`` unchanged rather than
    chop the real first word. Audible "banana," leaking through is the
    signal that the detector needs tuning — better than silent garbling.
    """
    if samples.size == 0:
        return samples

    frame = TAIL_RMS_FRAME_SAMPLES
    search_frames = min(
        samples.size // frame,
        int(round(HEAD_TRIM_SEARCH_MS * sample_rate / 1000.0)) // frame,
    )
    if search_frames <= HEAD_TRIM_DWELL_FRAMES:
        return samples

    window = samples[:search_frames * frame].astype(np.float64)
    rms = np.sqrt(np.mean(window.reshape(search_frames, frame) ** 2, axis=1))

    peak = float(rms.max()) if rms.size else 0.0
    threshold = max(TAIL_RMS_THRESHOLD, peak * HEAD_TRIM_REL_THRESHOLD)

    speed_scale = max(speed, 0.1)  # guard against zero/negative
    min_cut_frame = (
        int(round(HEAD_TRIM_MIN_CUT_AT_1X_MS * sample_rate / 1000.0 / speed_scale))
        // frame
    )
    max_cut_frame = (
        int(round(HEAD_TRIM_MAX_CUT_AT_1X_MS * sample_rate / 1000.0 / speed_scale))
        // frame
    )

    # Kokoro's ONNX model emits ~400 ms of leading silence before the first
    # phoneme (a model-warm-up artifact, independent of KOKORO_PAD_START_MS).
    # That silence shifts the entire banana-prefix window rightward on the
    # time axis, so min/max cut bounds measured from WAV t=0 reject the real
    # gap as "past max". Anchor the bounds to the first audible frame instead
    # — the bounds are semantically about "how far through banana are we",
    # not "how far into the WAV".
    first_audible = -1
    for i in range(search_frames):
        if rms[i] > threshold:
            first_audible = i
            break
    if first_audible < 0:
        return samples

    silent_run = 0
    for i in range(first_audible, search_frames):
        if rms[i] > threshold:
            silent_run = 0
            continue
        silent_run += 1
        if silent_run < HEAD_TRIM_DWELL_FRAMES:
            continue
        cut_frame = i + 1
        offset_frames = cut_frame - first_audible
        if offset_frames > max_cut_frame:
            return samples
        if offset_frames < min_cut_frame:
            continue
        return samples[cut_frame * frame:]
    return samples


def synth_with_durations(
    phonemes: str,
    voice,
    speed: float,
    lang: str,
):
    """
    Synthesize audio from a phoneme string and capture the Gather duration
    output from the patched model.

    Returns (audio_samples, sample_rate, gather_output_or_None).

    ``gather_output`` is a 1-D INT64 array of per-phoneme frame counts (one
    entry per phoneme token).  Only available when _HAS_DURATION_OUTPUTS is
    True; otherwise returns None so callers can fall back gracefully.

    This function replicates the logic of kokoro_onnx.Kokoro._create_audio()
    but calls sess.run() with explicit output names so we can capture the
    extra outputs the patched model exposes, without modifying the library.
    """
    from kokoro_onnx.config import MAX_PHONEME_LENGTH, SAMPLE_RATE

    # Replicate _create_audio: tokenize, build inputs, run session.
    phonemes_trunc = phonemes[:MAX_PHONEME_LENGTH]
    tokens = np.array(KOKORO.tokenizer.tokenize(phonemes_trunc), dtype=np.int64)
    # Voice style is indexed by token count (length of phoneme sequence).
    if isinstance(voice, str):
        voice_style = KOKORO.get_voice_style(voice)
    else:
        voice_style = voice
    voice_for_call = voice_style[len(tokens)]
    padded_tokens = [[0, *tokens.tolist(), 0]]

    # Build inputs matching the session's expected input names.
    input_names = [i.name for i in KOKORO.sess.get_inputs()]
    if "input_ids" in input_names:
        inputs = {
            "input_ids": padded_tokens,
            "style": np.array(voice_for_call, dtype=np.float32),
            "speed": np.array([speed], dtype=np.int32),
        }
    else:
        inputs = {
            "tokens": padded_tokens,
            "style": voice_for_call,
            "speed": np.ones(1, dtype=np.float32) * speed,
        }

    if _HAS_DURATION_OUTPUTS:
        # Request all outputs so we get the duration data alongside audio.
        outputs = KOKORO.sess.run(None, inputs)
        # Output order (patched model): [audio, Cast_output_0, Gather_output_0, CumSum_output_0]
        audio = outputs[0]
        # Find Gather by position in the session output list.
        gather_idx = _sess_output_names.index("/encoder/Gather_output_0")
        gather = np.asarray(outputs[gather_idx], dtype=np.int64).ravel()
        # The Gather output covers the padded token sequence (with leading/
        # trailing pad-0 tokens).  Strip those two boundary entries so the
        # length matches the phoneme token count.
        if len(gather) == len(tokens) + 2:
            gather = gather[1:-1]
    else:
        audio = KOKORO.sess.run(None, inputs)[0]
        gather = None

    return audio, SAMPLE_RATE, gather


def samples_to_wav(samples: np.ndarray, sample_rate: int) -> bytes:
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write(f"[kokoro] {self.address_string()} - {fmt % args}\n")

    def _json(self, code, obj):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"ok": True, "voices": len(VOICES), "loaded": True})
        elif self.path == "/voices":
            # Sorted so clients (the menu bar, for one) can display a stable
            # ordering without doing the sort themselves.
            self._json(200, {"voices": sorted(VOICES)})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/speak":
            self._json(404, {"error": "not found"})
            return
        # Content-Length required: we don't support chunked. Rejecting up
        # front with 411 is clearer than accepting 0 and 400ing later.
        if "Content-Length" not in self.headers:
            self._json(411, {"error": "Content-Length required"})
            return
        try:
            length = int(self.headers["Content-Length"])
        except ValueError:
            self._json(400, {"error": "Content-Length not an integer"})
            return
        if length <= 0:
            self._json(400, {"error": "empty body"})
            return
        if length > MAX_REQUEST_BYTES:
            self._json(413, {"error": f"body too large "
                                      f"({length} > {MAX_REQUEST_BYTES} bytes)"})
            return
        try:
            payload = json.loads(self.rfile.read(length))
        except json.JSONDecodeError as e:
            self._json(400, {"error": f"bad json: {e}"})
            return

        text = (payload.get("text") or "").strip()
        voice_spec = payload.get("voice") or "af_bella"
        lang = payload.get("lang") or "en-gb"
        sacrificial_head_word = (payload.get("sacrificial_head_word") or "").strip()

        if not text:
            self._json(400, {"error": "text required"})
            return
        # Keep the sacrificial word small and alphabetic — it runs through
        # Kokoro as-is, and anything exotic (punctuation, digits, multi-word
        # phrases) defeats the "one extra token of acoustic warm-up" trick.
        if sacrificial_head_word and (
            len(sacrificial_head_word) > 32
            or not sacrificial_head_word.replace("-", "").isalpha()
        ):
            self._json(400, {"error": "sacrificial_head_word must be a single "
                                      "alphabetic word ≤32 chars"})
            return
        if len(text) > MAX_TEXT_CHARS:
            self._json(400, {"error": f"text: {len(text)} chars exceeds "
                                      f"{MAX_TEXT_CHARS}"})
            return

        # EQ params are per-request (not env) so the client's cache key
        # can include them and toggling EQ auto-invalidates stale cache
        # entries instead of silently mixing pre/post-EQ audio.
        try:
            speed = _check_range(
                _parse_float(payload.get("speed"), field="speed", default=1.0),
                field="speed", lo=SPEED_RANGE[0], hi=SPEED_RANGE[1],
            )
            eq_gain_db = _check_range(
                _parse_float(payload.get("eq_gain_db"), field="eq_gain_db", default=0.0),
                field="eq_gain_db", lo=EQ_GAIN_DB_RANGE[0], hi=EQ_GAIN_DB_RANGE[1],
            )
            eq_freq = _check_range(
                _parse_float(payload.get("eq_freq"), field="eq_freq", default=2500.0),
                field="eq_freq", lo=EQ_FREQ_RANGE[0], hi=EQ_FREQ_RANGE[1],
            )
            eq_q = _check_range(
                _parse_float(payload.get("eq_q"), field="eq_q", default=1.0),
                field="eq_q", lo=EQ_Q_RANGE[0], hi=EQ_Q_RANGE[1],
            )
            pad_start_ms = _check_range(
                _parse_float(payload.get("pad_start_ms"), field="pad_start_ms", default=0.0),
                field="pad_start_ms", lo=PAD_START_MS_RANGE[0], hi=PAD_START_MS_RANGE[1],
            )
            tail_trim_ms = _check_range(
                _parse_float(payload.get("tail_trim_ms"), field="tail_trim_ms", default=0.0),
                field="tail_trim_ms", lo=TAIL_TRIM_MS_RANGE[0], hi=TAIL_TRIM_MS_RANGE[1],
            )
        except ValidationError as e:
            self._json(400, {"error": str(e)})
            return

        if lang not in SUPPORTED_LANGS:
            self._json(400, {"error": f"unsupported lang {lang!r}; "
                                      f"supported: {sorted(SUPPORTED_LANGS)}"})
            return

        try:
            voice = resolve_voice(voice_spec)
        except ValueError as e:
            self._json(400, {"error": str(e)})
            return

        # Prepend a throwaway word so Kokoro's first-token warm-up is spent
        # on the prefix instead of on the real first consonant. We trim the
        # prefix audio off after synthesis. The comma gives us a detectable
        # prosodic pause to cut at.
        synth_text = (
            f"{sacrificial_head_word}, {text}" if sacrificial_head_word else text
        )

        # Optional cache path for sidecar writing. When provided, the server
        # writes cache/<hash>.words.json alongside the WAV on the server's
        # filesystem. kokoro-tts.sh passes this when it has a definite cache
        # target path so the sidecar is written atomically with the WAV.
        cache_path = (payload.get("cache_path") or "").strip()

        t0 = time.time()
        try:
            # Phonemize once, then chunk into ≤MAX_PHONEMES_PER_CALL pieces
            # so we never trip Kokoro-82M's 510-phoneme IndexError on long
            # unpunctuated spans. Short inputs → one chunk → single create()
            # call, identical to the old path.
            phonemes = KOKORO.tokenizer.phonemize(synth_text, lang)
            chunks = chunk_phonemes(phonemes)
            parts = []
            gather_parts: list[np.ndarray] = []
            for chunk in chunks:
                part, sr, gather = synth_with_durations(chunk, voice, speed, lang)
                parts.append(part)
                if gather is not None:
                    gather_parts.append(gather)
            samples = parts[0] if len(parts) == 1 else np.concatenate(parts)
            # Concatenate gather outputs across chunks (if multiple).
            combined_gather = (
                np.concatenate(gather_parts) if gather_parts else None
            )
        except Exception as e:
            self._json(500, {"error": f"synth failed: {e}"})
            return
        t1 = time.time()
        samples = peaking_eq(samples, sr, eq_freq, eq_gain_db, eq_q)
        head_trimmed_samples = 0
        if sacrificial_head_word:
            before = samples.size
            samples = trim_sacrificial_head(samples, sr, speed=speed)
            head_trimmed_samples = before - samples.size
        # Trim before pad so the pad length is exactly what the client asked
        # for regardless of how much native tail Kokoro baked in.
        if tail_trim_ms > 0:
            samples = trim_tail_silence(samples, sr, tail_trim_ms)
        if pad_start_ms > 0:
            # Prepend zero samples after EQ so we don't run the filter over
            # silence (pointless cost; silence stays silent anyway). The
            # WAV header's frame count is derived from the array length in
            # samples_to_wav, so just lengthening the array is enough.
            pad_samples = int(round(sr * pad_start_ms / 1000.0))
            if pad_samples > 0:
                silence = np.zeros(pad_samples, dtype=samples.dtype)
                samples = np.concatenate([silence, samples])
        t2 = time.time()
        wav = samples_to_wav(samples, sr)
        t3 = time.time()
        head_trim_ms = (head_trimmed_samples * 1000.0 / sr) if sr else 0.0

        # Compute and write timing sidecar if we have duration data.
        sidecar_written = False
        if combined_gather is not None and cache_path:
            try:
                from kokoro.timing import compute_word_timings
                word_timings = compute_word_timings(
                    source_text=text,
                    gather_output=combined_gather,
                    lang=lang,
                    sample_rate=sr,
                    pad_start_ms=pad_start_ms,
                )
                sidecar = {
                    "version": 1,
                    "sample_rate": sr,
                    "audio_samples": samples.size,
                    "words": word_timings,
                }
                sidecar_path = cache_path.rsplit(".", 1)[0] + ".words.json"
                tmp_path = sidecar_path + ".tmp"
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(sidecar, f, indent=2)
                os.replace(tmp_path, sidecar_path)
                sidecar_written = True
                sys.stderr.write(f"[kokoro] sidecar written: {sidecar_path} ({len(word_timings)} words)\n")
            except Exception as e:
                sys.stderr.write(f"[kokoro] WARNING: sidecar write failed: {e}\n")

        sys.stderr.write(
            f"[kokoro] synth chars={len(text)} phonemes={len(phonemes)} "
            f"chunks={len(chunks)} voice={voice_spec} speed={speed} "
            f"eq={eq_gain_db}dB@{eq_freq}Hz pad={pad_start_ms}ms "
            f"trim={tail_trim_ms}ms "
            f"head_word={sacrificial_head_word or '-'} "
            f"head_trim={head_trim_ms:.0f}ms "
            f"synth={t1-t0:.2f}s eq_time={t2-t1:.3f}s "
            f"encode={t3-t2:.2f}s bytes={len(wav)} "
            f"sidecar={'yes' if sidecar_written else 'no'}\n"
        )

        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(wav)))
        self.end_headers()
        self.wfile.write(wav)


def _warmup():
    # onnxruntime JIT-compiles its execution graph lazily on the first
    # inference, not at model-load. Without this, the first real /speak
    # after boot pays ~500ms on top of actual synth. A throwaway call on
    # a short string here amortizes that cost before the socket opens,
    # so the first user-visible request lands on the warm path.
    try:
        default_voice = os.environ.get("KOKORO_VOICE", "af_bella")
        # Blend specs go through resolve_voice; a bare name passes through
        # as a string that KOKORO.create() accepts directly.
        voice = resolve_voice(default_voice.split(",")[0].split(":")[0])
        t = time.time()
        KOKORO.create("ready", voice=voice, speed=1.0, lang="en-us")
        print(f"[kokoro] warmed in {time.time()-t:.2f}s", flush=True)
    except Exception as e:  # noqa: BLE001 — warm-up must never block startup
        print(f"[kokoro] warm-up skipped: {e}", flush=True)


def main():
    _warmup()
    server = HTTPServer((HOST, PORT), Handler)
    print(f"[kokoro] listening on http://{HOST}:{PORT}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
