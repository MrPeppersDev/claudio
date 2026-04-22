#!/usr/bin/env python3
"""
Minimal local HTTP server wrapping kokoro-onnx.

Endpoints:
  GET  /health        -> {"ok": true, "voices": N, "loaded": bool}
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
import os
import sys
import time
import wave
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
from kokoro_onnx import Kokoro

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "kokoro-v1.0.onnx")
VOICES_PATH = os.path.join(HERE, "voices-v1.0.bin")

HOST = os.environ.get("KOKORO_HOST", "127.0.0.1")
PORT = int(os.environ.get("KOKORO_PORT", "8880"))

print(f"[kokoro] loading model from {MODEL_PATH}", flush=True)
t0 = time.time()
KOKORO = Kokoro(MODEL_PATH, VOICES_PATH)
VOICES = set(KOKORO.get_voices())
print(f"[kokoro] loaded in {time.time()-t0:.2f}s ({len(VOICES)} voices)", flush=True)


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
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/speak":
            self._json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            self._json(400, {"error": "empty body"})
            return
        try:
            payload = json.loads(self.rfile.read(length))
        except json.JSONDecodeError as e:
            self._json(400, {"error": f"bad json: {e}"})
            return

        text = (payload.get("text") or "").strip()
        voice = payload.get("voice") or "bm_george"
        speed = float(payload.get("speed") or 1.0)
        lang = payload.get("lang") or "en-gb"

        if not text:
            self._json(400, {"error": "text required"})
            return
        if voice not in VOICES:
            self._json(400, {"error": f"unknown voice {voice!r}"})
            return

        t0 = time.time()
        try:
            samples, sr = KOKORO.create(text, voice=voice, speed=speed, lang=lang)
        except Exception as e:
            self._json(500, {"error": f"synth failed: {e}"})
            return
        t1 = time.time()
        wav = samples_to_wav(samples, sr)
        t2 = time.time()
        sys.stderr.write(
            f"[kokoro] synth chars={len(text)} voice={voice} speed={speed} "
            f"synth={t1-t0:.2f}s encode={t2-t1:.2f}s bytes={len(wav)}\n"
        )

        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(wav)))
        self.end_headers()
        self.wfile.write(wav)


def main():
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
