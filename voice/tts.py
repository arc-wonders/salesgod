"""Fish Audio TTS over their HTTP API (model s2.1-pro-free by default: free, no API credit needed), streamed as 8 kHz PCM.

Natural-voice settings (owner 15 Nov: the voice sounded modulated, not human):
- the same text with the same reference voice came back at pitch 117 / 140 / 129 Hz and 7.0 / 7.4 / 8.0 s on three
  requests (Fish samples randomly), so the agent's pitch and pace jumped from one reply to the next. A low
  temperature / top_p keeps each reply close to the same voice and tempo (FISH_TEMPERATURE, FISH_TOP_P).
- Fish is asked for 24 kHz and we do ONE clean 24k -> 8k conversion (windowed-sinc low-pass, streamed) instead of
  letting the 8 kHz request path decide; FISH_RATE=8000 restores the old path.
- every reply starts and ends with a short fade so there is no click or clipped word at the joins.
- latency stays "low": measured "normal" costs 3-4.5 s before the first audio, too slow for a phone call.
No pitch or speed is ever derived from the caller's voice here.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

import httpx
import numpy as np

FISH_URL = "https://api.fish.audio/v1/tts"
LINE_RATE = 8000
FADE_MS = 8


class StreamDownsampler:
    """Streaming integer-ratio downsampler: Kaiser-windowed sinc low-pass (cut-off 3.6 kHz at 8 kHz out), then decimate."""

    def __init__(self, src_rate: int, dst_rate: int = LINE_RATE, taps: int = 121) -> None:
        if src_rate % dst_rate:
            raise ValueError("integer ratio only")
        self.n = src_rate // dst_rate
        fc = 0.45 * dst_rate / src_rate  # cycles per source sample
        k = np.arange(taps) - (taps - 1) / 2
        h = 2 * fc * np.sinc(2 * fc * k) * np.kaiser(taps, 8.0)
        self.h = (h / h.sum()).astype(np.float32)
        self.tail = np.zeros(taps - 1, dtype=np.float32)
        self.phase = 0  # offset of the next kept sample inside the next block (carried between blocks)

    def push(self, x: np.ndarray) -> np.ndarray:
        buf = np.concatenate([self.tail, x.astype(np.float32)])
        y = np.convolve(buf, self.h, mode="valid")  # len(x) samples
        out = y[self.phase::self.n]
        self.phase = (self.phase - len(y)) % self.n
        self.tail = buf[len(buf) - (len(self.h) - 1):]
        return out


def _pcm(f: np.ndarray) -> bytes:
    return np.clip(np.round(f * 32768.0), -32768, 32767).astype("<i2").tobytes()


class FishTTS:
    def __init__(self) -> None:
        self.key = os.environ["FISH_API_KEY"]
        self.model = os.getenv("FISH_MODEL", "s2.1-pro-free")
        self.voices = {"en": os.getenv("FISH_VOICE_EN") or None, "hi": os.getenv("FISH_VOICE_HI") or None}
        self.rate = int(os.getenv("FISH_RATE", "24000"))
        self.latency = os.getenv("FISH_LATENCY", "low")
        self.temperature = float(os.getenv("FISH_TEMPERATURE", "0.3"))
        self.top_p = float(os.getenv("FISH_TOP_P", "0.5"))
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=5.0))

    async def stream(self, text: str, language: str = "en") -> AsyncIterator[bytes]:
        """Yields 16-bit mono PCM at 8 kHz (even-length chunks), faded in and out."""
        hi = language in ("hi", "hinglish")
        body = {"text": text, "format": "pcm", "sample_rate": self.rate, "latency": self.latency,
                "normalize": True, "language": "hi" if hi else "en"}
        if self.rate != LINE_RATE:  # new path only; FISH_RATE=8000 keeps the old request exactly
            body["temperature"] = self.temperature
            body["top_p"] = self.top_p
        voice = self.voices.get("hi" if hi else "en")
        if voice:
            body["reference_id"] = voice
        headers = {"Authorization": f"Bearer {self.key}", "model": self.model}
        carry = b""
        down = StreamDownsampler(self.rate) if self.rate != LINE_RATE else None
        fade_n = int(LINE_RATE * FADE_MS / 1000)
        held = np.zeros(0, dtype=np.float32)  # last few samples held back so the end can be faded
        started = False
        async with self.client.stream("POST", FISH_URL, json=body, headers=headers) as r:
            if r.status_code != 200:
                raise RuntimeError(f"Fish TTS {r.status_code}: {(await r.aread())[:200]!r}")
            async for chunk in r.aiter_bytes():
                data = carry + chunk
                cut = len(data) - (len(data) % 2)
                carry = data[cut:]
                if not cut:
                    continue
                x = np.frombuffer(data[:cut], dtype="<i2").astype(np.float32) / 32768.0
                f = np.concatenate([held, down.push(x) if down is not None else x])
                if len(f) <= fade_n:
                    held = f
                    continue
                out, held = f[:-fade_n].copy(), f[-fade_n:]
                if not started:
                    m = min(len(out), fade_n)
                    out[:m] *= np.linspace(0.0, 1.0, m, dtype=np.float32)
                    started = True
                yield _pcm(out)
        if len(held):
            held = held.copy()
            if not started:
                m = min(len(held), fade_n)
                held[:m] *= np.linspace(0.0, 1.0, m, dtype=np.float32)
            held *= np.linspace(1.0, 0.0, len(held), dtype=np.float32)
            yield _pcm(held)

    async def aclose(self) -> None:
        await self.client.aclose()
