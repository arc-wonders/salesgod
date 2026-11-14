"""Fish TTS wrapper: clean 24k -> 8k conversion, fades, request settings (owner 15 Nov: voice must sound natural)."""

import asyncio
import json

import httpx
import numpy as np

from voice.tts import FishTTS, StreamDownsampler


def test_downsampler_streaming_equals_one_shot():
    x = (np.random.default_rng(1).standard_normal(48000) * 0.1).astype(np.float32)
    one = StreamDownsampler(24000).push(x)
    d = StreamDownsampler(24000)
    cuts = [0, 777, 1778, 2111, 6207, 6208, 48000]
    streamed = np.concatenate([d.push(x[a:b]) for a, b in zip(cuts, cuts[1:])])
    assert len(one) == len(streamed) == 16000
    assert np.abs(one - streamed).max() < 1e-6


def test_downsampler_keeps_speech_band_and_removes_aliasing():
    t = np.arange(24000) / 24000
    keep = StreamDownsampler(24000).push(np.sin(2 * np.pi * 1000 * t).astype(np.float32))
    alias = StreamDownsampler(24000).push(np.sin(2 * np.pi * 5000 * t).astype(np.float32))
    assert abs(np.abs(keep[200:]).max() - 1.0) < 0.02
    assert np.abs(alias[200:]).max() < 0.01  # 5 kHz would fold back to 3 kHz without the low-pass


def _tts(monkeypatch, rate: int, pcm: np.ndarray, seen: dict, split: int = 1000) -> FishTTS:
    monkeypatch.setenv("FISH_API_KEY", "k")
    monkeypatch.setenv("FISH_RATE", str(rate))
    raw = np.round(pcm * 32767).astype("<i2").tobytes()

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, content=raw)

    t = FishTTS()
    t.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return t


def _collect(t: FishTTS) -> np.ndarray:
    async def go() -> bytes:
        return b"".join([c async for c in t.stream("hello", "en")])
    out = asyncio.run(go())
    assert len(out) % 2 == 0
    return np.frombuffer(out, "<i2").astype(np.float32) / 32768


def test_new_path_requests_24k_with_low_temperature_and_fades(monkeypatch):
    seen: dict = {}
    loud = np.full(24000, 0.5, dtype=np.float32)  # 1 s of constant level: any click shows at the edges
    out = _collect(_tts(monkeypatch, 24000, loud, seen))
    assert seen["body"]["sample_rate"] == 24000 and seen["body"]["latency"] == "low"
    assert seen["body"]["temperature"] == 0.3 and seen["body"]["top_p"] == 0.5
    assert abs(len(out) - 8000) <= 2
    assert abs(out[0]) < 0.02 and abs(out[-1]) < 0.02  # faded in and out
    assert abs(out[4000] - 0.5) < 0.01  # body untouched: no gain change


def test_old_8k_path_request_unchanged(monkeypatch):
    seen: dict = {}
    out = _collect(_tts(monkeypatch, 8000, np.full(8000, 0.5, dtype=np.float32), seen))
    assert seen["body"]["sample_rate"] == 8000
    assert "temperature" not in seen["body"] and "top_p" not in seen["body"]
    assert len(out) == 8000
