"""Fish Audio TTS over their HTTP API (model s2.1-pro by default), streamed as 8 kHz PCM."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

import httpx

FISH_URL = "https://api.fish.audio/v1/tts"


class FishTTS:
    def __init__(self) -> None:
        self.key = os.environ["FISH_API_KEY"]
        self.model = os.getenv("FISH_MODEL", "s2.1-pro")
        self.voices = {"en": os.getenv("FISH_VOICE_EN") or None, "hi": os.getenv("FISH_VOICE_HI") or None}
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=5.0))

    async def stream(self, text: str, language: str = "en") -> AsyncIterator[bytes]:
        """Yields 16-bit mono PCM at 8 kHz (even-length chunks)."""
        body = {"text": text, "format": "pcm", "sample_rate": 8000, "latency": "low",
                "normalize": True, "language": "hi" if language in ("hi", "hinglish") else "en"}
        voice = self.voices.get("hi" if language in ("hi", "hinglish") else "en")
        if voice:
            body["reference_id"] = voice
        headers = {"Authorization": f"Bearer {self.key}", "model": self.model}
        carry = b""
        async with self.client.stream("POST", FISH_URL, json=body, headers=headers) as r:
            if r.status_code != 200:
                raise RuntimeError(f"Fish TTS {r.status_code}: {(await r.aread())[:200]!r}")
            async for chunk in r.aiter_bytes():
                data = carry + chunk
                cut = len(data) - (len(data) % 2)
                carry = data[cut:]
                if cut:
                    yield data[:cut]

    async def aclose(self) -> None:
        await self.client.aclose()
