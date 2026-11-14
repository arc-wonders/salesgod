"""Audio helpers: G.711 mu-law (the phone line codec Plivo streams) and resampling.

mu-law tables follow the reference g711.c (adapted from uno_pipeline/backend/app/audio.py).
"""

from __future__ import annotations

import numpy as np

LINE_RATE = 8000  # Plivo stream: audio/x-mulaw at 8 kHz
STT_RATE = 16000  # Whisper + Silero VAD


def pcm16_to_float(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def _ulaw_encode_table() -> np.ndarray:
    v = np.arange(-32768, 32768, dtype=np.int32) >> 2
    mask = np.where(v < 0, 0x7F, 0xFF)
    mag = np.minimum(np.abs(v), 8159) + 0x21
    seg = np.searchsorted(np.array([0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF]), mag)
    uval = np.where(seg >= 8, 0x7F, (seg << 4) | ((mag >> (seg + 1)) & 0x0F))
    return ((uval ^ mask) & 0xFF).astype(np.uint8)


def _ulaw_decode_table() -> np.ndarray:
    u = ~np.arange(256, dtype=np.int32) & 0xFF
    exponent, mantissa = (u >> 4) & 0x07, u & 0x0F
    mag = (((mantissa << 3) + 0x84) << exponent) - 0x84
    return np.where(u & 0x80, -mag, mag).astype(np.float32) / 32768.0


_ULAW_ENC = _ulaw_encode_table()
_ULAW_DEC = _ulaw_decode_table()


def ulaw_to_float(data: bytes) -> np.ndarray:
    return _ULAW_DEC[np.frombuffer(data, dtype=np.uint8)]


def pcm16_to_ulaw(data: bytes) -> bytes:
    pcm = np.frombuffer(data, dtype="<i2").astype(np.int32)
    return _ULAW_ENC[pcm + 32768].tobytes()


def upsample_8k_to_16k(audio: np.ndarray) -> np.ndarray:
    """Exact 2x linear interpolation; good enough for VAD + ASR on telephone audio."""
    if len(audio) == 0:
        return audio
    out = np.empty(len(audio) * 2, dtype=np.float32)
    out[0::2] = audio
    out[1:-1:2] = (audio[:-1] + audio[1:]) / 2
    out[-1] = audio[-1]
    return out
