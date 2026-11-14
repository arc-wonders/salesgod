"""Prosody features from a customer utterance: pitch level, pitch variation, loudness.

Pitch (F0) is tracked with YIN (de Cheveigné & Kawahara 2002) in plain numpy, 40 ms frames,
10 ms hop, 75-400 Hz. Telephone audio cuts everything below ~300 Hz, but YIN still finds the
period from the harmonics, so F0 survives the phone line (validated in tests/test_prosody.py).

Raw Hz means little across people (a calm woman's voice can be higher than an angry man's),
so the estimator compares each turn with the SAME caller's earlier turns (see acie.state.estimator).
"""

from __future__ import annotations

import numpy as np

from acie.schemas import Prosody

RATE = 16000
FRAME = 640  # 40 ms
HOP = 160  # 10 ms
FMIN, FMAX = 75.0, 400.0
YIN_THRESHOLD = 0.15


def _frames(x: np.ndarray) -> np.ndarray:
    n = 1 + (len(x) - FRAME) // HOP
    if n <= 0:
        return np.zeros((0, FRAME), dtype=np.float32)
    idx = np.arange(FRAME)[None, :] + HOP * np.arange(n)[:, None]
    return x[idx]


def f0_track(audio: np.ndarray, rate: int = RATE) -> tuple[np.ndarray, np.ndarray]:
    """Returns (f0 per frame in Hz, 0 where unvoiced; rms per frame)."""
    x = audio.astype(np.float32)
    fr = _frames(x)
    if len(fr) == 0:
        return np.zeros(0), np.zeros(0)
    rms = np.sqrt((fr ** 2).mean(axis=1))
    tmin, tmax = int(rate / FMAX), int(rate / FMIN)
    w = FRAME - tmax
    # difference function d(tau) = sum (x[j] - x[j+tau])^2 over the first w samples
    base = fr[:, :w]
    d = np.empty((len(fr), tmax + 1), dtype=np.float32)
    d[:, 0] = 0
    for tau in range(1, tmax + 1):
        diff = base - fr[:, tau:tau + w]
        d[:, tau] = (diff * diff).sum(axis=1)
    # cumulative mean normalised difference
    cum = np.cumsum(d[:, 1:], axis=1)
    cmnd = np.ones_like(d)
    cmnd[:, 1:] = d[:, 1:] * np.arange(1, tmax + 1)[None, :] / np.maximum(cum, 1e-9)
    f0 = np.zeros(len(fr))
    gate = rms > max(0.01, 0.1 * np.percentile(rms, 95))  # ignore silence / line noise
    for i in np.nonzero(gate)[0]:
        c = cmnd[i, tmin:]
        below = np.nonzero(c < YIN_THRESHOLD)[0]
        if len(below) == 0:
            continue
        t = below[0]
        while t + 1 < len(c) and c[t + 1] < c[t]:  # walk down to the local minimum
            t += 1
        tau = t + tmin
        # parabolic interpolation for sub-sample period
        if 1 <= tau < tmax:
            a, b, cc = cmnd[i, tau - 1], cmnd[i, tau], cmnd[i, tau + 1]
            den = a - 2 * b + cc
            tau = tau + (0.5 * (a - cc) / den if abs(den) > 1e-9 else 0.0)
        f0[i] = rate / tau
    # drop isolated octave jumps: keep frames within an octave of the median
    voiced = f0 > 0
    if voiced.sum() >= 3:
        med = np.median(f0[voiced])
        f0[voiced & ((f0 > 2 * med) | (f0 < med / 2))] = 0
    return f0, rms


def prosody_from_track(f0: np.ndarray, rms: np.ndarray, n_samples: int, words: int = 0,
                       rate: int = RATE) -> Prosody | None:
    voiced = f0[f0 > 0]
    if len(voiced) < 10:  # < 0.1 s of voicing: not enough to say anything
        return None
    st = 12 * np.log2(voiced / 100.0)  # semitones re 100 Hz: perceptual scale
    speech = rms > max(0.01, 0.1 * np.percentile(rms, 95))
    db = 20 * np.log10(np.maximum(rms[speech], 1e-6))
    dur = n_samples / rate
    return Prosody(
        f0_median_hz=round(float(np.median(voiced)), 1),
        f0_median_st=round(float(np.median(st)), 2),
        f0_std_st=round(float(np.std(st)), 2),
        f0_range_st=round(float(np.percentile(st, 90) - np.percentile(st, 10)), 2),
        loudness_db=round(float(np.mean(db)), 1) if len(db) else -60.0,
        voiced_ratio=round(float(len(voiced) / max(len(f0), 1)), 2),
        words_per_s=round(words / dur, 2) if words and dur > 0 else None,
    )


def extract(audio: np.ndarray, words: int = 0, rate: int = RATE) -> Prosody | None:
    f0, rms = f0_track(audio, rate)
    return prosody_from_track(f0, rms, len(audio), words, rate)
