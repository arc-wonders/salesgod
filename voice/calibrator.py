"""Per-call voice calibrator: learns the line's noise and the caller's normal voice from the moment
they pick up, then reads every later turn against THAT person's own baseline.

Why per caller: raw pitch says almost nothing across people (a calm woman can sit higher than an
angry man), and phone lines differ by 10+ dB in level and noise. Changes from the same person on the
same line are what carry emotion.

What it tracks (all cheap numpy, ~10-60 ms per turn on CPU, run in parallel with Whisper):
  noise floor  RMS of the line while nobody speaks (VAD prob < 0.15, agent not talking)
  pitch        median F0 in semitones (YIN, voice/prosody.py)
  pitch_var    F0 spread within the turn (agitation widens it, tiredness flattens it)
  loudness     mean dBFS of speech frames
  effort       energy 1-3.4 kHz vs 0.3-1 kHz in dB: raised/strained voices push energy up the
               spectrum even when the phone's gain control flattens loudness
  rate         syllable nuclei per second of speech (energy peaks), language independent

Emotion is read as ACOUSTIC AROUSAL vs baseline (that is what voice reliably carries) and the
direction (angry vs excited) comes from the words, see `label()`. This is what is realistic: pitch +
loudness + effort up means "activated"; the voice alone cannot tell anger from delight.
"""

from __future__ import annotations

import json
import math
import os
from collections import deque
from pathlib import Path

import numpy as np

from acie.schemas import Prosody, VoiceReading

from .prosody import FRAME, HOP, RATE, _frames, f0_track, prosody_from_track

FEATURES = ("pitch", "pitch_var", "loudness", "effort", "rate")
# Normal turn-to-turn spread of ONE calm caller (1 "z" unit). Defaults; replaced by the values fitted
# on real calls (voice/calibrate_fit.py -> voice/assets/calib_stats.json) when that file exists.
DEFAULT_SPREAD = {"pitch": 1.5, "pitch_var": 0.6, "loudness": 3.0, "effort": 2.5, "rate": 0.9}
WEIGHTS = {"pitch": 0.32, "pitch_var": 0.18, "loudness": 0.22, "effort": 0.18, "rate": 0.10}
STATS_PATH = Path(__file__).parent / "assets" / "calib_stats.json"
NOISE_PROB = 0.15  # VAD probability below which a frame is "line, not voice"
MAX_BASELINE = 12
MAX_SHIFT_ST = 7.0  # emotional pitch shifts stay well under this; bigger = another voice
# The pickup "Hello?" is not the person's normal voice (fitted on real calls: ~1.4 dB quieter and
# ~1.8 dB duller than their later turns). A raw pickup baseline read "fix this shit" as LOW on a real
# call, so: with a fitted pickup offset the pickup is shifted and used from turn 2 at half confidence;
# without one, emotion waits for 2+ turns with 1 s+ of voicing. The pickup leaves the baseline once
# 3 later turns exist.
MIN_BASE_TURNS = 2
MIN_BASE_VOICED_S = 1.0


def _load_stats() -> tuple[dict[str, float], dict[str, float]]:
    """(spread, pickup offset) fitted on real calls, with safe defaults."""
    spread, pickup = dict(DEFAULT_SPREAD), {}
    try:
        st = json.loads(STATS_PATH.read_text())
        # never let a tiny fitted spread make the reading jumpy
        spread.update({k: max(float(v), 0.5 * DEFAULT_SPREAD[k]) for k, v in st.get("spread", {}).items()
                       if k in spread})
        # how the pickup "Hello?" differs from the same person's normal voice, clamped to 1.5 spreads
        pickup = {k: max(-1.5 * spread[k], min(1.5 * spread[k], float(v)))
                  for k, v in st.get("pickup_offset", {}).items() if k in spread}
    except (OSError, ValueError):
        pass
    return spread, pickup


def _db(x: float) -> float:
    return 20 * math.log10(max(x, 1e-6))


def _fill_short_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    """Treat dips shorter than max_gap frames (between syllables / words) as speech; longer = a pause."""
    out = mask.copy()
    idx = np.nonzero(mask)[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= max_gap:
            out[a:b] = True
    return out


def syllable_rate(rms: np.ndarray, speech: np.ndarray) -> float | None:
    """Articulation rate: energy peaks (syllable nuclei) per second of talking, pauses >= 250 ms
    excluded. rms is per 10 ms hop. Language independent (no transcript needed)."""
    if speech.sum() < 20:
        return None
    env = 20 * np.log10(np.maximum(rms, 1e-6))
    env = np.convolve(env, np.ones(5) / 5, mode="same")  # 50 ms smoothing
    peaks, last = 0, -100
    for i in range(1, len(env) - 1):
        if not speech[i] or i - last < 8:  # <= 12 syllables/s
            continue
        win = env[max(0, i - 8):i + 9]
        if env[i] == win.max() and env[i] - win.min() >= 2.0:
            peaks, last = peaks + 1, i
    talking = _fill_short_gaps(speech, 25).sum() * HOP / RATE
    return peaks / talking if talking >= 0.3 else None


def spectral_effort(audio: np.ndarray, speech: np.ndarray) -> float | None:
    fr = _frames(audio.astype(np.float32))[:len(speech)]
    fr = fr[speech[:len(fr)]]
    if len(fr) < 5:
        return None
    spec = (np.abs(np.fft.rfft(fr * np.hanning(FRAME), axis=1)) ** 2).mean(axis=0)
    f = np.fft.rfftfreq(FRAME, 1 / RATE)
    lo = spec[(f >= 300) & (f < 1000)].sum()
    hi = spec[(f >= 1000) & (f < 3400)].sum()
    return 10 * math.log10(max(hi, 1e-12) / max(lo, 1e-12))


def measure(audio: np.ndarray, words: int = 0) -> tuple[Prosody | None, dict | None]:
    """One pass over a 16 kHz utterance: (Prosody for ACIE, calibrator feature dict)."""
    f0, rms = f0_track(audio)
    prosody = prosody_from_track(f0, rms, len(audio), words)
    if prosody is None:
        return None, None
    speech = rms > max(0.01, 0.1 * np.percentile(rms, 95))
    feats = {
        "pitch": prosody.f0_median_st,
        # robust spread (0.74 x IQR): plain std is dominated by tracking jitter at syllable edges
        "pitch_var": round(0.7413 * float(np.subtract(*np.percentile(12 * np.log2(f0[f0 > 0] / 100.0), [75, 25]))), 2),
        # power mean over voiced frames: does not move with how many quiet edge frames pass the gate
        "loudness": round(10 * math.log10(max(float(np.mean(rms[f0 > 0] ** 2)), 1e-12)), 1),
        "effort": spectral_effort(audio, speech),
        "rate": syllable_rate(rms, speech),
        "voiced_s": float((f0 > 0).sum() * HOP / RATE),
    }
    return prosody, feats


class VoiceCalibrator:
    """One per call. Feed it line frames (`observe`) and accepted caller turns (`read` then `accept`)."""

    def __init__(self, spread: dict[str, float] | None = None, pickup_offset: dict[str, float] | None = None):
        if spread is None:
            spread, pickup_offset = _load_stats()
        self.spread = spread
        self.pickup_offset = pickup_offset or {}
        self.noise_rms: deque[float] = deque(maxlen=160)  # last ~5 s of quiet line (32 ms frames)
        self.base: list[dict] = []

    # ---------------------------------------------------------------- line noise

    def observe(self, frame: np.ndarray, vad_prob: float, agent_speaking: bool = False) -> None:
        if vad_prob < NOISE_PROB and not agent_speaking:  # our own voice echoes back: not line noise
            self.noise_rms.append(float(np.sqrt(np.mean(frame * frame))))

    @property
    def noise_db(self) -> float | None:
        if len(self.noise_rms) < 15:  # ~0.5 s of quiet line needed
            return None
        return round(_db(float(np.median(self.noise_rms))), 1)

    @property
    def noise_level(self) -> float | None:
        return None if self.noise_db is None else 10 ** (self.noise_db / 20)

    # ---------------------------------------------------------------- caller baseline

    @property
    def calibrated(self) -> bool:
        if len(self.base) == 1 and self.pickup_offset:
            # pickup alone, corrected by the fitted "Hello?" offset: usable from turn 2, at lower confidence
            return self.base[0]["voiced_s"] >= 0.5
        return len(self.base) >= MIN_BASE_TURNS and sum(h["voiced_s"] for h in self.base) >= MIN_BASE_VOICED_S

    def _hist(self) -> list[dict]:
        hist = self.base
        if len(hist) >= 4 and hist[0].get("pickup"):
            hist = hist[1:]
        return hist

    def baseline(self) -> dict[str, tuple[float, float]]:
        """feature -> (centre, spread) from this caller's turns so far."""
        out = {}
        hist = self._hist()
        if len(hist) >= 3:  # drop turns that are clearly another voice (voicemail, TV) from the centre
            med = float(np.median([h["pitch"] for h in hist]))
            hist = [h for h in hist if abs(h["pitch"] - med) <= MAX_SHIFT_ST] or hist
        for k in FEATURES:
            vals = np.array([h[k] for h in hist if h.get(k) is not None], dtype=float)
            if len(vals) == 0:
                continue
            centre = float(np.median(vals))
            s = self.spread[k]
            if len(vals) >= 4:  # this caller is unusually variable: widen, never narrow
                s = max(s, 1.4826 * float(np.median(np.abs(vals - centre))))
            out[k] = (centre, s)
        return out

    def read(self, feats: dict | None) -> VoiceReading:
        noise = self.noise_db
        if feats is None:
            return VoiceReading(calibrated=self.calibrated, baseline_turns=len(self.base))
        snr = round(feats["loudness"] - noise, 1) if noise is not None else None  # voiced level over the floor
        if not self.calibrated:
            return VoiceReading(calibrated=False, baseline_turns=len(self.base), snr_db=snr,
                                evidence=["learning this caller's normal voice"])
        base = self.baseline()
        z = {}
        for k in FEATURES:
            if feats.get(k) is not None and k in base:
                c, s = base[k]
                z[k] = round((feats[k] - c) / s, 2)
        if abs(feats["pitch"] - base["pitch"][0]) > MAX_SHIFT_ST:
            return VoiceReading(calibrated=True, baseline_turns=len(self.base), z=z, snr_db=snr,
                                evidence=[f"pitch {feats['pitch'] - base['pitch'][0]:+.1f} st: "
                                          "probably not the same voice, ignored"])
        wsum = sum(WEIGHTS[k] for k in z)
        # z clipped: a 0.3 s "haan" can show +12 on pitch movement from tracking error alone
        arousal = sum(WEIGHTS[k] * math.tanh(max(-4.0, min(4.0, z[k])) / 2) for k in z) / wsum if wsum else 0.0
        arousal = max(-1.0, min(1.0, arousal / 0.55))  # all cues at +2 z -> ~ +1
        conf = {1: 0.5, 2: 0.75}.get(len(self._hist()), 1.0) * min(1.0, feats["voiced_s"] / 0.6)
        ev = []
        names = {"pitch": "pitch", "pitch_var": "pitch movement", "loudness": "loudness",
                 "effort": "vocal effort", "rate": "speed"}
        for k, v in z.items():
            if abs(v) >= 1.5:
                ev.append(f"{names[k]} {'up' if v > 0 else 'down'} ({v:+.1f}z vs their baseline)")
        return VoiceReading(calibrated=True, baseline_turns=len(self.base), arousal=round(arousal, 3),
                            confidence=round(conf, 2), z=z, snr_db=snr, evidence=ev)

    def accept(self, feats: dict | None) -> None:
        """A turn confirmed as the caller's voice joins the baseline. The median keeps short emotional
        bursts from moving it; a long change of mood slowly becomes the new normal."""
        if feats is not None and feats["voiced_s"] >= 0.3:
            if not self.base:
                # the pickup, shifted to where this person's normal voice usually sits
                feats = {**feats, "pickup": True, **{k: feats[k] - off for k, off in self.pickup_offset.items()
                                                      if feats.get(k) is not None}}
            self.base = (self.base + [feats])[-MAX_BASELINE:]

    def same_voice(self, feats: dict | None, voice_sim: float | None) -> bool:
        """Borderline voice-print score: does pitch + loudness say it is still the caller? A background
        voice (live call 8) sat ~5 dB quieter; a different person usually sits 3+ semitones away."""
        if feats is None or voice_sim is None or voice_sim < 0.25 or len(self.base) < 2:
            return False
        base = self.baseline()
        return (abs(feats["pitch"] - base["pitch"][0]) <= 3.0
                and feats["loudness"] >= base["loudness"][0] - 4.0)


# ---------------------------------------------------------------- emotion label

def label(v: VoiceReading | None, frustration: float = 0.0, positive: bool = False,
          confusion: float = 0.0) -> dict:
    """Voice arousal x word valence -> one honest label.

    Voice gives HOW activated; words/state give WHICH way:
      raised + negative words -> angry      raised + clearly positive (wants to buy) -> excited
      raised + neutral words  -> animated   (energetic; fast + higher pitch, not louder -> anxious)
      flat/quiet/slow         -> low        (tired, sad, bored or disengaged; never with negative words)
    Text alone still yields frustrated/confused when the voice is calm or unknown. Questions are NOT
    positive: a replayed "Why is it so cheap? Is it fake?" was wrongly "excited" when they counted.
    """
    a = v.arousal if v and v.calibrated else 0.0
    conf = v.confidence if v and v.calibrated else 0.0
    z = v.z if v else {}
    negative = frustration >= 0.4
    positive = positive and frustration < 0.2
    if negative:  # words decide the direction: a quiet, flat "is it fake?" is frustrated, not sad
        name = "angry" if conf >= 0.4 and a >= 0.35 else "frustrated"
        c = conf * min(1.0, a / 0.6) if name == "angry" else min(1.0, frustration)
    elif conf >= 0.4 and a >= 0.35:
        if positive:
            name = "excited"
        elif z.get("rate", 0) >= 1.0 and z.get("loudness", 0) < 1.0:
            name = "anxious"
        else:
            name = "animated"
        c = conf * min(1.0, a / 0.6)
    elif conf >= 0.4 and a <= -0.35 and frustration < 0.25:  # some irritation + flat voice is not "sad"
        name, c = "low", conf * min(1.0, -a / 0.6)
    elif confusion >= 0.5:
        name, c = "confused", confusion
    elif not (v and v.calibrated):
        name, c = "calibrating", 0.0
    else:
        name, c = "calm", conf * (1 - min(1.0, abs(a) / 0.35))
    return {"label": name, "confidence": round(c, 2), "arousal": round(a, 2)}


_STATS: tuple[dict[str, float], dict[str, float]] | None = None


def new_calibrator() -> VoiceCalibrator:
    global _STATS
    if _STATS is None or os.getenv("VOICE_CALIB_RELOAD") == "1":
        _STATS = _load_stats()
    return VoiceCalibrator(dict(_STATS[0]), dict(_STATS[1]))
