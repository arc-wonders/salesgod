"""Whisper STT (faster-whisper, CTranslate2 on the A100). Restricted to English / Hindi."""

from __future__ import annotations

import asyncio
import logging
import os
import time

import numpy as np
from faster_whisper import WhisperModel

log = logging.getLogger("voice.stt")

ALLOWED = ("en", "hi")


def normalise(audio: np.ndarray, target_db: float = -20.0, max_gain: float = 10.0) -> np.ndarray:
    """Speech level to about -20 dBFS (at most +20 dB) before Whisper. Phone callers arrive at -15 to -30 dBFS;
    on the phone-line benchmark (voice/bench) this cut Hindi CER 8.4% -> 7.3%. Only Whisper's copy is scaled:
    the noise gates below still measure the real line level, so quiet noise is not boosted into words."""
    if not len(audio):
        return audio
    r = float(np.sqrt(np.mean(np.square(audio)))) + 1e-9
    g = min(10 ** ((target_db - 20 * np.log10(r)) / 20), max_gain)
    return np.clip(audio * g, -1.0, 1.0).astype(np.float32)


class WhisperSTT:
    def __init__(self, model: str | None = None, device: str = "cuda", compute_type: str = "float16"):
        self.model_name = model or os.getenv("STT_MODEL", "large-v3")
        self.model = WhisperModel(self.model_name, device=device, compute_type=compute_type)
        self.lock = asyncio.Semaphore(int(os.getenv("STT_CONCURRENCY", "4")))
        # Hindi turns go to a Hindi phone-speech model: IISc/ARTPARK whisper-large-v3-vaani-hindi (Apache-2.0, trained on
        # Project Vaani, crowd-sourced Hindi recorded on phones across India), CTranslate2 build. Phone-line benchmark
        # (voice/bench, 8 kHz mu-law, known text): Hindi CER 8.4% (large-v3) -> 1.7%; on real call turns it turned
        # "आब दोस्तों" into "आप बताओ" and "आपको मेरी वास्तव मदारी है" into "आपको मेरी बात समझ आ रही है". It writes
        # English speech in Devanagari, so English turns and language ID stay on large-v3. Never with an initial
        # prompt (it loops: "जजजज…"). STT_HI_MODEL="" turns it off; any load error falls back to large-v3 alone.
        self.hi_model = None
        self.hi_model_name = os.getenv("STT_HI_MODEL", "bjollans/whisper-large-v3-vaani-hindi-ct2")
        if self.hi_model_name:
            try:
                self.hi_model = WhisperModel(self.hi_model_name, device=device, compute_type=compute_type)
            except Exception as e:  # no network / no GPU memory: keep working on large-v3
                log.warning("Hindi STT model %s not loaded (%s); using %s for Hindi", self.hi_model_name, e,
                            self.model_name)

    def _transcribe(self, audio: np.ndarray, hint: str | None, prefer: str | None,
                    prompt: str | None) -> tuple[str, str, str, float]:
        lang = hint if hint in ALLOWED else None
        if lang is None:
            # Hindi speech is sometimes detected as Urdu/Punjabi/etc: pick the likelier of en/hi,
            # and stay in the call's current language unless the other one is clearly more likely.
            _, _, all_probs = self.model.detect_language(audio)
            p = dict(all_probs or [])
            en, hi = p.get("en", 0.0), p.get("hi", 0.0) + p.get("ur", 0.0)
            share_en = en / ((en + hi) or 1.0)
            # Forcing "en" on Hindi speech makes Whisper TRANSLATE it ("मैं समझा नहीं हिंदी में बताना" came
            # out as "I don't understand. Tell me in Hindi."), while "hi" on English speech still
            # transcribes it (Latin or Devanagari, which ACIE reads as English). So "en" needs to be clear.
            need = 0.5 if prefer == "en" else 0.92
            lang = "en" if share_en >= need else "hi"
        else:
            share_en = None
        mirror_en = lang == "hi" and hint is None and prefer == "en" and share_en is not None and share_en >= 0.15
        mirror_hi = lang == "en" and hint is None and prefer != "en"
        if lang == "hi" and self.hi_model is not None and not mirror_en:
            return self._filter(audio, self._run_hi(audio), "hi")
        segs = self._run(audio, lang, prompt)
        if mirror_en:
            # the mirror case: an ENGLISH call where the detector leaned Hindi on an unclear turn (live 2c4ec7a4ff:
            # the first English reply came out as "पूर्ण लोग"). Unless it is clearly Hindi, transcribe both and keep
            # English when it is at least as likely (it is the call's language).
            en_segs = self._run(audio, "en", prompt)
            score = lambda ss: float(np.mean([x.avg_logprob for x in ss])) if ss else -9.0
            if score(en_segs) >= score(segs):
                segs, lang = en_segs, "en"
        if lang == "en" and hint is None and prefer != "en":
            # a Hindi call where the detector said English: call a7152cdb0a turned "क्यूं इतना सस्ता किया,
            # असली है?" into "Thank you. It was just a given a clear." Transcribe both, keep the likelier
            # (Hindi gets a small edge, since it is the call's language).
            hi_segs = self._run(audio, "hi", prompt)
            score = lambda ss: float(np.mean([x.avg_logprob for x in ss])) if ss else -9.0
            if score(hi_segs) + 0.15 >= score(segs):
                segs, lang = hi_segs, "hi"
        if lang == "hi" and self.hi_model is not None:
            segs = self._run_hi(audio)  # the language comparison above used large-v3; the words come from the Hindi model
        return self._filter(audio, segs, lang)

    def _run_hi(self, audio: np.ndarray) -> list:
        segments, _ = self.hi_model.transcribe(
            # beam 2: phone-line set CER 2.6% (beam 5: 1.7%, beam 1: 2.6%) but no slow tail (p90 424 ms vs 1.6 s with
            # beam 5; Devanagari output is token-heavy). STT_HI_BEAM to change.
            normalise(audio), language="hi", beam_size=int(os.getenv("STT_HI_BEAM", "2")), vad_filter=False,
            condition_on_previous_text=False,
            without_timestamps=True, temperature=0.0)
        return list(segments)

    def _filter(self, audio: np.ndarray, segs: list, lang: str) -> tuple[str, str, str, float]:
        # Whisper hallucinates on noise ("प्रेम प्रेम प्रेम ..." on the first live call): drop segments
        # it marks as probably-not-speech or that are highly repetitive / very low confidence.
        # Live call 07b5082791: background noise at rms 0.013-0.022 came out as "हुआ है" / "कर दो" with
        # no_speech_prob 0.6-0.85, and the agent "sold" on it. Real speech on these calls is rms 0.04-0.15, so
        # quiet + probably-not-speech is dropped (a quiet real "हाँ" still has a low no_speech_prob).
        # The per-call calibrator decides later whether a quiet drop was really the caller.
        rms = float(np.sqrt(np.mean(np.square(audio)))) if len(audio) else 0.0
        keep, quiet = [], []
        for s in segs:
            garbage = s.compression_ratio >= 2.4 or s.avg_logprob <= -1.2 or (
                s.no_speech_prob > 0.6 and s.avg_logprob < -0.7)
            too_quiet = ((s.no_speech_prob > 0.4 and rms < 0.025)
                         or (s.no_speech_prob > 0.5 and s.avg_logprob < -0.3 and rms < 0.035)
                         or rms < 0.008)
            if not garbage and not too_quiet:
                keep.append(s)
            elif not garbage and s.no_speech_prob <= 0.75 and s.avg_logprob > -0.9:
                # dropped only for being quiet: voice/call.py gets it back if the voice is the
                # caller's own (a soft "हाँ" from a quiet phone is still the customer talking)
                quiet.append(s)
        text = " ".join(s.text.strip() for s in keep).strip()
        self_nsp = max((s.no_speech_prob for s in keep), default=1.0)
        return text, lang, " ".join(s.text.strip() for s in quiet).strip(), self_nsp

    def _run(self, audio: np.ndarray, lang: str, prompt: str | dict | None) -> list:
        if isinstance(prompt, dict):  # per-language vocabulary hint (voice/call.py builds it from the campaign)
            prompt = prompt.get(lang)
        segments, _ = self.model.transcribe(
            normalise(audio), language=lang, beam_size=5,  # beam 1 turned a clear "हाँ" into "हुआ है" on a live call
            vad_filter=False, condition_on_previous_text=False,
            without_timestamps=True, temperature=0.0, initial_prompt=prompt or None,
        )
        return list(segments)

    async def transcribe(self, audio: np.ndarray, hint: str | None = None, prefer: str | None = None,
                         prompt: str | None = None) -> tuple[str, str, float]:
        text, lang, ms, _, _ = await self.transcribe_detail(audio, hint, prefer, prompt)
        return text, lang, ms

    async def transcribe_detail(self, audio: np.ndarray, hint: str | None = None, prefer: str | None = None,
                                prompt: str | None = None) -> tuple[str, str, float, str, float]:
        """Like transcribe, plus the words dropped ONLY for being quiet (not hallucination-like) and Whisper's
        highest no-speech probability over the kept words (real speech ~0.1-0.3, its noise phrases 0.5+)."""
        t0 = time.perf_counter()
        async with self.lock:
            text, lang, quiet, nsp = await asyncio.to_thread(self._transcribe, audio, hint, prefer, prompt)
        return text, lang, (time.perf_counter() - t0) * 1000, quiet, nsp
