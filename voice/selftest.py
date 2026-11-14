"""End-to-end self test on the server, without a phone call.

For each test line: Fish TTS (8 kHz PCM) -> mu-law (as the phone line carries it) ->
VAD turn detection -> Whisper -> ACIE (LLM backend) -> validator. Prints transcripts,
intents, the reply and per-stage latency.

  /opt/salesgod/venv/bin/python -m voice.selftest
"""

from __future__ import annotations

import asyncio
import time

import numpy as np

from acie.decision.domains.bank_demo import BankDemoDomain
from acie.decision.engine import DecisionEngine
from acie.expression import OpenAICompatibleBackend
from acie.pipeline import ACIEPipeline
from acie.schemas import TurnInput

from .audio import pcm16_to_ulaw, ulaw_to_float, upsample_8k_to_16k
from .stt import WhisperSTT
from .tts import FishTTS
from .vad import TurnDetector

CONVERSATION = [
    "नमस्ते, मुझे पचास हज़ार रुपये ट्रांसफर करने हैं",
    "मतलब? लिमिट क्या होती है, समझ नहीं आया",
    "अच्छा ठीक है, दस हज़ार भेज दो",
    "हाँ",
    "what is my balance now",
    "okay thank you, bye",
]


async def main() -> None:
    tts, stt = FishTTS(), await asyncio.to_thread(WhisperSTT)
    pipe = ACIEPipeline(DecisionEngine(BankDemoDomain()), backend=OpenAICompatibleBackend(), buffer_reply=True)
    for i, line in enumerate(CONVERSATION):
        pcm = b"".join([c async for c in tts.stream(line, "hi" if any("ऀ" <= ch <= "ॿ" for ch in line) else "en")])
        # phone line: mu-law 8 kHz, plus 1 s of silence so the VAD sees the end of the turn
        line_audio = ulaw_to_float(pcm16_to_ulaw(pcm + b"\x00\x00" * 8000))
        det = TurnDetector()
        utt = None
        for k in range(0, len(line_audio), 160):  # 20 ms frames, as Plivo sends them
            for kind, data in det.feed(upsample_8k_to_16k(line_audio[k:k + 160])):
                if kind == "utterance":
                    utt = data
        if utt is None:
            print(f"[{i}] VAD found no speech in {line!r}")
            continue
        sess = pipe.sessions.get("selftest")
        hint = None
        if sess and len(utt) / 16000 < 1.6:  # same rule as voice/call.py
            hint = "hi" if sess.expression.language in ("hi", "hinglish") else "en"
        text, lang, stt_ms = await stt.transcribe(utt, hint)
        t0 = time.perf_counter()
        first = {}

        def on_sentence(s: str) -> None:
            first.setdefault("ms", (time.perf_counter() - t0) * 1000)

        r = await pipe.run_turn(TurnInput(session_id="selftest", customer_id="c1", text=text), on_sentence)
        print(f"\n[{i}] said : {line}\n    heard: {text}  ({lang}, STT {stt_ms:.0f} ms, VAD {len(utt)/16000:.1f}s)\n"
              f"    intent={r.plan.intent} lang={r.expression.language} cplx={r.expression.complexity} "
              f"frus={r.state.frustration} conf={r.state.confusion}\n"
              f"    reply: {r.text}\n    LLM first sentence {first.get('ms', 0):.0f} ms, total {r.timings_ms.get('total')} ms, "
              f"valid={r.validation.ok} fallback={r.validation.used_fallback} {r.validation.issues}")
    await tts.aclose()


if __name__ == "__main__":
    asyncio.run(main())
