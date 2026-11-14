"""voice/call.py turn handling against the live-call failures of 14 Nov: noise heard as words, the agent
speaking over itself, barge-in on noise, and hanging up at the wrong time."""

import asyncio
import json
import time
import types

import numpy as np

from acie.decision.domains.bank_demo import BankDemoDomain
from acie.decision.engine import DecisionEngine
from acie.expression import TemplateBackend
from acie.pipeline import ACIEPipeline
from voice import call as callmod
from voice.call import CallRecord, CallSession


class FakeSTT:
    def __init__(self):
        self.next = ("", "")

    async def transcribe_detail(self, audio, hint=None, prefer=None, prompt=None):
        text, quiet = self.next
        return text, "en", 5.0, quiet


class FakeTTS:
    def __init__(self, delay=0.0):
        self.delay = delay

    async def stream(self, text, language="en"):
        for i in range(3):
            await asyncio.sleep(self.delay)
            level = 2000 if text.startswith("A") else 12000  # which sentence a chunk came from survives mu-law
            yield np.full(400, level, dtype="<i2").tobytes()


class FakeWS:
    def __init__(self):
        self.sent = []

    async def send_text(self, t):
        self.sent.append(json.loads(t))

    async def close(self):
        pass


class FakePlivo:
    def __init__(self):
        self.hung = None

    async def hangup(self, call):
        self.hung = call.end_reason


def session(tts_delay=0.0):
    events = []
    app = types.SimpleNamespace(
        cfg=types.SimpleNamespace(vad_end_silence_ms=700, vad_threshold=0.5, barge_in_ms=800, barge_in_rms=0.04),
        stt=FakeSTT(), tts=FakeTTS(tts_delay), plivo=FakePlivo(), products=lambda: {},
        broadcast=events.append, save_audio=lambda *a: None)
    app.make_pipeline = lambda call: (ACIEPipeline(DecisionEngine(BankDemoDomain()), backend=TemplateBackend()), "")
    cs = CallSession(app, CallRecord(to="+910", backend="template"), FakeWS(), "st")
    return cs, events


def voice(cs, sim, confirmed=True):
    """Pretend the voice print scores this utterance at `sim` vs a confirmed (or not yet known) caller."""
    cs.speaker.score = lambda audio: (np.ones(256) / 16, sim is None or sim >= 0.4, sim)
    type(cs.speaker).confirmed = property(lambda self: confirmed)


def speech(seconds=1.5, amp=0.06):
    t = np.arange(int(16000 * seconds)) / 16000
    return (amp * np.sin(2 * np.pi * 150 * t) * 1.4).astype(np.float32)


def kinds(cs):
    return [(e["type"], e.get("why")) for e in cs.call.events]


async def test_noise_phrase_while_agent_speaks_is_not_a_yes():
    """Live call 9d69a31bae: noise heard as "कर दो" / "करते हैं" sent a payment link."""
    cs, _ = session()
    voice(cs, None, confirmed=False)
    cs.app.stt.next = ("कर दो", "")
    await cs.handle_utterance(speech(amp=0.04), time.monotonic(), False, overlap=True)
    assert not [t for t in cs.call.turns if t["role"] == "customer"]
    assert ("noise", "noise phrase, not the caller's voice") in kinds(cs)


async def test_same_words_in_the_callers_own_voice_are_a_real_turn():
    cs, _ = session()
    voice(cs, 0.82)
    cs.app.stt.next = ("कर दो", "")
    await cs.handle_utterance(speech(), time.monotonic(), False, overlap=True)
    assert [t["text"] for t in cs.call.turns if t["role"] == "customer"] == ["कर दो"]


async def test_noise_never_teaches_the_voice_print():
    cs, _ = session()
    voice(cs, None, confirmed=False)
    learned = []
    cs.speaker.learn = lambda e, n: learned.append(n)
    cs.app.stt.next = ("I want to know the balance please", "")
    await cs.handle_utterance(speech(amp=0.01), time.monotonic(), False, overlap=True)  # quiet, over the agent
    assert learned == []
    await cs.handle_utterance(speech(amp=0.06), time.monotonic(), False, overlap=False)  # clean turn
    assert learned == [24000]


async def test_voicemail_words_only_hang_up_at_the_start():
    cs, _ = session()
    voice(cs, 0.8)
    cs.app.stt.next = ("what is my balance", "")
    await cs.handle_utterance(speech(), time.monotonic(), False)
    cs.app.stt.next = ("the trainer is not available in the morning?", "")
    await cs.handle_utterance(speech(), time.monotonic(), False)
    assert cs.app.plivo.hung is None
    cs2, _ = session()
    voice(cs2, None, confirmed=False)
    cs2.app.stt.next = ("the number you are calling is not available", "")
    await cs2.handle_utterance(speech(), time.monotonic(), False)
    assert cs2.app.plivo.hung == "voicemail"


async def test_after_goodbye_noise_does_not_reopen_and_hangup_is_capped():
    """Live call 9d69a31bae: two more goodbyes to "हुआ है" heard on noise, line left open."""
    cs, _ = session()
    voice(cs, 0.2)  # not the caller
    cs.hangup_after_speaking = True
    cs.app.stt.next = ("okay done thank you sir", "")
    await cs.handle_utterance(speech(), time.monotonic(), False)
    assert not [t for t in cs.call.turns if t["role"] == "customer"]
    now = time.monotonic()
    cs.play_until, cs.last_activity = now - 3, now - 0.5  # noise kept the VAD busy
    cs.detector.in_speech = True
    assert not cs.goodbye_due(now)
    assert cs.goodbye_due(now + callmod.GOODBYE_MAX_S)  # capped: noise cannot hold the line open
    cs.in_turn = True  # but a real new caller turn is always answered first
    assert not cs.goodbye_due(now + callmod.GOODBYE_MAX_S)


async def test_two_speakers_never_interleave():
    """Greeting + reply streamed at once and Plivo played them chopped together ("multiple voices")."""
    cs, _ = session(tts_delay=0.01)
    await asyncio.gather(cs.speak(["AAAA greeting"], "en"), cs.speak(["BBBB reply"], "en"))
    import base64
    from voice.audio import ulaw_to_float  # noqa: F401
    tags = [m["media"]["payload"] for m in cs.ws.sent if m["event"] == "playAudio"]
    order = [base64.b64decode(p) for p in tags]
    first = order[:len(order) // 2]
    second = order[len(order) // 2:]
    assert len(set(first)) == 1 and len(set(second)) == 1 and first[0] != second[0]


async def test_barge_in_needs_the_callers_voice_once_known():
    cs, _ = session()
    cs.play_until = time.monotonic() + 5  # agent is talking
    d = cs.detector
    d.in_speech, d.run_frames, d.run_start, d.run_energy = True, 20, 7, 0.2  # 640 ms of loud "speech"
    d.frames = [np.zeros(512, np.float32)] * 30
    voice(cs, 0.15)  # a background voice / noise
    await cs.check_barge_in()
    assert not cs.barge_in_pending and ("barge_in_ignored", None) in kinds(cs)
    d.run_start = 8  # a new run, this time the caller
    voice(cs, 0.7)
    await cs.check_barge_in()
    assert cs.barge_in_pending  # stopped after 0.64 s, not 0.8 s
