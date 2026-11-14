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
        self.nsp = 0.6

    async def transcribe_detail(self, audio, hint=None, prefer=None, prompt=None):
        text, quiet = self.next
        return text, "en", 5.0, quiet, self.nsp


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
    voice(cs, None, confirmed=False)
    cs.hangup_after_speaking = True
    cs.app.stt.next = ("हुआ है", "")
    await cs.handle_utterance(speech(), time.monotonic(), False, overlap=False)
    assert not [t for t in cs.call.turns if t["role"] == "customer"]
    # but a real sentence after "I think we've lost the line" is answered (replay of 339d4ace25)
    cs.app.stt.next = ("That is fine, but what do you have in the gym?", "")
    await cs.handle_utterance(speech(), time.monotonic(), False, overlap=True)
    assert [t["text"] for t in cs.call.turns if t["role"] == "customer"] == [
        "That is fine, but what do you have in the gym?"]
    cs.hangup_after_speaking, cs.in_turn = True, False  # turn_worker clears in_turn after the reply
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


def _speaking(cs, seconds=5.0):
    """The agent has `seconds` of audio queued at Plivo, in 200 ms parts."""
    now = time.monotonic()
    cs.play_until = now + seconds
    cs.sent_parts = [(now - 0.4 + 0.2 * i, bytes([i % 256]) * 1600) for i in range(int(seconds / 0.2) + 2)]


def _caller_speaking(cs, run_ms):
    d = cs.detector
    d.in_speech, d.run_frames, d.run_start, d.run_energy = True, run_ms // 32, 5, 0.2
    d.frames = [np.zeros(512, np.float32)] * 40


async def test_agent_pauses_within_about_250ms_of_caller_speech():
    """Owner: "if the person is saying something, it does not cut it off" (old rule: 800 ms + a voice check)."""
    cs, _ = session()
    _speaking(cs)
    _caller_speaking(cs, 160)
    await cs.check_barge_in()
    assert cs.paused is None  # 160 ms: not yet
    _caller_speaking(cs, 224)
    await cs.check_barge_in()
    assert cs.paused is not None and not cs.agent_speaking()
    assert any(m["event"] == "clearAudio" for m in cs.ws.sent)
    ev = [e for e in cs.call.events if e["type"] == "barge_pause"][0]
    assert ev["after_speech_ms"] <= 256


async def test_short_sound_resumes_the_paused_sentence():
    cs, _ = session()
    _speaking(cs)
    _caller_speaking(cs, 224)
    await cs.check_barge_in()
    held = len(cs.paused["parts"])
    assert held > 0
    cs.detector.in_speech = False  # it was a "haan" / a cough: over at once
    await cs.check_barge_in()
    assert cs.paused is None and not cs.barge_in_pending
    assert sum(m["event"] == "playAudio" for m in cs.ws.sent) == held  # the unplayed rest went out again
    assert cs.agent_speaking()
    assert [e for e in cs.call.events if e["type"] == "barge_resume"]


async def test_sustained_caller_speech_is_a_real_interruption():
    cs, _ = session()
    voice(cs, 0.7)
    _speaking(cs)
    _caller_speaking(cs, 224)
    await cs.check_barge_in()
    epoch = cs.speaking_epoch
    cs.paused["at"] -= 0.4  # still talking 400 ms after the pause
    await cs.check_barge_in()
    assert cs.paused is None and cs.barge_in_pending and cs.speaking_epoch == epoch + 1
    assert [e for e in cs.call.events if e["type"] == "barge_in"]


async def test_another_voice_resumes_the_agent():
    cs, _ = session()
    voice(cs, 0.12)  # the caller is known; this is someone else
    _speaking(cs)
    _caller_speaking(cs, 224)
    await cs.check_barge_in()
    cs.paused["at"] -= 0.4
    await cs.check_barge_in()
    assert cs.paused is None and not cs.barge_in_pending
    assert [e for e in cs.call.events if e["type"] == "barge_resume"]


async def test_tts_arriving_while_paused_is_held_not_spoken():
    cs, _ = session()
    _speaking(cs)
    _caller_speaking(cs, 224)
    await cs.check_barge_in()
    before = sum(m["event"] == "playAudio" for m in cs.ws.sent)
    await cs.play_pcm(np.zeros(1600, dtype="<i2").tobytes(), cs.speaking_epoch)
    assert sum(m["event"] == "playAudio" for m in cs.ws.sent) == before  # held while we decide


async def test_noise_phrase_unlike_the_first_clean_turn_is_dropped_before_confirmation():
    cs, _ = session()
    voice(cs, None, confirmed=False)
    cs.speaker.embs = [np.ones(256) / 16]
    cs.speaker.provisional_sim = lambda e: 0.12
    cs.app.stt.next = ("करते हैं", "")
    await cs.handle_utterance(speech(amp=0.05), time.monotonic(), False, overlap=False)  # loud, agent silent
    assert not [t for t in cs.call.turns if t["role"] == "customer"]


async def test_short_okay_from_the_confirmed_caller_is_kept_when_the_agent_is_silent():
    cs, _ = session()
    voice(cs, 0.42)
    cs.app.stt.next = ("Okay.", "")
    await cs.handle_utterance(speech(), time.monotonic(), False, overlap=False)
    assert [t["text"] for t in cs.call.turns if t["role"] == "customer"] == ["Okay."]


async def test_quiet_real_yes_is_kept_when_whisper_heard_speech():
    """Live 6bb1ca00ad: the caller's first "Yes." (rms 0.02, agent silent, no-speech 0.17)."""
    cs, _ = session()
    voice(cs, None, confirmed=False)
    cs.app.stt.next, cs.app.stt.nsp = ("Yes.", ""), 0.17
    await cs.handle_utterance(speech(amp=0.02), time.monotonic(), False, overlap=False)
    assert [t["text"] for t in cs.call.turns if t["role"] == "customer"] == ["Yes."]
    cs2, _ = session()
    voice(cs2, None, confirmed=False)
    cs2.app.stt.next, cs2.app.stt.nsp = ("Okay.", ""), 0.59  # the same words from noise
    await cs2.handle_utterance(speech(amp=0.02), time.monotonic(), False, overlap=False)
    assert not [t for t in cs2.call.turns if t["role"] == "customer"]


async def test_degraded_caller_voice_is_not_dropped_as_background():
    """Live 2c4ec7a4ff: the caller's own turns at 0.29-0.40 were dropped as 'different voice'."""
    from voice.speaker import SpeakerLock
    assert SpeakerLock().drop_below <= 0.25
    cs, _ = session()
    cs.speaker.score = lambda audio: (np.ones(256) / 16, True, 0.33)  # 0.33 >= 0.25: the caller
    type(cs.speaker).confirmed = property(lambda self: True)
    cs.app.stt.next = ("वहाँ तो डबल तरफ करूंगे", "")
    await cs.handle_utterance(speech(), time.monotonic(), False)
    assert [t["text"] for t in cs.call.turns if t["role"] == "customer"] == ["वहाँ तो डबल तरफ करूंगे"]


def test_two_agreeing_rejected_turns_mean_the_callers_voice_changed():
    from voice.speaker import SpeakerLock
    lock = SpeakerLock()
    a = np.zeros(256); a[0] = 1.0
    b = np.zeros(256); b[0] = 0.9; b[1] = 0.436  # cos(a, b) ~ 0.9
    assert not lock.voice_changed(a, 24000)
    assert lock.voice_changed(b, 24000)  # sounds like the earlier rejected turn: re-learn, keep it
    assert any(float(x @ a) > 0.99 for x in lock.embs)


async def test_unusable_caller_speech_gets_repeat_request_not_hangup():
    """Live 2c4ec7a4ff: six 'can you hear me?' and a hang-up while the caller was talking."""
    cs, _ = session()
    spoken = []

    async def fake_speak(sentences, lang, *a, **k):
        spoken.extend(sentences)
    cs.speak = fake_speak
    cs.last_dropped_speech = time.monotonic()  # we dropped / could not use what the caller said
    await cs.no_response()
    assert spoken and ("फिर से" in spoken[-1] or "फिर बोलेंगे" in spoken[-1] or "once more" in spoken[-1])
    assert cs.call.turns[-1]["intent"] == "repeat_request" and not cs.hangup_after_speaking
    # silence after that, but speech was heard within 20 s: still never a hang-up as "no response"
    for _ in range(3):
        await cs.no_response()
    assert not cs.hangup_after_speaking


def test_noise_phrase_variants():
    from voice.call import CallSession as C
    for t in ["कर दो कि अ", "कर दो कर दो कर दो कर दो", "करते हैं।", "हुआ है ना"]:
        assert C.stock_noise(t), t
    for t in ["कर दो, बताइए price", "जी हाँ कर दो link", "मुझे join करना है"]:
        assert not C.stock_noise(t), t


async def test_english_call_uses_the_hindi_model_only_after_the_caller_asks_for_hindi():
    cs, _ = session()
    cs.call.language = "en"
    assert not cs.hindi_model_ok()
    voice(cs, 0.8)
    cs.app.stt.next = ("Yes, I can hear you, but can you speak in Hindi?", "")
    await cs.handle_utterance(speech(), time.monotonic(), False)
    assert cs.hindi_model_ok()
    cs2, _ = session()
    cs2.call.language = "hi"
    assert cs2.hindi_model_ok()
