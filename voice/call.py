"""One live phone call: Plivo audio <-> VAD -> Whisper -> ACIE -> Fish TTS -> Plivo audio.

Turn-taking:
  - Silero VAD finds the start and end of each customer turn.
  - If the customer starts talking while the agent is speaking (barge-in), playback is
    cleared on the Plivo side and the rest of the reply is dropped. The ACIE turn itself
    is never cancelled half-way (it may already have changed dialog state, e.g. a transfer).
  - Customer turns are processed one at a time, in order.
"""

from __future__ import annotations

import asyncio
import base64
import difflib
import re
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from acie.trace import trace
from acie.text import detect_language
from acie.schemas import ExpressionParams, TurnInput, TurnTiming, Utterance
from acie.validator import StreamingValidator

from .audio import LINE_RATE, pcm16_to_ulaw, ulaw_to_float, upsample_8k_to_16k
from .calibrator import label as emotion_label
from .calibrator import measure as measure_voice
from .calibrator import new_calibrator
from .speaker import SpeakerLock
from .vad import TurnDetector

if TYPE_CHECKING:
    from .server import App

log = logging.getLogger("voice.call")

VOICEMAIL = re.compile(r"leave (a|your) message|after the (tone|beep)|is not available|switched off|not reachable|"
                       r"उपलब्ध नहीं|स्विच ऑफ|संदेश छोड़|बीप के बाद|पहुंच से बाहर|abhi uplabdh nahi", re.IGNORECASE)
DEVA = re.compile(r"[\u0900-\u097F]")
PLAY_CHUNK_BYTES = int(LINE_RATE * 0.2)  # 200 ms of mu-law per playAudio message
SILENCE_HANGUP_S = 25.0
# Plivo's machine detection called a real person ("hello? haan bolo...") a machine on the first
# live try (call 7dbfc951d4) and we hung up on them. So by default it is only logged; the
# transcript check (VOICEMAIL above) still hangs up on real voicemail greetings.
AMD_HANGUP = os.getenv("PLIVO_AMD_HANGUP", "0") == "1"
HEARD_RECENTLY_S = 20.0  # any speech this recent (used or not) means the line is alive
REPEAT_PLEASE = {"hi": "माफ़ कीजिए, आवाज़ थोड़ी साफ़ नहीं आई। एक बार फिर बोलेंगे?",
                 "en": "Sorry, I didn't catch that clearly. Could you say it once more?",
                 "hinglish": "Sorry, awaaz thodi saaf nahi aayi. Ek baar phir bolenge?"}
NO_RESPONSE_S = 8.0  # quiet this long after the agent finished -> "can you hear me?" (voice_platform watchdog)
MAX_NO_RESPONSE_PROMPTS = 2
OPENING_PROTECTED_S = 4.0  # the opening's "AI assistant / recorded" disclosure can't be barged off
NO_RESPONSE = {
    "hi": ["हेलो, क्या आपको मेरी आवाज़ आ रही है?",
           "लगता है आवाज़ नहीं आ रही है। मैं बाद में कॉल करता हूँ, धन्यवाद।"],
    "en": ["Hello, can you hear me?", "I think we've lost the line. I'll call back later, thank you."],
    "hinglish": ["Hello, kya aapko meri awaaz aa rahi hai?",
                 "Lagta hai awaaz nahi aa rahi. Main baad mein call karta hoon, thank you."],
}
GOODBYE_GRACE_S = 2.5
GOODBYE_MAX_S = 8.0  # after our goodbye, noise / half-heard words never keep the line open longer than this
# Whisper's stock outputs on line noise (diag of live calls 6bb1ca00ad / 9d69a31bae / 7920edcd26: these came
# out of pure noise while the agent spoke, caller-voice similarity 0.07-0.2, and "कर दो" + "करते हैं" from
# noise sent a payment link). They are only believed when the voice is the caller's.
# fillers Whisper wraps around its noise phrases ("कर दो कि अ", "हुआ है ना"); stripped, with repeats, before matching
NOISE_FILLERS = {"कि", "अ", "आ", "उम", "हम्म", "हम", "ना", "जी", "तो", "और", "uh", "um", "hmm", "so", "and", "oh"}
STOCK_NOISE = {"कर दो", "कर दो कर दो", "हुआ है", "हुआ है हुआ है", "करते हैं", "कि अ", "कर दो कि अ", "thank you",
               "thank you very much", "thanks for watching", "okay", "ok", "you", "bye", "we'll be on", "we're ready",
               "हां", "हाँ", "जी", "कि", "अ", "हम्म", "hmm", "yes", "yeah", "उनको लेकर", "कि उनको लेकर"}
CLEAN_RMS = 0.03
BARGE_STAGES = [(500, 0.35), (800, 0.35), (1200, 0.32)]  # (ms of the caller's run, voice similarity needed)  # a turn loud enough (with the agent silent) to teach the caller's voice print
MAX_CALL_S = 360.0


@dataclass
class CallRecord:
    to: str
    customer_id: str = "c1"
    language: str = "hi"  # opening language
    backend: str = "llm"  # llm (default reply model) | gemini | qwen | template
    campaign: str = "bank_demo"  # bank_demo | <product id>
    customer_name: str = ""
    customer_age: int | None = None  # optional: how the agent pitches and explains (a 21-year-old is not a 70-year-old)
    call_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    status: str = "dialing"  # dialing | ringing | live | ended | failed
    request_uuid: str | None = None
    call_uuid: str | None = None
    token: str | None = None
    created_at: float = field(default_factory=time.time)
    answered_at: float | None = None
    ended_at: float | None = None
    end_reason: str | None = None
    turns: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)  # barge-ins, dropped noise, errors (for review)
    machine: bool = False  # Plivo answering-machine detection said a machine answered
    sms: list[dict] = field(default_factory=list)
    context: dict = field(default_factory=dict)  # the call's context ledger (acie/context.py), saved at the end

    def public(self) -> dict:
        return {k: getattr(self, k) for k in ("call_id", "to", "customer_id", "language", "backend", "campaign",
                                              "customer_name", "customer_age", "status",
                                              "created_at", "answered_at", "ended_at", "end_reason", "turns", "events", "sms", "context")}


class _SafeMap(dict):
    def __missing__(self, key):  # an unknown {placeholder} in sms_text stays as typed, it never crashes the send
        return "{" + key + "}"


class CallSession:
    def __init__(self, app: App, call: CallRecord, ws, stream_id: str):
        self.app, self.call, self.ws, self.stream_id = app, call, ws, stream_id
        self.detector = TurnDetector(end_silence_ms=app.cfg.vad_end_silence_ms, threshold=app.cfg.vad_threshold)
        self.pipeline, self.stt_prompt = app.make_pipeline(call)
        # every ledger event goes to the console and the call record, live
        self.pipeline.context_sinks.append(lambda ev: self.event("context", ev.model_dump(exclude={"t"})))
        self.session_id = f"call-{call.call_id}"
        self.utterances: asyncio.Queue = asyncio.Queue()
        self.play_until = 0.0  # monotonic time the agent's queued audio finishes
        self.speaking_epoch = 0  # bumped on barge-in: stale TTS stops sending
        self.barge_in_pending = False
        self.speech_started_at: float | None = None
        self.last_activity = time.monotonic()
        self.closed = asyncio.Event()
        # recording: customer = every inbound mu-law frame in order; agent = what we played, placed
        # at its playback time (relative to the first inbound frame), cut where a barge-in cleared it
        self.rec_in = bytearray()
        self.rec_t0: float | None = None
        self.rec_out: list[tuple[float, bytes]] = []
        self.rec_clears: list[float] = []
        self.ts0: int | None = None
        self.hangup_after_speaking = False
        self.busy = False  # a customer turn is being processed
        self.max_loop_lag = 0.0
        self.send_lock = asyncio.Lock()
        # one voice at a time: the greeting, a reply and the "can you hear me?" prompt each streamed their own
        # TTS at once, and Plivo played the chunks interleaved ("multiple voices", owner 14 Nov)
        self.speak_lock = asyncio.Lock()
        self.in_turn = False  # a turn ACCEPTED as the caller's is being answered
        self.barge_checked_run = -1  # VAD run already voice-checked for barge-in
        self.barge_stage = 0
        self.accepted_turns = 0
        self.last_dropped_speech = 0.0  # a caller utterance with words that we dropped or could not use
        self.last_replied_at = 0.0
        self.primary: list = []  # prosody of the caller's accepted turns (speaker lock)
        self.speaker = SpeakerLock()
        # learns this line's noise floor and the caller's normal voice from pickup (voice/calibrator.py)
        self.calib = new_calibrator()
        self.detector.on_frame = lambda frame, p: self.calib.observe(frame, p, self.agent_speaking())
        self.no_response_count = 0
        self.protected_until = 0.0
        product = app.products().get(call.campaign) if call.campaign != "bank_demo" else None
        self.product = product
        self.pronounce = [(re.compile(r"(?<![A-Za-z])" + re.escape(k) + r"(?![A-Za-z])"), v)
                          for k, v in (product.pronunciations if product else {}).items() if k and v]

    # ------------------------------------------------------------------ plivo I/O

    async def send(self, msg: dict) -> None:
        async with self.send_lock:
            try:
                await self.ws.send_text(json.dumps(msg))
            except Exception:
                self.closed.set()

    async def clear_audio(self) -> None:
        if self.rec_t0 is not None:
            self.rec_clears.append(time.monotonic() - self.rec_t0)
        await self.send({"event": "clearAudio", "streamId": self.stream_id})

    async def play_pcm(self, pcm16: bytes, epoch: int) -> bool:
        ulaw = pcm16_to_ulaw(pcm16)
        for i in range(0, len(ulaw), PLAY_CHUNK_BYTES):
            if epoch != self.speaking_epoch or self.closed.is_set():
                return False
            part = ulaw[i:i + PLAY_CHUNK_BYTES]
            await self.send({"event": "playAudio", "media": {
                "contentType": "audio/x-mulaw", "sampleRate": LINE_RATE,
                "payload": base64.b64encode(part).decode()}})
            now = time.monotonic()
            start = max(self.play_until, now)
            if self.rec_t0 is not None:
                self.rec_out.append((start - self.rec_t0, part, now - self.rec_t0))
            self.play_until = start + len(part) / LINE_RATE
        return True

    def agent_speaking(self) -> bool:
        return time.monotonic() < self.play_until

    # ------------------------------------------------------------------ main loop

    async def run(self) -> None:
        self.call.status, self.call.answered_at = "live", time.time()
        self.app.broadcast({"type": "status", "call": self.call.public()})
        workers = [asyncio.create_task(self.turn_worker()), asyncio.create_task(self.watchdog())]
        self.rec_t0 = time.monotonic()  # stream start (the greeting plays before any inbound audio)
        # the greeting plays in the background: awaiting it here left the caller's audio unread for
        # its whole length (backlog -> late VAD -> false barge-ins)
        workers.append(asyncio.create_task(self.greet()))
        try:
            while not self.closed.is_set():
                msg = json.loads(await self.ws.receive_text())
                ev = msg.get("event")
                if ev == "media":
                    raw = base64.b64decode(msg["media"]["payload"])
                    ts = msg["media"].get("timestamp")
                    if ts is not None:
                        # place by Plivo's timestamp: exact even if we read late. It is an absolute
                        # clock (epoch ms), not ms since stream start: treating it as the latter made a
                        # ~10^13-byte pad -> MemoryError -> the stream died after the greeting (calls
                        # 383e71e9bf and e5e10e25c0 "hung up" with no customer audio at all)
                        now_s = time.monotonic() - self.rec_t0
                        if self.ts0 is None:
                            self.ts0 = int(ts) - int(now_s * 1000)
                        rel = (int(ts) - self.ts0) / 1000
                        pos = int(rel * LINE_RATE)
                        if 0 < pos - len(self.rec_in) < 5 * LINE_RATE:  # a gap in the media: keep time aligned
                            self.rec_in += b"\xff" * (pos - len(self.rec_in))
                        lag = now_s - rel
                        if lag > self.max_loop_lag + 0.25:
                            self.max_loop_lag = lag
                            if lag > 0.5:
                                self.event("loop_lag", {"seconds": round(lag, 2)})
                    self.rec_in += raw
                    audio = upsample_8k_to_16k(ulaw_to_float(raw))
                    for kind, data in self.detector.feed(audio):
                        await self.on_vad(kind, data)
                    await self.check_barge_in()
                elif ev == "stop":
                    self.call.end_reason = self.call.end_reason or "customer_hangup"
                    break
        except Exception as e:  # websocket closed by Plivo, etc.
            log.info("[%s] stream ended: %s", self.call.call_id, type(e).__name__)
            self.call.end_reason = self.call.end_reason or "customer_hangup"
        finally:
            self.closed.set()
            for w in workers:
                w.cancel()
            try:
                self.app.save_recording(self.call.call_id, bytes(self.rec_in), self.rec_out, self.rec_clears)
            except Exception:
                log.exception("[%s] saving recording failed", self.call.call_id)
            await self.app.finish(self.call, self.pipeline, self.session_id)

    async def on_vad(self, kind: str, data: np.ndarray | None) -> None:
        """First live call showed line noise / echo of the agent's own voice triggering barge-ins
        every few seconds. So: a barge-in needs BARGE_IN_MS of continuous speech, and a short
        burst heard while the agent was talking (and not a barge-in) is dropped as echo/noise."""
        self.last_activity = time.monotonic()
        if kind == "speech_start":
            self.speech_started_at = time.monotonic()
            self.started_during_agent = self.agent_speaking()
        elif kind == "utterance":
            started = self.speech_started_at or time.monotonic()
            during_agent = getattr(self, "started_during_agent", False)
            dur = len(data) / 16000
            # only short bursts fully inside the agent's speech are echo/noise; live call 9 dropped a
            # real 7 s complaint that started while the agent was still finishing a sentence
            # call a7152cdb0a dropped three 1.2-1.4 s bursts of the caller trying to cut in; only very short
            # ones are dropped now, and echo of our own voice is caught by its transcript (handle_utterance)
            if during_agent and not self.barge_in_pending and dur < 0.8 and self.agent_speaking():
                self.event("noise", {"duration_s": round(dur, 2), "why": "while agent speaking"})
            else:
                overlap = during_agent or self.agent_speaking()
                await self.utterances.put((data, started, self.barge_in_pending, overlap))
            self.barge_in_pending = False
            self.speech_started_at = None
            self.started_during_agent = False

    def is_background(self, p) -> bool:
        """Live call 8, the caller: "जो प्राइमरी वॉइस है उसको पकड़ो, बैकग्राउंड वाली को हटा दो".
        Lock onto the caller from their first answered turns; a turn much quieter than them (and/or
        in a clearly different pitch) is someone in the background or a TV, not the caller."""
        if len(self.primary) < 2:
            return False
        db = float(np.median([x.loudness_db for x in self.primary]))
        st = float(np.median([x.f0_median_st for x in self.primary]))
        quieter = db - p.loudness_db
        other_pitch = abs(p.f0_median_st - st) > 6
        return quieter >= 10 or (quieter >= 5 and other_pitch)

    def primary_rms(self) -> float | None:
        if len(self.primary) < 2:
            return None
        return 10 ** (float(np.median([x.loudness_db for x in self.primary])) / 20)

    async def check_barge_in(self) -> None:
        d = self.detector
        if not (self.agent_speaking() and not self.barge_in_pending and d.in_speech
                and time.monotonic() >= self.protected_until):
            return
        run_ms = d.run_frames * 32
        if self.speaker.confirmed:
            # we know the caller's voice: stop for THEIR voice, never for noise or a background voice. Checked in
            # stages because a short clip scores lower (voice/diag_bargein.py on 122 real caller turns: the
            # first 0.5 / 0.8 / 1.0 s pass >= 0.35 for 66% / 88% / 94% of callers, noise 0% / 0% / 7%)
            stage = self.barge_stage if self.barge_checked_run == d.run_start else 0
            if stage >= len(BARGE_STAGES) or run_ms < BARGE_STAGES[stage][0]:
                return
            self.barge_checked_run, self.barge_stage = d.run_start, stage + 1
            clip = np.concatenate(d.frames[-max(16, min(d.run_frames, 40)):])
            _, _, sim = await asyncio.to_thread(self.speaker.score, clip)
            if sim is None or sim < BARGE_STAGES[stage][1]:
                self.event("barge_in_ignored", {"voice_similarity": sim, "run_ms": run_ms})
                return
            if not (self.agent_speaking() and not self.barge_in_pending):
                return
        # before the caller's voice is known: continuous, confident (p>=0.7) and loud enough. Quiet echo of
        # our own voice and line noise pass the VAD but sit far below real speech on the customer channel
        elif not (run_ms >= self.app.cfg.barge_in_ms
                  and d.run_energy >= max(self.app.cfg.barge_in_rms, 0.25 * (self.primary_rms() or 0),
                                          4 * (self.calib.noise_level or 0))):  # noisy line: 12 dB over its floor
            return
        self.speaking_epoch += 1
        self.barge_in_pending = True
        self.play_until = time.monotonic()
        await self.clear_audio()
        self.event("barge_in", {})

    async def watchdog(self) -> None:
        t0 = time.monotonic()
        while not self.closed.is_set():
            await asyncio.sleep(1.0)
            now = time.monotonic()
            idle = now - max(self.last_activity, self.play_until)
            if self.call.machine and AMD_HANGUP and not any(t["role"] == "customer" for t in self.call.turns):
                await self.clear_audio()
                await self.hangup("answering_machine")
            elif now - t0 > MAX_CALL_S:
                await self.hangup("max_duration")
            elif (not self.hangup_after_speaking and not self.busy and not self.agent_speaking()
                  and not self.detector.in_speech and self.utterances.empty()
                  and now - max(self.last_activity, self.play_until) > NO_RESPONSE_S):
                await self.no_response()
            elif idle > SILENCE_HANGUP_S and self.utterances.empty():
                await self.hangup("silence")
            elif self.goodbye_due(now):
                # wait for a quiet moment after the goodbye: the caller often says "ok thanks" or asks
                # something else, and hanging up 0.6 s after the agent stops felt like being cut off
                await self.hangup(getattr(self, "hangup_reason", "agent_ended"))

    def goodbye_due(self, now: float) -> bool:
        """After our goodbye: hang up at the first quiet moment (the caller often says "ok thanks"), and
        in any case GOODBYE_MAX_S after it finished, unless the caller's own new turn is being answered."""
        if not self.hangup_after_speaking or self.agent_speaking() or self.in_turn:
            return False
        since = now - self.play_until
        quiet = (self.utterances.empty() and not self.detector.in_speech and not self.busy
                 and now - self.last_activity > GOODBYE_GRACE_S)
        return since > GOODBYE_GRACE_S and (quiet or since > GOODBYE_MAX_S)

    async def no_response(self) -> None:
        sess = self.pipeline.sessions.get(self.session_id)
        lang = sess.expression.language if sess else self.call.language
        heard = time.monotonic() - max(getattr(self, "last_dropped_speech", 0.0), self.last_activity) < HEARD_RECENTLY_S
        if getattr(self, "last_dropped_speech", 0.0) > self.last_replied_at:
            # the caller DID speak but we could not use it (unclear / dropped): ask them to repeat, never
            # "can you hear me?" and never a hang-up (live 2c4ec7a4ff: six prompts and a hang-up while he talked)
            self.last_replied_at = time.monotonic()
            text = REPEAT_PLEASE.get(lang, REPEAT_PLEASE["hi"])
            self.event("no_response", {"count": self.no_response_count, "why": "speech heard but unusable"})
            turn = {"role": "agent", "text": text, "intent": "repeat_request", "t": time.time()}
            self.call.turns.append(turn)
            self.event("turn", turn)
            if sess:
                sess.history.append(Utterance(role="agent", text=text))
            self.last_activity = time.monotonic()
            await self.speak([text], lang)
            return
        self.no_response_count += 1
        if heard and self.no_response_count > MAX_NO_RESPONSE_PROMPTS - 1:
            self.no_response_count = MAX_NO_RESPONSE_PROMPTS - 1  # someone spoke recently: never hang up as silent
        last = self.no_response_count > MAX_NO_RESPONSE_PROMPTS - 1
        text = NO_RESPONSE.get(lang, NO_RESPONSE["hi"])[1 if last else 0]
        self.event("no_response", {"count": self.no_response_count})
        turn = {"role": "agent", "text": text, "intent": "no_response_hangup" if last else "no_response_check",
                "t": time.time()}
        self.call.turns.append(turn)
        self.event("turn", turn)
        if sess:
            sess.history.append(Utterance(role="agent", text=text))
        self.last_activity = time.monotonic()
        await self.speak([text], lang)
        if last:
            self.hangup_after_speaking, self.hangup_reason = True, "no_response"

    async def hangup(self, reason: str) -> None:
        if self.closed.is_set():
            return
        self.call.end_reason = reason
        await self.app.plivo.hangup(self.call)
        self.closed.set()
        try:
            await self.ws.close()
        except Exception:
            pass

    # ------------------------------------------------------------------ turns

    async def greet(self) -> None:
        sess = self.pipeline.session(TurnInput(session_id=self.session_id, customer_id=self.call.customer_id, text=""))
        sess.expression = ExpressionParams(language=self.call.language)  # type: ignore[arg-type]
        plan = await self.pipeline.decision.decide(
            "greet", TurnInput(session_id=self.session_id, customer_id=self.call.customer_id, text=""), sess)
        v = StreamingValidator(plan, sess.expression)
        text = " ".join(v.finish())  # safe template, slots filled
        sess.history.append(Utterance(role="agent", text=text))
        sess.context.record_turn(0, "", plan.intent, False, sinks=self.pipeline.context_sinks)
        turn = {"role": "agent", "text": text, "intent": "greet", "t": time.time()}
        self.call.turns.append(turn)
        self.event("turn", turn)
        self.protected_until = time.monotonic() + OPENING_PROTECTED_S
        await self.speak([text], sess.expression.language)

    async def turn_worker(self) -> None:
        while not self.closed.is_set():
            audio, started, barge, overlap = await self.utterances.get()
            self.busy = True
            try:
                await self.handle_utterance(audio, started, barge, overlap)
            except Exception as e:
                log.exception("[%s] turn failed", self.call.call_id)
                self.event("error", {"message": f"{type(e).__name__}: {e}"})
            finally:
                self.busy = False
                self.in_turn = False

    async def handle_utterance(self, audio: np.ndarray, started: float, barge: bool, overlap: bool = False) -> None:
        t_end = time.monotonic()
        sess = self.pipeline.sessions.get(self.session_id)
        dur_s = len(audio) / 16000
        # Whisper's language ID is unreliable on short clips ("हाँ" came out as "Huh?"), so short
        # turns are transcribed in the call's current language; longer ones auto-detect en/hi.
        hint = prefer = None
        if sess:
            prefer = "hi" if sess.expression.language in ("hi", "hinglish") else "en"
            hint = prefer if dur_s < 1.6 else None
        n = sum(1 for t in self.call.turns if t["role"] == "customer") + 1
        self.app.save_audio(self.call.call_id, n, audio)  # for listening back / STT debugging
        # Whisper (GPU), the voice print and the voice features (CPU) run at the same time: no added latency
        (text, stt_lang, stt_ms, quiet, nsp), (prosody, feats), (emb, is_caller, voice_sim) = await asyncio.gather(
            self.transcribe(audio, hint, prefer),
            asyncio.to_thread(measure_voice, audio),
            asyncio.to_thread(self.speaker.score, audio))
        reading = self.calib.read(feats)
        if not is_caller and self.calib.same_voice(feats, voice_sim):
            # borderline voice print (short turns / a changed tone score low) but pitch and loudness are the
            # caller's own: keep it. Replay of 29 real calls: "Hello." / "Okay, bye." from the caller at 0.37-0.40
            self.event("kept_caller", {"voice_similarity": voice_sim, "text": text})
            is_caller = True
        if not text and quiet and voice_sim is not None and voice_sim >= 0.55:
            # Whisper dropped these words only for being quiet, and the voice is the confirmed caller's.
            # Never before the caller is confirmed: the replay showed quiet drops at pickup were mostly
            # Whisper's noise phrases ("हुआ है", "Thank you."), and these lines are digitally silent
            # between words, so loudness over the noise floor cannot tell them apart.
            self.event("recovered_speech", {"text": quiet, "voice_similarity": voice_sim, "snr_db": reading.snr_db})
            text = quiet
        if VOICEMAIL.search(text) and self.accepted_turns == 0:
            # carrier announcement / voicemail greeting: only at the start. Later, "trainer is not available?"
            # from the caller must not hang up the call
            self.event("voicemail", {"text": text})
            await self.hangup("voicemail")
            return
        last_agent = next((t["text"] for t in reversed(self.call.turns) if t["role"] == "agent"), "")
        # echo of our own voice only arrives while (or just after) the agent is playing, and never sounds
        # like the caller: the replay showed "Yes, I have two minutes." (answering "do you have two
        # minutes?") dropped as echo
        overlapped = self.play_until and started <= self.play_until + 0.8
        if text and last_agent and overlapped and not (voice_sim is not None and voice_sim >= 0.5) \
                and len(text.split()) >= 3 and difflib.SequenceMatcher(
                None, text.lower(), last_agent.lower()).find_longest_match().size >= 0.6 * len(text):
            self.event("noise", {"duration_s": round(dur_s, 2), "why": "echo of the agent's own voice", "text": text})
            return
        if not text or len(text.strip(" .?!,।")) < 1:
            if dur_s >= 1.0 and voice_sim is not None and voice_sim >= 0.4:
                self.last_dropped_speech = time.monotonic()  # the caller spoke, we got no words: ask to repeat
            self.event("noise", {"stt_ms": round(stt_ms), "duration_s": round(dur_s, 2), "why": "no words",
                                 "voice_similarity": voice_sim, "quiet_text": quiet or None})
            return
        latency_ms = None
        if sess and sess.history and self.play_until:
            latency_ms = max(0, int((started - self.play_until) * 1000)) if not barge else 0
        timing = TurnTiming(barge_in=barge, response_latency_ms=latency_ms,
                            speech_rate_wps=round(len(text.split()) / max(dur_s, 0.5), 2))
        if prosody is not None:
            prosody.words_per_s = round(len(text.split()) / dur_s, 2) if dur_s > 0 else None
        rms = float(np.sqrt(np.mean(audio * audio))) if len(audio) else 0.0
        confirmed = self.speaker.confirmed
        caller_voice = voice_sim is not None and voice_sim >= (0.5 if overlap else 0.4)
        psim = None if confirmed else self.speaker.provisional_sim(emb)
        # not confirmed yet, but one clean turn is known: noise scored 0.07-0.2 against it, the caller 0.49+
        unlike_first = psim is not None and psim < 0.35
        # a quiet stock phrase with the agent silent is believed only if Whisper heard speech in it
        # (live 6bb1ca00ad: the caller's real "Yes." no-speech 0.17; noise phrases 0.55-0.88)
        if is_caller and self.stock_noise(text) and not caller_voice and (
                confirmed or overlap or unlike_first or (rms < 0.035 and nsp >= 0.3)):
            self.event("noise", {"duration_s": round(dur_s, 2), "why": "noise phrase, not the caller's voice",
                                 "text": text, "voice_similarity": voice_sim if confirmed else psim,
                                 "rms": round(rms, 3), "overlap": overlap})
            return
        if is_caller and self.hangup_after_speaking and self.stock_noise(text) and not caller_voice:
            # after our goodbye a noise phrase never reopens the call (live call 9d69a31bae said goodbye twice
            # more to noise heard as "हुआ है"). Real sentences always do: after "I think we've lost the line"
            # the caller's "what do you have in the gym?" must be answered
            self.event("noise", {"duration_s": round(dur_s, 2), "why": "after goodbye, not clearly the caller",
                                 "text": text, "voice_similarity": voice_sim})
            return
        if not is_caller and self.speaker.voice_changed(emb, len(audio)):
            self.event("kept_caller", {"voice_similarity": voice_sim, "text": text, "why": "caller's voice changed"})
            is_caller = True
        if not is_caller:
            self.last_dropped_speech = time.monotonic()
            self.event("noise", {"duration_s": round(dur_s, 2), "why": "different voice (background)",
                                 "text": text, "voice_similarity": voice_sim})
            return
        if prosody is not None and os.getenv("ACIE_PROSODY", "0") == "1" and self.is_background(prosody):
            self.event("noise", {"duration_s": round(dur_s, 2), "why": "background voice", "text": text,
                                 "loudness_db": prosody.loudness_db, "f0_hz": prosody.f0_median_hz})
            return
        # accepted as the caller. Only CLEAN turns teach the voice print (agent silent, loud enough, real
        # words): the old lock learned from everything, so noise heard as words could become "the caller"
        self.accepted_turns += 1
        self.in_turn = True
        # callers who talk over the agent still get learned from their clearly loud, longer turns
        if not self.stock_noise(text) and rms >= CLEAN_RMS and (not overlap or (rms >= 0.05 and dur_s >= 1.5)):
            self.speaker.learn(emb, len(audio))
        self.calib.accept(feats)
        if prosody is not None:
            self.primary.append(prosody)
        self.no_response_count = 0
        turn_in = TurnInput(session_id=self.session_id, customer_id=self.call.customer_id, text=text, timing=timing,
                            prosody=prosody, voice=reading)

        sentences: asyncio.Queue = asyncio.Queue()
        epoch = self.speaking_epoch
        first_audio: dict = {}
        speaker = asyncio.create_task(self.speak_queue(sentences, epoch, first_audio, t_end))

        t_acie = time.monotonic()
        # shielded: an ACIE turn may move money / change dialog state, so it always completes
        result = await asyncio.shield(self.pipeline.run_turn(turn_in, on_sentence=sentences.put_nowait))
        sess = self.pipeline.sessions.get(self.session_id)
        acie_ms = (time.monotonic() - t_acie) * 1000
        await sentences.put(None)
        await speaker

        cust = {"role": "customer", "text": text, "stt_lang": stt_lang, "stt_ms": round(stt_ms),
                "prosody": prosody.model_dump() if prosody else None, "voice_similarity": voice_sim,
                "voice": reading.model_dump(), "noise_db": self.calib.noise_db,
                "emotion": emotion_label(reading, result.state.frustration, self.wants_to_buy(sess),
                                         result.state.confusion),
                "duration_s": round(dur_s, 2), "barge_in": barge, "t": time.time()}
        agent = {
            "role": "agent", "text": result.text, "intent": result.plan.intent, "t": time.time(),
            "state": result.state.model_dump(), "expression": result.expression.model_dump(),
            "validation": result.validation.model_dump(), "acie_timings_ms": result.timings_ms,
            "latency_ms": {"stt": round(stt_ms), "acie_total": round(acie_ms),
                           "first_audio_after_speech_end": first_audio.get("ms"),
                           "tts_first_chunk": first_audio.get("tts_ms")},
            "interrupted": epoch != self.speaking_epoch,
            "nlu": json.loads(sess.dialog["_nlu"]) if sess and sess.dialog.get("_nlu") else None,
            "rule_intent": sess.dialog.get("_rule_intent") if sess else None,
            "trace": trace(result, sess.dialog if sess else None),
        }
        self.call.turns += [cust, agent]
        self.last_replied_at = time.monotonic()
        self.event("turn", cust)
        self.event("turn", agent)
        self.hangup_after_speaking = False  # a new reply supersedes an earlier goodbye
        if result.plan.intent == "link_sent":
            asyncio.create_task(self.send_link())
        if result.plan.end_call:
            self.hangup_after_speaking = True
            self.hangup_reason = "safety_distress" if result.plan.intent == "safety_distress" else "agent_ended"

    @staticmethod
    def stock_noise(text: str) -> bool:
        # punctuation only: \w does not cover Devanagari vowel signs ("कर दो" became "कर द")
        t = re.sub(r"[.,!?।॥…\"“”‘’()\[\]:;~-]", " ", text.lower())
        t = re.sub(r"\s+", " ", t).strip()
        if t in STOCK_NOISE:
            return True
        # near-variants: fillers around it and repeats ("कर दो कि अ", "कर दो कर दो कर दो कर दो", "करते हैं।")
        words = [w for w in t.split() if w not in NOISE_FILLERS]
        if not words:
            return bool(t)
        for n in (1, 2, 3):  # collapse a repeated 1-3 word unit
            unit = words[:n]
            if len(words) % n == 0 and unit * (len(words) // n) == words:
                words = unit
                break
        return " ".join(words) in STOCK_NOISE

    @staticmethod
    def wants_to_buy(sess) -> bool:
        """The call's NLU says they are leaning to buy (acie/understanding.py buying_signal)."""
        try:
            return float(json.loads(sess.dialog["_nlu"]).get("buying_signal", 0)) >= 0.6
        except Exception:
            return False

    async def transcribe(self, audio: np.ndarray, hint: str | None, prefer: str | None):
        stt = self.app.stt
        if hasattr(stt, "transcribe_detail"):
            out = await stt.transcribe_detail(audio, hint, prefer)
            return out if len(out) == 5 else (*out, 0.5)
        text, lang, ms = await stt.transcribe(audio, hint, prefer)  # test doubles
        return text, lang, ms, "", 0.5

    async def send_link(self) -> None:
        """The agent just said it is sending the payment link: send it (once per call)."""
        p = self.product
        if not p or not p.sms_text or self.call.sms:
            return
        if p.allowed_numbers and self.call.to not in p.allowed_numbers:
            return  # same guard as the dial gate: a restricted campaign never texts other numbers
        sess = getattr(getattr(self, "pipeline", None), "sessions", {}).get(getattr(self, "session_id", ""))
        link = (sess.dialog.get("payment_link_url") if sess is not None else "") or p.payment_link  # business API link
        text = p.sms_text.format_map(_SafeMap({"seller": p.seller_name, "product": p.product_name,
                                               "offer_price": p.offer_price, "link": link}))
        res = await self.app.plivo.send_sms(self.call.to, text)
        self.call.sms.append({"text": text, "to": self.call.to, "at": time.time(), **res})
        self.event("sms", res)

    async def speak_queue(self, q: asyncio.Queue, epoch: int, first_audio: dict, t_end: float) -> None:
        while True:
            s = await q.get()
            if s is None:
                return
            if epoch != self.speaking_epoch:
                continue  # barged in: drain without speaking
            while not q.empty():  # one TTS request for everything that's ready: no gaps between sentences
                nxt = q.get_nowait()
                if nxt is None:
                    await q.put(None)
                    break
                s = f"{s} {nxt}"
            # the turn's ExpressionParams aren't stored until run_turn returns, so read the script. One language
            # per reply: switching Fish's language between sentences changed the accent mid-answer
            lang = first_audio.setdefault("lang", detect_language(s))
            await self.speak([s], lang, epoch, first_audio, t_end)

    async def speak(self, sentences: list[str], language: str, epoch: int | None = None,
                    first_audio: dict | None = None, t_end: float | None = None) -> None:
        epoch = self.speaking_epoch if epoch is None else epoch
        async with self.speak_lock:
            await self._speak(sentences, language, epoch, first_audio, t_end)

    async def _speak(self, sentences: list[str], language: str, epoch: int,
                     first_audio: dict | None, t_end: float | None) -> None:
        for s in sentences:
            if epoch != self.speaking_epoch:
                return
            if DEVA.search(s):  # Hindi turn: respell English names the Hindi voice misreads
                for rx, say in self.pronounce:
                    s = rx.sub(say, s)
            t0 = time.monotonic()
            try:
                async for pcm in self.app.tts.stream(s, language):
                    if first_audio is not None and "ms" not in first_audio and t_end is not None:
                        first_audio["ms"] = round((time.monotonic() - t_end) * 1000)
                        first_audio["tts_ms"] = round((time.monotonic() - t0) * 1000)
                    if not await self.play_pcm(pcm, epoch):
                        return
            except Exception as e:
                log.warning("[%s] TTS failed: %s", self.call.call_id, e)
                self.event("error", {"message": f"TTS: {e}"})
                return

    def event(self, kind: str, data: dict) -> None:
        if kind != "turn":
            self.call.events.append({"type": kind, "t": time.time(),
                                     "after_turn": len(self.call.turns), **data})
        self.app.broadcast({"type": kind, "call_id": self.call.call_id, **data})
