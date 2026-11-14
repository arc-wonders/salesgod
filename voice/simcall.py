"""Simulated phone call through the real CallSession (everything except Plivo itself).

A fake Plivo WebSocket feeds the customer's lines (Fish TTS -> 8 kHz mu-law, in 20 ms frames,
in real time) and collects the agent's playAudio. The last line barges in while the agent is
still talking.

  /opt/salesgod/venv/bin/python -m voice.simcall [llm|template]
"""

from __future__ import annotations

import asyncio
import base64
import json
import sys
import time

from acie.expression import OpenAICompatibleBackend, TemplateBackend
from acie.personalization.profile_store import MemoryProfileStore

from .audio import pcm16_to_ulaw
from .call import CallRecord, CallSession
from .server import App, Config
from .stt import WhisperSTT
from .tts import FishTTS

SCRIPTS = {
    "bank_demo": [  # (text, wait for agent to finish first?)
        ("मुझे पचास हज़ार रुपये भेजने हैं", True),
        ("अच्छा, पच्चीस हज़ार ही भेज दो", True),
        ("हाँ कर दो", True),
        ("credit card ke baare mein batao", True),
        ("नहीं नहीं, अभी नहीं चाहिए", False),  # barge-in
    ],
    "claude-pro": [
        ("कौन बोल रहा है?", True),
        ("हाँ बोलिए, क्या है", True),
        ("ये कितने का है?", True),
        ("नहीं, अभी जरूरत नहीं है", True),
        ("अच्छा ठीक है, लिंक भेज दो", True),
    ],
}


class FakePlivo:
    async def hangup(self, call):
        print("  [plivo] hangup requested:", call.end_reason)

    async def send_sms(self, to, text):
        print("  [plivo] SMS to", to, ":", text)
        return {"status": "simulated"}


class FakeWS:
    def __init__(self):
        self.inbox: asyncio.Queue = asyncio.Queue()
        self.audio_s = 0.0
        self.last_play = 0.0
        self.clears = 0

    async def receive_text(self) -> str:
        return await self.inbox.get()

    async def send_text(self, text: str) -> None:
        m = json.loads(text)
        if m["event"] == "playAudio":
            self.audio_s += len(base64.b64decode(m["media"]["payload"])) / 8000
            self.last_play = time.monotonic()
        elif m["event"] == "clearAudio":
            self.clears += 1
            print("  [plivo] clearAudio (barge-in)")

    async def close(self):
        await self.inbox.put(json.dumps({"event": "stop"}))


async def main(backend: str, campaign: str) -> None:
    app = App(cfg=Config(), plivo=FakePlivo())
    app.stt, app.tts = await asyncio.to_thread(WhisperSTT), FishTTS()
    app.store = MemoryProfileStore()
    app.backends = {"llm": OpenAICompatibleBackend(), "template": TemplateBackend()}
    events = []
    app.broadcast = lambda ev: events.append(ev) or (ev.get("type") == "turn" and print(
        f"  {ev['role']:8s}: {ev['text']}" + (f"   [{ev.get('intent')}] first-audio "
                                               f"{(ev.get('latency_ms') or {}).get('first_audio_after_speech_end')} ms"
                                               if ev['role'] == 'agent' and ev.get('latency_ms') else
                                               f"   [STT {ev.get('stt_ms')} ms]" if ev['role'] == 'customer' else "")))

    async def finish(call, pipeline, sid):
        print("  call finished:", call.end_reason)
    app.finish = finish

    call = CallRecord(to="+910000000000", customer_id="c1", language="hi", backend=backend, campaign=campaign,
                      customer_name="Yash")
    ws = FakeWS()
    session = CallSession(app, call, ws, "stream-1")
    runner = asyncio.create_task(session.run())

    agent_turns = lambda: sum(1 for e in events if e.get("type") == "turn" and e["role"] == "agent")

    async def agent_done(expected: int):
        while True:  # the agent has replied `expected` times and finished playing it
            await asyncio.sleep(0.2)
            if agent_turns() >= expected and ws.audio_s > 0 and not session.agent_speaking() and time.monotonic() - ws.last_play > 0.8:
                return

    # like Plivo: a 20 ms frame every 20 ms for the whole call (silence between lines), each with an
    # absolute epoch-ms timestamp (a relative-timestamp assumption crashed two live calls)
    silence = pcm16_to_ulaw(b"\x00\x00" * 160)
    pending: asyncio.Queue = asyncio.Queue()

    async def feeder():
        t0 = int(time.time() * 1000)
        n = 0
        while not runner.done():
            frame = pending.get_nowait() if not pending.empty() else silence
            await ws.inbox.put(json.dumps({"event": "media", "media": {
                "payload": base64.b64encode(frame).decode(), "timestamp": str(t0 + 20 * n)}}))
            n += 1
            await asyncio.sleep(0.02)
    feed = asyncio.create_task(feeder())

    for n, (text, wait) in enumerate(SCRIPTS[campaign]):
        if wait:
            await agent_done(n + 1)
        else:
            while not session.agent_speaking():
                await asyncio.sleep(0.05)
            await asyncio.sleep(0.8)  # interrupt mid-sentence
        pcm = b"".join([c async for c in app.tts.stream(text, "hi")])
        ulaw = pcm16_to_ulaw(pcm)
        for i in range(0, len(ulaw), 160):
            pending.put_nowait(ulaw[i:i + 160].ljust(160, b"\xff"))
        while not pending.empty():
            await asyncio.sleep(0.05)
    await agent_done(len(SCRIPTS[campaign]) + 1)
    await asyncio.sleep(2)
    if not runner.done():
        await ws.close()
    await asyncio.wait_for(runner, 15)
    feed.cancel()
    print(f"\nagent audio sent: {ws.audio_s:.1f}s, clearAudio: {ws.clears}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "llm", sys.argv[2] if len(sys.argv) > 2 else "claude-pro"))
