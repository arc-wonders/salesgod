"""Replay RECORDED customer audio through the real CallSession, to test the calibrator on real phones.

The left channel of data/voice/calls/<id>/call.wav (the customer line exactly as Plivo sent it,
noise, echo and background voices included) is fed in real time as 20 ms mu-law frames. Whisper is
real; the agent uses the template backend and a silent stand-in TTS (no LLM / Fish cost), so the
agent's words differ from the original call but the customer audio is the real one.

Prints per call: what the original call heard vs what the replay heard, every noise drop and why,
recovered quiet speech, the line noise floor, and the emotion read on each turn.

  VOICE_DATA_DIR=/tmp/replay python -m voice.replaycall SRC_CALLS_DIR [call_id ...] [--jobs 8]
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import sys
import time
import wave
from pathlib import Path

import numpy as np

from acie.expression import TemplateBackend
from acie.personalization.profile_store import MemoryProfileStore

from .audio import pcm16_to_ulaw
from .call import CallRecord, CallSession
from .server import DATA, App, Config
from .simcall import FakePlivo, FakeWS
from .stt import WhisperSTT


class SilentTTS:
    """Stands in for Fish: silence about as long as the sentence would take to say."""

    async def stream(self, text: str, language: str):
        n = int(8000 * max(0.6, len(text) / 14))
        for i in range(0, n, 1600):
            yield b"\x00\x00" * min(1600, n - i)
            await asyncio.sleep(0)


def customer_line(path: Path) -> bytes:
    with wave.open(str(path)) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").reshape(-1, w.getnchannels())[:, 0]
    return pcm16_to_ulaw(x.astype("<i2").tobytes())


async def replay(app: App, src: Path, call_id: str, report: dict) -> None:
    meta = json.loads((src / f"{call_id}.json").read_text())
    events: list[dict] = []
    call = CallRecord(to="+910000000000", customer_id=meta.get("customer_id", "c1"), language=meta.get("language", "hi"),
                      backend="template", campaign=meta.get("campaign", "bank_demo"),
                      customer_name=meta.get("customer_name", ""))
    report["listeners"][call.call_id] = events
    ws = FakeWS()
    session = CallSession(app, call, ws, "stream-" + call_id)
    runner = asyncio.create_task(session.run())
    ulaw = customer_line(src / call_id / "call.wav")
    t0 = int(time.time() * 1000)
    start = time.monotonic()
    for n, i in enumerate(range(0, len(ulaw), 160)):
        if runner.done():
            break
        await ws.inbox.put(json.dumps({"event": "media", "media": {
            "payload": base64.b64encode(ulaw[i:i + 160].ljust(160, b"\xff")).decode(), "timestamp": str(t0 + 20 * n)}}))
        await asyncio.sleep(max(0.0, start + 0.02 * (n + 1) - time.monotonic()))
    for _ in range(100):  # let the last turn finish
        if not session.busy and session.utterances.empty():
            break
        await asyncio.sleep(0.2)
    if not runner.done():
        await ws.close()
    try:
        await asyncio.wait_for(runner, 30)
    except Exception:
        pass
    report["calls"][call_id] = {"original": meta, "replay": call.public(), "noise_db": session.calib.noise_db}


def show(call_id: str, r: dict) -> dict:
    orig = [t for t in r["original"]["turns"] if t["role"] == "customer"]
    rep = [t for t in r["replay"]["turns"] if t["role"] == "customer"]
    ev = r["replay"]["events"]
    print(f"\n=== {call_id}  line noise {r['noise_db']} dBFS  | original heard {len(orig)} turns, replay heard {len(rep)}")
    for t in orig:
        print(f"   orig: {t['text']}")
    print(f"   replay ended: {r['replay'].get('end_reason')}")
    for t in r["replay"]["turns"]:
        if t["role"] == "agent":
            print(f"   agent: {t['text'][:110]} [{t.get('intent')}]")
    allt = r["replay"]["turns"]
    for t in rep:
        i = allt.index(t)
        st = next((x.get("state") or {} for x in allt[i + 1:] if x["role"] == "agent"), {})
        e, v = t.get("emotion") or {}, t.get("voice") or {}
        z = " ".join(f"{k[:5]}{x:+.1f}" for k, x in (v.get("z") or {}).items())
        print(f"   now : {t['text']}\n         -> {e.get('label')} ({e.get('confidence')}) arousal {e.get('arousal')} "
              f"words-frustration {st.get('frustration')} sim {t.get('voice_similarity')}  [{z}]")
    for e in ev:
        if e["type"] in ("noise", "recovered_speech", "voicemail", "kept_caller", "barge_in", "barge_in_ignored"):
            print(f"   {e['type']}: {e.get('why', '')} {e.get('text', '')!r} sim={e.get('voice_similarity')} snr={e.get('snr_db')}")
    return {"orig": len(orig), "replay": len(rep),
            "recovered": sum(e["type"] == "recovered_speech" for e in ev),
            "kept": sum(e["type"] == "kept_caller" for e in ev),
            "noise": [e.get("why") for e in ev if e["type"] == "noise"],
            "labels": [(t.get("emotion") or {}).get("label") for t in rep],
            "stt_ms": [t.get("stt_ms") for t in rep]}


async def main(src: Path, ids: list[str], jobs: int) -> None:
    assert DATA != src.parent, "set VOICE_DATA_DIR to a scratch folder: replays must not write into real call logs"
    (DATA / "calls").mkdir(parents=True, exist_ok=True)
    if (src.parent / "products.json").exists() and not (DATA / "products.json").exists():
        shutil.copy(src.parent / "products.json", DATA / "products.json")
    report: dict = {"calls": {}, "listeners": {}}
    app = App(cfg=Config(), plivo=FakePlivo())
    app.stt, app.tts = await asyncio.to_thread(WhisperSTT), SilentTTS()
    app.store = MemoryProfileStore()
    app.backends = {"template": TemplateBackend()}
    app.broadcast = lambda ev: report["listeners"].get(ev.get("call_id"), []).append(ev)

    async def finish(call, pipeline, sid):
        pass
    app.finish = finish
    ids = ids or sorted(p.stem for p in src.glob("*.json") if (src / p.stem / "call.wav").exists())
    sem = asyncio.Semaphore(jobs)

    async def one(cid):
        async with sem:
            try:
                await replay(app, src, cid, report)
            except Exception as e:
                print("replay failed", cid, type(e).__name__, e)
    t0 = time.monotonic()
    await asyncio.gather(*(one(c) for c in ids))
    summary = {cid: show(cid, r) for cid, r in sorted(report["calls"].items())}
    tot = lambda k: sum(s[k] for s in summary.values())
    labels: dict[str, int] = {}
    for s in summary.values():
        for lab in s["labels"]:
            labels[lab] = labels.get(lab, 0) + 1
    why: dict[str, int] = {}
    for s in summary.values():
        for w in s["noise"]:
            why[w] = why.get(w, 0) + 1
    stt = [x for s in summary.values() for x in s["stt_ms"] if x]
    print(f"\n##### {len(summary)} calls in {time.monotonic() - t0:.0f}s | customer turns: original {tot('orig')}, "
          f"replay {tot('replay')} | recovered quiet speech {tot('recovered')} | borderline voices kept {tot('kept')} | noise drops {why} | emotions {labels}"
          f" | STT median {np.median(stt) if stt else 0:.0f} ms")
    out = DATA / "replay_report.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print("report:", out)


if __name__ == "__main__":
    args = sys.argv[1:]
    jobs = 8
    if "--jobs" in args:
        k = args.index("--jobs")
        jobs = int(args[k + 1])
        del args[k:k + 2]
    asyncio.run(main(Path(args[0]), args[1:], jobs))
