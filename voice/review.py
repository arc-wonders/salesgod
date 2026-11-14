"""Post-call review: what went wrong on a call, component by component.

  /opt/salesgod/venv/bin/python -m voice.review <call_id> [--json]

Checks:
  STT        live transcript vs a slower second opinion (beam 5, forced hi and en) per customer turn
  VAD/turns  barge-ins, dropped-noise events, customer talk vs agent talk from the stereo recording
  decision   intent path, fallbacks, open_question share (= the FSM didn't understand)
  expression long agent turns, script/language mismatch, repeated replies
  latency    end of customer speech -> first agent audio, per turn
  outcome    how the call ended
"""

from __future__ import annotations

import difflib
import json
import re
import sys
import wave
from pathlib import Path

import numpy as np

from .server import CALLS_DIR

DEVA = re.compile(r"[ऀ-ॿ]")


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path)) as w:
        a = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768
        return a.reshape(-1, w.getnchannels()), w.getframerate()


def speech_seconds(x: np.ndarray, rate: int, thr: float = 0.02) -> float:
    frame = rate // 50
    n = len(x) // frame
    if n == 0:
        return 0.0
    rms = np.sqrt((x[: n * frame].reshape(n, frame) ** 2).mean(axis=1))
    return float((rms > thr).sum() / 50)


def main(call_id: str, as_json: bool = False) -> dict:
    call = json.loads((CALLS_DIR / f"{call_id}.json").read_text())
    d = CALLS_DIR / call_id
    report: dict = {"call_id": call_id, "campaign": call.get("campaign"), "end_reason": call.get("end_reason"),
                    "problems": [], "turns": []}
    P = report["problems"]

    from faster_whisper import WhisperModel
    model = WhisperModel("large-v3", device="cuda", compute_type="float16")

    def second_opinion(audio: np.ndarray) -> dict:
        out = {}
        for lang in ("hi", "en"):
            segs, _ = model.transcribe(audio, language=lang, beam_size=5, condition_on_previous_text=False,
                                       without_timestamps=True)
            segs = list(segs)
            out[lang] = (" ".join(s.text.strip() for s in segs).strip(),
                         min([s.avg_logprob for s in segs] or [-9]))
        return out

    turns = call["turns"]
    cust_n = 0
    intents, latencies, agent_words = [], [], []
    prev_agent = ""
    for t in turns:
        row = {"role": t["role"], "text": t["text"]}
        if t["role"] == "customer":
            cust_n += 1
            wav = d / f"customer_{cust_n:02d}.wav"
            if wav.exists():
                a, _ = load_wav(wav)
                so = second_opinion(a[:, 0])
                best_lang = max(so, key=lambda k: so[k][1])
                best = so[best_lang][0]
                sim = difflib.SequenceMatcher(None, t["text"].lower(), best.lower()).ratio()
                row.update(second_opinion=best, second_lang=best_lang, agreement=round(sim, 2))
                if sim < 0.6:
                    P.append(f"STT: turn {cust_n} live={t['text']!r} vs careful={best!r} ({best_lang})")
            if t.get("barge_in"):
                row["barge_in"] = True
        else:
            intent = t.get("intent")
            intents.append(intent)
            row["intent"] = intent
            words = len(t["text"].split())
            agent_words.append(words)
            lat = (t.get("latency_ms") or {}).get("first_audio_after_speech_end")
            if lat:
                latencies.append(lat)
                row["first_audio_ms"] = lat
                if lat > 2500:
                    P.append(f"latency: {lat} ms before the agent spoke ({intent})")
            if words > 35:
                P.append(f"expression: agent turn too long ({words} words, {intent}): {t['text'][:80]}...")
            lang = (t.get("expression") or {}).get("language")
            if lang == "hi" and len(re.findall(r"[A-Za-z]{3,}", t["text"])) > 6:
                P.append(f"expression: lots of English inside a Hindi turn ({intent})")
            if prev_agent and difflib.SequenceMatcher(None, prev_agent, t["text"]).ratio() > 0.75:
                P.append(f"decision/expression: agent repeated itself ({intent})")
            prev_agent = t["text"]
            v = t.get("validation") or {}
            if v.get("used_fallback"):
                row["fallback"] = v.get("issues")
            if t.get("interrupted"):
                row["interrupted"] = True
            if intent in ("open_question", "unknown"):
                P.append(f"decision: FSM didn't understand the customer ({intent}) after {turns[turns.index(t) - 1]['text']!r}")
        report["turns"].append(row)

    ev = call.get("events", [])
    barges = sum(1 for e in ev if e["type"] == "barge_in")
    noise = sum(1 for e in ev if e["type"] == "noise")
    errors = [e for e in ev if e["type"] == "error"]
    if errors:
        P.extend(f"error: {e.get('message')}" for e in errors)

    rec = d / "call.wav"
    talk = {}
    if rec.exists():
        a, rate = load_wav(rec)
        talk = {"customer_s": round(speech_seconds(a[:, 0], rate), 1), "agent_s": round(speech_seconds(a[:, 1], rate), 1),
                "total_s": round(len(a) / rate, 1)}
        if talk["agent_s"] > 3 * max(talk["customer_s"], 1):
            P.append(f"pitch: agent talked {talk['agent_s']}s vs customer {talk['customer_s']}s (monologue)")

    # compliance (voice_platform's platform floor + QA scorecard): did the call sound legitimate?
    from acie.decision.domains.product_sales import _ARE_YOU_AI, _OPT_OUT
    agent_turns = [t for t in turns if t["role"] == "agent"]
    compliance = {}
    if agent_turns and call.get("campaign") != "bank_demo":
        first = agent_turns[0]["text"]
        compliance["disclosed_ai"] = bool(re.search(r"\bAI\b|ए आई|एआई", first))
        compliance["disclosed_recording"] = bool(re.search(r"record|रिकॉर्ड", first, re.I))
        for i, t in enumerate(turns[:-1]):
            if t["role"] != "customer":
                continue
            nxt = turns[i + 1]
            if _OPT_OUT.search(t["text"]) and nxt.get("intent") not in ("opt_out", "callback"):
                P.append(f"compliance: opt-out not honoured: {t['text']!r} -> {nxt.get('intent')}")
                compliance["opt_out_honoured"] = False
            if _ARE_YOU_AI.search(t["text"]) and nxt.get("intent") != "are_you_ai" and not re.search(r"\bAI\b", nxt["text"]):
                P.append(f"compliance: 'are you a bot?' not answered honestly: {t['text']!r}")
                compliance["admitted_ai"] = False
        reoffers = sum(1 for t in agent_turns if t.get("intent") == "reoffer")
        if reoffers > 1:
            P.append(f"compliance: {reoffers} re-offers after a no (max 1)")
        asks = [t["text"] for t in agent_turns
                if re.search(r"otp|ओटीपी|cvv|card number|कार्ड नंबर|पिन|\bpin\b", t["text"], re.I)
                and not re.search(r"मत|don't|do not|never|कभी नहीं|mat ", t["text"], re.I)]
        if asks:
            P.append(f"compliance: agent mentioned payment secrets without a warning: {asks[0][:80]}")
        for k, v in compliance.items():
            if v is False and k.startswith("disclosed"):
                P.append(f"compliance: opening missing {k}")
        for m in call.get("sms", []):
            if m.get("status") != "sent":
                P.append(f"sms: link not sent ({m.get('status')}: {str(m.get('reason'))[:120]})")
    report["compliance"] = compliance

    report["summary"] = {
        "intent_path": intents, "barge_ins": barges, "noise_dropped": noise,
        "median_first_audio_ms": int(np.median(latencies)) if latencies else None,
        "max_first_audio_ms": max(latencies) if latencies else None,
        "avg_agent_words": round(float(np.mean(agent_words)), 1) if agent_words else None,
        "fallbacks": sum(1 for r in report["turns"] if r.get("fallback")), **talk,
        # a sale (link sent) stays a sale even if a later turn was classified declined, unless the link was cancelled
        "outcome": ("link_sent" if "link_sent" in intents and "link_cancelled" not in intents else
                    next((i for i in reversed(intents) if i in ("declined", "callback", "safety_distress",
                                                                 "human_agent", "goodbye")), "no outcome")),
    }
    if as_json:
        print(json.dumps(report, ensure_ascii=False, indent=1))
    else:
        print(f"== call {call_id}  ({call.get('campaign')}, ended: {call.get('end_reason')})")
        print("summary:", json.dumps(report["summary"], ensure_ascii=False))
        for r in report["turns"]:
            extra = {k: v for k, v in r.items() if k not in ("role", "text")}
            print(f"  {r['role']:8s} {r['text']}\n           {extra}" if extra else f"  {r['role']:8s} {r['text']}")
        print("problems:")
        for p in P or ["(none found)"]:
            print("  -", p)
    d.mkdir(exist_ok=True)
    (d / "review.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    return report


if __name__ == "__main__":
    main(sys.argv[1], "--json" in sys.argv)
