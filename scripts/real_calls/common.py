"""Shared helpers for the real-call scripts: load calls, detect repetition and other failure kinds.

Real calls live in data/real/calls (git-ignored). Nothing here ever writes them anywhere committable.
"""

from __future__ import annotations

import difflib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CALLS = ROOT / "data" / "real" / "calls"
REP_RATIO = 0.6
_DEV = re.compile(r"[ऀ-ॿ]")
_LAT = re.compile(r"[A-Za-z]")
_PUNCT = re.compile(r"[\s\.,!?;:\"'()\[\]।\-]+")
_QWORD = re.compile(r"\?|\b(why|what|how|which|who|where|when|is it|are you|can i|can you|do you|does)\b|"
                    r"क्या|कैसे|कौन|कितन|क्यों|कहाँ|कहां|कब|किस", re.I)
_REQ_EN = re.compile(r"(english|speak english|shift to english|in english)", re.I)
_REQ_HI = re.compile(r"(hindi (mein|me|में)|हिंदी में|हिन्दी में|tell me in hindi|speak hindi)|मुझे हिंदी", re.I)
_NO_HI = re.compile(r"(don.?t|do not|can.?t|not able to).{0,20}(hindi|understand your language)|don.?t know hindi|हिंदी (नहीं|शो) (आती|आता|समझ)", re.I)
_NO_EN = re.compile(r"(english|इंग्लिश|अंग्रेज).{0,10}(नहीं|nahi)", re.I)
_NO = re.compile(r"नहीं चाहिए|नही चाहिए|नहीं जानना|नहीं$|^नहीं|^ने$|not interested|no, thank you|i don.?t know$|मुझे नहीं चाहिए|8 में नहीं|don.?t want", re.I)
_SELL_INTENTS = {"pitch", "price", "reoffer", "confirm_close", "recover", "open_question"}
_OFFERISH = re.compile(r"\?\s*$")


def norm(t: str) -> str:
    return _PUNCT.sub(" ", t.lower()).strip()


def ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()


def script(t: str) -> str:
    """'hi' (mostly Devanagari), 'en' (mostly Latin) or 'mix'."""
    d, l = len(_DEV.findall(t)), len(_LAT.findall(t))
    if d + l == 0:
        return "none"
    f = d / (d + l)
    return "hi" if f > 0.6 else ("en" if f < 0.15 else "mix")


def load_calls() -> list[dict]:
    out = []
    for f in sorted(CALLS.glob("*.json")):
        d = json.loads(f.read_text())
        d["_id6"] = f.stem[:6]
        rv = CALLS / f.stem / "review.json"
        d["_review"] = json.loads(rv.read_text()) if rv.exists() else None
        out.append(d)
    return out


def is_q(text: str) -> bool:
    return bool(_QWORD.search(text))


def customer_wants_lang(text: str) -> str | None:
    if _NO_HI.search(text) or _REQ_EN.search(text):
        return "en"
    if _REQ_HI.search(text) or _NO_EN.search(text):
        return "hi"
    return None


def analyse_call(call: dict) -> dict:
    """Failure instances in one logged call. Every instance: {kind, turn, detail...}."""
    turns = call["turns"]
    inst: list[dict] = []
    agents = [(i, t) for i, t in enumerate(turns) if t["role"] == "agent"]
    seen: list[tuple[int, str, str]] = []
    for i, t in agents:
        for j, prev, pint in seen:
            r = ratio(t["text"], prev)
            if r > REP_RATIO:
                # who said what right before
                before = next((turns[k]["text"] for k in range(i - 1, -1, -1) if turns[k]["role"] == "customer"), "")
                inst.append({"kind": "repeat_text", "turn": i, "of": j, "ratio": round(r, 2), "intent": t.get("intent"),
                             "prev_intent": pint, "customer_before": before})
                break
        seen.append((i, t["text"], t.get("intent")))
    # same intent said again (informational/selling intents only; mechanical ones are counted separately)
    cnt: dict[str, int] = {}
    for i, t in agents:
        it = t.get("intent")
        cnt[it] = cnt.get(it, 0) + 1
        if cnt[it] > 1 and it in ("pitch", "price", "reoffer", "open_question", "faq", "confirm_close", "answer", "not_understood", "off_topic"):
            inst.append({"kind": "repeat_intent", "turn": i, "intent": it, "count": cnt[it]})
    # the same question asked again after the customer answered in between
    asks = []
    for i, t in agents:
        sents = [s for s in re.split(r"(?<=[.!?।])\s+", t["text"]) if s.strip().endswith("?")]
        if sents:
            q = sents[-1]
            for j, pq in asks:
                if ratio(q, pq) > REP_RATIO and any(turns[k]["role"] == "customer" for k in range(j, i)):
                    inst.append({"kind": "reasked_question", "turn": i, "of": j, "q": q})
                    break
            asks.append((i, q))
    # customer questions that the next agent reply did not address
    for i, t in enumerate(turns):
        if t["role"] != "customer" or i + 1 >= len(turns) or turns[i + 1]["role"] != "agent":
            continue
        nxt = turns[i + 1]
        if is_q(t["text"]) and nxt.get("intent") in ("not_understood", "off_topic", "reoffer", "pitch", "confirm_close",
                                                    "link_sent", "price", "recover") and len(t["text"].split()) >= 2:
            inst.append({"kind": "ignored_question", "turn": i, "text": t["text"], "agent_intent": nxt.get("intent")})
    # pushy: an offer or ask for the link right after a clear "no", or more than one re-offer
    n_reoffer = 0
    for i, t in enumerate(turns):
        if t["role"] == "agent" and t.get("intent") == "reoffer":
            n_reoffer += 1
            if n_reoffer > 1:
                inst.append({"kind": "pushy", "turn": i, "why": "second re-offer"})
        if t["role"] == "customer" and _NO.search(t["text"]) and i + 1 < len(turns):
            nxt = turns[i + 1]
            if nxt.get("intent") in ("reoffer", "pitch", "price", "confirm_close", "open_question", "recover"):
                inst.append({"kind": "pushy", "turn": i + 1, "why": "offer right after a no", "customer": t["text"]})
    # wrong language: after the caller asked for a language the reply is in the other one, or an English
    # sentence of 3+ words got a Devanagari-only reply
    want = None
    for i, t in enumerate(turns):
        if t["role"] != "customer" or i + 1 >= len(turns) or turns[i + 1]["role"] != "agent":
            continue
        w = customer_wants_lang(t["text"])
        want = w or want
        s = script(turns[i + 1]["text"])
        if want and s in ("hi", "en") and s != want:
            inst.append({"kind": "wrong_language", "turn": i + 1, "wanted": want, "got": s, "asked_now": bool(w)})
        elif not want and script(t["text"]) == "en" and len(t["text"].split()) >= 3 and s == "hi":
            inst.append({"kind": "wrong_language", "turn": i + 1, "wanted": "en", "got": s, "asked_now": False})
    # misheard: the agent says it did not understand / FSM did not understand / careful STT disagrees
    for i, t in enumerate(turns):
        if t["role"] == "agent" and t.get("intent") in ("not_understood", "clarify"):
            inst.append({"kind": "misheard", "turn": i, "why": "agent had to ask again"})
    rv = call.get("_review")
    if rv:
        for p in rv["problems"]:
            if p.startswith("STT:"):
                inst.append({"kind": "misheard", "turn": -1, "why": "live vs careful STT differ"})
    # frustration markers in customer words
    frus = re.compile(r"पागल|fix this|shit|fuck|बार बार|bar bar|repeat क्यों|रिपेट|answer me|नकली|fake|बेवकूफ|stupid|fed up", re.I)
    for i, t in enumerate(turns):
        if t["role"] == "customer" and frus.search(t["text"]):
            inst.append({"kind": "frustration_words", "turn": i, "text": t["text"]})
    return {"call": call["_id6"], "campaign": call.get("campaign"), "end": call.get("end_reason"),
            "n_agent": len(agents), "n_customer": sum(1 for t in turns if t["role"] == "customer"), "instances": inst}


def count_repetitions(agent_texts: list[str]) -> int:
    """Agent replies with ratio > 0.6 to any earlier agent reply in the same call (the headline metric)."""
    n = 0
    for i, t in enumerate(agent_texts):
        if any(ratio(t, p) > REP_RATIO for p in agent_texts[:i]):
            n += 1
    return n
