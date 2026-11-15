"""Long-term caller memory: how this person talks and what happened before, carried into the next call.

One JSON record per caller (key = customer id, i.e. the phone number), written at call end:
  NATS KV bucket `sg_customer` (when the NATS store is on) and always a JSON file next to the profiles
  (data/voice/profiles/caller_memory/<id>.json), so it works with NATS down.
Everything is derived deterministically from the finished session (no LLM call): language mix, words per turn,
question rate, pace, formality, objections, last deal state, outcome, a short summary per call (last 3 kept).
At call start the record becomes a compact "ABOUT THIS CALLER" block on the session context (decision layer + reply
prompt). A previous price is only ever presented as agreed when it was agreed; otherwise it is history, not an offer.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from pathlib import Path

log = logging.getLogger("acie.caller_memory")

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_ROMAN_HI = re.compile(r"\b(hai|haan|nahi|nahin|kya|kaise|mujhe|aap|theek|accha|acha|bhai|karo|kar|mein|ke|ka|ki|ji|bolo|batao)\b", re.I)
_HUMOUR = re.compile(r"\b(haha+|hehe+|lol|lmao|mazaak|majak)\b|😂|😄", re.I)
_SAFE = re.compile(r"[^A-Za-z0-9_-]")
_BUCKET = "sg_customer"


def key_for(customer_id: str) -> str:
    return _SAFE.sub("_", customer_id or "unknown")[:120]


def _lang_mix(texts: list[str]) -> dict:
    n = len(texts) or 1
    hi = sum(1 for t in texts if _DEVANAGARI.search(t))
    hg = sum(1 for t in texts if not _DEVANAGARI.search(t) and len(_ROMAN_HI.findall(t)) >= 1)
    en = len(texts) - hi - hg
    mix = {"hi": round(hi / n, 2), "hinglish": round(hg / n, 2), "en": round(en / n, 2)}
    return mix


def style_from_session(session, turns: list[dict] | None = None) -> dict:
    cust = [u.text for u in session.history if u.role == "customer" and (u.text or "").strip()]
    words = [len(t.split()) for t in cust]
    avg = round(sum(words) / len(words), 1) if words else 0.0
    qrate = round(sum(1 for t in cust if "?" in t or re.search(r"\b(what|how|why|kya|kaise|kitna)\b", t, re.I)) / len(cust), 2) if cust else 0.0
    wps = [t.get("prosody", {}).get("words_per_s") for t in (turns or []) if t.get("role") == "customer" and isinstance(t.get("prosody"), dict)]
    wps = [w for w in wps if w]
    pace = round(sum(wps) / len(wps), 2) if wps else None
    expr = getattr(session, "expression", None)
    return {"language_mix": _lang_mix(cust), "last_language": getattr(expr, "language", "") or "",
            "formality": getattr(expr, "formality", "med"), "avg_words_per_turn": avg,
            "answers": "short" if avg and avg < 6 else ("long" if avg > 18 else "medium"),
            "question_rate": qrate, "pace_words_per_s": pace,
            "pace": None if pace is None else ("fast" if pace > 3.2 else "slow" if pace < 2.0 else "normal"),
            "humour": any(_HUMOUR.search(t) for t in cust), "customer_turns": len(cust)}


def _call_summary(session, outcome: str, product: str) -> str:
    ctx = session.context
    c, dv = ctx.checklist, ctx.deal_view
    bits = [f"{time.strftime('%d %b', time.gmtime())}: {product or 'call'}, {len([u for u in session.history if u.role == 'customer'])} turns, outcome {outcome or ctx.decision}."]
    if ctx.objections:
        bits.append("Objections: " + ", ".join(ctx.objections) + ".")
    if dv.get("agreed_price"):
        bits.append(f"Agreed {dv.get('currency', '')}{float(dv['agreed_price']):,.0f}.")
    elif dv.get("last_customer_counter") or dv.get("last_agent_offer"):
        bits.append(f"Haggled: they asked {dv.get('last_customer_counter', '-')}, last offer {dv.get('last_agent_offer', '-')}, not agreed.")
    qs = [q["q"] for q in ctx.questions[-2:]]
    if qs:
        bits.append("Asked: " + " | ".join(x[:70] for x in qs))
    return " ".join(bits)


def build(prev: dict | None, session, turns: list[dict] | None = None, product: str = "", profile=None) -> dict:
    """The updated memory record after a finished call (prev = the record from before, or None)."""
    prev = dict(prev or {})
    ctx = session.context
    style = style_from_session(session, turns)
    outcome = ctx.outcome or ctx.decision
    calls = (prev.get("calls") or [])[-2:] + [{"t": time.time(), "session_id": session.session_id, "product": product,
                                               "outcome": outcome, "summary": _call_summary(session, outcome, product)}]
    objections = list(dict.fromkeys((prev.get("objections") or []) + list(ctx.objections)))[-8:]
    dnp = list(prev.get("do_not_push") or [])
    if outcome in ("declined", "opt_out") and f"declined {product}" not in dnp:
        dnp.append(f"declined {product}" if outcome == "declined" else "asked not to be called again")
    if session.dialog.get("sad_hold") == "1" and "was going through a hard time: no hard sell" not in dnp:
        dnp.append("was going through a hard time: no hard sell")
    if "price" in ctx.objections and "price-sensitive: lead with value, don't push the full price twice" not in dnp:
        dnp.append("price-sensitive: lead with value, don't push the full price twice")
    rec = {"customer_id": session.customer_id, "updated": time.time(), "calls_total": int(prev.get("calls_total", 0)) + 1,
           "preferred_language": style["last_language"] or prev.get("preferred_language", ""), "style": style,
           "topics": list(dict.fromkeys((prev.get("topics") or []) + list(getattr(profile, "topics", []) or [])))[-10:],
           "likes": list(getattr(profile, "likes", []) or prev.get("likes") or [])[-10:],
           "dislikes": list(getattr(profile, "dislikes", []) or prev.get("dislikes") or [])[-10:],
           "objections": objections, "last_deal": {k: v for k, v in (ctx.deal_view or {}).items() if k != "floor"},
           "last_outcome": outcome, "calls": calls, "do_not_push": dnp[-6:]}
    return rec


def about_block(mem: dict | None) -> str:
    """Compact block for the decision layer and the reply prompt."""
    if not mem:
        return ""
    st = mem.get("style") or {}
    lines = [f"ABOUT THIS CALLER ({mem.get('calls_total', 0)} earlier call(s)):"]
    mix = st.get("language_mix") or {}
    lines.append(f"talks in {mem.get('preferred_language') or '?'} (mix {', '.join(f'{k} {v:.0%}' for k, v in mix.items() if v)}), "
                 f"{st.get('answers', 'medium')} answers (~{st.get('avg_words_per_turn', 0)} words), formality {st.get('formality', 'med')}"
                 + (f", pace {st['pace']}" if st.get("pace") else "") + (", jokes around" if st.get("humour") else ""))
    if mem.get("topics") or mem.get("likes"):
        lines.append("cares about: " + ", ".join((mem.get("topics") or []) + (mem.get("likes") or []))[:200])
    if mem.get("objections"):
        lines.append("objections before: " + ", ".join(mem["objections"]))
    d = mem.get("last_deal") or {}
    if d.get("agreed_price"):
        lines.append(f"last time they AGREED {d.get('currency', '')}{float(d['agreed_price']):,.0f} (outcome {mem.get('last_outcome', '')})")
    elif d.get("last_agent_offer") or d.get("last_customer_counter"):
        lines.append("last time a price was discussed but NOT agreed: history only, never quote it as today's offer")
    for c in (mem.get("calls") or [])[-3:]:
        lines.append("- " + c.get("summary", "")[:220])
    if mem.get("do_not_push"):
        lines.append("do not push: " + "; ".join(mem["do_not_push"]))
    lines.append("Use this only to fit their style; mention the last call only if it helps them; never invent what they said.")
    return "\n".join(lines)


class CallerMemoryStore:
    """NATS KV `sg_customer` (via the NatsSessionStore connection) + JSON files; reads are time-boxed."""

    def __init__(self, root: str | Path, nats_store=None, read_timeout_s: float = 0.15) -> None:
        self.root = Path(root)
        self.nats, self.read_timeout_s = nats_store, read_timeout_s

    def _path(self, customer_id: str) -> Path:
        return self.root / f"{key_for(customer_id)}.json"

    async def _kv(self):
        st = self.nats
        if st is None or not getattr(st, "connected", False):
            return None
        kv = getattr(st, "_customer_kv", None)
        if kv is None:
            from nats.js.api import KeyValueConfig, StorageType
            try:
                kv = await st._js.key_value(_BUCKET)
            except Exception:
                kv = await st._js.create_key_value(KeyValueConfig(bucket=_BUCKET, history=5, storage=StorageType.FILE,
                                                                  max_value_size=256 * 1024, max_bytes=256 * 1024 ** 2))
            st._customer_kv = kv
        return kv

    async def get(self, customer_id: str) -> dict | None:
        if not customer_id:
            return None
        try:
            kv = await asyncio.wait_for(self._kv(), self.read_timeout_s)
            if kv is not None:
                e = await asyncio.wait_for(kv.get(key_for(customer_id)), self.read_timeout_s)
                if e is not None and e.value:
                    return json.loads(e.value)
        except Exception:
            pass
        p = self._path(customer_id)
        try:
            return json.loads(p.read_text()) if p.exists() else None
        except Exception:
            return None

    async def put(self, rec: dict) -> None:
        cid = rec.get("customer_id", "")
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            self._path(cid).write_text(json.dumps(rec, ensure_ascii=False, indent=1))
        except Exception as e:
            log.warning("caller memory file write failed: %s", e)
        try:
            kv = await asyncio.wait_for(self._kv(), 1.0)
            if kv is not None:
                await asyncio.wait_for(kv.put(key_for(cid), json.dumps(rec, ensure_ascii=False).encode()), 1.0)
        except Exception as e:
            log.warning("caller memory NATS write failed (file kept): %s", e)
