"""What the last call(s) with this customer left behind, for the follow-up call: read from data/voice/calls/*.json.

Everything here is a pure function of the saved call record (the call's context ledger, its turns and, when a call-type
pack ran, its `result`). No model, no randomness; the clock is passed in.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .extract import DEFAULT_TZ, date_time

# plan intents whose answer to a customer question was NOT a real answer: the question is still open
_OPEN_ANSWERS = {"clarify", "unknown", "callback", "human_agent", "goodbye", "annoyed_exit", "declined", "recover", "garbled"}
_PRICE_INTENTS = {"link_sent", "confirm_close", "confirm_buy", "price_floor", "reoffer", "price", "neg_close", "negotiate"}
_MONEY = re.compile(r"(₹|rs\.?|inr|\$)\s?\d[\d,]*(?:\.\d+)?|\d[\d,]*(?:\.\d+)?\s?(rupees|rupaye|रुपये|रुपए)", re.I)
_PLACEHOLDER_IDS = {"", "c1"}  # CallRecord's default customer_id: not a real customer


@dataclass
class PreviousCall:
    call_id: str = ""
    created_at: float = 0.0
    call_type: str = ""
    campaign: str = ""
    outcome: str = ""
    decision: str = ""
    told: list[str] = field(default_factory=list)
    quoted: list[str] = field(default_factory=list)
    promised: list[str] = field(default_factory=list)
    objections: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    agreed_price: str = ""
    callback_text: str = ""
    callback_at: str = ""  # ISO, when the caller named a time and we could read it
    link_sent: bool = False
    customer_name: str = ""

    def has_content(self) -> bool:
        return bool(self.callback_text or self.callback_at or self.link_sent or self.agreed_price or self.objections or self.open_questions
                    or self.outcome or self.decision not in ("", "undecided"))

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def _is_open(q: dict) -> bool:
    return str(q.get("answer", "")) in _OPEN_ANSWERS


def from_record(rec: dict, tz: str = DEFAULT_TZ) -> PreviousCall:
    ctx = rec.get("context") or {}
    res = ctx.get("result") or rec.get("result") or {}
    turns = rec.get("turns") or []
    p = PreviousCall(call_id=str(rec.get("call_id", "")), created_at=float(rec.get("created_at") or 0), call_type=str(rec.get("call_type") or ctx.get("call_type") or ""),
                     campaign=str(rec.get("campaign", "")), outcome=str(ctx.get("outcome") or res.get("outcome") or ""), decision=str(ctx.get("decision") or ""),
                     told=list(ctx.get("told") or []), quoted=list(ctx.get("quoted") or []), promised=list(ctx.get("promised") or []),
                     objections=list(ctx.get("objections") or []), customer_name=str(rec.get("customer_name", "")))
    p.link_sent = "payment_link" in p.promised or str((res.get("facts") or {}).get("link_sent", "")) in ("1", "true", "yes")
    # open questions: unanswered ones from the ledger, plus any a call-type pack left behind
    qs = [str(q.get("q", "")).strip() for q in (ctx.get("questions") or []) if isinstance(q, dict) and _is_open(q)]
    qs += [x for x in str((res.get("facts") or {}).get("open_questions", "")).split("|") if x.strip()]
    p.open_questions = list(dict.fromkeys(q for q in qs if q))[:3]
    # the callback they asked for: the customer's words just before our callback turn
    for i, t in enumerate(turns):
        if t.get("role") == "agent" and t.get("intent") == "callback" and i > 0:
            prev = next((x for x in reversed(turns[:i]) if x.get("role") == "customer"), None)
            if prev and prev.get("text"):
                p.callback_text = str(prev["text"])[:80]
                base = datetime.fromtimestamp(p.created_at or 0, ZoneInfo(tz)) if p.created_at else None
                ok, iso = date_time(str(prev["text"]), base, tz)
                if ok:
                    p.callback_at = iso or ""
    nxt = (res.get("next_action") or {})
    if nxt.get("due_at"):
        p.callback_at = p.callback_at or str(nxt["due_at"])
    fact_price = str((res.get("facts") or {}).get("agreed_price", ""))
    if fact_price:
        p.agreed_price = fact_price
    else:
        for t in reversed(turns):
            if t.get("role") == "agent" and t.get("intent") in _PRICE_INTENTS:
                found = _MONEY.search(str(t.get("text", "")))
                if found:
                    p.agreed_price = found.group(0).strip()
                    break
    return p


def load_previous_calls(data_dir: Path | str, customer_id: str = "", phone: str = "", exclude_call_id: str = "", limit: int = 3,
                        scan: int = 300, tz: str = DEFAULT_TZ) -> list[PreviousCall]:
    """The customer's earlier calls, newest first. Matched by a real customer_id, else by phone number."""
    cdir = Path(data_dir) / "calls"
    if not cdir.is_dir():
        return []
    cid = "" if customer_id in _PLACEHOLDER_IDS else customer_id
    if not cid and not phone:
        return []
    files = sorted(cdir.glob("*.json"), key=lambda p: (p.stat().st_mtime, p.name), reverse=True)[:scan]
    out: list[PreviousCall] = []
    for f in files:
        try:
            rec = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        if not isinstance(rec, dict) or str(rec.get("call_id", f.stem)) == exclude_call_id:
            continue
        same = (cid and rec.get("customer_id") == cid) or (phone and rec.get("to") == phone)
        if not same or not rec.get("turns"):
            continue
        out.append(from_record(rec | {"call_id": rec.get("call_id", f.stem)}, tz))
        if len(out) >= limit:
            break
    return sorted(out, key=lambda c: c.created_at, reverse=True)
