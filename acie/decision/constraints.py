"""Declarative constraints, evaluated before a plan is returned.

A product's `constraints` is a list of plain dicts (editable from the console, default empty):

  {"type": "min_age", "value": 18}
  {"type": "region_allow", "regions": ["delhi", "mumbai"]}       customer region must be in the list
  {"type": "exclude_existing_subscriber"}
  {"type": "price_floor", "min": 5}                              offer price (first number) must be >= min
  {"type": "max_reoffers", "value": 1}                           (also capped by Product.max_reoffers)
  {"type": "call_hours", "start": 9, "end": 21, "utc_offset_minutes": 330}   local hour in [start, end)
  {"type": "dnc", "numbers": ["+91..."]}                         plus an optional live lookup callable

Unknown facts never violate (no age on file = rule does not apply), except `dnc`/`call_hours`
which only need the number / the clock. A violated constraint yields a compliant TurnPlan: a
reviewed template, llm_allowed=False, call ended.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable

from ..schemas import TurnPlan

PRE = ("min_age", "region_allow", "exclude_existing_subscriber", "price_floor", "call_hours", "dnc")


@dataclass
class ConstraintContext:
    product: object
    phone: str = ""
    age: int | None = None
    region: str = ""
    existing_subscriber: bool | None = None
    reoffers: int = 0
    now: datetime | None = None
    dnc_lookup: Callable[[str], bool] | None = None


@dataclass
class Violation:
    type: str
    reason: str


def _price_number(s: str) -> float | None:
    m = re.search(r"\d[\d,]*(?:\.\d+)?", s or "")
    return float(m.group(0).replace(",", "")) if m else None


def _digits(n: str) -> str:
    return re.sub(r"\D", "", n or "")[-10:]


def check_one(c: dict, ctx: ConstraintContext) -> Violation | None:
    t = c.get("type", "")
    if t == "min_age":
        if ctx.age is not None and ctx.age < int(c.get("value", 18)):
            return Violation(t, f"age {ctx.age} below {c.get('value', 18)}")
    elif t == "region_allow":
        allow = {r.strip().lower() for r in c.get("regions", [])}
        if allow and ctx.region and ctx.region.strip().lower() not in allow:
            return Violation(t, f"region {ctx.region} not allowed")
    elif t == "exclude_existing_subscriber":
        if ctx.existing_subscriber:
            return Violation(t, "already a subscriber")
    elif t == "price_floor":
        price = _price_number(getattr(ctx.product, "offer_price", ""))
        if price is not None and price < float(c.get("min", 0)):
            return Violation(t, f"offer price {price} below floor {c.get('min')}")
    elif t == "max_reoffers":
        if ctx.reoffers > int(c.get("value", 1)):
            return Violation(t, "re-offer limit reached")
    elif t == "call_hours":
        now = ctx.now or datetime.now(timezone.utc)
        utc = now.replace(tzinfo=timezone.utc) if now.tzinfo is None else now.astimezone(timezone.utc)
        local = utc + timedelta(minutes=int(c.get("utc_offset_minutes", 330)))
        start, end = float(c.get("start", 9)), float(c.get("end", 21))
        h = local.hour + local.minute / 60
        if not (start <= h < end):
            return Violation(t, f"outside call hours {start:g}-{end:g}")
    elif t == "dnc":
        me = _digits(ctx.phone)
        if me and (me in {_digits(n) for n in c.get("numbers", [])} or (ctx.dnc_lookup and ctx.dnc_lookup(ctx.phone))):
            return Violation(t, "number on do-not-call list")
    return None


def evaluate(constraints: list[dict], ctx: ConstraintContext, types: tuple[str, ...] | None = None) -> Violation | None:
    for c in constraints or []:
        if types is not None and c.get("type") not in types:
            continue
        v = check_one(c, ctx)
        if v is not None:
            return v
    return None


_TPL = {
    "dnc": ("I'm sorry for the disturbance. Your number is on our do-not-call list and will not be contacted. Goodbye.",
            "परेशानी के लिए माफ़ी। आपका नंबर हमारी do-not-call लिस्ट में है, अब कॉल नहीं आएगी। नमस्ते।",
            "Pareshani ke liye maafi. Aapka number hamari do-not-call list mein hai, ab call nahi aayegi. Namaste."),
    "call_hours": ("Sorry, I've called at a bad time. I'll try again during the day. Goodbye.",
                   "माफ़ कीजिए, मैंने गलत समय पर कॉल कर दिया। मैं दिन में दोबारा कोशिश करूँगा। नमस्ते।",
                   "Sorry, galat time pe call ho gaya. Main din mein dobara try karunga. Namaste."),
}
_DEFAULT = ("Sorry to disturb you. This offer is not applicable for you, so I won't take more of your time. Have a good day.",
            "परेशानी के लिए माफ़ी। ये ऑफर आपके लिए लागू नहीं है, इसलिए मैं आपका और समय नहीं लूँगा। आपका दिन शुभ हो।",
            "Pareshani ke liye maafi. Ye offer aapke liye applicable nahi hai, isliye main aapka aur time nahi lunga. Aapka din accha ho.")


def violation_plan(v: Violation) -> TurnPlan:
    en, hi, hing = _TPL.get(v.type, _DEFAULT)
    return TurnPlan(intent="constraint_stop", llm_allowed=False, end_call=True,
                    decision=f"blocked by constraint: {v.type}",
                    objective="Politely end the call; the offer cannot be made to this customer",
                    fallback={"en": en, "hi": hi, "hinglish": hing})
