"""The deal: one serialisable record of where the price talk stands (contract with agent "ctx": team/INTERFACE.md).

`DealState` lives on `session.context.deal` (ctx persists and restores it with the whole context; a mirror copy is kept in
`session.dialog["_deal"]` so a process that does not have the context field yet still works). Every number the agent says about
the price comes from here; the LLM never invents one.

State machine (status):

    none --first price talk--> open --counter / concession--> countering --both say yes to one price--> agreed --link--> closed
      \\                         \\                              \\                                         |
       +-------------------------+------------------------------+--walk-away--> walked <------------------+ (cancel)
    walked --a counter at/above the floor, or "ok, that price"--> countering / agreed again

Rules enforced here (the negotiation wording layer may only pick a price this module allows):
  * an agent offer only ever goes DOWN (monotonic), one step per explicit request, and the steps shrink (only the smallest friendly
    step may repeat), at most `max_concessions` of them;
  * never below the floor, never below the customer's own best counter, never below an already agreed price;
  * once agreed, the price is fixed: a lower counter or a push is answered with the agreed price, never a concession;
  * a counter is never an acceptance of the list price; a vague line ("maybe more or less"), silence or confusion is not a request.

This module imports only the standard library and pydantic at import time (acie/context.py may import it); acie.text is imported lazily.
"""

from __future__ import annotations

import json
import math
import re
from typing import Literal

from pydantic import BaseModel, Field

_I = re.IGNORECASE
STATUSES = ("none", "open", "countering", "agreed", "closed", "walked")
DEFAULT_SCHEDULE = (0.33, 0.25, 0.17, 0.12, 0.08)   # each concession as a share of (offer - floor): shrinking


class DealOffer(BaseModel):
    turn: int = 0
    by: Literal["agent", "customer"] = "agent"
    amount: float = 0.0
    kind: Literal["offer", "counter", "accept", "reject"] = "offer"


class DealState(BaseModel):
    """Where the price talk stands. `floor` is never said to the customer."""
    currency: str = "₹"
    list_price: float = 0.0            # the normal price (e.g. ₹6,000) when the product states one, else the offer price
    offer_price: float = 0.0           # the advertised offer the call opens with (e.g. ₹2,999)
    floor: float = 0.0                 # the lowest price allowed (never spoken)
    status: str = "none"               # none | open | countering | agreed | closed | walked
    offers: list[DealOffer] = Field(default_factory=list)
    agreed_price: float = 0.0
    agreed_perk: str = ""              # a non-price sweetener that is part of the agreed deal (economics.ladder perk id)
    last_agent_offer: float = 0.0      # 0 = nothing said yet (the standing price is offer_price)
    last_customer_counter: float = 0.0
    best_customer_counter: float = 0.0  # the highest number the customer has said they would pay
    concessions_made: int = 0
    updated_turn: int = 0

    # ------------------------------------------------------------------ reading
    @property
    def active(self) -> bool:
        return self.status in ("open", "countering", "agreed", "walked")

    def standing(self) -> float:
        """The price the agent currently stands at: the agreed price, else the last offer, else the advertised offer."""
        return self.agreed_price or self.last_agent_offer or self.offer_price

    def lower_bound(self) -> float:
        """No agent offer may go below this."""
        return max(self.floor, self.best_customer_counter, self.agreed_price)

    def facts(self) -> dict:
        """The numbers the reply layer may use (never the floor)."""
        return {"status": self.status, "standing": self.standing(), "agreed_price": self.agreed_price or None,
                "last_agent_offer": self.last_agent_offer or None, "last_customer_counter": self.last_customer_counter or None,
                "offer_price": self.offer_price, "currency": self.currency}

    def summary(self) -> str:
        """One line for the LLM prompt (numbers only from the state)."""
        f = lambda n: fmt(n, self.currency)
        if self.status in ("agreed", "closed") and self.agreed_price:
            return f"Price agreed with the customer: {f(self.agreed_price)}. Never quote any other price for this plan."
        if self.status in ("open", "countering"):
            s = f"Price talk in progress: your current offer is {f(self.standing())}"
            if self.last_customer_counter:
                s += f"; the customer's last number was {f(self.last_customer_counter)}"
            return s + ". Never quote any other price and never offer a new discount yourself."
        return ""

    # ------------------------------------------------------------------ transitions
    def _log(self, turn: int, by: str, amount: float, kind: str) -> None:
        self.offers.append(DealOffer(turn=turn, by=by, amount=float(amount), kind=kind))
        self.offers = self.offers[-40:]
        self.updated_turn = turn

    def open(self, turn: int) -> None:
        if self.status == "none":
            self.status = "open"
            self.updated_turn = turn

    def customer_counter(self, amount: float, turn: int) -> None:
        self.open(turn)
        self.last_customer_counter = float(amount)
        if amount <= self.standing():  # a number above our price is not a counter we hold them to
            self.best_customer_counter = max(self.best_customer_counter, float(amount))
        if self.status in ("open", "walked"):
            self.status = "countering"
        self._log(turn, "customer", amount, "counter")

    def customer_reject(self, turn: int) -> None:
        if self.active:
            self._log(turn, "customer", self.standing(), "reject")

    def agent_offer(self, amount: float, turn: int) -> float:
        """Record a price the agent says. Clamped to the rules: never up, never below the lower bound. Returns the price recorded."""
        self.open(turn)
        amount = float(amount)
        if self.status == "agreed":
            return self.agreed_price  # an agreed price never moves
        cur = self.last_agent_offer or self.offer_price
        amount = min(amount, cur) if cur else amount
        amount = max(amount, self.lower_bound())
        if cur and amount < cur - 1e-9:
            self.concessions_made += 1
        self.last_agent_offer = amount
        if self.status in ("open", "walked"):
            self.status = "countering"
        self._log(turn, "agent", amount, "offer")
        return amount

    def agree(self, amount: float, turn: int, by: str = "customer", perk: str = "") -> float:
        """Both sides said yes to one price. Never below the floor; never above the standing offer."""
        amount = max(float(amount), self.floor)
        if self.status == "agreed" and self.agreed_price:
            amount = min(amount, self.agreed_price)
        self.open(turn)
        self.agreed_price = amount
        self.last_agent_offer = min(self.last_agent_offer or amount, amount)
        if perk:
            self.agreed_perk = perk
        self.status = "agreed"
        self._log(turn, by, amount, "accept")
        return amount

    def walk(self, turn: int) -> None:
        if self.status in ("open", "countering"):
            self.status = "walked"
            self.updated_turn = turn

    def close(self, turn: int) -> None:
        if self.status == "agreed" or self.agreed_price:
            self.status = "closed"
            self.updated_turn = turn


# ---------------------------------------------------------------------- config (per product)

class DealConfig(BaseModel):
    currency: str = "₹"
    offer_price: float = 0.0
    list_price: float = 0.0
    floor: float = 0.0
    max_concessions: int = 4
    schedule: tuple[float, ...] = DEFAULT_SCHEDULE
    sweeteners: list[str] = Field(default_factory=list)  # economics.ladder perk ids, in the order they may be offered
    joke_below_pct: float = 40.0                         # a counter under this % of the floor is a joke: humour, no movement


def _money(s: str) -> float:
    m = re.search(r"\d[\d,]*(?:\.\d+)?", s or "")
    return float(m.group(0).replace(",", "")) if m else 0.0


def config_for(p) -> DealConfig | None:
    """The deal config of a product, or None when it has no haggling (no economics, e.g. a real company's fixed price)."""
    eco = getattr(p, "economics", None)
    if eco is None or not eco.list_price or not (0 < eco.profit_floor < eco.list_price):
        return None
    neg = getattr(p, "negotiation", None)
    offer = float(eco.list_price)
    floor = float(eco.profit_floor)
    if neg is not None:
        if getattr(neg, "floor", 0):
            floor = max(floor, float(neg.floor))
        if getattr(neg, "floor_pct", 0):
            floor = max(floor, offer * float(neg.floor_pct) / 100.0)
    floor = min(floor, offer)
    normal = _money(getattr(p, "original_price", ""))
    return DealConfig(currency=eco.currency or "₹", offer_price=offer, list_price=normal if normal > offer else offer, floor=floor,
                      max_concessions=int(getattr(neg, "max_concessions", 4) or 0) if neg is not None else 4,
                      schedule=tuple(getattr(neg, "step_schedule", None) or DEFAULT_SCHEDULE) if neg is not None else DEFAULT_SCHEDULE,
                      sweeteners=list(getattr(neg, "sweeteners", None) or []) if neg is not None else [],
                      joke_below_pct=float(getattr(neg, "joke_below_pct", 40.0) or 40.0) if neg is not None else 40.0)


def new_deal(cfg: DealConfig) -> DealState:
    return DealState(currency=cfg.currency, list_price=cfg.list_price, offer_price=cfg.offer_price, floor=cfg.floor)


# ---------------------------------------------------------------------- the concession schedule

def grid(price: float) -> int:
    return 50 if price >= 1000 else 10 if price >= 200 else 5 if price >= 50 else 1


def near(x: float, g: int) -> int:
    """The friendly price (...49 / ...99 on a 50 grid) nearest to x."""
    return int(round((x + 1) / g)) * g - 1 if g > 1 else int(round(x))


def up(x: float, g: int) -> int:
    """The smallest friendly price at or above x (a rounded floor never drops under the real floor)."""
    return int(math.ceil((x + 1) / g - 1e-9)) * g - 1 if g > 1 else int(math.ceil(x - 1e-9))


def ladder(offer: float, floor: float, schedule=DEFAULT_SCHEDULE, max_steps: int | None = None) -> list[int]:
    """The concession prices, highest first. Every cut is smaller than the one before it (shrinking steps; only the smallest friendly
    step, one grid unit, may repeat), every price is
    a friendly number at or above the floor, and there are at most `max_steps` of them. Empty when there is nothing to give."""
    g = grid(offer)
    lowest = min(up(floor, g), int(math.ceil(offer)))
    if offer - lowest < g:
        return []
    room, prev, last_cut, out = offer - floor, float(offer), float("inf"), []
    for fr in schedule:
        x = near(prev - fr * room, g)
        if x >= prev:
            x = near(prev - g, g)
        x = max(x, lowest)
        cut = prev - x
        if cut >= last_cut and cut > g:           # a cut must be smaller than the last one (the smallest friendly step may repeat)
            cut = max(last_cut - g, g)
            x = int(prev - cut)
        if cut < g or x < lowest:
            break
        out.append(int(x))
        prev, last_cut = x, cut
    if max_steps is not None:
        out = out[:max(0, max_steps)]
    return out


# ---------------------------------------------------------------------- reading the customer

_ACCEPT = re.compile(r"^\W*(ok(ay)?|okk+|done|deal|done deal|sure|yes|yeah|yep|fine|agreed|alright|all right|sounds good|perfect|great|"
                     r"theek( hai)?|thik( hai)?|chalega|chalo|pakka|haan|haan ji|ha|ji|bilkul|kar do|kardo|"
                     r"ठीक( है)?|चलेगा|चलो|पक्का|हाँ|हां|जी|बिल्कुल|कर दो|डन|डील|ओके)(\W+(ok(ay)?|done|deal|sure|yes|yeah|fine|theek( hai)?|chalega|pakka|"
                     r"haan|ji|then|it is|that|thats|that's|works|for me|bhai|boss|sir|ठीक( है)?|चलेगा|पक्का|हाँ|हां|जी|है))*\W*$", _I)
_RESTATE = re.compile(r"where (were|are) we|where did we (leave|get|land)|what (was|is) (the|our|my|your) (price|deal|offer|final|number|rate)|"
                      r"what (price|number|rate) (did|were|was|are)|what did (we|you) (agree|settle|say|offer)|which price|remind me|"
                      r"(last|latest) (price|offer|number) (you|we)|(lost|missed) (you|track|it|the price)|"
                      r"kitne (pe|par|mein|me) (baat|deal)|kya (price|rate|keemat|daam) (tha|thi|hua|hui|bola|boli)|kahan (the|they) hum|"
                      r"baat kahan (tak|pe|par)|kitna (bola|kaha|tay) (tha|hua)|"
                      r"कितने (में|पे|पर) (बात|डील)|क्या (प्राइस|रेट|कीमत|दाम) (था|थी|हुआ|हुई|बोला)|कहाँ (थे|पहुँचे) हम|बात कहाँ (तक|पर|पे)|कितना (बोला|कहा|तय)", _I)
_VAGUE = re.compile(r"more or less|kam (ya|ki) (zyada|jyada)|kam zyada|kam jyada|idhar udhar|upar neeche|whatever you (think|want|like)|"
                    r"(up|upto) to you|aap dekh lo|aap dekh lijiye|jo (aapko )?(theek|sahi) lage|कम ज़्यादा|कम ज्यादा|इधर उधर|जो (आपको )?ठीक लगे", _I)
_WALK = re.compile(r"leave it|forget it|not (interested|worth it)|i.?ll pass|no deal|i.?ll (go|look) elsewhere|rehne do|chhodo|chodo|jane do|"
                   r"nahi (chahiye|lena)|रहने दो|छोड़ो|जाने दो|नहीं (चाहिए|लेना)", _I)
_REJECT = re.compile(r"^\W*(no|nope|nah|nahi|nahin|na|नहीं|ना)\b(\W+(no|nahi|that.?s|thats|still|too|it.?s|बहुत|zyada|jyada|much|high|expensive|costly|mehenga))*", _I)
_PRICE_CUE = re.compile(r"₹|\brs\b|rupe|रुपए|रुपये|\bk\b|how about|what about|make it|for\s+\d|at\s+\d|around|about\s+\d|"
                        r"(mein|me|में|pe|par|पे|पर|tak|तक)\s*(do|de|dedo|दो|दे|chalega|चलेगा|done|kar|कर|final|ho|deal)?\b|chalega|चलेगा|kar ?do|कर दो|"
                        r"de ?do|दे दो|deal|done|final|budget|bajet|बजट|offer|price|rate|discount|allowed to|can you do|could you do|give (it|me)", _I)
_NOT_PRICE = re.compile(r"\b(?:days?|din|months?|mahin[ae]|weeks?|hafte|sessions?|minutes?|mins?|hours?|baje|bje|o.?clock|am|pm|"
                        r"years?|saal|percent|%|kg|kilo|times|baar|branch|number|otp|pin|sector|plot|block|flat|house|room|gali|street|road|phase|floor|"
                        r"km|kilometer|minute|age|years old|saal ka|batch)\b", _I)


# a number attached to a unit is a distance / time / count, never money ("21 km marathon", "hafte mein 4 din", "5 baje", "3x10 reps")
_UNIT_AFTER = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*-?\s*(?:km|kms|kilomet\w*|k run|k marathon|marathon|miles?|meters?|metres?|mtr|kg|kgs|kilo\w*|lbs?|pounds|"
                         r"days?|din|dino|weeks?|hafte|hafton|months?|mahin\w*|years?|saal|sal|times|baar|bar\b|sessions?|reps?|sets?|rounds?|"
                         r"am|pm|a\.m\.|p\.m\.|o.?clock|baje|bje|bajey|%|percent|minutes?|mins?|hours?|hrs?|ghant\w*|"
                         r"दिन|हफ़्ते|हफ्ते|महीने|साल|बजे|बार|किलो|किलोमीटर|मिनट|घंटे)(?![a-z])", _I)


def amounts(text: str) -> list[int]:
    """Every price-like amount in the line (digits, ₹, 2.8k, English / Hindi / Hinglish number words, STT noise around them),
    with durations, times, phone numbers and OTPs left out."""
    from ..text import find_amounts
    try:
        found = find_amounts(text)
    except Exception:  # a parse failure never breaks a turn
        found = []
    if not found:
        found = [int(x) for x in re.findall(r"\d+", (text or "").replace(",", "")) if not _NOT_PRICE.search(text or "")]
    units = set()
    for m in _UNIT_AFTER.finditer(text or ""):
        try:
            units.add(int(round(float(m.group(1).replace(",", "")))))
        except ValueError:
            pass
    return [x for x in found if 0 < x < 10_000_000 and x not in units]


def amount(text: str, deal: DealState | None = None) -> int | None:
    """The amount the customer names: the last price-like number ("not 2,999, I want 2,500" -> 2500). When several are said and a deal
    is known, the one that is not the price already on the table wins."""
    xs = amounts(text)
    if not xs:
        return None
    big = [x for x in xs if x >= 50] or xs
    if deal is not None and len(big) > 1:
        table = {int(round(deal.standing())), int(round(deal.offer_price)), int(round(deal.list_price))}
        other = [x for x in big if x not in table]
        if other:
            return other[-1]
    return big[-1]


def plausible(n: float, deal: DealState) -> bool:
    """A number that can be a price for this deal (not "3 months", not a phone number): up to 2x the normal price."""
    top = max(deal.list_price, deal.offer_price) * 2 or float("inf")
    return 0 < n <= top


def is_joke(n: float, cfg_or_pct, deal: DealState) -> bool:
    pct = cfg_or_pct.joke_below_pct if hasattr(cfg_or_pct, "joke_below_pct") else float(cfg_or_pct)
    return deal.floor > 0 and n < deal.floor * pct / 100.0


_ADDR_BEFORE = re.compile(r"(sector|plot|block|flat|house|room|gali|street|road|phase|floor|no\.?|number|batch|pin|otp|platform|gate|bus|route|"
                          r"tower|wing|unit|apartment|सेक्टर|प्लॉट|नंबर|ब्लॉक|मकान|गली)\W{0,3}$", _I)


def _addressish(text: str, n: int) -> bool:
    """The number is part of an address / identifier ("Sector 49", "flat 302", "PIN 1234"), not a price."""
    t = text or ""
    for m in re.finditer(r"\d[\d,]*", t):
        if m.group(0).replace(",", "") == str(n) and _ADDR_BEFORE.search(t[max(0, m.start() - 14):m.start()]):
            return True
    return False


def price_cue(text: str) -> bool:
    """The number is said like a price ("₹2,800", "2800 mein do", "how about 2,800", "make it 100")."""
    return bool(_PRICE_CUE.search(text or ""))


def wants_restate(text: str) -> bool:
    return bool(_RESTATE.search(text or ""))


def is_vague(text: str) -> bool:
    return bool(_VAGUE.search(text or ""))


def is_accept(text: str) -> bool:
    return bool(_ACCEPT.match(text or ""))


def is_walk(text: str) -> bool:
    return bool(_WALK.search(text or ""))


def is_reject(text: str) -> bool:
    return bool(_REJECT.match(text or "")) and not is_accept(text)


def classify(text: str, deal: DealState, cfg: DealConfig | None = None) -> tuple[str, int | None]:
    """(move, amount) for one customer line during a deal. Moves: restate | joke | counter | accept | reject | walk | vague | none.
    `ask_more` (a push for a lower price with no number) is decided by the negotiation detector, not here."""
    if wants_restate(text):
        return "restate", None
    n = amount(text, deal)
    if n is not None and plausible(n, deal) and not _addressish(text, n):
        if abs(n - deal.standing()) < 0.5:
            return ("accept" if not is_reject(text) else "reject"), n
        if cfg is not None and is_joke(n, cfg, deal):
            return "joke", n
        return "counter", n
    if is_walk(text):
        return "walk", None
    if is_vague(text):
        return "vague", None
    if is_accept(text):
        return "accept", None
    if is_reject(text):
        return "reject", None
    return "none", None


# ---------------------------------------------------------------------- formatting and storage

def fmt(n: float, currency: str = "₹") -> str:
    return f"{currency}{int(round(n)):,}"


def load(ctx, dialog: dict) -> DealState:
    """The deal from session.context.deal (when ctx has the field), else the dialog mirror, else a blank deal."""
    d = getattr(ctx, "deal", None) if ctx is not None else None
    if isinstance(d, DealState) and (d.status != "none" or d.offer_price or not dialog.get("_deal")):
        return d.model_copy(deep=True)
    if isinstance(d, dict) and d:
        try:
            return DealState.model_validate(d)
        except Exception:
            pass
    raw = dialog.get("_deal") if dialog is not None else None
    if raw:
        try:
            return DealState.model_validate(json.loads(raw))
        except Exception:
            pass
    return DealState()


def save(deal: DealState, ctx, dialog: dict) -> None:
    """Write the deal to session.context.deal (if the context has that field) and to the dialog mirror."""
    if ctx is not None and "deal" in getattr(type(ctx), "model_fields", {}):
        try:
            setattr(ctx, "deal", deal.model_copy(deep=True))
        except Exception:
            pass
    if dialog is not None:
        dialog["_deal"] = deal.model_dump_json()
