"""Call context ledger: what has happened in this call, kept for the whole call.

Live call 912c391917: the agent said it was sending the payment link, and a few turns later
asked "shall I send you the link?" again. The LLM only sees the last few utterances and the FSM
only knows its current state, so anything older was forgotten.

The ledger is an append-only event log per call (the shape of a Kafka/NATS topic, one per
call) plus a small state folded from it:

  promised    what the agent committed to (payment link, callback, number removed, team call)
  quoted      offer facts already told to the customer (price, discount)
  questions   what the customer asked, and how it was answered
  objections  concerns raised (fake? too costly? language?)
  decision    the customer's current answer to the offer: accepted / declined / undecided

It lives in-process on the SessionState (one call = one process; sub-millisecond, no broker
to run). Every event is also handed to `sinks` (the voice server streams them to the console
and stores them on the call record); a NATS or Kafka publisher can be added as another sink.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any, Callable

from pydantic import BaseModel, Field, field_validator


_QUESTION = re.compile(r"\?|\b(what|how|why|when|where|which|who|kya|kaise|kab|kyun|kyon|kaun|kahan|kitna|kitne)\b|"
                       # "do/is/can" only open a question at the start ("okay fine, let's do it" is not one)
                       r"^\W*(can|could|does|do|is|are|will|would|should)\b|क्या|कैसे|कब|क्यों|कौन|कहां|कहाँ|कितना|कितने|कितनी",
                       re.IGNORECASE)


_PAY_BYE = re.compile(r"\b(will pay|i.?ll pay|paying|pay (now|kar|kardunga|kar dunga|karunga)|thanks?|thank you|bye|ok(ay)? fine|done|"
                      r"kar dunga|kar doonga|bhar dunga|dhanyavad|shukriya)\b|धन्यवाद|शुक्रिया|बाय|पे कर", re.I)
_CANCEL_LINK = re.compile(r"\b(don.?t (send|want)|cancel|mat bhej|nahi chahiye|not interested|changed my mind)\b|मत भेज|नहीं चाहिए", re.I)


def is_question(text: str) -> bool:
    return bool(text) and bool(_QUESTION.search(text))


class ContextEvent(BaseModel):
    turn: int
    who: str  # agent | customer | system
    kind: str
    detail: str = ""
    t: float = Field(default_factory=time.time)


# agent plan intent -> (event kind, what it commits to / tells the customer)
_AGENT_EVENTS: dict[str, tuple[str, str]] = {
    "link_sent": ("promised", "payment_link"),
    "callback": ("promised", "callback_later"),
    "human_agent": ("promised", "team_member_will_call"),
    "opt_out": ("promised", "number_removed_no_more_calls"),
    "price": ("quoted", "offer_price"),
    "reoffer": ("quoted", "offer_price"),
    "recover": ("quoted", "offer_price"),
    "price_floor": ("quoted", "no_lower_price"),
    "confirm_close": ("quoted", "offer_price"),
    "pitch": ("told", "what_the_product_is"),
    "are_you_ai": ("told", "agent_is_an_ai"),
    "identify": ("told", "who_is_calling"),
    "opening": ("told", "who_is_calling"),
    "safety_distress": ("promised", "helpline_given_call_stopped"),
    "constraint_stop": ("promised", "call_ended_by_policy"),
}
_DECISIONS = {"link_sent": "accepted", "declined": "declined", "declined_price": "declined",
              "annoyed_exit": "declined", "opt_out": "declined", "opt_out_confirmed": "declined",
              "callback": "later"}
# terminal paths -> the call's outcome label (what the feedback lane reads next to `decision`):
# accepted | declined | later | opt_out | safety | human_agent | blocked ("" = still open)
_OUTCOMES = {"link_sent": "accepted", "declined": "declined", "declined_price": "declined", "annoyed_exit": "declined",
             "callback": "later", "opt_out": "opt_out", "opt_out_confirmed": "opt_out", "safety_distress": "safety",
             "safety_followup": "safety", "safety_goodbye": "safety", "safety_noted": "safety",
             "human_agent": "human_agent", "constraint_stop": "blocked"}
# the customer takes the answer back / reconsiders: the open question is open again
_REOPEN = {"link_cancelled", "confirm_close"}


class SaleChecklist(BaseModel):
    """Where the sale stands: what has been said and done so far (updated after every turn, persisted with the session)."""
    product_explained: bool = False
    price_quoted: bool = False
    offer_terms_read: bool = False          # the full offer was read back (confirm_close)
    objections_raised: list[str] = Field(default_factory=list)
    objections_handled: list[str] = Field(default_factory=list)
    counter_offers: list[dict] = Field(default_factory=list)  # {"turn", "by": agent|customer, "amount"}
    agreed_price: float | None = None
    consent: bool = False                   # the customer said yes to buying / to the link
    link_sent: bool = False
    language: str = ""
    updated_turn: int = -1

    def lines(self) -> list[str]:
        done = [k for k, v in (("product explained", self.product_explained), ("price quoted", self.price_quoted),
                               ("offer terms read back", self.offer_terms_read), ("customer consented", self.consent),
                               ("payment link sent", self.link_sent)) if v]
        out = ["done: " + (", ".join(done) if done else "nothing yet")]
        open_obj = [o for o in self.objections_raised if o not in self.objections_handled]
        if self.objections_raised:
            out.append("objections: " + ", ".join(self.objections_raised) + (f" (still open: {', '.join(open_obj)})" if open_obj else " (handled)"))
        if self.agreed_price is not None:
            out.append(f"agreed price: {self.agreed_price:g}")
        return out


# agent lines that are not "the last thing I told you" (call.py fillers, our own replays)
_FILLER_INTENTS = {"no_response_check", "no_response_hangup", "repeat_request", "repeat_last", "where_were_we", "garbled"}
_AMOUNT = re.compile(r"(?:₹|rs\.?|inr|rupees?|\$)?\s*(\d{1,3}(?:,\d{2,3})+|\d{2,6})(?:\.\d+)?", re.I)


_REFUSE = re.compile(r"too low|below what|can.?t (do|go)|cannot|not possible|bahut kam|बहुत कम|नहीं हो पाएगा|nahi ho payega", re.I)


def amounts(text: str) -> list[float]:
    out = []
    for m in _AMOUNT.finditer(text or ""):
        try:
            out.append(float(m.group(1).replace(",", "")))
        except ValueError:
            pass
    return out


def _deal_get(deal, key: str, default=None):
    if deal is None:
        return default
    if isinstance(deal, dict):
        return deal.get(key, default)
    return getattr(deal, key, default)


def _coerce_deal(v):
    """`deal` is the negotiation agent's DealState (acie/decision/deal.py or negotiation.py). A restored snapshot carries
    it as a dict: turn it back into the model when that model exists, else keep the dict (both read the same way)."""
    if isinstance(v, dict):
        for mod in ("acie.decision.deal", "acie.decision.negotiation"):
            try:
                cls = getattr(__import__(mod, fromlist=["DealState"]), "DealState", None)
            except Exception:
                cls = None
            if cls is not None:
                try:
                    return cls.model_validate(v)
                except Exception:
                    return v
    return v


class CallContext(BaseModel):
    events: list[ContextEvent] = Field(default_factory=list)
    promised: list[str] = Field(default_factory=list)
    quoted: list[str] = Field(default_factory=list)
    told: list[str] = Field(default_factory=list)
    questions: list[dict] = Field(default_factory=list)  # {"q": text, "answer": intent}
    objections: list[str] = Field(default_factory=list)
    decision: str = "undecided"
    outcome: str = ""  # see _OUTCOMES
    call_type: str = ""  # which call type ran (acie/calltypes); "" = the plain sales call
    result: dict = Field(default_factory=dict)  # the call-type pack's CallResult (acie/calltypes/schema.py), rebuilt every turn
    # ---- where the conversation is (agent ctx): persisted every turn to NATS (acie/store/nats_session.py), restored on loss
    checklist: SaleChecklist = Field(default_factory=SaleChecklist)
    deal: Any = None  # negotiation state, owned by agent "haggle" (DealState); ctx only stores and restores it
    deal_view: dict = Field(default_factory=dict)  # plain numbers for the summary/console: list, last offers, agreed, status
    last_agent_line: str = ""  # the last real thing the agent said (fillers and replays excluded): "repeat that" replays it
    last_agent_intent: str = ""
    last_customer_line: str = ""
    caller: dict = Field(default_factory=dict)  # long-term caller memory loaded at call start (acie/caller_memory.py)

    @field_validator("deal", mode="after")
    @classmethod
    def _deal_model(cls, v):
        return _coerce_deal(v)

    def has(self, promise: str) -> bool:
        return promise in self.promised

    def add(self, ev: ContextEvent, sinks: list[Callable[[ContextEvent], None]] | None = None) -> None:
        self.events.append(ev)
        for s in sinks or ():
            try:
                s(ev)
            except Exception:  # a sink must never break the call
                pass

    def record_turn(self, turn: int, customer_text: str, intent: str, is_question: bool,
                    objection: str | None = None, sinks: list | None = None) -> list[ContextEvent]:
        """Fold one finished turn (customer said X, agent answered with plan `intent`) into the ledger."""
        new: list[ContextEvent] = []

        def add(who: str, kind: str, detail: str) -> None:
            ev = ContextEvent(turn=turn, who=who, kind=kind, detail=detail)
            self.add(ev, sinks)
            new.append(ev)

        if intent == "declined" and "payment_link" in self.promised and _PAY_BYE.search(customer_text or "") \
                and not _CANCEL_LINK.search(customer_text or ""):
            # gate run 7: after the link, "Okay fine, will pay now. Thanks, bye" was classified declined and the sale was
            # recorded as lost (link removed, CRM stage lost). A thank-you/goodbye/"will pay" after the link is a goodbye.
            intent = "goodbye"
        if customer_text and is_question:
            self.questions.append({"q": customer_text[:160], "answer": intent})
            add("customer", "asked", customer_text[:160])
        if objection and objection not in self.objections:
            self.objections.append(objection)
            add("customer", "objection", objection)
        if intent in _AGENT_EVENTS:
            kind, what = _AGENT_EVENTS[intent]
            bucket = getattr(self, kind)
            if what not in bucket:
                bucket.append(what)
                add("agent", kind, what)
        if intent in _DECISIONS and self.decision != _DECISIONS[intent]:
            self.decision = _DECISIONS[intent]
            add("customer", "decision", self.decision)
        if intent in _OUTCOMES:
            if self.outcome != _OUTCOMES[intent]:
                self.outcome = _OUTCOMES[intent]
                add("system", "outcome", self.outcome)
        elif intent in _REOPEN and self.decision == "undecided" and self.outcome in ("accepted", "declined", "later"):
            self.outcome = ""
        if intent == "declined" and "payment_link" in self.promised:  # cancelled after agreeing
            self.promised.remove("payment_link")
        return new

    def summary(self) -> str:
        """Plain lines for the LLM prompt: facts that stay true for the rest of the call."""
        lines = []
        if self.decision != "undecided":
            lines.append(f"customer's answer to the offer: {self.decision}")
        if self.promised:
            lines.append("you already promised (don't offer or ask again): " + ", ".join(self.promised))
        if self.quoted:
            lines.append("already told the customer: " + ", ".join(self.quoted + self.told))
        elif self.told:
            lines.append("already told the customer: " + ", ".join(self.told))
        if self.objections:
            lines.append("concerns the customer raised: " + ", ".join(self.objections))
        if self.questions:
            lines.append("customer's earlier questions: " + " | ".join(q["q"] for q in self.questions[-5:]))
        where = self.where_we_are(include_last=False)
        if where:
            lines.append(where)
        if self.caller:
            from .caller_memory import about_block
            lines.append(about_block(self.caller))
        return "\n".join(lines)

    # ------------------------------------------------------------------ where we are (checklist + deal + last line)

    def _refresh_deal_view(self, dialog: dict | None, list_price: float = 0.0, currency: str = "") -> None:
        """Plain numbers from haggle's DealState when there is one, else from the ladder's NegState (dialog["_neg"])
        plus the amounts heard this call. Never contains the floor."""
        v = dict(self.deal_view)
        d = self.deal
        if d is not None:
            for k in ("list_price", "offer_price", "currency", "status", "agreed_price", "last_agent_offer", "last_customer_counter"):
                val = _deal_get(d, k)
                if val not in (None, "", 0, 0.0):
                    v[k] = val
        else:
            try:
                neg = json.loads((dialog or {}).get("_neg") or "{}")
            except (ValueError, TypeError):
                neg = {}
            if neg.get("price"):
                v["last_agent_offer"] = neg["price"]
            if neg.get("agreed"):
                v["agreed_price"] = neg["agreed"]
            if neg.get("closed"):
                v["status"] = "agreed"
            elif neg.get("rounds") or neg.get("pushes"):
                v.setdefault("status", "open")
        if list_price and not v.get("list_price"):
            v["list_price"] = list_price
        if currency and not v.get("currency"):
            v["currency"] = currency
        cust = [c for c in self.checklist.counter_offers if c.get("by") == "customer"]
        if cust and not v.get("last_customer_counter"):
            v["last_customer_counter"] = cust[-1]["amount"]
        elif cust and d is None:
            v["last_customer_counter"] = cust[-1]["amount"]
        agent = [c for c in self.checklist.counter_offers if c.get("by") == "agent"]
        if agent and (d is None or not v.get("last_agent_offer")):
            v["last_agent_offer"] = agent[-1]["amount"]
        if v.get("agreed_price") and v.get("status") in (None, "", "none", "open"):
            v["status"] = "agreed"
        if (v.get("last_customer_counter") or v.get("last_agent_offer")) and not v.get("status"):
            v["status"] = "open"
        v.pop("floor", None)
        self.deal_view = v

    def after_reply(self, turn: int, customer_text: str, intent: str, agent_text: str, language: str = "",
                    dialog: dict | None = None, list_price: float = 0.0, currency: str = "") -> dict:
        """Fold the finished turn (with the words actually spoken) into the checklist; -> the checklist fields that changed."""
        before = self.checklist.model_dump()
        c = self.checklist
        if customer_text:
            self.last_customer_line = customer_text[:300]
        if agent_text and intent not in _FILLER_INTENTS:
            self.last_agent_line, self.last_agent_intent = agent_text[:600], intent
        c.product_explained = c.product_explained or "what_the_product_is" in self.told or intent == "pitch"
        c.price_quoted = c.price_quoted or "offer_price" in self.quoted
        c.offer_terms_read = c.offer_terms_read or intent == "confirm_close"
        for o in self.objections:
            if o not in c.objections_raised:
                c.objections_raised.append(o)
        # an objection is handled once the agent answered it with something other than a clarify/filler
        if intent not in _FILLER_INTENTS | {"clarify", "goodbye", "declined", "declined_price", "annoyed_exit"}:
            for o in c.objections_raised:
                if o not in c.objections_handled and any(e.kind == "objection" and e.detail == o and e.turn <= turn for e in self.events):
                    c.objections_handled.append(o)
        negotiating = intent in ("negotiation", "price_floor") or bool((dialog or {}).get("_neg"))
        hi = list_price * 1.5 if list_price else 10 ** 7
        if negotiating and intent not in _FILLER_INTENTS:
            for a in amounts(customer_text)[-1:]:
                if 1 <= a <= hi and not (c.counter_offers and c.counter_offers[-1] == {"turn": turn, "by": "customer", "amount": a}):
                    c.counter_offers.append({"turn": turn, "by": "customer", "amount": a})
            lo = list_price * 0.3 if list_price else 50
            offers = [a for a in amounts(agent_text) if lo <= a <= hi]
            said = {c_["amount"] for c_ in c.counter_offers if c_["by"] == "customer" and c_["turn"] == turn}
            if said and _REFUSE.search(agent_text or ""):  # "₹2,700 is too low": their number echoed, not our offer
                offers = [a for a in offers if a not in said]
            if offers and intent in ("negotiation", "price_floor"):
                c.counter_offers.append({"turn": turn, "by": "agent", "amount": offers[-1]})
            c.counter_offers = c.counter_offers[-30:]
        self._refresh_deal_view(dialog, list_price, currency)
        if self.deal_view.get("agreed_price"):
            try:
                c.agreed_price = float(self.deal_view["agreed_price"])
            except (TypeError, ValueError):
                pass
        c.consent = c.consent or self.decision == "accepted" or intent == "link_sent"
        c.link_sent = "payment_link" in self.promised
        if language:
            c.language = language
        c.updated_turn = turn
        after = c.model_dump()
        return {k: v for k, v in after.items() if before.get(k) != v and k != "updated_turn"}

    def where_we_are(self, include_last: bool = True) -> str:
        """One compact block for the decision layer and the reply LLM: checklist + deal + the last thing said."""
        c = self.checklist
        if c.updated_turn < 0 and not self.last_agent_line:
            return ""
        lines = ["WHERE WE ARE: " + "; ".join(c.lines())]
        dv = self.deal_view
        if dv.get("status") and dv.get("status") != "none":
            cur = dv.get("currency") or ""
            bits = [f"status {dv['status']}"]
            for k, label in (("list_price", "list"), ("last_customer_counter", "customer's last counter"),
                             ("last_agent_offer", "your last offer"), ("agreed_price", "AGREED")):
                if dv.get(k):
                    bits.append(f"{label} {cur}{float(dv[k]):,.0f}")
            lines.append("deal: " + ", ".join(bits) + (" (never offer more than the agreed price, never re-open it)" if dv.get("agreed_price") else ""))
        if include_last and self.last_agent_line:
            lines.append(f"last thing you said: \"{self.last_agent_line}\"")
        return "\n".join(lines)

    def public_view(self) -> dict:
        """For GET /api/calls/{id}/context: no floor, no raw ledger."""
        deal = self.deal.model_dump() if hasattr(self.deal, "model_dump") else (dict(self.deal) if isinstance(self.deal, dict) else None)
        if isinstance(deal, dict):
            deal.pop("floor", None)
        return {"checklist": self.checklist.model_dump(), "deal": deal, "deal_view": self.deal_view,
                "summary": self.where_we_are(), "last_agent_line": self.last_agent_line,
                "last_customer_line": self.last_customer_line, "decision": self.decision, "outcome": self.outcome,
                "promised": self.promised, "objections": self.objections, "caller": self.caller or None}


# ---------------------------------------------------------------------- "say that again" / "where were we?"

_REPEAT = re.compile(r"\brepeat\b|say (that|it|this)? ?(once more|again)|\bsay again\b|come again|what (did|was it) you (just )?(say|said|told)|"
                     r"what was (that|the last thing)|(tell|say) (me )?(the )?last thing|didn.?t (get|catch|hear) (that|you|it)|"
                     r"(could|can).{0,12}(say|tell).{0,10}again|pardon|"
                     r"phir se (bol|bata|kah|suna)|dobara (bol|bata|kah|suna)|dubara (bol|bata|kah|suna)|ek baar (aur|phir) (bol|bata)|kya bola|kya kaha|"
                     r"फिर से (बोल|बता|कह|सुना)|दोबारा (बोल|बता|कह|सुना)|दुबारा (बोल|बता|कह|सुना)|एक बार (और|फिर) (बोल|बता)|क्या बोला|क्या कहा", re.I)
# complaints about repetition, not requests for it
_NOT_REPEAT = re.compile(r"stop repeat|don.?t repeat|repeating (yourself|the same|again)|keep repeating|same thing again and again|"
                         r"(you.?re|you are|are you) repeating|repeat kar (rahe|raha|rahi)|ek hi (line|baat|dialogue|cheez)|दोहरा (रहे|रहा|रही)|वही (बात|लाइन) (बार|फिर)|"
                         r"baar baar|bar bar|बार बार|बार-बार|repeat (mat|na) kar|रिपीट (मत|ना)", re.I)
# a repeat of something specific ("repeat the price") is a question about that thing, not "the last thing"
_REPEAT_TARGET = re.compile(r"(repeat|again).{0,20}\b(price|cost|offer|terms|number|link|name|address|timing)", re.I)
_WHERE = re.compile(r"where (were|are|was) we|what were we (talking|discussing|saying)|(i|we) (got|get|was|were|am) (a bit |a little )?lost|"
                    r"lost (track|the thread|you for a|for a (bit|moment|second|minute))|i just lost|"
                    r"what did we (agree|decide|settle|say)|what (price|amount) (did we|were we|we) (agree|settle|at)|"
                    r"where did we (leave|stop|get to)|recap|summari[sz]e (it|that|where)|"
                    r"kahan (the|thhe|the hum|tak pahunche)|kahaan the|hum kahan|kya tay hua|kitne mein tay|"
                    r"कहाँ थे|कहां थे|कहाँ तक|कहां तक|क्या तय हुआ|कितने में तय", re.I)


def recall_kind(text: str) -> str | None:
    """'repeat' (say the last thing again), 'where' (restate where we are), or None."""
    t = (text or "").strip()
    if not t or len(t.split()) > 22:
        return None
    if _WHERE.search(t):
        return "where"
    if _REPEAT.search(t) and not _NOT_REPEAT.search(t) and not _REPEAT_TARGET.search(t):
        return "repeat"
    return None


def _money(cur: str, n) -> str:
    try:
        return f"{cur or '₹'}{int(round(float(n))):,}"
    except (TypeError, ValueError):
        return str(n)


def _last_question(line: str) -> str:
    parts = [x.strip() for x in re.split(r"(?<=[.!?।])\s+", line or "") if x.strip()]
    qs = [x for x in parts if x.endswith("?")]
    return qs[-1] if qs else ""


def repeat_texts(ctx: "CallContext", fallback_line: str = "") -> dict[str, str] | None:
    """'Can you repeat the last thing?' -> the last real agent line, word for word."""
    line = ctx.last_agent_line or fallback_line
    if not line:
        return None
    return {"en": f"Sure. {line}", "hi": f"जी, ज़रूर। {line}", "hinglish": f"Ji, zaroor. {line}"}


def where_texts(ctx: "CallContext", product_name: str = "", deal_recap: dict | str | None = None) -> dict[str, str]:
    """'Where were we?' -> the state, restated from the ledger (never a new concession, never a new pitch).
    `deal_recap` (from the negotiation domain, agent haggle) replaces the default deal sentence when given."""
    c, dv = ctx.checklist, ctx.deal_view
    cur = dv.get("currency") or "₹"
    pn = product_name or "the offer"
    if isinstance(deal_recap, str):
        deal_recap = {"en": deal_recap, "hi": deal_recap, "hinglish": deal_recap}
    if deal_recap:
        deal = deal_recap
    elif dv.get("agreed_price"):
        a = _money(cur, dv["agreed_price"])
        deal = {"en": f"We had agreed on {a} for {pn}.", "hi": f"हमने {pn} के लिए {a} तय किया था।",
                "hinglish": f"Humne {pn} ke liye {a} tay kiya tha."}
    elif dv.get("last_agent_offer") or dv.get("last_customer_counter"):
        en, hi, hg = ["We were settling the price for " + pn + "."], [f"हम {pn} की कीमत तय कर रहे थे।"], [f"Hum {pn} ki price tay kar rahe the."]
        if dv.get("last_customer_counter"):
            m = _money(cur, dv["last_customer_counter"])
            en.append(f"You asked for {m}."), hi.append(f"आपने {m} कहा था।"), hg.append(f"Aapne {m} kaha tha.")
        if dv.get("last_agent_offer"):
            m = _money(cur, dv["last_agent_offer"])
            en.append(f"My last offer was {m}."), hi.append(f"मेरा आख़िरी ऑफ़र {m} था।"), hg.append(f"Mera last offer {m} tha.")
        deal = {"en": " ".join(en), "hi": " ".join(hi), "hinglish": " ".join(hg)}
    elif c.price_quoted:
        deal = {"en": f"I had told you about {pn} and its offer price.", "hi": f"मैंने आपको {pn} और उसकी ऑफ़र कीमत के बारे में बताया था।",
                "hinglish": f"Maine aapko {pn} aur uski offer price ke baare mein bataya tha."}
    elif c.product_explained:
        deal = {"en": f"I was telling you about {pn}.", "hi": f"मैं आपको {pn} के बारे में बता रहा था।",
                "hinglish": f"Main aapko {pn} ke baare mein bata raha tha."}
    else:
        deal = {"en": f"I had just started telling you about {pn}.", "hi": f"मैंने अभी {pn} के बारे में बताना शुरू ही किया था।",
                "hinglish": f"Maine abhi {pn} ke baare mein batana shuru hi kiya tha."}
    if c.link_sent:
        nxt = {"en": "The payment link is already on its way to you.", "hi": "पेमेंट लिंक आपको भेजा जा चुका है।",
               "hinglish": "Payment link aapko bheja ja chuka hai."}
    elif dv.get("agreed_price"):
        nxt = {"en": "Shall I go ahead at that price?", "hi": "क्या मैं इसी कीमत पर आगे बढ़ूँ?", "hinglish": "Kya main isi price par aage badhun?"}
    else:
        q = _last_question(ctx.last_agent_line)
        nxt = {"en": q, "hi": q, "hinglish": q} if q else {"en": "Shall we continue?", "hi": "क्या हम आगे बात करें?", "hinglish": "Kya hum aage baat karein?"}
    lead = {"en": "No problem.", "hi": "कोई बात नहीं।", "hinglish": "Koi baat nahi."}
    return {k: " ".join(x for x in (lead[k], deal.get(k) or deal.get("en", ""), nxt.get(k) or nxt["en"]) if x) for k in ("en", "hi", "hinglish")}
