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

import re
import time
from typing import Callable

from pydantic import BaseModel, Field


_QUESTION = re.compile(r"\?|\b(what|how|why|when|where|which|who|kya|kaise|kab|kyun|kyon|kaun|kahan|kitna|kitne)\b|"
                       # "do/is/can" only open a question at the start ("okay fine, let's do it" is not one)
                       r"^\W*(can|could|does|do|is|are|will|would|should)\b|क्या|कैसे|कब|क्यों|कौन|कहां|कहाँ|कितना|कितने|कितनी",
                       re.IGNORECASE)


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


class CallContext(BaseModel):
    events: list[ContextEvent] = Field(default_factory=list)
    promised: list[str] = Field(default_factory=list)
    quoted: list[str] = Field(default_factory=list)
    told: list[str] = Field(default_factory=list)
    questions: list[dict] = Field(default_factory=list)  # {"q": text, "answer": intent}
    objections: list[str] = Field(default_factory=list)
    decision: str = "undecided"
    outcome: str = ""  # see _OUTCOMES

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
        return "\n".join(lines)
