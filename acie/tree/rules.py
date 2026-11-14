"""The documented transition tables of the multi-layer decision tree. Pure data + tiny predicates; no randomness.

LAYER 1  STAGE    open -> discover -> educate -> recommend -> objection -> negotiate -> close -> after_close
LAYER 2  CLUSTER  interest tags inferred from the caller's words (acie/tree/interest.py)
LAYER 3  TOPIC    a node of the KB taxonomy: the best match for the question, else the least explored one that speaks to
                  the top cluster
LAYER 4  ACTION   leaf: answer | teach | ask_discovery | bridge_offering | close_ask | handle_objection | acknowledge_close

STAGE_RULES are evaluated top to bottom; the first whose predicate holds wins.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

STAGES = ("open", "discover", "educate", "recommend", "objection", "negotiate", "close", "after_close")
ACTIONS = ("answer", "teach", "ask_discovery", "bridge_offering", "close_ask", "handle_objection", "acknowledge_close")

HAGGLE = re.compile(r"discount|lower|cheaper|less price|kam kar|कम कर|reduce|bargain|negotiat|डिस्काउंट|सस्ता|"
                    r"(want|give|make).{0,25}\bfor\b.{0,6}\d[\d,]{2,}|\bfor\b\s*\d[\d,]{2,}", re.I)
QUESTION = re.compile(r"\?|\b(what|how|why|when|where|which|who|kya|kaise|kab|kyun|kyon|kaun|kahan|kitna|kitne|do you|can i|can you|"
                      r"is there|are there|does)\b|क्या|कैसे|कब|क्यों|कौन|कहां|कहाँ|कितना|कितने", re.I)

STRONG_FIT = 2.0         # offering fit at/above which a stated interest is strong: answer, then bridge in the same turn
EXPLORED_FOR_BRIDGE = 2  # entries asked/given in the offering's cluster before teach turns into bridge


@dataclass(frozen=True)
class Ctx:
    customer_turns: int
    text: str
    intent: str
    objection: str
    buying: float
    decision: str
    promised_link: bool
    is_question: bool
    has_cluster: bool
    offering_ready: bool


Rule = tuple[str, Callable[[Ctx], bool], str, str]  # (stage, predicate, rule id, plain-words condition)

STAGE_RULES: list[Rule] = [
    ("after_close", lambda c: c.decision == "accepted" or c.promised_link, "R1", "caller accepted / payment link promised"),
    ("negotiate", lambda c: c.objection == "price" and bool(HAGGLE.search(c.text)), "R2", "price objection and asking for a lower price"),
    ("objection", lambda c: c.objection not in ("none", ""), "R3", "any other objection raised this turn"),
    ("close", lambda c: c.buying >= 0.7 or (c.intent == "affirm" and c.customer_turns > 1), "R4", "buying interest >= 0.7 or a clear yes"),
    ("open", lambda c: c.customer_turns == 0 or (c.customer_turns == 1 and not c.has_cluster and not c.is_question), "R5",
     "first customer turn with nothing stated yet"),
    ("educate", lambda c: c.is_question or c.intent in ("benefits", "tell_more", "clarify"), "R6", "the caller asked something: teach by answering"),
    ("recommend", lambda c: c.offering_ready, "R7", "an offering fits the stated interests and the cluster was explored"),
    ("discover", lambda c: True, "R8", "otherwise: find out what they want"),
]


def stage_for(c: Ctx) -> tuple[str, str]:
    for stage, pred, rid, _ in STAGE_RULES:
        if pred(c):
            return stage, rid
    return "discover", "R8"


def next_move(stage: str, offering: bool, pitched: bool, strong: bool, explored_enough: bool, entries_left: bool,
              comfortable: bool) -> str:
    """Layer-4 action from the stage. `offering` = a fitting offering exists; `pitched` = it was already pitched;
    `comfortable` = caller not frustrated and not declining (we never pitch past discomfort)."""
    if stage == "after_close":
        return "acknowledge_close"
    if stage in ("objection", "negotiate"):
        return "handle_objection"
    if stage == "close":
        return "close_ask"
    if offering and comfortable and not pitched and (strong or explored_enough or stage == "recommend"):
        return "bridge_offering"   # the answer comes first in the same turn, then the bridge
    if stage == "educate":
        return "answer"
    if stage == "recommend":
        return "teach"
    if stage == "open":
        return "ask_discovery"
    return "teach" if entries_left and offering else "ask_discovery"
