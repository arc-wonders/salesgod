"""enrich_plan(): add the tree hint to a TurnPlan so the LLM gets more to work with. Behind ACIE_KB=1 (default off).

  * KB answers go into plan.context as NON-verbatim talking points (the LLM paraphrases them in the caller's language);
  * the call map (stage, caller's interests, what was already told, next move) goes into plan.reference (<= 700 chars);
  * prices never appear as text: an offering price (or any {price:ID} in an entry) becomes a Fact slot (verbatim) that
    the LLM can only mention through its {slot};
  * entries handed over are recorded as given, so the next turn never gets the same ones again;
  * high-stakes plans (llm_allowed False), ended calls and safety / stop plans are left alone.
"""

from __future__ import annotations

import re
from typing import Any

from ..knowledge.render import SLOT
from ..knowledge.retrieve import retrieve
from ..knowledge.store import kb_enabled, load_kb
from ..schemas import Fact, TurnPlan
from . import state as S
from .hint import TreeHint, tree_hint

SKIP_INTENTS = {"safety_distress", "safety_followup", "safety_goodbye", "safety_noted", "constraint_stop", "opt_out", "opt_out_confirmed",
                "declined", "declined_price", "annoyed_exit", "link_sent", "human_agent", "callback", "confirm_close", "greet", "opening",
                "identify", "are_you_ai", "language_switch"}
_SLOT_NAME = re.compile(r"[^a-z0-9_]")


def slot_name(offering_id: str) -> str:
    return "kb_price_" + _SLOT_NAME.sub("_", offering_id.lower())


def _to_slots(raw: str, facts: dict[str, Fact], kb, lang: str) -> str | None:
    """Entry text with prices turned into slots; None when it needs a product price slot we cannot safely fill here."""
    offs = kb.offering_map()
    bad = False

    def sub(m: re.Match) -> str:
        nonlocal bad
        name, arg = m.group(1), m.group(2)
        if name == "price" and arg in offs and offs[arg].price_label:
            sn = slot_name(arg)
            facts[sn] = Fact(value=offs[arg].price_label.get(lang), description=f"price of {offs[arg].name.en}", must_mention=False)
            return "{" + sn + "}"
        if name == "name" and arg in offs:
            return offs[arg].name.get(lang)
        bad = True
        return ""

    out = SLOT.sub(sub, raw)
    return None if bad else out


def apply_hint(plan: TurnPlan, hint: TreeHint, kb, session: Any) -> TurnPlan:
    facts = dict(plan.facts)
    lines: list[str] = []
    used: list[str] = []
    for p in hint.entries:
        e = kb.entry_map().get(p.id)
        if e is None:  # a topic overview standing in for "nothing matched"
            tp = kb.topic_map().get(p.topic)
            raw = tp.overview.get(hint.lang) if tp else ""
            txt = _to_slots(raw, facts, kb, hint.lang) if raw else None
            if txt:
                lines.append("BACKGROUND ON THE TOPIC: " + txt.strip())
            continue
        txt = _to_slots(e.answer.get(hint.lang), facts, kb, hint.lang)
        if txt is None:
            continue
        lines.append(("ANSWER: " if p.kind == "answer" else "TIP TO TEACH: ") + txt.strip())
        used.append(p.id)
    off_id = None
    if hint.offering is not None and hint.move == "bridge_offering":
        o = kb.offering_map()[hint.offering.id]
        off_id = o.id
        pitch = "; ".join(x.rstrip(".") for x in hint.offering.pitch[:2])
        line = f"OFFER ONE THING THAT FITS: {o.name.get(hint.lang)}: {pitch}. Why it fits: {hint.offering.why}."
        if o.price_label and not o.included:
            sn = slot_name(o.id)
            facts[sn] = Fact(value=o.price_label.get(hint.lang), description=f"price of {o.name.en}; say it only if asked or when making the offer",
                             must_mention=False)
            line += " Price only through {" + sn + "}."
        lines.append(line + " Bring it up after answering, in one natural sentence, then let them decide.")
    ctx = plan.context or ""
    if lines:
        ctx = (ctx + "\n\n" if ctx else "") + ("Product details you may use now, in your own words (talking points, not a script; not yet told to the caller):\n"
                                              + "\n".join("- " + l for l in lines))
    ref = plan.reference or ""
    cov = hint.coverage_text()
    ref = (ref + "\n" if ref else "") + cov
    update = {"facts": facts, "context": ctx or None, "reference": ref or None}
    new = plan.model_copy(update=update)
    S.mark_given(session, used, off_id)
    return new


def enrich_plan(plan: TurnPlan, session: Any, signals: Any, text: str, product: Any, lang: str | None = None) -> TurnPlan:
    """Plan + KB talking points + call map. Returns the plan unchanged when ACIE_KB is off, the product has no KB, or the turn
    must stay exactly as designed."""
    if not kb_enabled() or plan.end_call or not plan.llm_allowed or plan.intent in SKIP_INTENTS:
        return plan
    kb = load_kb(product)
    if kb is None:
        return plan
    hint = tree_hint(session, signals, text, product, lang)
    if hint is None:
        return plan
    return apply_hint(plan, hint, kb, session)


def kb_knowledge(product: Any, text: str, lang: str | None = None, k: int = 3) -> str:
    """For product_sales' background text: the best KB talking points for a question (or the nearest topic overview), so the
    LLM never has to say "I don't know, contact the owner". Empty when ACIE_KB is off or there is no KB. Prices are dropped
    (price answers stay with the reviewed templates)."""
    if not kb_enabled():
        return ""
    kb = load_kb(product)
    if kb is None:
        return ""
    r = retrieve(kb, text, lang, k=k, product=product)
    lines = []
    for h in r.hits:
        if "{" in " ".join(h.entry.answer.all()):
            continue
        lines.append("- " + h.text)
    if not lines:
        return ""
    head = "Nearest topic overview" if r.fallback else "Product details for this question (paraphrase, do not recite)"
    return head + ":\n" + "\n".join(lines)
