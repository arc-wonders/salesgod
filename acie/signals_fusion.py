"""Deterministic fusion of provider outputs into one `TurnSignals`.

No randomness, no clocks, no dict-order luck: providers are folded in the order they are configured
(`ACIE_SIGNAL_PROVIDERS`, rules always first), ties break by that order, and the same provider outputs always
give the same result. Rules (all in CONFIG below, override with `fuse(..., config=...)`):

INTENT   The rules' intent is the base. Every later provider may REPLACE it only when its policy allows:
           min_confidence  its confidence (None = 1.0) must reach this
           only_over       replace only if the current intent is in this list ("fill the gaps")
           protected       never replace an intent in this list (the rules' hard-priority intents)
           respect_gates   a gated intent (affirm / deny) needs its dialog key, like the rules
         Today's behaviour: the keyword rules keep the last word on priority intents (clarify, human_agent,
         callback, gated affirm / deny); the LLM fills unknown / tell_more / benefits / goodbye / affirm / deny;
         a trained model replaces any intent the rules do not protect when it is >= 0.6 sure.
         Foreign labels are mapped by the domain vocabulary first; "unknown" never replaces anything.
FIELDS   objection / question_topic / buying_interest / garbled / gist: the first provider in the field's
         priority list that supplied a value (not None / "" / "none") wins.
STATE    The 14 dims start from the rules' reading; every other provider is blended in turn with a fixed weight
         (0.5 each): v = round((1 - w) * v + w * x, 3). `learned_state` is the learned providers' own reading.
"""

from __future__ import annotations

import copy
import os
from typing import Any

from .signals import DEFAULT_LEARNED, IntentVocabulary, PartialSignals, TurnSignals

FILL_SET = ["unknown", "tell_more", "benefits", "goodbye", "affirm", "deny"]
CONFIG: dict[str, Any] = {
    "intent": {
        "model": {"min_confidence": 0.6, "protected": "rule_first", "respect_gates": True},
        "default": {"min_confidence": 0.0, "only_over": FILL_SET},   # llm, http, fake, anything else
    },
    "field_priority": ["llm", "http", "model", "fake", "rules"],
    "state_weight": {"default": 0.5},
}


def default_config() -> dict[str, Any]:
    cfg = copy.deepcopy(CONFIG)
    try:
        cfg["intent"]["model"]["min_confidence"] = float(os.getenv("ACIE_TURN_MODEL_THRESHOLD", "0.6"))
    except ValueError:
        pass
    return cfg


def _present(v: Any) -> bool:
    return v is not None and v != "" and v != "none"


def fuse(parts: list[PartialSignals], vocab: IntentVocabulary, dialog: dict | None = None,
         config: dict | None = None) -> TurnSignals:
    """`parts` in provider order, the rules' partial first (providers that failed or timed out are simply absent)."""
    cfg = config or default_config()
    dialog = dialog or {}
    parts = [p for p in parts if p is not None]
    out = TurnSignals()
    rules = next((p for p in parts if p.provider == "rules"), None)
    learned = [p for p in parts if p.provider != "rules"]
    out.learned = bool(learned)

    # ---- intent
    cur, conf, src = "unknown", 0.0, "rules"
    if rules is not None and rules.intent:
        cur, conf = rules.intent, 1.0 if rules.confidence is None else rules.confidence
    out.rule_intent = cur
    scores: dict[str, float] = dict(rules.intent_scores) if rules else {}
    for p in learned:
        label = vocab.map(p.intent)
        for k, v in vocab.map_scores(p.intent_scores).items():
            scores[k] = max(scores.get(k, 0.0), v)
        out.votes[p.provider] = label
        pol = {**cfg["intent"].get("default", {}), **cfg["intent"].get(p.provider, {})}
        if label == "unknown":
            continue
        c = 1.0 if p.confidence is None else float(p.confidence)
        if c < pol.get("min_confidence", 0.0):
            continue
        only = pol.get("only_over")
        if only is not None and cur not in only:
            continue
        prot = pol.get("protected")
        prot = vocab.rule_first if prot == "rule_first" else (prot or [])
        if cur in prot:
            continue
        if pol.get("respect_gates") and vocab.gated.get(label) and vocab.gated[label] not in dialog:
            continue
        cur, conf, src = label, c, p.provider
    out.intent, out.confidence = cur, round(conf, 4)
    out.intent_scores = {k: round(scores[k], 4) for k in sorted(scores)}
    out.sources["intent"] = src

    # ---- scalar fields: first provider in the priority list that has a value
    order = {n: i for i, n in enumerate(cfg["field_priority"])}
    by_priority = sorted(enumerate(learned), key=lambda t: (order.get(t[1].provider, len(order)), t[0]))
    ranked = [p for _, p in by_priority]
    for fld in ("objection", "question_topic"):
        for p in ranked:
            v = getattr(p, fld)
            if _present(v):
                setattr(out, fld, v)
                out.sources[fld] = p.provider
                break
    for p in ranked:
        if p.buying_interest is not None:
            out.buying_interest = float(p.buying_interest)
            out.sources["buying_interest"] = p.provider
            break
    for p in ranked:
        if p.garbled is not None:
            out.garbled, out.garbled_reason = bool(p.garbled), p.garbled_reason
            out.sources["garbled"] = p.provider
            break
    for p in ranked:
        if p.gist:
            out.gist = p.gist
            out.sources["gist"] = p.provider
            break

    # ---- state dims
    base = dict(rules.state) if rules else {}
    ev = list(rules.evidence) if rules else []
    lstate: dict[str, float] = {}
    for p in learned:
        w = cfg["state_weight"].get(p.provider, cfg["state_weight"]["default"])
        for d in sorted(p.state):
            x = max(0.0, min(1.0, float(p.state[d])))
            if d in base:
                base[d] = round((1 - w) * base[d] + w * x, 3)
            lstate[d] = round((1 - w) * lstate[d] + w * x, 3) if d in lstate else x
        ev += list(p.evidence)
    for k, v in DEFAULT_LEARNED.items():
        if learned and k not in lstate:
            lstate[k] = v
    out.state, out.evidence, out.learned_state = base, ev, lstate
    for p in parts:
        if p.state:
            out.sources.setdefault("state", p.provider)
    return out
