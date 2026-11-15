"""Decision trace: which branch of the decision tree produced a turn, for the live console.

The tree has fixed policy gates (checked in this order on every turn) and, under "Conversation", the
tenant's playbook situations. `GATES` is also served to the console to draw the canvas.
"""

from __future__ import annotations

GATES = [
    {"id": "safety", "title": "Safety", "rule": "Distress or self-harm: stop selling, give the helpline, stay kind",
     "intents": ["safety_distress", "safety_followup", "safety_noted", "safety_goodbye"]},
    {"id": "compliance", "title": "Compliance", "rule": "Opt-out, card/OTP stop, admit AI, who is calling, number source",
     "intents": ["opt_out", "opt_out_confirmed", "sensitive_stop", "are_you_ai", "identify", "number_source"]},
    {"id": "decided", "title": "Already decided", "rule": "After a sale, a no or a callback: answer, never re-sell",
     "intents": ["link_cancelled", "goodbye_after"]},
    {"id": "unclear", "title": "Unclear line", "rule": "Garbled or noisy speech: ask again, never sell on it",
     "intents": ["clarify"]},
    {"id": "close", "title": "Consent & close", "rule": "Explicit yes + full terms read back before any link",
     "intents": ["confirm_close", "confirm_buy", "link_sent"]},
    {"id": "routing", "title": "Routing", "rule": "Callback with time, human handoff, language switch",
     "intents": ["callback", "human_agent", "language_switch"]},
    {"id": "exit", "title": "Refusal & goodbye", "rule": "One soft re-offer at most, then respect the no",
     "intents": ["reoffer", "declined", "goodbye"]},
    {"id": "scripted", "title": "Scripted steps", "rule": "Opening, operator pitch, exact price line, hello, slow down",
     "intents": ["opening", "greet", "pitch", "price", "hello", "slow_down"]},
    {"id": "conversation", "title": "Conversation", "rule": "Answer first, then the playbook situation's talking points",
     "intents": ["answer", "small_talk", "personal"]},
]
_BY_INTENT = {i: g["id"] for g in GATES for i in g["intents"]}


def gate_of(intent: str) -> str:
    return _BY_INTENT.get(intent, "conversation")


def trace(result, dialog: dict | None) -> dict:
    d = dialog or {}
    plan = result.plan
    return {
        "gate": gate_of(plan.intent),
        "decision": plan.decision or "",
        "objective": (plan.objective or "")[:400],
        "pb_node": d.get("pb_node_turn") or None,
        "pb_title": d.get("pb_title_turn") or None,
        "lead": bool(plan.lead),
        "llm": plan.llm_allowed,
        "end_call": plan.end_call,
        "rule_intent": d.get("_rule_intent"),
        "pending": d.get("pending"),
        "issues": result.validation.issues,
        "fallback": result.validation.used_fallback,
        "social": _social(d),
    }


def _social(d: dict) -> dict | None:
    """The caller's social signals this turn (acie/state/social.py), for the console's meters."""
    import json
    try:
        v = json.loads(d.get("_social") or "null")
    except (ValueError, TypeError):
        return None
    return {k: v[k] for k in ("sarcasm", "humour", "hostility", "interest", "impatience", "scepticism", "rapport") if k in v} if isinstance(v, dict) else None
