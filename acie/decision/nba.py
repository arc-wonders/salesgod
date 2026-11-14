"""Deterministic next-best-action scorer (used only when ACIE_NBA=1).

Inputs: FSM stage, buying_signal (0..1, NLU), open objections, the context ledger.

  score(a) = BASE[stage][a]
           + 2.0 * buying      * READY[a]          readiness pushes the closing actions up
           - 1.0 * open_obj    * PUSHY[a]          each unresolved objection pushes pushy actions down
           + 2.5 * open_obj    * (a == "answer_objection")
           - 2.0 * already_done(a)                 do not repeat what the ledger says was already said
           - 6.0 * blocked(a)                      hard rules: no sale after a decision, no close before the price

stage in {intro, pitch, discovery, close, confirm}. The highest score wins; ties go to the earlier action in
ACTIONS. The result is advice for the objective line (never a replacement for the compliance policy).
"""

from __future__ import annotations

ACTIONS = ["answer_objection", "discover", "pitch", "quote_price", "benefit", "confirm_close", "soft_exit"]

BASE = {
    "intro":     {"pitch": 3.0, "discover": 1.0, "quote_price": 0.5},
    "pitch":     {"quote_price": 3.0, "benefit": 1.5, "discover": 1.0},
    "discovery": {"benefit": 2.5, "quote_price": 1.5, "discover": 1.0},
    "close":     {"confirm_close": 2.0, "benefit": 1.5, "soft_exit": 0.5},
    "confirm":   {"confirm_close": 3.0},
}
READY = {"confirm_close": 1.5, "quote_price": 0.5, "benefit": 0.2}
PUSHY = {"confirm_close": 1.5, "quote_price": 0.5, "pitch": 0.3, "soft_exit": -0.5}
# which ledger entry means an action was already done
DONE = {"pitch": ("told", "what_the_product_is"), "quote_price": ("quoted", "offer_price")}


def enabled() -> bool:
    import os
    return os.getenv("ACIE_NBA") == "1"


def next_best_action(stage: str, buying: float, objections: list[str], ctx=None, resolved: int = 0,
                     benefit_turns: int = 0) -> tuple[str, dict[str, float]]:
    open_obj = max(0, len(objections) - resolved)
    quoted = ctx is not None and "offer_price" in getattr(ctx, "quoted", [])
    decided = ctx is not None and (getattr(ctx, "decision", "undecided") != "undecided" or ctx.has("payment_link"))
    scores: dict[str, float] = {}
    for a in ACTIONS:
        s = BASE.get(stage, BASE["intro"]).get(a, 0.0)
        s += 2.0 * buying * READY.get(a, 0.0)
        s -= 1.0 * open_obj * PUSHY.get(a, 0.0)
        if a == "answer_objection":
            s += 2.5 * open_obj
        if a == "benefit":
            s -= 0.5 * benefit_turns
        if a in DONE and ctx is not None and DONE[a][1] in getattr(ctx, DONE[a][0], []):
            s -= 2.0
        if a == "confirm_close" and not quoted:
            s -= 6.0
        if decided and a != "soft_exit":
            s -= 6.0
        scores[a] = round(s, 3)
    best = max(ACTIONS, key=lambda a: (scores[a], -ACTIONS.index(a)))
    return best, scores


HINT = {
    "answer_objection": "Next best step: address their concern directly first, no selling line.",
    "discover": "Next best step: ask one open question about what they need.",
    "pitch": "Next best step: briefly say what the product is.",
    "quote_price": "Next best step: only if it fits, offer to give the price.",
    "benefit": "Next best step: tie one concrete benefit to what you know about them.",
    "confirm_close": "Next best step: you may ask once, lightly, whether they'd like to try it.",
    "soft_exit": "Next best step: keep it light and leave the door open; do not push.",
}
