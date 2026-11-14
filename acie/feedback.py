"""Layer 8 - Feedback engine: turns what happened in a turn into long-term memory updates.

Implicit evidence is deliberately weak (small lr) so one odd call never
rewrites a customer's profile; explicit statements win immediately.
"""

from __future__ import annotations

from .personalization.profile_store import update_preference
from .schemas import CustomerProfile, ExpressionParams, HumanState, TurnPlan
from .text import detect_language


class FeedbackEngine:
    def after_turn(self, profile: CustomerProfile, text: str, explicit: dict[str, str],
                   prev_expr: ExpressionParams | None, state: HumanState, plan: TurnPlan) -> None:
        for key, value in explicit.items():
            update_preference(profile, key, value, explicit=True)

        # The language the customer actually speaks is the strongest implicit signal.
        if "language" not in explicit:
            update_preference(profile, "language", detect_language(text), lr=0.15)

        # If the previous response style caused confusion, nudge complexity preference down;
        # if it went smoothly, reinforce it.
        if prev_expr is not None and "complexity" not in explicit:
            if state.confusion >= 0.5:
                update_preference(profile, "complexity", "low", lr=0.15)
            elif state.confusion < 0.2 and state.frustration < 0.2:
                update_preference(profile, "complexity", prev_expr.complexity, lr=0.05)

        if plan.intent == "card_declined":
            profile.outcomes.append("offer_declined")
        elif plan.intent == "transfer_done":
            profile.outcomes.append("transfer_done")

    def end_session(self, profile: CustomerProfile) -> None:
        profile.interaction_count += 1
        profile.outcomes = profile.outcomes[-50:]
