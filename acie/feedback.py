"""Layer 8 - Feedback engine: turns what happened in a turn into long-term memory updates.

Implicit evidence is deliberately weak (small lr) so one odd call never
rewrites a customer's profile; explicit statements win immediately.

Sales outcomes (end_session, once per call)
  Read from session.context.decision / promised and recorded in profile.outcomes (last 50), one primary label:
  safety (distress flag) > opt_out (number removed) > accepted / declined / later > no_answer (the customer never
  spoke), plus human_agent whenever a team member was promised. The bank-demo labels (offer_declined,
  transfer_done) are still written per turn. end_session(profile, session) is idempotent per session.

Per-turn reward in [-1, 1]  (needs turn t+1, so (state, expr) of turn t stays pending until the next turn / call end)
  reward = clip( - w_frustration * d_frustration  - w_confusion * d_confusion
                 + w_positive * mean(d_engagement, d_satisfaction, d_receptiveness)
                 - w_validator * penalty - w_barge * barge_in , -1, 1 )
  d_x = x(next turn) - x(this turn); engagement/satisfaction/receptiveness missing on HumanState (state lane
  adds them) are read with getattr default 0.5. penalty = 1.0 if the reply used the safe fallback, else 0.4 if
  it carried real validation issues (issue_codes if the report has them, else issues; trimming/echo/regenerate
  notes are benign). barge_in = the customer interrupted that reply (next turn's timing.barge_in).
  Default weights: frustration 0.8, confusion 0.6, positive 0.5, validator 0.5, barge 0.3 (WEIGHTS; override per
  engine via FeedbackEngine(weights={...}) or env ACIE_REWARD_WEIGHTS="frustration=0.9,barge=0.2").

Delayed call-level reward credited to every turn of the call at end_session (discounted 0.9 per turn back from the
last one, sent as a second policy.observe): accepted +1, later +0.2, declined -0.3, opt_out -0.6; safety and
no outcome are excluded from learning (no observe, and per-turn learning stops once the safety flag is set).

Learning hooks (called when present, never break a turn)
  acie.personalization.learner.learn_from_turn(profile, text, state, expr, explicit)   (persona lane; lazy import)
  policy.observe(state, expr, reward, profile)                                       (BanditPolicy)

Validator learning
  Whether each reply passed validation is kept per language as the last 4 results in the profile preference
  "validator_recent_<lang>" ("1" = failed). If >= 2 of the last 4 failed, complexity is nudged to "low" and the
  preference "prefer_templates" = "true" is set; the decision layer may read it to speak safe template replies for
  this caller. It flips back to "false" after 4 clean replies in a row.

Semantic memory (ACIE_MEMORY=1; see acie/memory): remember() stores a compact call summary per customer at the end
of the call; recall() puts the top-3 similar memories on session.dialog["_memories"] (JSON list of texts) at call
start for the compiler.
"""

from __future__ import annotations

import json
import os

from .memory.base import MemoryStore, make_store, memory_enabled
from .personalization.profile_store import update_preference
from .schemas import CustomerProfile, ExpressionParams, HumanState, SessionState, TurnPlan
from .text import detect_language

WEIGHTS = {"frustration": 0.8, "confusion": 0.6, "positive": 0.5, "validator": 0.5, "barge": 0.3}
CALL_REWARD = {"accepted": 1.0, "later": 0.2, "declined": -0.3, "opt_out": -0.6}
_BENIGN = ("trimmed", "dropped echo", "regenerated", "llm repeated")
CREDIT_DISCOUNT = 0.9


def _env_weights() -> dict[str, float]:
    out = {}
    for part in os.getenv("ACIE_REWARD_WEIGHTS", "").split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            try:
                out[k.strip()] = float(v)
            except ValueError:
                pass
    return out


def _g(state, name: str, default: float = 0.5) -> float:
    return float(getattr(state, name, default))


def validation_penalty(validation) -> float:
    """1.0 safe fallback used, 0.4 real issues, 0 clean."""
    if validation is None:
        return 0.0
    if getattr(validation, "used_fallback", False):
        return 1.0
    codes = getattr(validation, "issue_codes", None) or getattr(validation, "issues", None) or []
    return 0.4 if any(not str(c).startswith(_BENIGN) for c in codes) else 0.0


def validation_failed(validation) -> bool:
    return validation_penalty(validation) > 0


def sales_outcome(session: SessionState) -> tuple[str | None, bool]:
    """(primary outcome or None, human_agent promised)."""
    ctx = session.context
    human = "team_member_will_call" in ctx.promised
    if session.dialog.get("safety_flag"):
        return "safety", human
    if "number_removed_no_more_calls" in ctx.promised:
        return "opt_out", human
    if ctx.decision in ("accepted", "declined", "later"):
        return ctx.decision, human
    if not any(u.role == "customer" and u.text.strip() for u in session.history):
        return "no_answer", human
    return None, human


class FeedbackEngine:
    def __init__(self, weights: dict[str, float] | None = None, policy=None, memory: MemoryStore | None = None) -> None:
        self.weights = {**WEIGHTS, **_env_weights(), **(weights or {})}
        self.policy = policy  # pipeline sets this to its policy when None
        self.memory = memory if memory is not None else (make_store() if memory_enabled() else None)
        self._pending: dict[str, dict] = {}  # session_id -> {"state","expr","penalty"}
        self._turns: dict[str, list[tuple[HumanState, ExpressionParams, float]]] = {}

    # ------------------------------------------------------------------ reward
    def turn_reward(self, prev: HumanState, nxt: HumanState, penalty: float = 0.0, barge_in: bool = False) -> float:
        w = self.weights
        pos = sum(_g(nxt, n) - _g(prev, n) for n in ("engagement", "satisfaction", "receptiveness")) / 3
        r = (-w["frustration"] * (nxt.frustration - prev.frustration) - w["confusion"] * (nxt.confusion - prev.confusion)
             + w["positive"] * pos - w["validator"] * penalty - w["barge"] * (1.0 if barge_in else 0.0))
        return max(-1.0, min(1.0, r))

    def _observe(self, state, expr, reward, profile) -> None:
        obs = getattr(self.policy, "observe", None)
        if obs is not None:
            try:
                obs(state, expr, reward, profile)
            except Exception:
                pass

    def _resolve(self, sid: str, profile, nxt: HumanState | None, barge_in: bool) -> None:
        p = self._pending.pop(sid, None)
        if p is None:
            return
        r = self.turn_reward(p["state"], nxt if nxt is not None else p["state"], p["penalty"], barge_in)
        self._turns.setdefault(sid, []).append((p["state"], p["expr"], r))
        self._observe(p["state"], p["expr"], r, profile)

    # ------------------------------------------------------------------ per turn
    def after_turn(self, profile: CustomerProfile, text: str, explicit: dict[str, str],
                   prev_expr: ExpressionParams | None, state: HumanState, plan: TurnPlan, *,
                   session: SessionState | None = None, expr: ExpressionParams | None = None,
                   validation=None, barge_in: bool = False) -> None:
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

        cur = expr or prev_expr
        if cur is not None and validation is not None:
            self._validator_learning(profile, cur, validation, explicit)

        if session is not None and session.dialog.get("safety_flag"):
            return  # a caller in crisis is never learning material
        if cur is not None:
            self._learn_hook(profile, text, state, cur, explicit)
            sid = session.session_id if session is not None else profile.customer_id
            self._resolve(sid, profile, state, barge_in)  # reward of the PREVIOUS reply: how the caller reacted
            self._pending[sid] = {"state": state, "expr": cur, "penalty": validation_penalty(validation)}

    def _learn_hook(self, profile, text, state, expr, explicit) -> None:
        try:
            from .personalization.learner import learn_from_turn  # persona lane; may not exist yet
        except Exception:
            return
        try:
            learn_from_turn(profile, text, state, expr, explicit)
        except Exception:
            pass

    def _validator_learning(self, profile, expr, validation, explicit) -> None:
        key = f"validator_recent_{expr.language}"
        cur = profile.preferences.get(key)
        recent = ((cur.value if cur else "") + ("1" if validation_failed(validation) else "0"))[-4:]
        update_preference(profile, key, recent, explicit=True)  # raw history string; explicit = exact overwrite
        if recent.count("1") >= 2:
            if "complexity" not in explicit:
                update_preference(profile, "complexity", "low", lr=0.2)
            update_preference(profile, "prefer_templates", "true", lr=0.4)
        elif recent == "0000" and "prefer_templates" in profile.preferences:
            update_preference(profile, "prefer_templates", "false", lr=0.4)

    # ------------------------------------------------------------------ end of call
    def end_session(self, profile: CustomerProfile, session: SessionState | None = None) -> None:
        if session is not None:
            if session.dialog.get("_fb_ended"):
                return  # already closed (end_call turn, then the server's hang-up handler)
            session.dialog["_fb_ended"] = "1"
        profile.interaction_count += 1
        outcome = None
        if session is not None:
            outcome, human = sales_outcome(session)
            if outcome:
                profile.outcomes.append(outcome)
            if human:
                profile.outcomes.append("human_agent")
            self._delayed_credit(session, profile, outcome)
        profile.outcomes = profile.outcomes[-50:]

    def _delayed_credit(self, session: SessionState, profile, outcome: str | None) -> None:
        sid = session.session_id
        if outcome != "safety":
            self._resolve(sid, profile, None, False)  # last turn has no next state
        self._pending.pop(sid, None)
        turns = self._turns.pop(sid, [])
        reward = CALL_REWARD.get(outcome or "")
        if reward is None:
            return
        n = len(turns)
        for i, (st, ex, _) in enumerate(turns):
            self._observe(st, ex, reward * CREDIT_DISCOUNT ** (n - 1 - i), profile)

    # ------------------------------------------------------------------ semantic memory
    @staticmethod
    def call_summary(session: SessionState, profile: CustomerProfile | None = None) -> str:
        ctx = session.context
        parts = [f"decision: {ctx.decision}"]
        if ctx.objections:
            parts.append("objections: " + ", ".join(ctx.objections))
        if ctx.questions:
            parts.append("questions: " + " | ".join(q["q"] for q in ctx.questions[-5:]))
        likes = getattr(profile, "likes", None) or []
        if likes:
            parts.append("likes: " + ", ".join(likes[:5]))
        return "; ".join(parts)

    async def remember(self, session: SessionState, profile: CustomerProfile | None = None) -> None:
        if self.memory is None or session.dialog.get("_fb_remembered"):
            return
        session.dialog["_fb_remembered"] = "1"
        try:
            await self.memory.add(session.customer_id, self.call_summary(session, profile),
                                  {"session_id": session.session_id, "decision": session.context.decision})
        except Exception:
            pass  # memory never breaks a call

    async def recall(self, session: SessionState, query: str = "", k: int = 3) -> None:
        if self.memory is None or "_memories" in session.dialog:
            return
        try:
            found = await self.memory.search(session.customer_id, query, k)
        except Exception:
            return
        if found:
            session.dialog["_memories"] = json.dumps([m["text"] for m in found], ensure_ascii=False)
