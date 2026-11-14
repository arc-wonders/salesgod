"""Deterministic lead score, 0-100. Same customer + same `now` => same score. No randomness, no LLM.

score = clamp(stage + engagement + outcomes + recency + call_deltas, 0, 100), rounded to int
  - stage        new 5, contacted 15, interested 40, qualified 55, demo_booked 70, negotiating 80   (won = 100, lost = 0, do_not_call = 0: fixed)
  - engagement   +5 per connected call (a call with an outcome other than no_answer/voicemail/failed), max +15
  - outcomes     last outcome: accepted +15, interested +10, callback +8, later +3, undecided 0, declined -20;
                 each no_answer/voicemail/failed among the last 3 outcomes -3 (max -9)
  - recency      days since last contact: <=2 +10, <=7 +6, <=14 +3, <=30 0, older -5; never contacted 0
  - call_deltas  sum of the calls' structured score_delta, clamped to +-20
"""

from __future__ import annotations

import time

from .models import Customer

STAGE_POINTS = {"new": 5, "contacted": 15, "interested": 40, "qualified": 55, "demo_booked": 70, "negotiating": 80}
LAST_OUTCOME = {"accepted": 15, "interested": 10, "callback": 8, "later": 3, "undecided": 0, "declined": -20}
UNREACHED = {"no_answer", "voicemail", "failed"}


def score(c: Customer, now: float | None = None) -> int:
    if c.stage == "won":
        return 100
    if c.stage in ("lost", "do_not_call"):
        return 0
    now = time.time() if now is None else now
    pts = STAGE_POINTS.get(c.stage, 5)
    outs = [o.get("outcome", "") for o in c.history.outcomes]
    connected = sum(1 for o in outs if o and o not in UNREACHED)
    pts += min(15, 5 * connected)
    if outs:
        pts += LAST_OUTCOME.get(outs[-1], 0)
        pts -= min(9, 3 * sum(1 for o in outs[-3:] if o in UNREACHED))
    if c.last_contact_at:
        days = max(0.0, (now - c.last_contact_at) / 86400)
        pts += 10 if days <= 2 else 6 if days <= 7 else 3 if days <= 14 else 0 if days <= 30 else -5
    pts += max(-20, min(20, c.score_delta_total))
    return int(max(0, min(100, round(pts))))
