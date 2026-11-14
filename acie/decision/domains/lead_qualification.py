"""Lead qualification call: opening + purpose, the configured questions in order (need / budget / authority / timeline ...),
never re-asked once answered, questions answered from the product facts, no hard sell; ends by booking a demo or a callback
(the `book_demo` step) or a polite close. The score is deterministic: the sum of the steps' points (0-100)."""

from __future__ import annotations

from .custom_steps import StepCallDomain


class LeadQualificationDomain(StepCallDomain):
    name = "lead_qualification"

    def _fallback_outcome(self, d: dict) -> str:
        """No step earned an outcome (no demo booked): by score. >= 60 qualified, >= 40 interested, else not qualified."""
        score = int(d.get("ct_score", "0") or 0)
        if score >= self.score_qualified:
            return "qualified"
        if score >= self.score_interested:
            return "interested"
        return "not_qualified"
