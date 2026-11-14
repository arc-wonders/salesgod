"""Rule-pack protocol: a pack inspects one sentence and returns a short reason if it must not be spoken."""

from __future__ import annotations

from typing import Protocol

from ..schemas import ExpressionParams, TurnPlan


class RulePack(Protocol):
    name: str

    def check(self, sentence: str, plan: TurnPlan, expr: ExpressionParams) -> str | None: ...
