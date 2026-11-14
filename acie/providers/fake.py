"""A scripted provider for tests: returns the PartialSignals it was given for each text (or a default)."""

from __future__ import annotations

from ..schemas import SessionState, TurnInput
from ..signals import PartialSignals
from .base import ProviderBase, SignalGuide


class FakeProvider(ProviderBase):
    def __init__(self, table: dict[str, PartialSignals | dict] | None = None, name: str = "fake",
                 default: PartialSignals | dict | None = None, fail: bool = False, delay_s: float = 0.0,
                 budget_s: float = 3.0, needs_guide: bool = False, skip_when_rules_sure: bool = False):
        self.name, self.table, self.default, self.fail, self.delay_s = name, table or {}, default, fail, delay_s
        self.budget_s, self.needs_guide, self.skip_when_rules_sure = budget_s, needs_guide, skip_when_rules_sure
        self.calls = 0

    async def provide(self, turn: TurnInput, session: SessionState, guide: SignalGuide) -> PartialSignals | None:
        import asyncio
        self.calls += 1
        if self.delay_s:
            await asyncio.sleep(self.delay_s)
        if self.fail:
            raise RuntimeError("fake provider down")
        v = self.table.get(turn.text, self.default)
        if v is None:
            return None
        part = v if isinstance(v, PartialSignals) else PartialSignals.model_validate(v)
        return part.model_copy(update={"provider": self.name})
