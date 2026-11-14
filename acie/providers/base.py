"""The provider protocol: anything that can look at a turn and say what it heard."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from ..schemas import SessionState, TurnInput
from ..signals import IntentVocabulary, PartialSignals


@dataclass
class SignalGuide:
    """What the domain tells every provider: its vocabulary, what the call is about, what the rules heard."""
    vocab: IntentVocabulary
    about: str = ""
    history: list[tuple[str, str]] = field(default_factory=list)
    rule_intent: str = "unknown"

    @property
    def intents(self) -> dict[str, str]:
        return self.vocab.guide


@runtime_checkable
class SignalProvider(Protocol):
    name: str                          # the key in ACIE_SIGNAL_PROVIDERS and in TurnSignals.sources
    budget_s: float                    # hard time budget; a provider that is late or fails is skipped
    needs_guide: bool                  # only runs when the domain declares an intent guide
    skip_when_rules_sure: bool         # skipped when the rules are already sure (saves a slow call)

    async def provide(self, turn: TurnInput, session: SessionState, guide: SignalGuide) -> PartialSignals | None: ...


class ProviderBase:
    name = "provider"
    budget_s = 3.0
    needs_guide = False
    skip_when_rules_sure = False


def asdict_safe(obj: Any) -> dict:
    return obj.model_dump() if hasattr(obj, "model_dump") else dict(obj)
