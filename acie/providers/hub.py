"""Runs the configured providers for one turn and fuses their output (acie.signals_fusion)."""

from __future__ import annotations

import asyncio
import logging
import os

from ..schemas import SessionState, TurnInput
from ..signals import PartialSignals, TurnSignals, build_vocabulary
from ..signals_fusion import fuse
from .base import SignalGuide

log = logging.getLogger("acie.providers")


def provider_names() -> list[str] | None:
    """ACIE_SIGNAL_PROVIDERS as an ordered list, or None when unset (= today's behaviour)."""
    raw = os.getenv("ACIE_SIGNAL_PROVIDERS", "").strip()
    if not raw:
        return None
    return list(dict.fromkeys(n.strip().lower() for n in raw.split(",") if n.strip()))


class SignalHub:
    def __init__(self, providers: list, config: dict | None = None):
        # rules always go first: they are the base the others are fused onto
        self.providers = sorted(providers, key=lambda p: 0 if p.name == "rules" else 1)
        self.config = config

    @property
    def names(self) -> list[str]:
        return [p.name for p in self.providers]

    async def _one(self, p, turn, session, guide) -> PartialSignals | None:
        try:
            part = await asyncio.wait_for(p.provide(turn, session, guide), p.budget_s)
        except Exception as e:  # late, down, bad payload: skipped; the others still decide
            log.info("signal provider %s skipped: %s", p.name, type(e).__name__)
            return None
        if part is not None and part.provider != p.name:
            part = part.model_copy(update={"provider": p.name})
        return part

    async def run(self, turn: TurnInput, session: SessionState, domain, classifier=None) -> TurnSignals:
        vocab = build_vocabulary(domain, classifier)
        guide = SignalGuide(vocab=vocab, about=getattr(domain, "about", ""),
                            history=[(u.role, u.text) for u in session.history])
        rules = next((p for p in self.providers if p.name == "rules"), None)
        parts: list[PartialSignals | None] = []
        if rules is not None:
            rp = await rules.provide(turn, session, guide)  # the rules are never skipped (and never swallowed)
            guide.rule_intent = rp.intent or "unknown"
            parts.append(rp)
        sure = guide.rule_intent in getattr(domain, "confident_intents", ()) and len(turn.text.split()) <= 6
        runnable = [p for p in self.providers if p.name != "rules" and turn.text.strip()
                    and not (p.needs_guide and not vocab.guide) and not (p.skip_when_rules_sure and sure)]
        parts += list(await asyncio.gather(*[self._one(p, turn, session, guide) for p in runnable]))
        return fuse([p for p in parts if p is not None], vocab, session.dialog, self.config)
