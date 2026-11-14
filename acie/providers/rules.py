"""Keyword rules (the domain's classifier) + the lexicon state scorer, as one provider. Always first."""

from __future__ import annotations

from ..schemas import SessionState, TurnInput
from ..signals import PartialSignals
from .base import ProviderBase, SignalGuide


class RulesProvider(ProviderBase):
    name = "rules"
    budget_s = 1.0

    def __init__(self, classifier, scorer=None):
        self.classifier = classifier   # anything with classify(text, session) -> str (a hybrid is fine: legacy mode)
        self.scorer = scorer           # a RawScorer: score(turn, session) -> (dims, evidence)

    async def provide(self, turn: TurnInput, session: SessionState, guide: SignalGuide) -> PartialSignals:
        intent = self.classifier.classify(turn.text, session)
        scores: dict[str, float] = {}
        try:
            raw = self.classifier.scores(turn.text, session)
            tot = sum(v for v in raw.values() if v > 0)
            scores = {k: round(v / tot, 4) for k, v in sorted(raw.items()) if v > 0} if tot > 0 else {}
        except Exception:  # a classifier without scores() simply has no distribution
            pass
        state, ev = {}, []
        if self.scorer is not None:
            state, ev = self.scorer.score(turn, session)
        return PartialSignals(provider="rules", intent=intent, intent_scores=scores, confidence=1.0,
                              state=dict(state), evidence=list(ev))
