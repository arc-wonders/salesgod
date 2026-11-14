"""A trained classifier (whatever ACIE_TURN_MODEL points at, via the loader registry) as a provider."""

from __future__ import annotations

from ..schemas import SessionState, TurnInput
from ..signals import PartialSignals
from .base import ProviderBase, SignalGuide
from .loaders import load_model, predict_label


class TrainedProvider(ProviderBase):
    name = "model"
    budget_s = 1.0

    def __init__(self, model=None, path: str | None = None):
        self._model, self._path = model, path

    @property
    def model(self):
        if self._model is None:
            self._model = load_model(self._path)
        return self._model

    async def provide(self, turn: TurnInput, session: SessionState, guide: SignalGuide) -> PartialSignals | None:
        m = self.model
        if m is None or not turn.text.strip():
            return None
        label, p, scores = predict_label(m, turn.text, session)
        state, ev = {}, []
        if hasattr(m, "score"):
            try:
                state, ev = m.score(turn, session)
                ev = list(ev) + ["turn model"]
                state = {k: v for k, v in state.items() if v is not None}
            except Exception:  # the intent may still be usable
                state, ev = {}, []
        return PartialSignals(provider="model", intent=label, intent_scores=scores, confidence=p, state=state, evidence=ev)
