"""Hybrid intent classifier: trained model when it is confident, keyword rules otherwise.

Turned on by ACIE_TURN_MODEL=<dir> (the ml lane's `TrainedTurnModel`, imported lazily). The keyword
classifier keeps the last word where rules must not be second-guessed:

  * its hard-priority intents (clarify, human_agent, callback ...) always win,
  * affirm / deny are only accepted from the model when a question is pending (same gate as the rules),
  * an intent the domain does not know, or "unknown", falls back to the keywords.

Any error (missing deps, bad dir, model crash) silently means keywords. A model object may offer
`classify_with_confidence(text, session) -> (label, p)`; a TrainedTurnModel offers `predict`, from
which the top probability is read; a bare `classify` is trusted at confidence 1.0.
"""

from __future__ import annotations

import os

from ..schemas import SessionState
from .engine import KeywordIntentClassifier

DEFAULT_THRESHOLD = 0.6
_MODEL_CACHE: dict[str, object] = {}


def load_turn_model(path: str | None = None):
    """The model at ACIE_TURN_MODEL (cached), or None when unset or loading fails."""
    path = path if path is not None else os.getenv("ACIE_TURN_MODEL", "")
    if not path:
        return None
    if path in _MODEL_CACHE:
        return _MODEL_CACHE[path]
    try:
        from training.state_classifier.predict import TrainedTurnModel  # lazy: torch is optional
        model = TrainedTurnModel.load(path) if hasattr(TrainedTurnModel, "load") else TrainedTurnModel(path)
    except Exception:
        model = None
    _MODEL_CACHE[path] = model
    return model


def _threshold() -> float:
    try:
        return float(os.getenv("ACIE_TURN_MODEL_THRESHOLD", DEFAULT_THRESHOLD))
    except ValueError:
        return DEFAULT_THRESHOLD


class HybridIntentClassifier:
    def __init__(self, keyword: KeywordIntentClassifier, model, threshold: float | None = None,
                 known: set[str] | None = None):
        self.keyword = keyword
        self.model = model
        self.threshold = _threshold() if threshold is None else threshold
        self.known = known if known is not None else set(keyword.keywords)
        # intents the rules decide before any model: priority groups that are not gated (clarify, callback ...)
        self.rule_first = {i for g in keyword.priority for i in g if i not in keyword.gated}

    # keep the keyword classifier's public surface
    @property
    def keywords(self):
        return self.keyword.keywords

    def scores(self, text: str, session: SessionState) -> dict[str, float]:
        return self.keyword.scores(text, session)

    def _predict(self, text: str, session: SessionState) -> tuple[str, float]:
        m = self.model
        if hasattr(m, "classify_with_confidence"):
            label, p = m.classify_with_confidence(text, session)
            return label, float(p)
        if hasattr(m, "predict"):
            from training.state_classifier.predict import _session_context, INTENTS  # lazy
            probs = m.predict(text, _session_context(session, None))["intent_probs"]
            k = max(range(len(probs)), key=probs.__getitem__)
            return INTENTS[k], float(probs[k])
        return m.classify(text, session), 1.0

    def classify(self, text: str, session: SessionState) -> str:
        kw = self.keyword.classify(text, session)
        if self.model is None or not text.strip() or kw in self.rule_first:
            return kw
        try:
            label, conf = self._predict(text, session)
        except Exception:
            return kw
        if conf < self.threshold or label in ("unknown", "", None) or label not in self.known:
            return kw
        gate = self.keyword.gated.get(label)
        if gate and gate not in session.dialog:
            return kw
        return label
