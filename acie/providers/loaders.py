"""Loader registry for trained turn models: whatever is at ACIE_TURN_MODEL becomes a model object.

A loader is `(name, match(path) -> bool, load(path) -> model)`; the first match wins and new ones plug in with
`register_loader`. A model object offers any of: classify_with_confidence(text, session) -> (label, p),
predict(...) (the encoder wrapper), classify(text, session) -> label; and optionally score(turn, session) ->
(dims, evidence). Heavy imports are lazy; any failure means "no model" (the rules decide alone).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger("acie.providers")

Loader = tuple[str, Callable[[str], bool], Callable[[str], Any]]
LOADERS: list[Loader] = []
_CACHE: dict[str, Any] = {}


def register_loader(name: str, match: Callable[[str], bool], load: Callable[[str], Any], first: bool = False) -> None:
    LOADERS[:] = [x for x in LOADERS if x[0] != name]
    LOADERS.insert(0 if first else len(LOADERS), (name, match, load))


def _load_xgboost(path: str):
    from training.state_classifier.predict import TrainedTurnModel  # lazy: xgboost / torch are optional
    return TrainedTurnModel.load(path) if hasattr(TrainedTurnModel, "load") else TrainedTurnModel(path)


def _load_muril(path: str):
    from training.state_classifier.predict_muril import MurilTurnModel  # lazy
    return MurilTurnModel.load(path)


register_loader("muril", lambda p: Path(p, "encoder").is_dir(), _load_muril)
register_loader("xgboost", lambda p: True, _load_xgboost)  # the default: a dir of XGBoost JSON boosters


def load_model(path: str | None = None):
    """The model at `path` / ACIE_TURN_MODEL (cached), or None when unset or loading fails."""
    path = path if path is not None else os.getenv("ACIE_TURN_MODEL", "")
    if not path:
        return None
    if path in _CACHE:
        return _CACHE[path]
    model = None
    for name, match, load in LOADERS:
        try:
            if not match(path):
                continue
            model = load(path)
            break
        except Exception as e:  # missing deps, bad dir, model crash: the rules decide alone
            log.warning("turn model loader %s failed for %s (%s)", name, path, type(e).__name__)
            model = None
    _CACHE[path] = model
    return model


def predict_label(model: Any, text: str, session: Any) -> tuple[str, float, dict[str, float]]:
    """(top label, its probability, {label: probability} when the model exposes a distribution)."""
    if hasattr(model, "classify_with_confidence"):
        label, p = model.classify_with_confidence(text, session)
        return label, float(p), {}
    if hasattr(model, "predict"):
        from training.state_classifier.predict import INTENTS, _session_context  # lazy
        probs = model.predict(text, _session_context(session, None))["intent_probs"]
        k = max(range(len(probs)), key=probs.__getitem__)
        return INTENTS[k], float(probs[k]), {INTENTS[i]: float(p) for i, p in enumerate(probs)}
    return model.classify(text, session), 1.0, {}
