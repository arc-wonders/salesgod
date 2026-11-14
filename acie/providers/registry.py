"""Provider names -> provider objects. `ACIE_SIGNAL_PROVIDERS=rules,llm` (unset = rules, plus llm when the
pipeline was given an NLU, exactly as before). Add your own with `register_provider(name, factory)`; a factory
gets the pipeline and returns a provider, or None when it is not configured (it is then skipped)."""

from __future__ import annotations

import logging
import os
from typing import Any, Callable

from .hub import SignalHub, provider_names
from .rules import RulesProvider

log = logging.getLogger("acie.providers")
FACTORIES: dict[str, Callable[[Any], Any]] = {}


def register_provider(name: str, factory: Callable[[Any], Any]) -> None:
    FACTORIES[name.lower()] = factory


def _rules(pipe):
    names = provider_names()
    cls = pipe.decision.domain.classifier
    scorer = getattr(pipe.estimator, "scorer", None)
    if names is not None:  # explicit list: the rules are the plain rules, models come in as their own providers
        cls = getattr(cls, "keyword", cls)
        scorer = getattr(scorer, "rules", scorer)
        return RulesProvider(cls, scorer)
    return RulesProvider(pipe.decision.domain.classifier, scorer)  # legacy: whatever the domain / estimator wired


def _llm(pipe):
    from .llm import LLMProvider
    nlu = pipe.nlu
    if nlu is None:
        from ..understanding import LLMUnderstanding  # configured by env (ACIE_LLM_BASE_URL / ACIE_LLM_MODEL)
        nlu = LLMUnderstanding()
    return LLMProvider(nlu)


def _model(pipe):
    from .loaders import load_model
    from .trained import TrainedProvider
    m = load_model()
    return TrainedProvider(model=m) if m is not None else None


def _http(pipe):
    from .http import HttpProvider
    return HttpProvider() if os.getenv("ACIE_SIGNAL_HTTP_URL") else None


register_provider("rules", _rules)
register_provider("llm", _llm)
register_provider("model", _model)
register_provider("trained", _model)
register_provider("http", _http)


def build_hub(pipe, config: dict | None = None) -> SignalHub:
    names = provider_names()
    if names is None:
        names = ["rules"] + (["llm"] if pipe.nlu is not None else [])
    if "rules" not in names:
        names = ["rules"] + names
    providers = []
    for n in names:
        f = FACTORIES.get(n)
        if f is None:
            log.warning("unknown signal provider %r ignored", n)
            continue
        p = f(pipe)
        if p is not None:
            providers.append(p)
    return SignalHub(providers, config)
