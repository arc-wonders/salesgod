"""The model-neutral contract between "what was heard" and "what to do" (Layer 4 reads ONLY this).

Whatever produced the signals (keyword rules, an LLM, a trained classifier, an HTTP service) the decision
layer, the psyche (needs / moves) and the playbook see one object: `TurnSignals`. Providers return a
`PartialSignals` (only the fields they know); `acie.signals_fusion.fuse` combines them deterministically.

Labels are declared by the DOMAIN (`IntentVocabulary`): providers map their own label sets onto it through an
alias table, and anything the domain does not know becomes "unknown". No provider label reaches the domain.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# the 14 human-state dims (same names as acie.schemas.HumanState); the first four are always read from a learned provider
STATE_DIMS = ("frustration", "confusion", "urgency", "engagement", "satisfaction", "anxiety", "excitement", "calmness",
              "trust", "certainty", "cognitive_load", "attention", "participation", "receptiveness")
DEFAULT_LEARNED = {"frustration": 0.0, "confusion": 0.0, "urgency": 0.0, "engagement": 0.5}

# "the line was bad / the words are not what they meant": kept here so every provider is judged the same way
GARBLED_GIST = re.compile(r"unclear|unrelated|garbled|nonsens|not clear|incomprehensible", re.I)


def _norm(label: Any) -> str:
    return re.sub(r"[\s\-]+", "_", str(label or "").strip().lower())


class IntentVocabulary(BaseModel):
    """The intent labels one domain understands, and how foreign labels map onto them."""
    labels: list[str] = Field(default_factory=list)
    guide: dict[str, str] = Field(default_factory=dict)       # label -> one-line meaning (shown to LLM providers)
    aliases: dict[str, str] = Field(default_factory=dict)     # foreign label -> domain label
    rule_first: list[str] = Field(default_factory=list)       # intents the rules decide before any model
    gated: dict[str, str] = Field(default_factory=dict)       # intent -> dialog key that must exist

    def map(self, label: Any) -> str:
        """The domain's label for a provider's label; "unknown" when it has none."""
        lab = _norm(label)
        if not lab or lab == "unknown":
            return "unknown"
        known = {_norm(x): x for x in self.labels}
        if lab in known:
            return known[lab]
        al = {_norm(k): v for k, v in self.aliases.items()}
        tgt = al.get(lab)
        if tgt is not None and _norm(tgt) in known:
            return known[_norm(tgt)]
        return "unknown"

    def map_scores(self, scores: dict[str, float]) -> dict[str, float]:
        out: dict[str, float] = {}
        for k in sorted(scores):
            m = self.map(k)
            if m != "unknown":
                out[m] = max(out.get(m, 0.0), float(scores[k]))
        return out


class PartialSignals(BaseModel):
    """What ONE provider knows about the turn. None / empty = "I do not know"; the fusion skips it."""
    model_config = ConfigDict(extra="ignore", populate_by_name=True)
    provider: str = ""
    intent: str | None = None
    intent_scores: dict[str, float] = Field(default_factory=dict)
    confidence: float | None = None
    objection: str | None = None
    question_topic: str | None = None
    buying_interest: float | None = Field(None, ge=0, le=1, alias="buying_signal")
    garbled: bool | None = None
    garbled_reason: str = ""
    gist: str = ""
    state: dict[str, float] = Field(default_factory=dict)
    evidence: list[str] = Field(default_factory=list)


class TurnSignals(BaseModel):
    """The fused, model-neutral reading of one customer turn."""
    model_config = ConfigDict(extra="ignore")
    intent: str = "unknown"                       # top intent, in the domain's vocabulary
    intent_scores: dict[str, float] = Field(default_factory=dict)
    confidence: float = 0.0                       # confidence of the provider the top intent came from
    objection: str = "none"                       # trust | price | need | timing | language | other | none
    question_topic: str = "none"                  # product | offer_terms | general | conversation | none
    buying_interest: float = Field(0.0, ge=0, le=1)
    garbled: bool = False
    garbled_reason: str = ""
    gist: str = ""
    state: dict[str, float] = Field(default_factory=dict)          # the 14 dims, rules and learned fused
    learned_state: dict[str, float] = Field(default_factory=dict)  # dims read by learned providers alone
    evidence: list[str] = Field(default_factory=list)
    sources: dict[str, str] = Field(default_factory=dict)          # field -> provider that produced it
    votes: dict[str, str] = Field(default_factory=dict)            # provider -> its own (mapped) top intent
    rule_intent: str = "unknown"
    learned: bool = False                         # at least one non-rules provider contributed

    # ---- compat names (the old NLU object) and helpers for decision code
    @property
    def buying_signal(self) -> float:
        return self.buying_interest

    def _ls(self, dim: str) -> float:
        return self.learned_state.get(dim, DEFAULT_LEARNED.get(dim, 0.5))

    frustration = property(lambda s: s._ls("frustration"))
    confusion = property(lambda s: s._ls("confusion"))
    urgency = property(lambda s: s._ls("urgency"))
    engagement = property(lambda s: s._ls("engagement"))

    def heard(self, *labels: str) -> bool:
        """True when a learned (non-rules) provider's own top intent is one of `labels`."""
        return any(v in labels for v in self.votes.values())

    def to_understanding(self):
        """The legacy NLU object (for custom estimators and the console's `nlu` field)."""
        from .understanding import Understanding
        return Understanding(**self.legacy_nlu())

    def legacy_nlu(self) -> dict:
        d = {"intent": next(iter(self.votes.values()), self.intent), "buying_signal": self.buying_interest,
             "question_topic": self.question_topic, "objection": self.objection, "gist": self.gist}
        for k, v in self.learned_state.items():
            d[k] = v
        return d


def as_signals(obj: Any) -> TurnSignals | None:
    """TurnSignals from a TurnSignals, a legacy Understanding / any duck with its fields, or None."""
    if obj is None or isinstance(obj, TurnSignals):
        return obj
    g = lambda k, d=None: getattr(obj, k, d)  # noqa: E731
    ls = {k: float(g(k)) for k in STATE_DIMS if g(k) is not None}
    for k, v in DEFAULT_LEARNED.items():
        ls.setdefault(k, v)
    gist = g("gist", "") or ""
    intent = g("intent", "unknown") or "unknown"
    return TurnSignals(intent=intent, intent_scores={intent: 1.0} if intent != "unknown" else {}, confidence=1.0,
                       objection=g("objection", "none") or "none", question_topic=g("question_topic", "none") or "none",
                       buying_interest=float(g("buying_signal", 0.0) or 0.0), gist=gist,
                       garbled=bool(GARBLED_GIST.search(gist)), learned_state=ls, state=dict(ls),
                       votes={"llm": intent}, sources={"intent": "llm"}, learned=True)


def signals_from_dialog(dialog: dict) -> TurnSignals | None:
    """The learned-provider signals the pipeline stored for this turn (None = the rules spoke alone)."""
    raw = dialog.get("_signals")
    if raw:
        sig = TurnSignals.model_validate_json(raw)
        return sig if sig.learned else None
    legacy = dialog.get("_nlu")  # logs / consoles that only wrote the old NLU JSON
    if legacy:
        from .understanding import Understanding
        return as_signals(Understanding.model_validate_json(legacy))
    return None


DEFAULT_ALIASES = {
    "yes": "affirm", "agree": "affirm", "accept": "affirm", "no": "deny", "refuse": "deny", "reject": "deny",
    "decline": "deny", "repeat": "clarify", "not_understood": "clarify", "human": "human_agent",
    "live_agent": "human_agent", "agent": "human_agent", "call_back": "callback", "later": "callback",
    "bye": "goodbye", "pricing": "price", "cost": "price", "features": "benefits", "chitchat": "small_talk",
    "chit_chat": "small_talk",
}


def build_vocabulary(domain: Any, classifier: Any = None) -> IntentVocabulary:
    """The domain's declared vocabulary: its intent guide + keyword intents + alias table (+ env extras)."""
    import os
    guide = dict(getattr(domain, "intent_guide", None) or {})
    cls = classifier if classifier is not None else getattr(domain, "classifier", None)
    kw = getattr(cls, "keyword", cls)  # a hybrid wraps the keyword classifier
    labels = list(dict.fromkeys(list(getattr(kw, "keywords", {}) or {}) + list(guide)))
    gated = dict(getattr(kw, "gated", {}) or {})
    rule_first = sorted({i for g in (getattr(kw, "priority", []) or []) for i in g if i not in gated})
    aliases = {**DEFAULT_ALIASES, **dict(getattr(domain, "intent_aliases", {}) or {})}
    try:
        aliases.update(json.loads(os.getenv("ACIE_INTENT_ALIASES", "") or "{}"))
    except ValueError:
        pass
    return IntentVocabulary(labels=labels, guide=guide, aliases=aliases, rule_first=rule_first, gated=gated)
