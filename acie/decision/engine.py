"""Layer 4 - Response Determination Engine.

Domain-agnostic shell. Each business domain is a `DomainPack` that owns its
intent classifier, dialog FSM, backend calls and fallback templates. The
engine only guarantees the output contract: a `TurnPlan`, never prose.
"""

from __future__ import annotations

import re
from typing import Protocol

from ..schemas import SessionState, TurnInput, TurnPlan
from ..text import tokens


class IntentClassifier(Protocol):
    def classify(self, text: str, session: SessionState) -> str: ...


class DomainPack(Protocol):
    name: str
    classifier: IntentClassifier

    async def plan(self, intent: str, turn: TurnInput, session: SessionState) -> TurnPlan: ...


Keyword = str | tuple[str, float]
Priority = str | tuple[str, ...]

# Non-lexical fillers ignored by whole-utterance ("=...") keywords.
_FILLERS = {"um", "umm", "ummm", "uh", "uhh", "uhm", "er", "erm", "ah", "aa", "aaa", "sir", "madam",
            "maam", "mam", "please", "ji", "जी", "सर", "मैडम", "उम", "अं", "आ", "plz"}


class KeywordIntentClassifier:
    """v1 classifier: weighted keyword hits over normalized ASR text. Replace with a
    fine-tuned small encoder (same `classify` signature) once there is labeled data.

    Keyword syntax (keywords are normalized like the text: lowercase, no nukta,
    anusvara for chandrabindu, apostrophes dropped):
      "card block"   whole-word phrase match
      "bhej*"        prefix match on the last word (bhej, bhejna, bhejo ...)
      "=matlab"      the whole utterance (fillers like um/sir/ji removed) is exactly this
      ("ok", 0.5)    weighted keyword (default weight 1.0)
    `masks[intent]` lists phrases ("x*" = prefix allowed) removed from the text before scoring that intent
    ("koi baat nahi" is not a deny; "debit card" is not a card offer).
    `priority` entries (an intent or a tuple of competing intents) are checked in
    order; inside a tuple the highest score wins, ties go to the earlier intent.
    A *gated* priority intent (e.g. affirm/deny) only wins if its score is at least
    the best open intent's score, so a filler "haan" does not swallow
    "haan mera balance kitna hai".
    """

    def __init__(self, keywords: dict[str, list[Keyword]], gated: dict[str, str] | None = None,
                 priority: list[Priority] | None = None, masks: dict[str, list[str]] | None = None,
                 weak: list[str] | None = None):
        # weak intents (greetings, thanks) only win when nothing else matched:
        # "नमस्ते, मुझे पचास हज़ार ट्रांसफर करने हैं" is a transfer, not a greeting
        self.weak = set(weak or [])
        self.gated = gated or {}  # intent -> dialog key that must be present for it to fire
        self.priority = [p if isinstance(p, tuple) else (p,) for p in (priority or [])]
        self.keywords = {i: [self._compile(k) for k in ks] for i, ks in keywords.items()}
        self.masks = {i: [self._compile_mask(m) for m in sorted(ms, key=len, reverse=True)]
                      for i, ms in (masks or {}).items()}

    @staticmethod
    def _compile_mask(m: str) -> re.Pattern[str]:
        m = m.strip()
        stem = m.endswith("*")
        body = re.escape(" ".join(tokens(m.rstrip("*"))))
        return re.compile(r"(?<= )" + body + (r"\S*" if stem else "") + r"(?= )")

    @staticmethod
    def _compile(k: Keyword) -> tuple[str, str, float]:
        word, weight = (k, 1.0) if isinstance(k, str) else k
        word = word.strip()
        if word.startswith("="):
            return "exact", " ".join(tokens(word[1:])), weight
        if word.endswith("*"):
            return "prefix", " " + " ".join(tokens(word[:-1])), weight
        return "word", " " + " ".join(tokens(word)) + " ", weight

    def _score(self, intent: str, low: str, bare: str) -> float:
        for m in self.masks.get(intent, ()):
            low = m.sub("|", low)
        total = 0.0
        for kind, k, w in self.keywords[intent]:
            if (kind == "exact" and bare == k) or (kind != "exact" and k in low):
                total += w
        return total

    def scores(self, text: str, session: SessionState) -> dict[str, float]:
        toks = tokens(text)
        low = " " + " ".join(toks) + " "
        bare = " ".join(t for t in toks if t not in _FILLERS)
        return {i: self._score(i, low, bare) for i in self.keywords
                if i not in self.gated or self.gated[i] in session.dialog}

    def classify(self, text: str, session: SessionState) -> str:
        sc = self.scores(text, session)
        in_priority = {i for group in self.priority for i in group}
        best_open = max((v for i, v in sc.items() if i not in in_priority), default=0.0)
        for group in self.priority:
            cands = [(sc[i], -n, i) for n, i in enumerate(group) if sc.get(i, 0) > 0]
            if not cands:
                continue
            h, _, intent = max(cands)
            if intent in self.gated and h < best_open:
                continue
            return intent
        best, best_hits = "unknown", 0.0
        strong = {i: h for i, h in sc.items() if i not in self.weak and h > 0}
        for intent, h in (strong or sc).items():
            if h > best_hits:
                best, best_hits = intent, h
        return best


class DecisionEngine:
    def __init__(self, domain: DomainPack):
        self.domain = domain

    def classify(self, turn: TurnInput, session: SessionState) -> str:
        return self.domain.classifier.classify(turn.text, session)

    async def decide(self, intent: str, turn: TurnInput, session: SessionState) -> TurnPlan:
        plan = await self.domain.plan(intent, turn, session)
        if plan.intent != "unknown":  # kept so "samajh nahi aaya" can re-explain the last answer
            session.dialog["last_plan"] = plan.model_dump_json()
        return plan
