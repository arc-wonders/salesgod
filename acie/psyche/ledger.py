"""SaidLedger: everything the agent has already said in this call.

Stored as JSON in `session.dialog["_said"]` (dialog is a str->str dict, so it survives the session
stores). It remembers normalized sentences, intents, fact slots, asks and phrasing rotations, so the
move selector can pick the next *unsaid* thing instead of repeating. Pure Python, deterministic.
"""

from __future__ import annotations

import difflib
import json
import re

KEY = "_said"
SIMILAR = 0.6  # SequenceMatcher ratio above which two sentences count as the same thing
_PUNCT = re.compile(r"[\s\.,!?;:\"'()\[\]।\-]+")
_SPLIT = re.compile(r"(?<=[.!?।])\s+")


def norm(text: str) -> str:
    return _PUNCT.sub(" ", text.lower()).strip()


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SPLIT.split(text.strip()) if s.strip()]


def ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()


class SaidLedger:
    def __init__(self, data: dict | None = None):
        d = data or {}
        self.sentences: list[str] = list(d.get("sentences", []))  # normalized, in order
        self.replies: list[str] = list(d.get("replies", []))      # whole agent replies, normalized
        self.intents: dict[str, int] = dict(d.get("intents", {}))
        self.slots: list[str] = list(d.get("slots", []))
        self.asks: list[str] = list(d.get("asks", []))
        self.moves: dict[str, int] = dict(d.get("moves", {}))
        self.offers: int = int(d.get("offers", 0))
        self.refused: bool = bool(d.get("refused", False))  # the caller said no / later / stop: never offer again

    # -- persistence
    @classmethod
    def load(cls, dialog: dict) -> "SaidLedger":
        raw = dialog.get(KEY)
        try:
            return cls(json.loads(raw)) if raw else cls()
        except (ValueError, TypeError):
            return cls()

    def save(self, dialog: dict) -> None:
        dialog[KEY] = json.dumps({"sentences": self.sentences[-80:], "replies": self.replies[-30:], "intents": self.intents, "slots": self.slots,
                                  "asks": self.asks[-30:], "moves": self.moves, "offers": self.offers,
                                  "refused": self.refused}, ensure_ascii=False, sort_keys=True)

    # -- queries
    def has_slot(self, slot: str) -> bool:
        return slot in self.slots

    def similar_to(self, text: str, thr: float = SIMILAR) -> str | None:
        """The earlier sentence this one repeats (ratio > thr), else None. Whole text and each sentence are checked."""
        n = norm(text)
        if not n:
            return None
        for s in self.sentences:
            if difflib.SequenceMatcher(None, n, s).ratio() > thr:
                return s
        return None

    def repeats(self, text: str, thr: float = SIMILAR) -> bool:
        parts = sentences(text) or [text]
        return any(self.similar_to(p, thr) for p in parts if len(norm(p).split()) >= 3)

    def reply_repeats(self, text: str, thr: float = SIMILAR) -> bool:
        """The whole reply is (almost) a copy of an earlier whole reply: the headline repetition metric."""
        n = norm(text)
        return bool(n) and any(difflib.SequenceMatcher(None, n, r).ratio() > thr for r in self.replies)

    def count_move(self, name: str) -> int:
        return self.moves.get(name, 0)

    # -- updates
    def add_text(self, text: str) -> None:
        if norm(text):
            self.replies.append(norm(text))
        for s in sentences(text):
            n = norm(s)
            if n and n not in self.sentences:
                self.sentences.append(n)
        for s in sentences(text):
            if s.endswith("?"):
                self.asks.append(norm(s))

    def add_intent(self, intent: str) -> None:
        self.intents[intent] = self.intents.get(intent, 0) + 1

    def add_slot(self, slot: str | None) -> None:
        if slot and slot not in self.slots:
            self.slots.append(slot)

    def add_move(self, name: str) -> None:
        self.moves[name] = self.moves.get(name, 0) + 1
