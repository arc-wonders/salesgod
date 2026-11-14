from __future__ import annotations

import re

from ..schemas import ExpressionParams, TurnPlan

# textbook / very formal words nobody says on a casual phone call (Devanagari and Roman)
DEFAULT_FORMAL_WORDS = (
    "अतः", "तथा", "अथवा", "किंतु", "परंतु", "पश्चात", "पश्चात्", "उपरांत", "प्रयोग", "प्रदान", "आवश्यकता",
    "धनराशि", "सम्पर्क", "संपर्क", "निवेदन", "कृपया", "प्रतीक्षा", "सूचित", "अनुरोध", "उपलब्ध", "समस्या का समाधान",
    "kripya", "tatha", "athva", "athwa", "kintu", "paschat", "uprant", "prayog", "pradan", "avashyakta",
    "dhanrashi", "nivedan", "pratiksha", "suchit", "anurodh",
)


class HindiRegisterPack:
    """Flags formal Hindi words in casual registers. `formal_words` and `casual_levels` are configurable."""

    name = "hindi_register"

    def __init__(self, formal_words: tuple[str, ...] | list[str] | None = None,
                 casual_levels: tuple[str, ...] = ("low",), languages: tuple[str, ...] = ("hi", "hinglish")):
        words = tuple(formal_words) if formal_words is not None else DEFAULT_FORMAL_WORDS
        self.casual_levels, self.languages = casual_levels, languages
        self._rx = re.compile("|".join(
            (re.escape(w) if re.search(r"[ऀ-ॿ]", w) else rf"\b{re.escape(w)}\b") for w in words),
            re.I) if words else None

    def check(self, sentence: str, plan: TurnPlan, expr: ExpressionParams) -> str | None:
        if not self._rx or expr.language not in self.languages or expr.formality not in self.casual_levels:
            return None
        m = self._rx.search(sentence)
        return f"too formal for a casual register: {m.group(0)}" if m else None
