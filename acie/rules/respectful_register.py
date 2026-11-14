"""Respectful register: the agent keeps आप and a polite tone even when the caller is casual (no यार / तू / तेरी / तुम, no 'अरे भाई').
In the casual tone (campaign tone_level) bhai / boss / yaar are allowed; तू / तेरी / तुम / बे / ओए are blocked in every tone.
A product persona that explicitly allows casual speech can switch the pack off by not listing it."""

from __future__ import annotations

import re

from ..schemas import ExpressionParams, TurnPlan

# bhai / boss / yaar are fine ONLY in the casual tone (or when the caller used them and the tone drifted casual); तू/तेरी/तुम etc. are never fine
_CASUAL_OK = {"yaar", "yar", "यार", "भाई", "bhai", "bhaiya", "boss", "बॉस"}
_RX = re.compile(
    r"(?<![\wऀ-ॿ])(यार|तू|तेरी|तेरा|तेरे|तुझे|तुझसे|तुम|तुम्हें|तुम्हारी|तुम्हारा|तुम्हारे|तुमसे|भाई|भाईसाहब|बे|अबे|ओए|"
    r"अरे भाई|सोरी यार|बॉस)(?![\wऀ-ॿ])|"
    r"\b(yaar|yar|boss|tu|teri|tera|tere|tujhe|tujhse|tum|tumhe|tumhein|tumhari|tumhara|tumhare|tumse|bhai|bhaiya|abe|oye|bro|dude|mate)\b", re.I)


class RespectfulRegisterPack:
    name = "respectful_register"

    def check(self, sentence: str, plan: TurnPlan, expr: ExpressionParams) -> str | None:
        casual = (getattr(plan, "tone_level", "") or getattr(expr, "tone", "") or "").lower() == "casual"
        for m in _RX.finditer(sentence):
            if casual and m.group(0).lower() in _CASUAL_OK:
                continue
            return f"disrespectful or over-casual address: {m.group(0)}"
        return None
