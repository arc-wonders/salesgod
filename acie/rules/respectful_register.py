"""Respectful register: the agent keeps आप and a polite tone even when the caller is casual (no यार / तू / तेरी / तुम, no 'अरे भाई').
A product persona that explicitly allows casual speech can switch the pack off by not listing it."""

from __future__ import annotations

import re

from ..schemas import ExpressionParams, TurnPlan

_RX = re.compile(
    r"(?<![\wऀ-ॿ])(यार|तू|तेरी|तेरा|तेरे|तुझे|तुझसे|तुम|तुम्हें|तुम्हारी|तुम्हारा|तुम्हारे|तुमसे|भाई|भाईसाहब|बे|अबे|ओए|"
    r"अरे भाई|सोरी यार)(?![\wऀ-ॿ])|"
    r"\b(yaar|yar|tu|teri|tera|tere|tujhe|tujhse|tum|tumhe|tumhein|tumhari|tumhara|tumhare|tumse|bhai|bhaiya|abe|oye|bro|dude|mate)\b", re.I)


class RespectfulRegisterPack:
    name = "respectful_register"

    def check(self, sentence: str, plan: TurnPlan, expr: ExpressionParams) -> str | None:
        m = _RX.search(sentence)
        return f"disrespectful or over-casual address: {m.group(0)}" if m else None
