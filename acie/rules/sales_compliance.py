from __future__ import annotations

import re

from ..schemas import ExpressionParams, TurnPlan

_RULES: list[tuple[str, re.Pattern]] = [
    ("guarantee", re.compile(
        r"\bguarantee[sd]?\b|\bassured (results?|returns?|profit)\b|100\s*%\s*(safe|sure|certain|secure|working)"
        r"|गारंटी|गारण्टी|\bgaranti\b", re.I)),
    ("false urgency", re.compile(
        r"limited[- ]time( only)?|\bonly (today|tonight|for today)\b|\blast chance\b|hurry( up)?\b|"
        r"offer (ends|expires)|\bact now\b|\bwhile (stocks|seats) last\b|सीमित समय|सिर्फ़? आज|आज ही (ले|खरीद|बुक)|"
        r"जल्दी कीजिए|\bsirf aaj\b|\baaj hi (le|lijiye|book)|jaldi kijiye|offer khatam", re.I)),
    ("pressure (you must)", re.compile(
        r"\byou (must|have to|need to buy|should buy)\b|\byou('| a)?re required\b|आपको .{0,25}(करना ही होगा|लेना ही होगा)|"
        r"\baapko .{0,25}(karna hi hoga|lena hi hoga)|\bzaroor lena padega", re.I)),
    ("health or financial promise", re.compile(
        r"\b(cures?|heals?|risk[- ]free|no risk|zero risk|double your money|guaranteed (returns?|income|profit)|"
        r"you will (earn|make|get rich|lose weight)|get rich|permanent relief)\b|बिना जोखिम|पैसे दोगुने|"
        r"\bpaise (double|dugne)|\bbina risk\b", re.I)),
    ("competitor disparagement", re.compile(
        r"\b(competitors?|other (apps?|companies|providers|banks?|brands?)|they|others)\b.{0,40}"
        r"\b(scams?|fraud(ulent)?|cheats?|cheating|useless|worthless|terrible|rubbish|fake)\b|"
        r"\b(worse|worst) than\b|दूसरी कंपनियाँ? .{0,30}(धोखा|फ्रॉड|बेकार)|\bdusre log .{0,30}(fraud|bekaar|dhokha)", re.I)),
]


class SalesCompliancePack:
    name = "sales_compliance"

    def check(self, sentence: str, plan: TurnPlan, expr: ExpressionParams) -> str | None:
        for reason, rx in _RULES:
            if rx.search(sentence):
                return reason
        return None
