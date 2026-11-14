"""Reply length budget for the phone: the same rule the expression eval scores (training/expression/checks.length_issue).

The tuned reply model writes one sentence too many in ~15% of turns (8B vs Gemini eval, 15 Nov 2026), mostly an empty
opener ("Aapka sawal samajh gaya.", "I'm sorry, I didn't quite get that.") before the real answer. Live replies are
buffered, so every extra sentence is also extra silence before the caller hears anything.

  budget(plan, expr)          -> (max_sentences, max_words)
  fit(sentences, plan, expr)  -> sentences that fit: drop filler first, never a sentence carrying a fact/number, keep
                                 the closing question
  enough(sentences, plan, expr) -> True when generation can stop early (the reply is complete and at budget)
"""

from __future__ import annotations

import re

_WORD = re.compile(r"[\wऀ-ॣॱ-ॿ{}₹]+", re.UNICODE)
_FACTY = re.compile(r"[{}₹\d]|रुपय|rupay|rupee", re.I)
_SPLIT = re.compile(r"(?<=[.!?।])\s+")
LANG_WORDS = {"en": 1.0, "hi": 1.3, "hinglish": 1.15}


def words(s: str) -> list[str]:
    return _WORD.findall(s)


def split(sentences: list[str]) -> list[str]:
    return [p for s in sentences for p in _SPLIT.split(s.strip()) if p.strip()]


def _counts(s: str) -> bool:
    return len(words(s)) > 2  # "Sure." / "जी।" do not count as sentences (same as the eval)


def budget(plan, expr) -> tuple[int, int]:
    ask = bool(getattr(plan, "ask", None))
    ack = bool(getattr(expr, "acknowledge_emotion", False))
    verb = getattr(expr, "verbosity", "med")
    max_s = {"low": 1, "med": 2, "high": 4}.get(verb, 2) + ask + ack
    nfacts = len(getattr(plan, "facts", {}) or {})
    max_w = {"low": 17, "med": 38, "high": 75}.get(verb, 38) + 10 * ask + 9 * ack + 3 * max(0, nfacts - 1)
    return max_s, int(max_w * LANG_WORDS.get(getattr(expr, "language", "en"), 1.0))


def _n_sent(ss: list[str]) -> int:
    return sum(1 for s in ss if _counts(s))


def _n_words(ss: list[str]) -> int:
    return sum(len(words(s)) for s in ss)


def fit(sentences: list[str], plan, expr) -> list[str]:
    ss = split(sentences)
    max_s, max_w = budget(plan, expr)
    if _n_sent(ss) <= max_s and _n_words(ss) <= max_w:
        return ss
    ask = ss[-1] if len(ss) > 1 and ss[-1].rstrip().endswith("?") else None
    body = ss[:-1] if ask else list(ss)

    def total(b):
        return b + ([ask] if ask else [])

    # 1. empty openers first: a short leading sentence with no fact in it
    while len(body) > 1 and (_n_sent(total(body)) > max_s or _n_words(total(body)) > max_w) \
            and len(words(body[0])) <= 8 and not _FACTY.search(body[0]):
        body.pop(0)
    # 2. then from the end of the body (before the closing question), never a sentence carrying a fact
    i = len(body) - 1
    while i > 0 and (_n_sent(total(body)) > max_s or _n_words(total(body)) > max_w):
        if not _FACTY.search(body[i]):
            body.pop(i)
        i -= 1
    return total(body)


def enough(sentences: list[str], plan, expr) -> bool:
    """Stop generating: the reply already ends with its question and has reached the sentence budget, or it has run
    well past the budget (a runaway reply; fit() trims it)."""
    ss = split(sentences)
    if not ss:
        return False
    max_s, max_w = budget(plan, expr)
    n = _n_sent(ss)
    if getattr(plan, "ask", None) and ss[-1].rstrip().endswith("?") and n >= max_s:
        return True
    return n >= max_s + 2 or _n_words(ss) >= max_w + 25
