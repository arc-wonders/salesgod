"""Layer 7 - Response Validation (streaming).

Validates the LLM output sentence by sentence so TTS can start on the first
good sentence instead of waiting for the full reply.

Per sentence:
  - placeholders must be facts from the plan (no invented slots)
  - no digits / currency written by the LLM itself (all numbers come from slots)
  - then slots are replaced with the real fact values
At the end:
  - every must_mention fact must have appeared, else the safe template is used
"""

from __future__ import annotations

import re

from .expression import render_fallback
from .rules import load_packs
from .text import detect_language
from .schemas import ExpressionParams, TurnPlan, ValidationReport
from .psyche.ledger import invented_memory, unsourced_claims

_SLOT = re.compile(r"\{(\w+)\}")
_RAW_NUMBER = re.compile(r"[0-9०-९]|₹|\brs\.?\s|\binr\b", re.IGNORECASE)
# Amounts spelled out in words ("दस हजार रुपए", "ten thousand") are also numbers the LLM invented.
_NUMBER_WORDS = re.compile(
    r"\b(hundred|thousand|lakh|lakhs|lac|crore|million|billion|hazaar|hazar|rupees?|rupaye)\b"
    r"|हजार|हज़ार|लाख|करोड़|करोड|रुपए|रुपये|रुपया", re.IGNORECASE)
_DEVA = re.compile(r"[\u0900-\u097F]")
_LATIN_WORD = re.compile(r"[A-Za-z]{2,}")
_SENT_END = re.compile(r"(?<=[.!?।])\s+|\n+")
_MARKDOWN = re.compile(r"[*#`>]+")
_EMOJI = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F]")
_CURRENCY_BEFORE_SLOT = re.compile(r"(₹|rs\.?|inr)\s*(?=\{)", re.IGNORECASE)  # values already carry ₹


_OFFER_PAD = re.compile(
    r"(for the first (3|three) months,? (at |of )?|(पहले )?(3|तीन) महीन(े|ों) (में |के लिए |तक )?|(pehle )?(3|teen) mahine (mein |ke liye |tak )?)(?=\{offer\})"
    r"|(?<=\{offer\})( for the first (3|three) months| (पहले )?(3|तीन) महीन(े|ों) (के लिए|तक))", re.IGNORECASE)


# "अगर आप इस हफ़्ते join करते हैं तो {offer}": the offer phrase already says "if you join this week" (persona test: said twice in one sentence)
_THIS_WEEK_PAD = re.compile(
    r"(अगर आप |यदि आप |if you |agar aap )?(इस (हफ्ते|हफ़्ते|सप्ताह)|this week|is hafte) ?(join|जॉइन|जुड़ते|जुड़ें|जुड़ने|karte|करते|करें|karein|karne|करने)( करते| करने| karte| karne)?( हैं| hain| पर| par)?,? ?(आपको |aapko |you get |you'll get )?(तो |to |then )?(?=\{offer\})",
    re.IGNORECASE)


# problem text (as returned by _check) -> machine-readable code; first matching prefix wins
_CODES = (("invented placeholder", "invented_slot"), ("price repeated", "price_repeated"),
          ("offer phrase repeated", "offer_repeated"), ("offer phrase padded", "offer_padded"),
          ("money written", "money_in_general_answer"), ("raw number", "raw_number"),
          ("wrong language", "wrong_script"), ("rule pack", "rulepack"), ("invented memory", "invented_memory"),
          ("unsourced claim", "unsourced_claim"), ("repeated sentence", "repeated_sentence"), ("restated fact", "restated_fact"),
          ("re-asked question", "reasked_question"), ("repeated opener", "repeated_opener"))


class StreamingValidator:
    def __init__(self, plan: TurnPlan, expr: ExpressionParams, trusted: bool = False, customer_text: str = "",
                 rule_packs=None, known=None):
        # known = (SaidLedger, CustomerFacts): the call so far. A generation that repeats a sentence we already said, says the caller's own
        # facts back, or re-asks a question they answered fails like any other check (the pipeline regenerates once, then uses the template)
        self.known = known
        # rule packs: explicit list (names or objects) or the plan's own; none = unchanged behaviour
        self.rule_packs = load_packs(rule_packs if rule_packs is not None else getattr(plan, "rule_packs", []))
        self.issue_codes: list[str] = []
        # trusted: operator-written template text (llm_allowed=False) - "first 3 months" in a reviewed
        # FAQ answer is not an LLM inventing a number (live call 194d15aef7 spoke only "Fair question.")
        self.trusted = trusted
        # numbers the customer said ("call me at 7", "8 dollars?") or the operator's background states
        # ("founded in 2021", "about ₹700") may be repeated; prices still come only through slots
        self.said = customer_text + " " + (plan.context or "") + " " + (plan.reference or "")
        self.plan = plan
        self.expr = expr
        self.buf = ""
        self.emitted: list[str] = []
        self.mentioned: set[str] = set()
        self.issues: list[str] = []
        self.failed = False
        self.used_fallback = False

    # ------------------------------------------------------------- public

    def feed(self, delta: str) -> list[str]:
        if self.failed:
            return []
        self.buf += delta
        parts = _SENT_END.split(self.buf)
        self.buf = parts.pop()  # last part may be incomplete
        return self._accept(parts)

    def finish(self) -> list[str]:
        tail = self.buf.strip()
        if tail and self.emitted and not re.search(r"[.!?।]$", tail):
            # cut off mid-sentence (max_tokens): speaking half a sentence sounds broken, drop it
            self.add_issue(f"dropped unfinished sentence: {tail[:40]!r}", "unfinished")
            tail = ""
        out = [] if self.failed else self._accept([tail])
        self.buf = ""
        spoken = " ".join(self.emitted)
        # a value copied verbatim (e.g. the customer's name from history) still counts as mentioned
        missing = [k for k, f in self.plan.facts.items()
                   if f.must_mention and f.verbatim and not f.hidden and k not in self.mentioned and f.value not in spoken]
        if missing:
            self.add_issue(f"missing facts: {missing}", "missing_fact")
        if missing or self.failed or not (self.emitted or out):
            self.used_fallback = True
            out.append(self.fill(render_fallback(self.plan, self.expr)))
        return out

    def report(self) -> ValidationReport:
        codes = list(self.issue_codes)
        if self.used_fallback:
            codes.append("fallback")
        signals: dict[str, int] = {}
        for c in codes:
            signals[c] = signals.get(c, 0) + 1
        return ValidationReport(ok=not self.issues, used_fallback=self.used_fallback, issues=self.issues,
                                issue_codes=codes, signals=signals)

    def add_issue(self, message: str, code: str) -> None:
        self.issues.append(message)
        self.issue_codes.append(code)

    def fill(self, text: str) -> str:
        out = _SLOT.sub(lambda m: self.plan.facts[m.group(1)].value if m.group(1) in self.plan.facts else "", text)
        return re.sub(r"\b(off|छूट)( \1)+\b", r"\1", out)  # "{discount} off" -> "60% off off"

    # ------------------------------------------------------------- internals

    def _accept(self, sentences: list[str]) -> list[str]:
        out = []
        for s in sentences:
            # U+FFFD: a Devanagari character split across stream chunks ("बता�ऊँ")
            s = _CURRENCY_BEFORE_SLOT.sub("", _EMOJI.sub("", _MARKDOWN.sub("", s))).replace("\ufffd", "").strip()
            s = _OFFER_PAD.sub("", s)  # "पहले 3 महीने में {offer}" -> "{offer}" (the phrase already says it)
            s = _THIS_WEEK_PAD.sub("", s)
            if not s:
                continue
            problem = self._check(s)
            if problem:
                self.add_issue(f"{problem}: {s!r}", self._code_for(problem))
                self.failed = True  # stop trusting this generation
                break
            self.mentioned.update(_SLOT.findall(s))
            filled = self.fill(s)
            self.emitted.append(filled)
            out.append(filled)
        return out

    @staticmethod
    def _code_for(problem: str) -> str:
        if problem.startswith("rule pack "):
            return "rulepack:" + problem.split()[2].rstrip(":")
        return next((c for p, c in _CODES if problem.startswith(p)), "other")

    def _claim_sources(self) -> str:
        """Everything the reply may lean on: the caller's words, the plan's operator-written text and its fact values."""
        p = self.plan
        bits = [self.said, p.objective or ""]
        bits += [f.value + " " + (f.description or "") for f in p.facts.values()]
        bits += list((p.lead or {}).values()) + list(p.fallback.values())
        return " ".join(bits)

    def _numbers_left(self, bare: str) -> bool:
        """A number the LLM wrote itself. Numbers inside operator-written talking points ("for the first
        3 months") may be repeated, as digits or words."""
        allowed = set(re.findall(r"\d+", self.said.replace(",", "")))
        if "₹" in self.said:
            bare = bare.replace("₹", "")
        bare = re.sub(r"(?<=\d),(?=\d{3})", "", bare)
        for f in self.plan.facts.values():
            if not f.verbatim and not f.hidden:
                allowed.update(re.findall(r"\d+", f.value))
        if allowed:
            bare = re.sub(r"\d+", lambda m: "" if m.group(0) in allowed else m.group(0), bare)
            if not _RAW_NUMBER.search(bare):
                return False  # number words here are most likely the same talking-point number
        return bool(_RAW_NUMBER.search(bare) or _NUMBER_WORDS.search(bare))

    def _check(self, s: str) -> str | None:
        unknown = [k for k in _SLOT.findall(s) if k not in self.plan.facts or self.plan.facts[k].hidden]
        if unknown:
            return f"invented placeholder {unknown}"
        bare = _SLOT.sub("", s)
        mem = invented_memory(bare)
        if mem:  # always, even in operator text: the agent only knows this call
            return f"invented memory of the caller ({mem})"
        if self.trusted:
            return None
        claims = unsourced_claims(bare, self._claim_sources())
        if claims:
            return f"unsourced claim {claims[0]!r}"
        if self.known is not None:
            ledger, facts = self.known
            if len(bare.split()) >= 3 and ledger.similar_to(bare):
                return "repeated sentence"
            if facts.restates(bare):
                return "restated fact"
            if facts.reasks(bare):
                return "re-asked question"
            if not self.emitted and facts.opener_used(bare) and len(bare.split()) >= 3:
                return "repeated opener"  # starts the way an earlier reply started (same first words)
        if self.plan.lead and re.search(r"\$|₹|रुपय|रुपए|रुपये|डॉलर|dollar|rupe|rupay|\d+ ?%", bare, re.I):
            return "price repeated after the exact price line"
        if "offer" in self.plan.facts and "{offer}" in s:
            # "The {offer} is {offer} after that" / "{offer} for the first 3 months": the phrase already says it all
            if s.count("{offer}") > 1 or "offer" in self.mentioned:
                return "offer phrase repeated"
            if re.search(r"first (3|three) months|then \$?20|3 महीने|तीन महीने|teen mahine|3 mahine", bare, re.I):
                return "offer phrase padded with its own terms"
        if self.plan.general_knowledge:
            # world-knowledge answers need numbers (a year, a temperature); business facts are not in
            # these plans, and currency is still refused
            if re.search(r"[₹$]|rs\.?\s*\d|rupe|रुपए|रुपये|डॉलर|dollar", bare, re.I):
                return "money written in a general-knowledge answer"
        elif self._numbers_left(bare):
            return "raw number/currency written by LLM"
        lang = self.expr.language
        if lang == "en" and _DEVA.search(bare):
            return "wrong language: Devanagari in an English reply"
        if lang == "en" and len(_LATIN_WORD.findall(bare)) >= 5 and detect_language(bare) == "hinglish":
            return "wrong language: Roman Hindi in an English reply"
        if lang == "hi" and not _DEVA.search(bare) and len(_LATIN_WORD.findall(bare)) >= 3:
            return "wrong language: no Devanagari in a Hindi reply"
        if lang == "hinglish" and len(_DEVA.findall(bare)) > 0.6 * max(1, len(re.sub(r"\s", "", bare))):
            return "wrong language: Devanagari in a Hinglish reply"  # a Devanagari word or two is fine
        if lang in ("hi", "hinglish") and len(_LATIN_WORD.findall(bare)) >= 4 and not _DEVA.search(bare) and detect_language(bare) == "en":
            return "wrong language: an English sentence in a Hindi/Hinglish reply"
        for pack in self.rule_packs:
            why = pack.check(s, self.plan, self.expr)
            if why:
                return f"rule pack {getattr(pack, 'name', type(pack).__name__)}: {why}"
        return None
