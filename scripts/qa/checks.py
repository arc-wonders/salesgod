"""Machine checks for the conversation regression harness.

Every check takes a `Ctx` (one agent reply plus what came before) and returns None (ok) or a short detail string.
Default checks run on every agent turn; a fixture turn can list `skip` names to switch some off for that turn, and
`must` for positive expectations. Customer turns carry `tags` that steer the state-dependent checks:

  interest / ask_link / accept   the caller wants the offer: a payment link may now be offered
  refuse                         the caller said no (or cannot use links): no selling, no link, from now on
  hardship                       caller is down / money trouble / crisis: no selling until they ask for the offer
  asks_price                     caller asks for a price: selling words are fine in that reply
  garbled                        speech-to-text noise: "didn't catch that" is allowed once
  end_ok                         the agent may end the call after this turn (bye, opt-out, clear no)
  answered_goal                  caller has told us their goal: asking for it again is a failure
  repeat_ask                     caller asked for a repeat: a near copy of the earlier reply is fine
"""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REP_RATIO = 0.6
_PUNCT = re.compile(r"[\s\.,!?;:\"'()\[\]।\-]+")
_DEV = re.compile(r"[ऀ-ॿ]")
_LAT = re.compile(r"[A-Za-z]")

SELL_INTENTS = {"pitch", "price", "reoffer", "confirm_close", "recover", "open_question", "price_floor"}
_LINK = re.compile(r"\blink\b|\bsms\b|payment|लिंक|एसएमएस|पेमेंट|joining link", re.I)
_SELL_WORDS = re.compile(r"\boffer\b|% ?off|discount|\bprice\b|payment|\blink\b|try karna|try करना|कीमत|ऑफर|डिस्काउंट|"
                         r"keemat|₹|\$|bata doon|बता दूँ|join kar|जॉइन", re.I)
_DIDNT_CATCH = re.compile(r"didn.?t (quite )?(get|catch)|say (that )?(it )?(once more|again)|दोबारा|phir se bol|"
                          r"samajh nahi aaya|सुनाई नहीं|dobara bol|ek baar phir", re.I)
_FEMININE = re.compile(
    r"\b(kar|ho|de|le|bata|dekh|sun|jaa|ja|chaah|samajh|bhej|puch|aa|soch|rakh|mil)\s*(sakti|rahi|chahti|paati|"
    r"leti|deti|karti|jaati|samajhti|dekhti)\b|\b\w*(ungi)\b|\b(sakti|karti|chahti|rahi|samajhti|bolti|batati|"
    r"jaanti|sunti|dekhti)\s+(hoon|hun|hu|hu\.|hoon\.)\b|(सकती|करती|चाहती|रही|समझती|बोलती|जानती|देखती|सुनती)\s*(हूँ|हूं)|"
    r"(ूँगी|ूंगी|ऊँगी|ऊंगी|ुंगी)\b|\bmain\s+\w+\s+gayi\b|मैं\s+\S+\s+गई\b", re.I)
_EN_WORDS = {"the", "is", "are", "you", "your", "i", "to", "of", "and", "could", "would", "sorry", "please", "say",
             "once", "more", "didn't", "quite", "get", "that", "have", "for", "it", "we", "can", "will", "with",
             "this", "about", "if", "what", "do", "does", "not", "am", "my", "me", "tell", "price", "offer",
             "week", "per", "month", "months", "after", "included", "there", "any", "or", "on", "at", "be", "as"}
_HI_MARK = {"hai", "hain", "ho", "hoon", "main", "mein", "me", "aap", "ap", "ke", "ki", "ka", "ko", "nahi", "nahin",
            "kya", "to", "toh", "ye", "yeh", "wo", "woh", "bhi", "se", "par", "pe", "ji", "kar", "karna", "sakte",
            "sakta", "raha", "rahe", "chahenge", "bataun", "aur", "agar", "koi", "kuch", "hi", "ek", "isme", "iske",
            "mere", "mujhe", "hum", "hume", "abhi", "baad", "samajh", "theek", "thik", "baat", "bhej", "doon",
            "milta", "milti", "milega", "shamil", "keemat", "mahine", "hafte", "hafta", "kijiye", "kijiyega"}


def norm(t: str) -> str:
    return _PUNCT.sub(" ", t.lower()).strip()


def ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()


def flat(t: str) -> str:
    """Lower-case, drop commas inside numbers and symbols so '₹2,999' matches '2999'."""
    t = re.sub(r"(?<=\d),(?=\d)", "", t.lower())
    return t.replace("₹", "").replace("$", "")


def reply_lang(t: str) -> str:
    """'hi' (Devanagari), 'hinglish' (romanised Hindi), 'en' (English) or 'none'."""
    d, l = len(_DEV.findall(t)), len(_LAT.findall(t))
    if d + l == 0:
        return "none"
    if d / (d + l) > 0.5:
        return "hi"
    words = re.findall(r"[a-z']+", t.lower())
    hi = sum(w in _HI_MARK for w in words)
    en = sum(w in _EN_WORDS for w in words)
    if d and d / (d + l) > 0.25:
        return "hi"
    return "hinglish" if hi >= 2 and hi >= en * 0.6 else "en"


def english_sentences(t: str) -> list[str]:
    """Sentences of a reply that read as English (used to catch English inside a Hindi/Hinglish reply)."""
    out = []
    for s in re.split(r"(?<=[.!?।])\s+", t):
        if _DEV.search(s):
            continue
        words = re.findall(r"[A-Za-z']+", s)
        if len(words) < 4:
            continue
        en = sum(w.lower() in _EN_WORDS for w in words)
        hi = sum(w.lower() in _HI_MARK for w in words)
        if en >= 3 and hi <= 1:
            out.append(s)
    return out


@dataclass
class Ctx:
    reply: str
    intent: str
    end_call: bool
    ask: str | None
    say: str                       # the caller's words this turn
    tags: set[str]
    lang: str                      # caller's language this turn (en/hi/hinglish)
    prior_replies: list[str]       # every earlier agent line, opening included
    prior_tags: list[set[str]]     # tags of earlier caller turns
    allowed_numbers: set[str]
    product_name: str
    other_products: list[str] = field(default_factory=list)
    expect_lang: str | None = None  # language the reply must be in (language-switch fixtures)

    @property
    def accepted(self) -> bool:
        return any(t & {"accept", "ask_link", "interest"} for t in self.prior_tags + [self.tags])

    @property
    def refused(self) -> bool:
        return any("refuse" in t for t in self.prior_tags + [self.tags])

    @property
    def hardship_open(self) -> bool:
        """Hardship was voiced and the caller has not since asked for the offer or a price."""
        state = False
        for t in self.prior_tags + [self.tags]:
            if "hardship" in t:
                state = True
            elif t & {"asks_price", "accept", "ask_link"}:
                state = False
        return state


# ---------------------------------------------------------------- default checks (must_not)

def c_repeat(c: Ctx):
    if "repeat_ask" in c.tags:
        return None
    for i, p in enumerate(c.prior_replies):
        if i == 0:
            continue  # the opening line is a greeting, not a reply
        r = ratio(c.reply, p)
        if r > REP_RATIO:
            return f"{r:.2f} copy of agent line #{i}: {p[:70]!r}"


def c_end_call_pending_question(c: Ctx):
    if c.end_call and not (c.tags & {"end_ok"}):
        return "ended the call on a caller turn that still needed an answer: " + c.say[:50]


def c_link_unasked(c: Ctx):
    if not _LINK.search(c.reply):
        return None
    if c.refused:
        return "link/payment mentioned after the caller refused"
    if not c.accepted:
        return "link/payment mentioned before the caller showed interest"


def c_push_after_no(c: Ctx):
    if c.refused and (c.intent in SELL_INTENTS) and not (c.tags & {"asks_price"}):
        return f"sales intent {c.intent!r} after a no"


def c_english_in_hindi(c: Ctx):
    if c.lang in ("hi", "hinglish"):
        if reply_lang(c.reply) == "en":
            return "English reply to a Hindi/Hinglish caller: " + c.reply[:60]
        e = english_sentences(c.reply)
        if e:
            return "English sentence inside a Hindi reply: " + e[0][:60]


def c_wrong_language(c: Ctx):
    if c.expect_lang:
        got = reply_lang(c.reply)
        ok = {"hi": {"hi", "hinglish"}, "hinglish": {"hi", "hinglish"}, "en": {"en"}}[c.expect_lang]
        if got not in ok and got != "none":
            return f"asked for {c.expect_lang} but reply is {got}: {c.reply[:60]!r}"
    elif c.lang == "en" and reply_lang(c.reply) in ("hi", "hinglish"):
        return "Hindi reply to an English caller: " + c.reply[:60]


def c_feminine(c: Ctx):
    m = _FEMININE.search(c.reply)
    if m:
        return f"feminine first-person form {m.group(0)!r} (Arjun is male)"


def c_digits_not_in_facts(c: Ctx):
    caller = set(re.findall(r"\d+", flat(c.say)))
    for n in re.findall(r"\d+", flat(c.reply)):
        if n not in c.allowed_numbers and n not in caller:
            return f"number {n} is not in the product facts"


def c_sell_after_hardship(c: Ctx):
    if c.hardship_open:
        safety = c.intent.startswith("safety")  # the helpline reply may say "I will not talk about any offer"
        if c.intent in SELL_INTENTS or (not safety and _SELL_WORDS.search(c.reply)):
            return f"selling after hardship (intent {c.intent!r}): {c.reply[:60]!r}"


def c_didnt_catch(c: Ctx):
    if _DIDNT_CATCH.search(c.reply) and "garbled" not in c.tags:
        return "'didn't catch that' to a clear caller turn: " + c.say[:40]


def c_wrong_product(c: Ctx):
    low = c.reply.lower()
    for o in c.other_products:
        if o.lower() in low:
            return f"names another campaign's product {o!r}"


_MEMORY = re.compile(r"your (details|record|data|number|info\w*) (are|is)? ?(in|with|on) (our|the) (system|file|records?)|"
                     r"(previous|earlier|last) (inquiry|enquiry|conversation|call|visit|chat)|we (already )?have your (details|info)|"
                     r"remember (you|your)|from (our|your) (last|previous)|पिछली (बातचीत|पूछताछ|कॉल)|"
                     r"(आपकी|आपके) (डिटेल|जानकारी).{0,25}(हमारे पास|सिस्टम)|हमारे पास आपकी|pichli (baat|inquiry|call)|"
                     r"aapki details.{0,25}(hamare paas|system)", re.I)
_GOAL_Q = re.compile(r"(main|your|aapka|आपका)\s+(main\s+)?goal|goal (kya|क्या)|losing weight, building muscle|"
                     r"weight (kam|कम).{0,25}muscle|fit rehna|fit रहना|what.s your goal|kya goal", re.I)
_LISTENING = re.compile(r"i.?m listening|^\W*please go ahead|^\W*go ahead\W*$|main sun raha|मैं सुन रहा|^\W*(ji )?(boliye|बोलिए|bataiye)\W*$", re.I)
_LINK_SENT = re.compile(r"(link|लिंक).{0,40}(bhej (raha|diya)|भेज (रहा|दिया)|sent|on its way|is on)|"
                        r"(sent|sending|bhej raha|भेज रहा).{0,30}(link|लिंक)", re.I)


def c_invented_memory(c: Ctx):
    m = _MEMORY.search(c.reply)
    if m:
        return f"invented memory of the caller: {m.group(0)!r}"


def c_goal_reasked(c: Ctx):
    if any("answered_goal" in t for t in c.prior_tags + [c.tags]) and _GOAL_Q.search(c.reply):
        return "discovery (goal) question asked again after the caller answered it"


def c_listening_cue(c: Ctx):
    if "garbled" in c.tags or "repeat_ask" in c.tags:
        return None
    if _LISTENING.search(c.reply) and len(c.say.split()) >= 3:
        return f"listening cue instead of an answer: {c.reply[:60]!r}"


def c_link_without_consent(c: Ctx):
    if (c.intent == "link_sent" or _LINK_SENT.search(c.reply)) and not any(
            t & {"accept", "ask_link"} for t in c.prior_tags + [c.tags]):
        return "link sent / promised although the caller never said yes: " + c.say[:50]


DEFAULT_CHECKS = {
    "repeat": c_repeat,
    "end_call_pending_question": c_end_call_pending_question,
    "link_unasked": c_link_unasked,
    "push_after_no": c_push_after_no,
    "english_in_hindi": c_english_in_hindi,
    "wrong_language": c_wrong_language,
    "feminine_forms": c_feminine,
    "digits_not_in_facts": c_digits_not_in_facts,
    "sell_after_hardship": c_sell_after_hardship,
    "didnt_catch": c_didnt_catch,
    "wrong_product": c_wrong_product,
    "invented_memory": c_invented_memory,
    "goal_reasked": c_goal_reasked,
    "listening_cue": c_listening_cue,
    "link_without_consent": c_link_without_consent,
}


# ---------------------------------------------------------------- positive expectations (must)

def check_must(c: Ctx, must: dict) -> list[tuple[str, str]]:
    while isinstance(must.get("must"), dict):  # fixtures wrote {"must": {"must": {...}}}: unwrap, or the expectations never ran
        must = {**{k: v for k, v in must.items() if k != "must"}, **must["must"]}
    out = []
    fr = flat(c.reply)
    if "contains_any" in must and not any(flat(x) in fr for x in must["contains_any"]):
        out.append(("must_contain", f"reply lacks any of {must['contains_any']}: {c.reply[:80]!r}"))
    if "contains_all" in must:
        miss = [x for x in must["contains_all"] if flat(x) not in fr]
        if miss:
            out.append(("must_contain", f"reply lacks {miss}: {c.reply[:80]!r}"))
    if "not_contains_any" in must:
        hit = [x for x in must["not_contains_any"] if flat(x) in fr]
        if hit:
            out.append(("must_not_contain", f"reply contains {hit}: {c.reply[:80]!r}"))
    if "intent_in" in must and c.intent not in must["intent_in"]:
        out.append(("intent", f"intent {c.intent!r} not in {must['intent_in']}"))
    if "intent_not_in" in must and c.intent in must["intent_not_in"]:
        out.append(("intent", f"intent {c.intent!r} is in the forbidden set {must['intent_not_in']}"))
    if must.get("end_call") is True and not c.end_call:
        out.append(("must_end_call", "caller said bye / opted out but the call did not end"))
    if must.get("end_call") is False and c.end_call:
        out.append(("must_not_end_call", "call ended but the caller was still talking"))
    if "lang" in must and reply_lang(c.reply) not in {"hi": {"hi", "hinglish"}, "hinglish": {"hi", "hinglish"},
                                                       "en": {"en"}}[must["lang"]]:
        out.append(("must_lang", f"reply language {reply_lang(c.reply)} != {must['lang']}"))
    return out


def numbers_in_product(prod_raw: dict) -> set[str]:
    """Every digit group anywhere in the product record, plus a small set of harmless ones (helpline, 1-2)."""
    text = flat(json.dumps(prod_raw, ensure_ascii=False))
    nums = set(re.findall(r"\d+", text))
    return nums | {"1", "2", "14416", "112", "5", "11"}
