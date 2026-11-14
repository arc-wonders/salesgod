"""Target-quality checks for Expression-Engine SFT data and eval.

`check_target(text, plan, expr)` returns a list of issue strings (empty = pass).
It runs the real runtime StreamingValidator (fed in small chunks, like a token
stream) plus extra checks the runtime cannot do cheaply:

  language/script match, length vs verbosity, ask at the end, must_not_say,
  number/magnitude words and month names (numbers must come from slots),
  acknowledgement present when requested, formatting, AI-isms / over-apology,
  textbook (shuddh) Hindi.

Only acie + stdlib.
"""

from __future__ import annotations

import re

from acie.schemas import ExpressionParams, TurnPlan
from acie.validator import StreamingValidator

_SLOT = re.compile(r"\{(\w+)\}")
_SENT_SPLIT = re.compile(r"(?<=[.!?।])\s+|\n+")
_DEV = re.compile(r"[ऀ-ॿ]")
_LAT = re.compile(r"[A-Za-z]")
_WORD = re.compile(r"[\wऀ-ॿ{}]+", re.UNICODE)
_EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿️]")

# romanised Hindi function words that basically never occur in English sentences
_HINGLISH_STRONG = {
    "hai", "hain", "nahi", "nahin", "kya", "aap", "aapka", "aapki", "aapke", "aapko", "mein", "hoon", "karna",
    "kar", "ke", "ki", "ka", "ko", "se", "ho", "gaya", "gayi", "gaye", "diya", "di", "raha", "rahi", "theek",
    "abhi", "toh", "bhi", "aur", "liye", "wala", "wali", "chahiye", "chahenge", "doon", "sakta", "sakte",
    "hum", "humne", "maine", "mujhe", "batao", "bataiye", "ji", "haan", "bas", "par", "pe", "jayega", "jayenge",
    "zaroor", "kuch", "koi", "baat", "dijiye", "kijiye", "lijiye", "accha", "achha", "badhiya",
}

NUMBER_WORDS = re.compile(
    r"\b(hundred|thousand|lakh|lakhs|lac|crore|crores|million|billion|percent|per cent|"
    r"hazaar|hazar|hajar|sau|pachas|pachaas|bees|tees|chalis|chaalis|dus|paanch|"
    r"half|double|twice|"
    r"january|february|march|april|june|july|august|september|october|november|december)\b"
    r"|(?<![ऀ-ॿ])(हज़ार|हजार|लाख|करोड़|करोड|प्रतिशत|परसेंट|पचास|बीस|तीस|चालीस|दस|पाँच|पांच|सौ|आधा|दुगना|"
    r"जनवरी|फ़रवरी|फरवरी|मार्च|अप्रैल|जून|जुलाई|अगस्त|सितंबर|अक्टूबर|नवंबर|दिसंबर)(?![ऀ-ॿ])",
    re.IGNORECASE,
)

ACK_MARKERS = {
    "en": ["sorry", "understand", "apologi", "frustrat", "i know", "i hear", "hassle", "annoying", "not fun",
           "must be", "the wait", "upsetting", "disappointing", "stressful", "worrying", "i get it", "fair"],
    "hi": ["माफ़", "माफ", "समझ", "परेशान", "दिक्कत", "अफ़सोस", "अफसोस", "बुरा", "तकलीफ़", "तकलीफ", "इंतज़ार",
           "इंतजार", "sorry", "सॉरी", "चिंता", "झुंझलाहट", "जायज़"],
    "hinglish": ["sorry", "samajh", "pareshan", "dikkat", "maaf", "bura", "takleef", "intezaar", "wait",
                 "frustrat", "tension", "chinta", "jayaz"],
}
APOLOGY = re.compile(r"sorry|apologi[sz]e|माफ़ी|माफ़ कीजिए|माफ कीजिए|क्षमा|सॉरी|maafi|maaf kijiye", re.IGNORECASE)

AI_ISMS = [
    "as an ai", "language model", "i apologize for any inconvenience", "inconvenience caused",
    "sorry for the inconvenience", "i'd be happy to assist", "i would be happy to assist", "rest assured",
    "feel free to", "i hope this helps", "great question", "kindly note", "please be informed",
    "do not hesitate", "don't hesitate", "valued customer", "thank you for your patience",
    "is there anything else i can assist you with today", "dear customer",
    "असुविधा के लिए खेद", "आपकी असुविधा", "असुविधा के लिए क्षमा", "कृपया ध्यान दें", "सहर्ष", "प्रिय ग्राहक",
    "हमें खेद है", "सम्मानित ग्राहक",
]
# textbook Hindi the demo should avoid (spoken Hindi uses the English loanword instead)
SHUDDH = ["शेष राशि", "स्थानांतरण", "स्थानांतरित", "धनराशि", "सीमा", "अनुरोध", "आवेदन", "पूर्व-स्वीकृत",
          "पूर्व स्वीकृत", "लेनदेन संख्या", "संदर्भ संख्या", "वार्षिक शुल्क", "ऋण", "सत्यापन", "विद्युत",
          "मासिक किस्त", "ब्याज दर", "प्रेषित", "निवेदन", "अवगत", "उपरोक्त", "कृपया प्रतीक्षा"]

# must_not_say entries are often descriptive ("limited time pressure"), so map keywords -> surface patterns.
MNS_PATTERNS = [
    ("guarant", r"guarante|गारंटी|pakka|पक्का|definitely (get|be approved)|100\s*%"),
    ("limited time", r"limited[- ]time|only today|today only|hurry|offer ends|jaldi (karein|kijiye|karo)|जल्दी (करें|कीजिए|करो)|सिर्फ़ आज|aaj hi|आज ही|ख़त्म हो जाएगा|khatam ho jayega"),
    ("urgency", r"hurry|right away before|jaldi (karein|kijiye|karo)|जल्दी (करें|कीजिए|करो)|aaj hi|आज ही"),
    ("repeat the offer", r"pre-?approved|प्री-?अप्रूव्ड|annual fee|{card_limit}|{card_benefit}|apply (karna|karenge|now)|अप्लाई कर"),
    ("pressure to reconsider", r"reconsider|think (it )?(over|again)|soch (lijiye|lo)|सोच (लीजिए|लो)|dobara soch|दोबारा सोच|are you sure|sure hain\?|पक्का\?"),
    ("credit score", r"credit score|cibil|सिबिल|क्रेडिट स्कोर|score"),
    ("promises of approval", r"will be approved|approve ho jayega|अप्रूव हो जाएगा|next time (you will|for sure)|zaroor milega|ज़रूर मिलेगा|मिल जाएगा"),
    ("approval is guaranteed", r"approved for sure|guarante|गारंटी|pakka approve|पक्का अप्रूव|approve ho (jayega|jaayega)|अप्रूव हो जाएगा"),
    ("fees other than stated", r"joining fee|processing fee|जॉइनिंग|प्रोसेसिंग|hidden"),
    ("waiver", r"waive|wave off|माफ़ करवा|माफ करवा|maaf karwa|maaf kar (denge|dunga|sakte)|reverse (kar|it)|रिवर्स|हटा (देंगे|दूँगा)"),
    ("fee waiver", r"waive|free of cost|free mein|फ़्री में|मुफ़्त"),
    ("refund already done", r"refund (ho gaya|has been (done|processed|credited)|is done|already)|already refunded|रिफ़ंड हो गया|पैसे (वापस )?आ गए|wapas aa gaye"),
    ("blame the merchant", r"merchant('s)? (fault|galti)|मर्चेंट की ग़लती|मर्चेंट की गलती"),
    ("blame the customer", r"your (fault|mistake)|aapki (galti|ghalti)|आपकी (ग़लती|गलती)|you should have"),
    ("break existing", r"\bbreak\b|तोड़|\btod\b|premature"),
    ("only the minimum", r"(just|only) (pay )?the minimum|sirf minimum|सिर्फ़ मिनिमम|minimum (hi|ही) (bhar|भर)"),
    ("account closure", r"clos(e|ed|ure)|बंद (हो|कर) (जाएगा|देंगे)|band (ho|kar) (jayega|denge)|freez|फ़्रीज़|फ्रीज"),
    ("verification is already complete", r"verif(ied|ication (is )?(done|complete))|वेरिफ़ाई हो गया|verify ho gaya"),
    ("accusing", r"did you (steal|lie)|fraud (kiya|किया)|आपने धोखा"),
]


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def words(text: str) -> list[str]:
    return _WORD.findall(text)


def run_validator(text: str, plan: TurnPlan, expr: ExpressionParams, chunk: int = 6) -> list[str]:
    v = StreamingValidator(plan, expr)
    for i in range(0, len(text), chunk):
        v.feed(text[i:i + chunk])
    v.finish()
    out = list(v.issues)
    if v.used_fallback and not out:
        out.append("validator fell back to template")
    return [f"validator: {x}" for x in out]


def language_issue(text: str, lang: str) -> str | None:
    bare = _SLOT.sub(" ", text)
    dev, lat = len(_DEV.findall(bare)), len(_LAT.findall(bare))
    toks = [w.lower() for w in re.findall(r"[A-Za-z']+", bare)]
    hits = sum(1 for t in toks if t in _HINGLISH_STRONG)
    if lang == "hi":
        if dev + lat == 0 or dev / (dev + lat) < 0.7:
            return f"language: expected Devanagari Hindi (dev={dev}, latin={lat})"
    elif lang == "en":
        if dev:
            return "language: Devanagari in English reply"
        if hits >= 2:
            return f"language: looks like Hinglish ({hits} Hindi markers) in English reply"
    else:
        if dev:
            return "language: Devanagari in Hinglish reply (must be Roman script)"
        need = 1 if len(toks) <= 6 else 2
        if hits < need:
            return f"language: not Hinglish enough ({hits} Hindi markers)"
    return None


def length_issue(text: str, plan: TurnPlan, expr: ExpressionParams) -> str | None:
    ss = [s for s in sentences(text) if len(words(s)) > 2]  # "Sure." / "जी।" don't count as sentences
    n_words = len(words(text))
    extra_s = (1 if plan.ask else 0) + (1 if expr.acknowledge_emotion else 0)
    max_s = {"low": 1, "med": 2, "high": 4}[expr.verbosity] + extra_s
    nfacts = len(plan.facts)
    max_w = {"low": 17, "med": 38, "high": 75}[expr.verbosity] + 10 * bool(plan.ask) \
        + 9 * expr.acknowledge_emotion + 3 * max(0, nfacts - 1)
    max_w = int(max_w * {"en": 1.0, "hi": 1.3, "hinglish": 1.15}[expr.language])
    if len(ss) > max_s:
        return f"length: {len(ss)} sentences > {max_s} for verbosity={expr.verbosity}"
    if n_words > max_w:
        return f"length: {n_words} words > {max_w} for verbosity={expr.verbosity}"
    if n_words < 2:
        return "length: empty/too short"
    return None


def must_not_say_issues(text: str, plan: TurnPlan) -> list[str]:
    low = text.lower()
    out = []
    for item in plan.must_not_say:
        it = item.lower()
        if it in low:
            out.append(f"must_not_say: literal {item!r}")
            continue
        for key, pat in MNS_PATTERNS:
            if key in it and re.search(pat, text, re.IGNORECASE):
                out.append(f"must_not_say: {item!r} matched /{key}/")
                break
    return out


def check_target(text: str, plan: TurnPlan, expr: ExpressionParams) -> list[str]:
    issues = run_validator(text, plan, expr)
    t = text.strip()
    low = t.lower()

    if "\n" in t or re.search(r"^\s*([-*•]|\d+\.)\s", t, re.M) or _EMOJI.search(t) or re.search(r"[*#`\[\]<>]", t):
        issues.append("format: newline/list/markdown/emoji/brackets")
    if re.search(r"\((pause|laughs|sighs|smiles)[^)]*\)", low) or t[:1] in "\"'“":
        issues.append("format: stage direction or quoted output")
    if re.search(r"\{\s*\}|\{[^}]*\s[^}]*\}", t):
        issues.append("format: malformed placeholder")

    li = language_issue(t, expr.language)
    if li:
        issues.append(li)
    le = length_issue(t, plan, expr)
    if le:
        issues.append(le)

    if plan.ask and not t.rstrip().endswith("?"):
        issues.append("ask: reply must end with the plan's question")
    if plan.end_call and t.rstrip().endswith("?"):
        issues.append("end_call: closing turn should not end with a question")

    issues += must_not_say_issues(t, plan)

    bare = _SLOT.sub(" ", t)
    m = NUMBER_WORDS.search(bare)
    if m:
        issues.append(f"numbers: number/date word {m.group(0)!r} outside a slot")

    if expr.acknowledge_emotion:
        first = (sentences(t) or [""])[0].lower()
        if not any(k in first for k in ACK_MARKERS[expr.language]):
            issues.append("ack: first sentence should acknowledge the customer's feeling")
    if len(APOLOGY.findall(t)) >= 2:
        issues.append("tone: over-apologetic (2+ apologies)")
    for p in AI_ISMS:
        if p in low:
            issues.append(f"tone: generic AI/IVR phrase {p!r}")
    if expr.language == "hi":
        for w in SHUDDH:
            if w in t:
                issues.append(f"tone: textbook Hindi {w!r} (use the spoken loanword)")
    return issues
