"""Implicit long-term preference learning (Layer 3 -> profile).

`learn_from_turn(profile, text, state, expr, explicit)` is called once per customer turn (by the
feedback lane). Every long-term key (formality, verbosity, complexity, warmth, explanation_style,
info_density, regional_variant) gets WEAK evidence through the shared confidence update
(`update_preference`, small lr): ~10 consistent turns lift a preference to the 0.6 confidence the
policy needs before it acts on it, and one odd turn never flips it. Keys the customer stated
explicitly this turn are skipped (explicit statements are handled by the feedback lane).

Also extracts likes / dislikes / topics ("mujhe X pasand hai", "I hate X", "I use it for coding"),
each list capped at LIST_CAP, newest last, de-duplicated, and a thing is never in likes and dislikes
at once (the latest statement wins).
"""

from __future__ import annotations

import re

from ..schemas import CustomerProfile, ExpressionParams, HumanState
from ..text import padded
from .profile_store import update_preference

LIST_CAP = 20
LR = 0.1  # pronoun / explicit-marker evidence
LR_WEAK = 0.05  # length / state-derived evidence

_HIGH_FORMAL = {"aap", "aapka", "aapki", "aapke", "aapko", "aapse", "ji", "sir", "madam", "maam", "mam", "sahab", "sahib",
                "kindly", "please", "pls", "regards", "आप", "आपका", "आपकी", "आपके", "आपको", "जी", "सर", "मैडम", "कृपया",
                "janab", "huzoor", "boliye", "bataiye", "kijiye", "dijiye", "बोलिए", "बताइए", "कीजिए", "दीजिए"}
_LOW_FORMAL = {"tum", "tu", "tera", "teri", "tere", "tujhe", "tumhara", "tumhari", "tumhe", "tumko", "bhai", "bro", "dude",
               "yaar", "abe", "oye", "bol", "bata", "तू", "तुम", "तेरा", "तेरी", "तुझे", "तुम्हारा", "तुम्हें", "भाई", "यार",
               "अबे", "ओए", "bhaiya", "mate", "dost"}
_WARM = {"thanks", "thank", "thankyou", "thx", "shukriya", "dhanyavad", "dhanyawad", "great", "awesome", "wonderful", "lovely",
         "love", "nice", "amazing", "good", "welcome", "pleasure", "glad", "happy", "badhiya", "shandar", "mast",
         "dhanyavaad", "धन्यवाद", "शुक्रिया", "बढ़िया", "बढिया", "शानदार", "बहुत", "बेहतरीन", "hello", "hi", "namaste", "नमस्ते"}
_COLD = re.compile(r" (?:no small talk|kaam (?:ki|ka) baat|business only|just business|skip (?:the )?pleasantries|"
                   r"to the point|seedha (?:kaam|point)|get to the point|time (?:nahi|nhi) hai|काम की बात) ")
_EXAMPLE = re.compile(r" (?:example|examples|udaharan|misaal|for instance|for example|jaise|उदाहरण|मिसाल|matlab kya|for eg|eg) ")
_STEPS = re.compile(r" (?:step|steps|stepwise|process|procedure|pehle (?:kya|kaise)|phir kya|next (?:step|kya)|uske baad|"
                    r"kaise karna|how (?:do|to|should) i|kya kya karna|kaise karu|kaise karun|स्टेप|प्रक्रिया|फिर क्या|उसके बाद) ")
_DIRECT = re.compile(r" (?:seedh[ae]|sidha|direct(?:ly)?|just tell|straight|bottom line|bas itna|yes or no|haan ya na|सीधा|सीधे) ")
_WANT_MORE = re.compile(r" (?:aur batao|aur bataiye|aur kuch|tell me more|more (?:info|details?)|elaborate|sab kuch|everything|"
                        r"poori|puri|saari|sari|details?|vistar|और बताओ|और बताइए|विस्तार|पूरी|सारी) ")
_ENOUGH = re.compile(r" (?:bas|enough|itna hi|thats (?:enough|all|it)|ok ok|theek hai bas|that will do|chalega|kaafi hai|"
                     r"zyada nahi|bas itna|बस|काफी है|इतना ही) ")
_JARGON = re.compile(r"\b(?:api|latency|throughput|roi|kpi|sla|apr|nav|xirr|cagr|underwriting|amortization|"
                     r"webhook|schema|deploy\w*|backend|frontend|sdk|oauth|encryption|compliance|kyc|aml)\b")
_SIMPLE = re.compile(r" (?:samajh (?:nahi|nhi) (?:aaya|aya|aa raha|aata)|simple|aasan|easy|dumb it down|layman|not clear|"
                     r"samjha nahi|समझ नहीं|आसान|सरल) ")

# regional markers: weights per marker
_BIHAR = {"hamra", "hamar", "humka", "hamko", "ka ho", "kaa ho", "ka kare", "kahe", "kaahe", "ee", "ehi", "okra", "ohi",
          "kauno", "kuchhu", "kachhu", "sunaa", "bhaiya", "हमरा", "हमार", "का हो", "काहे", "बाबू", "babu", "ka haal"}
_MUMBAI = {"apun", "bantai", "bidu", "bhidu", "kaiko", "bhau", "tapori", "lafda", "kya bolta", "kya bolti", "ek dum", "bhidu",
           "वांदा", "vanda", "jhakaas", "झक्कास", "अपुन", "बंटाई"}
_DELHI = {"bhai", "yaar", "oye", "abe", "bawal", "ekdum", "scene", "bhai sahab", "bhaisahab", "jugaad", "यार", "भाई",
          "ओए", "अबे", "बवाल", "एकदम", "chill", "scene hai", "kya scene"}
_LUCKNOW = {"janab", "huzoor", "tashreef", "farmaiye", "farmaye", "farmaaiye", "nawab", "miyan", "mian", "jee", "zaroor",
            "pehle aap", "kijiyega", "boliyega", "dijiyega", "bataiyega", "aiyega", "जनाब", "हुजूर", "तशरीफ", "फरमाइए",
            "मियां", "मियाँ"}
_SOUTH_RX = re.compile(r"(?:\b(?:itself|macha|machan|ayyo|aiyo|aiyoo|enna|prepone|revert back|do the needful|kindly adjust)\b|"
                       r"\bda\b[.?!,]?\s*$|\bonly[.?!,]\s*$|\b\w+ only\s*$|\bwhat da\b|\bis it\b[.?!]*\s*$|\bno\?\s*$)")
_HUM_NOT_US = re.compile(r" hum (?!log|sab|sabhi|dono|bhi log|saath|sath|aur|dono)")

_FILLER = {"bahut", "bilkul", "ye", "yeh", "wo", "woh", "kaafi", "really", "very", "the", "a", "an", "to", "toh", "hi", "bhi", "hai",
           "hain", "mujhe", "mera", "meri", "mere", "main", "i", "my", "so", "much", "ki", "ka", "ke", "is", "are", "bohot", "bahot",
           "zyada", "jyada", "sach", "mein", "me", "sirf", "ek", "kuch", "bas", "yaar", "बहुत", "बिल्कुल", "यह", "ये", "वो", "है", "हैं",
           "को", "भी", "तो", "ही", "और", "ज्यादा", "कि", "का", "की", "के"}
_STOP = {"aur", "and", "but", "lekin", "par", "because", "kyunki", "i", "so", "also", "or", "though", "when", "jab", "magar", "isliye"}
_LIKE = [re.compile(p) for p in (
    r" (?:mujhe|muje|mjhe) (.{2,40}?) (?:bahut |bohot |bahot |kaafi |bilkul )?(?:pasand (?:hai|h|aata|aati|aate)|pasand|acha lagta|achha lagta|accha lagta|achi lagti|achhi lagti|love lagta) ",
    r" (?:मुझे) (.{2,40}?) (?:बहुत )?(?:पसंद(?: है| आता| आती)?|अच्छा लगता|अच्छी लगती) ",
    r" i (?:really |absolutely |just )?(?:like|love|enjoy|prefer|adore|am into|m into) (\S+(?: \S+){0,3}) ",
    r" (?:mera|meri|mere) (?:favourite|favorite|fav|pasandida) (.{2,40}?) (?:hai|h|he) ",
    r" my (?:favourite|favorite|fav) (?:\w+ )?(?:is|are) (\S+(?: \S+){0,3}) ",
)]
_DISLIKE = [re.compile(p) for p in (
    r" i (?:really |absolutely |just )?(?:hate|dislike|despise|cant stand|can not stand|dont like|do not like|detest) (\S+(?: \S+){0,3}) ",
    r" (?:mujhe|muje|mjhe) (.{2,40}?) (?:bilkul |bahut |zyada |ek dum )?(?:pasand (?:nahi|nhi|nai)|(?:acha|achha|accha) (?:nahi|nhi) lagta|bura lagta|irritate karta|irritating lagta|bakwas lagta|nafrat) ",
    r" (?:mujhe|muje|mjhe) (?:(?:bilkul|bahut) )?(?:nafrat hai|dikkat hai|problem hai|chidh hai) (\S+(?: \S+){0,3}) ",
    r" (.{2,40}?) (?:mujhe |muje )(?:bilkul )?(?:pasand (?:nahi|nhi)|bura lagta|acha nahi lagta) ",
    r" (?:मुझे) (.{2,40}?) (?:बिल्कुल )?(?:पसंद (?:नहीं|नही)|बुरा लगता|अच्छा नहीं लगता) ",
)]
_USE_MARK = re.compile(r" (?:i|my|we|our|mera|meri|mere|hamara|hamari|hamare|main|mai|mujhe|humara|humari|hum|मेरा|मेरी|मेरे|हमारा|हमारी|मैं|मुझे|हम) ")
_TOPICS: dict[str, tuple[str, ...]] = {
    "coding": ("coding", "programming", "programmer", "developer", "software", "code", "python", "javascript", "app development", "kode"),
    "study": ("study", "studying", "padhai", "padhta", "padhti", "exam", "exams", "student", "college", "school", "neet", "jee",
              "upsc", "homework", "assignment", "thesis", "पढ़ाई", "पढाई", "परीक्षा"),
    "business": ("business", "dukaan", "shop", "startup", "company", "vyapar", "dhandha", "dhanda", "invoice", "gst", "customers",
                 "sales", "store", "दुकान", "व्यापार", "बिज़नेस", "बिजनेस"),
    "finance": ("loan", "emi", "investment", "invest", "sip", "mutual fund", "insurance", "savings", "credit card", "stocks", "tax",
                "लोन", "निवेश", "बीमा"),
    "health": ("health", "doctor", "fitness", "gym", "diet", "medicine", "yoga", "सेहत", "स्वास्थ्य"),
    "travel": ("travel", "trip", "vacation", "holiday", "tour", "flight", "yatra", "ghoomna", "यात्रा"),
    "cooking": ("cooking", "recipe", "recipes", "khana", "kitchen", "baking"),
    "cricket": ("cricket", "ipl", "match", "क्रिकेट"),
    "music": ("music", "songs", "gaana", "gaane", "singing", "गाने", "संगीत"),
    "movies": ("movies", "movie", "film", "films", "series", "netflix", "फिल्म"),
    "gaming": ("gaming", "games", "pubg", "bgmi", "valorant", "minecraft"),
    "writing": ("writing", "essay", "blog", "content writing", "emails", "likhna"),
}
_PREFERENCE_VERBS = r"(?:use|using|need|want|learn|learning|work|working|do|doing|build|building|run|running|handle|handling|prepare|preparing|karta|karti|karna|chahiye|ke liye|seekh|chala|sikhna|sikh|kar raha|kar rahi|lagta)"


def _append(lst: list[str], item: str) -> None:
    key = item.lower()
    lst[:] = [x for x in lst if x.lower() != key]
    lst.append(item)
    del lst[:-LIST_CAP]


def _clean_phrase(raw: str) -> str | None:
    cut = []
    for w in raw.split():
        if w in _STOP:
            break
        cut.append(w)
    words = [w for w in cut if w not in _FILLER]
    # drop trailing verbs/aux that sneak into the capture
    while words and words[-1] in {"hai", "h", "he", "hain", "lagta", "lagti", "karta", "karti"}:
        words.pop()
    if not words or len(words) > 4:
        return None
    phrase = " ".join(words)
    return phrase if len(phrase) >= 2 and not phrase.isdigit() else None


def _extract_phrases(patterns: list[re.Pattern[str]], low: str) -> list[str]:
    out: list[str] = []
    for rx in patterns:
        for m in rx.finditer(low):
            ph = _clean_phrase(m.group(1))
            if ph and ph not in out:
                out.append(ph)
    return out


def _learn_lists(profile: CustomerProfile, low: str) -> list[str]:
    touched: list[str] = []
    dislikes = _extract_phrases(_DISLIKE, low)
    likes = [p for p in _extract_phrases(_LIKE, low) if p not in dislikes]
    # "i dont like X" must not also be read as a like ("i like"/"dont like" overlap): drop likes starting with negations
    likes = [p for p in likes if not p.startswith(("not ", "nahi ", "nhi "))]
    for ph in likes:
        profile.dislikes[:] = [x for x in profile.dislikes if x.lower() != ph]
        _append(profile.likes, ph)
        touched.append("likes")
    for ph in dislikes:
        profile.likes[:] = [x for x in profile.likes if x.lower() != ph]
        _append(profile.dislikes, ph)
        touched.append("dislikes")
    if _USE_MARK.search(low) or re.search(_PREFERENCE_VERBS, low):
        for topic, words in _TOPICS.items():
            if any(f" {w} " in low for w in words) and (_USE_MARK.search(low)):
                _append(profile.topics, topic)
                touched.append("topics")
    return touched


def _regional_evidence(text: str, low: str) -> str | None:
    scores = {"bihar": 0, "mumbai": 0, "delhi": 0, "lucknow": 0, "south_indian_english": 0}
    toks = low.strip()
    padded_low = " " + toks + " "
    for name, marks in (("bihar", _BIHAR), ("mumbai", _MUMBAI), ("delhi", _DELHI), ("lucknow", _LUCKNOW)):
        for m in marks:
            if f" {m} " in padded_low:
                scores[name] += 1
    if _HUM_NOT_US.search(padded_low) and " aap " not in padded_low and " hum " in padded_low and not _WE_CONTEXT.search(padded_low):
        scores["bihar"] += 1
    if _SOUTH_RX.search(text.lower().strip()):
        scores["south_indian_english"] += 1
    best = max(scores, key=lambda k: scores[k])
    if scores[best] == 0 or list(scores.values()).count(scores[best]) > 1:
        return None
    return best


_WE_CONTEXT = re.compile(r" hum (?:sab|log|dono|saath|sath|sabhi|aapke|aapki|aapko|company|team)\b")


def learn_from_turn(profile: CustomerProfile, text: str, state: HumanState, expr: ExpressionParams | None,
                    explicit: dict[str, str] | None = None) -> list[str]:
    """Apply one turn of implicit evidence to `profile`. Returns the keys that received evidence."""
    explicit = explicit or {}
    low = padded(text)
    toks = low.split()
    n = len(toks)
    touched: list[str] = []
    if n == 0:
        return touched

    def ev(key: str, value: str, lr: float) -> None:
        if key in explicit:
            return
        update_preference(profile, key, value, lr=lr)
        touched.append(key)

    # formality: how they address the agent ("aap"/"ji"/"sir" vs "tum"/"tu"/"bhai")
    hi = sum(t in _HIGH_FORMAL for t in toks)
    lo = sum(t in _LOW_FORMAL for t in toks)
    if hi > lo:
        ev("formality", "high", LR)
    elif lo > hi:
        ev("formality", "low", LR)

    # verbosity: their own reply length (weak: one-word answers to yes/no questions are common)
    if n <= 4:
        ev("verbosity", "low", LR_WEAK)
    elif n >= 25:
        ev("verbosity", "high", LR_WEAK)
    elif n >= 8:
        ev("verbosity", "med", LR_WEAK / 2)
    if _ENOUGH.search(low) and n <= 6 and not _WANT_MORE.search(low):
        ev("verbosity", "low", LR)

    # warmth
    if _COLD.search(low):
        ev("warmth", "low", LR)
    elif any(t in _WARM for t in toks) or re.search(r"[:;]-?[)D]|[❤😊😀🙏👍]", text):
        ev("warmth", "high", LR_WEAK * 2)
    elif state.frustration >= 0.5:
        ev("warmth", "high", LR_WEAK)  # upset people want warmth back

    # complexity
    if _JARGON.search(text.lower()):
        ev("complexity", "high", LR_WEAK)
    if _SIMPLE.search(low):
        ev("complexity", "low", LR)

    # explanation style
    if _DIRECT.search(low):
        ev("explanation_style", "direct", LR)
    elif _EXAMPLE.search(low) or (n <= 6 and re.search(r"\bexamples?\b\??", text.lower())):
        ev("explanation_style", "example", LR)
    elif _STEPS.search(low):
        ev("explanation_style", "step_by_step", LR)

    # information density
    if _WANT_MORE.search(low) and not _ENOUGH.search(low):
        ev("info_density", "high", LR)
    elif _ENOUGH.search(low):
        ev("info_density", "low", LR)
    elif getattr(state, "cognitive_load", 0.0) >= 0.6:
        ev("info_density", "low", LR_WEAK)

    # regional variant from dialect markers
    region = _regional_evidence(text, low)
    if region:
        ev("regional_variant", region, LR)

    # likes / dislikes / topics
    touched += _learn_lists(profile, low)
    return touched
