"""Small text utilities shared across layers (normalization, language detection,
tokenizing, spoken-amount parsing).

Everything here is pure Python and allocation-light: it runs on every ASR turn.
Input is assumed to be raw ASR output in English, Hindi (Devanagari) or Roman
Hindi (Hinglish): lowercase, little punctuation, spoken numbers, misspellings.
"""

from __future__ import annotations

import re
import unicodedata

from .schemas import Language

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_WORD = re.compile(r"[\wऀ-ॣॱ-ॿ]+", re.UNICODE)  # excludes danda (।॥)

# --------------------------------------------------------------------------- normalization

_NORM_TABLE = {
    0x093C: None,  # nukta: ज़ -> ज, ड़ -> ड (ASR is inconsistent about it)
    0x0901: 0x0902,  # chandrabindu -> anusvara: हाँ -> हां
    0x200C: None, 0x200D: None,  # ZWNJ / ZWJ
    0x0964: ord(" "), 0x0965: ord(" "),  # danda
    0x2019: None, 0x2018: None, ord("'"): None, ord("`"): None,  # don't -> dont (as ASR writes it)
}
_NORM_TABLE.update({0x0966 + i: ord(str(i)) for i in range(10)})  # ०-९ -> 0-9


def normalize(text: str) -> str:
    """Lowercase + canonical Devanagari (no nukta, anusvara for chandrabindu,
    ASCII digits) + apostrophes dropped. Apply to both text and lexicons."""
    return unicodedata.normalize("NFD", text).translate(_NORM_TABLE).lower()


def tokens(text: str) -> list[str]:
    return _WORD.findall(normalize(text))


def padded(text: str) -> str:
    """' tok1 tok2 ... ' for whole-word phrase matching."""
    return " " + " ".join(tokens(text)) + " "


# --------------------------------------------------------------------------- language detection

# Roman-script Hindi. Grammatical words are strong evidence; nouns/fillers that
# English speakers in India drop into English sentences are weak.
HINGLISH_STRONG = {
    "hai", "hain", "hu", "hoon", "hun", "ho", "tha", "thi", "nahi", "nahin", "nhi", "nai",
    "kya", "kyu", "kyun", "kyon", "kaise", "kaisa", "kitna", "kitni", "kitne", "kab", "kahan", "kaun",
    "kaunsa", "konsa", "kis", "kisi", "mera", "meri", "mere", "mujhe", "mujhko", "humko", "hum",
    "aap", "aapka", "aapki", "aapke", "apna", "apne", "apni", "tum", "tumhara", "karna", "karni",
    "karne", "karo", "kar", "kariye", "kijiye", "karein", "karunga", "dijiye", "dena", "de", "diya",
    "ko", "se", "mein", "mei", "ka", "ki", "ke", "bata", "batao", "bataiye", "bataye", "bolo",
    "boliye", "bol", "chahiye", "chahta", "chahti", "chahte", "bhejna", "bhejo", "bhej", "bhejni",
    "bhejne", "raha", "rahi", "rahe", "gaya", "gayi", "gaye", "hua", "hui", "wala", "wali", "wale",
    "toh", "bhi", "aur", "kuch", "koi", "yeh", "ye", "woh", "wo", "iska", "uska", "sakta", "sakti",
    "sakte", "samajh", "samjha", "samjhi", "samjhao", "baat", "abhi", "pehle", "baad", "sirf",
    "matlab", "lekin", "par", "paisa", "paise", "rupaye", "hazaar", "hazar", "lakh", "accha",
}
HINGLISH_WEAK = {
    "haan", "han", "haa", "ji", "acha", "achha", "theek", "thik", "yaar", "bhai", "arre", "arey",
    "jaldi", "bas", "chalo", "bilkul", "zaroor", "shukriya", "dhanyavad", "namaste", "main", "na",
}
# English function words: evidence the sentence is English even if it carries a Hindi noun.
ENGLISH_MARKERS = {
    "the", "is", "are", "am", "was", "my", "me", "i", "you", "your", "what", "how", "why", "when",
    "where", "which", "who", "please", "want", "can", "could", "would", "will", "should", "to",
    "of", "and", "this", "that", "it", "do", "does", "did", "have", "has", "need", "with", "for",
    "about", "tell", "check", "send", "not", "yes", "no", "much", "many", "there", "here", "know",
    "an", "a", "be", "been", "just", "only", "again", "now", "today", "money", "dont", "im",
}
# English words that Hindi ASR writes in Devanagari. Banking nouns and courtesy
# words carry no language signal; English function words are English evidence.
_DEVA_NEUTRAL = {
    "ओके", "ओक", "ओकै", "हेलो", "हैलो", "हलो", "हाय", "बाय", "थैंक", "थैंक्स", "थैंकयू", "थेंक",
    "प्लीज", "सॉरी", "सोरी", "बैलेंस", "बेलेंस", "बैलन्स", "ट्रांसफर", "ट्रान्सफर", "ट्रांसफार",
    "लिमिट", "कार्ड", "क्रेडिट", "डेबिट", "अकाउंट", "एकाउंट", "बैंक", "मनी", "चेक", "सर", "मैडम",
    "मैम", "एजेंट", "कस्टमर", "केयर", "सर्विस", "ऑफर", "प्रॉब्लम", "प्रोब्लम", "फोन", "कॉल",
    "ओटीपी", "पिन", "नंबर", "डेली", "टाइम", "गुड", "फाइन", "ग्रेट", "नाइस", "डन", "ओह", "हम्म",
    "मॉर्निंग", "इवनिंग", "यस", "नो", "कूल", "रिपीट", "अपडेट", "इंग्लिश", "एटीएम", "यूपीआई",
    "मैनेजर", "ब्लॉक", "स्टेटमेंट", "ऑनलाइन", "मोबाइल", "लोन", "ईएमआई", "पेमेंट", "रिक्वेस्ट",
}
_DEVA_ENGLISH = {
    "यू", "व्हाट", "इज", "माय", "आई", "वांट", "टू", "द", "कैन", "डू", "मी", "योर", "हाउ", "मच",
    "सेंड", "वेरी", "दिस", "दैट", "इट", "नॉट", "डोंट", "विद", "फॉर", "एंड", "प्लीज", "एम",
}


def _roman_scores(toks: list[str]) -> tuple[float, float]:
    hi = sum(1.0 if t in HINGLISH_STRONG else 0.5 if t in HINGLISH_WEAK else 0.0 for t in toks)
    en = sum(1.0 for t in toks if t in ENGLISH_MARKERS)
    return hi, en


def detect_language(text: str, neutral: Language | None = None) -> Language:
    """Devanagari Hindi -> 'hi'; Roman Hindi -> 'hinglish'; else 'en'.

    - An English sentence with one dropped-in Hindi word ("check my balance yaar") stays 'en'.
    - A Devanagari turn made only of English loanwords ("ओके थैंक यू") carries no signal:
      returns `neutral` if given, else 'hi' (the Hindi ASR channel was active). If it is
      clearly English function words ("व्हाट इज माय बैलेंस") it returns 'en'.
    - A Roman turn with no discriminative words ("ok", "hmm") returns `neutral` or 'en'.
    """
    toks = tokens(text)
    deva = [t for t in toks if _DEVANAGARI.search(t)]
    latin = [t for t in toks if not _DEVANAGARI.search(t) and not t.isdigit()]
    if deva:
        deva_en = sum(1 for t in deva if t in _DEVA_ENGLISH)
        deva_hi = [t for t in deva if t not in _DEVA_NEUTRAL and t not in _DEVA_ENGLISH]
        _, en_latin = _roman_scores(latin)
        if deva_hi and len(deva_hi) * 2 >= en_latin:
            return "hi"
        if not deva_hi:
            if deva_en + en_latin >= 2:
                return "en"
            if not latin:
                return neutral or "hi"
    hi, en = _roman_scores(latin)
    if (hi >= 2 and hi >= en * 0.5) or (hi >= 1 and hi >= en):
        return "hinglish"
    if hi == 0 and en == 0:
        return neutral or "en"
    return "en"


def jaccard(a: str, b: str) -> float:
    sa, sb = set(tokens(a)), set(tokens(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# --------------------------------------------------------------------------- spoken amounts

_EN_SMALL = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_EN_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fourty": 40, "fifty": 50, "sixty": 60,
            "seventy": 70, "eighty": 80, "ninety": 90}

# Hindi 1-99 are irregular, so they are a lookup table. Each entry: Devanagari
# spellings (canonical first), then common romanizations.
_HI_1_99: dict[int, tuple[str, ...]] = {
    1: ("एक", "ek", "ik"), 2: ("दो", "do"), 3: ("तीन", "teen", "tin"),
    4: ("चार", "char", "chaar"), 5: ("पांच", "पाच", "paanch", "panch", "paach", "pach"),
    6: ("छह", "छः", "छै", "छे", "chhe", "che", "chah", "chhah", "chheh", "chhah"),
    7: ("सात", "saat", "sat"), 8: ("आठ", "aath", "ath", "aat"), 9: ("नौ", "nau", "nao"),
    10: ("दस", "das", "dus"),
    11: ("ग्यारह", "ग्यारा", "gyarah", "gyara", "gyaarah", "gyaara", "giyara"),
    12: ("बारह", "बारा", "barah", "bara", "baarah", "baara"),
    13: ("तेरह", "terah", "tehra"),
    14: ("चौदह", "चौदा", "chaudah", "chauda", "chodah", "choda", "chaudha"),
    15: ("पंद्रह", "पन्द्रह", "पंद्रा", "pandrah", "pandra", "pundrah", "pandhra"),
    16: ("सोलह", "सोला", "solah", "sola"),
    17: ("सत्रह", "सतरह", "सत्रा", "satrah", "satra", "sattrah", "sattra"),
    18: ("अठारह", "अट्ठारह", "अठारा", "atharah", "athara", "attharah", "atthara", "athaarah"),
    19: ("उन्नीस", "उनीस", "unnis", "unees", "unnees", "unis"),
    20: ("बीस", "bees", "bis"),
    21: ("इक्कीस", "ikkees", "ikkis", "ekkis"), 22: ("बाईस", "baees", "bais", "baaees", "baais", "baaes"),
    23: ("तेईस", "teis", "teees", "teyis"), 24: ("चौबीस", "chaubees", "chaubis", "chobis", "choubees"),
    25: ("पच्चीस", "पचीस", "pachees", "pachchis", "pacchis", "pachis", "pachchees", "pacchees"),
    26: ("छब्बीस", "chhabbees", "chabbis", "chhabbis", "chabbees"),
    27: ("सत्ताईस", "sattaees", "sattais", "sattayis", "sataees"),
    28: ("अट्ठाईस", "अठ्ठाईस", "अठाईस", "atthaees", "athais", "atthais", "athaais", "athaees"),
    29: ("उनतीस", "untees", "untis", "unatis"), 30: ("तीस", "tees", "tis"),
    31: ("इकतीस", "iktees", "ikatis", "iktis", "ikattis"), 32: ("बत्तीस", "battees", "battis", "batis"),
    33: ("तैंतीस", "taintees", "tetis", "taintis", "tentees", "tentis"),
    34: ("चौंतीस", "chauntees", "chautis", "chontis", "chauntis", "chautees"),
    35: ("पैंतीस", "paintees", "paintis", "pentis", "pentees"),
    36: ("छत्तीस", "chhattees", "chattis", "chhattis", "chattees"),
    37: ("सैंतीस", "saintees", "saintis", "sentis", "sentees"),
    38: ("अड़तीस", "adtees", "artis", "adatis", "adtis", "artees"),
    39: ("उनतालीस", "untaalis", "untalis", "untalees", "unchalis"),
    40: ("चालीस", "chaalis", "chalis", "chalees", "chaalees"),
    41: ("इकतालीस", "iktaalis", "iktalis", "ektalis", "iktalees"),
    42: ("बयालीस", "बियालीस", "bayalis", "byalis", "bayaalis", "biyalis", "bayalees"),
    43: ("तैंतालीस", "taintaalis", "taintalis", "tentalis", "taintalees"),
    44: ("चवालीस", "चौवालीस", "chavalis", "chawalis", "chauvalis", "chauwalis", "chavalees"),
    45: ("पैंतालीस", "paintalis", "paintaalis", "pentalis", "paintalees"),
    46: ("छियालीस", "छयालीस", "chhiyalis", "chiyalis", "chhiyaalis", "chhiyalees"),
    47: ("सैंतालीस", "saintalis", "sentalis", "saintaalis", "saintalees"),
    48: ("अड़तालीस", "adtalis", "artalis", "adtaalis", "adtalees"),
    49: ("उनचास", "unchaas", "unchas"),
    50: ("पचास", "पच्चास", "pachaas", "pachas", "pachchas"),
    51: ("इक्यावन", "इक्कावन", "ikyavan", "ikyaavan", "ekyavan", "ikkavan"),
    52: ("बावन", "baavan", "bavan", "bawan"), 53: ("तिरपन", "tirpan", "trepan", "tirepan"),
    54: ("चौवन", "चौव्वन", "chauvan", "chauwan", "chovan", "chauvvan"),
    55: ("पचपन", "pachpan"), 56: ("छप्पन", "chhappan", "chappan"),
    57: ("सत्तावन", "sattavan", "sattaavan", "sattawan"),
    58: ("अट्ठावन", "अठावन", "atthavan", "athavan", "atthaavan", "athaavan"),
    59: ("उनसठ", "unsath", "unsaath"), 60: ("साठ", "saath", "sath"),
    61: ("इकसठ", "iksath", "iksaath"), 62: ("बासठ", "baasath", "basath"),
    63: ("तिरसठ", "tirsath", "tirsaath", "tresath"),
    64: ("चौंसठ", "चौसठ", "chaunsath", "chausath", "chausaath"),
    65: ("पैंसठ", "painsath", "pensath", "painsaath"),
    66: ("छियासठ", "chhiyasath", "chiyasath", "chhiyaasath"),
    67: ("सड़सठ", "सरसठ", "sadsath", "sarsath", "sadsaath"),
    68: ("अड़सठ", "adsath", "arsath", "adsaath"), 69: ("उनहत्तर", "unhattar", "unhatar"),
    70: ("सत्तर", "sattar"), 71: ("इकहत्तर", "ikhattar", "ikahattar"),
    72: ("बहत्तर", "bahattar", "behattar", "bahatar"),
    73: ("तिहत्तर", "tihattar", "tehattar", "tihatar"),
    74: ("चौहत्तर", "chauhattar", "chohattar", "chauhatar"),
    75: ("पचहत्तर", "pachhattar", "pachattar", "pachahattar", "pachhatar"),
    76: ("छिहत्तर", "chhihattar", "chihattar"),
    77: ("सतहत्तर", "satattar", "sathattar", "satahattar", "sathhattar"),
    78: ("अठहत्तर", "athhattar", "athattar", "athahattar"),
    79: ("उन्यासी", "उनासी", "उन्नासी", "unaasi", "unasi", "unyasi", "unyaasi"),
    80: ("अस्सी", "assi", "assee", "asi"), 81: ("इक्यासी", "ikyaasi", "ikyasi"),
    82: ("बयासी", "बियासी", "bayaasi", "bayasi", "biyasi"),
    83: ("तिरासी", "tiraasi", "tirasi"), 84: ("चौरासी", "chauraasi", "chaurasi", "chorasi"),
    85: ("पचासी", "pachaasi", "pachasi"), 86: ("छियासी", "chhiyaasi", "chhiyasi", "chiyasi"),
    87: ("सत्तासी", "sattaasi", "sattasi"), 88: ("अट्ठासी", "atthaasi", "atthasi", "athasi"),
    89: ("नवासी", "navaasi", "navasi", "nawasi"), 90: ("नब्बे", "nabbe", "nabbay"),
    91: ("इक्यानवे", "ikyaanve", "ikyanve", "ikyanave"), 92: ("बानवे", "baanve", "banve", "baanave"),
    93: ("तिरानवे", "tiraanve", "tiranve"), 94: ("चौरानवे", "chauraanve", "chauranve"),
    95: ("पचानवे", "pachaanve", "pachanve"), 96: ("छियानवे", "chhiyaanve", "chhiyanve", "chiyanve"),
    97: ("सत्तानवे", "sattaanve", "sattanve"), 98: ("अट्ठानवे", "atthaanve", "atthanve", "athanve"),
    99: ("निन्यानवे", "ninyaanve", "ninyanve", "ninnyanve", "ninanve"),
}


def _deva_variants(w: str) -> set[str]:
    """Systematic Devanagari spelling variation seen in ASR output."""
    out = {w}
    for a, b in (("ई", "इ"), ("ैं", "ें"), ("ौं", "ों"), ("ौं", "ौ"), ("वे", "बे"), ("ठ्ठ", "ट्ठ"),
                 ("ट्ठ", "ठ"), ("ड", "र")):
        out |= {v.replace(a, b) for v in out if a in v}
    return out


_SMALL: dict[str, int] = {}
for _n, _forms in _HI_1_99.items():
    for _f in _forms:
        for _v in (_deva_variants(normalize(_f)) if _DEVANAGARI.search(_f) else {_f}):
            _SMALL.setdefault(_v, _n)
_SMALL.update(_EN_SMALL)
_SMALL.update(_EN_TENS)

# Number words that are also common non-number words: only count when a
# multiplier or currency word makes the amount reading explicit.
_AMBIGUOUS = {"do", "sat", "das", "tin", "bis", "char", "bara", "sola", "assi", "asi", "sath",
              "saath", "saat", "nau", "nao", "tis", "ek", "ik", "che", "ath", "aat", "pach", "unis",
              "one", "ten", "दो", "एक", "satra", "choda", "chauda", "tehra", "sattar", "bavan", "बारा", "सोला"}
_FRACTION = {"dedh": 1.5, "derh": 1.5, "dhed": 1.5, "deedh": 1.5, "डेढ": 1.5,
             "dhai": 2.5, "dhaai": 2.5, "dhaayi": 2.5, "adhai": 2.5, "adhaai": 2.5, "ढाई": 2.5, "अढाई": 2.5}
_MODIFIER = {"sava": 0.25, "sawa": 0.25, "savva": 0.25, "सवा": 0.25,
             "saadhe": 0.5, "sadhe": 0.5, "saade": 0.5, "sade": 0.5, "साढे": 0.5,
             "paune": -0.25, "pone": -0.25, "paunay": -0.25, "पौने": -0.25}
_HUNDRED = {"hundred", "sau", "saw", "सौ"}
_BIG = {"thousand": 1_000, "thousands": 1_000, "hazaar": 1_000, "hazar": 1_000, "hajar": 1_000,
        "hajaar": 1_000, "hazzar": 1_000, "hazaaar": 1_000, "hazaron": 1_000, "हजार": 1_000,
        "हजर": 1_000, "हजारो": 1_000, "हज्जार": 1_000,  # Whisper spellings seen on real calls
        "k": 1_000, "lakh": 100_000, "lakhs": 100_000, "lac": 100_000, "lacs": 100_000,
        "laakh": 100_000, "lak": 100_000, "लाख": 100_000, "million": 1_000_000,
        "crore": 10_000_000, "crores": 10_000_000, "karod": 10_000_000, "karor": 10_000_000,
        "करोड": 10_000_000}
_CURRENCY = {"₹", "rs", "inr", "rupees", "rupee", "rupaye", "rupaiye", "rupay", "rupaya", "rupiya",
             "rupye", "rupe", "रुपये", "रुपए", "रूपये", "रूपए", "रुपया", "रुपयों", "रुपय",
             "रु", "रुपीस", "रूपीस", "bucks"}
# The number right after these is an identifier, not money.
_BLOCK_BEFORE = {"otp", "pin", "cvv", "mpin", "digits", "digit", "number", "num", "date", "tarikh",
                 "tareekh", "ending", "ends", "reference", "ref", "mobile", "phone", "code",
                 "password", "ओटीपी", "पिन", "नंबर", "तारीख", "अंक", "कोड", "पासवर्ड", "डिजिट"}
_BLOCK_ADJACENT = {"last", "no", "lastly", "aakhri", "akhri", "आखिरी", "लास्ट", "option", "press"}
# The number right before these is a time/date/count, not money.
_BLOCK_AFTER = {
    "minute", "minutes", "min", "mins", "second", "seconds", "sec", "secs", "hour", "hours", "hrs",
    "ghanta", "ghante", "baje", "bajey", "din", "day", "days", "week", "weeks", "hafte", "hafta",
    "mahina", "mahine", "month", "months", "saal", "sal", "year", "years", "baar", "times", "time",
    "tarikh", "tareekh", "th", "st", "nd", "rd", "pm", "am", "baje", "percent", "digit", "digits",
    "january", "february", "march", "april", "may", "june", "july", "august", "september",
    "october", "november", "december", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep",
    "sept", "oct", "nov", "dec", "मिनट", "घंटे", "घंटा", "बजे", "दिन", "हफ्ते", "महीने", "महीना",
    "साल", "बार", "तारीख", "प्रतिशत", "जनवरी", "फरवरी", "मार्च", "अप्रैल", "मई", "जून", "जुलाई",
    "अगस्त", "सितंबर", "अक्टूबर", "नवंबर", "दिसंबर", "डिजिट", "अंक",
}

_AMT_TOKEN = re.compile(r"₹|\d[\d,]*(?:\.\d+)?|[a-zऀ-ॣॱ-ॿ]+")


def _amount_tokens(text: str) -> list[str]:
    return [t.rstrip(",") for t in _AMT_TOKEN.findall(normalize(text))]


def _is_digit(t: str) -> bool:
    return t[:1].isdigit()


def _digit_value(t: str) -> float:
    return float(t.replace(",", ""))


def _starts_number(toks: list[str], i: int) -> bool:
    t = toks[i]
    if _is_digit(t) or t in _SMALL or t in _FRACTION or t in _MODIFIER:
        return True
    nxt = toks[i + 1] if i + 1 < len(toks) else ""
    return t in ("a", "an") and (nxt in _BIG or nxt in _HUNDRED)


def _parse_phrase(toks: list[str], i: int) -> tuple[float, int, dict]:
    """Greedy Indian-number-system parser starting at toks[i].
    Returns (value, index after phrase, info)."""
    total, cur = 0.0, None
    last_big = float("inf")
    mod = 0.0
    info = {"mult": False, "digit": False, "ambiguous_only": True, "words": 0}
    prev_kind = ""
    seg_start, seg_ok = i, True  # trailing segment after a big multiplier
    k = i
    n = len(toks)
    while k < n:
        t = toks[k]
        nxt = toks[k + 1] if k + 1 < n else ""
        if cur is None and total and prev_kind in ("big", "and", "mod"):
            if prev_kind == "big":
                seg_start, seg_ok = k, False
            elif prev_kind == "and":
                seg_ok = True
        if _is_digit(t):
            if cur is not None:
                break
            if "," not in t and "." not in t and len(t) >= 9:  # phone / card / account number
                info["ambiguous_only"] = True
                info["identifier"] = True
                cur = _digit_value(t)
                k += 1
                break
            cur = _digit_value(t) + mod
            mod = 0.0
            info["digit"] = True
            info["ambiguous_only"] = False
            prev_kind = "digit"
        elif t in _SMALL:
            v = _SMALL[t]
            if cur is None:
                cur = v + mod
                mod = 0.0
            elif prev_kind in ("hundred",) and cur % 100 == 0 and v < 100:
                cur += v
            elif prev_kind == "tens" and v < 10:
                cur += v
            elif prev_kind == "and" and v < 100 and cur % 100 == 0:
                cur += v
            else:
                break
            info["words"] += 1
            if t not in _AMBIGUOUS:
                info["ambiguous_only"] = False
            prev_kind = "tens" if t in _EN_TENS else "small"
        elif t in _FRACTION:
            if cur is not None:
                break
            cur = _FRACTION[t]
            info["ambiguous_only"] = False
            prev_kind = "frac"
        elif t in _MODIFIER:
            if cur is not None or mod:
                break
            mod = _MODIFIER[t]
            info["ambiguous_only"] = False
            prev_kind = "mod"
        elif t in ("a", "an") and k == i:
            cur = 1.0
            prev_kind = "small"
        elif t in _HUNDRED or (t in ("so", "सो") and cur is not None and prev_kind == "small"
                                and not info["ambiguous_only"]):
            if prev_kind == "hundred" or (cur is not None and cur >= 100 and prev_kind != "digit"):
                break
            base = cur if cur is not None else 1.0 + mod
            cur = base * 100
            mod = 0.0
            info["mult"] = True
            seg_ok = True
            prev_kind = "hundred"
        elif t in _BIG and (t != "k" or prev_kind in ("digit", "small", "tens")):
            m = _BIG[t]
            if m >= last_big:
                break  # a new amount starts; the trailing segment is dropped below
            base = cur if cur is not None else 1.0 + mod
            total += base * m
            cur = None
            mod = 0.0
            last_big = m
            info["mult"] = True
            seg_ok = True
            prev_kind = "big"
        elif t == "and" and (cur is not None or total) and (nxt in _SMALL or _is_digit(nxt)):
            prev_kind = "and"
        elif t == "point" and cur is not None and nxt in _SMALL and _SMALL[nxt] < 10:
            cur += _SMALL[nxt] / 10
            k += 1
        else:
            break
        k += 1
    if prev_kind == "and":
        k -= 1
    if total and cur is not None and not seg_ok:
        # "pachas hazaar do" (do = give), "pachas hazaar ek lakh": the bare number
        # after the multiplier is not part of this amount.
        return total, seg_start, info
    return total + (cur or 0.0), k, info


def find_amounts(text: str) -> list[int]:
    """All rupee amounts in `text`, in order. Skips identifiers (OTP, PIN, last 4
    digits, account/phone numbers), dates and durations where detectable."""
    toks = _amount_tokens(text)
    out: list[int] = []
    i = 0
    while i < len(toks):
        if not _starts_number(toks, i):
            i += 1
            continue
        val, j, info = _parse_phrase(toks, i)
        if j == i:
            i += 1
            continue
        prev = toks[i - 1] if i > 0 else ""
        nxt = toks[j] if j < len(toks) else ""
        window = toks[max(0, i - 2):i]
        currency = prev in _CURRENCY or nxt in _CURRENCY or (i > 1 and toks[i - 2] in _CURRENCY)
        explicit = info["mult"] or currency
        blocked = (
            info.get("identifier")
            or (not explicit and (any(w in _BLOCK_BEFORE for w in window) or prev in _BLOCK_ADJACENT))
            or (not info["mult"] and nxt in _BLOCK_AFTER)
        )
        if info["digit"] and not info["mult"]:
            ok = val >= 10 or currency
        else:
            ok = explicit and not (info["ambiguous_only"] and not info["mult"] and not currency)
        if ok and not blocked and val > 0:
            out.append(int(round(val)))
        i = j
    return out


def parse_amount(text: str) -> int | None:
    """First rupee amount: '50,000', '₹50000', '50k', '50 hazaar', '1.5 lakh',
    'twenty five thousand', 'pachas hazaar', 'dedh lakh', 'पच्चीस हजार रुपये', '५०,०००'."""
    found = find_amounts(text)
    return found[0] if found else None


def inr(amount: int) -> str:
    """Indian digit grouping: 1250000 -> ₹12,50,000."""
    s = str(abs(int(amount)))
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        head = re.sub(r"(\d)(?=(\d\d)+$)", r"\1,", head)
        s = f"{head},{tail}"
    return f"₹{s}"
