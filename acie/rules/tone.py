"""Tone (register) for every reply: professional | mid | casual, per product/campaign (`tone_level`, default mid).

  effective_tone()  the register this caller actually gets: an elderly caller (age >= 55, or "beta" in their words) always gets
                    a respectful register (casual drops to mid: "aap", no bhai/yaar), and a mid tone drifts casual when the caller
                    themselves says bhai/yaar/boss (mirroring).
  restyle()         deterministic register for a template reply that is not tone-aware (pitch, answers, closing, goodbye): a short address
                    lead ("Sir," / "Bhai,") and a few word swaps. The legal content of the sentence is never touched.
Everything here is pure and deterministic (zlib.crc32 for variant choice, never hash()).
"""

from __future__ import annotations

import re
import zlib

TONES = ("professional", "mid", "casual")
_ELDER = re.compile(r"(?<![\wऀ-ॿ])(beta|beta ji|beti|बेटा|बेटे|बेटी)(?![\wऀ-ॿ])", re.I)
_MIRROR = re.compile(r"(?<![\wऀ-ॿ])(bhai|bhaiya|yaar|yar|boss|यार|भाई|बॉस)(?![\wऀ-ॿ])", re.I)
ELDER_AGE = 55
# replies that keep their legal / safety wording exactly: no address lead is added
SKIP_INTENTS = {"greet", "opening", "opt_out", "safety_distress", "sensitive_stop", "are_you_ai", "identify", "number_source", "human_agent"}
_MARK = re.compile(r"(?<![\wऀ-ॿ])(sir|ma'am|madam|सर|मैम|bhai|boss|yaar|भाई|बॉस|यार)(?![\wऀ-ॿ])", re.I)


def clean(t: str | None) -> str:
    return t if t in TONES else "mid"


def effective_tone(configured: str | None, dialog: dict, text: str = "", age: int | None = None) -> str:
    """The register for this turn; remembers the elderly / mirror signals in the dialog so they hold for the whole call."""
    tone = clean(configured)
    if (age is not None and age >= ELDER_AGE) or _ELDER.search(text or ""):
        dialog["_elder"] = "1"
    try:
        calm = float(dialog.get("_frustration", "0") or 0) < 0.3
    except ValueError:
        calm = True
    if tone == "mid" and calm and _MIRROR.search(text or ""):  # only when the caller used these words themselves, and is not upset
        dialog["_mirror"] = "1"
    if dialog.get("_elder") == "1":
        return "mid" if tone == "casual" else tone
    if tone == "mid" and dialog.get("_mirror") == "1":
        return "casual"
    return tone


_LEAD = {"professional": {"en": "Sir, ", "hi": "सर, ", "hinglish": "Sir, "}}   # normalised by address() to the caller's address word
_FEMALE = {"pooja", "priya", "neha", "anjali", "sneha", "kavita", "sunita", "anita", "rekha", "meena", "sita", "geeta", "seema", "ritu", "nisha", "swati", "divya",
           "shweta", "komal", "sakshi", "ananya", "aarti", "preeti", "ritika", "simran", "kajal", "manisha", "payal", "radha", "lakshmi", "deepika", "shalini"}
_MALE = {"rahul", "rohit", "ravi", "amit", "raj", "suresh", "ramesh", "vikram", "arjun", "sanjay", "ajay", "vijay", "manoj", "deepak", "anil", "sunil", "mohit",
         "yash", "karan", "nikhil", "aman", "vivek", "sachin", "gaurav", "rajesh", "mukesh", "harsh", "kunal", "varun", "ankit", "pankaj"}


def who(name: str | None) -> str:
    """sir | ma'am | ji: never a default "Sir". Only a clearly male / clearly female first name picks a gendered word; otherwise "ji"."""
    first = (name or "").strip().split(" ")[0].lower()
    return "ma'am" if first in _FEMALE else "sir" if first in _MALE else "ji"


_ADDR = {"sir": ("Sir", "सर"), "ma'am": ("Ma'am", "मैम"), "ji": ("Ji", "जी")}
_PRO_M = r"(?:Sir|Ma'am|Ji|सर|मैम|जी)"
_CAS_M = r"(?:Bhai|Boss|Yaar|भाई|बॉस|यार)"


def _once(text: str, marker: str, repl: str | None) -> str:
    """Keep the first address word in a reply (swapped for `repl` if given), remove every later one."""
    rx = re.compile(r"(?<![\wऀ-ॿ])" + marker + r"(?![\wऀ-ॿ])", re.I)
    ms = list(rx.finditer(text))
    if not ms:
        return text
    head = text[:ms[0].start()] + (repl or ms[0].group(0)) + text[ms[0].end():]
    cut = ms[0].start() + len(repl or ms[0].group(0))
    tail = head[cut:]
    tail = re.sub(r"(^|[.?!।]\s+)" + marker + r",\s+(\S)", lambda m: m.group(1) + m.group(2).upper(), tail, flags=re.I)
    tail = re.sub(r",\s*" + marker + r"(?=[?.!।]|$)", "", tail, flags=re.I)
    tail = re.sub(r"\s+" + marker + r"(?=,)", "", tail, flags=re.I)
    tail = re.sub(r"(?<![\wऀ-ॿ])" + marker + r"(?![\wऀ-ॿ]),?\s*", "", tail, flags=re.I)
    return head[:cut] + tail


def address(fallback: dict[str, str], tone: str, addr: str) -> dict[str, str]:
    """One address word per reply. Professional: the caller's address word (sir / ma'am / ji); casual: the first bhai/boss/yaar only."""
    out = {}
    for l, t in fallback.items():
        if tone == "professional":
            en, hi = _ADDR[addr]
            out[l] = _once(t, _PRO_M, hi if l == "hi" else en)
        elif tone == "casual":
            out[l] = _once(t, _CAS_M, None)
        else:
            out[l] = t
    return out
_CASUAL_LEADS = {"en": ("Bhai, ", "Boss, ", "Bhai, "), "hi": ("भाई, ", "बॉस, ", "भाई, "), "hinglish": ("Bhai, ", "Boss, ", "Bhai, ")}
_LOWER_OK = {"ye", "yeh", "aap", "aapka", "aapki", "kya", "main", "bas", "theek", "badhiya", "koi", "is", "iske", "haan", "bilkul", "achha",
             "zaroor", "dekhiye", "the", "it", "it's", "that", "that's", "here", "here's", "that", "this", "sure", "great", "okay", "ok", "yes", "no", "of", "alright", "you", "we", "there", "to", "ji", "samajh", "samajhta"}
_SWAPS = {
    "professional": [(r"\bBadhiya\b", "Bahut achha"), (r"\bchalo\b", "chaliye"), (r"\bChalo\b", "Chaliye"), (r"\bkoi baat nahi\b", "koi baat nahi, Sir"),
                     (r"\bOkay\b", "Certainly")],
    "casual": [(r"\bBilkul\b", "Bilkul bhai"), (r"\bKoi baat nahi\b", "Koi baat nahi bhai")],
}


def _lead_for(text: str, lang: str, tone: str) -> str:
    if tone == "professional":
        return _LEAD["professional"].get(lang, "Sir, ")
    leads = _CASUAL_LEADS.get(lang, _CASUAL_LEADS["hinglish"])
    return leads[zlib.crc32(text.encode("utf-8")) % len(leads)]


def restyle_text(text: str, lang: str, tone: str) -> str:
    tone = clean(tone)
    if tone == "mid" or not text or _MARK.search(text):
        return text
    out = text
    for rx, rep in _SWAPS.get(tone, []):
        if lang != "hi":
            out = re.sub(rx, rep, out, count=1)
    if _MARK.search(out):  # a swap already added the address
        return out
    first = re.match(r"[A-Za-z']+", out)
    if first and first.group(0).lower() in _LOWER_OK and lang != "hi":
        out = out[0].lower() + out[1:]
    return _lead_for(out, lang, tone) + out


def restyle(fallback: dict[str, str], tone: str) -> dict[str, str]:
    return {l: restyle_text(t, l, tone) for l, t in fallback.items()}
