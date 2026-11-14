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
_MIRROR = re.compile(r"(?<![\wऀ-ॿ])(bhai|bhaiya|yaar|yar|boss|bro|यार|भाई|बॉस)(?![\wऀ-ॿ])", re.I)
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
    if tone == "mid" and _MIRROR.search(text or ""):
        dialog["_mirror"] = "1"
    if dialog.get("_elder") == "1":
        return "mid" if tone == "casual" else tone
    if tone == "mid" and dialog.get("_mirror") == "1":
        return "casual"
    return tone


_LEAD = {"professional": {"en": "Sir, ", "hi": "सर, ", "hinglish": "Sir, "}}
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
