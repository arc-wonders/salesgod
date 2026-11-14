"""Layer 2 - Human State Estimator.

v1 is rule/signal based so it works with zero training data. The `RawScorer`
protocol is the seam where a trained classifier (e.g. XLM-R / MuRIL fine-tuned
on synthetic + labeled turns) replaces `RuleScorer` later. Smoothing and
evidence tracking stay the same either way.
"""

from __future__ import annotations

import logging
import os

import re
from typing import Protocol

import numpy as np

from ..schemas import HumanState, SessionState, TurnInput
from ..text import jaccard, tokens
from . import dims as _dims
from .dims import state_dims  # noqa: F401  (re-exported)

log = logging.getLogger("acie.state")

# Lexicons: phrase -> weight. Phrases are matched on whole words over normalized
# text (see acie.text.normalize); "x*" allows a suffix on the last word; "=x"
# means the whole utterance (fillers removed). The per-turn score of a dimension
# is its strongest phrase plus a small bonus for each further distinct hit.
# Weights: 0.8 = one turn should cross the policy threshold; 0.5 = needs support
# (repetition, barge-in, a second phrase); <=0.3 = weak hint.
CONFUSION_PHRASES: dict[str, float] = {
    # English
    "what do you mean": 0.8, "didnt understand": 0.8, "dont understand": 0.8,
    "did not understand": 0.8, "do not understand": 0.8, "not clear": 0.7, "i dont get it": 0.8,
    "didnt get that": 0.7, "didnt get you": 0.7, "confused": 0.8, "confusing": 0.7, "im lost": 0.7,
    "lost me": 0.7, "come again": 0.7, "sorry what": 0.7, "say that again": 0.6, "say again": 0.6,
    "repeat": 0.6, "pardon": 0.7, "huh": 0.7, "which one": 0.6, "which limit": 0.6,
    "what does that mean": 0.8, "meaning": 0.4, "again": 0.3,
    # Hinglish
    "samajh nahi": 0.8, "samjha nahi": 0.8, "samjhi nahi": 0.8, "samaj nahi": 0.8,
    "samajh nai": 0.8, "samajh nhi": 0.8, "samjh nahi": 0.8, "nahi samjha": 0.8, "nahi samjhi": 0.8,
    "nahi samajh*": 0.8, "samajh mein nahi": 0.8, "samajh me nahi": 0.8, "kya matlab": 0.8,
    "matlab kya": 0.8, "kya bola": 0.6, "kya kaha": 0.6, "kya bol rahe": 0.6, "phir se": 0.5,
    "fir se": 0.5, "dobara": 0.5, "dubara": 0.5, "kaun sa": 0.4, "konsa": 0.4, "kaunsa": 0.4,
    "confuse ho": 0.8, "confusion": 0.6, "नहीं समझते": 0.8, "समझ नहीं आता": 0.8, "समझ नहीं आया": 0.8,
    "क्या बोले": 0.6, "फिर से बोलो": 0.6, "धीरे बोलो": 0.6, "ये क्या होता है": 0.6, "क्या होता है": 0.4,
    "samajh nahi aaya": 0.8, "kya bole": 0.6, "dheere bolo": 0.6,
    # Devanagari
    "समझ नहीं": 0.8, "समझा नहीं": 0.8, "समझी नहीं": 0.8, "नहीं समझ*": 0.8, "समझ में नहीं": 0.8,
    "क्या मतलब": 0.8, "मतलब क्या": 0.8, "क्या बोला": 0.6, "क्या बोले": 0.6, "क्या कहा": 0.6,
    "फिर से": 0.5, "दोबारा": 0.5, "दुबारा": 0.5, "कौन सा": 0.4, "कौनसा": 0.4, "कन्फ्यूज": 0.8,
    # whole-utterance confusion ("matlab?" / "sorry?" / "huh")
    "=matlab": 0.8, "=मतलब": 0.8, "=sorry": 0.7, "=what": 0.7, "=kya": 0.7, "=क्या": 0.7,
    "=hain": 0.7, "=हैं": 0.7, "=kaun sa": 0.6, "=कौन सा": 0.6, "=which": 0.6, "=excuse me": 0.6,
}
CONFUSION_MASKS = ["no confusion", "koi confusion nahi", "samajh gaya", "samajh gayi", "समझ गया",
                   "समझ गई", "phir se transfer*", "phir se bhej*", "फिर से भेज*", "फिर से ट्रांसफर*",
                   "again thank*", "repeat transaction"]

FRUSTRATION_PHRASES: dict[str, float] = {
    # English
    "how many times": 0.5, "how many times do i": 0.8, "how many times have i": 0.8,
    "i already told you": 0.8, "already told you": 0.8, "i told you": 0.6, "told you already": 0.8,
    "i just said": 0.6, "are you even listening": 0.8, "you are not listening": 0.8,
    "youre not listening": 0.8, "not listening": 0.6, "useless": 0.8, "ridiculous": 0.8,
    "this is ridiculous": 0.9, "pathetic": 0.8, "worst": 0.7, "stupid": 0.8, "nonsense": 0.8,
    "fed up": 0.8, "irritating": 0.8, "annoying": 0.7, "frustrating": 0.8, "frustrated": 0.8,
    "waste of time": 0.8, "wasting my time": 0.8, "terrible": 0.7, "horrible": 0.7,
    "disgusting": 0.8, "unacceptable": 0.8, "what the hell": 0.8, "wtf": 0.8, "damn": 0.5,
    "bloody": 0.6, "seriously": 0.3, "come on": 0.4, "ugh": 0.4, "rubbish": 0.8,
    # Hinglish
    "kitni baar": 0.5, "kitni bar": 0.5, "kitne baar": 0.5, "kitni baar bol*": 0.8,
    "kitni baar bata*": 0.8, "kitni baar samjha*": 0.8, "kitni baar puch*": 0.8,
    "maine bola na": 0.8, "maine pehle hi bola": 0.8, "pehle hi bola": 0.8, "pehle hi bata*": 0.8,
    "pehle bhi bola": 0.7, "bola na": 0.5, "bakwas": 0.8, "bakwaas": 0.8, "kya bakwas": 0.9,
    "bekar": 0.8, "bekaar": 0.8, "faltu": 0.7, "pagal": 0.8, "bas karo": 0.8, "hadd hai": 0.8,
    "had hai": 0.8, "hadd ho gayi": 0.8, "kya mazak": 0.8, "mazak hai": 0.7, "mazaak hai": 0.7,
    "time waste": 0.8, "dimag kharab": 0.8, "dimaag kharab": 0.8, "pareshan": 0.6,
    "tang aa gaya": 0.8, "tang aa gayi": 0.8, "ghatiya": 0.8, "sun nahi rahe": 0.7,
    "sun hi nahi": 0.8, "arre yaar": 0.5, "are yaar": 0.5, "arey yaar": 0.5, "kya yaar": 0.5,
    "yaar": 0.2, "uff": 0.4, "ufff": 0.4, "ye kya hai": 0.5, "yeh kya hai": 0.5, "kya hai ye": 0.5,
    "kya hai yeh": 0.5, "kab se": 0.4, "kab tak": 0.5,
    # Devanagari
    "कितनी बार": 0.5, "कितनी बार बोल*": 0.8, "कितनी बार बता*": 0.8, "कितनी बार समझा*": 0.8,
    "मैंने पहले ही बोला": 0.8, "पहले ही बोला": 0.8, "पहले ही बता*": 0.8, "मैंने बोला ना": 0.8,
    "बोला ना": 0.5, "बकवास": 0.8, "बेकार": 0.8, "फालतू": 0.7, "पागल": 0.8, "हद है": 0.8,
    "हद हो गई": 0.8, "हद हो गयी": 0.8, "बस करो": 0.8, "मजाक": 0.6, "टाइम वेस्ट": 0.8,
    "दिमाग खराब": 0.8, "परेशान": 0.6, "तंग आ गया": 0.8, "तंग आ गई": 0.8, "घटिया": 0.8,
    "सुन नहीं रहे": 0.7, "सुन ही नहीं": 0.8, "अरे यार": 0.5, "क्या यार": 0.5, "यार": 0.2, "उफ्फ": 0.4,
    "उफ": 0.4, "ये क्या है": 0.5, "यह क्या है": 0.5, "क्या है ये": 0.5, "कब से": 0.4, "कब तक": 0.5,
    "ये क्या बकवास": 0.9,
    # seen on live calls (the agent repeating itself / talking over the caller)
    "बार बार": 0.6, "बार-बार": 0.6, "baar baar": 0.6, "bar bar": 0.5, "रिपेट": 0.6, "रिपीट": 0.6,
    "repeat kyu": 0.8, "repeat kyon": 0.8, "रिपेट क्यों": 0.8, "रिपीट क्यों": 0.8, "एक ही चीज": 0.6,
    "एक चीज": 0.4, "वही बात": 0.6, "wahi baat": 0.6, "same baat": 0.6, "same thing": 0.5,
    "keep repeating": 0.8, "repeating the same": 0.8, "again and again": 0.6, "पागले": 0.8, "पगले": 0.8,
    "बोले जा रहे": 0.8, "बोले जा रहा": 0.8, "bole ja rahe": 0.8, "bole ja raha": 0.8, "चुप": 0.6,
    "chup": 0.6, "shut up": 0.9, "stop talking": 0.8, "मत बोलो": 0.6, "fuck": 0.8, "fucking": 0.9,
    "not listening": 0.7, "you are not listening": 0.8, "सुनते ही नहीं": 0.8, "सुन ही नहीं": 0.7, "sunte hi nahi": 0.8,
    "same line": 0.6, "same thing again": 0.7, "repeat kar rahe": 0.7, "एक ही बात": 0.6, "फिर वही": 0.7, "phir wahi": 0.7,
    "dodge": 0.6, "dodged": 0.7, "third time": 0.6, "तीसरी बार": 0.6, "price repeat": 0.6,
    "bullshit": 0.9, "shit": 0.8, "answer me": 0.6, "answer my question": 0.7, "you didnt answer": 0.7,
    "जवाब दो": 0.5, "जवाब नहीं दिया": 0.7, "ढंग से": 0.5, "खराब चल": 0.6, "jawab do": 0.5, "bc": 0.5, "saala": 0.6, "साला": 0.6, "chutiya": 0.9, "चूतिया": 0.9,
    # Whisper censors swearing ("Why the f**k is it so cheap?" scored 0.06 on a replayed real call)
    "f**k": 0.8, "f**king": 0.9, "f*ck": 0.8, "is it fake": 0.5, "it is fake": 0.5, "scam": 0.6,
    "नकली है": 0.5, "nakli hai": 0.5,
}
FRUSTRATION_MASKS = ["how many times can*", "how many times a day", "how many transactions",
                     "kitni baar transfer*", "kitni baar kar sak*", "kitni baar bhej sak*",
                     "कितनी बार ट्रांसफर*", "कितनी बार कर सक*", "कितनी बार भेज सक*",
                     "worst case", "no worries", "not bad", "koi pareshani nahi", "pareshan mat",
                     "kab se kab tak", "ye kya hai card", "yeh kya hai offer", "kab tak aayega"]

URGENCY_PHRASES: dict[str, float] = {
    # English
    "urgent": 0.7, "urgently": 0.7, "emergency": 0.9, "hospital": 0.8, "accident": 0.8,
    "immediately": 0.7, "right now": 0.7, "right away": 0.7, "asap": 0.7, "hurry": 0.7,
    "quickly": 0.6, "quick": 0.4, "fast": 0.4, "no time": 0.6, "need it now": 0.7, "medical": 0.5,
    # Hinglish
    "jaldi": 0.7, "jaldi jaldi": 0.8, "jaldi karo": 0.8, "jaldi se": 0.7, "turant": 0.8,
    "abhi ke abhi": 0.9, "abhi turant": 0.9, "abhi chahiye": 0.6, "abhi": 0.3, "urgent hai": 0.8,
    "time nahi hai": 0.7, "time nahi": 0.6, "fauran": 0.8, "aspatal": 0.8, "ilaj": 0.5,
    # Devanagari
    "जल्दी": 0.7, "जल्दी जल्दी": 0.8, "जल्दी करो": 0.8, "जल्दी से": 0.7, "तुरंत": 0.8,
    "अभी के अभी": 0.9, "अभी तुरंत": 0.9, "अभी": 0.3, "अर्जेंट": 0.7, "इमरजेंसी": 0.9,
    "अस्पताल": 0.8, "हॉस्पिटल": 0.8, "टाइम नहीं है": 0.7, "समय नहीं है": 0.7, "फौरन": 0.8,
}
URGENCY_MASKS = [
    "no rush", "no hurry", "not urgent", "not in a hurry", "not in hurry", "take your time",
    "no need to hurry", "not right now", "not now", "isnt urgent", "nothing urgent",
    "koi jaldi nahi", "jaldi nahi", "jaldi nahin", "jaldi koi nahi", "urgent nahi",
    "koi urgent nahi", "jaldbazi nahi", "aaram se", "araam se", "abhi nahi", "abhi nahin",
    "कोई जल्दी नहीं", "जल्दी नहीं", "जल्दी कोई नहीं", "अर्जेंट नहीं", "आराम से", "अभी नहीं",
    "abhi tak", "अभी तक",
]
LOW_ENGAGEMENT_REPLIES = {"ok", "okay", "okk", "hmm", "hmmm", "hm", "mm", "haan", "han", "ha", "achha",
                          "acha", "accha", "theek", "thik", "fine", "k", "hai", "ji", "हम्म", "हां",
                          "ओके", "ठीक", "है", "अच्छा", "जी", "हु", "हूं"}
QUESTION_WORDS = {"what", "how", "why", "which", "when", "where", "kya", "kaise", "kitna", "kitni",
                  "kitne", "kyun", "kyu", "kab", "kaun", "kahan", "क्या", "कैसे", "कितना", "कितनी",
                  "कितने", "क्यों", "कब", "कौन", "कहां"}
_FILLERS = {"um", "umm", "ummm", "uh", "uhh", "er", "erm", "ah", "sir", "madam", "maam", "please",
            "ji", "जी", "सर", "मैडम", "उम", "अं"}


class RawScorer(Protocol):
    def score(self, turn: TurnInput, session: SessionState) -> tuple[dict[str, float], list[str]]:
        """Return per-turn raw scores in [0,1] for each dimension plus evidence strings."""


class _Lexicon:
    def __init__(self, phrases: dict[str, float], masks: list[str] = ()):
        self.word: list[tuple[str, str, float]] = []
        self.prefix: list[tuple[str, str, float]] = []
        self.exact: dict[str, tuple[str, float]] = {}
        for p, w in phrases.items():
            if p.startswith("="):
                self.exact[" ".join(tokens(p[1:]))] = (p, w)
            elif p.endswith("*"):
                self.prefix.append((p, " " + " ".join(tokens(p[:-1])), w))
            else:
                self.word.append((p, " " + " ".join(tokens(p)) + " ", w))
        self.masks = [re.compile(r"(?<= )" + re.escape(" ".join(tokens(m.replace("*", " "))))
                                 + (r"\S*" if m.endswith("*") else "") + r"(?= )")
                      for m in sorted(masks, key=len, reverse=True)]

    def hits(self, low: str, bare: str) -> list[tuple[str, float]]:
        for m in self.masks:
            low = m.sub("|", low)
        out = [(p, w) for p, k, w in self.word if k in low]
        out += [(p, w) for p, k, w in self.prefix if k in low]
        if bare in self.exact:
            out.append(self.exact[bare])
        # "arre yaar" also contains "yaar": count only the most specific phrase
        keyed = sorted(((" ".join(tokens(p.strip("=*"))), p, w) for p, w in out),
                       key=lambda x: (-len(x[0]), -x[2]))
        kept: list[tuple[str, str, float]] = []
        for key, p, w in keyed:
            if not any(key in k for k, _, _ in kept):
                kept.append((key, p, w))
        return sorted(((p, w) for _, p, w in kept), key=lambda x: -x[1])

    @staticmethod
    def score(hits: list[tuple[str, float]]) -> float:
        if not hits:
            return 0.0
        return min(1.0, hits[0][1] + 0.15 * (len(hits) - 1))


_CONF = _Lexicon(CONFUSION_PHRASES, CONFUSION_MASKS)
_FRUS = _Lexicon(FRUSTRATION_PHRASES, FRUSTRATION_MASKS)
_URG = _Lexicon(URGENCY_PHRASES, URGENCY_MASKS)


def _hits(text: str, phrases: dict[str, float] | list[str]) -> list[str]:
    """Phrases of a lexicon present in `text` (whole-word, normalized). Kept for callers/tests."""
    lex = _Lexicon(dict.fromkeys(phrases, 1.0) if isinstance(phrases, list) else phrases)
    toks = tokens(text)
    return [p for p, _ in lex.hits(" " + " ".join(toks) + " ", " ".join(t for t in toks if t not in _FILLERS))]


class RuleScorer:
    def score(self, turn: TurnInput, session: SessionState) -> tuple[dict[str, float], list[str]]:
        text = turn.text
        toks = tokens(text)
        low = " " + " ".join(toks) + " "
        content = [t for t in toks if t not in _FILLERS]
        bare = " ".join(content)
        ev: list[str] = []
        s = {"frustration": 0.0, "confusion": 0.0, "urgency": 0.0, "engagement": 0.5}

        last_customer = next((u.text for u in reversed(session.history) if u.role == "customer"), None)
        repeated = last_customer is not None and jaccard(text, last_customer) >= 0.6

        if h := _CONF.hits(low, bare):
            s["confusion"] += _Lexicon.score(h)
            ev.append(f"confusion phrase: {h[0][0]}")
        if repeated:
            s["confusion"] += 0.3
            s["frustration"] += 0.3
            ev.append("customer repeated previous turn")
        if text.strip().endswith("?") and len(toks) <= 3:
            s["confusion"] += 0.2
            ev.append("short question")

        if h := _FRUS.hits(low, bare):
            s["frustration"] += _Lexicon.score(h)
            ev.append(f"frustration phrase: {h[0][0]}")
        if text.count("!") >= 2 or (len(text) > 8 and text.isupper()):
            s["frustration"] += 0.3
            ev.append("emphatic punctuation/caps")
        if turn.timing.barge_in:
            s["frustration"] += 0.25
            s["urgency"] += 0.2
            ev.append("barge-in")

        if h := _URG.hits(low, bare):
            s["urgency"] += _Lexicon.score(h)
            ev.append(f"urgency phrase: {h[0][0]}")
        if turn.timing.speech_rate_wps and turn.timing.speech_rate_wps > 3.5:
            s["urgency"] += 0.2
            ev.append("fast speech")

        if content and all(t in LOW_ENGAGEMENT_REPLIES for t in content):
            s["engagement"] = 0.2
            ev.append("minimal reply")
        elif "?" in text or len(toks) >= 8 or (len(content) >= 3 and any(t in QUESTION_WORDS for t in content)):
            s["engagement"] = 0.8
        lat = turn.timing.response_latency_ms
        if lat is not None and lat > 4000:
            s["engagement"] -= 0.2
            ev.append("slow to respond")

        core = {k: max(0.0, min(1.0, v)) for k, v in s.items()}
        minimal = bool(content) and all(t in LOW_ENGAGEMENT_REPLIES for t in content)
        core.update(_dims.score_extended(turn, session, core, low, bare, toks, content, minimal, ev))
        return core, ev


# ----------------------------------------------------------------------------- prosody

MIN_BASELINE_TURNS = 1  # need at least one earlier turn from the same caller
MAX_EMOTION_SHIFT_ST = 7.0


def _clean_baseline(hist: list) -> list:
    """Drop baseline turns far from the others (a different voice: voicemail, echo, TV)."""
    if len(hist) < 3:
        return hist
    med = float(np.median([h.f0_median_st for h in hist]))
    return [h for h in hist if abs(h.f0_median_st - med) <= MAX_EMOTION_SHIFT_ST]


def prosody_arousal(turn: TurnInput, session: SessionState) -> tuple[float, list[str]]:
    """How much more (or less) activated the caller sounds than THEY did earlier in the call.

    Uses the change, not absolute pitch: raw Hz differs hugely between people. Combines
      pitch level   (semitones above this caller's median; anger/excitement raise it ~2-4 st)
      pitch variation (std in semitones vs their usual; agitation widens it, flat = tired/bored)
      loudness      (dB vs their usual)
    Returns arousal in [-1, 1] and evidence strings. 0 when there is no baseline yet.
    """
    p = turn.prosody
    hist = _clean_baseline(session.prosody_history[-8:])
    if p is None or len(hist) < MIN_BASELINE_TURNS:
        return 0.0, []
    base_st = float(np.median([h.f0_median_st for h in hist]))
    base_var = max(float(np.median([h.f0_std_st for h in hist])), 0.5)
    base_db = float(np.median([h.loudness_db for h in hist]))
    d_pitch = p.f0_median_st - base_st
    var_ratio = p.f0_std_st / base_var
    d_loud = p.loudness_db - base_db
    if abs(d_pitch) > MAX_EMOTION_SHIFT_ST:
        # live call: a voicemail prompt at 369 Hz became the "baseline" and the caller then read
        # as -12 st. Emotional pitch shifts stay well under ~7 st; bigger = another voice/echo.
        return 0.0, [f"pitch {d_pitch:+.1f} st: probably not the same voice, ignored"]
    arousal = 0.45 * np.clip(d_pitch / 4.0, -1, 1) + 0.3 * np.clip(var_ratio - 1.0, -1, 1) \
        + 0.25 * np.clip(d_loud / 6.0, -1, 1)
    arousal *= min(1.0, len(hist) / 3)  # 1-2 turns of baseline = low confidence
    ev = []
    if abs(d_pitch) >= 1.5:
        ev.append(f"pitch {d_pitch:+.1f} st vs caller baseline")
    if var_ratio >= 1.4 or var_ratio <= 0.65:
        ev.append(f"pitch variation {var_ratio:.1f}x usual")
    if abs(d_loud) >= 3:
        ev.append(f"loudness {d_loud:+.0f} dB")
    return float(np.clip(arousal, -1, 1)), ev


def voice_arousal(turn: TurnInput, session: SessionState) -> tuple[float, list[str]]:
    """The calibrated reading (voice/calibrator.py: pitch, movement, loudness, vocal effort and speed vs
    this caller's own baseline) when there is one; otherwise the older pitch-history estimate."""
    v = turn.voice
    if v is not None and v.calibrated:
        return v.arousal * v.confidence, list(v.evidence)
    if v is not None and turn.prosody is not None and not session.prosody_history:
        return 0.0, []
    return prosody_arousal(turn, session)


def apply_prosody(raw: dict[str, float], arousal: float, text_negative: bool) -> dict[str, float]:
    """Voice tells HOW activated someone is; words tell WHICH way. So high arousal amplifies
    frustration only when the words/behaviour already lean negative; otherwise it reads as
    engagement (an excited 'haan haan batao!'). Low arousal (flat, quiet) lowers engagement.

    Extended dims (only touched when present in `raw`):
      high arousal + negative words -> anxiety up (more if the words already sounded worried), calmness down
      high arousal + positive words -> excitement up, calmness down a little
      low arousal (flat, quiet)     -> calmness up, excitement down
    """
    out = dict(raw)
    ext = "anxiety" in out
    if arousal > 0.25:
        if text_negative:
            out["frustration"] = min(1.0, out["frustration"] + 0.4 * arousal)
            out["urgency"] = min(1.0, out["urgency"] + 0.15 * arousal)
            if ext:
                worried = out["anxiety"] > 0 or out["trust"] < 0.5
                out["anxiety"] = min(1.0, out["anxiety"] + (0.4 if worried else 0.2) * arousal)
                out["calmness"] = max(0.0, out["calmness"] - 0.4 * arousal)
        else:
            out["engagement"] = min(1.0, out["engagement"] + 0.25 * arousal)
            out["frustration"] = min(1.0, out["frustration"] + 0.1 * arousal)
            if ext:
                out["excitement"] = min(1.0, out["excitement"] + 0.5 * arousal)
                out["calmness"] = max(0.0, out["calmness"] - 0.2 * arousal)
    elif arousal < -0.25:
        out["engagement"] = max(0.0, out["engagement"] - 0.4 * abs(arousal))
        if ext:
            out["calmness"] = min(1.0, out["calmness"] + 0.4 * abs(arousal))
            out["excitement"] = max(0.0, out["excitement"] - 0.3 * abs(arousal))
    return out


class _FusedScorer:
    """Rules + a trained turn model, 50/50 per dim the model returns (ACIE_TURN_MODEL)."""

    def __init__(self, rules: RawScorer, model):
        self.rules, self.model = rules, model

    def score(self, turn: TurnInput, session: SessionState) -> tuple[dict[str, float], list[str]]:
        raw, ev = self.rules.score(turn, session)
        try:
            m, mev = self.model.score(turn, session)
        except Exception as e:  # a broken model must never break a live turn
            log.warning("turn model failed (%s); rules only for this turn", type(e).__name__)
            return raw, ev
        for dim, v in m.items():
            if dim in raw and v is not None:
                raw[dim] = round(0.5 * raw[dim] + 0.5 * max(0.0, min(1.0, float(v))), 3)
        return raw, ev + list(mev) + ["turn model"]


_MODEL_FAILED = False


def load_turn_model(path: str | None = None):
    """TrainedTurnModel from ACIE_TURN_MODEL (lazy import), or None; logs the failure once."""
    global _MODEL_FAILED
    path = path or os.getenv("ACIE_TURN_MODEL")
    if not path or _MODEL_FAILED:
        return None
    try:
        from training.state_classifier.predict import TrainedTurnModel
        return TrainedTurnModel.load(path)
    except Exception as e:
        _MODEL_FAILED = True
        log.warning("ACIE_TURN_MODEL=%s could not be loaded (%s: %s); using the rules", path, type(e).__name__, e)
        return None


_BASELINE = {k: f.default for k, f in HumanState.model_fields.items() if k != "signals"}


class HumanStateEstimator:
    """Asymmetric EMA: states rise fast (react to a problem now) and decay slowly
    (don't forget the customer was annoyed one turn ago).

    Smoothing math, per dim d with neutral baseline b_d (0 for frustration/anxiety/..., 0.5 for
    engagement/trust/...; the HumanState default) and previous value p, raw reading r:
        a = alpha_up if r > p else alpha_down
        H = b + (a * (r - b) + (1 - a) * (p - b))
    i.e. an EMA of the deviation from the baseline, added back to the baseline. With no signal the
    raw reading equals b, so a neutral-default dim decays toward 0.5, not toward 0. For b = 0 this is
    the plain EMA used before; for b = 0.5 it is algebraically the same as the plain EMA of r.
    """

    def __init__(self, scorer: RawScorer | None = None, alpha_up: float = 0.7, alpha_down: float = 0.3):
        if scorer is None:
            scorer = RuleScorer()
            if (model := load_turn_model()) is not None:
                scorer = _FusedScorer(scorer, model)
        self.scorer = scorer
        self.alpha_up = alpha_up
        self.alpha_down = alpha_down

    def estimate(self, turn: TurnInput, session: SessionState, nlu=None, signals=None) -> HumanState:
        if signals is not None:
            # acie.signals.TurnSignals: the providers' state readings are already fused (acie/signals_fusion.py)
            raw, ev = dict(signals.state), list(signals.evidence)
            nlu = None
        else:
            raw, ev = self.scorer.score(turn, session)
        if nlu is not None:
            # neural reading of the turn in context (acie/understanding.py), fused half-half with the
            # lexicon: the lexicon is precise on words it knows, the model hears everything else
            for dim in list(raw):
                v = getattr(nlu, dim, None)
                if v is not None:
                    raw[dim] = round(0.5 * raw[dim] + 0.5 * float(v), 3)
            ev = ev + [f"nlu: {nlu.gist}" if nlu.gist else "nlu", f"buying {nlu.buying_signal:.2f}"]
        # pitch/loudness arousal: off unless ACIE_PROSODY=1 (the caller: "turn the frequency one off")
        arousal, pev = voice_arousal(turn, session) if os.getenv("ACIE_PROSODY", "0") == "1" else (0.0, [])
        if pev or arousal:
            text_negative = raw["frustration"] > 0 or turn.timing.barge_in
            raw = apply_prosody(raw, arousal, text_negative)
            ev = ev + pev + [f"voice arousal {arousal:+.2f}"]
        prev = session.state
        out = {}
        for dim, r in raw.items():
            p = getattr(prev, dim)
            b = _BASELINE.get(dim, 0.0)
            a = self.alpha_up if r > p else self.alpha_down
            out[dim] = round(b + (a * (r - b) + (1 - a) * (p - b)), 3)
        return HumanState(**out, signals=ev)
