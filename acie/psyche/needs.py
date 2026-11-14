"""The caller's current conversational NEED, from the 14 HumanState dims, the call ledger and the words.

Deterministic and documented. One of nine needs (first matching row wins, fixed order):

  row  need         when (any of)
  ---  -----------  --------------------------------------------------------------------------------
   0   be_heard     crisis words (the safety plan speaks; the need is recorded for the move policy)
   1   closure      goodbye / "that's all" / thanks right after the link went out / hang-up wishes
   2   autonomy     no, not interested, stop, don't call, later, call me back, "I don't want it"
   3   be_heard     frustration >= 0.60, or a complaint about the agent (repeating, voice, rushing,
                    not listening), or swearing
   4   reassurance  trust words (fake, scam, why so cheap, where did you get my number, is it safe),
                    anxiety >= 0.60 or trust <= 0.30
   5   clarity      language trouble, confusion >= 0.60, "didn't understand / repeat / summarise",
                    or words that carry no meaning (garbled speech-to-text)
   6   answer       a question or an information request (price, how, what, why)
   7   space        busy / in a hurry words, frustration 0.40-0.59, cognitive_load >= 0.70
   8   connection   personal talk (family, study, mood, small talk) with no question about the offer
   9   information  anything else: agreement, "go on", curiosity

The table always wins on closure, autonomy and be_heard-crisis. An optional XGBoost classifier
(acie/psyche/need_model.json, fixed params, trained offline on labelled real turns + synthetic ones)
may change the table's answer only among the soft needs (clarity, space, connection, information,
answer) and only when it is >= 0.70 sure. No model file -> the table alone. Never random.
"""

from __future__ import annotations

import json
import os
import re
import zlib
from dataclasses import dataclass, field
from pathlib import Path

NEEDS = ("be_heard", "clarity", "answer", "space", "reassurance", "autonomy", "information", "connection", "closure")
SOFT = ("clarity", "space", "connection", "information", "answer")
MODEL_PATH = Path(__file__).with_name("need_model.json")
FEATURE_BUCKETS = 128
DIMS = ("frustration", "confusion", "urgency", "engagement", "satisfaction", "anxiety", "excitement", "calmness", "trust",
        "certainty", "cognitive_load", "attention", "participation", "receptiveness")

# ---------------------------------------------------------------- lexicons (en / hinglish / Devanagari)
_I = re.IGNORECASE
CRISIS = re.compile(r"मरने का (मन|दिल)|मर जाना चाहत|आत्महत्या|खुदकुशी|जीने का (मन|दिल) नहीं|\bsuicid|want to die|kill myself|"
                    r"marne ka (man|mann|dil)|jeene ka (man|dil) nahi", _I)
CLOSE = re.compile(r"\b(bye|goodbye|good bye|that.?s all|that is all|nothing else|thank you,? bye|ok bye|okay bye)\b|बाय|रखता हूँ|रखती हूँ|"
                   r"रखते हैं|चलते हैं|फोन रख|फ़ोन रख|rakhta hoon|rakhti hoon|rakhte hain|bas itna hi|बस इतना ही|और कुछ नहीं", _I)
THANKS = re.compile(r"\b(thanks?|thank you|shukriya|dhanyavad)\b|धन्यवाद|शुक्रिया|थैंक", _I)
NO = re.compile(r"नहीं चाहिए|नही चाहिए|नहीं जानना|नहीं लेना|नही लेना|जरूरत नहीं|ज़रूरत नहीं|\bnot interested\b|\bno,? thank|\bno thanks\b|"
                r"don.?t want|do not want|don.?t need|i don.?t think|not really|nahi chahiye|nahin chahiye|nahi lena|zarurat nahi|"
                r"don.?t call|do not call|stop calling|remove my number|number hata|कॉल मत|call mat|रहने दो|rehne do|मुझे नहीं|"
                r"नहीं चाहिये|इतना नहीं चाहिए|नहीं चाहिए\b|8 में नहीं", _I)
# asking for a lower price is negotiation (a question about the price), never a decline, even with "नहीं" in it
NEGOTIATE = re.compile(r"कम (नहीं |नही )?(कर|हो)|और कम|थोड़ा कम|थोडा कम|बोरा सा कम|कुछ कम|kam (nahi |nhi )?(kar|ho)|thoda kam|cheaper|lower (the )?price|"
                       r"\breduce\b|bargain|(some|any|a little|more) discount|discount (do|de|aur|mil)|सस्ता कर|sasta kar|और डिस्काउंट", _I)
from ..signals import NEGOTIATE_MORE  # noqa: E402
NEGOTIATE = re.compile(NEGOTIATE.pattern + "|" + NEGOTIATE_MORE, _I)
BARE_NO = {"no", "nope", "nahi", "nahin", "nai", "na", "नहीं", "नही", "ना", "ने", "नो"}
LATER = re.compile(r"\b(later|call me back|call back|call me later|some other time|another time|not now|abhi nahi|baad mein|baad me)\b|"
                   r"बाद में|बाद मे|अभी नहीं|अभी मत|कल कॉल|कल फोन|फिर कभी|बाद में कॉल", _I)
COMPLAINT = re.compile(r"बार बार|bar bar|repeat (क्यों|kyun|kyu|why)|रिपेट|रिपीट|same (thing|cheez)|same बात|तू (चार|बार)|आवाज.{0,12}(बदल|change)|"
                       r"voice.{0,12}(change|chang)|you (keep|are) repeat|stop repeat|not listening|सुन नहीं|जल्दी जा रहा|too fast|rushing|"
                       r"fix this|पागल|बेवकूफ|stupid|idiot|\bshit\b|\bfuck|bakwas|बकवास|harass|परेशान मत|annoy|irritat|तंग|चिढ़|"
                       r"ढंग से (नहीं|reply)|not what i (asked|said)|that.?s not what|तुमने (नहीं|कहा ही नहीं)|you didn.?t (listen|answer)", _I)
DISTRUST = re.compile(r"\b(fake|fraud|scam|cheat|legit|genuine|real or|is it real|is this real|is it safe|spam|trust)\b|नकली|nakli|धोखा|dhokha|"
                      r"फ्रॉड|असली|so cheap|why.{0,12}cheap|इतना सस्ता|इतना सक्ता|इतना कम|itna sasta|sasta kyu|where did you get my number|"
                      r"number (kaha|kahan|kaise)|नंबर (कहाँ|कहां|कैसे)|मेरा नंबर|how did you get|privacy|प्राइवेसी|my data|डेटा|safe hai|सुरक्षित|"
                      r"paise (kat|kate)|पैसे (कट|कटे)|hidden|छुपा|छिपा|कोई (चार्ज|charge)", _I)
CONFUSED = re.compile(r"(did ?n.?t|do ?n.?t|can.?t|cannot|could ?n.?t) (understand|follow|get)|not able to understand|what do you mean|"
                      r"(say|repeat) (that |it )?(again|once)|can you repeat|once again|summari[sz]e|in one line|in simple|"
                      r"समझ नहीं|समझा नहीं|समझी नहीं|क्या मतलब|मतलब क्या|दोबारा|फिर से बोल|एक बार फिर|कितने का पढ़|"
                      r"samajh nahi|samjha nahi|kya matlab|phir se bol|dobara bol|sorry,? what|\bhuh\b|\bpardon\b|क्या बोला|क्या कहा|"
                      r"what is this (deal|about)|what are you talking|कौन से दिन|what link|कौन सा लिंक", _I)
LANG_TROUBLE = re.compile(r"(don.?t|do not|can.?t|not able to).{0,20}(speak|understand|know).{0,12}(hindi|english)|don.?t know (hindi|english)|"
                          r"(shift|switch|talk|speak|tell me|reply).{0,14}(in )?(english|hindi)|(english|hindi).{0,8}(please|plz|में बोल|mein bol)|"
                          r"हिंदी (में|नहीं|शो)|हिन्दी में|इंग्लिश|अंग्रेज|not able to understand your language|your language", _I)
ASKING = re.compile(r"\?|\b(why|what|how|which|who|where|when|is it|are you|can i|can you|do you|does it|explain|answer me|"
                    r"how much|price|cost)\b|क्या|कैसे|कौन|कितन|क्यों|कहाँ|कहां|कब|किस|बताइए|बताओ|बताएं|बता दो|समझाइए|कीमत|दाम|"
                    r"kya|kaise|kaun|kitn|kyu|kyon|batao|bataiye|kab|kahan", _I)
BUSY = re.compile(r"\b(busy|in a meeting|driving|in a hurry|quickly|make it quick|short)\b|बिजी|व्यस्त|मीटिंग|जल्दी में|जल्दी बोल|जल्दी बताइए|"
                  r"\bbusy hoon|jaldi mein|abhi time nahi|time nahi|वक्त नहीं|टाइम नहीं|मैं बिजी|मैं बिजीओ", _I)
PERSONAL = re.compile(r"\b(my (son|daughter|wife|husband|mother|father|family|mom|dad|friend)|i am (studying|a student|tired|sad|lonely|"
                      r"stressed)|i.?m (studying|a student|tired|sad|lonely|stressed)|how are you|feeling|lonely|marathon|college|exam|"
                      r"engineering|studying for)\b|मेरा (बेटा|बेटी|परिवार|दोस्त)|पढ़ाई|थका|अकेला|कैसे हो|कैसी हो|तबियत|चिंता में|मूड", _I)
SALE = re.compile(r"price|कीमत|offer|ऑफर|plan|प्लान|link|लिंक|pay|पेमेंट|subscri|सब्सक्रा|discount|gym|membership|claude|pro\b|trainer|ट्रेनर|"
                  r"machine|मचीन|मशीन|treadmill|class|bench|cardio|weights?", _I)
WEAK_YES = {"ok", "okay", "yes", "yeah", "yep", "sure", "haan", "han", "ha", "haa", "ji", "hmm", "हाँ", "हां", "जी", "हम्म", "ओके", "ठीक", "theek", "bilkul"}
GO_ON = re.compile(r"\b(go ahead|go on|tell me|yes|yeah|sure|please|continue|tell|okay|ok)\b|बताइए|बताओ|बोलिए|बोलो|हाँ|हां|कर दो|भेज दो|भेज दीजिए|"
                   r"भेजिए|ले लूंगा|सब्सक्राइब|subscribe|bhej|batao|bataiye|boliye|kar do", _I)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[\wऀ-ॿ]+", text.lower())


def garbled(text: str) -> bool:
    """Words that carry no meaning: one word repeated, very short filler with no known cue."""
    toks = _tokens(text)
    if len(toks) >= 6 and len(set(toks)) <= max(2, len(toks) // 4):
        return True
    if re.fullmatch(r"[\d\.\s,]+", text.strip() or "x"):
        return True
    return False


@dataclass
class NeedInput:
    text: str
    state: dict[str, float] = field(default_factory=dict)  # the 14 HumanState dims (0..1)
    intent: str = ""                                       # domain intent: affirm, deny, tell_more, ...
    nlu_intent: str = ""                                   # neural understanding's intent, when present
    question_topic: str = "none"
    has_link: bool = False                                 # the payment link already went out
    decided: str = "undecided"                             # call context decision
    turn_index: int = 0
    prev_agent: str = ""
    signals: object = None                                 # acie.signals.TurnSignals (learned providers' fused reading), or None

    def heard(self, *labels: str) -> bool:
        """A learned provider (any model) heard one of these intents; `nlu_intent` is the legacy single-label form."""
        sg = self.signals
        return self.nlu_intent in labels or (sg is not None and sg.heard(*labels))


@dataclass
class NeedResult:
    need: str
    rule: str                    # which table row fired
    source: str = "table"        # table | model
    flags: dict[str, bool] = field(default_factory=dict)  # complaint, lang_trouble, later, distrust, garbled, ...
    model_probs: dict[str, float] = field(default_factory=dict)


def signals(inp: NeedInput) -> dict[str, bool]:
    t, toks = inp.text, _tokens(inp.text)
    f = {
        "crisis": bool(CRISIS.search(t)),
        "bye": bool(CLOSE.search(t)) or inp.intent == "goodbye",
        "thanks": bool(THANKS.search(t)),
        "no": bool(NO.search(t)) or (0 < len(toks) <= 2 and all(w in BARE_NO for w in toks)) or inp.intent == "deny" or inp.heard("deny"),
        "negotiate": bool(NEGOTIATE.search(t)),
        "later": bool(LATER.search(t)) or inp.intent == "callback",
        "complaint": bool(COMPLAINT.search(t)),
        "distrust": bool(DISTRUST.search(t)),
        "confused": bool(CONFUSED.search(t)),
        "lang_trouble": bool(LANG_TROUBLE.search(t)),
        "asking": bool(ASKING.search(t)) or inp.question_topic not in ("none", ""),
        "busy": bool(BUSY.search(t)),
        "personal": bool(PERSONAL.search(t)) or inp.heard("personal", "small_talk"),
        "sale": bool(SALE.search(t)),
        "garbled": garbled(t),
        "go_on": bool(GO_ON.search(t)) or inp.intent in ("affirm", "tell_more", "benefits", "price"),
        "weak_yes": 0 < len(toks) <= 2 and all(w in WEAK_YES for w in toks),
        "short": len(toks) <= 3,
    }
    f["asking"] = f["asking"] or f["negotiate"]
    if f["negotiate"] and not re.search(r"नहीं चाहिए|not interested|don.?t want|stop calling", t, _I):
        f["no"] = False
    # "I don't think that's it" style no must not fire on a question
    if f["no"] and f["asking"] and not NO.search(t) and not f["later"]:
        f["no"] = False
    return f


def table_need(inp: NeedInput) -> NeedResult:
    """The deterministic decision table (see module docstring). First matching row wins."""
    s, f = inp.state, signals(inp)
    g = lambda k, d=0.0: float(s.get(k, d))
    if f["crisis"]:
        return NeedResult("be_heard", "0:crisis", flags=f)
    if f["bye"] and not (f["asking"] and not CLOSE.search(inp.text)) or (inp.has_link and f["short"] and not f["asking"] and (f["thanks"] or f["weak_yes"] or f["bye"])):
        return NeedResult("closure", "1:closure", flags=f)
    if f["no"] or f["later"]:
        return NeedResult("autonomy", "2:autonomy", flags=f)
    if f["complaint"] or g("frustration") >= 0.60:
        return NeedResult("be_heard", "3:be_heard", flags=f)
    if f["distrust"] or g("anxiety") >= 0.60 or ("trust" in s and g("trust", 0.5) <= 0.30):
        return NeedResult("reassurance", "4:reassurance", flags=f)
    if f["lang_trouble"] or f["confused"] or g("confusion") >= 0.60:
        return NeedResult("clarity", "5:clarity", flags=f)
    if f["garbled"] or (f["short"] and not f["go_on"] and not f["weak_yes"] and not f["asking"] and not f["personal"] and not f["busy"]
                        and inp.intent in ("unknown", "clarify", "") and not f["sale"]):
        return NeedResult("clarity", "5:garbled", flags=f)
    if f["asking"]:
        return NeedResult("answer", "6:answer", flags=f)
    if f["busy"] or g("frustration") >= 0.40 or g("cognitive_load") >= 0.70:
        return NeedResult("space", "7:space", flags=f)
    if f["personal"] and not f["sale"]:
        return NeedResult("connection", "8:connection", flags=f)
    return NeedResult("information", "9:information", flags=f)


# ---------------------------------------------------------------- optional XGBoost classifier
def features(inp: NeedInput) -> list[float]:
    """Hashed (zlib.crc32) word uni/bigrams + the 14 dims + a few flags. Same vector at train and run time."""
    vec = [0.0] * FEATURE_BUCKETS
    toks = _tokens(inp.text)
    for g in toks + [a + "_" + b for a, b in zip(toks, toks[1:])]:
        vec[zlib.crc32(g.encode("utf-8")) % FEATURE_BUCKETS] += 1.0
    f = signals(inp)
    extra = [float(inp.state.get(k, 0.5 if k in ("engagement", "satisfaction", "calmness", "trust", "certainty", "attention",
                                                  "participation", "receptiveness") else 0.0)) for k in DIMS]
    extra += [float(f[k]) for k in sorted(f)]
    extra += [float(len(toks)), float(inp.turn_index), float(inp.has_link), 0.0]  # (intent is deliberately not a feature: it differs between labelling and runtime)
    return vec + extra


_MODEL = {"loaded": False, "booster": None, "classes": ()}


def load_model(path: str | os.PathLike | None = None):
    """Lazy, cached. Returns (booster, classes) or (None, ()) when the file/xgboost is missing or ACIE_NEED_MODEL=off."""
    env = os.getenv("ACIE_NEED_MODEL")
    if env == "off":
        return None, ()
    p = Path(path or env or MODEL_PATH)
    if path is None and _MODEL["loaded"] and not env:
        return _MODEL["booster"], _MODEL["classes"]
    booster, classes = None, ()
    try:
        if p.exists():
            import xgboost as xgb
            booster = xgb.Booster()
            booster.load_model(str(p))
            meta = json.loads(p.with_suffix(".meta.json").read_text())
            classes = tuple(meta["classes"])
            booster.set_param({"nthread": 1})
    except Exception:
        booster, classes = None, ()
    if path is None and not env:
        _MODEL.update(loaded=True, booster=booster, classes=classes)
    return booster, classes


def predict_proba(inp: NeedInput, booster, classes) -> dict[str, float]:
    import numpy as np
    import xgboost as xgb
    d = xgb.DMatrix(np.array([features(inp)], dtype="float32"))
    p = booster.predict(d)[0]
    return {c: float(v) for c, v in zip(classes, p)}


def estimate_need(inp: NeedInput, use_model: bool = True) -> NeedResult:
    """Table first; the model may refine only the soft needs, at >= 0.70 confidence. Table wins elsewhere."""
    res = table_need(inp)
    if not use_model or res.need not in SOFT:
        return res
    booster, classes = load_model()
    if booster is None:
        return res
    try:
        probs = predict_proba(inp, booster, classes)
    except Exception:
        return res
    res.model_probs = {k: round(v, 3) for k, v in sorted(probs.items())}
    best = sorted(probs.items(), key=lambda kv: (-kv[1], kv[0]))[0]
    if best[0] in SOFT and best[1] >= 0.70 and best[0] != res.need:
        return NeedResult(best[0], res.rule, "model", res.flags, res.model_probs)
    return res
