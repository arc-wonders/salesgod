"""Principled negotiation, as a deterministic state machine (no randomness, no learning, no profit optimisation).

Price talk comes in a few shapes. Each is detected with fixed patterns, in this order:

  discount_group  student / friend / group / bulk / family / senior discount
  free_trial      "can I try it free first", a trial or demo
  counter_offer   "500 kam karo", "1000 mein do", "give it for $5"
  competitor      "Cult is cheaper", "ChatGPT is also $20", comparing with someone else
  budget          "budget tight", "can't afford" (real hardship is handled earlier by the hard-time pause: selling stops)
  value_doubt     "is it worth it", "too expensive", "mehenga hai"
  haggle          "reduce it", "cheaper", "discount do"

The call's negotiation state (kept in `session.dialog["_neg"]`, the per-call ledger next to `_said`) says what already happened:
the reason (budget / value / comparing) if known, whether the floor was stated, which levers were offered, how many rounds.
The flow never loops and never repeats a line:

  1. acknowledge the concern; if the reason is unknown (plain haggling) ask ONE short question about it;
  2. say the floor once, kindly: "this is already the lowest price, I can't go below it", with no invented discount;
  3. reframe the value against what they said they need (the stated goal from the call), never against a made-up number;
  4. offer only the levers the product config explicitly allows, in a fixed order: trial_pass, cheaper_plan, annual_saving,
     referral (`Product.negotiation.levers`). Defaults: no levers, so nothing changes unless the operator configures them;
  5. after the floor and every allowed lever, if they still push or decline: accept warmly, offer a callback or details,
     and never push again. Walking away is always respected.

Nothing here lowers a price: the floor is the offer price. No urgency, no pressure, no "last chance".
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field

from ..signals import NEGOTIATE_MORE
from . import honest

_I = re.IGNORECASE
LANGS = ("en", "hi", "hinglish")
LEVER_ORDER = ("trial_pass", "cheaper_plan", "annual_saving", "referral")

_GROUP = re.compile(r"(student|स्टूडेंट|छात्र|friend|दोस्त|group|ग्रुप|bulk|family|फैमिली|senior|सीनियर|couple|corporate|referral|refer).{0,40}(discount|डिस्काउंट|छूट|offer|ऑफर|off\b|cheaper|सस्ता|sasta|कम|kam|milega|मिलेगा)|"
                    r"(discount|डिस्काउंट|छूट|offer|ऑफर|cheaper|सस्ता|sasta|कम|kam).{0,40}(student|स्टूडेंट|छात्र|friend|दोस्त|group|ग्रुप|bulk|family|फैमिली|senior|सीनियर|couple|corporate)", _I)
_TRIAL = re.compile(r"free trial|trial|ट्रायल|demo|try (it )?(first|for free|free)|pehle try|पहले (try|ट्राई)|free (mein|me|में) (try|use)|free pass|guest pass", _I)
_COUNTER = re.compile(r"\d[\d,]*\s*(rupe\w*|rs\.?|₹|dollars?|\$|bucks)?\s*(kam|कम|off|less|discount)\b|"
                      r"(₹|rs\.?|\$)\s*\d[\d,]*\s*(mein|me|में|for|tak|तक)?\s*(do|de|dedo|दो|दे|chahiye|चाहिए)?|\d[\d,]*\s*(rupe\w*|rs|₹|dollars?)?\s*(mein|me|में|for|tak|तक)\s*(do|de|dedo|दो|दे|chahiye|चाहिए|milega|मिलेगा|ho jayega|हो जाएगा)|"
                      r"(give|do|de|दो|दे)\s*(it|me)?\s*(for|at|in)?\s*(₹|rs\.?|\$)?\s*\d[\d,]*", _I)
_COMP_NAME = re.compile(r"cult|gold.?s|anytime fitness|competitor|chatgpt|gpt|gemini|copilot|other (gym|app|plan)|dusr[ae] (gym|app|jagah)|doosre|दूसरे|कहीं और|kahin aur|elsewhere|वहाँ|wahan", _I)
_COMP_PRICE = re.compile(r"cheap|sasta|सस्ता|सस्ती|price|dollar|\$|₹|rupe|रुपये|\b\d{3,5}\b|kitna|कितना|cost|monthly|मंथली|महंगा|mehenga|expensive|same|20|free hai|फ्री है", _I)
_BUDGET = re.compile(r"budget|afford|bajet|बजट|tight|tangi|तंगी|paise (kam|nahi|nhi)|पैसे (कम|नहीं)|expensive for me|mere liye (mehenga|महंगा)|मेरे लिए (महंगा|ज़्यादा|ज्यादा)", _I)
_VALUE = re.compile(r"worth|value for money|paisa vasool|पैसा वसूल|itna (paisa|mehenga|महंगा)|इतना (पैसा|महंगा)|mehenga|mehnga|महंगा|expensive|too much|"
                    r"zyada hai|jyada hai|ज़्यादा है|ज्यादा है|costly|pricey|over ?priced", _I)
_HAGGLE = re.compile(honest.NEGOTIATE.pattern + "|" + NEGOTIATE_MORE + r"|bargain|negotiat|best price|final price|last price|aakhri (rate|price)|आखिरी (रेट|प्राइस)|thoda (aur )?(kam|sasta)", _I)
_NUM = re.compile(r"\d[\d,]*")


def detect(text: str) -> str | None:
    """The negotiation type for this line, or None when it is not price talk."""
    t = text.lower()
    for name, rx in (("discount_group", _GROUP), ("free_trial", _TRIAL), ("counter_offer", _COUNTER)):
        if rx.search(t):
            return name
    if _COMP_NAME.search(t) and _COMP_PRICE.search(t):
        return "competitor"
    for name, rx in (("budget", _BUDGET), ("value_doubt", _VALUE), ("haggle", _HAGGLE)):
        if rx.search(t):
            return name
    return None


def counter_number(text: str) -> int | None:
    m = _NUM.search(text.replace(",", ""))
    return int(m.group(0)) if m else None


@dataclass
class NegState:
    reason: str = ""                                  # budget | value | comparing ("" = unknown)
    asked_reason: bool = False
    floor: int = 0                                    # how many times the floor was spoken
    levers: list[str] = field(default_factory=list)   # levers already offered, in order
    rounds: int = 0
    closed: bool = False                              # accepted warmly: never push again
    last: str = ""

    @classmethod
    def load(cls, dialog: dict) -> "NegState":
        try:
            return cls(**json.loads(dialog.get("_neg") or "{}"))
        except (ValueError, TypeError):
            return cls()

    def save(self, dialog: dict) -> None:
        dialog["_neg"] = json.dumps(asdict(self), sort_keys=True)


@dataclass
class NegReply:
    intent: str                 # price_floor (the floor was spoken) | negotiation
    text: dict[str, str]
    state: NegState
    end_call: bool = False
    note: str = ""


def _t(en: str, hi: str, hl: str) -> dict[str, str]:
    return {"en": en, "hi": hi, "hinglish": hl}


# ---------------------------------------------------------------- phrase banks (fixed text, three languages)
ACK = {
    "haggle": _t("I understand, everyone wants the best value.", "मैं समझता हूँ, हर कोई सही कीमत चाहता है।", "Main samajhta hoon, har koi sahi keemat chahta hai."),
    "value_doubt": _t("That's a fair thing to ask.", "ये पूछना बिल्कुल ठीक है।", "Ye poochna bilkul theek hai."),
    "competitor": _t("It is good that you compare.", "तुलना करना अच्छी बात है।", "Compare karna achhi baat hai."),
    "budget": _t("Thank you for telling me, that's fair.", "बताने के लिए शुक्रिया, आपकी बात सही है।", "Batane ke liye shukriya, aapki baat sahi hai."),
    "counter_offer": _t("Thank you for saying what works for you.", "आपने बताया कि आपके लिए क्या ठीक रहेगा, शुक्रिया।", "Aapne bataya ki aapke liye kya theek rahega, shukriya."),
    "discount_group": _t("Good question.", "अच्छा सवाल है।", "Achha sawal hai."),
    "free_trial": _t("It makes sense to want to look first.", "पहले देखकर तय करना ठीक है।", "Pehle dekhkar tay karna theek hai."),
}
FLOOR = [  # {offer} is the full offer phrase (a locked fact); the second line never repeats the first
    _t("I can't go lower: the offer is fixed at {offer}, and that is already the lowest price I have.",
       "मैं इससे कम नहीं कर सकता: ऑफर {offer} पर तय है, और ये पहले से मेरी सबसे कम कीमत है।",
       "Main isse kam nahi kar sakta: offer {offer} par fixed hai, aur ye pehle se meri sabse kam keemat hai."),
    _t("That part truly can't move, and I'd rather be straight with you than promise something I can't give.",
       "ये हिस्सा सच में नहीं बदल सकता, और मैं ऐसा कुछ वादा नहीं करना चाहता जो दे न सकूँ।",
       "Ye hissa sach mein nahi badal sakta, aur main aisa kuch vaada nahi karna chahta jo de na sakoon."),
]
REASON_Q = _t("May I ask what matters most to you here: the budget, how much use you'd get from it, or comparing with something else?",
              "क्या मैं पूछ सकता हूँ कि आपके लिए सबसे ज़रूरी क्या है: बजट, इससे कितना काम निकलेगा, या किसी और से तुलना?",
              "Kya main pooch sakta hoon ki aapke liye sabse zaroori kya hai: budget, isse kitna kaam niklega, ya kisi aur se compare?")
CLOSE_WARM = [
    _t("Of course, that is completely fine. Would you like me to call you back another time, or shall I leave it here? No pressure at all.",
       "बिल्कुल, कोई बात नहीं। क्या मैं किसी और समय आपको वापस कॉल करूँ, या यहीं छोड़ दूँ? कोई दबाव नहीं है।",
       "Bilkul, koi baat nahi. Kya main kisi aur time aapko wapas call karoon, ya yahin chhod doon? Koi pressure nahi hai."),
    _t("That's fine, take all the time you need; I'm here if a question comes up.",
       "ठीक है, आप आराम से सोचिए; कोई सवाल हो तो मैं यहीं हूँ।",
       "Theek hai, aap aaram se sochiye; koi sawal ho to main yahin hoon."),
]
NO_GROUP = _t("There is no student, friend or group discount in this offer, and I can't make one up.",
              "इस ऑफर में स्टूडेंट, दोस्त या ग्रुप डिस्काउंट नहीं है, और मैं अपनी तरफ़ से कोई बना नहीं सकता।",
              "Is offer mein student, friend ya group discount nahi hai, aur main apni taraf se koi bana nahi sakta.")
NO_TRIAL = _t("I don't have a free trial to offer; nothing is charged on this call, and you can decide whenever you like.",
              "मेरे पास देने के लिए कोई फ्री ट्रायल नहीं है; इस कॉल पर कुछ नहीं कटता, और आप जब चाहें तय कर सकते हैं।",
              "Mere paas dene ke liye koi free trial nahi hai; is call par kuch nahi katta, aur aap jab chahein tay kar sakte hain.")
COMPARE = _t("I can't speak for their plan, so I won't compare. Here you get {b1}.",
             "मैं उनके प्लान के बारे में नहीं बोल सकता, इसलिए तुलना नहीं करूँगा। यहाँ आपको {b1} मिलता है।",
             "Main unke plan ke baare mein nahi bol sakta, isliye compare nahi karunga. Yahan aapko {b1} milta hai.")
VALUE = _t("What you'd get: {b1}.", "आपको मिलेगा: {b1}।", "Aapko milega: {b1}.")
VALUE_NEED = _t("You mentioned \"{need}\", so the part that helps is {b1}.", "आपने \"{need}\" बताया था, तो इसमें काम का हिस्सा है: {b1}।",
                "Aapne \"{need}\" bataya tha, to isme kaam ka hissa hai: {b1}.")
BUDGET_NOTE = _t("Please don't stretch your budget for this.", "इसके लिए अपना बजट मत खींचिए।", "Iske liye apna budget mat kheenchiye.")


def _benefit(p, lang: str, i: int = 0) -> str:
    if lang == "en":
        return p.benefits[i] if len(p.benefits) > i else ""
    lst = (p.benefits_hi if lang == "hi" else p.benefits_hinglish) or []
    return lst[i] if len(lst) > i else ""


def _lever_text(p, name: str, lang: str) -> str:
    cfg = getattr(p, "negotiation", None)
    d = (cfg.levers.get(name) if cfg is not None else None) or {}
    return d.get(lang) or ""


def _next_lever(p, st: NegState, lang: str) -> tuple[str, str] | None:
    """The next allowed lever not yet offered, in the fixed order. A lever with no text in this language is skipped."""
    cfg = getattr(p, "negotiation", None)
    if cfg is None:
        return None
    for name in LEVER_ORDER:
        if name in cfg.levers and name not in st.levers and _lever_text(p, name, lang):
            return name, _lever_text(p, name, lang)
    return None


def _join(*parts: str) -> str:
    return " ".join(x for x in parts if x)


def step(p, text: str, st: NegState, need_known: str = "", faq_line=None) -> NegReply | None:
    """One negotiation turn. Returns the reply for all three languages and the new state, or None when this is not price talk.
    `faq_line(lang)` gives the operator's approved FAQ answer for the question, when there is one."""
    kind = detect(text)
    if kind is None:
        return None
    st = NegState(**asdict(st))
    st.rounds += 1
    if st.closed and kind in ("competitor", "discount_group", "free_trial"):
        return None  # a question again after walking away: the conversation answers it (never pushed)
    if st.closed:  # they already walked away once: never push, just stay kind
        st.last = "closed"
        return NegReply("negotiation", CLOSE_WARM[1], st, note="They said no to the options: be kind, do not push.")
    if kind == "budget":
        st.reason = "budget"
    elif kind == "value_doubt":
        st.reason = st.reason or "value"
    elif kind == "competitor":
        st.reason = "comparing"

    def lang_text(bank: dict, **kw) -> dict[str, str]:
        clean = lambda v: str(v).replace("{", "(").replace("}", ")")
        return {l: bank[l].format(**{k: clean(v[l] if isinstance(v, dict) else v) for k, v in kw.items()}) if kw else bank[l] for l in LANGS}

    ack = ACK.get(kind, ACK["haggle"])
    parts: dict[str, str] = {l: ack[l] for l in LANGS}
    intent = "negotiation"

    def add(bank: dict) -> None:
        for l in LANGS:
            parts[l] = _join(parts[l], bank[l])

    # ----- type-specific honest content (no invented discount)
    if kind == "discount_group":
        faq = {l: (faq_line(l) if faq_line else "") for l in LANGS}
        if all(faq.values()):
            parts = {l: _join(ack[l], faq[l] + ("।" if l == "hi" else ".")) for l in LANGS}
        else:
            add(NO_GROUP)
        cfg = getattr(p, "negotiation", None)
        if cfg is not None and "referral" in cfg.levers and "referral" not in st.levers and _lever_text(p, "referral", "en"):
            st.levers.append("referral")
            for l in LANGS:
                parts[l] = _join(parts[l], _lever_text(p, "referral", l))
        st.last = kind
        return NegReply("negotiation", parts, st)
    if kind == "free_trial":
        got = _next_lever(p, st, "en") if "trial_pass" in getattr(getattr(p, "negotiation", None), "levers", {}) and "trial_pass" not in st.levers else None
        if got is not None:
            st.levers.append("trial_pass")
            for l in LANGS:
                parts[l] = _join(parts[l], _lever_text(p, "trial_pass", l))
        else:
            faq = {l: (faq_line(l) if faq_line else "") for l in LANGS}
            if all(faq.values()):  # e.g. "there is a free plan": the operator's approved answer
                parts = {l: _join(ack[l], faq[l] + ("।" if l == "hi" else ".")) for l in LANGS}
            else:
                add(NO_TRIAL)
        st.last = kind
        return NegReply("negotiation", parts, st)

    # ----- first contact: acknowledge, say the floor once, then (if the reason is unknown) one short question
    if st.floor == 0:
        st.floor = 1
        intent = "price_floor"
        add(FLOOR[0])
        if kind == "competitor":
            faq = {l: (faq_line(l) if faq_line else "") for l in LANGS}
            b = {l: _benefit(p, l) for l in LANGS}
            if all(faq.values()):  # the operator's approved comparison answer
                add({l: faq[l] + ("।" if l == "hi" else ".") for l in LANGS})
            elif all(b.values()):
                add(lang_text(COMPARE, b1=b))
        elif kind == "value_doubt":
            b = {l: _benefit(p, l) for l in LANGS}
            if all(b.values()):
                add(lang_text(VALUE_NEED if need_known else VALUE, b1=b, **({"need": need_known} if need_known else {})))
        if not st.reason and not st.asked_reason and kind in ("haggle", "counter_offer"):
            st.asked_reason = True
            add(REASON_Q)
        elif kind == "budget":
            add(BUDGET_NOTE)
        st.last = kind
        return NegReply(intent, parts, st, note="Acknowledge, state the floor once and kindly, no pressure.")

    # ----- later rounds: an allowed lever (fixed order), else walk away warmly
    lever = _next_lever(p, st, "en")
    if lever is not None:
        name = lever[0]
        st.levers.append(name)
        if kind in ("value_doubt", "competitor"):
            b = {l: _benefit(p, l, 1) or _benefit(p, l) for l in LANGS}
            if all(b.values()):
                parts = {l: _join(lang_text(VALUE, b1=b)[l], "") for l in LANGS}
        for l in LANGS:
            parts[l] = _join(parts[l], _lever_text(p, name, l))
        if st.floor < 2 and kind in ("haggle", "counter_offer"):  # the floor again only as a short, different sentence
            st.floor = 2
            add(FLOOR[1])
        st.last = f"lever:{name}"
        return NegReply("negotiation", parts, st, note="Offer only this lever; no pressure.")
    st.closed = True
    st.last = "close"
    return NegReply("negotiation", CLOSE_WARM[0], st, note="Accept warmly; offer a callback or details; never push.")
