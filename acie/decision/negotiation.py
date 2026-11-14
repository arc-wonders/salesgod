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
import math
import re
from dataclasses import asdict, dataclass, field

from ..signals import NEGOTIATE_MORE
from . import honest
from . import negotiation_banks as nb

_I = re.IGNORECASE
LANGS = ("en", "hi", "hinglish")
LEVER_ORDER = ("trial_pass", "cheaper_plan", "annual_saving", "referral")

_GROUP = re.compile(r"(student|स्टूडेंट|छात्र|friend|दोस्त|group|ग्रुप|bulk|family|फैमिली|senior|सीनियर|couple|corporate|referral|refer).{0,40}(discount|डिस्काउंट|छूट|offer|ऑफर|off\b|cheaper|सस्ता|sasta|कम|kam|milega|मिलेगा)|"
                    r"(discount|डिस्काउंट|छूट|offer|ऑफर|cheaper|सस्ता|sasta|कम|kam).{0,40}(student|स्टूडेंट|छात्र|friend|दोस्त|group|ग्रुप|bulk|family|फैमिली|senior|सीनियर|couple|corporate)", _I)
_TRIAL = re.compile(r"free trial|trial|ट्रायल|demo|try (it )?(first|for free|free)|pehle try|पहले (try|ट्राई)|free (mein|me|में) (try|use)|free pass|guest pass", _I)
_COUNTER = re.compile(r"(?:₹|rs\.?|rupees?)\s*\d[\d,]{2,}|\b\d[\d,]{2,}\s*(?:pe|par|tak|mein|me|में)\s*(?:ho|chalega|done|final|kar|de)|\b\d[\d,]{2,}\s*(?:rupe\w*|rs)\b|\d[\d,]*\s*(rupe\w*|rs\.?|₹|dollars?|\$|bucks)?\s*(kam|कम|off|less|discount)\b|"
                      r"(₹|rs\.?|\$)\s*\d[\d,]*\s*(mein|me|में|for|tak|तक)?\s*(do|de|dedo|दो|दे|chahiye|चाहिए)?|\d[\d,]*\s*(rupe\w*|rs|₹|dollars?)?\s*(mein|me|में|for|tak|तक)\s*(do|de|dedo|दो|दे|chahiye|चाहिए|milega|मिलेगा|ho jayega|हो जाएगा)|"
                      r"(give|do|de|दो|दे)\s*(it|me)?\s*(for|at|in)?\s*(₹|rs\.?|\$)?\s*\d[\d,]*", _I)
_COUNTER_MORE = re.compile(r"\b(?:for|at|around|about|to|in)\s*(?:₹|rs\.?|rupees?|\$)?\s*\d[\d,]{2,}\b|\b(?:want|need|make|give|take|do)\b.{0,18}\b\d[\d,]{2,}\b|"
                          r"\b\d[\d,]{2,}\s*(?:max|only|hi|bas|tak|se zyada nahi)\b", _I)
_COMP_NAME = re.compile(r"cult|gold.?s|anytime fitness|competitor|chatgpt|gpt|gemini|copilot|other (gym|app|plan)|dusr[ae] (gym|app|jagah)|doosre|दूसरे|कहीं और|kahin aur|elsewhere|वहाँ|wahan", _I)
_COMP_PRICE = re.compile(r"cheap|sasta|सस्ता|सस्ती|price|dollar|\$|₹|rupe|रुपये|\b\d{3,5}\b|kitna|कितना|cost|monthly|मंथली|महंगा|mehenga|expensive|same|20|free hai|फ्री है", _I)
_BUDGET = re.compile(r"budget|afford|bajet|बजट|tight|tangi|तंगी|paise (kam|nahi|nhi)|पैसे (कम|नहीं)|expensive for me|mere liye (mehenga|महंगा)|मेरे लिए (महंगा|ज़्यादा|ज्यादा)", _I)
_VALUE = re.compile(r"worth|value for money|paisa vasool|पैसा वसूल|itna (paisa|mehenga|महंगा)|इतना (पैसा|महंगा)|mehenga|mehnga|महंगा|expensive|too much|"
                    r"zyada hai|jyada hai|ज़्यादा है|ज्यादा है|costly|pricey|over ?priced", _I)
_HAGGLE = re.compile(honest.NEGOTIATE.pattern + "|" + NEGOTIATE_MORE + r"|still (too )?(high|much|expensive)|(ab|abhi|phir|fir) bhi (mehenga|zyada|jyada|kam)|aur kam|और कम|bargain|negotiat|how much (lower|less|cheaper)|(price|rate|keemat|कीमत) (lower|kam|कम)|more lower|\\blower\\b.{0,12}\\b(price|rate)\\b|(any|more|further|extra|koi|aur) (more |further |extra )?(discount|छूट|डिस्काउंट)|(give|do|de|dedo) (me )?(a |the |some )?(discount|छूट|डिस्काउंट)|best price|final price|last price|aakhri (rate|price)|आखिरी (रेट|प्राइस)|thoda (aur )?(kam|sasta)", _I)
_NUM = re.compile(r"\d[\d,]*")
_NOT_A_PRICE = re.compile(r"\b(?:days?|din|months?|mahine|weeks?|hafte|sessions?|minutes?|baje|bje|o.?clock|am|pm)\b", _I)


def detect(text: str) -> str | None:
    """The negotiation type for this line, or None when it is not price talk."""
    t = text.lower()
    for name, rx in (("discount_group", _GROUP), ("free_trial", _TRIAL), ("counter_offer", _COUNTER)):
        if rx.search(t):
            return name
    if _COUNTER_MORE.search(t) and not _NOT_A_PRICE.search(t):
        return "counter_offer"
    if _COMP_NAME.search(t) and _COMP_PRICE.search(t):
        return "competitor"
    for name, rx in (("budget", _BUDGET), ("value_doubt", _VALUE), ("haggle", _HAGGLE)):
        if rx.search(t):
            return name
    return None


def counter_number(text: str) -> int | None:
    """The amount the caller names: the last number of 3+ digits ("not sold at 2,999, I want 2,500" -> 2500), else the last number."""
    nums = [int(x) for x in re.findall(r"\d+", text.replace(",", ""))]
    big = [x for x in nums if x >= 100]
    return (big or nums or [None])[-1]


@dataclass
class NegState:
    reason: str = ""                                  # budget | value | comparing ("" = unknown)
    asked_reason: bool = False
    floor: int = 0                                    # how many times the floor was spoken
    levers: list[str] = field(default_factory=list)   # levers already offered, in order
    rounds: int = 0
    closed: bool = False                              # accepted warmly: never push again
    last: str = ""
    pushes: int = 0                                   # price pushes seen (the first one only reframes value: anchor at list)
    ladder_i: int = 0                                 # next economics.ladder step to offer
    price: float = 0.0                                # the price currently on the table (0 = list)
    final_said: bool = False                          # the final price was stated (once)
    used: list[str] = field(default_factory=list)     # phrase variants already said ("bank:index"): no sentence repeats
    perks: list[str] = field(default_factory=list)    # ladder perks already offered
    ask_i: int = -1                                   # the closing-ask form used last (never the same twice in a row)
    sweet: bool = False                               # the last perk-only sweetener at the floor was given
    agreed: float = 0.0                               # the price the caller said yes to (never retracted, never re-offered higher)

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
    offer: dict | None = None   # {"price": n, "perk": id}: a closing ask is pending on this
    toned: bool = False         # the text already carries the call's tone (else the domain restyles it)


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
CLOSED_AGAIN = [  # after the warm goodbye was said, a caller who keeps pushing hears something new each time, never a push
    CLOSE_WARM[1],
    _t("I hear you. The price stays where we agreed, and the link is ready whenever you want it.",
       "मैं समझता हूँ। कीमत वही रहेगी जो तय हुई, और लिंक जब चाहें तब तैयार है।",
       "Main samajhta hoon. Keemat wahi rahegi jo tay hui, aur link jab chahein tab ready hai."),
    _t("No hurry at all; if you like, I can also call you back at a time that suits you.",
       "कोई जल्दी नहीं है; चाहें तो मैं आपके सुविधा वाले समय पर वापस कॉल कर सकता हूँ।",
       "Koi jaldi nahi hai; chahein to main aapke suvidha wale time par wapas call kar sakta hoon."),
    _t("Thank you for hearing me out; whatever you decide, it was good talking to you.",
       "मेरी बात सुनने के लिए शुक्रिया; आप जो भी तय करें, आपसे बात करके अच्छा लगा।",
       "Meri baat sunne ke liye shukriya; aap jo bhi tay karein, aapse baat karke achha laga."),
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


# a real push for a LOWER price (a number, "kam karo", "discount do", "cheaper", "reduce"). A comparison question ("what's better than Cult?",
# "Gemini to free hai, uska kya?") or a plain question about how the price works is NOT a push: it is answered, and nothing is conceded.
_PUSH = re.compile(r"kam kar|कम कर|\bsasta\b|सस्त[ाी]|\bcheap|discount|छूट|डिस्काउंट|reduce|lower|cheaper|sasta kar|सस्ता कर|best price|final price|last price|aur kam|और कम|"
                   r"thoda (aur )?(kam|sasta)|bargain|negotiat|match|beat (that|their|it)|give (me )?(it )?(for|at)|(mein|में|me) (do|de|dedo|दो|दे)\b|"
                   r"too (much|expensive|costly)|bahut (mehenga|mehnga|महंगा)|बहुत (महंगा|महँगा)|can.?t afford|afford nahi|budget", _I)


def explicit_push(text: str) -> bool:
    return bool(_PUSH.search(text)) or counter_number(text) is not None and detect(text) == "counter_offer"


def _compare_reply(p, st: NegState, tone: str = "mid") -> NegReply:
    """"What's better than Cult?" / "Gemini is free, what about it?": an honest comparison question gets an honest answer (what you get
    here), never a price floor or a concession."""
    st.last = "compare"
    tone = clean_tone(tone)
    if ladder_enabled(p):  # tone-aware: lead + what you get here
        v = _value_lines(p, st, tone, "")
        lead = _say(nb.COMPARE_LEAD, tone, st, "cmp")
        return NegReply("negotiation", {l: _join(lead[l], v[l]) for l in LANGS}, st, toned=True,
                        note="Honest comparison answer only: no concession, no floor line; the price is not on the table.")
    b = {l: " ; ".join(x for x in (_benefit(p, l, 0), _benefit(p, l, 1)) if x) for l in LANGS}
    parts = {l: ACK["competitor"][l] for l in LANGS}
    if all(b.values()):
        clean = lambda v: str(v).replace("{", "(").replace("}", ")")
        for l in LANGS:
            parts[l] = _join(parts[l], COMPARE[l].format(b1=clean(b[l])))
    return NegReply("negotiation", parts, st, note="Honest comparison answer only: no floor line, no concession; the price is not on the table.")


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


def _fmt(p, n: float) -> str:
    cur = getattr(p.economics, "currency", "₹") or "₹"
    return f"{cur}{int(round(n)):,}"


# ---------------------------------------------------------------- the Indian ladder (numbers only from the product's economics)
# Wording lives in negotiation_banks.py: three tones x three languages, every bank rotated so no call repeats a line.
SWEET_ORDER = ("referral", "cheaper_plan", "trial_pass", "annual_saving")
_STEP_FRACTIONS = (0.33, 0.25, 0.17, 0.12, 0.08)   # each concession as a share of (list - floor): shrinking, e.g. 200 -> 150 -> 100 on a 599 room


def clean_tone(t: str | None) -> str:
    return t if t in nb.TONES else "mid"


def ladder_enabled(p) -> bool:
    """The cash ladder runs only for a product that has economics: a list price and a floor (cost + margin, or an explicit floor) below it.
    A product with no economics (a real company's price, e.g. Claude Pro) gets value and reassurance only: no concessions."""
    eco = getattr(p, "economics", None)
    return eco is not None and eco.list_price > 0 and 0 < eco.profit_floor < eco.list_price


def _grid(L: float) -> int:
    return 50 if L >= 1000 else 10 if L >= 200 else 5 if L >= 50 else 1


def _near(x: float, g: int) -> int:
    """The friendly price (...49 / ...99 on a 50 grid) nearest to x."""
    return int(round((x + 1) / g)) * g - 1 if g > 1 else int(round(x))


def _up(x: float, g: int) -> int:
    """The smallest friendly price at or above x (so the rounded floor never drops under the real floor)."""
    return int(math.ceil((x + 1) / g - 1e-9)) * g - 1 if g > 1 else int(math.ceil(x - 1e-9))


def ladder_prices(list_price: float, floor: float) -> list[int]:
    """The concession prices, highest first, last = the final price (the floor rounded UP to a friendly number, never below it).
    Steps shrink: each cut is a fraction of (list - floor). Empty when there is nothing to give (final == list)."""
    g = _grid(list_price)
    final = min(_up(floor, g), int(math.ceil(list_price)))
    if list_price - final < g:
        return []
    room, prev, out = list_price - floor, float(list_price), []
    for fr in _STEP_FRACTIONS:
        x = _near(prev - fr * room, g)
        if x >= prev:
            x = _near(prev - g, g)
        if x - final < max(g, 0.1 * room):  # too close to the final price to be a step of its own
            break
        out.append(x)
        prev = x
    return out + [final]


def effective_floor(eco, st: NegState) -> float:
    """The profit floor plus what the perks already promised cost us: a sale with a perk must still clear the margin."""
    return eco.profit_floor + sum(x.cost for x in eco.ladder if x.kind == "perk" and x.id in st.perks)


def margin_ok(eco, price: float, perk_ids=()) -> bool:
    """price - cost - perk cost >= the minimum margin (the profit check every offer must pass)."""
    perk = sum(x.cost for x in eco.ladder if x.kind == "perk" and x.id in perk_ids)
    return price - perk >= eco.profit_floor - 1e-9


_UNIT = re.compile(r"(\d+(?:\.\d+)?)?\s*(day|din|week|hafte|month|mahin|year|saal|annual)", _I)


def per_day(eco) -> float | None:
    m = _UNIT.search(eco.unit or "")
    if not m:
        return None
    n = float(m.group(1) or 1)
    u = m.group(2).lower()
    per = 1 if u in ("day", "din") else 7 if u in ("week", "hafte") else 30 if u.startswith("mahin") or u == "month" else 365
    days = per * n
    return eco.list_price / days if days else None


def _pick(st: NegState, key: str, n: int) -> int:
    """The first variant of this bank not yet said on the call; once all were, round robin."""
    seen = [u for u in st.used if u.startswith(key + ":")]
    for i in range(n):
        if f"{key}:{i}" not in seen:
            st.used.append(f"{key}:{i}")
            return i
    i = len(seen) % n
    st.used.append(f"{key}:{i}")
    return i


def _say(bank, tone: str, st: NegState, key: str, **kw) -> dict[str, str]:
    """One row of the bank for this tone (rotated), filled in for the three languages. kw values are strings or {lang: str}."""
    rows = bank[tone]
    row = rows[_pick(st, f"{key}:{tone}", len(rows))]
    clean = lambda v: str(v).replace("{", "(").replace("}", ")")
    return {l: row[i].format(**{k: clean(v[l] if isinstance(v, dict) else v) for k, v in kw.items()}) for i, l in enumerate(LANGS)}


def _p(p, price: float) -> dict[str, str]:
    return {l: _fmt(p, price) for l in LANGS}


def _ask(p, st: NegState, tone: str, price: float) -> dict[str, str]:
    return _say(nb.ASK, tone, st, "ask", p=_fmt(p, price))


def _value_lines(p, st: NegState, tone: str, need_known: str, n_benefits: int = 2) -> dict[str, str]:
    if need_known and (_VALUE.search(need_known) or _BUDGET.search(need_known) or _HAGGLE.search(need_known) or _PUSH.search(need_known)):
        need_known = ""  # the "need" the call recorded is the price complaint itself: not a goal to tie the value to
    joiner = {"en": " and ", "hi": " और ", "hinglish": " aur "}
    b = {l: joiner[l].join(x for x in (_benefit(p, l, i) for i in range(n_benefits)) if x) for l in LANGS}
    out = {l: "" for l in LANGS}
    if all(b.values()):
        out = _say(nb.VALUE_NEED if need_known else nb.VALUE, tone, st, "value", v=b, **({"need": need_known} if need_known else {}))
    pd = per_day(p.economics)
    if pd is not None and pd >= 1:
        day = _say(nb.PER_DAY, tone, st, "perday", d=_fmt(p, pd))
        out = {l: _join(out[l], day[l]) for l in LANGS}
    return out


def _lowest_offer(st: NegState, cur: float) -> float:
    return min(x for x in (st.price, cur) if x) if st.price else cur


def _ladder_reply(p, kind: str, text: str, st: NegState, need_known: str, committed: bool = False, tone: str = "mid") -> NegReply | None:
    """The Indian way, per product economics (list, floor = max(cost x (1+margin), floor_price), perk costs against the margin):
      1. first push: a warm, firm no plus concrete value, no cut (no number given -> ask the budget; a number -> say no to it, ask them to come up);
      2. every later push: ONE step down the shrinking ladder (fractions of list - floor, friendly ...49/...99 prices), framed personally,
         tied to a closing ask; "last/final" only on the true last step above the floor, otherwise "going lower is difficult";
      3. a counter at/above our current offer, or at/above the next step, or at/after the last step and >= floor: accepted at once;
         a counter between the floor and the next step: meet in the middle (friendly rounding, never below the floor);
         a counter below the floor: a no to that number, then the next step;
      4. at the floor: one honest sweetener (a perk the margin allows, then a configured lever), then a warm walk-away. Never below the
         floor, never back up (st.price only goes down), never retracts the agreed price, no urgency or invented approvals."""
    tone = clean_tone(tone)
    eco = p.economics
    st = NegState(**asdict(st))
    L = float(eco.list_price)
    floor = effective_floor(eco, st)
    cur = st.price or L
    st.pushes += 1
    first = st.pushes == 1
    n, target = None, None
    if kind == "counter_offer" or (re.search(r"\d[\d,]{2,}", text) and not _NOT_A_PRICE.search(text)):  # "mera budget 2000 hai" names a rate too
        n = counter_number(text)
        if n is not None:
            target = float(max(0, cur - n)) if re.search(r"kam|कम|off|less", text, _I) and n < cur else float(n)
    say = lambda *banks: {l: _join(*(b[l] for b in banks)) for l in LANGS}
    g = _grid(L)

    def accept(price: float, note: str) -> NegReply:
        seq_ = ladder_prices(L, floor)
        st.price, st.final_said = min(price, cur), st.final_said or bool(seq_ and price <= seq_[-1])
        return NegReply("negotiation", say(_say(nb.ACCEPT, tone, st, "accept", p=_fmt(p, st.price)), _ask(p, st, tone, st.price)), st,
                        offer={"price": st.price}, note=note, toned=True)

    if st.agreed:  # the agreed price stands: never retracted, never lowered, never raised
        return NegReply("negotiation", _say(nb.CLOSED_AGAIN, tone, st, "closed"), st, note="Price already agreed: it stays.", toned=True)
    if target is not None and target >= cur:
        return accept(cur, "Their number is at or above our offer: accepted at once, at our offer.")
    if st.closed:
        return NegReply("negotiation", _say(nb.CLOSED_AGAIN, tone, st, "closed"), st, toned=True)
    seq = ladder_prices(L, floor)
    nxt = next((x for x in seq if x < cur), None)
    if first:  # one warm firm no, with value: no cut yet
        if target is not None:
            lead = _say(nb.NO_NUM, tone, st, "nonum", n=_fmt(p, target))
            tail = _say(nb.ASK_COME_UP, tone, st, "comeup")
        elif committed:
            return NegReply("negotiation", say(_say(nb.NO_FIRST, tone, st, "no"), _say(nb.ASK_KEEP, tone, st, "keep")), st,
                            note="Already keen: a warm no, let them name their number.", toned=True)
        else:
            lead = _say(nb.COMPARE_LEAD, tone, st, "cmp") if kind == "competitor" else _say(nb.NO_FIRST, tone, st, "no")
            tail = _say(nb.ASK_BUDGET, tone, st, "budget")
        return NegReply("negotiation", say(lead, _value_lines(p, st, tone, need_known), tail), st,
                        note="Warm firm no with value; no concession yet; ask their budget/number.", toned=True)
    if target is not None and target >= floor:
        if nxt is None or target >= nxt:  # at/after the last step, or at/above the step we would give anyway: yes
            return accept(target, "Their number clears the floor and is at/above our next step: accepted.")
        mid = max(_near((cur + target) / 2.0, g), seq[-1] if seq else int(math.ceil(floor)))
        if mid < cur:
            st.price = float(mid)
            final = bool(seq) and mid == seq[-1]
            st.final_said = st.final_said or final
            return NegReply("price_floor" if final else "negotiation",
                            say(_say(nb.MID, tone, st, "mid", p=_fmt(p, mid)), _say(nb.NOT_LAST, tone, st, "notlast") if not final else {l: "" for l in LANGS},
                                _ask(p, st, tone, mid)), st, offer={"price": float(mid)}, note="Met in the middle, never below the floor; one closing ask.", toned=True)
    if nxt is not None:
        st.price = float(nxt)
        final = nxt == seq[-1]
        st.final_said = st.final_said or final
        lead = _say(nb.NO_NUM_THEN, tone, st, "nothen", n=_fmt(p, target)) if target is not None else {l: "" for l in LANGS}
        body = _say(nb.FINAL, tone, st, "final", p=_fmt(p, nxt)) if final else say(
            _say(nb.STEP, tone, st, "step", p=_fmt(p, nxt)), _say(nb.NOT_LAST, tone, st, "notlast") if cur < L else {l: "" for l in LANGS})
        return NegReply("price_floor" if final else "negotiation", say(lead, body, _ask(p, st, tone, nxt)), st, offer={"price": float(nxt)},
                        note="Final price, said once, warmly." if final else "One concession tied to a closing ask.", toned=True)
    # at the floor: no more cash. State it once, then one honest sweetener, then a warm goodbye
    if not st.final_said:
        st.final_said = True
        return NegReply("price_floor", say(_say(nb.LOWEST, tone, st, "lowest", p=_fmt(p, cur)), _ask(p, st, tone, cur)), st, offer={"price": cur},
                        note="Floor stated once, warmly.", toned=True)
    if not st.sweet:
        st.sweet = True
        perks = [x for x in eco.ladder if x.kind == "perk" and len(st.perks) < eco.max_perks and x.id not in st.perks and all(x.text.get(l) for l in LANGS)
                 and cur >= eco.profit_floor + x.cost]
        if perks:  # a perk instead of cash, only if the margin still clears with its cost
            stp = perks[0]
            st.perks.append(stp.id)
            return NegReply("negotiation", say(_say(nb.PERK_LEAD, tone, st, "perk"), {l: stp.text[l] for l in LANGS}, _ask(p, st, tone, cur)), st,
                            offer={"price": cur, "perk": stp.id}, note="One perk instead of cash (margin allows), then a closing ask.", toned=True)
        cfg = getattr(p, "negotiation", None)
        for name in SWEET_ORDER:
            if cfg is not None and name in cfg.levers and name not in st.levers and all(_lever_text(p, name, l) for l in LANGS):
                st.levers.append(name)
                return NegReply("negotiation", say(_say(nb.SWEET_LEAD, tone, st, "sweet"), {l: _lever_text(p, name, l) for l in LANGS}, _ask(p, st, tone, cur)), st,
                                offer={"price": cur}, note="Last honest sweetener from the config, then a warm goodbye.", toned=True)
    st.closed = True
    return NegReply("negotiation", _say(nb.CLOSE_WARM, tone, st, "close"), st, note="Final no: warm goodbye, callback offered.", toned=True)


def lang_bank_value(b: dict, need_known: str) -> dict:
    if not all(b.values()):
        return {l: "" for l in LANGS}
    clean = lambda v: str(v).replace("{", "(").replace("}", ")")
    bank = VALUE
    return {l: bank[l].format(b1=clean(b[l])) for l in LANGS}


def step(p, text: str, st: NegState, need_known: str = "", faq_line=None, committed: bool = False, tone: str = "mid") -> NegReply | None:
    """One negotiation turn. Returns the reply for all three languages and the new state, or None when this is not price talk.
    `faq_line(lang)` gives the operator's approved FAQ answer for the question, when there is one."""
    kind = detect(text)
    if kind is None:
        return None
    st = NegState(**asdict(st))
    st.rounds += 1
    if kind == "competitor" and not explicit_push(text):
        return _compare_reply(p, st, tone)
    if kind == "counter_offer" and _COMP_NAME.search(text) and honest.is_question(text) and not _PUSH.search(text):
        return _compare_reply(p, st, tone)  # "Cult 2,500 mein deta hai, aapka kyun lu?": their rate is the competitor's, a question: answered, no concession
    ladder_on = ladder_enabled(p)
    if ladder_on and kind in ("value_doubt", "budget") and honest.is_question(text) and not explicit_push(text):
        return None  # a question about how the price works is answered by the conversation first, not met with a concession
    if ladder_on and kind in ("haggle", "value_doubt", "budget", "counter_offer", "competitor"):
        return _ladder_reply(p, kind, text, st, need_known, committed, tone)
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
