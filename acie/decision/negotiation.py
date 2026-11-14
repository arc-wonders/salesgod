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
_COUNTER = re.compile(r"(?:₹|rs\.?|rupees?)\s*\d[\d,]{2,}|\b\d[\d,]{2,}\s*(?:pe|par|tak|mein|me|में)\s*(?:ho|chalega|done|final|kar|de)|\b\d[\d,]{2,}\s*(?:rupe\w*|rs)\b|\d[\d,]*\s*(rupe\w*|rs\.?|₹|dollars?|\$|bucks)?\s*(kam|कम|off|less|discount)\b|"
                      r"(₹|rs\.?|\$)\s*\d[\d,]*\s*(mein|me|में|for|tak|तक)?\s*(do|de|dedo|दो|दे|chahiye|चाहिए)?|\d[\d,]*\s*(rupe\w*|rs|₹|dollars?)?\s*(mein|me|में|for|tak|तक)\s*(do|de|dedo|दो|दे|chahiye|चाहिए|milega|मिलेगा|ho jayega|हो जाएगा)|"
                      r"(give|do|de|दो|दे)\s*(it|me)?\s*(for|at|in)?\s*(₹|rs\.?|\$)?\s*\d[\d,]*", _I)
_COMP_NAME = re.compile(r"cult|gold.?s|anytime fitness|competitor|chatgpt|gpt|gemini|copilot|other (gym|app|plan)|dusr[ae] (gym|app|jagah)|doosre|दूसरे|कहीं और|kahin aur|elsewhere|वहाँ|wahan", _I)
_COMP_PRICE = re.compile(r"cheap|sasta|सस्ता|सस्ती|price|dollar|\$|₹|rupe|रुपये|\b\d{3,5}\b|kitna|कितना|cost|monthly|मंथली|महंगा|mehenga|expensive|same|20|free hai|फ्री है", _I)
_BUDGET = re.compile(r"budget|afford|bajet|बजट|tight|tangi|तंगी|paise (kam|nahi|nhi)|पैसे (कम|नहीं)|expensive for me|mere liye (mehenga|महंगा)|मेरे लिए (महंगा|ज़्यादा|ज्यादा)", _I)
_VALUE = re.compile(r"worth|value for money|paisa vasool|पैसा वसूल|itna (paisa|mehenga|महंगा)|इतना (पैसा|महंगा)|mehenga|mehnga|महंगा|expensive|too much|"
                    r"zyada hai|jyada hai|ज़्यादा है|ज्यादा है|costly|pricey|over ?priced", _I)
_HAGGLE = re.compile(honest.NEGOTIATE.pattern + "|" + NEGOTIATE_MORE + r"|still (too )?(high|much|expensive)|(ab|abhi|phir|fir) bhi (mehenga|zyada|jyada|kam)|aur kam|और कम|bargain|negotiat|best price|final price|last price|aakhri (rate|price)|आखिरी (रेट|प्राइस)|thoda (aur )?(kam|sasta)", _I)
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
    pushes: int = 0                                   # price pushes seen (the first one only reframes value: anchor at list)
    ladder_i: int = 0                                 # next economics.ladder step to offer
    price: float = 0.0                                # the price currently on the table (0 = list)
    final_said: bool = False                          # the final price was stated (once)
    used: list[str] = field(default_factory=list)     # phrase variants already said ("bank:index"): no sentence repeats
    perks: list[str] = field(default_factory=list)    # ladder perks already offered
    ask_i: int = -1                                   # the closing-ask form used last (never the same twice in a row)
    sweet: bool = False                               # the last perk-only sweetener at the floor was given

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


def _fmt(p, n: float) -> str:
    cur = getattr(p.economics, "currency", "₹") or "₹"
    return f"{cur}{int(round(n)):,}"


# ---------------------------------------------------------------- close-in-profit bargaining banks (fixed text, three languages)
# Every bank is a list of variants; NegState.used remembers which were said, so no round repeats a line.
OPENERS = [
    _t("Fair enough.", "ठीक है।", "Theek hai."),
    _t("I understand.", "मैं समझता हूँ।", "Main samajhta hoon."),
    _t("Okay, let me see what I can do.", "अच्छा, देखता हूँ मैं क्या कर सकता हूँ।", "Achha, dekhta hoon main kya kar sakta hoon."),
    _t("Alright.", "चलिए।", "Chaliye."),
]
FIRST_NUMBER = _t("What number did you have in mind?", "आप बताइए, आपके मन में कितना है?", "Aap batao, aapke mann mein kitna hai?")
PERK_LEADS = [
    _t("Let me add something instead of cutting the price: one thing free.", "कीमत घटाने की जगह एक चीज़ फ्री कर देता हूँ।", "Keemat ghatane ki jagah ek cheez free kar deta hoon."),
    _t("How about this: a bit more value rather than a lower number.", "एक काम करता हूँ: कम कीमत की जगह थोड़ा और वैल्यू जोड़ देता हूँ।", "Ek kaam karta hoon: kam keemat ki jagah thoda aur value jod deta hoon."),
    _t("Here's one more thing from my side.", "लीजिए, एक चीज़ और मेरी तरफ़ से।", "Ye lijiye, ek cheez aur meri taraf se."),
]
PRICE_LINES = [  # a price concession, each tied to a commitment (never a deadline or scarcity); {p} = the price
    ("I can do {p} for you, if we finalise it now.", "मैं आपके लिए {p} कर सकता हूँ, अगर अभी तय हो जाए।", "Aapke liye main {p} kar sakta hoon, agar abhi tay ho jaye."),
    ("It has been nice talking to you, so I'll come down to {p} if you're ready to join now.", "आपसे बात करके अच्छा लगा, इसलिए अगर आप अभी जुड़ने को तैयार हैं तो {p} कर देता हूँ।",
     "Aapse baat karke achha laga, isliye agar aap abhi judne ko taiyaar hain to {p} kar deta hoon."),
    ("Let me stretch a little more: {p}, if you'll go ahead today.", "थोड़ा और खींचता हूँ: {p}, अगर आप आज आगे बढ़ें।", "Thoda aur kheenchta hoon: {p}, agar aap aaj aage badhein."),
    ("One more small step down to {p}, on the promise that we close it now.", "एक छोटा क़दम और, {p}, इस वादे पर कि अभी बात पक्की करें।", "Ek chhota kadam aur, {p}, is vaade par ki abhi baat pakki karein."),
]
MIDDLE_LINE = ("Let's meet in the middle: {p}, if you'll join now.", "चलिए बीच का करते हैं: {p}, अगर आप अभी जुड़ें।", "Chaliye beech ka karte hain: {p}, agar aap abhi judein.")
FINAL_LINE = ("{p} is my final price, I can't go below it; it is a fair one and I'd be glad to have you.",
              "{p} मेरी आख़िरी कीमत है, इससे नीचे नहीं जा सकता; ये ठीक कीमत है और आपका स्वागत है।",
              "{p} meri aakhri keemat hai, isse neeche nahi ja sakta; ye theek keemat hai aur aapka swagat hai.")
ACCEPT_LINES = [
    ("Done, {p} it is, that works for me.", "ठीक है, {p} पक्का, ये मेरे लिए चलेगा।", "Done, {p} pakka, ye mere liye chalega."),
    ("Deal, {p} works for me.", "चलिए, {p} में बात पक्की।", "Chaliye, {p} mein baat pakki."),
    ("Alright, {p} it is.", "ठीक है, {p} कर देता हूँ।", "Theek hai, {p} kar deta hoon."),
]
ASKS = [  # closing-ask forms; never the same twice in a row
    ("Shall I lock {p} and send the link?", "क्या मैं {p} पक्का कर दूँ और लिंक भेज दूँ?", "Kya main {p} pakka kar doon aur link bhej doon?"),
    ("Should I send you the joining link at {p}?", "तो {p} पर जॉइनिंग लिंक भेज दूँ?", "To {p} par joining link bhej doon?"),
    ("Is {p} a deal, then?", "तो {p} में बात पक्की?", "To {p} mein baat pakki?"),
    ("Want me to SMS the link for {p} now?", "{p} का लिंक अभी SMS कर दूँ?", "{p} ka link abhi SMS kar doon?"),
]
LOWEST_LINE = ("{p} is already the lowest I can go.", "{p} मेरी सबसे कम कीमत है।", "{p} meri sabse kam keemat hai.")
SWEET_LEAD = _t("The price can't move any further, but here is something that may help.", "कीमत इससे आगे नहीं हिल सकती, पर एक चीज़ शायद काम आए।",
                "Keemat isse aage nahi hil sakti, par ek cheez shayad kaam aaye.")
SWEET_ORDER = ("referral", "cheaper_plan", "trial_pass", "annual_saving")


def _fill(row, p, price: float) -> dict[str, str]:
    return {l: row[i].format(p=_fmt(p, price)) for i, l in enumerate(LANGS)}


def _pick(st: NegState, key: str, n: int) -> int:
    """The first variant of this bank not yet said on the call; once all were, the least repeated (round robin)."""
    seen = [u for u in st.used if u.startswith(key + ":")]
    for i in range(n):
        if f"{key}:{i}" not in seen:
            st.used.append(f"{key}:{i}")
            return i
    i = len(seen) % n
    st.used.append(f"{key}:{i}")
    return i


def _ask(p, st: NegState, price: float) -> dict[str, str]:
    st.ask_i = (st.ask_i + 1) % len(ASKS)
    return _fill(ASKS[st.ask_i], p, price)


def _round50(x: float) -> int:
    return int(round(x / 50.0)) * 50


def _remaining(eco, st: NegState, cur: float, floor: float):
    perks = [x for x in eco.ladder if x.kind == "perk" and x.id not in st.perks and all(x.text.get(l) for l in LANGS)]
    prices = sorted((x for x in eco.ladder if x.kind == "price" and floor <= x.price < cur), key=lambda x: -x.price)
    return perks, prices


def _ladder_reply(p, kind: str, text: str, st: NegState, need_known: str, committed: bool = False) -> NegReply | None:
    """Close-in-profit bargaining (economics in the product config): anchor at list, reframe value on the first push and let them name
    their number, then concede one ladder step per push (value perks before cash, shrinking price steps, friendly rounding), accept a
    counter at or above the profit floor at once, meet a counter below the current price halfway (never below the floor), tie every price
    to a commitment (no deadline, no scarcity), state the final price once, give one last honest sweetener, then walk away warmly.
    Every price on the table is at or above the profit floor; every line is picked so no sentence repeats on the call."""
    eco = p.economics
    floor = eco.profit_floor
    st = NegState(**asdict(st))
    cur = st.price or eco.list_price
    st.pushes += 1
    say = lambda *banks: {l: _join(*(b[l] for b in banks)) for l in LANGS}
    lead = ACK.get(kind, ACK["haggle"]) if st.pushes == 1 else OPENERS[_pick(st, "open", len(OPENERS))]
    n = None
    target = None
    if kind == "counter_offer":
        n = counter_number(text)
        if n is not None:
            target = float(max(0, cur - n)) if re.search(r"kam|कम|off|less", text, _I) and n < cur else float(n)
    if target is not None and target >= floor:  # at or above the floor: accepted at once, even on the very first push
        price = min(target, cur)
        st.price, st.final_said = price, True
        a = _fill(ACCEPT_LINES[_pick(st, "accept", len(ACCEPT_LINES))], p, price)
        return NegReply("negotiation", say(a, _ask(p, st, price)), st, offer={"price": price}, note="Accepted at once; one closing ask.")
    if st.closed:
        return NegReply("negotiation", CLOSE_WARM[1], st)
    if st.pushes == 1 and target is None:  # first vague push: anchor at list, value, and let them name their number first
        if committed:
            return NegReply("negotiation", say(lead, FIRST_NUMBER), st, note="Already keen: let them name their number first.")
        bank = lang_bank_value({l: _benefit(p, l) for l in LANGS}, need_known)
        keep = _t(f"Would {_fmt(p, cur)} work for you with that, or what number do you have in mind?", f"इसके साथ क्या {_fmt(p, cur)} ठीक रहेगा, या आपके मन में कितना है?",
                  f"Iske saath kya {_fmt(p, cur)} theek rahega, ya aapke mann mein kitna hai?")
        return NegReply("negotiation", say(lead, bank, keep), st, note="Anchor at list; let them name a number; no concession yet.")
    perks, prices = _remaining(eco, st, cur, floor)
    if perks:  # value before cash
        stp = perks[0]
        st.perks.append(stp.id)
        line = {l: stp.text[l] for l in LANGS}
        return NegReply("negotiation", say(PERK_LEADS[_pick(st, "perk", len(PERK_LEADS))], line, _ask(p, st, cur)), st,
                        offer={"price": cur, "perk": stp.id}, note="A perk instead of a price cut, then a closing ask.")
    if prices:
        price = float(prices[0].price)
        row = PRICE_LINES[_pick(st, "price", len(PRICE_LINES))]
        mid_used = False
        if n is not None and kind == "counter_offer":  # below the floor or not yet reachable: meet them part of the way, never under the floor
            mid = float(_round50((cur + n) / 2.0))
            if floor <= mid <= cur - 50 and mid != price:
                price, row, mid_used = mid, MIDDLE_LINE, True
        st.price = price
        _, after = _remaining(eco, st, price, floor)
        if not after:
            st.final_said = True
            return NegReply("price_floor", say(_fill(MIDDLE_LINE, p, price) if mid_used else {l: '' for l in LANGS}, _fill(FINAL_LINE, p, price), _ask(p, st, price)), st, offer={"price": price}, note="Final price, said once, warmly.")
        return NegReply("negotiation", say(lead, _fill(row, p, price), _ask(p, st, price)), st, offer={"price": price}, note="One concession tied to a commitment, one closing ask.")
    if not st.final_said:
        st.final_said = True
        return NegReply("price_floor", say(_fill(LOWEST_LINE, p, cur), _ask(p, st, cur)), st, offer={"price": cur}, note="Floor stated once, warmly.")
    if not st.sweet:  # one last NEW honest offer before the goodbye: a configured lever, never an invented one
        st.sweet = True
        cfg = getattr(p, "negotiation", None)
        for name in SWEET_ORDER:
            if cfg is not None and name in cfg.levers and name not in st.levers and _lever_text(p, name, "en") and all(_lever_text(p, name, l) for l in LANGS):
                st.levers.append(name)
                return NegReply("negotiation", say(SWEET_LEAD, {l: _lever_text(p, name, l) for l in LANGS}, _ask(p, st, cur)), st,
                                offer={"price": cur}, note="Last honest sweetener from the config, then a warm goodbye.")
    st.closed = True
    return NegReply("negotiation", CLOSE_WARM[0], st, note="Final no: warm goodbye, callback offered.")


def lang_bank_value(b: dict, need_known: str) -> dict:
    if not all(b.values()):
        return {l: "" for l in LANGS}
    clean = lambda v: str(v).replace("{", "(").replace("}", ")")
    bank = VALUE
    return {l: bank[l].format(b1=clean(b[l])) for l in LANGS}


def step(p, text: str, st: NegState, need_known: str = "", faq_line=None, committed: bool = False) -> NegReply | None:
    """One negotiation turn. Returns the reply for all three languages and the new state, or None when this is not price talk.
    `faq_line(lang)` gives the operator's approved FAQ answer for the question, when there is one."""
    kind = detect(text)
    if kind is None:
        return None
    st = NegState(**asdict(st))
    st.rounds += 1
    if getattr(p, "economics", None) is not None and p.economics.list_price > 0 and p.economics.ladder \
            and kind in ("haggle", "value_doubt", "budget", "counter_offer", "competitor"):
        return _ladder_reply(p, kind, text, st, need_known, committed)
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
