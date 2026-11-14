"""NEED -> MOVE: what the agent does this turn, with content that is never repeated.

The move policy is profit-free: a selling move is possible only from the need `information`, only once per
call, only after the caller has heard the terms and shown interest, and never after a no, a "later" or a
"stop". Every other need gets a human move: acknowledge, rephrase, answer, give space, reassure, respect,
chat, close.

Content comes from the Product (what it is, benefits, offer terms, approved FAQ answers) and from fixed phrase
banks in three languages (en / hi / hinglish). Rotating phrasings are chosen by a deterministic index (how many
times that move was already used), never random. The SaidLedger (acie/psyche/ledger.py) remembers every
sentence, slot, ask and move; if the chosen text was already said the next unsaid alternative is taken, and if
nothing new is left the move says so honestly, offers to stop, and finally closes politely.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field

from .ledger import SaidLedger
from .needs import NeedResult

LANGS = ("en", "hi", "hinglish")
NO_SELL = ("autonomy", "space", "reassurance", "be_heard", "connection", "closure", "clarity")


@dataclass
class Move:
    name: str                      # ack | simplify | answer_fact | offer_stop | reassure | respect_close | inform | connect | close | nothing_new | silent_close
    need: str = ""
    text: dict[str, str] = field(default_factory=dict)  # lang -> deterministic reply text ("" = say nothing)
    slot: str | None = None        # the fact slot this move tells (ledger.slots)
    sells: bool = False            # asks for the sale (only the one gentle offer)
    brief: bool = False            # keep the reply short
    end_call: bool = False
    avoid: list[str] = field(default_factory=list)  # slots already told: the LLM must not repeat them
    note: str = ""                 # instruction for an LLM-written reply (what this move is for)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_json(cls, raw: str) -> "Move":
        return cls(**json.loads(raw))

    def line(self, lang: str) -> str:
        return self.text.get(lang) or self.text.get("en") or ""


def T(en: str, hi: str, hl: str) -> dict[str, str]:
    return {"en": en, "hi": hi, "hinglish": hl}


def _all(s: str) -> dict[str, str]:
    return {k: s for k in LANGS}


# ---------------------------------------------------------------- phrase banks (rotated by use count)
HEARD_REPEAT = [
    T("You're right, I was repeating myself, sorry. What would you like to know?",
      "आप सही कह रहे हैं, मैं बार-बार एक ही बात बोल रहा था, माफ़ कीजिए। आप क्या जानना चाहते हैं?",
      "Aap sahi keh rahe hain, main baar-baar ek hi baat bol raha tha, sorry. Aap kya jaanna chahte hain?"),
    T("Sorry, that was my mistake. Tell me what you'd like and I'll just answer it.",
      "माफ़ कीजिए, ये मेरी ग़लती थी। आप बताइए क्या चाहिए, मैं बस वही बताऊँगा।",
      "Sorry, ye meri galti thi. Aap bataiye kya chahiye, main bas wahi bataunga."),
    T("I hear you, and I'm sorry. I'll listen first: what should I help with?",
      "मैं आपकी बात समझ रहा हूँ, और माफ़ी चाहता हूँ। पहले आप बताइए, मैं किस बारे में मदद करूँ?",
      "Main aapki baat samajh raha hoon, aur maafi chahta hoon. Pehle aap bataiye, main kis baare mein madad karoon?"),
]
HEARD_GENERIC = [
    T("I'm sorry, I can hear this is frustrating. What would help you most right now?",
      "माफ़ कीजिए, मैं समझ सकता हूँ कि आपको परेशानी हुई। अभी आपकी सबसे ज़्यादा क्या मदद हो सकती है?",
      "Sorry, main samajh sakta hoon ki aapko pareshani hui. Abhi aapki sabse zyada kya madad ho sakti hai?"),
    T("Thank you for telling me, that's fair. Tell me what you'd like me to do.",
      "बताने के लिए शुक्रिया, आपकी बात सही है। आप बताइए मैं क्या करूँ।",
      "Batane ke liye shukriya, aapki baat sahi hai. Aap bataiye main kya karoon."),
    T("You're right to say so, and I'm sorry. I'm listening.",
      "आपने सही कहा, और मैं माफ़ी चाहता हूँ। मैं सुन रहा हूँ।",
      "Aapne sahi kaha, aur main maafi chahta hoon. Main sun raha hoon."),
]
REPEAT_COMPLAINT = re.compile(r"बार बार|bar bar|repeat|रिपेट|रिपीट|same (thing|cheez|बात)|keep (saying|repeating)|तू (चार|बार)", re.I)

ASK_REPEAT = [
    T("Sorry, I didn't catch that. Could you say it once more?",
      "माफ़ कीजिए, मैं ठीक से सुन नहीं पाया। एक बार फिर बोलेंगे?",
      "Sorry, main theek se sun nahi paya. Ek baar phir bolenge?"),
]
LINE_POOR = [
    T("The line doesn't seem clear on my side. Shall I call you another time?",
      "लगता है लाइन साफ़ नहीं है। क्या मैं आपको किसी और समय कॉल करूँ?",
      "Lagta hai line saaf nahi hai. Kya main aapko kisi aur time call karoon?"),
]
ASK_UNCLEAR = [
    T("Which part should I explain better: what it is, or what it costs?",
      "कौन-सा हिस्सा और अच्छे से समझाऊँ: ये क्या है, या इसकी कीमत?",
      "Kaun sa hissa aur achhe se samjhaun: ye kya hai, ya iski keemat?"),
]
SPACE = [
    T("Sorry to catch you at a bad time. Shall I call another time, or leave it here?",
      "माफ़ कीजिए, शायद आपका समय ठीक नहीं है। क्या मैं बाद में कॉल करूँ, या यहीं छोड़ दूँ?",
      "Sorry, shayad aapka time theek nahi hai. Kya main baad mein call karoon, ya yahin chhod doon?"),
    T("I'll keep it short. Would you rather I stop here?",
      "मैं छोटा रखूँगा। क्या आप चाहते हैं कि मैं यहीं रुक जाऊँ?",
      "Main chhota rakhunga. Kya aap chahte hain ki main yahin ruk jaun?"),
]
REASSURE = [
    T("Fair to ask. I'm an AI assistant calling for {seller}. Nothing is charged on this call, and I'd only send a link if you ask me to.",
      "पूछना ठीक है। मैं {seller} की तरफ़ से एक AI असिस्टेंट बोल रहा हूँ। इस कॉल पर कुछ नहीं कटता, और लिंक मैं तभी भेजूँगा जब आप कहेंगे।",
      "Poochna theek hai. Main {seller} ki taraf se ek AI assistant bol raha hoon. Is call par kuch nahi katta, aur link main tabhi bhejunga jab aap kahenge."),
    T("You can check everything yourself on the official website, and there's no pressure from me either way.",
      "आप सब कुछ ऑफिशियल वेबसाइट पर ख़ुद देख सकते हैं, और मेरी तरफ़ से कोई दबाव नहीं है।",
      "Aap sab kuch official website par khud dekh sakte hain, aur meri taraf se koi pressure nahi hai."),
]
RESPECT = [
    T("Of course, no problem at all. Thank you for your time, take care.",
      "बिल्कुल, कोई बात नहीं। आपके समय के लिए धन्यवाद, ध्यान रखिए।",
      "Bilkul, koi baat nahi. Aapke time ke liye thank you, dhyan rakhiye."),
]
CONNECT = [
    T("That's nice to hear, thank you for sharing it. I'll leave the offer aside; ask me anything, or I can let you go.",
      "ये जानकर अच्छा लगा, बताने के लिए शुक्रिया। ऑफर की बात मैं अभी छोड़ देता हूँ; कुछ पूछना हो तो पूछिए, या मैं आपको जाने दूँ।",
      "Ye jaankar achha laga, batane ke liye shukriya. Offer ki baat main abhi chhod deta hoon; kuch poochna ho to poochiye, ya main aapko jaane doon."),
    T("I'm an AI, so I can't match that with a story of my own, but I'm glad you told me.",
      "मैं एक AI हूँ, इसलिए अपनी कोई कहानी नहीं सुना सकता, लेकिन आपने बताया, अच्छा लगा।",
      "Main ek AI hoon, isliye apni koi kahani nahi suna sakta, lekin aapne bataya, achha laga."),
]
CLOSE = [
    T("Thank you for your time. Take care, goodbye.",
      "आपके समय के लिए धन्यवाद। ध्यान रखिए, नमस्ते।",
      "Aapke time ke liye thank you. Dhyan rakhiye, namaste."),
    T("Alright, bye for now.", "ठीक है, फिर मिलते हैं।", "Theek hai, phir milte hain."),
]
NOTHING_NEW = [
    T("Honestly, that's everything I have on this. Is there anything else you'd like to ask, or shall I let you go?",
      "ईमानदारी से, इस बारे में मेरे पास बस इतना ही है। कुछ और पूछना हो तो बताइए, या मैं आपको जाने दूँ?",
      "Imaandaari se, is baare mein mere paas bas itna hi hai. Kuch aur poochna ho to bataiye, ya main aapko jaane doon?"),
]
NO_DETAIL = [
    T("I don't have that detail with me; the official website will have it. Is there anything else I can answer?",
      "वो जानकारी मेरे पास नहीं है; ऑफिशियल वेबसाइट पर मिल जाएगी। और कुछ है जो मैं बता सकूँ?",
      "Wo jaankari mere paas nahi hai; official website par mil jaayegi. Aur kuch hai jo main bata sakoon?"),
]
SWITCH_EN = ["Sorry about that, I'll speak English from now on. What would you like to know?"]
SWITCH_HI = ["माफ़ कीजिए, अब मैं हिंदी में बात करूँगा। आप क्या जानना चाहेंगे?"]
GENTLE_OFFER = T("If you'd like it, I can send the link by SMS; only if you want it.",
                 "अगर आप चाहें तो मैं SMS से लिंक भेज सकता हूँ; सिर्फ़ तभी जब आप चाहें।",
                 "Agar aap chahein to main SMS se link bhej sakta hoon; sirf tabhi jab aap chahein.")

_PRICE_Q = re.compile(r"price|cost|how much|kitn|कितन|कीमत|दाम|पैसे|rupe|रुपए|रुपये|discount|offer|ऑफर|छूट|expensive|महंगा|mehenga|cheap|सस्ता", re.I)
_HOW_Q = re.compile(r"how (to|do|can) (i )?(pay|buy|join|subscribe|take)|कैसे (खरीद|पेमेंट|ले|जॉइन|join)|link|लिंक|payment|पेमेंट|join", re.I)
_WHAT_Q = re.compile(r"what (is|does|do)|what.?s (in|inside)|features?|benefits?|क्या (है|होता|मिलता)|फीचर|कौन कौन|क्या क्या|machines?|मचीन|मशीन|classes|क्लास|tell me about", re.I)


# ---------------------------------------------------------------- catalog from the Product
@dataclass
class Item:
    slot: str
    kind: str                 # fact | benefit | faq
    text: dict[str, str]
    keywords: list[str] = field(default_factory=list)


def _esc(s: str) -> str:
    return s.replace("{", "(").replace("}", ")")


def build_catalog(p) -> list[Item]:
    """Content the agent may say, in the order `information` uses it. Built only from Product fields."""
    name, one = _esc(p.product_name), _esc(p.one_liner)
    items = [Item("what", "fact", T(_esc(p.pitch_en or f"{name} is {one}."), _esc(p.pitch_hi or p.pitch_en or f"{name} is {one}."),
                                    _esc(p.pitch_hinglish or p.pitch_hi or p.pitch_en or f"{name} is {one}.")))]
    for i, b in enumerate(p.benefits):
        b = _esc(b)
        items.append(Item(f"benefit_{i + 1}", "benefit",
                          T(f"One thing you get: {b}.", f"एक चीज़ जो मिलती है: {b}।", f"Ek cheez jo milti hai: {b}.")))
    d, o, ov, op = _esc(p.discount), _esc(p.offer_price), _esc(p.offer_validity), _esc(p.original_price)
    items.append(Item("offer", "fact", T(f"The offer is {d}: {o} {ov}, normally {op}.",
                                         f"ऑफर है {d}: {o} {ov}, वैसे {op}।",
                                         f"Offer hai {d}: {o} {ov}, waise {op}."),
                      ["price", "cost", "how much", "kitn", "कितन", "कीमत", "दाम", "रुपए", "रुपये", "discount", "छूट", "offer", "ऑफर", "expensive", "महंगा"]))
    items.append(Item("cancel", "fact", T("You can cancel any time.", "आप कभी भी cancel कर सकते हैं।", "Aap kabhi bhi cancel kar sakte hain."),
                      ["cancel", "कैंसल", "refund", "रिफंड", "stop paying"]))
    items.append(Item("how", "fact", T(f"Nothing is charged on this call; if you want it, a secure link comes by SMS to this number.",
                                       "इस कॉल पर कुछ नहीं कटता; आप चाहें तो इसी नंबर पर SMS से एक सुरक्षित लिंक आ जाएगा।",
                                       "Is call par kuch nahi katta; aap chahein to isi number par SMS se ek surakshit link aa jaayega."),
                      ["pay", "link", "लिंक", "payment", "पेमेंट", "join", "buy", "खरीद", "kaise"]))
    for k, f in enumerate(p.faqs):
        if f.answer and not f.not_eligible:
            items.append(Item(f"faq_{k}", "faq", T(_esc(f.answer) + ".", _esc(f.answer_hi or f.answer) + "।", _esc(f.answer) + "."),
                              [x.lower() for x in f.keywords if not x.startswith("re:")]))
    return items


@dataclass
class MoveContext:
    need: NeedResult
    lang: str = "hi"
    product: object | None = None
    ledger: SaidLedger = field(default_factory=SaidLedger)
    text: str = ""
    turn_index: int = 0
    frustration: float = 0.0
    engagement: float = 0.5
    decided: str = "undecided"     # call context decision
    has_link: bool = False
    refused: bool = False          # said no / later / stop earlier in this call


def _fill(s: str, p) -> str:
    return s.replace("{seller}", getattr(p, "seller_name", "us")) if p is not None else s.replace("{seller}", "us")


def _rotate(bank, ledger: SaidLedger, name: str, p, lang: str):
    """The next phrasing of a bank that is not a repeat (start index = how often `name` was used). None if all used."""
    n = len(bank)
    start = ledger.count_move(name)
    for k in range(n):
        v = bank[(start + k) % n]
        text = {l: _fill(v[l], p) for l in LANGS} if isinstance(v, dict) else _all(v)
        if not ledger.repeats(text.get(lang) or text["en"]):
            return text
    return None


def _mk(name: str, mc: MoveContext, text: dict[str, str] | None, **kw) -> Move | None:
    return None if text is None else Move(name=name, need=mc.need.need, text=text, **kw)


def _exhausted(mc: MoveContext, why: str) -> Move:
    """Nothing new to say: be honest, offer to stop, then close, then fall silent. Never repeat."""
    L, p, lang = mc.ledger, mc.product, mc.lang
    for name, bank, end in (("nothing_new", NOTHING_NEW, False), ("close", CLOSE, True)):
        m = _mk(name, mc, _rotate(bank, L, name, p, lang), end_call=end, brief=True, note=why)
        if m is not None:
            return m
    return Move(name="silent_close", need=mc.need.need, text=_all(""), end_call=True, brief=True, note=why)


def _info_items(mc: MoveContext, kinds=("fact", "benefit", "faq")) -> list[Item]:
    if mc.product is None:
        return []
    return [i for i in build_catalog(mc.product) if i.kind in kinds and not mc.ledger.has_slot(i.slot)
            and not mc.ledger.repeats(i.text.get(mc.lang) or i.text["en"])]


def _matches(item: Item, text: str) -> bool:
    low = text.lower()
    return any(k.lower() in low for k in item.keywords if k.strip())


def _do_be_heard(mc: MoveContext) -> Move:
    bank = HEARD_REPEAT if REPEAT_COMPLAINT.search(mc.text) else HEARD_GENERIC
    m = _mk("ack", mc, _rotate(bank, mc.ledger, "ack", mc.product, mc.lang), brief=True,
            note="Acknowledge the caller's feeling in a few words and apologise; do not sell, do not repeat earlier lines.")
    return m or _exhausted(mc, "acknowledged enough")


def _do_clarity(mc: MoveContext) -> Move:
    f, L, p = mc.need.flags, mc.ledger, mc.product
    t = mc.text
    if f.get("lang_trouble"):
        wants_hi = bool(re.search(r"हिंदी|hindi", t, re.I)) and not re.search(r"(don.?t|नहीं).{0,20}hindi|hindi (नहीं|nahi)|speak english|in english", t, re.I)
        bank = SWITCH_HI if wants_hi else SWITCH_EN
        m = _mk("ack", mc, _rotate(bank, L, "lang_switch", p, mc.lang), brief=True,
                note="Switch to the caller's language, apologise briefly, and ask what they would like to know.")
        if m:
            return m
    if f.get("garbled") or not L.sentences:
        ladder = [("ask_repeat", ASK_REPEAT), ("offer_stop", LINE_POOR), ("close", CLOSE)]
    else:
        ladder = [("simplify", None), ("simplify2", None), ("ask_unclear", ASK_UNCLEAR), ("offer_stop", LINE_POOR), ("close", CLOSE)]
    start = max(0, L.count_move("need:clarity") - 1)  # (the pipeline counts the need before choosing)  # one rung further each time the caller still needs clarity
    for step, (name, bank) in enumerate(ladder[start:], start):
        if bank is None:
            # simpler words: the one-line version of the next unsaid fact, introduced as a rephrase
            facts = [i for i in _info_items(mc) if i.kind in ("fact", "benefit")]
            if not facts:
                continue
            it = facts[0]
            intro = T("Let me say it more simply. ", "इसे और आसान शब्दों में बताता हूँ। ", "Ise aur aasan shabdon mein batata hoon. ")
            text = {l: intro[l] + it.text[l] for l in LANGS}
            mv = Move(name=name, need="clarity", text=text, slot=it.slot, brief=True,
                      note="Rephrase in simpler everyday words with a different example; never repeat the earlier wording.")
            return mv
        text = _rotate(bank, L, name, p, mc.lang)
        if text is None:
            continue
        return Move(name=name, need="clarity", text=text, brief=True, end_call=(name == "close"),
                    note="Ask what was unclear or offer to stop; one short line.")
    return _exhausted(mc, "clarity ladder done")


_GENERIC_TOPIC = {"it", "this", "ye", "yeh", "यह", "ये", "इसमें", "इसका", "इसको", "इसे", "isme", "iska", "isko", "ise", "plan", "प्लान",
                  "offer", "ऑफर", "app", "ऐप", "pro", "प्रो", "ai", "एआई", "assistant", "असिस्टेंट", "subscription", "सब्सक्रिप्शन"}
_PRICE_WORDS = re.compile(r"price|cost|how much|kitn|कितन|कीमत|दाम|पैसे|charge|चार्ज|फीस|fees?\b|rate\b|रेट", re.I)


def item_words(p) -> dict[str, str]:
    """Words naming something the offer INCLUDES (a trainer, classes, a feature): topic words that appear in a
    benefit. A Devanagari word listed right after its English twin shares that benefit. word -> benefit text."""
    ben = [b for b in getattr(p, "benefits", [])]
    own = {w.lower() for w in re.findall(r"\w+", f"{p.product_name} {p.seller_name}")}
    out: dict[str, str] = {}
    words = list(getattr(p, "topic_words", []))
    for i, w in enumerate(words):
        lw = w.lower()
        if lw in _GENERIC_TOPIC or lw in own or not re.fullmatch(r"[a-z]+", lw):
            continue
        for b in ben:
            if lw in b.lower():
                out[lw] = b
                if i + 1 < len(words) and re.search(r"[\u0900-\u097f]", words[i + 1]):
                    out[words[i + 1].lower()] = b
                break
    return out


def item_price_answer(p, text: str, lang: str) -> tuple[str, str] | None:
    """A price question that names something already described ("personal trainer कितने का?") is answered from
    the product facts (it is included), not with the membership price. Returns (benefit, text) or None."""
    if p is None or not _PRICE_WORDS.search(text):
        return None
    low = text.lower()
    for w, b in sorted(item_words(p).items()):
        if re.search(r"(?<![\w\u0900-\u097f])" + re.escape(w), low):
            b = _esc(b)
            t = T(f"{b}: that part is included in the offer, with no separate price. For anything beyond it I don't have a rate; the official website has it.",
                  f"{b}: ये हिस्सा ऑफर में शामिल है, इसकी अलग कीमत नहीं है। इससे ज़्यादा के लिए मेरे पास कोई रेट नहीं है; ऑफिशियल वेबसाइट पर मिल जाएगा।",
                  f"{b}: ye hissa offer mein shamil hai, iski alag keemat nahi hai. Isse zyada ke liye mere paas koi rate nahi hai; official website par mil jaayega.")
            return b, t[lang] if lang in t else t["en"]
    return None


def _do_answer(mc: MoveContext) -> Move:
    t, L = mc.text, mc.ledger
    if mc.product is None:
        return _exhausted(mc, "no catalog")
    ip = item_price_answer(mc.product, t, mc.lang)
    if ip is not None:
        slot = "item:" + re.sub(r"\W+", "_", ip[0].lower())[:30]
        full = {l: item_price_answer(mc.product, t, l)[1] for l in LANGS}
        if not L.has_slot(slot) and not L.repeats(full.get(mc.lang) or full["en"]):
            return Move(name="answer_fact", need="answer", text=full, slot=slot, brief=True,
                        note="Answer what they asked about this included item from the facts; no selling question.")
        return _exhausted(mc, "already answered this item question")
    cat = [i for i in build_catalog(mc.product)]
    cands = [i for i in cat if _matches(i, t)]
    if not cands:
        if _WHAT_Q.search(t):
            cands = [i for i in cat if i.kind == "benefit"] or cat[:1]
        elif _PRICE_Q.search(t):
            cands = [i for i in cat if i.slot == "offer"]
        elif _HOW_Q.search(t):
            cands = [i for i in cat if i.slot == "how"]
    fresh = [i for i in cands if not L.has_slot(i.slot) and not L.repeats(i.text.get(mc.lang) or i.text["en"])]
    if fresh:
        it = fresh[0]
        return Move(name="answer_fact", need="answer", text=it.text, slot=it.slot, brief=mc.frustration >= 0.4,
                    note="Answer the question first and fully from the product facts; do not add a selling question.")
    if cands:  # they asked about something we already said: honest, not a copy
        return _exhausted(mc, "already answered this")
    m = _mk("no_detail", mc, _rotate(NO_DETAIL, L, "no_detail", mc.product, mc.lang), brief=True,
            note="Say honestly you do not have that detail; the official website has it.")
    if m:
        return m
    return _exhausted(mc, "no further detail")


def _do_information(mc: MoveContext) -> Move:
    L = mc.ledger
    items = _info_items(mc)
    if not items:
        return _exhausted(mc, "nothing new to tell")
    it = items[0]
    text = dict(it.text)
    sells = False
    interest = (mc.need.flags.get("go_on") or mc.need.flags.get("sale")) and mc.engagement >= 0.4
    if (interest and L.offers == 0 and not L.refused and not mc.refused and mc.decided == "undecided" and not mc.has_link
            and L.has_slot("offer") and L.has_slot(it.slot) is False and it.slot != "offer"):
        text = {l: it.text[l] + " " + GENTLE_OFFER[l] for l in LANGS}
        sells = True
    return Move(name="inform", need="information", text=text, slot=it.slot, sells=sells,
                note="Give the next useful fact in one or two short sentences; only one gentle offer per call.")


def _simple(name: str, bank, mc: MoveContext, need_name: str, **kw) -> Move:
    m = _mk(name, mc, _rotate(bank, mc.ledger, name, mc.product, mc.lang), brief=True, **kw)
    return m or _exhausted(mc, f"{name} already said")


def choose_move(mc: MoveContext) -> Move:
    """The policy. Deterministic: same context -> same move."""
    need = mc.need.need
    if need == "closure":
        return _simple("close", CLOSE, mc, need, end_call=True, note="Close politely.")
    if need == "autonomy":
        mc.ledger.refused = True
        return _simple("respect_close", RESPECT, mc, need, end_call=True,
                       note="Respect the no or the later; close politely; never offer again.")
    if need == "be_heard":
        return _do_be_heard(mc)
    if need == "reassurance":
        t = mc.text
        if mc.product is not None:
            trust = [i for i in build_catalog(mc.product) if i.kind == "faq" and _matches(i, t) and not mc.ledger.has_slot(i.slot)
                     and not mc.ledger.repeats(i.text.get(mc.lang) or i.text["en"])]
            if trust:
                return Move(name="reassure", need=need, text=trust[0].text, slot=trust[0].slot, brief=True,
                            note="Give honest facts, no pressure, no selling question.")
        return _simple("reassure", REASSURE, mc, need, note="Give honest facts, no pressure, no selling question.")
    if need == "clarity":
        return _do_clarity(mc)
    if need == "answer":
        return _do_answer(mc)
    if need == "space":
        return _simple("offer_stop", SPACE, mc, need, note="Be brief and offer to stop or call back.")
    if need == "connection":
        return _simple("connect", CONNECT, mc, need, note="Reply like a warm human in one short line; no selling.")
    return _do_information(mc)


def repeat_move(mc: MoveContext, intent: str) -> Move:
    """A deterministic plan was about to repeat an earlier reply word for word (or nearly). Say something that
    helps instead: a garbled/unclear turn gets a different request, a finished call a polite close, anything
    else the honest "that's all I have"."""
    if mc.has_link or intent in ("goodbye", "link_sent", "declined"):
        return _simple("close", CLOSE, mc, "closure", end_call=True, note="Close politely; do not repeat.")
    if intent in ("not_understood", "clarify", "hello", "off_topic"):
        mc.need.flags["garbled"] = True
        return _do_clarity(mc)
    return _exhausted(mc, "the reply would have repeated an earlier one")
