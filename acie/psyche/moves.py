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

from ..decision import honest
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
KEEP_LISTENING = [
    T("Sure. What else would you like to know?", "जी, और क्या जानना चाहेंगे?", "Ji, aur kya jaanna chahenge?"),
    T("I'm listening, please go ahead.", "मैं सुन रहा हूँ, बताइए।", "Main sun raha hoon, bataiye."),
    T("If it helps, someone from the team can call you with more detail. Would you like that?",
      "अगर चाहें तो टीम से कोई आपको पूरी जानकारी के साथ कॉल कर सकता है। ऐसा करें?",
      "Agar chahein to team se koi aapko poori jaankari ke saath call kar sakta hai. Aisa karein?"),
    T("I can also call back at a time that suits you. Would that be better?",
      "मैं आपके सुविधा के समय पर दोबारा कॉल भी कर सकता हूँ। वो ठीक रहेगा?",
      "Main aapke suvidha ke time pe dobara call bhi kar sakta hoon. Wo theek rahega?"),
]
# the last rung: a short listening cue. Reused in a fixed rotation if every one was said: a two-word cue said again
# is far better than hanging up on a caller who is still asking (live replay: hung up with ten questions left)
LISTEN_CUES = [T("Yes, go on.", "जी, बताइए।", "Ji, bataiye."), T("Yes?", "जी?", "Ji?"),
               T("I'm here.", "जी, मैं यहीं हूँ।", "Ji, main yahin hoon.")]
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
SOFT_CHECK = T("No problem at all. Would another time suit you better, or shall I just leave it?",
               "कोई बात नहीं। क्या किसी और समय बात करना ठीक रहेगा, या मैं यहीं छोड़ दूँ?",
               "Koi baat nahi. Kya kisi aur time baat karna theek rahega, ya main yahin chhod doon?")
SWITCH_EN = ["Sorry about that, I'll speak English from now on. What would you like to know?"]
SWITCH_HI = ["माफ़ कीजिए, अब मैं हिंदी में बात करूँगा। आप क्या जानना चाहेंगे?"]
GENTLE_OFFER = T("If you'd like it, I can send the link by SMS; only if you want it.",
                 "अगर आप चाहें तो मैं SMS से लिंक भेज सकता हूँ; सिर्फ़ तभी जब आप चाहें।",
                 "Agar aap chahein to main SMS se link bhej sakta hoon; sirf tabhi jab aap chahein.")

_PRICE_Q = re.compile(r"price|cost|how much|kitn|कितन|कीमत|दाम|पैसे|rupe|रुपए|रुपये|discount|offer|ऑफर|छूट|expensive|महंगा|mehenga|cheap|सस्ता", re.I)
_HOW_Q = re.compile(r"how (to|do|can) (i )?(pay|buy|join|subscribe|take)|कैसे (खरीद|पेमेंट|ले|जॉइन|join)|link|लिंक|payment|पेमेंट|join", re.I)
_WHAT_Q = re.compile(r"what (does|do) (it|this|that)\b.{0,20}(do|include|have|offer|give)|what do i get|what.?s (in|inside|included)|features?|benefits?|"
                     r"क्या क्या|क्या (मिलता|मिलेगा|करता)|फीचर|कौन कौन|tell me about|kya kya|kya (milta|milega|karta)|इसमें क्या|isme kya", re.I)


# ---------------------------------------------------------------- catalog from the Product
@dataclass
class Item:
    slot: str
    kind: str                 # fact | benefit | faq
    text: dict[str, str]      # only the languages the product really has text for ("en" always); never an English paste
    keywords: list[str] = field(default_factory=list)
    rx: list[str] = field(default_factory=list)  # regex keywords ("re:" entries of an FAQ)

    def line(self, lang: str) -> str | None:
        return self.text.get(lang)


def _esc(s: str) -> str:
    return s.replace("{", "(").replace("}", ")")


_STOP = {"every", "like", "with", "your", "plus", "much", "more", "than", "free", "plan", "access", "most", "that", "this", "from", "have", "week",
         "open", "unlimited", "session", "personal", "organise", "documents", "chats"}


def _derived_words(b: str, p) -> set[str]:
    """Words a caller would use for this benefit: its own content words, plus time words when it states opening hours."""
    own = {w.lower() for w in re.findall(r"\w+", f"{p.product_name} {p.seller_name}")}
    out = {w for w in re.findall(r"[a-z]{5,}", b.lower()) if w not in _STOP and w not in own}
    if re.search(r"\b\d+ ?(am|pm)\b|\bopen\b|hours", b, re.I):
        out |= {"timing", "timings", "hours", "टाइमिंग", "समय", "kab tak", "कब तक खुल", "khula", "खुला", "open", "opening"}
    return out


def build_catalog(p) -> list[Item]:
    """Content the agent may say, in the order `information` uses it. Built only from Product fields. A language with no
    text for an item is simply absent from `Item.text`: the move policy then lets the LLM paraphrase the English fact
    instead of pasting English into a Hindi or Hinglish reply."""
    name, one = _esc(p.product_name), _esc(p.one_liner)
    items = [Item("what", "fact", T(_esc(p.pitch_en or f"{name} is {one}."), _esc(p.pitch_hi or p.pitch_en or f"{name} is {one}."),
                                    _esc(p.pitch_hinglish or p.pitch_hi or p.pitch_en or f"{name} is {one}.")),
                  ["membership", "मेंबरशिप", "what is", "kya hota", "क्या होता", "fitzone", "फिटज़ोन", "फिट-ज़ोन", "फिट ज़ोन", "फिटजोन"])]
    words = item_words(p)
    ben_hi, ben_hl = list(getattr(p, "benefits_hi", []) or []), list(getattr(p, "benefits_hinglish", []) or [])
    for i, b in enumerate(p.benefits):
        be = _esc(b)
        text = {"en": f"One thing you get: {be}."}
        if i < len(ben_hi) and ben_hi[i].strip():
            text["hi"] = f"एक चीज़ जो मिलती है: {_esc(ben_hi[i])}।"
        if i < len(ben_hl) and ben_hl[i].strip():
            text["hinglish"] = f"Ek cheez jo milti hai: {_esc(ben_hl[i])}."
        kws = sorted({w for w, bb in words.items() if bb == b} | _derived_words(b, p))
        items.append(Item(f"benefit_{i + 1}", "benefit", text, kws))
    d, o, ov, op = _esc(p.discount), _esc(p.offer_price), _esc(p.offer_validity), _esc(p.original_price)
    items.append(Item("offer", "fact", T(f"The offer is {d}: {o} {ov}, normally {op}.",
                                         f"ऑफर है {d}: {o} {ov}, वैसे {op}।",
                                         f"Offer hai {d}: {o} {ov}, waise {op}."),
                      ["price", "cost", "how much", "kitn", "कितन", "कीमत", "दाम", "रुपए", "रुपये", "रुपया", "rupee", "rupaye", "dollar", "डॉलर", "discount",
                       "छूट", "offer", "ऑफर", "expensive", "महंगा", "months", "महीने", "mahine", "3 month", "after that", "uske baad", "उसके बाद"]))
    if honest.can_cancel(p):
        items.append(Item("cancel", "fact", T("You can cancel any time.", "आप कभी भी cancel कर सकते हैं।", "Aap kabhi bhi cancel kar sakte hain."),
                          ["cancel", "कैंसल", "stop paying"]))
    items.append(Item("how", "fact", T(f"Nothing is charged on this call; if you want it, a secure link comes by SMS to this number.",
                                       "इस कॉल पर कुछ नहीं कटता; आप चाहें तो इसी नंबर पर SMS से एक सुरक्षित लिंक आ जाएगा।",
                                       "Is call par kuch nahi katta; aap chahein to isi number par SMS se ek surakshit link aa jaayega."),
                      ["pay", "link", "लिंक", "payment", "पेमेंट", "join", "buy", "खरीद", "kaise"]))
    for k, f in enumerate(p.faqs):
        if f.answer and not f.not_eligible:
            text = {"en": _esc(f.answer) + "."}
            if f.answer_hi:
                text["hi"] = _esc(f.answer_hi) + "।"
            if getattr(f, "answer_hinglish", ""):
                text["hinglish"] = _esc(f.answer_hinglish) + "."
            items.append(Item(f"faq_{k}", "faq", text, [x.lower() for x in f.keywords if not x.startswith("re:")],
                              [x[3:] for x in f.keywords if x.startswith("re:")]))
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
    hold: bool = False             # sad / money trouble earlier in the call: selling is paused until they re-open it


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
    """Nothing new to say. For a NON-question: be honest, then keep listening (ask what else / offer a person or a
    callback), then a short listening cue; it never ends the call (only the caller's own goodbye, no or stop does).
    For a QUESTION it is never used: a question always gets an answer (the fact in other words, or an honest "I don't
    have that detail"), and "shall I let you go?" is never the reply to one."""
    L, p, lang = mc.ledger, mc.product, mc.lang
    if honest.is_question(mc.text) and p is not None:
        return answer_again(mc, why)
    for name, bank in (("nothing_new", NOTHING_NEW), ("keep_listening", KEEP_LISTENING)):
        m = _mk(name, mc, _rotate(bank, L, name, p, lang), brief=True, note=why)
        if m is not None:
            return m
    m = _mk("listen", mc, _rotate(LISTEN_CUES, L, "listen", p, lang), brief=True, note=why)
    if m is not None:
        return m
    cue = LISTEN_CUES[L.count_move("listen") % len(LISTEN_CUES)]  # deterministic reuse, never silence + hang-up
    return Move(name="listen", need=mc.need.need, text=dict(cue), brief=True, note=why)


def _info_items(mc: MoveContext, kinds=("fact", "benefit", "faq")) -> list[Item]:
    if mc.product is None:
        return []
    return [i for i in build_catalog(mc.product) if i.kind in kinds and mc.lang in i.text and not mc.ledger.has_slot(i.slot)
            and not mc.ledger.repeats(i.text[mc.lang])]


def _matches(item: Item, text: str) -> bool:
    low = text.lower()
    return any(k.lower() in low for k in item.keywords if k.strip()) or any(re.search(r, low) for r in item.rx)


def _by_lang_one(d: dict[str, str]) -> dict[str, str]:
    return dict(d)


def _resolvable(mc: MoveContext) -> bool:
    """The question names something the facts cover (a catalog item, a price) or a known gap: it can be answered now."""
    p, t = mc.product, mc.text
    if honest.find_topics(p, t) or honest.health_question(t):
        return True
    return any(_matches(i, t) for i in build_catalog(p)) or bool(_PRICE_Q.search(t) or _WHAT_IS.search(t))


def _by_lang(fn) -> dict[str, list[str]]:
    return {l: fn(l) for l in LANGS}


def _say(mc: MoveContext, name: str, variants: dict[str, list[str]], slot: str | None = None, note: str = "") -> Move:
    """One honest/answering move from per-language variant lists. The variant is chosen by how many times THIS answer was
    already given (the ledger counts moves named "<name>:<slot>"), so asking the same thing again gives the next wording and
    a fact restated for a caller who asked again is never held back as a "repeat". After the last, the last again."""
    L, lang = mc.ledger, mc.lang
    key = f"{name}:{slot}" if slot else name
    mine = variants.get(lang) or variants["en"]
    idx = min(L.count_move(key), len(mine) - 1)
    text = {l: v[min(idx, len(v) - 1)] for l, v in variants.items() if v}
    return Move(name=key, need=mc.need.need, text=text, slot=slot, brief=mc.frustration >= 0.4 or mc.need.need in ("clarity", "space"),
                note=note or "Answer what they asked; if it is not in the facts say so honestly; never offer to end the call.")


def _unknown(mc: MoveContext, topic=None) -> Move:
    p = mc.product
    topics = honest.find_topics(p, mc.text) if topic is not None else []
    if topic is not None and len(topics) > 1 and topics[0].id == topic.id:
        n = min(mc.ledger.count_move(f"answer_unknown:topic:{topic.id}"), 2)
        return Move(name=f"answer_unknown:topic:{topic.id}", need=mc.need.need, slot="topic:" + topic.id, brief=True,
                    text={l: honest.unknown_combo(p, l, topics, n) for l in LANGS},
                    note="Say honestly you do not have those details; do not guess.")
    return _say(mc, "no_detail" if topic is None else "answer_unknown", _by_lang(lambda l: honest.unknown_variants(p, l, topic)),
                slot="topic:" + (topic.id if topic is not None else "generic"),
                note="Say honestly you do not have that detail and where they will find it; do not guess.")


def _restate(it: Item, p, lang: str) -> list[str]:
    """The same fact in other shapes (never the first wording): for a re-asked question."""
    if it.slot == "offer":
        return honest.price_variants(p, lang)
    base = it.text.get(lang) or ""
    if it.kind == "benefit" and base:
        core = base.split(":", 1)[-1].strip(" .।")
        return {"en": [f"{honest.cap(core)} is part of it.", f"It includes {core}."],
                "hi": [f"{core} इसमें शामिल है।", f"इसमें {core} भी मिलता है।"],
                "hinglish": [f"{honest.cap(core)} isme shamil hai.", f"Isme {core} bhi milta hai."]}[lang]
    lead = {"en": ["Once more, in short: ", "To say it again: "], "hi": ["छोटे में फिर से: ", "दोबारा बता देता हूँ: "],
            "hinglish": ["Chhote mein phir se: ", "Dobara bata deta hoon: "]}[lang]
    return [l + base for l in lead] if base else []


def answer_again(mc: MoveContext, why: str = "") -> Move:
    """The caller asked (again) and the fresh facts are used up: answer anyway. Order: a topic the facts do not cover ->
    the honest line (a different one each time); a catalog item already said -> the same fact in other words, finally
    "that's all I have on this, <where to look>"; nothing matched -> the honest generic line. Never a listening cue."""
    t, p, L, lang = mc.text, mc.product, mc.ledger, mc.lang
    kind = honest.info_request(t)
    if kind is not None:
        return _say(mc, "answer_fact", {l: honest.info_request_variants(p, kind, l) + honest.end_variants(p, l) for l in LANGS}, slot="info:" + kind,
                    note="Say honestly you can only send the payment link by SMS; do not offer to end the call.")
    topics = honest.find_topics(p, t)
    if topics and (topics[0].beats_faq or not honest.covered_by_faq(p, t)):
        return _unknown(mc, topics[0])
    ip = item_price_answer(p, t, lang)
    if ip is not None:
        bl = {l: item_price_answer(p, t, l)[0] if l == "en" else _benefit_in(p, item_price_answer(p, t, "en")[0], l) for l in LANGS}
        again = {"en": f"So it is included: {bl['en']}. There is no separate charge for it.",
                 "hi": f"तो वो शामिल है: {bl['hi']}। उसका अलग से कोई चार्ज नहीं है।",
                 "hinglish": f"Toh wo shamil hai: {bl['hinglish']}. Uska alag se koi charge nahi hai."}
        variants = {l: [again[l]] + honest.end_variants(p, l) for l in LANGS}
        return _say(mc, "answer_fact", variants, slot="item_price", note="They asked again: answer again; do not offer to end the call.")
    if honest.NEGOTIATE.search(t):
        return _say(mc, "answer_fact", _by_lang(lambda l: honest.negotiate_variants(p, l)), slot="negotiate", note="Price is fixed; say so again in other words.")
    cat = build_catalog(p)
    cands = sorted([i for i in cat if _matches(i, t)], key=lambda i: 0 if i.kind == "faq" else 1) \
        or ([i for i in cat if i.slot == "offer"] if _PRICE_Q.search(t) else [])
    cands = [i for i in cands if mc.lang in i.text]
    if cands:
        it = cands[0]
        variants = {l: (_restate(it, p, l) if it.text.get(l) or it.slot == "offer" else []) + honest.end_variants(p, l) for l in LANGS}
        return _say(mc, "answer_fact", variants, slot=it.slot,
                    note="They asked again: say the same fact in different, simpler words; do not offer to end the call.")
    return _unknown(mc)


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
        ladder = [("ask_repeat", ASK_REPEAT), ("offer_stop", LINE_POOR), ("keep_listening", KEEP_LISTENING)]
    else:
        ladder = [("simplify", None), ("simplify2", None), ("ask_unclear", ASK_UNCLEAR), ("offer_stop", LINE_POOR), ("keep_listening", KEEP_LISTENING)]
    start = max(0, L.count_move("need:clarity") - 1)  # (the pipeline counts the need before choosing)  # one rung further each time the caller still needs clarity
    for step, (name, bank) in enumerate(ladder[start:], start):
        if bank is None:
            # simpler words: the one-line version of the next unsaid fact, introduced as a rephrase
            facts = [i for i in _info_items(mc) if i.kind in ("fact", "benefit")]
            if not facts:
                continue
            it = facts[0]
            intro = T("Let me say it more simply. ", "इसे और आसान शब्दों में बताता हूँ। ", "Ise aur aasan shabdon mein batata hoon. ")
            text = {l: intro[l] + it.text[l] for l in LANGS if l in it.text}
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


def _benefit_in(p, b: str, lang: str) -> str:
    """The benefit's wording in the caller's language (benefits_hi / benefits_hinglish), else None-ish ''."""
    try:
        i = list(p.benefits).index(b)
    except ValueError:
        return ""
    lst = {"hi": getattr(p, "benefits_hi", []), "hinglish": getattr(p, "benefits_hinglish", [])}.get(lang)
    return (lst[i] if lst and i < len(lst) else "") or ""


def item_price_answer(p, text: str, lang: str) -> tuple[str, str] | None:
    """A price question that names something already described ("personal trainer कितने का?") is answered from
    the product facts (it is included), not with the membership price. Returns (benefit, text) or None. The benefit is
    spoken in the caller's language; with no translation the answer is skipped for that language (the LLM paraphrases)."""
    if p is None or not _PRICE_WORDS.search(text) or honest.covered_by_faq(p, text):
        return None  # an approved FAQ answer (e.g. the rate of extra trainer sessions) beats the generic "it is included"
    low = text.lower()
    for w, b in sorted(item_words(p).items()):
        if re.search(r"(?<![\w\u0900-\u097f])" + re.escape(w), low):
            bl = _esc(b) if lang == "en" else _esc(_benefit_in(p, b, lang))
            if not bl:
                return None
            S = honest.source(p, lang)
            t = {"en": f"{honest.cap(bl)}: that part is included in the offer, with no separate price. For anything beyond it I don't have a rate; {S}.",
                 "hi": f"{bl}: ये हिस्सा ऑफर में शामिल है, इसकी अलग कीमत नहीं है। इससे ज़्यादा के लिए मेरे पास कोई रेट नहीं है; {S}।",
                 "hinglish": f"{honest.cap(bl)}: ye hissa offer mein shamil hai, iski alag keemat nahi hai. Isse zyada ke liye mere paas koi rate nahi hai; {S}."}
            return _esc(b), t[lang] if lang in t else t["en"]
    return None


_WHAT_IS = re.compile(r"what (is|are) (a |an |the )?(membership|fitzone|claude|this|it)|(membership|मेंबरशिप|fitzone|फिट.?ज़ोन|फिट.?जोन|claude|क्लॉड)\W+(\w+\W+){0,2}(kya|क्या)\W+(hota|hai|होता|है|hoti|होती)|"
                      r"(kya|क्या) (hota|होता) (hai|है)", re.I)
_DIFF = re.compile(r"\b(different|difference|better than|versus|vs|compare|comparison|why should i|why switch)\b|अलग क्या|क्या अलग|फर्क|फ़र्क|बेहतर|तुलना|"
                   r"alag kya|kya alag|kya fark|fark kya|behtar|cult\.?fit|gold.?s gym|competitor|switch", re.I)


def _do_answer(mc: MoveContext) -> Move:
    """Answer the question that was asked: from the catalog when it matches, else the honest "I don't have that" line.
    The reply is always in the caller's language (a language with no text for an item is not pasted in English)."""
    t, L, p, lang = mc.text, mc.ledger, mc.product, mc.lang
    if p is None:
        return _unknown(mc)
    if honest.info_request(t) is not None:
        return answer_again(mc, "a request for something that cannot be sent")
    if honest.health_question(t):
        return _say(mc, "answer_health", _by_lang(lambda l: honest.health_variants(p, l)), slot="health",
                    note="Never promise what is safe for a health condition; send them to their doctor; no selling.")
    ip = item_price_answer(p, t, lang)
    if ip is not None:
        slot = "item:" + re.sub(r"\W+", "_", ip[0].lower())[:30]
        full = {l: item_price_answer(p, t, l)[1] for l in LANGS}
        if not L.has_slot(slot) and not L.repeats(full.get(lang) or full["en"]):
            return Move(name="answer_fact", need="answer", text=full, slot=slot, brief=True,
                        note="Answer what they asked about this included item from the facts; no selling question.")
        return answer_again(mc, "already answered this item question")
    topics = honest.find_topics(p, t)
    if topics and (topics[0].beats_faq or not honest.covered_by_faq(p, t)):
        return _unknown(mc, topics[0])
    if re.search(r"kab tak|कब तक|how long|valid", t, re.I) and re.search(r"offer|ऑफर|discount|डिस्काउंट|price|कीमत", t, re.I) and not honest.covered_by_faq(p, t):
        return _say(mc, "answer_fact", _by_lang(lambda l: honest.offer_end_variants(p, l)), slot="offer_end",
                    note="Say how long the offer price lasts from the facts, and honestly that no end date is known.")
    cat = [i for i in build_catalog(p)]
    cands = sorted([i for i in cat if _matches(i, t)], key=lambda i: 0 if i.kind == "faq" else 1)  # an approved FAQ beats a generic fact
    if not cands:
        if _WHAT_IS.search(t):
            cands = [i for i in cat if i.slot == "what"]
        elif _DIFF.search(t):
            cands = []
            fresh_b = [i for i in cat if i.kind == "benefit" and lang in i.text and not L.has_slot(i.slot) and not L.repeats(i.text[lang])]
            if fresh_b:
                lead = {"en": "I don't know their plans well enough to compare honestly. Here, ",
                        "hi": "मुझे उनके प्लान ठीक से नहीं पता, इसलिए ईमानदारी से तुलना नहीं कर सकता। यहाँ, ",
                        "hinglish": "Mujhe unke plan theek se nahi pata, isliye imaandaari se comparison nahi kar sakta. Yahan, "}[lang]
                it = fresh_b[0]
                body = it.text[lang]
                body = body[0].lower() + body[1:] if lang != "hi" else body
                return Move(name="answer_fact", need="answer", text={lang: lead + body}, slot=it.slot, brief=True,
                            note="Do not run the other company down and do not claim to be better; say what this offer includes.")
        elif _WHAT_Q.search(t):
            cands = [i for i in cat if i.kind == "benefit" and lang in i.text] or cat[:1]
        elif _PRICE_Q.search(t):
            cands = [i for i in cat if i.slot == "offer"]
        elif _HOW_Q.search(t):
            cands = [i for i in cat if i.slot == "how"]
    have = [i for i in cands if lang in i.text]
    fresh = [i for i in have if not L.has_slot(i.slot) and not L.repeats(i.text[lang])]
    if fresh:
        it = fresh[0]
        return Move(name="answer_fact", need="answer", text=it.text, slot=it.slot, brief=mc.frustration >= 0.4,
                    note="Answer the question first and fully from the product facts; do not add a selling question.")
    if cands and not have:  # the fact exists but has no text in this language: the LLM paraphrases it (never pasted in English)
        return _unknown(mc)
    if cands:  # they asked about something we already said: the same fact in other words, never a copy, never a cue
        return answer_again(mc, "already answered this")
    m = _mk("no_detail", mc, _rotate(NO_DETAIL, L, "no_detail", p, lang), brief=True,
            note="Say honestly you do not have that detail; the official website has it.")
    return m or _unknown(mc)


def _do_information(mc: MoveContext) -> Move:
    L = mc.ledger
    items = _info_items(mc)
    if not items:
        return _exhausted(mc, "nothing new to tell")
    it = items[0]
    text = dict(it.text)
    sells = False
    interest = (mc.need.flags.get("go_on") or mc.need.flags.get("sale")) and mc.engagement >= 0.4
    if (interest and not mc.hold and L.offers == 0 and not L.refused and not mc.refused and mc.decided == "undecided" and not mc.has_link
            and L.has_slot("offer") and L.has_slot(it.slot) is False and it.slot != "offer"):
        text = {l: it.text[l] + " " + GENTLE_OFFER[l] for l in LANGS if l in it.text}
        sells = True
    return Move(name="inform", need="information", text=text, slot=it.slot, sells=sells,
                note="Give the next useful fact in one or two short sentences; only one gentle offer per call.")


def _simple(name: str, bank, mc: MoveContext, need_name: str, **kw) -> Move:
    m = _mk(name, mc, _rotate(bank, mc.ledger, name, mc.product, mc.lang), brief=True, **kw)
    if m is None and kw.get("end_call"):
        # the caller said goodbye / no and every closing line was used: end quietly rather than repeat
        return Move(name="silent_close", need=mc.need.need, text=_all(""), end_call=True, brief=True)
    return m or _exhausted(mc, f"{name} already said")


def choose_move(mc: MoveContext) -> Move:
    """The policy. Deterministic: same context -> same move."""
    need = mc.need.need
    if need == "closure":
        return _simple("close", CLOSE, mc, need, end_call=True, note="Close politely.")
    if need == "autonomy":
        f = mc.need.flags
        if (f.get("soft_no") and not mc.ledger.refused and mc.ledger.count_move("soft_check") == 0 and not mc.refused
                and not mc.has_link and mc.decided == "undecided"):
            # "not now" is not "never": one gentle check (a better time?), then the next no closes
            return Move(name="soft_check", need=need, text=_by_lang_one(SOFT_CHECK), brief=True,
                        note="They said not now: one gentle check about a better time; no pitch, no price.")
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
        if (honest.is_question(mc.text) and len(mc.text.split()) >= 4 and mc.product is not None
                and not mc.need.flags.get("lang_trouble") and not mc.need.flags.get("garbled") and _resolvable(mc)):
            return _do_answer(mc)  # "कीमत बताओ बेटा, सीधे सीधे" is a question with a known answer, not a reason to ask "which part?"
        return _do_clarity(mc)
    if need == "answer":
        return _do_answer(mc)
    if need == "space":
        return _simple("offer_stop", SPACE, mc, need, note="Be brief and offer to stop or call back.")
    if need == "connection":
        if mc.need.flags.get("sad") or mc.hold:
            first = mc.ledger.count_move("connect") == 0
            return _say(mc, "connect", _by_lang(lambda l: honest.empathy_variants(mc.product, l, first)),
                        note="They are going through a hard time: warm, short, human; no selling, no price, no link; offer another time.")
        return _simple("connect", CONNECT, mc, need, note="Reply like a warm human in one short line; no selling.")
    return _do_information(mc)


def repeat_move(mc: MoveContext, intent: str) -> Move:
    """A deterministic plan was about to repeat an earlier reply word for word (or nearly). Say something that
    helps instead: a garbled/unclear turn gets a different request, a finished call a polite close, anything
    else the honest "that's all I have"."""
    if intent in ("goodbye", "declined"):  # the caller's own goodbye / no: the only repeat that may close the call
        return _simple("close", CLOSE, mc, "closure", end_call=True, note="Close politely; do not repeat.")
    if (honest.is_question(mc.text) or honest.info_request(mc.text)) and mc.product is not None:  # a question, asked again: answer it in other words
        return answer_again(mc, "the reply would have repeated an earlier one")
    if intent in ("not_understood", "clarify", "hello", "off_topic"):
        mc.need.flags["garbled"] = True
        return _do_clarity(mc)
    return _exhausted(mc, "the reply would have repeated an earlier one")
