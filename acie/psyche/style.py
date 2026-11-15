"""Conversational style: how a good human salesperson would say this turn (team/human/PLAN.md, owner-approved).

decide_style(...) -> Style picks ONE row per turn (first match wins) from the emotion row, the 14 state dims, the caller's words
and per-call budgets kept in session.dialog["_style_b"]:

  1 crisis/concern        -> safety (the safety layer owns the words; style does nothing)
  2 angry/frustrated      -> held-back: label + own it, no new pitch, only "what would help?"
  3 confused / garbled    -> brief: one point, no question
  4 sad / personal        -> empathetic: label, no selling, no question
  5 suspicious / trust    -> reassuring: honest specifics, no question
  6 price objection (1st) -> curious: label + one calibrated question
  7 price objection again -> persuasive-calm: value with their goal (callback)
  8 busy / "make it fast" -> brief: one line, one ask max
  9 caller gave a detail  -> warm: mirror 1-3 of their words (<= 1 per 3 turns), ask only if >= 2 turns since the last one
 10 caller asked          -> warm: answer first, no new ask
 11 engaged + calm        -> playful: light self/situation humour (budgets, never elder / turn < 3)
 12 buying / yes          -> confident-brief: approved proof line (product `proof` list only), then close
 13 otherwise             -> warm: backchannel now and then, ask every 2nd turn max

Partial utterances ("So basically,", "I was thinking here about...", trailing "and" / "ki" / "matlab") get a short
backchannel ("Mm-hmm, go on.") instead of a new fact: is_partial(text). Pure text, no timing.

Hard limits (tests/test_style.py): no humour in crisis/angry/frustrated/anxious/sad/confused/overwhelmed/suspicious, elder,
garbled, turn < 3 or money hardship; humour <= 1 per 5 agent turns and <= 2 per call; questions in <= 50% of agent turns,
never 2 discovery questions in a row, <= 3 discovery questions per call, never a known slot; mirror <= 1 per 3 turns and
<= 4 words; always aap forms. Deterministic (zlib.crc32 rotation, no randomness). ACIE_STYLE=0 turns it all off.
"""

from __future__ import annotations

import json
import os
import re
import zlib
from dataclasses import asdict, dataclass, field

from ..agents import gendered
from ..state import social

STYLE_KEY = "_style"
BUDGET_KEY = "_style_b"
_I = re.I


def enabled() -> bool:
    return os.getenv("ACIE_STYLE", "1") != "0"


@dataclass
class Style:
    style: str = "warm"
    move: str = "next_point"
    lead_key: str = ""            # LABEL_* / MIRROR / BACKCHANNEL / HUMOR / PROOF / "" (none)
    ask_ok: bool = True
    humor_ok: bool = False
    note: str = ""
    row: int = 13
    mirror: list[str] = field(default_factory=list)
    callback: str = ""            # "goal:marathon"
    partial: bool = False

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)

    @classmethod
    def from_json(cls, raw: str | None) -> "Style | None":
        try:
            return cls(**json.loads(raw)) if raw else None
        except (ValueError, TypeError):
            return None


# ---------------------------------------------------------------------------------------------- caller-text detectors

_REPEAT_COMPLAINT = re.compile(r"repeat(ing)?|forg[eo]t|already (told|said)|i told you|i just said|same thing|again and again|"
                               r"pehle (hi )?bata|bata (chuka|diya)|baar baar|बार बार|बता (चुका|दिया)|पहले (ही )?बता|फिर से वही|"
                               r"stupid|dumb|idiot|bakwas|बकवास", _I)
PUSHY = re.compile(r"\brude\b|pushy|too fast|so fast|suddenly|did you (even )?ask|didn.?t (even )?ask|jump(ed|ing)? (ahead|straight)|"
                   r"why are you (telling|giving)|bina pooch|बिना पूछ|zabardasti|ज़बरदस्ती|बदतमीज़|badtameez", _I)
APPRECIATE = re.compile(r"^\W*(okay[.,!]?\s*)?(love it|loved it|nice|cool|great|awesome|wow|interesting|sounds (good|nice|great)|achha laga|"
                        r"badhiya|बढ़िया|बहुत (बढ़िया|अच्छा)|mast|मस्त|okay|ok|hmm+|achha|अच्छा|theek hai|ठीक है)\W*$", _I)
_BUSY = re.compile(r"make it (fast|quick)|quickly|be quick|in short|short mein|jaldi|जल्दी|टाइम नहीं|time nahi|no time|busy", _I)
_REPEAT_ONLY = re.compile(r"repeat(ing)?|forg[eo]t|already (told|said)|i told you|i just said|same thing|again and again|"
                          r"pehle (hi )?bata|bata (chuka|diya)|baar baar|बार बार|बता (चुका|दिया)|पहले (ही )?बता|फिर से वही|ek hi (line|baat|dialogue)", _I)
_PRICE_OBJ = re.compile(r"expensive|too much|costly|discount|lower|cheaper|negotiate|mehnga|mahenga|महंगा|sasta|सस्ता|kam kar|कम कर|"
                        r"bahut zyada|बहुत ज़्यादा|afford|budget", _I)
_HARDSHIP = re.compile(r"can.?t afford|no money|lost (my )?job|paise nahi|पैसे नहीं|naukri (chali|gayi)|नौकरी (चली|गई)|tight on money|"
                       r"loan|karz|कर्ज़|कर्ज", _I)
_YES = re.compile(r"\b(yes|yeah|sure|ok(ay)?|done|deal|let'?s do it|send (me )?(the )?link|i('ll| will) (take|join|buy))\b|"
                  r"haan|हाँ|हां|ठीक है|theek hai|chalo|चलो|bhej do|भेज दो|le leta|ले लेता", _I)
_PARTIAL_END = re.compile(r"(,|\.\.\.|…|-|—|(?:^|\s)(and|and also|but|so|because|like|or|then|about|with|for|the|a|to|of|um|uh|"
                          r"ki|ke|ka|aur|matlab|toh|to|lekin|par|ya|jo|ki agar|और|कि|मतलब|तो|लेकिन|पर|या|जो))\s*$", _I)
_PARTIAL_START = re.compile(r"^(so basically|basically|actually|i was thinking|i think|i mean|what i mean|the thing is|matlab|"
                            r"मतलब|असल में|asal mein|dekho|देखिए|देखो)\b", _I)
_TOK = re.compile(r"[\wऀ-ॿ]+")
_POSITIVE = re.compile(r"\b(haha|lol|nice|cool|great|awesome|perfect|sounds good|love it|interesting)\b|badhiya|बढ़िया|mast|मस्त|"
                       r"accha laga|अच्छा लगा|बहुत अच्छा|bahut achha", _I)
_CONCERN = re.compile(r"not comfortable|worried|scared|problem|issue|nahi|नहीं|\bno\b|don.?t|can.?t|won.?t|dikkat|दिक्कत|tension|टेंशन", _I)
_Q = re.compile(r"\?\s*$|^(what|why|how|which|who|where|when|is|are|can|do|does|will|kya|kaise|kitn|kab|kaun|क्या|कैसे|कितन|कब|कौन|क्यों)\b", _I)


def is_partial(text: str) -> bool:
    """The caller's turn is clearly unfinished: a trailing comma / ellipsis / connective ("and", "ki", "matlab") or a lone
    discourse opener ("So basically,"). Never for a question, a number (a price counter is never swallowed) or a farewell."""
    t = (text or "").strip()
    if not t or "?" in t or re.search(r"\d", t):
        return False
    try:
        from ..context import is_question
        if is_question(t) and not t.endswith(("...", "…", ",")):
            return False
    except Exception:
        pass
    toks = _TOK.findall(t.lower())
    if not toks or len(toks) > 14:
        return False
    if re.search(r"\b(bye|thank|thanks|no|nahi|नहीं|stop|cut)\b", t, _I) or _YES.search(t):
        return False  # "Sure, I'll just..." is consent (the negotiation closes on it), never swallowed by a backchannel
    if t.endswith(("...", "…")) or (t.endswith(",") and len(toks) <= 6):
        return True
    if _PARTIAL_END.search(t) and len(toks) >= 2:
        return True
    return bool(_PARTIAL_START.match(t)) and len(toks) <= 3


# ---------------------------------------------------------------------------------------------- lead banks (aap forms only)

BANKS: dict[str, dict[str, list[str]]] = {
    "BACKCHANNEL": {"en": ["Right.", "Mm-hmm.", "Okay."], "hi": ["जी।", "अच्छा।", "हम्म।"], "hinglish": ["Ji.", "Achha.", "Hmm."]},
    "GO_ON": {"en": ["Mm-hmm, go on.", "Sure, take your time.", "I'm listening, go ahead."],
              "hi": ["हाँ जी, बोलिए।", "जी, आराम से बताइए।", "जी, सुन रहा हूँ, बोलिए।"],
              "hinglish": ["Haan ji, boliye.", "Ji, aaram se bataiye.", "Ji, sun raha hoon, boliye."]},
    "MIRROR": {"en": ["{m}? Nice.", "{m}, got it.", "Oh, {m}. Okay."], "hi": ["{m}, बढ़िया।", "अच्छा, {m}।", "{m}, समझ गया।"],
               "hinglish": ["{m}, badhiya.", "Achha, {m}.", "{m}, samajh gaya."]},
    "LABEL_UPSET": {"en": ["You're right, that's on me.", "Fair, I repeated myself there.", "You're right, I asked that already."],
                    "hi": ["आप सही हैं, ग़लती मेरी है।", "सही कहा आपने, मैंने दोबारा कह दिया।", "आप सही हैं, मैंने दोबारा पूछ लिया।"],
                    "hinglish": ["Aap sahi hain, galti meri hai.", "Sahi kaha aapne, maine dobara keh diya.", "Sahi kaha aapne, maine dobara pooch liya."]},
    "LABEL_PRICE": {"en": ["Sounds like the price is the sticking point.", "Fair, it's real money."],
                    "hi": ["लगता है कीमत ही अटक रही है।", "ठीक बात है, पैसे का सवाल है।"],
                    "hinglish": ["Lagta hai price hi atak raha hai.", "Theek baat hai, paise ka sawaal hai."]},
    "LABEL_SAD": {"en": ["That sounds hard.", "I'm sorry, that's a lot."], "hi": ["ये सुनकर अच्छा नहीं लगा।", "ये मुश्किल लग रहा है।"],
                  "hinglish": ["Ye sunkar achha nahi laga.", "Ye mushkil lag raha hai."]},
    "LABEL_DOUBT": {"en": ["Fair to doubt it.", "Good that you're careful."], "hi": ["शक होना ठीक है।", "सावधान रहना अच्छी बात है।"],
                    "hinglish": ["Shak hona theek hai.", "Saavdhaan rehna achhi baat hai."]},
    "HUMOR": {"en": ["I'd sign up myself, but I'm just a voice on the phone.", "No treadmill needed for this part, I promise."],
              "hi": ["मैं ख़ुद जॉइन कर लेता, पर मैं तो बस फ़ोन की आवाज़ हूँ।", "इस बात के लिए पसीना नहीं बहाना पड़ेगा।"],
              "hinglish": ["Main khud join kar leta, par main toh bas phone ki awaaz hoon.", "Is baat ke liye pasina nahi bahana padega."]},
    "HANDBACK": {"en": ["You're right, I jumped ahead, sorry. Would you like to hear what's included first, or is now a bad time?",
                        "Fair point, I rushed that, sorry. Shall I tell you what you get first, or would you rather stop here?"],
                 "hi": ["आप सही हैं, मैं जल्दी आगे बढ़ गया, माफ़ कीजिए। पहले सुनना चाहेंगे कि इसमें क्या मिलता है, या अभी सही समय नहीं है?",
                        "सही कहा आपने, मैंने जल्दबाज़ी की, माफ़ कीजिए। पहले बताऊँ कि क्या मिलता है, या यहीं रुक जाएँ?"],
                 "hinglish": ["Aap sahi hain, main jaldi aage badh gaya, maaf kijiye. Pehle sunna chahenge isme kya milta hai, ya abhi sahi time nahi hai?",
                              "Sahi kaha aapne, maine jaldbaazi ki, maaf kijiye. Pehle bataun kya milta hai, ya yahin ruk jaayein?"]},
    "PERMISSION": {"en": ["Glad you like it! Want to hear this week's offer?", "Good to hear! Shall I tell you about the offer?"],
                   "hi": ["सुनकर अच्छा लगा! इस हफ़्ते का ऑफर बताऊँ?", "बढ़िया! ऑफर के बारे में बताऊँ?"],
                   "hinglish": ["Sunkar achha laga! Is hafte ka offer bataun?", "Badhiya! Offer ke baare mein bataun?"]},
    # social-signal leads (acie/state/social.py): sarcasm gets a dry honest nod (never taken literally, never mocked back),
    # a joke gets a short laugh back, rudeness gets a calm, plain "straight answer" (never matched, never an apology loop)
    "SARCASM_ACK": {"en": ["Fair, nobody hopes for a sales call.", "Ha, fair. I'll keep it short.", "Understood, I'll be quick and straight."],
                    "hi": ["सही बात है, सेल्स कॉल कोई नहीं चाहता।", "ठीक है, छोटा रखता हूँ।", "समझ गया, सीधी और छोटी बात करता हूँ।"],
                    "hinglish": ["Sahi baat hai, sales call koi nahi chahta.", "Theek hai, chhota rakhta hoon.", "Samajh gaya, seedhi aur chhoti baat karta hoon."]},
    "LAUGH": {"en": ["Ha, good one!", "Haha, fair enough!", "Ha, I like that."],
              "hi": ["हाहा, ये बढ़िया था!", "हाहा, सही कहा!", "हाहा, मज़ेदार!"],
              "hinglish": ["Haha, ye badhiya tha!", "Haha, sahi kaha!", "Haha, mazedaar!"]},
    "CALM": {"en": ["Sure, straight answer.", "Okay, plainly."], "hi": ["जी, सीधी बात।", "ठीक है, सीधे बताता हूँ।"],
             "hinglish": ["Ji, seedhi baat.", "Theek hai, seedha batata hoon."]},
    "SOFT": {"en": ["Sure.", "Of course.", "Fair enough."], "hi": ["जी।", "बिल्कुल।", "ज़रूर।"], "hinglish": ["Ji.", "Bilkul.", "Zaroor."]},
    "ASK_HELP": {"en": ["What would help most right now?"], "hi": ["अभी आपके लिए सबसे काम की बात क्या होगी?"],
                 "hinglish": ["Abhi aapke liye sabse kaam ki baat kya hogi?"]},
}
_GENERIC_HUMOR_EN = ["I'd sign up myself, but I'm just a voice on the phone."]  # non-gym products: self only
_HUMOR_BANNED_ROWS = {"crisis", "angry", "frustrated", "anxious", "sad", "confused", "overwhelmed", "suspicious"}
_NEVER_IN_HUMOR = re.compile(r"\b(fat|weight|body|old|age|caste|region|gender|girl|boy|money|cheap|poor)\b|मोटा|बूढ़|जाति|पैसे", _I)


def pick(key: str, lang: str, n: int, gender: str = "", mirror: str = "") -> str:
    lines = BANKS.get(key, {}).get(lang) or BANKS.get(key, {}).get("en") or []
    if not lines:
        return ""
    s = lines[n % len(lines)].replace("{m}", mirror)
    if mirror and s.startswith(mirror) and lang != "hi":
        s = s[0].upper() + s[1:]
    if gender == "female":  # the bank's own first-person verbs, then the validator's general rewrite
        for m, f in _FEM:
            s = s.replace(m, f)
    return gendered(s, gender) if gender else s


_FEM = (("समझ गया, सीधी", "समझ गई, सीधी"), ("Samajh gaya, seedhi", "Samajh gayi, seedhi"), ("छोटा रखता", "छोटा रखती"), ("chhota rakhta", "chhota rakhti"),
        ("करता हूँ।", "करती हूँ।"), ("karta hoon.", "karti hoon."), ("बताता हूँ", "बताती हूँ"), ("batata hoon", "batati hoon"),
        ("बढ़ गया", "बढ़ गई"), ("badh gaya", "badh gayi"), ("kar leta", "kar leti"), ("कर लेता", "कर लेती"), ("samajh gaya", "samajh gayi"), ("समझ गया", "समझ गई"),
        ("sun raha", "sun rahi"), ("सुन रहा", "सुन रही"))


# ---------------------------------------------------------------------------------------------- per-call budgets

def budgets(dialog: dict) -> dict:
    try:
        b = json.loads(dialog.get(BUDGET_KEY) or "{}")
    except (ValueError, TypeError):
        b = {}
    b.setdefault("turns", 0)          # agent turns so far
    b.setdefault("q_turns", [])       # agent turn numbers that ended in / held a question
    b.setdefault("disc_turns", [])    # agent turn numbers with a discovery question
    b.setdefault("humor_turns", [])
    b.setdefault("mirror_turns", [])
    b.setdefault("prices", {})        # price string -> last agent turn number it was said
    b.setdefault("leads", 0)          # rotation counter
    b.setdefault("upset_turns", [])   # agent turn numbers answered in the held-back row (no humour for the rest of the call)
    b.setdefault("label_turns", [])
    return b


def save_budgets(dialog: dict, b: dict) -> None:
    dialog[BUDGET_KEY] = json.dumps(b, ensure_ascii=False, sort_keys=True)


_DISC = re.compile(r"\b(what|which)\b.{0,40}\b(use|using|goal|work|study|need|looking for|spend|focus)\b|\b(goal|use it for|mainly use)\b|"
                   r"starting out|trained before|how many days|morning or evening|aesthetic|किस (काम|चीज़|चीज|तरह)|goal|लक्ष्य|इस्तेमाल करेंगे|"
                   r"kis (kaam|cheez)|use karenge|शुरुआत कर रहे|ट्रेनिंग की है|shuru kar rahe|training ki hai|कितने दिन|kitne din", _I)
PRICE_RX = re.compile(r"(₹|rs\.?\s?|\$)\s?\d[\d,]*", _I)


def is_discovery(sentence: str) -> bool:
    return sentence.rstrip().endswith("?") and bool(_DISC.search(sentence))


def note_reply(dialog: dict, text: str) -> None:
    """After the turn: count what the agent actually said (questions, discovery, prices, humour, mirror)."""
    if not enabled():
        return
    b = budgets(dialog)
    b["turns"] += 1
    n = b["turns"]
    sents = [s for s in re.split(r"(?<=[.!?।])\s+", (text or "").strip()) if s]
    if any(s.rstrip().endswith("?") for s in sents):
        b["q_turns"].append(n)
    if any(is_discovery(s) for s in sents):
        b["disc_turns"].append(n)
    for p in set(PRICE_RX.findall(text or "")) | set(m.group(0) for m in PRICE_RX.finditer(text or "")):
        b["prices"][p.replace(" ", "")] = n
    st = Style.from_json(dialog.get(STYLE_KEY))
    if st and dialog.get("_style_used"):
        used = dialog.pop("_style_used")
        if used.startswith("LABEL"):
            b["label_turns"].append(n)
        if used == "HUMOR":
            b["humor_turns"].append(n)
        elif used == "MIRROR":
            b["mirror_turns"].append(n)
        b["leads"] += 1
    if st and st.row == 2:
        b["upset_turns"].append(n)
    for k in ("q_turns", "disc_turns", "humor_turns", "mirror_turns", "upset_turns", "label_turns"):
        b[k] = b[k][-30:]
    save_budgets(dialog, b)


def ask_budget_ok(b: dict) -> bool:
    """Questions in <= 50% of agent turns: never two asking turns in a row."""
    n = b["turns"]
    return not (b["q_turns"] and b["q_turns"][-1] == n)


def discovery_ok(dialog: dict) -> bool:
    """May this turn carry a discovery question? Never 2 in a row, <= 3 per call, and only on a turn that may ask at all."""
    if not enabled():
        return True
    b = budgets(dialog)
    st = Style.from_json(dialog.get(STYLE_KEY))
    if st is not None and not st.ask_ok:
        return False
    if len(b["disc_turns"]) >= 3:
        return False
    if b["disc_turns"] and b["disc_turns"][-1] >= b["turns"] - 1:  # the last or the one before was a discovery question
        return False
    return ask_budget_ok(b)


def humor_budget_ok(b: dict) -> bool:
    """<= 2 per call, <= 1 per 5 agent turns, never after the caller was upset in this call."""
    h = b["humor_turns"]
    return len(h) < 2 and not (h and b["turns"] - h[-1] < 5) and not b.get("upset_turns")


def label_budget_ok(b: dict) -> bool:
    lt = b.get("label_turns") or []
    return not (lt and b["turns"] - lt[-1] < 3)


def mirror_budget_ok(b: dict) -> bool:
    m = b["mirror_turns"]
    return not (m and b["turns"] - m[-1] < 3)


# ---------------------------------------------------------------------------------------------- the decision

def _g(state, k: str, d: float = 0.0) -> float:
    if isinstance(state, dict):
        return float(state.get(k, d))
    return float(getattr(state, k, d) or d) if getattr(state, k, None) is not None else d


def mirror_words(text: str, hits: list[str]) -> list[str]:
    """1-3 of the caller's own words (a matched answer such as 'marathon' / 'powerlifting'), never more than 4."""
    toks = _TOK.findall(text or "")
    low = [x.lower() for x in toks]
    for h in hits:
        w = _TOK.findall(h.lower())
        if not 1 <= len(w) <= 3:
            continue
        for i in range(len(low)):  # the caller's own spelling ("power lift" matched -> "power lifting")
            if all(i + j < len(low) and low[i + j].startswith(w[j]) for j in range(len(w))):
                return toks[i:i + len(w)][:3]
        return w[:3]
    return []


def decide_style(*, row: str, state, text: str, intent: str, turn_index: int, dialog: dict, facts=None, elder: bool = False,
                 garbled: bool = False, new_detail: list[str] | None = None) -> Style:
    """One row per turn (module table). Pure: reads the budgets, never changes them (note_reply does)."""
    b = budgets(dialog)
    t = (text or "").strip()
    q_ok = ask_budget_ok(b)
    try:
        from ..context import is_question
        asked = bool(_Q.search(t)) or is_question(t)
    except Exception:
        asked = bool(_Q.search(t))
    objs = list(getattr(facts, "objections", []) or [])
    hardship = bool(_HARDSHIP.search(t))
    goal = ""
    if facts is not None:
        for k in ("lvl:goal", "goal", "lvl:subgoal"):
            v = (getattr(facts, "caller", {}) or {}).get(k)
            if v:
                goal = v[-1]
                break
    callback = f"goal:{goal}" if goal else ""
    humour_banned = row in _HUMOR_BANNED_ROWS or elder or garbled or turn_index < 3 or hardship
    base = dict(humor_ok=False, callback=callback)
    soc = social.load(dialog)
    sarcasm_now = max(social.turn_value(dialog, "sarcasm"), soc["sarcasm"] if soc["sarcasm"] >= 0.6 else 0.0)
    hostile_now = social.turn_value(dialog, "hostility")
    joking_now = social.turn_value(dialog, "humour")

    if row == "crisis":
        return Style("safety", "defer_to_safety", "", False, row=1, note="the safety layer speaks", **base)
    if is_partial(t) and turn_index > 0:
        return Style("listening", "backchannel", "GO_ON", False, row=0, partial=True, note="caller is mid-sentence: let them finish", **base)
    if hostile_now >= 0.6 and not _REPEAT_ONLY.search(t) and not PUSHY.search(t):
        # rude AT us (not about our repetition): calm and plain, never matched, never a joke, no apology loop; answer what they
        # asked in one short line, or (nothing asked) offer to stop
        return Style("calm", "calm_answer_offer_stop", "CALM" if asked else "", False, row=2,
                     note="the caller is being rude: stay calm and polite, do not match their tone, no joke, no repeated apology; "
                          + ("answer exactly what they asked in one short line" if asked else "offer to stop the call or answer one question"), **base)
    if row in ("angry", "frustrated") or _g(state, "frustration") >= 0.4 or _REPEAT_COMPLAINT.search(t) or (PUSHY.search(t) and not _PRICE_OBJ.search(t)) or re.search(r"\brude\b|did you (even )?ask", t, _I):
        lab = "LABEL_UPSET" if label_budget_ok(b) and (_REPEAT_COMPLAINT.search(t) or not b.get("upset_turns")) else ""
        if PUSHY.search(t) and not (_PRICE_OBJ.search(t) and not re.search(r"\brude\b|did you (even )?ask|didn.?t (even )?ask", t, _I)):
            return Style("held-back", "own_and_hand_back", "HANDBACK", True, row=2,
                         note="own it (you jumped ahead) and hand control back with ONE question: hear what's included first, or stop", **base)
        return Style("held-back", "label_own_mistake", lab, True, row=2,
                     note="own the mistake in a few words, then hand control back with ONE short question (what would help, or shall I stop); no new pitch", **base)
    if row == "confused" or garbled or re.fullmatch(r"\s*(what|huh|kya|क्या|sorry)\s*\??\s*", t, _I):
        return Style("brief", "simplify", "", False, row=3, note="one simple point, no question", **base)
    if sarcasm_now >= 0.6 and row not in ("sad",):
        # "oh wow, another sales call, my lucky day": never literal ("glad you're excited!"), never sarcasm back; a dry honest nod,
        # then exactly what they asked, short; no question unless it is the one they asked for
        return Style("dry-honest", "acknowledge_sarcasm", "SARCASM_ACK", False, row=5,
                     note="the caller is being sarcastic: do NOT take it literally (no 'glad you like it', no 'haha yes'), no sarcasm back; "
                          "one dry honest acknowledgement, then exactly what they asked, in one or two short sentences", **base)
    if row == "sad" or intent == "personal":
        return Style("empathetic", "label_connect", "LABEL_SAD", False, row=4, note="label the feeling, no selling, no question", **base)
    if row == "suspicious" or "trust" in objs and re.search(r"scam|fake|fraud|trust|bharosa|भरोसा", t, _I) \
            or (soc["scepticism"] >= 0.6 and re.search(r"scam|fake|fraud|genuine|legit|hidden|guarantee|bharosa|भरोसा|फ्रॉड|फ़्रॉड|धोखा|otp|ओटीपी", t, _I)):
        return Style("reassuring", "honest_specifics", "LABEL_DOUBT", False, row=5,
                     note="honest specifics + risk reversal (nothing charged on this call, cancel terms only if in the facts)", **base)
    if _PRICE_OBJ.search(t) or intent in ("negotiation", "price_objection"):
        repeated = int(dialog.get("_style_price_obj", "0") or 0) >= 1
        if repeated:
            return Style("persuasive-calm", "value_reframe", "", q_ok, row=7,
                         note="tie the value to what they told you (callback), no new number from you", **base)
        return Style("curious", "label_calibrated_q", "LABEL_PRICE" if label_budget_ok(b) else "", True, row=6,
                     note="label the price concern, then one calibrated what/how question", **base)
    if row == "busy" or _BUSY.search(t) or soc["impatience"] >= 0.5:
        return Style("brief", "one_line_value", "", q_ok, row=8, note="one line of value, then one ask or offer a callback", **base)
    if joking_now >= 0.6 and hostile_now < 0.3 and sarcasm_now < 0.6 and _g(state, "frustration") < 0.25 and not elder and not garbled \
            and not hardship and row not in _HUMOR_BANNED_ROWS:
        # the caller joked: laugh back briefly (reciprocal, about the moment, never about them), then answer / carry on plainly.
        # Our own jokes (HUMOR) still wait for turn 3 and the humour budget.
        return Style("playful", "laugh_then_answer", "LAUGH" if humor_budget_ok(b) else "", False if asked else q_ok,
                     not humour_banned and humor_budget_ok(b), row=11,
                     note="the caller is joking: you may laugh back in a few words or be lightly playful about the situation or yourself "
                          "(never the caller, body, age or money), then " + ("answer what they asked plainly" if asked else "carry on plainly"),
                     callback=callback)
    if new_detail and not asked:
        mw = mirror_words(t, new_detail) if mirror_budget_ok(b) else []
        two_since = not b["q_turns"] or b["turns"] - b["q_turns"][-1] >= 2
        return Style("warm", "mirror_tailor", "MIRROR" if mw else "", two_since, row=9, mirror=mw,
                     note="reuse 1-3 of their words once, then one fact tailored to it", **base)
    if asked:
        if soc["interest"] >= 0.65:
            return Style("warm", "answer_then_step", "", False, row=10,
                         note="they are genuinely interested: answer exactly, then offer the next concrete step (how to join / the offer); "
                              "no extra features", **base)
        return Style("warm", "answer_first", "", False, row=10, note="answer first (because / for example), no new pitch, no question", **base)
    if _YES.search(t) and (_g(state, "buying_intent", 0.0) >= 0.7 or intent in ("confirm_close", "accept", "yes", "buy")):
        return Style("confident-brief", "proof_close", "PROOF", True, row=12, note="short, confident, close", **base)
    if not humour_banned and _g(state, "engagement", 0.5) >= 0.6 and _g(state, "frustration") < 0.25 and humor_budget_ok(b) \
            and _POSITIVE.search(t) and not _CONCERN.search(t):
        return Style("playful", "light_humour", "HUMOR", q_ok, True, row=11, note="one light line about the situation or yourself; never the person, body or money",
                     callback=callback)
    lead = "BACKCHANNEL" if b["turns"] % 2 == 1 else ""
    return Style("warm", "next_point", lead, q_ok, row=13, note="the next unsaid point; ask only every 2nd turn", **base)


def humour_allowed(style: Style, row: str, elder: bool, turn_index: int, text: str, dialog: dict) -> bool:
    """The hard limit, checked again where the lead is applied."""
    return (style.humor_ok and row not in _HUMOR_BANNED_ROWS and not elder and turn_index >= 3 and not _HARDSHIP.search(text or "")
            and not _CONCERN.search(text or "") and humor_budget_ok(budgets(dialog)))


def crc_n(*parts: str) -> int:
    return zlib.crc32("|".join(parts).encode("utf-8"))


# ---------------------------------------------------------------------------------------------- the LLM card

def style_card(st: Style) -> str:
    if st is None or st.style == "safety":
        return ""
    bits = [f"style={st.style}", f"move={st.move}", f"ask={'ok' if st.ask_ok else 'NO question this turn'}"]
    if st.note:
        bits.append(f"note={st.note}")
    if st.mirror:
        bits.append("mirror_words=" + json.dumps(st.mirror, ensure_ascii=False))
    if st.callback:
        bits.append("callback=" + st.callback)
    return ("THIS TURN: " + ", ".join(bits) + "\n"
            "STYLE CARD: talk like a real salesperson on the phone: short, listen more than you talk, answer what they said first, "
            "no stacked questions. Humour only if style=playful (about the situation or yourself, never the caller, body, age or money). "
            "Never invent stories, numbers or other customers. Never call the caller bhai, boss, yaar, dude, bro or beta, even if they "
            "talk that way; use aap / their name with ji.\n\n")
