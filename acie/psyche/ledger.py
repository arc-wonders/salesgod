"""SaidLedger: everything the agent has already said in this call.

Stored as JSON in `session.dialog["_said"]` (dialog is a str->str dict, so it survives the session
stores). It remembers normalized sentences, intents, fact slots, asks and phrasing rotations, so the
move selector can pick the next *unsaid* thing instead of repeating. Pure Python, deterministic.
"""

from __future__ import annotations

import difflib
import json
import re

KEY = "_said"
SIMILAR = 0.6  # SequenceMatcher ratio above which two sentences count as the same thing
_PUNCT = re.compile(r"[\s\.,!?;:\"'()\[\]।\-]+")
_SPLIT = re.compile(r"(?<=[.!?।])\s+")


def norm(text: str) -> str:
    return _PUNCT.sub(" ", text.lower()).strip()


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SPLIT.split(text.strip()) if s.strip()]


def ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()


class SaidLedger:
    def __init__(self, data: dict | None = None):
        d = data or {}
        self.sentences: list[str] = list(d.get("sentences", []))  # normalized, in order
        self.replies: list[str] = list(d.get("replies", []))      # whole agent replies, normalized
        self.intents: dict[str, int] = dict(d.get("intents", {}))
        self.slots: list[str] = list(d.get("slots", []))
        self.asks: list[str] = list(d.get("asks", []))
        self.moves: dict[str, int] = dict(d.get("moves", {}))
        self.offers: int = int(d.get("offers", 0))
        self.refused: bool = bool(d.get("refused", False))  # the caller said no / later / stop: never offer again

    # -- persistence
    @classmethod
    def load(cls, dialog: dict) -> "SaidLedger":
        raw = dialog.get(KEY)
        try:
            return cls(json.loads(raw)) if raw else cls()
        except (ValueError, TypeError):
            return cls()

    def save(self, dialog: dict) -> None:
        dialog[KEY] = json.dumps({"sentences": self.sentences[-80:], "replies": self.replies[-30:], "intents": self.intents, "slots": self.slots,
                                  "asks": self.asks[-30:], "moves": self.moves, "offers": self.offers,
                                  "refused": self.refused}, ensure_ascii=False, sort_keys=True)

    # -- queries
    def has_slot(self, slot: str) -> bool:
        return slot in self.slots

    def similar_to(self, text: str, thr: float = SIMILAR) -> str | None:
        """The earlier sentence this one repeats (ratio > thr), else None. Whole text and each sentence are checked."""
        n = norm(text)
        if not n:
            return None
        for s in self.sentences:
            if difflib.SequenceMatcher(None, n, s).ratio() > thr:
                return s
        return None

    def repeats(self, text: str, thr: float = SIMILAR) -> bool:
        parts = sentences(text) or [text]
        return any(self.similar_to(p, thr) for p in parts if len(norm(p).split()) >= 3)

    def reply_repeats(self, text: str, thr: float = SIMILAR) -> bool:
        """The whole reply is (almost) a copy of an earlier whole reply: the headline repetition metric."""
        n = norm(text)
        return bool(n) and any(difflib.SequenceMatcher(None, n, r).ratio() > thr for r in self.replies)

    def count_move(self, name: str) -> int:
        return self.moves.get(name, 0)

    # -- updates
    def add_text(self, text: str) -> None:
        if norm(text):
            self.replies.append(norm(text))
        for s in sentences(text):
            n = norm(s)
            if n and n not in self.sentences:
                self.sentences.append(n)
        for s in sentences(text):
            if s.endswith("?"):
                self.asks.append(norm(s))

    def add_intent(self, intent: str) -> None:
        self.intents[intent] = self.intents.get(intent, 0) + 1

    def add_slot(self, slot: str | None) -> None:
        if slot and slot not in self.slots:
            self.slots.append(slot)

    def add_move(self, name: str) -> None:
        self.moves[name] = self.moves.get(name, 0) + 1


# ====================================================================================================================
# CustomerFacts: what the CALLER told us, and the topics we have asked / said. "No repeating context" (owner, 15 Nov):
# never re-ask what was answered, never parrot the caller's own facts back, never say a benefit twice, never say a
# sentence twice. Stored as JSON in session.dialog["_cf"] next to the SaidLedger. Deterministic: regexes only,
# zlib.crc32 (never hash()), sorted iteration, fixed tie-breaks.
# ====================================================================================================================

import zlib  # noqa: E402

FACTS_KEY = "_cf"
_I = re.I

# topic -> regex on the CALLER's words that answers it
_ANSWERS = {
    "goal": re.compile(r"muscle|bulk|मसल|बॉडी|body ?build|physique|aesthetic|strength|power ?lift|stamina|weight ?(loss|lose|gain)|lose weight|"
                       r"fat ?loss|वज़न|वजन|stay fit|fitness|fit rehna|fit रहना|फिट रहना|toning|coding|code|कोडिंग|writing|study|padhai|पढ़ाई|"
                       r"resume|assignment|report|research|business|dukaan|दुकान|content|marketing|design", _I),
    "schedule": re.compile(r"\b(\d+|one|two|three|four|five|six|seven|ek|do|teen|char|chaar|paanch|panch)\s*(to|-|se)?\s*(\d+|four|five|six|char|paanch)?\s*"
                           r"(days?|din|times|baar|hours?|ghante)\b|\b(morning|evening|night|weekend|weekends|subah|shaam|raat)\b|"
                           r"सुबह|शाम|रात|हफ़्ते में|हफ्ते में|दिन|\bevery day\b|daily|रोज़|roz\b", _I),
    "budget": re.compile(r"\bbudget\b|afford|\bsasta\b|सस्ता|महंगा|mehnga|mahenga|expensive|too much|bahut zyada|बहुत ज़्यादा|\bcheap\b|"
                         r"(\d+|k|thousand|hazaar)\s*(rs|rupees|₹|rupaye)|₹\s*\d+", _I),
    "experience": re.compile(r"beginner|first time|pehli baar|पहली बार|new to|never (done|tried|used)|already (go|train|use|using|have)|"
                             r"\d+\s*(years?|saal|साल|months?|mahine)\b|experience|experienced|used it before|pehle se|पहले से|कॉलेज|college|"
                             r"student|स्टूडेंट|छात्र|working|job karta|नौकरी", _I),
    "name": re.compile(r"(my name is|this is|i am|i'm|mera naam|मेरा नाम|main hoon|मैं)\s+([A-Za-zऀ-ॿ]{3,})", _I),
}
_NOT_NAME = {"interested", "busy", "driving", "student", "working", "fine", "good", "sorry", "ready", "looking", "just", "not", "only",
             "calling", "thinking", "trying", "already", "going", "here", "in", "at", "from", "hoon", "hun", "kar", "karta", "bol", "bahut",
             "तैयार", "ठीक", "बिज़ी", "busy", "कर", "बोल", "सुन", "पढ़"}

# topic -> regex on OUR question (what we asked)
_ASKS = {
    "goal": re.compile(r"\b(what|which)\b.{0,40}\b(use|using|goal|work|study|need|looking for|spend|focus)\b|\b(goal|use it for|mainly use)\b|"
                       r"किस (काम|चीज़|चीज)|आपका (main )?(goal|लक्ष्य)|इस्तेमाल करेंगे|\bkis (kaam|cheez)\b|aapka (main )?goal|use karenge|"
                       r"काम या पढ़ाई|kaam ya padhai|लक्ष्य|losing weight, building muscle|fit rehna", _I),
    "schedule": re.compile(r"how many days|which days|what time|kitne din|kab (aa|jaa|aana)|kis time|कितने दिन|किस समय|कब आ", _I),
    "budget": re.compile(r"your budget|what.s your budget|aapka budget|आपका बजट|kitna budget", _I),
    "experience": re.compile(r"beginner|first time|experience|pehle kabhi|पहले कभी|new to", _I),
    "name": re.compile(r"your name|aapka naam|आपका नाम|may i know who", _I),
}

_OBJ = {
    "price": re.compile(r"expensive|too much|costly|mehnga|mahenga|महंगा|bahut zyada|बहुत ज़्यादा|afford", _I),
    "think": re.compile(r"think about|sochna|सोचना|सोचता|soch(ta|ti)? hoon|let me think|get back", _I),
    "trust": re.compile(r"scam|fake|fraud|trust|bharosa|भरोसा|proof|सबूत", _I),
    "time": re.compile(r"no time|busy|time nahi|टाइम नहीं|अभी नहीं|not now|abhi nahi", _I),
    "competitor": re.compile(r"chatgpt|gemini|cult|other gym|doosra gym|free (plan|version)|already (have|use)", _I),
}
_DECISION = {
    "no": re.compile(r"not interested|don.?t want|do not want|nahi chahiye|नहीं चाहिए|no thanks|no,? thank|stop calling|don.?t call|"
                     r"(रुचि|interest|ज़रूरत|zaroorat|need) (नहीं|nahi|no)|interested nahi|नहीं (लेना|लूँगा|लूंगा|करना)|nahi (lena|lunga|karna)|no need|cancel|मत (करो|करना|भेजो)|don.?t send|do not send|nahi bhej|mat bhej|नहीं भेज", _I),
    "later": re.compile(r"call (me )?(later|tomorrow)|baad mein|बाद में|kal call|कल कॉल|later", _I),
    "yes": re.compile(r"\b(i('ll| will) (take|buy|join)|send (me )?(the )?link|link bhej|लिंक भेज|ok(ay)? (send|let.?s)|let.?s do it|karwa do|chalo le leta)\b", _I),
}
_STOPW = {"the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "i", "me", "my", "you", "your", "is", "am", "are", "it", "that",
          "this", "with", "want", "wanna", "like", "mujhe", "main", "mein", "hai", "hoon", "ko", "ka", "ki", "ke", "se", "karna", "karta",
          "chahta", "chahiye", "और", "मैं", "मुझे", "है", "हूँ", "हूं", "को", "का", "की", "के", "से", "में", "करना", "करता", "चाहता", "चाहिए",
          "yes", "ok", "okay", "haan", "ji", "so", "what", "will", "get", "kya", "about", "focus", "building", "build", "body"}
# our sentence is an echo of what the caller said: "it will help with muscle", "आप कॉलेज में पढ़ते हैं", "as a student"
_ECHO_FRAME = re.compile(r"help(s|ing)? (you )?(with|in|build|to build|achieve|reach)|you (want|said|mentioned|told|are|re|need)|as a |"
                         r"since you|because you|your goal|for your|aapka goal|आपका (goal|लक्ष्य)|आप (कॉलेज|स्टूडेंट|छात्र|.{0,25}(चाहते|करते|पढ़ते|सीखते))|"
                         r"aap (college|student|.{0,25}(chahte|karte|padhte|seekhte))|aapko .{0,25}(mein|ke liye) .{0,20}(madad|help)|"
                         r"आपको .{0,25}(में|के लिए) .{0,20}(मदद|हेल्प)|you.?re (into|focus)", _I)
_REPEAT_ASK = re.compile(r"(say|repeat|tell) (that|it|this)? ?(once more|again)|say (that )?again|repeat (that|it|please)|what was that|"
                         r"phir se bol|dobara bol|ek baar phir|फिर से बोल|दोबारा बोल|एक बार फिर|samajh nahi aaya|समझ नहीं आया|"
                         r"can.?t hear|sunai nahi|सुनाई नहीं|come again|pardon|\bsorry\?\s*$|क्या कहा|kya kaha|didn.?t (get|catch)", _I)
_BYE = re.compile(r"\b(bye|goodbye|good bye|bye ?bye|thanks?,? bye|ok(ay)? thanks?|thank you|thanks|that.?s all|that.?s it|talk later|take care|"
                  r"dhanyavaad|dhanyawad|shukriya|alvida|rakhta hoon|rakhti|rakh raha|chalo bye|theek hai bye|cut the call|cut call|hang up|disconnect|end the call|end call)\b|"
                  r"बाय|धन्यवाद|शुक्रिया|अलविदा|रखता हूँ|रखता हूं|फोन रख|कॉल (काट|खत्म)", _I)
_ASKS_AGAIN = re.compile(r"\baur\b|और (कुछ|क्या)|anything else|what else|kuch aur|कुछ और|\bmore\b|tell me more|aur batao|और बताओ", _I)
_INFO_Q = re.compile(r"what (will|do|can) i get|what.?s (in|inside|included)|what does it (do|include|give)|kya milega|kya milta|क्या मिलेगा|क्या मिलता|"
                     r"kya kya|क्या क्या|benefits?|features?|tell me about|kya karta|क्या करता|batao|बताओ|बताइए|bataiye|what (is|are) (it|this)|kya hai|क्या है", _I)

# claims the agent may only make when the product facts say so: phrase -> regex that must appear in the sources
_CLAIMS = (
    (re.compile(r"cancel (it )?(any ?time|whenever)|kabhi bhi cancel|कभी भी (cancel|कैंसल)|कैंसल कर सकते|cancel kar sakte", _I), r"cancel|कैंसल"),
    (re.compile(r"(five|5|panch|paanch|पाँच|पांच)\s*(times|x|guna|गुना|gunaa)|\b(\d+|panch|five)x\b", _I), r"(five|5|panch|paanch|पाँच|पांच)\s*(times|x|guna|गुना)|\d+x"),
    (re.compile(r"\bbest (ai |model|models|assistant|plan|gym|in)|sabse (achha|accha|behtar)|सबसे (अच्छा|बेहतर)|number (one|1)\b", _I), r"\bbest\b|sabse|सबसे|number (one|1)"),
    (re.compile(r"\bunlimited\b|असीमित|unlimited", _I), r"unlimited|असीमित"),
    (re.compile(r"money.?back|full refund|refund|पैसे वापस|paise wapas", _I), r"money.?back|refund|पैसे वापस|paise wapas"),
    (re.compile(r"free trial|trial period|ट्रायल|\btrial\b", _I), r"trial|ट्रायल"),
    (re.compile(r"no (hidden )?(charges|fees)|koi hidden|कोई hidden|lifetime|life.?time", _I), r"hidden|lifetime|no (extra )?(charge|fee)|no extra"),
    (re.compile(r"\d+\s*% ?(off|discount|छूट)|(\d+|sixty|fifty) percent", _I), r"\d+\s*% ?(off|discount|छूट)|percent"),
)
_MEMORY = re.compile(r"your (details|record|data|number|info\w*) (are|is)? ?(in|with|on) (our|the) (system|file|records?)|"
                     r"(previous|earlier|last) (inquiry|enquiry|conversation|call|visit|chat)|we (already )?have your (details|info)|"
                     r"remember (you|your)|from (our|your) (last|previous)|पिछली (बातचीत|पूछताछ|कॉल)|"
                     r"(आपकी|आपके) (डिटेल|जानकारी).{0,25}(हमारे पास|सिस्टम)|हमारे पास आपकी|pichli (baat|inquiry|call)|"
                     r"aapki details.{0,25}(hamare paas|system)|(you|aap) (had )?(enquired|inquired|called|visited|asked) (earlier|before|last)|"
                     r"aapne (pehle|pichli baar) (poocha|inquiry|enquiry|call)", _I)

_TOK = re.compile(r"[\wऀ-ॿ]+")


def content_words(text: str) -> list[str]:
    return sorted({w for w in _TOK.findall(text.lower()) if len(w) >= 4 and w not in _STOPW})


def invented_memory(text: str) -> str | None:
    m = _MEMORY.search(text)
    return m.group(0) if m else None


def unsourced_claims(text: str, sources: str) -> list[str]:
    """Claims in `text` that the source text (product facts / plan context / caller's words) does not support."""
    src = sources.lower()
    out = []
    for rx, need in _CLAIMS:
        m = rx.search(text)
        if m and not re.search(need, src, _I):
            out.append(m.group(0))
    return out


def is_repeat_ask(text: str) -> bool:
    return bool(_REPEAT_ASK.search(text or ""))


def explicit_bye(text: str) -> bool:
    from .needs import FIRM_NO
    return bool(_BYE.search(text or "") or FIRM_NO.search(text or "") or _DECISION["no"].search(text or ""))


_PLAIN_NO = {"no", "nahi", "nahin", "नहीं", "nope", "na", "ना", "nahi.", "thanks", "thank", "you", "ji", "जी", "please", "dhanyavaad", "धन्यवाद"}


def firm_no(text: str) -> bool:
    """A real no: opt-out wording, "not interested", "no thanks", or a bare "no / nahi nahi"."""
    t = text or ""
    if explicit_bye(t):
        return True
    if re.search(r"don.?t send|do not send|mat bhej|मत भेज|rehne do|रहने दो|changed my mind", t, re.I):
        return True  # cancelling the link they had agreed to is a firm no
    toks = _TOK.findall(t.lower())
    return bool(toks) and all(w in _PLAIN_NO for w in toks) and any(w in {"no", "nahi", "nahin", "नहीं", "nope", "na", "ना"} for w in toks) \
        or bool(_DECISION["no"].search(t))


def pure_farewell(text: str) -> bool:
    """Only a goodbye / thanks, nothing else to answer: "thanks, bye", "ok bye", "धन्यवाद"."""
    toks = _TOK.findall((text or "").lower())
    fill = {"ok", "okay", "thanks", "thank", "you", "bye", "goodbye", "good", "theek", "thik", "hai", "chalo", "achha", "acha", "so", "then", "alright",
            "dhanyavaad", "dhanyawad", "shukriya", "take", "care", "ठीक", "है", "चलो", "अच्छा", "ओके", "बाय", "धन्यवाद", "शुक्रिया", "तो", "फिर", "ji", "जी"}
    return bool(toks) and bool(_BYE.search(text or "")) and all(w in fill for w in toks)


def one_word_fragment(text: str) -> bool:
    """A lone opener the caller cut off after: "मुझे", "main", "mujhe", "I want"."""
    toks = _TOK.findall((text or "").lower())
    if (text or "").strip().endswith("?"):
        return False
    return len(toks) == 1 and toks[0] in {"मुझे", "main", "mujhe", "मैं", "मेरा", "mera", "mere", "i", "my", "so", "तो", "वो", "wo", "woh", "ye", "ये", "actually", "matlab",
                                           "मतलब", "lekin", "but", "and", "because", "kyunki", "क्योंकि", "hmm", "um", "uh"} - {"hmm", "um", "uh"}


def clear_input(text: str) -> bool:
    """The caller's words are plain speech, not line noise: a "didn't catch that" would be wrong."""
    from .needs import garbled
    t = (text or "").strip()
    if not t or garbled(t) or is_repeat_ask(t):
        return False
    if re.search(r"can.?t hear|breaking|आवाज़ नहीं|awaaz nahi|noise|\.\.\.|…", t, _I):
        return False
    toks = _TOK.findall(t.lower())
    if not toks:
        return False
    if _ANSWERS["goal"].search(t) or _BYE.search(t):
        return True
    ack = {"ok", "okay", "acha", "achha", "accha", "theek", "thik", "hai", "haan", "han", "ji", "yes", "yeah", "hmm", "alright", "sure", "fine",
           "right", "ठीक", "है", "हाँ", "हां", "जी", "अच्छा", "ओके", "हम्म", "सही", "then", "so", "to", "toh", "तो"}
    if all(w in ack for w in toks):
        return True
    return len(toks) >= 2 and len(set(toks)) >= 2  # two different real words are speech we can answer


def opener(text: str) -> str:
    """The first two words of a reply, normalised: two replies starting the same way sound like one template."""
    toks = norm(sentences(text)[0] if sentences(text) else text).split()
    return " ".join(toks[:2])


def phrases_list(ph: list[str]) -> str:
    return ", ".join(f'"{x}"' for x in ph[-10:])


def _grp(rx: str, en: list[str], hi: list[str], hl: list[str]) -> dict:
    return {"rx": re.compile(rx, re.I), "en": en, "hi": hi, "hinglish": hl}


# filler openers by meaning, with the synonyms rotation uses (male forms only)
PHRASE_GROUPS = [
    _grp(r"(bilkul|zaroor|sure|of course|absolutely|बिल्कुल|ज़रूर|जरूर)\b[,.!]?", ["Sure", "Of course", "Alright", "Okay"], ["ज़रूर", "ठीक है", "जी"], ["Zaroor", "Theek hai", "Ji", "Achha"]),
    _grp(r"(samajh sakta hoon|main samajhta hoon|i understand|i hear you|i get it|मैं समझता हूँ|समझ सकता हूँ)\b[,.!]?", ["I hear you", "That makes sense", "Fair enough", "Understood"],
         ["आपकी बात सही है", "समझ गया", "ठीक है"], ["Aapki baat sahi hai", "Samajh gaya", "Theek hai"]),
    _grp(r"(great|good|fair|nice) question\b[,.!]?|achha sawal hai|अच्छा सवाल है|accha sawal hai", ["Good point", "Fair to ask", "Happy to explain"], ["पूछना ठीक है", "बताता हूँ"], ["Poochna theek hai", "Batata hoon", "Achha point hai"]),
    _grp(r"(just to confirm|to confirm|bas confirm kar doon|बस कन्फ़र्म कर दूँ|बस confirm कर दूँ)\b[,:.!]?", ["One thing to check", "Quickly, to be sure", "So that it is clear"], ["एक बात पक्की कर दूँ", "साफ़ कर दूँ"], ["Ek baat pakki kar doon", "Saaf kar doon", "Ek cheez check kar leta hoon"]),
    _grp(r"(koi baat nahi|no problem|no worries|कोई बात नहीं)\b[,.!]?", ["That's fine", "No issue", "All good"], ["कोई दिक्कत नहीं", "ठीक है"], ["Koi dikkat nahi", "Theek hai", "Sab theek hai"]),
    _grp(r"(theek hai|ठीक है|okay|ok|alright|all right|got it|achha|अच्छा)\b[,.!]?", ["Alright", "Got it", "Understood", "Okay"], ["अच्छा", "समझ गया", "जी"], ["Achha", "Samajh gaya", "Ji", "Chalo"]),
    _grp(r"(waise|by the way|also|वैसे)\b[,.!]?", ["One more thing", "Also worth knowing", "On top of that"], ["एक और बात", "इसके अलावा"], ["Ek aur baat", "Iske alawa", "Aur ek cheez"]),
]


class CustomerFacts:
    """What the caller told us (goal, schedule, budget, experience, language, name, objections, decisions) and what we have
    asked / said (benefit and fact slots, question topics). Pure Python, serialised into session.dialog."""

    def __init__(self, data: dict | None = None):
        d = data or {}
        self.caller: dict[str, list[str]] = {k: list(v) for k, v in (d.get("caller") or {}).items()}
        self.objections: list[str] = list(d.get("objections", []))
        self.decisions: list[str] = list(d.get("decisions", []))
        self.said_slots: list[str] = list(d.get("said_slots", []))  # benefit_N / fact slots we told
        self.asked: dict[str, int] = dict(d.get("asked", {}))       # topic -> times we asked it
        self.heard: list[str] = list(d.get("heard", []))            # the caller's own raw sentences (for the "do not echo" check)
        self.guards: dict[str, int] = dict(d.get("guards", {}))     # how often each guard fired (diagnostics, rotation index)
        self.phrases: list[str] = list(d.get("phrases", []))        # filler openers / connectives already used (normalised), never reused
        self.openers: list[str] = list(d.get("openers", []))        # first two words of every earlier reply
        self.just_answered: list[str] = []                          # transient: ladder levels the caller answered in THIS turn
        self.ladder: list[dict] = []                                # transient: the product's discovery ladder (see use_ladder)

    # -- persistence
    @classmethod
    def load(cls, dialog: dict) -> "CustomerFacts":
        raw = dialog.get(FACTS_KEY)
        try:
            return cls(json.loads(raw)) if raw else cls()
        except (ValueError, TypeError):
            return cls()

    def save(self, dialog: dict) -> None:
        dialog[FACTS_KEY] = json.dumps({"caller": {k: self.caller[k][-6:] for k in sorted(self.caller)}, "objections": self.objections[-8:],
                                        "decisions": self.decisions[-8:], "said_slots": self.said_slots[-40:],
                                        "asked": self.asked, "heard": self.heard[-10:], "guards": self.guards,
                                        "phrases": self.phrases[-40:], "openers": self.openers[-20:]},
                                       ensure_ascii=False, sort_keys=True)

    # -- the caller
    def learn_caller(self, text: str, *, asked_goal: bool = False) -> None:
        t = (text or "").strip()
        if not t:
            return
        for topic in ("goal", "schedule", "budget", "experience"):
            hits = sorted({m.group(0).lower().strip() for m in _ANSWERS[topic].finditer(t) if m.group(0).strip()})
            if hits:
                cur = self.caller.setdefault(topic, [])
                for h in hits:
                    if h not in cur:
                        cur.append(h)
        m = _ANSWERS["name"].search(t)
        if m and m.group(2).lower() not in _NOT_NAME and "name" not in self.caller:
            self.caller["name"] = [m.group(2)]
        if "goal" not in self.caller and asked_goal and clear_input(t) and len(_TOK.findall(t)) <= 12 and not is_repeat_ask(t) \
                and not re.search(r"\?\s*$", t):
            self.caller["goal"] = [" ".join(content_words(t)[:3]) or t[:30].lower()]  # they answered our goal question in other words
        for k in sorted(_OBJ):
            if _OBJ[k].search(t) and k not in self.objections:
                self.objections.append(k)
        for k in sorted(_DECISION):
            if _DECISION[k].search(t) and k not in self.decisions:
                self.decisions.append(k)
        for lv in self.ladder:
            rx = lv.get("answers_regex") or ""
            try:
                hit = re.search(rx, t, _I) if rx else None
            except re.error:
                hit = None
            if hit and not is_repeat_ask(t):
                cur = self.caller.setdefault("lvl:" + lv["id"], [])
                if hit.group(0).lower() not in cur:
                    cur.append(hit.group(0).lower())
                    self.just_answered.append(lv["id"])
        if len(_TOK.findall(t)) >= 3 and len(self.heard) < 40:
            self.heard.append(t[:160])

    def knows(self, topic: str) -> bool:
        return bool(self.caller.get(topic))

    # -- what we said / asked
    def learn_agent(self, text: str, slots: list[str] | None = None) -> None:
        op = opener(text)
        if op:
            self.openers.append(op)
        for g in PHRASE_GROUPS:
            for sent in sentences(text):
                m = g["rx"].match(sent.strip())
                if m:
                    n = norm(m.group(0))
                    if n not in self.phrases:
                        self.phrases.append(n)
        for lv in self.ladder:
            for sent in sentences(text):
                if sent.rstrip().endswith("?") and any(ratio(sent, q) > 0.6 for q in (lv.get("question") or {}).values()):
                    self.asked["lvl:" + lv["id"]] = self.asked.get("lvl:" + lv["id"], 0) + 1
        for s in sentences(text):
            if s.rstrip().endswith("?"):
                for topic in sorted(_ASKS):
                    if _ASKS[topic].search(s):
                        self.asked[topic] = self.asked.get(topic, 0) + 1
        for sl in slots or []:
            if sl and sl not in self.said_slots:
                self.said_slots.append(sl)

    def reasks(self, text: str) -> list[str]:
        """Topics the reply asks about that the caller has already answered (or that we already asked once and they replied)."""
        out = []
        for s in sentences(text):
            if not s.rstrip().endswith("?"):
                continue
            for topic in sorted(_ASKS):
                if _ASKS[topic].search(s) and (self.knows(topic) or self.asked.get(topic, 0) >= 1):
                    out.append(topic)
            for lv in self.ladder:  # a ladder question whose level is answered is never asked again
                if self.caller.get("lvl:" + lv["id"]) and any(ratio(s, q) > 0.6 for q in (lv.get("question") or {}).values()):
                    out.append("lvl:" + lv["id"])
        return out

    def restates(self, text: str) -> list[str]:
        """Sentences of the reply that say the caller's own facts back to them ("it will help with muscle", "आप कॉलेज में हैं")."""
        mine = {w for k in ("goal", "experience", "schedule") for v in self.caller.get(k, []) for w in _TOK.findall(v)} - _STOPW
        heard = {w for h in self.heard[-3:] for w in content_words(h)}
        out = []
        for s in sentences(text):
            if s.rstrip().endswith("?"):
                continue
            low = s.lower()
            words = set(_TOK.findall(low))
            own = {w for w in words if w in mine and len(w) >= 3}
            echo = {w for w in words if w in heard}
            if _ECHO_FRAME.search(low) and (own or len(echo) >= 2):
                out.append(s)
            elif self.heard and any(len(norm(h).split()) >= 3 and ratio(s, h) > 0.62 for h in self.heard[-4:]):
                out.append(s)
        return out

    # -- the discovery ladder: goal -> sub-goal -> experience -> schedule ..., only ever deeper
    def use_ladder(self, product) -> None:
        """Load the product's discovery ladder (Product.discovery_ladder). With none, the product's single discovery_question is level one."""
        lad = [dict(x) for x in (getattr(product, "discovery_ladder", None) or []) if isinstance(x, dict) and x.get("id") and x.get("question")]
        if not lad:
            dq = getattr(product, "discovery_question", None)
            if isinstance(dq, dict) and dq.get("en"):
                lad = [{"id": "goal", "question": dict(dq), "answers_regex": _ANSWERS["goal"].pattern}]
        self.ladder = lad

    def answered_levels(self) -> list[str]:
        return [lv["id"] for lv in self.ladder if self.caller.get("lvl:" + lv["id"]) or (lv["id"] == "goal" and self.knows("goal"))
                or (lv["id"] in ("experience", "schedule") and self.knows(lv["id"]))]

    def next_discovery(self, product, lang: str) -> str | None:
        """The question of the NEXT UNANSWERED level of the ladder (never one the caller already answered, never one asked twice already),
        in `lang`; None when the ladder is done. A deeper level is only asked once the levels above it are answered or were asked twice."""
        if not self.ladder:
            self.use_ladder(product)
        done = set(self.answered_levels())
        for lv in self.ladder:
            if lv["id"] in done or self.asked.get("lvl:" + lv["id"], 0) + (self.asked.get(lv["id"], 0) if lv["id"] in _ASKS else 0) >= 2:
                continue
            q = lv["question"]
            return q.get(lang) or None  # no text in this language: never paste English into a Hindi call
        return None

    # -- phrasing: the same filler is never used twice
    def rotate_opener(self, text: str, lang: str) -> str:
        """Swap a leading filler ("Bilkul", "Samajh sakta hoon", "Great question", "Just to confirm", ...) that this call already used for the
        next unused synonym of its group; none left -> the filler is dropped. Deterministic."""
        parts = sentences(text)
        if not parts:
            return text
        first = parts[0]
        for g in PHRASE_GROUPS:
            m = g["rx"].match(first)
            if not m:
                continue
            used = norm(m.group(0))
            if used not in self.phrases:
                return text
            rest = first[m.end():].lstrip(" ,;:-")
            for alt in g.get(lang, []):
                if norm(alt) not in self.phrases and norm(alt) != used:
                    sep = ", " if rest else ""
                    new = alt + sep + (rest[0].lower() + rest[1:] if rest and lang != "hi" and rest[0].isupper() and not rest[:2].isupper() else rest)
                    return " ".join([new if rest else alt + ("." if not alt.endswith((".", "।")) else "")] + parts[1:]).strip()
            if rest:
                return " ".join([rest[0].upper() + rest[1:]] + parts[1:]).strip() if lang != "hi" else " ".join([rest] + parts[1:]).strip()
            return " ".join(parts[1:]) or text
        return text

    def opener_used(self, text: str) -> bool:
        op = opener(text)
        return bool(op) and op in self.openers

    def phrases_block(self) -> str:
        return ", ".join(self.phrases[-10:])

    # -- the prompt block
    def prompt_block(self, said: "SaidLedger | None" = None) -> str:
        """'Already known / already said: do not restate' for the LLM prompt. Empty when nothing is known yet."""
        known = []
        labels = {"goal": "goal / use", "schedule": "schedule", "budget": "budget", "experience": "background", "name": "name"}
        for k in ("goal", "schedule", "budget", "experience", "name"):
            if self.caller.get(k):
                known.append(f"{labels[k]}: {', '.join(self.caller[k][:3])}")
        if self.objections:
            known.append("objections raised: " + ", ".join(self.objections[:4]))
        if self.decisions:
            known.append("caller decisions: " + ", ".join(self.decisions[:4]))
        lines = []
        if known:
            lines.append("The caller ALREADY TOLD you (never ask it again, never say it back to them): " + "; ".join(known) + ".")
        if self.said_slots:
            lines.append("Points ALREADY SAID (do not say them again, pick something new): " + ", ".join(self.said_slots[:12]) + ".")
        if self.phrases:
            lines.append("Phrases / openers ALREADY USED (do not start with them again, use a different way to begin): " + phrases_list(self.phrases))
        if said is not None and said.sentences:
            lines.append("Sentences you already said (never repeat or reword them): " + " | ".join(s[:70] for s in said.sentences[-6:]))
        if not lines:
            return ""
        return "ALREADY KNOWN / ALREADY SAID (do not restate):\n" + "\n".join("- " + x for x in lines) + "\n\n"


# ---------------------------------------------------------------- the next new thing to say (deterministic)

_FORWARD = {
    "sell": {"en": ["Does that sound like what you need?", "Would you like to go ahead with it?"],
             "hi": ["क्या ये आपकी ज़रूरत के हिसाब से ठीक लग रहा है?", "क्या आप इसे आगे बढ़ाना चाहेंगे?"],
             "hinglish": ["Kya ye aapki zaroorat ke hisaab se theek lag raha hai?", "Kya aap ise aage badhana chahenge?"]},
    "link": {"en": ["Shall I send the link to this number?", "Should I send the payment link now?"],
             "hi": ["क्या मैं इसी नंबर पर लिंक भेज दूँ?", "क्या मैं अभी पेमेंट लिंक भेज दूँ?"],
             "hinglish": ["Kya main isi number par link bhej doon?", "Kya main abhi payment link bhej doon?"]},
    "ask": {"en": ["What else would you like to know?", "Is there anything specific you'd like me to cover?"],
            "hi": ["आप और क्या जानना चाहेंगे?", "क्या कोई ख़ास बात है जो आप जानना चाहते हैं?"],
            "hinglish": ["Aap aur kya jaanna chahenge?", "Koi khaas baat hai jo aap jaanna chahte hain?"]},
    "go": {"en": ["Ji, please go ahead.", "Sure, tell me more."],
           "hi": ["जी, बताइए।", "जी, आगे बताइए।"], "hinglish": ["Ji, bataiye.", "Ji, aage bataiye."]},
}
_BRIDGE = {"en": ["Here is something else: ", "One more thing: "], "hi": ["एक और बात: ", "और एक बात: "], "hinglish": ["Ek aur baat: ", "Aur ek baat: "]}


def covered(line: str, ledger: SaidLedger) -> bool:
    """The point of `line` (its words after the lead-in colon) was already said in other words earlier in the call."""
    core = line.split(":", 1)[-1]
    words = set(content_words(core))
    if len(words) < 2:
        return False
    said = {w for s in ledger.sentences for w in _TOK.findall(s)}
    return len(words & said) / len(words) >= 0.6


def next_unsaid(product, lang: str, ledger: SaidLedger, facts: "CustomerFacts | None" = None, text: str = "",
                allow_sell: bool = False, allow_link: bool = False, kinds=("benefit", "fact", "faq")) -> tuple[str | None, str]:
    """The next point the call has NOT said yet: (slot, text in `lang`). Prefers the catalog item the caller's words match, then the
    benefits in catalog order, then facts. Never the price line (that has its own moves). Else a short, NEW forward step
    (rotating by use count) -> (None, text). Deterministic. `text` is the caller's words."""
    from .moves import build_catalog, _matches
    said_slots = set(ledger.slots) | set(facts.said_slots if facts else [])
    if product is not None:
        cat = [i for i in build_catalog(product) if i.kind in kinds and lang in i.text and i.slot not in said_slots
               and i.slot not in ("offer", "how") and not ledger.repeats(i.text[lang]) and not covered(i.text[lang], ledger)]
        hit = next((i for i in cat if text and _matches(i, text)), None)
        order = {"benefit": 0, "fact": 1, "faq": 2}
        pick = hit or (sorted(cat, key=lambda i: (order.get(i.kind, 3)))[0] if cat else None)
        if pick is not None:
            line = pick.text[lang]
            return pick.slot, line
    n = ledger.count_move("forward") + (facts.guards.get("forward", 0) if facts else 0)
    bank = ("link" if allow_link else "sell") if allow_sell and not ledger.refused else ("go" if n % 2 else "ask")
    variants = _FORWARD[bank][lang]
    for k in range(len(variants)):
        v = variants[(n + k) % len(variants)]
        if not ledger.repeats(v) and not (ledger.reply_repeats(v)):
            return None, v
    for b in (("link",) if allow_link else ()) + (("sell",) if allow_sell else ()) + ("ask", "go"):
        for v in _FORWARD[b][lang]:
            if not ledger.repeats(v):
                return None, v
    return None, _FORWARD["go"][lang][n % 2]


def crc_pick(options: list[str], key: str) -> str:
    """Stable choice by zlib.crc32 (never hash())."""
    return options[zlib.crc32(key.encode("utf-8")) % len(options)] if options else ""
