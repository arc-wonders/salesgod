"""Layer 2 - lexicons and rule scoring for the ten extended human-state dimensions.

satisfaction, anxiety, excitement, calmness, trust, certainty, cognitive_load, attention,
participation, receptiveness. Phrases are English, Roman Hindi and Devanagari, in the same
format as the lexicons in estimator.py ("x*" suffix match, "=x" whole utterance).
Every dimension that is neutral by default (0.5) returns 0.5 when nothing is heard, so
the baseline-aware smoothing in the estimator decays it back to 0.5.
"""

from __future__ import annotations

from ..schemas import SessionState, TurnInput
from ..text import jaccard, tokens

NEUTRAL_DIMS = {"satisfaction": 0.5, "calmness": 0.5, "trust": 0.5, "certainty": 0.5,
                "attention": 0.5, "participation": 0.5, "receptiveness": 0.5, "engagement": 0.5}
NEW_DIMS = ["satisfaction", "anxiety", "excitement", "calmness", "trust", "certainty",
            "cognitive_load", "attention", "participation", "receptiveness"]

LIKE_PHRASES: dict[str, float] = {
    "accha hai": 0.7, "achha hai": 0.7, "acha hai": 0.7, "bahut accha": 0.8, "bahut achha": 0.8,
    "bahut badhiya": 0.9, "badhiya": 0.7, "badiya": 0.7, "bahut acha": 0.8, "zabardast": 0.9,
    "mast hai": 0.8, "mast": 0.5, "kamaal": 0.8, "shandar": 0.8, "pasand aaya": 0.8,
    "pasand aya": 0.8, "pasand hai": 0.7, "mujhe pasand": 0.7, "maza aa gaya": 0.8, "sahi hai": 0.5,
    "great": 0.7, "awesome": 0.8, "excellent": 0.8, "perfect": 0.8, "love it": 0.9, "i like": 0.6,
    "sounds good": 0.7, "thats good": 0.6, "very good": 0.7, "nice": 0.5, "wonderful": 0.8,
    "happy with": 0.8, "thank you so much": 0.7, "thanks a lot": 0.6, "helpful": 0.6, "good": 0.4,
    "बहुत अच्छा": 0.8, "अच्छा है": 0.7, "बढ़िया": 0.7, "बहुत बढ़िया": 0.9, "बढिया": 0.7,
    "बहुत बढिया": 0.9, "जबरदस्त": 0.9, "शानदार": 0.8, "कमाल": 0.8, "पसंद आया": 0.8,
    "पसंद है": 0.7, "मजा आ गया": 0.8, "मस्त": 0.5, "सही है": 0.5, "धन्यवाद": 0.4, "शुक्रिया": 0.4,
}
LIKE_MASKS = ["not good", "not so good", "no good", "not great", "not very good", "accha nahi",
              "achha nahi", "acha nahi", "accha nhi", "theek nahi", "pasand nahi", "अच्छा नहीं",
              "अच्छा नही", "पसंद नहीं", "not happy", "not nice", "not perfect", "good for nothing"]
DISLIKE_PHRASES: dict[str, float] = {
    "bekaar": 0.9, "bekar": 0.9, "bakwas": 0.9, "ghatiya": 0.9, "kharab hai": 0.8, "bura hai": 0.7,
    "accha nahi": 0.7, "achha nahi": 0.7, "acha nahi": 0.7, "accha nhi": 0.7, "pasand nahi": 0.8,
    "mujhe pasand nahi": 0.9, "theek nahi": 0.6, "maza nahi": 0.6, "bahut bura": 0.8,
    "not good": 0.7, "not great": 0.6, "not happy": 0.8, "dont like": 0.8, "i hate": 0.9, "hate it": 0.9,
    "disappointed": 0.8, "disappointing": 0.8, "terrible": 0.8, "awful": 0.8, "horrible": 0.8,
    "useless": 0.8, "worst": 0.8, "poor": 0.5, "bad": 0.5, "unhappy": 0.8, "waste": 0.6, "not satisfied": 0.8,
    "बेकार": 0.9, "बकवास": 0.9, "घटिया": 0.9, "खराब है": 0.8, "बुरा है": 0.7, "अच्छा नहीं": 0.7,
    "पसंद नहीं": 0.8, "ठीक नहीं": 0.6, "मजा नहीं": 0.6, "बहुत बुरा": 0.8,
}

ANXIETY_PHRASES: dict[str, float] = {
    "dar lag raha": 0.9, "dar lag rahi": 0.9, "dar lagta": 0.8, "darr lag": 0.8, "dar hai": 0.7,
    "ghabra*": 0.8, "tension ho rahi": 0.8, "tension hai": 0.6, "tension": 0.5, "chinta": 0.7,
    "fikar": 0.7, "fikr": 0.7, "safe hai na": 0.7, "safe hai kya": 0.7, "surakshit": 0.4,
    "kuch ho to nahi jayega": 0.8, "koi problem to nahi": 0.6, "kuch galat to nahi": 0.7,
    "paise doob": 0.8, "paisa doob": 0.8, "nuksan": 0.5, "risk": 0.4, "risky": 0.6,
    "worried": 0.8, "worry": 0.6, "scared": 0.9, "afraid": 0.8, "nervous": 0.8, "anxious": 0.9,
    "is it safe": 0.6, "will i lose": 0.8, "what if something goes wrong": 0.8, "what if": 0.4,
    "im not sure about this": 0.4, "panic": 0.8, "stress": 0.5, "stressed": 0.7, "i am afraid": 0.8,
    "डर लग रहा": 0.9, "डर लग रही": 0.9, "डर लगता": 0.8, "डर है": 0.7, "घबरा*": 0.8,
    "टेंशन": 0.6, "चिंता": 0.7, "फिक्र": 0.7, "सेफ है ना": 0.7, "सेफ है क्या": 0.7,
    "सुरक्षित": 0.4, "कुछ गलत तो नहीं": 0.7, "पैसे डूब": 0.8, "नुकसान": 0.5, "रिस्क": 0.4, "रिस्की": 0.6,
}
ANXIETY_MASKS = ["no worries", "dont worry", "do not worry", "not worried", "no tension", "tension nahi",
                 "tension mat", "koi tension nahi", "dar nahi", "not scared", "not afraid", "no risk",
                 "no problem", "टेंशन नहीं", "टेंशन मत", "डर नहीं", "चिंता नहीं", "चिंता मत"]

EXCITE_PHRASES: dict[str, float] = {
    "wow": 0.8, "amazing": 0.8, "awesome": 0.7, "fantastic": 0.8, "cant wait": 0.9, "so excited": 0.9,
    "excited": 0.8, "thrilled": 0.9, "really": 0.2, "oh nice": 0.6, "super": 0.5, "yay": 0.8,
    "love it": 0.7, "tell me more": 0.5, "haan haan": 0.4, "bahut badhiya": 0.6, "zabardast": 0.7,
    "kamaal": 0.7, "ekdum mast": 0.8, "maza aa gaya": 0.7, "wah": 0.6, "waah": 0.6, "arre wah": 0.8,
    "are wah": 0.8, "sachchi": 0.4, "sach mein": 0.4, "sach me": 0.4, "bahut khushi": 0.8,
    "khush": 0.5, "utsuk": 0.7, "excited hoon": 0.9, "jaldi batao": 0.6, "batao batao": 0.7,
    "वाह": 0.6, "अरे वाह": 0.8, "वाओ": 0.8, "कमाल": 0.7, "ज़बरदस्त": 0.7, "जबरदस्त": 0.7,
    "एकदम मस्त": 0.8, "मजा आ गया": 0.7, "सच में": 0.4, "बहुत खुशी": 0.8, "खुश": 0.5,
    "उत्सुक": 0.7, "बताओ बताओ": 0.7, "हां हां": 0.4,
}
EXCITE_MASKS = ["not excited", "not very excited", "not that excited", "excited nahi", "खुश नहीं"]

CALM_PHRASES: dict[str, float] = {
    "no problem": 0.6, "no worries": 0.7, "take your time": 0.8, "no rush": 0.8, "no hurry": 0.8,
    "its fine": 0.5, "its okay": 0.5, "all good": 0.6, "relax": 0.4, "sure": 0.3, "of course": 0.3,
    "koi baat nahi": 0.7, "koi dikkat nahi": 0.7, "koi problem nahi": 0.7, "koi jaldi nahi": 0.8,
    "aaram se": 0.8, "araam se": 0.8, "tension nahi": 0.7, "chalta hai": 0.5, "theek hai": 0.3,
    "shaanti se": 0.7, "dhire dhire": 0.5, "dheere se": 0.5, "sab theek": 0.5,
    "कोई बात नहीं": 0.7, "कोई दिक्कत नहीं": 0.7, "कोई प्रॉब्लम नहीं": 0.7, "कोई जल्दी नहीं": 0.8,
    "आराम से": 0.8, "टेंशन नहीं": 0.7, "चलता है": 0.5, "ठीक है": 0.3, "शांति से": 0.7,
    "धीरे धीरे": 0.5, "सब ठीक": 0.5,
}
CALM_MASKS = ["not fine", "not okay", "not ok", "not good", "no time", "not now"]

TRUST_UP: dict[str, float] = {
    "genuine": 0.7, "trust you": 0.8, "i trust": 0.8, "reliable": 0.7, "verified": 0.6, "legit": 0.7,
    "legitimate": 0.7, "trustworthy": 0.8, "not a scam": 0.8, "no scam": 0.7, "credible": 0.7,
    "bharosa hai": 0.8, "bharosa karta": 0.8, "bharosa karti": 0.8, "vishwas hai": 0.8, "vishwas": 0.5,
    "bharosemand": 0.8, "scam nahi": 0.8, "fraud nahi": 0.8, "asli hai": 0.7, "real hai": 0.6,
    "jaanta hoon aapki company": 0.7, "jaanta hu": 0.4, "recommended": 0.5, "sunna hai accha": 0.4,
    "भरोसा है": 0.8, "भरोसा करता": 0.8, "भरोसा करती": 0.8, "विश्वास है": 0.8, "विश्वास": 0.5,
    "भरोसेमंद": 0.8, "स्कैम नहीं": 0.8, "फ्रॉड नहीं": 0.8, "असली है": 0.7,
}
TRUST_DOWN: dict[str, float] = {
    "fraud": 0.9, "scam": 0.9, "fake": 0.9, "cheat": 0.8, "cheating": 0.8, "con": 0.4, "spam": 0.6,
    "suspicious": 0.8, "shady": 0.8, "dont trust": 0.9, "do not trust": 0.9, "cant trust": 0.9,
    "how do i trust": 0.8, "how can i trust": 0.8, "is it genuine": 0.5, "is this genuine": 0.5,
    "is this real": 0.5, "is it real": 0.5, "is it legit": 0.6, "prove it": 0.5, "sounds fishy": 0.8,
    "fishy": 0.7, "too good to be true": 0.8, "not sure about your company": 0.7,
    "kaise bharosa": 0.9, "kaise vishwas": 0.9, "bharosa nahi": 0.9, "bharosa kaise": 0.9,
    "vishwas nahi": 0.9, "fraud hai": 0.9, "scam hai": 0.9, "nakli": 0.8, "nakli hai": 0.9,
    "dhokha": 0.9, "dhoka": 0.9, "thagi": 0.9, "thug": 0.8, "jhooth": 0.6, "jhuth": 0.6, "jhoot": 0.6,
    "genuine hai kya": 0.5, "asli hai kya": 0.5, "sach hai kya": 0.5, "pata kaise": 0.4,
    "kaun si company": 0.4, "number kahan se mila": 0.7, "mera number kaise": 0.7, "number kaise mila": 0.7,
    "कैसे भरोसा": 0.9, "कैसे विश्वास": 0.9, "भरोसा नहीं": 0.9, "भरोसा कैसे": 0.9, "विश्वास नहीं": 0.9,
    "फ्रॉड": 0.9, "स्कैम": 0.9, "फर्जी": 0.9, "नकली": 0.8, "नकली है": 0.9, "धोखा": 0.9, "ठगी": 0.9,
    "झूठ": 0.6, "जेनुइन है क्या": 0.5, "असली है क्या": 0.5, "सच है क्या": 0.5,
    "मेरा नंबर कैसे": 0.7, "नंबर कहां से मिला": 0.7,
}
TRUST_DOWN_MASKS = ["not a scam", "no scam", "scam nahi", "fraud nahi", "not fraud", "not fake",
                    "fake nahi", "स्कैम नहीं", "फ्रॉड नहीं", "नकली नहीं", "nakli nahi"]

HEDGE_PHRASES: dict[str, float] = {
    "shayad": 0.8, "pata nahi": 0.9, "patta nahi": 0.9, "maloom nahi": 0.9, "nahi pata": 0.9,
    "kuch kah nahi sakta": 0.8, "ho sakta hai": 0.6, "lagta hai": 0.5, "pata nhi": 0.9, "dekhte hain": 0.5,
    "soch*": 0.5, "kya pata": 0.7, "ya phir": 0.4, "confuse hoon": 0.7, "pakka nahi": 0.8,
    "maybe": 0.7, "perhaps": 0.7, "not sure": 0.9, "dont know": 0.9, "i dont know": 0.9, "i guess": 0.7,
    "might": 0.5, "probably": 0.5, "possibly": 0.6, "i think": 0.4, "kind of": 0.4, "sort of": 0.4,
    "unsure": 0.9, "no idea": 0.9, "let me think": 0.7, "i suppose": 0.6, "not certain": 0.9,
    "शायद": 0.8, "पता नहीं": 0.9, "मालूम नहीं": 0.9, "नहीं पता": 0.9, "हो सकता है": 0.6,
    "लगता है": 0.5, "देखते हैं": 0.5, "सोच*": 0.5, "क्या पता": 0.7, "पक्का नहीं": 0.8,
}
HEDGE_MASKS = ["no idea about the delay", "i think so too"]
ASSERT_PHRASES: dict[str, float] = {
    "definitely": 0.8, "absolutely": 0.8, "for sure": 0.8, "certainly": 0.8, "of course": 0.5,
    "100 percent": 0.9, "hundred percent": 0.9, "i am sure": 0.9, "im sure": 0.9, "no doubt": 0.9,
    "without a doubt": 0.9, "exactly": 0.6, "i know": 0.5, "i want": 0.5, "i will take": 0.7,
    "pakka": 0.8, "pakka hai": 0.9, "bilkul": 0.8, "zaroor": 0.7, "jarur": 0.7, "yakeen hai": 0.9,
    "yakin hai": 0.9, "mujhe pata hai": 0.7, "sau percent": 0.9, "pura yakeen": 0.9, "confirm": 0.5,
    "पक्का": 0.8, "पक्का है": 0.9, "बिल्कुल": 0.8, "जरूर": 0.7, "ज़रूर": 0.7, "यकीन है": 0.9,
    "मुझे पता है": 0.7, "सौ प्रतिशत": 0.9, "पूरा यकीन": 0.9,
}
ASSERT_MASKS = ["pakka nahi", "पक्का नहीं", "not sure", "not certain", "not definitely", "not for sure",
                "yakeen nahi", "yakin nahi", "यकीन नहीं"]

LOAD_PHRASES: dict[str, float] = {
    "ek minute": 0.7, "ek min": 0.6, "ruko": 0.6, "ruk jao": 0.6, "ruk jaiye": 0.6, "one minute": 0.7,
    "one sec": 0.6, "hold on": 0.6, "wait": 0.5, "give me a minute": 0.8, "give me a second": 0.7,
    "let me think": 0.6, "too much": 0.7, "too many": 0.6, "slow down": 0.8, "bahut zyada": 0.6,
    "dhire bolo": 0.8, "dheere bolo": 0.8, "thoda dheere": 0.8, "thoda slow": 0.8, "complicated": 0.7,
    "bahut complicated": 0.8, "mushkil": 0.5, "dimag ghoom": 0.9, "sar ghoom": 0.9, "overwhelm*": 0.9,
    "dobara bolo": 0.7, "dubara bolo": 0.7, "ek baar aur": 0.7, "phir se bolo": 0.7, "fir se bolo": 0.7,
    "repeat": 0.6, "say that again": 0.6, "once more": 0.6, "come again": 0.6, "what was that": 0.6,
    "ek मिनट": 0.7, "एक मिनट": 0.7, "एक सेकंड": 0.6, "रुको": 0.6, "रुक जाओ": 0.6, "रुकिए": 0.6,
    "धीरे बोलो": 0.8, "धीरे बोलिए": 0.8, "थोड़ा धीरे": 0.8, "बहुत ज्यादा": 0.6, "मुश्किल": 0.5,
    "दिमाग घूम": 0.9, "सर घूम": 0.9, "दोबारा बोलो": 0.7, "दुबारा बोलो": 0.7, "एक बार और": 0.7,
    "फिर से बोलो": 0.7, "फिर से बताओ": 0.7, "रिपीट": 0.6,
}
HESITATION_TOKENS = {"um", "umm", "ummm", "uh", "uhh", "er", "erm", "hmm", "hmmm", "aaa", "aa", "ah",
                     "उम", "अं", "हम्म", "आ", "अम"}

AGREE_PHRASES: dict[str, float] = {
    "tell me more": 0.8, "go ahead": 0.8, "send it": 0.8, "send me": 0.8, "sounds good": 0.7,
    "interested": 0.8, "im interested": 0.9, "i am interested": 0.9, "yes please": 0.8, "lets do it": 0.9,
    "i would like": 0.7, "show me": 0.7, "how does it work": 0.6, "what are the options": 0.6,
    "makes sense": 0.7, "okay lets": 0.7, "i will take": 0.9, "sign me up": 0.9, "i want": 0.6,
    "batao": 0.6, "bataiye": 0.6, "batana": 0.5, "bhejo": 0.7, "bhej do": 0.8, "bhej dijiye": 0.8,
    "haan bhejo": 0.9, "haan batao": 0.8, "interest hai": 0.8, "interested hoon": 0.9, "dikhao": 0.6,
    "chalega": 0.6, "theek hai batao": 0.8, "samajh aaya": 0.5, "samajh gaya": 0.5, "samajh gayi": 0.5,
    "lena hai": 0.8, "le lunga": 0.9, "le lungi": 0.9, "karte hain": 0.7, "kar do": 0.7, "chaliye": 0.5,
    "बताओ": 0.6, "बताइए": 0.6, "भेजो": 0.7, "भेज दो": 0.8, "भेज दीजिए": 0.8, "हां भेजो": 0.9,
    "हां बताओ": 0.8, "इंटरेस्ट है": 0.8, "इंटरेस्टेड हूं": 0.9, "दिखाओ": 0.6, "चलेगा": 0.6,
    "समझ आया": 0.5, "समझ गया": 0.5, "समझ गई": 0.5, "लेना है": 0.8, "ले लूंगा": 0.9, "ले लूंगी": 0.9,
    "करते हैं": 0.7, "कर दो": 0.7,
}
AGREE_MASKS = ["not interested", "no interest", "interest nahi", "interested nahi", "not at all interested",
               "i dont want", "इंटरेस्ट नहीं", "इंटरेस्टेड नहीं", "not looking", "nahi bhejo", "mat bhejo"]
REFUSE_PHRASES: dict[str, float] = {
    "not interested": 0.9, "no thanks": 0.8, "no thank you": 0.8, "dont call": 0.9, "do not call": 0.9,
    "stop calling": 0.9, "remove my number": 0.9, "i am busy": 0.6, "im busy": 0.6, "busy right now": 0.7,
    "call later": 0.5, "call me later": 0.5, "not now": 0.5, "dont need": 0.8, "no need": 0.7,
    "not looking": 0.7, "leave me alone": 0.9, "cut the call": 0.8, "hang up": 0.8, "i dont want": 0.9,
    "nahi chahiye": 0.9, "nahin chahiye": 0.9, "nhi chahiye": 0.9, "interest nahi": 0.9, "interested nahi": 0.9,
    "zarurat nahi": 0.8, "jarurat nahi": 0.8, "mujhe nahi": 0.6, "busy hoon": 0.6, "busy hu": 0.6,
    "baad mein": 0.4, "baad me": 0.4, "phir kabhi": 0.5, "call mat karo": 0.9, "phone mat karo": 0.9,
    "number hata do": 0.9, "rakh do": 0.7, "phone rakh": 0.7, "mat karo": 0.5, "nahi lena": 0.9,
    "नहीं चाहिए": 0.9, "नही चाहिए": 0.9, "इंटरेस्ट नहीं": 0.9, "इंटरेस्टेड नहीं": 0.9, "जरूरत नहीं": 0.8,
    "ज़रूरत नहीं": 0.8, "बिजी हूं": 0.6, "बाद में": 0.4, "फिर कभी": 0.5, "कॉल मत करो": 0.9,
    "फोन मत करो": 0.9, "नंबर हटा दो": 0.9, "रख दो": 0.7, "नहीं लेना": 0.9,
}
REFUSE_MASKS = ["no problem", "no worries", "not now but", "i dont mind", "no need to worry",
                "zarurat nahi padegi", "no need to hurry", "no need to repeat"]


def _lex():
    from .estimator import _Lexicon  # circular-safe lazy import
    return _Lexicon


class _Set:
    """Compiled lexicons (built on first use to avoid an import cycle with estimator.py)."""

    def __init__(self):
        L = _lex()
        self.like = L(LIKE_PHRASES, LIKE_MASKS)
        self.dislike = L(DISLIKE_PHRASES)
        self.anx = L(ANXIETY_PHRASES, ANXIETY_MASKS)
        self.exc = L(EXCITE_PHRASES, EXCITE_MASKS)
        self.calm = L(CALM_PHRASES, CALM_MASKS)
        self.trust_up = L(TRUST_UP)
        self.trust_down = L(TRUST_DOWN, TRUST_DOWN_MASKS)
        self.hedge = L(HEDGE_PHRASES, HEDGE_MASKS)
        self.assertive = L(ASSERT_PHRASES, ASSERT_MASKS)
        self.load = L(LOAD_PHRASES)
        self.agree = L(AGREE_PHRASES, AGREE_MASKS)
        self.refuse = L(REFUSE_PHRASES, REFUSE_MASKS)
        self.score = L.score


_SET: _Set | None = None


def _s() -> _Set:
    global _SET
    if _SET is None:
        _SET = _Set()
    return _SET


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


def score_extended(turn: TurnInput, session: SessionState, base: dict[str, float],
                   low: str, bare: str, toks: list[str], content: list[str], minimal: bool,
                   ev: list[str]) -> dict[str, float]:
    """Rule scores for the ten extended dims. `base` holds the four core dims for this turn
    (frustration, confusion, urgency, engagement), already computed."""
    S = _s()
    sc = S.score
    t = turn.timing
    text = turn.text
    out: dict[str, float] = {}

    def hits(lex):
        return lex.hits(low, bare)

    like, dis = hits(S.like), hits(S.dislike)
    sat = 0.5 + 0.45 * sc(like) - 0.5 * sc(dis) - 0.25 * base["frustration"]
    if like:
        ev.append(f"liking: {like[0][0]}")
    if dis:
        ev.append(f"disliking: {dis[0][0]}")
    out["satisfaction"] = _clamp(sat)

    anx = hits(S.anx)
    a = sc(anx)
    if anx:
        ev.append(f"anxiety phrase: {anx[0][0]}")
    out["anxiety"] = _clamp(a)

    exc = hits(S.exc)
    e = sc(exc)
    if exc:
        ev.append(f"excitement phrase: {exc[0][0]}")
    if e > 0 and text.count("!") >= 1:
        e += 0.15
    if e > 0 and t.speech_rate_wps and t.speech_rate_wps > 3.5:
        e += 0.1
    out["excitement"] = _clamp(e)

    calm = hits(S.calm)
    c = 0.5 + 0.4 * sc(calm)
    if calm:
        ev.append(f"calm phrase: {calm[0][0]}")
    c -= 0.4 * base["frustration"] + 0.25 * out["anxiety"] + 0.2 * base["urgency"]
    if t.barge_in:
        c -= 0.1
    out["calmness"] = _clamp(c)

    tu, td = hits(S.trust_up), hits(S.trust_down)
    tr = 0.5 + 0.4 * sc(tu) - 0.5 * sc(td)
    if tu:
        ev.append(f"trust cue: {tu[0][0]}")
    if td:
        ev.append(f"distrust cue: {td[0][0]}")
    out["trust"] = _clamp(tr)

    hd, asr = hits(S.hedge), hits(S.assertive)
    ce = 0.5 - 0.45 * sc(hd) + 0.4 * sc(asr)
    if hd:
        ev.append(f"hedge: {hd[0][0]}")
    if asr:
        ev.append(f"assertive: {asr[0][0]}")
    if "..." in text or "…" in text:
        ce -= 0.1
    out["certainty"] = _clamp(ce)

    # cognitive load: needs time, many hesitations, asks to repeat, long silence before answering
    ld = hits(S.load)
    cl = sc(ld)
    if ld:
        ev.append(f"load phrase: {ld[0][0]}")
    hes = sum(1 for x in toks if x in HESITATION_TOKENS)
    if hes >= 2:
        cl += 0.1 * min(hes, 5)
        ev.append("many hesitations")
    elif hes == 1 and len(toks) <= 3:
        cl += 0.1
    if text.count("...") + text.count("…") >= 1:
        cl += 0.1
    if base["confusion"] >= 0.6:
        cl += 0.2
    lat = t.response_latency_ms
    if lat is not None and lat > 4000:
        cl += 0.15 if lat < 7000 else 0.3
        ev.append("long pause before answering")
    last_customer = next((u.text for u in reversed(session.history) if u.role == "customer"), None)
    if last_customer is not None and jaccard(text, last_customer) >= 0.6:
        cl += 0.15
    out["cognitive_load"] = _clamp(cl)

    # attention / participation / receptiveness
    q = "?" in text or (len(content) >= 3 and any(x in _qwords() for x in content))
    n = len(content)
    att = 0.5
    if minimal:
        att -= 0.2
    elif q or n >= 6:
        att += 0.3
    elif n >= 3:
        att += 0.1
    if lat is not None and lat > 6000:
        att -= 0.15
    if base["confusion"] >= 0.6 or out["cognitive_load"] >= 0.6:
        att -= 0.1
    if not content:
        att -= 0.2
    out["attention"] = _clamp(att)

    part = 0.5
    if minimal or not content:
        part = 0.2
    elif n >= 8:
        part = 0.8
    elif n >= 4:
        part = 0.65
    elif n <= 2:
        part = 0.4
    if q:
        part += 0.15
    if t.barge_in:
        part += 0.1
    out["participation"] = _clamp(part)

    ag, rf = hits(S.agree), hits(S.refuse)
    rc = 0.5 + 0.4 * sc(ag) - 0.5 * sc(rf) - 0.2 * base["frustration"] - 0.1 * out["anxiety"]
    if ag:
        ev.append(f"receptive: {ag[0][0]}")
    if rf:
        ev.append(f"rejecting: {rf[0][0]}")
    if minimal:
        rc -= 0.1
    if q and not rf:
        rc += 0.1
    out["receptiveness"] = _clamp(rc)
    return out


def _qwords():
    from .estimator import QUESTION_WORDS
    return QUESTION_WORDS


def state_dims(state) -> dict[str, float]:
    """All 14 numeric human-state dims of a HumanState (no signals), in a stable order."""
    from ..schemas import HumanState
    return {k: float(getattr(state, k)) for k in HumanState.model_fields if k != "signals"}
