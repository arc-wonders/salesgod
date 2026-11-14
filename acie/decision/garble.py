"""Is this turn garbled speech-to-text? Deterministic, conservative: when unsure it says no.

Signals (any one is enough): the NLU / fused TurnSignals say garbled, a low STT confidence (avg_logprob / no_speech, when the call record
carries them), or a very short fragment whose words are all outside a small lexicon of everyday words (Hindi, Hinglish and English)
and the call's product words. The agent never builds an answer, a pitch or a "you mean ..." guess on a garbled turn: it asks to repeat.
"""

from __future__ import annotations

import re

_TOK = re.compile(r"[ऀ-ॿ]+|[A-Za-z']+")
_Q = re.compile(r"\?|क्या|कैसे|कब|कहाँ|कहां|कौन|कितन|kya|kaise|kab|kahan|kaun|kitn|what|how|when|where|who|why|is |are |do |can ", re.I)

# everyday words that make a short fragment "meaningful" (it is then handled by the normal flow)
_LEX = set("""
हाँ हां नहीं नही जी ठीक ओके ok okay yes no haan han nahi nahin theek thik ji hello हेलो हैलो हलो hi bye बाय धन्यवाद thanks thank शुक्रिया
है हैं हूँ हूं था थी थे हो होगा होगी रहा रही रहे कर करो करना करें करूँ करूं दो दे देना देंगे दीजिए लो ले लेना लूँगा चाहिए चाहता चाहती चाहते
बोलो बोलिए बताइए बताओ बताना बता सुनो सुनिए सुन समझ समझा आया आना आओ जाना जाओ मिलेगा मिलता मिलती मिल
मैं मुझे मेरा मेरी मेरे हम हमें आप आपका आपकी आपके ये यह वो वह इस उस इसमें इसके उसके कुछ सब सारा बहुत थोड़ा ज्यादा ज़्यादा कम और या पर लेकिन तो भी ही
कितना कितने कितनी क्या कैसे कब कहाँ कहां कौन क्यों क्यूँ कीमत दाम पैसे रुपये रुपए फीस प्राइस ऑफर डिस्काउंट जिम मेंबरशिप ट्रेनर क्लास क्लासेस योगा
समय टाइम बाद अभी कल आज सुबह शाम रात बजे दिन महीने हफ़्ते हफ्ते साल घर काम
mujhe mera meri hum aap ye wo kya kaise kab kahan kaun kyun kitna kitne kitni hai hain hoon ho tha thi raha rahi rahe kar karo karna do dena le lena
chahiye bolo bataiye batao sunao suno samajh aaya aana milega milta price rate fees offer discount gym membership trainer class classes yoga zumba
time baad abhi kal aaj subah shaam raat baje din mahine hafte saal ghar kaam aur ya par lekin to bhi hi bas toh
the a an is are am was were be i you we me my your it this that these those and or but if so to of in on at for with from not do does did can could will would
please sorry what why how when where who which price cost much many call back later now today tomorrow money pay link send want need like know tell
बस रखो रख रखिए फ़ोन फोन bas rakho rakh phone fon mean hear audible sound voice line speak hindi english hmm hm hmmm acha achha accha अच्छा हम्म हम्मम umm uh refund upi cancel branch timing parking
""".split())


def tokens(text: str) -> list[str]:
    return _TOK.findall(text)


def stt_low(turn) -> bool:
    """Low recognition confidence from the call record (optional fields on TurnInput)."""
    lp = getattr(turn, "stt_avg_logprob", None)
    ns = getattr(turn, "stt_no_speech", None)
    return (lp is not None and lp < -1.0) or (ns is not None and ns > 0.6)


def looks_garbled(text: str, extra_words: set[str] | None = None, turn=None, signal: bool = False) -> bool:
    """`signal`: the NLU / TurnSignals already flagged it. `extra_words`: the product's own words (lower-case)."""
    if signal or (turn is not None and stt_low(turn)):
        return True
    toks = [t.lower() for t in tokens(text)]
    if not toks:
        return bool(text.strip()) and len(text.strip()) <= 12  # punctuation / digits noise only
    if len(toks) > 4 or _Q.search(text):
        return False
    if len(toks) == 1 and re.fullmatch(r"[a-z']+", toks[0]):
        return False  # one Latin word is a keyword (refund, upi, ...), not noise
    known = _LEX | (extra_words or set())
    hits = sum(1 for t in toks if t in known)
    return hits == 0 and len(toks) >= 1 and not (len(toks) == 1 and len(toks[0]) <= 2)
