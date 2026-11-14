"""Safety override that runs before any domain logic.

If a caller expresses suicidal thoughts or wanting to die, the agent stops whatever it was
doing (selling, collections, anything), responds with care, gives the national helpline and
the emergency number, flags the call for a human, and ends the call gently. The wording is a
fixed, reviewed template: the LLM never improvises here.

India: Tele-MANAS 14416 (Government of India mental health helpline, free, 24x7); emergency 112.
"""

from __future__ import annotations

import re

from .schemas import Fact, TurnPlan
from .text import normalize

_PATTERNS = [
    # Hindi (Devanagari, normalized: no nukta, anusvara)
    r"मरने का (मन|दिल|सोच)", r"मर जाना चाहत", r"मर जाऊं", r"मर जाउं", r"मरना चाहत", r"जीने का (मन|दिल) नहीं",
    r"जीना नहीं चाहत", r"आत्महत्या", r"खुदकुशी", r"जान दे (दूं|दूंगा|दूंगी|दू)", r"अपनी जान", r"खुद को (मार|खत्म)",
    r"सब खत्म कर (दूं|लूं)", r"जिंदगी से (तंग|परेशान) हो", r"मरना है", r"मर जाऊंगा", r"मर जाऊंगी",
    r"जीना नहीं", r"जी कर क्या", r"मौत", r"फांसी", r"सुसाइड",
    # Hinglish
    r"marne ka (man|mann|dil)", r"mar jana chah", r"mar jaun", r"jeene ka (man|mann|dil) nahi", r"jeena nahi chah",
    r"aatmahatya", r"atmahatya", r"khudkushi", r"jaan de (dunga|dungi|du)", r"khud ko (maar|khatam)",
    r"marna hai", r"mar jaunga", r"mar jaungi", r"suicide",
    # English
    r"\bkill myself\b", r"\bwant to die\b", r"\bwanna die\b", r"\bfeel like dying\b", r"\bend my life\b",
    r"\bsuicid", r"\bdon'?t want to live\b", r"\bno reason to live\b", r"\bhurt myself\b",
    # live call 2ccf3f1387 (14 Nov): "मेरा गोल है छत से कूदना" (ASR: "छट से कूदना") and "मैं जिंदगी से हारा हुआ हूँ
    # मुझे नहीं जीना" (ASR: "जिन्दगी ते खारा ... नहीं दीना") were NOT caught; the agent pitched the gym and then said
    # "आपका दिन शुभ हो". Word order, method phrases, and the ASR garbles seen on that call (only next to "जिंदगी")
    r"(नहीं|नही) (जीना|जीना है|जीना चाहत|जीने का)", r"(नहीं|नही) रहना (इस दुनिया|दुनिया)", r"दुनिया में (नहीं|नही) रहना", r"दुनिया छोड (दूं|दू|दूंगा|दूंगी|देना)",
    r"(जिंदगी|जिन्दगी|जिदगी) (से|ते)( मैं)? (हार|खार|तंग|परेशान|थक|ऊब)",
    r"(जिंदगी|जिन्दगी|जिदगी)\S*( \S+){0,6} (नहीं|नही) (जीना|दीना|जीणा)",
    r"(जिंदगी|जिन्दगी|जिदगी) (खत्म|बेकार|बोझ)",
    r"(छत|छट|बिल्डिंग|पुल|ब्रिज|टेरेस|बालकनी) (से|पर से) (कूद|छलांग|छलाग)",
    r"(ट्रेन|गाडी|गाड़ी) के (आगे|नीचे) (कूद|लेट|आ जा)", r"नस काट", r"(जहर|जेहर) खा", r"फंदा",
    r"nahi?n? jeena", r"jeena nahi", r"zindagi se (haar|tang|pareshan|thak)", r"(chhat|chat|roof|building|bridge|pul) se (kood|kud|jump)",
    r"(zeher|zehar|jahar) kha", r"nas kaat", r"duniya chhod",
    r"\bjump(ing)? off (the |a )?(roof|building|bridge|terrace|balcony)", r"\bend it all\b", r"\bno point (in )?living\b",
    r"\bdon'?t want to be alive\b", r"\btired of (living|life|being alive)\b", r"\bgive up on (life|living)\b",
]
_RX = re.compile("|".join(_PATTERNS), re.IGNORECASE)


def is_distress(text: str) -> bool:
    return bool(_RX.search(normalize(text))) or bool(_RX.search(text.lower()))


_SLOWLY = re.compile(r"number|नंबर|dhire|धीरे|slow|फिर से|phir se|again|repeat|likh|लिख|दोबारा|dobara", re.I)
_BYE = re.compile(r"bye|बाय|rakh|रख|chalta|चलता|chalti|चलती|thank|धन्यवाद|शुक्रिया|रखते|rakhte", re.I)


def distress_followup_plan(text: str = "") -> TurnPlan:
    """Any turn after a distress turn in the same call: no business, ever. Repeat the help - slowly when
    asked, and say a warm goodbye when they're leaving (the elderly persona got the same block 4 times)."""
    p = distress_plan()
    p.intent = "safety_followup"
    if _BYE.search(text):
        p.intent = "safety_goodbye"
        p.end_call = True
        p.fallback = {
            "en": "Please take care of yourself. Remember, Tele-MANAS 14416 is there any time. Goodbye.",
            "hi": "अपना ख़याल रखिए। याद रखिए, टेली-मानस 14416 पर कभी भी बात हो सकती है। नमस्ते।",
            "hinglish": "Apna khayal rakhiye. Yaad rakhiye, Tele-MANAS 14416 pe kabhi bhi baat ho sakti hai. Namaste.",
        }
        return p
    if re.search(r"लिख लिया|likh liya|noted|got it|note kar liya|नोट कर लिया", text, re.I):
        p.intent = "safety_noted"
        p.fallback = {
            "en": "Good. Please do call them. I'm glad you talked about it; please take care of yourself.",
            "hi": "बहुत अच्छा। उन्हें ज़रूर फ़ोन कीजिए। आपने बात की, ये अच्छा किया; अपना ख़याल रखिए।",
            "hinglish": "Bahut accha. Unhe zaroor phone kijiye. Aapne baat ki, ye accha kiya; apna khayal rakhiye.",
        }
        if _BYE.search(text):
            p.end_call = True
        return p
    if _SLOWLY.search(text):
        p.fallback = {
            "en": "Of course. Tele-MANAS: one, four, four, one, six. That's 1-4-4-1-6. It's free, any time. For an emergency: one, one, two.",
            "hi": "जी, लिखिए। टेली-मानस: एक, चार, चार, एक, छह। यानी एक-चार-चार-एक-छह। ये मुफ़्त है, कभी भी। इमरजेंसी में: एक, एक, दो।",
            "hinglish": "Ji, likhiye. Tele-MANAS: ek, chaar, chaar, ek, chhah. Yaani 1-4-4-1-6. Ye free hai, kabhi bhi. Emergency mein: ek, ek, do.",
        }
        return p
    p.fallback = {
        "en": "You matter, and you don't have to go through this alone. Tele-MANAS on {helpline} can talk with you right now, free. In an emergency call {emergency}. Please do call them, or someone close to you.",
        "hi": "आप अकेले नहीं हैं, आपकी बात मायने रखती है। टेली-मानस {helpline} पर अभी आपसे बात कर सकते हैं, मुफ़्त में। इमरजेंसी में {emergency} पर कॉल कीजिए। प्लीज़ उन्हें या अपने किसी अपने को अभी फ़ोन कीजिए।",
        "hinglish": "Please Tele-MANAS ko {helpline} pe call kijiye, wo abhi madad kar sakte hain. Emergency mein {emergency} pe call kijiye. Apna khayal rakhiye.",
    }
    return p


def distress_plan() -> TurnPlan:
    return TurnPlan(
        intent="safety_distress",
        objective="Respond with care, give the helpline and emergency number, stop the call's business",
        decision="safety: stop all selling, flag for human follow-up",
        facts={"helpline": Fact(value="14416", description="Tele-MANAS helpline"),
               "emergency": Fact(value="112", description="emergency number")},
        # the elderly persona review: hanging up right after this felt cold. Stay on the line; the call
        # ends when they say goodbye or go silent (the follow-up keeps repeating the help, never business).
        llm_allowed=False, end_call=False,
        fallback={
            "en": "I'm really sorry you're feeling this way, and I'm glad you said it. Please talk to someone who can "
                  "help right now: Tele-MANAS, {helpline}, it's free and open all day and night. If you're in danger, "
                  "please call {emergency}. I'm not going to talk about any offer. Is there someone with you right now, or someone you can call?",
            "hi": "मुझे बहुत दुख है कि आप ऐसा महसूस कर रहे हैं, और अच्छा किया कि आपने बताया। कृपया अभी किसी से बात "
                  "कीजिए जो मदद कर सकते हैं: टेली-मानस, {helpline}, ये मुफ़्त है और दिन-रात खुला है। अगर आप खतरे में हैं, "
                  "तो {emergency} पर कॉल कीजिए। मैं अब कोई ऑफर की बात नहीं करूँगा। क्या अभी आपके पास कोई है, या कोई अपना जिसे आप फ़ोन कर सकें?",
            "hinglish": "Mujhe bahut dukh hai ki aap aisa feel kar rahe hain, aur accha kiya ki aapne bataya. Please abhi "
                        "kisi se baat kijiye jo madad kar sakte hain: Tele-MANAS, {helpline}, ye free hai aur din-raat "
                        "khula hai. Agar aap khatre mein hain to {emergency} pe call kijiye. Main yahin rukta hoon. "
                        "Kya abhi aapke paas koi hai, ya koi apna jise aap phone kar sakein?",
        },
    )
