"""Safety override that runs before any domain logic.

If a caller expresses suicidal thoughts or wanting to die, the agent stops whatever it was
doing (selling, collections, anything), responds with care, gives the national helpline and
the emergency number, flags the call for a human, and ends the call gently. The wording is a
fixed, reviewed template: the LLM never improvises here.

India: Tele-MANAS 14416 (Government of India mental health helpline, free, 24x7); emergency 112.
"""

from __future__ import annotations

import os
import re


def enabled() -> bool:
    """Crisis/helpline guardrail switch. ON by default (owner's standing rule: a caller in distress stops all selling;
    live miss 2ccf3f1387 "मेरा गोल है छत से कूदना" got a gym pitch while it was off). The false trigger that had it switched
    off ("give me a good discount", 1e2a92c585) is fixed in skeletons()/_BENIGN_RAW. ACIE_SAFETY=0 turns it off."""
    return os.getenv("ACIE_SAFETY", "1") != "0"

from .schemas import Fact, TurnPlan
from .text import normalize, tokens

# ---------------------------------------------------------------------------------------------------------
# Detection. Deterministic (same text -> same score, always); no model, no randomness.
#
# Live call 2ccf3f1387: Whisper heard "मेरा गोल है छत से कूदना" as "...छट से कूदना" and "जिंदगी से हारा, मुझे नहीं
# जीना" as "जिन्दगी ते खारा हुआ हूँ मुझे नहीं दीना", and the old exact-spelling regex missed both. So every
# word is first reduced to a PHONETIC SKELETON that is the same for Devanagari and Roman spellings and
# survives typical ASR slips:
#   consonants -> classes  K (क ख ग घ ह k kh g gh h)   J (ज झ द ध ड ढ य j z d dh y)   S (च छ श ष स c ch s sh)
#                          T (त थ ट ठ)   P (प फ ब भ व f p b v w)   M   N   R (र ल)
#   vowels     -> I (इ ई ि ी i ee)  E (ए ऐ े ै e ai)  U (उ ऊ ो ौ u oo o)  ; the short "a" is dropped (the
#                 Devanagari inherent vowel is unwritten in Roman and vice versa), so मरना == marna == MRN
#   dropped    -> nukta, aspiration h, nasal before a consonant (जिन्दगी == zindagi), word-final nasal after
#                 I/E/U (नहीं == nahin == NK), word-final vowel (जीने == jeena == JIN)
# so छत/छट -> ST, जीना/दीना/जिना -> JIN, हारा/खारा -> KR, ज़िंदगी/जिन्दगी/zindagi -> JIJKI->JIJK, से/ते -> S/T.
# (देना "to give" stays JEN: e and ee are different classes, so "नहीं देना" is not "नहीं जीना".)
#
# Evidence is then built from word-window co-occurrence. Each piece of evidence has a GROUP and a WEIGHT
# (a documented, fixed probability-like score); within a group the max counts, across groups the scores
# combine as a noisy-or (1 - prod(1 - w)), so two independent worrying signals add up:
#   ideation   not wanting to live / wanting to die / self-harm phrases            0.55 - 0.95
#   method     jump from a height, train, wrist, poison, hanging, pills            0.55 alone, 0.85-0.9 with
#              a self-reference ("my goal is...", "I will...") within the window
#   hopeless   life + defeat words, "everything is over", no way out                 0.4 - 0.65
# HYPERBOLE guard: slang like "mar jaunga hans hans ke", "deadlift se jaan nikal gayi", "price se mar gaya" only
# lowers the entries marked hyper (x0.25) and only when a gym/price/laughing word is in the same turn.
# Explicit suicide words, methods and "नहीं जीना" are never lowered.
# Thresholds: score >= DISTRESS_T (0.75) is a crisis; CONCERN_T (0.45) <= score < DISTRESS_T gets a gentle
# check-in (concern_plan) instead of selling. Evidence from the previous 2 customer turns counts (x0.95,
# x0.85) when the current turn itself carries some signal (>= 0.30); cross-turn word windows work through
# the concatenated tokens ("मेरा गोल है" / "छत से कूदना").
# Wrong in the safe direction is the design: a false check-in costs one sentence, a miss costs a life.
# ---------------------------------------------------------------------------------------------------------

DISTRESS_T = 0.75
CONCERN_T = 0.45

_DEV = {}
for _chars, _cls in (("कखगघहङ", "K"), ("जझदधडढय", "J"), ("चछशषस", "S"), ("तथटठ", "T"), ("पफबभव", "P"),
                     ("म", "M"), ("नणञ", "N"), ("रलळ", "R")):
    for _c in _chars:
        _DEV[_c] = _cls
_DEV_VOWEL = {"अ": "a", "आ": "a", "इ": "I", "ई": "I", "उ": "U", "ऊ": "U", "ए": "E", "ऐ": "E", "ओ": "U", "औ": "U",
              "ऑ": "a", "ा": "a", "ि": "I", "ी": "I", "ु": "U", "ू": "U", "े": "E", "ै": "E", "ो": "U", "ौ": "U",
              "ॉ": "a", "ृ": "I"}
_ROM_C = {"k": "K", "g": "K", "q": "K", "h": "K", "j": "J", "z": "J", "d": "J", "y": "J", "c": "S", "s": "S",
          "t": "T", "p": "P", "b": "P", "f": "P", "v": "P", "w": "P", "m": "M", "n": "N", "r": "R", "l": "R",
          "x": "K", "ṭ": "T"}
_ROM_V = {"a": "a", "i": "I", "e": "E", "o": "U", "u": "U"}
_CONS = set(_DEV) | set(_ROM_C)


def _skel_word(w: str) -> str:
    """Phonetic skeleton of ONE (already normalized) word. See the block comment above."""
    out: list[str] = []
    if w.isascii():
        w = w.replace("ee", "i").replace("ii", "i").replace("oo", "u").replace("ie", "i")
    n = len(w)
    i = 0
    while i < n:
        ch = w[i]
        nxt = w[i + 1] if i + 1 < n else ""
        if ch in _DEV:
            cls = _DEV[ch]
            # nasal + virama + consonant (जिन्दगी): the nasal is not pronounced separately
            if cls in "NM" and nxt == "्" and i + 2 < n and w[i + 2] in _DEV:
                i += 2
                continue
            out.append(cls)
        elif ch in _DEV_VOWEL:
            out.append(_DEV_VOWEL[ch])
        elif ch in ("ं", "ँ", "ः", "्", "ॅ"):
            pass
        elif ch in _ROM_C:
            cls = _ROM_C[ch]
            prev = w[i - 1] if i else ""
            if ch == "h" and prev and prev in _ROM_C and prev not in "aeiou" and out:
                i += 1  # aspiration: kh gh ch sh th dh ph bh jh chh
                continue
            if ch in "nm" and nxt in _ROM_C and nxt not in "aeiou" and i > 0:
                i += 1  # zindagi: n before a consonant is not pronounced
                continue
            if ch == "c" and nxt == "k":
                cls = "K"
            out.append(cls)
        elif ch in _ROM_V:
            out.append(_ROM_V[ch])
        # digits / anything else: ignored
        i += 1
    # roman 'ai'/'ei'/'ay' -> E ; 'aa','ee','ii','oo' collapse by the dedupe below
    s: list[str] = []
    k = 0
    while k < len(out):
        if out[k] == "a" and k + 1 < len(out) and out[k + 1] == "I" and k + 1 < len(out):
            # "ai" (roman) is the vowel E; a Devanagari ा + ि sequence is rare, same reading is fine
            s.append("E")
            k += 2
            continue
        s.append(out[k])
        k += 1
    # roman "ee"/"ii" are one long I, but the Devanagari/roman "ie" etc. stay as they are
    dedup: list[str] = []
    for c in s:
        if not dedup or dedup[-1] != c:
            dedup.append(c)
    # word-final nasal directly after I/E/U (नहीं, nahin, main, hoon, jaun, train)
    if len(dedup) >= 3 and dedup[-1] == "N" and dedup[-2] in "IEU":
        dedup.pop()
    elif len(dedup) >= 2 and dedup[-1] == "N" and dedup[-2] in "IEU":
        dedup.pop()
    dedup = [c for c in dedup if c != "a"]
    if len(dedup) >= 2 and dedup[-1] in "IEU":
        dedup.pop()  # word-final vowel is noise
    sk = "".join(dedup)
    sk = re.sub(r"(.)\1+", r"\1", sk)
    return sk or "-"


def skeletons(text: str) -> list[str]:
    """Phonetic skeletons. A Latin-script word only gets one when it is a romanized Hindi/Hinglish word this module
    knows (any concept word list): ordinary English words collide with Hindi skeletons ("good" = KUJ = kud/jump,
    "for" = PUR = pul/bridge, owner's live call 15 Nov: "can't you just give me a good discount for me?" fired the
    crisis helpline). English distress is caught by the exact English patterns (_EN, legacy phrases) instead."""
    return [_skel_word(t) if (not t.isascii() or t in _ROMAN) else "~" + t for t in tokens(text)]


_ROMAN: set[str] = set()  # every romanized word any concept below was built from (filled by _set / _roman)


def _roman(words: str) -> list[str]:
    ws = [normalize(w) for w in words.split()]
    _ROMAN.update(w for w in ws if w.isascii())
    return ws


def _set(words: str) -> frozenset[str]:
    return frozenset(_skel_word(w) for w in _roman(words))


# --- single-word concepts (written in Roman; the Devanagari spelling reaches the same skeleton) -----------
_SELF = _set("main mai me mein mujhe mujhko mujhse mujhme mera meri mere merko meko mko apna apni apne khud i my "
             "myself mere hum hume")
_NEG = _set("nahi nahin nhi nai nahee na mat no not dont never cant")
_LIVE = _set("jeena jeene jina jine jeenaa jeeni jiyna jeenea")  # jeena / जीना / जिना / दीना (garbled)
_JUMP = _set("kud kuda kudna kudne kudu kudun kudoon kudke kudunga kudungi kood koodna koodne koodun koodunga "
             "kuddna kuuda")
_JUMP_RAW = frozenset(("jump", "jumps", "jumping", "jumped", "जंप", "जम्प"))
_PLACE = _set("chhat chhath terrace balcony balkani manzil mazil manjil pahad pahadi nadi naher samundar talab "
              "flyover overbridge ghati khidki pul")
_PLACE_RAW = frozenset(normalize(w) for w in (
    "chat roof rooftop building bildin bilding bridge cliff tower river sea window terrace "
    "बिल्डिंग ब्रिज रूफ रूफटॉप टैरेस बालकनी टावर नदी").split())
_roman("aage age kar kr kya kia jeekar jee ji jii jike jiye jiyo")  # used as literal skeletons in _evidence
_TRAIN = _set("trein patri")
_TRAIN_RAW = frozenset(normalize(w) for w in "train trains railway track metro ट्रेन रेलवे पटरी मेट्रो".split())
_DIEV = _set("marna marne maarna maarne")
_MAR = _set("mar maar")
_JA = _set("ja jao jau jaun jaoon jaunga jaungi jaye jaaye jayenge jaana jaane jana jane jaoonga jaungaa")
_WANT = _set("chahta chahti chahte chahiye chahun chahu chaahta chaahti man mann dil mood soch sochta sochti "
             "sochte ichha irada iraada plan goal ")
_ZINDAGI = _set("zindagi jindagi jindgi zindgi zindagee jeevan jivan life lyf lif")
_HOPE_W = {  # defeat words near "life": skeleton -> weight
    **{_skel_word(w): 0.78 for w in _roman("haar haara hara hare haari tang pareshan pareshaan bezaar bekar nafrat")},
    **{_skel_word(w): 0.55 for w in _roman("thak thaka thaki thakk")},
    **{_skel_word(w): 0.55 for w in _roman("bojh bhoj tut toot tuta toota")},
}
_KHATAM = _set("khatam khatm khtm")
_SAB = _set("sab sabkuch sabhi")
_KHUD = _set("khud")
_CUT = _set("kaat kaatna katna kat katlu kaatunga kaat lunga")
_NAS = _set("nas nass")
_POISON = _set("zeher zehar jahar jehar zahar poison vish")
_POISON_V = _set("khana khale khalun khaunga khakar pee peena pina pi piunga pikar pee lunga pi lunga")
_FANSI_RAW = frozenset(normalize(w) for w in "fansi phansi phaansi fanda phanda फांसी फाँसी फंदा फांसि".split())
_LATAK = _set("latak latakna latakke latkar latka")
_PANKHA = _set("pankha pankhe pankhey")
_NEEND = _set("neend nind nid")
_SUICIDE = _set("suicide susaid sucide suiside suiciide सुसाइड सुसाईड सुइसाइड aatmahatya atmahatya "
                "aatmahatyaa आत्महत्या khudkushi khudkhushi khudkusi khudkashi खुदकुशी")
_GOLI = _set("goli goliyan goliya goliyaan golian goliyo")
_GOLI_BIG = _set("saari sari sab bahut bohot zyada ek saath overdose ")
_GALA = _set("gala gale")
_HYPER_RAW = frozenset(normalize(w) for w in (
    "hans hansi hanste hasna hasi haste hanskar mazak mazaak joke jokes lol haha hehe funny comedy price rate kimat "
    "qeemat daam mehnga mehngi mehngai discount offer bill fees deadlift deadlifts squat squats workout exercise gym "
    "cardio treadmill reps mehnat thakan pasina garmi bhookh bhukh pyaas traffic "
    "हंस हँस हंसी हँसी हंसते हंसकर मजाक मज़ाक जोक लोल प्राइस रेट कीमत दाम महंगा महंगाई डिस्काउंट ऑफर बिल फीस डेडलिफ्ट "
    "स्क्वाट वर्कआउट एक्सरसाइज जिम कार्डियो ट्रेडमिल मेहनत थकान पसीना गर्मी भूख प्यास ट्रैफिक").split())
_FILLER = _set("hi to bhi ho hai he ki ka ke ko")  # tolerated between "mar" and "jaun"


# Everyday words (mostly English loanwords written in Devanagari, plus bol/khud/bas) whose phonetic skeleton collides with a
# method / place word: ऑफ़िस = PIS = vish (poison), बोल = PUR = pul (bridge), खुद = KUJ = kud (jump). They never carry a method
# concept (gate run 4: "ऑफ़िस से आकर" was a poison concern; "खुद आकर देख लूंगा" a jump). "khud" still counts for "khud ko ...".
_BENIGN_RAW = frozenset(normalize(w) for w in (
    "ऑफ़िस ऑफिस ऑफीस आफिस ऑफ़र ऑफर office offis offer ofis पॉलिसी पालिसी policy सर्विस सर्विसेज service services बस bus फ़ीस फीस fees fee "
    "प्लान plan पैकेज package पास pass बैच batch बोल बोलो बोला बोली बोलना bol bolo bola boli bolna बात baat बाद baad बार bar baar "
    "खुद khud sector सेक्टर सिस्टम system बिल bill ट्रायल trial फ्री free पोज़िशन position एक्सेस access कार्ड card कैश cash "
    "वीज़ा visa ऐप app ऐपल apple स्कूल school पूल pool फील feel फ़ील मॉल mall सेशन session लेसन lesson मिल mil पल pal "
    "पुल्स पुश push पुश-अप पीएफ pf").split()) | frozenset((
    # everyday ENGLISH words whose phonetic skeletons collide with Hindi method words (kud/kood = jump, pul = bridge,
    # zehar...): live call 1e2a92c585 "So can't you just give me a GOOD discount FOR me?" -> KUJ~kud + PUR~pul -> a false
    # helpline. English method words are matched raw (_JUMP_RAW etc.), never through these skeletons.
    "good goods god gold could cool coat code card cards kid kids cut cute cat get got give gave go goes going gone "
    "for four far fur fair fare fear pure poor pour pull pulled full fool fall fell fill feel few five fine phone "
    "put pot but bit bad bed bud bat boot book look like lake take make made need deal deals discount discounts price "
    "please plus pay paid per day days rate rates offer offers better best lower less cheap cheaper cost costs "
    "can cant can't just you your me my mine for from with this that then than there they them what when where why how "
    "sure yes yeah okay ok well will would should shall do does did done").split())
_METHOD_SKEL_CONCEPTS_NEED_RAW = True


def _scrub_benign(tok: list[str], raw: list[str]) -> list[str]:
    """Skeletons with the benign everyday words blanked out, for the method concepts (poison / jump / place / train)."""
    return ["-" if (i < len(raw) and raw[i] in _BENIGN_RAW) else t for i, t in enumerate(tok)]


def _poison_verb_ok(raw_w: str, sk: str) -> bool:
    """A 1-2 consonant verb skeleton (KR = खाकर, P = पी) is only the poison verb when the word really starts like it."""
    if len(sk) > 2:
        return True
    return raw_w[:1] in ("kh", "खा", "ख", "p", "प") or raw_w.startswith(("kh", "pi", "pe", "pee", "ख", "प"))


def _idx(tok: list[str], conc) -> list[int]:
    return [i for i, t in enumerate(tok) if t in conc]


def _near(a: list[int], b: list[int], window: int, ordered: bool = False) -> bool:
    for i in a:
        for j in b:
            if i != j and (j - i if ordered else abs(j - i)) <= window and (not ordered or j > i):
                return True
    return False


_EN = [  # (regex over the normalized text, weight, group, hyper_capable)
    (r"\bkill(ing)? (my ?self)\b", 0.95, "ideation", False),
    (r"\bkill me\b", 0.5, "ideation", True),
    (r"\bend(ing)? (my|this) life\b", 0.95, "ideation", False),
    (r"\bend (it all|everything)\b", 0.85, "ideation", False),
    (r"\btake my (own )?life\b", 0.95, "ideation", False),
    (r"\bsuicid\w*|\bsucide\b|\bsuside\b|\bsuiside\b", 0.95, "ideation", False),
    (r"\bwant(ed|s)? to die\b|\bwanna die\b|\bwant to be dead\b|\bwish (i|id) (was|were|be) dead\b|\bbetter off dead\b"
     r"|\bbetter off without me\b|\brather be dead\b|\bwish i was never born\b", 0.95, "ideation", False),
    (r"\b(dont|do not|didnt|no longer) want to (live|be alive|be here|exist|go on)\b", 0.95, "ideation", False),
    (r"\bno (reason|point|use) (in |to |of )?(living|live|life|going on|being alive)\b|\bnot worth living\b"
     r"|\bnothing to live for\b|\blife (is|isnt|is not) (worth|worthless)\b", 0.9, "ideation", False),
    (r"\b(cant|can not|cannot) (go on|do this anymore|take it anymore|take this anymore)\b", 0.6, "ideation", False),
    (r"\bhang(ing)? my ?self\b|\bcut(ting)? my (wrist|wrists|self)\b|\bslit my\b|\bhurt(ing)? my ?self\b"
     r"|\bharm(ing)? my ?self\b|\boverdose\b", 0.9, "method", False),
    (r"\bjump(ing)? (off|from|out of) (the |a |my |this )?(roof|rooftop|building|bridge|terrace|balcony|window|cliff|"
     r"tower)\b", 0.85, "method", False),
    (r"\b(in front of|under) (a |the )?train\b", 0.6, "method", False),
    (r"\bhopeless\b|\bworthless\b|\bno way out\b|\btired of (life|living)\b|\bsick of (life|living)\b"
     r"|\bgive up on life\b|\blife is over\b|\bnobody (would )?(care|miss me)\b", 0.5, "hopeless", False),
]
_EN_RX = [(re.compile(p), w, g, h) for p, w, g, h in _EN]
_LEGACY_PATTERNS = [  # exact-phrase net from the 14 Nov hotfix (merged in); kept as a floor under the phonetic engine
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
_LEGACY_RX = re.compile("|".join(_LEGACY_PATTERNS), re.IGNORECASE)
_EN_NEG = re.compile(r"\b(dont|do not|didnt|never|not|no)\b")


def _evidence(tok: list[str], turn: list[int], hyper: set[int], raw: list[str]) -> list[tuple[str, float, int, str]]:
    """-> [(group, weight, turn_index_of_last_word, rule)] ; tok = skeletons, turn = turn index per token."""
    ev: list[tuple[str, float, int, str]] = []
    def add(group, w, pos, rule, hyper_ok=False):
        t = turn[min(pos, len(turn) - 1)] if turn else 0
        if hyper_ok and t in hyper:
            w *= 0.25
        ev.append((group, w, t, rule))

    SELF, NEG, LIVE = _idx(tok, _SELF), _idx(tok, _NEG), _idx(tok, _LIVE)
    # 1. not wanting to live: "नहीं जीना", "जीना नहीं", "जीने का मन नहीं", "मुझे नहीं दीना" (window 4)
    for i in LIVE:
        for j in NEG:
            if abs(i - j) <= 4:
                near_self = any(abs(i - s) <= 6 for s in SELF)
                add("ideation", 0.95 if near_self else 0.85, max(i, j), "not_live")
                break
    # 2. dying: marna/marne + wanting; "mar jaun"; die-verb with self
    DIEV, MAR = _idx(tok, _DIEV), _idx(tok, _MAR)
    for i in DIEV:
        negd = any(0 < j - i <= 3 for j in NEG)
        if negd:
            continue
        wants = any(abs(i - j) <= 4 for j in _idx(tok, _WANT))
        near_self = any(abs(i - s) <= 4 for s in SELF)
        if wants:
            add("ideation", 0.92, i, "die_want", True)
        elif near_self:
            add("ideation", 0.8, i, "die_self", True)
        else:
            add("ideation", 0.35, i, "die_bare", True)
    for i in MAR:
        for k in (1, 2):
            if i + k < len(tok) and tok[i + k] in _JA and all(tok[i + x] in _FILLER for x in range(1, k)):
                if any(0 < j - (i + k) <= 3 for j in NEG):
                    break
                add("ideation", 0.8, i + k, "mar_jaun", True)
                break
    for i in _idx(tok, _SUICIDE):
        if not any(0 < j - i <= 3 for j in NEG):
            add("ideation", 0.95, i, "suicide_word")
    for i, r in enumerate(raw):  # "जान दे दूंगा": hyperbole-capable
        if r in ("jaan", "जान") and any(0 < j - i <= 3 and raw[j] in ("dunga", "dungi", "du", "दूंगा", "दूंगी", "दूं", "दू") for j in range(len(raw))):
            if any(abs(s - i) <= 5 for s in SELF):
                add("ideation", 0.78, i, "jaan_de", True)
    # 3. hopelessness around life: "जिंदगी से हारा / थक / तंग"
    ZIN = _idx(tok, _ZINDAGI)
    for i in ZIN:
        for j, t in enumerate(tok):
            if t in _HOPE_W and 0 < abs(i - j) <= 4:
                near_self = any(abs(i - s) <= 6 for s in SELF)
                add("hopeless", min(0.9, _HOPE_W[t] + (0.1 if near_self else 0)), max(i, j), "life_defeat", True)
        for j in _idx(tok, _KHATAM):
            if abs(i - j) <= 3:
                near_self = any(abs(i - s) <= 6 for s in SELF)
                add("ideation", 0.9 if near_self else 0.8, max(i, j), "life_khatam", True)
    for j in _idx(tok, _KHATAM):
        if any(0 < j - s <= 2 for s in _idx(tok, _SAB)):
            add("hopeless", 0.45, j, "sab_khatam", True)
        if any(0 < j - s <= 4 for s in _idx(tok, _KHUD)):
            add("ideation", 0.9, j, "khud_ko_khatam", True)
    for i in _idx(tok, _KHUD):
        for j in MAR + _idx(tok, _DIEV):
            if 0 < j - i <= 3:
                add("ideation", 0.9, j, "khud_ko_maar", True)
    # 4. methods (matched on the scrubbed skeletons, see _BENIGN_RAW)
    tok_all, tok = tok, _scrub_benign(tok, raw)
    JUMP = sorted(set(_idx(tok, _JUMP) + [i for i, r in enumerate(raw) if r in _JUMP_RAW]))
    TRAIN = sorted(set(_idx(tok, _TRAIN) + [i for i, r in enumerate(raw) if r in _TRAIN_RAW]))
    PLACE = sorted(set(_idx(tok, _PLACE) + [i for i, r in enumerate(raw) if r in _PLACE_RAW] + TRAIN))
    for i in JUMP:
        for j in PLACE:
            if abs(i - j) <= 5 and i != j:
                near_self = any(min(abs(s - i), abs(s - j)) <= 6 for s in SELF)
                wants = any(abs(w - i) <= 5 for w in _idx(tok, _WANT))
                add("method", 0.9 if (near_self or wants) else 0.8, max(i, j), "jump_place")
                break
    for i in TRAIN:
        for j, t in enumerate(tok):
            if t in _skel_word("aage") and 0 < j - i <= 2:
                near_self = any(abs(s - i) <= 6 for s in SELF)
                add("method", 0.85 if near_self else 0.4, j, "train_aage")
    for i in _idx(tok, _NAS):
        if any(0 < abs(i - j) <= 2 for j in _idx(tok, _CUT)):
            near_self = any(abs(s - i) <= 5 for s in SELF)
            add("method", 0.9 if near_self else 0.7, i, "nas_kaatna")
    pv = [j for j in _idx(tok, _POISON_V) if j < len(raw) and _poison_verb_ok(raw[j], tok[j])]
    for i in [i for i in _idx(tok, _POISON) if not raw[i].endswith(("कर", "के", "kar", "ke", "kr"))]:  # जाकर / आकर are not जहर
        if any(0 < j - i <= 3 or 0 < i - j <= 2 for j in pv):
            near_self = any(abs(s - i) <= 5 for s in SELF)
            add("method", 0.9 if near_self else 0.7, i, "poison")
    for i in [i for i, r in enumerate(raw) if r in _FANSI_RAW]:
        near_self = any(abs(s - i) <= 5 for s in SELF)
        add("method", 0.9 if near_self else 0.4, i, "fansi")
    for i in _idx(tok, _LATAK):
        near_self = any(abs(s - i) <= 5 for s in SELF)
        pk = any(abs(i - j) <= 3 for j in _idx(tok, _PANKHA))
        add("method", 0.9 if (near_self or pk) else 0.5, i, "latak")
    for i in _idx(tok, _NEEND):
        if any(0 < j - i <= 3 for j in _idx(tok, _GOLI)):
            big = any(abs(j - i) <= 6 for j in _idx(tok, _GOLI_BIG))
            near_self = any(abs(s - i) <= 6 for s in SELF)
            add("method", 0.85 if big else (0.6 if near_self else 0.35), i, "sleeping_pills")
    tok = tok_all
    # 5. "जी कर क्या" (what is the use of living)
    kr, kj = _skel_word("kar"), _skel_word("kya")
    for i in range(len(tok) - 2):
        if (tok[i] == _skel_word("jeekar") and tok[i + 1] == kj) or (tok[i] == "J" and tok[i + 1] == kr and tok[i + 2] == kj):
            add("ideation", 0.8 if any(abs(s - i) <= 6 for s in SELF) else 0.7, i + 1, "jee_kar_kya")
    return ev


def _english_evidence(norm_text: str, turn_idx: int) -> list[tuple[str, float, int, str]]:
    ev = []
    for rx, w, g, h in _EN_RX:
        for m in rx.finditer(norm_text):
            before = norm_text[max(0, m.start() - 14):m.start()]
            if g == "ideation" and _EN_NEG.search(before) and not rx.pattern.startswith(r"\b(dont"):
                continue  # "dont want to die", "never wanted to kill myself"
            if h and re.search(r"\b(workout|gym|traffic|heat|price|laughing|joke|jokes|lol|haha)\b", norm_text):
                w *= 0.25
            ev.append((g, w, turn_idx, "en:" + rx.pattern[:24]))
    return ev


def _combine(ev: list[tuple[str, float, int, str]], cur: int) -> tuple[float, list[str]]:
    """Group max, then noisy-or across groups. Earlier turns are decayed and only count when the current
    turn carries a signal of its own (>= 0.30)."""
    decay = {0: 1.0, 1: 0.95, 2: 0.85}
    scored = [(g, w * decay.get(cur - t, 0.0), t, r) for g, w, t, r in ev]
    current = [e for e in scored if e[2] == cur]
    if not current or max(w for _, w, _, _ in current) < 0.30:
        scored = current
    best: dict[str, tuple[float, str]] = {}
    for g, w, t, r in scored:
        label = r if t == cur else f"{r}@-{cur - t}"
        if w > best.get(g, (0.0, ""))[0]:
            best[g] = (w, label)
    prod = 1.0
    for w, _ in best.values():
        prod *= 1.0 - w
    return round(1.0 - prod, 3), [f"{g}:{lab}={w:.2f}" for g, (w, lab) in sorted(best.items())]


_LD_RAW = frozenset(normalize(w) for w in "maut death die dead dying life zindagi jindagi jeevan mar marna mara मौत मरना जिंदगी जिन्दगी जीना जीने जीवन".split())


def _voice_low(voice) -> bool:
    label = voice.get("label") if isinstance(voice, dict) else getattr(voice, "label", None)
    return str(label or "").lower() in ("low", "sad")


def _life_death_words(tok, raw, turn, cur) -> bool:
    return any(turn[i] == cur and (r in _LD_RAW or tok[i] in _ZINDAGI or tok[i] in _LIVE or tok[i] in _DIEV)
               for i, r in enumerate(raw))


def _customer_texts(history) -> list[str]:
    out: list[str] = []
    for h in history or []:
        if isinstance(h, str):
            out.append(h)
        elif isinstance(h, (tuple, list)) and len(h) == 2:
            if str(h[0]) in ("customer", "user", "caller"):
                out.append(str(h[1]))
        else:
            role, txt = getattr(h, "role", "customer"), getattr(h, "text", "")
            if role == "customer":
                out.append(txt)
    return out[-2:]


def distress_score(text: str, history=None, voice=None) -> tuple[float, list[str]]:
    """(score 0..1, evidence). history = previous turns (strings, or (role, text) pairs, or Utterances);
    only the last 2 CUSTOMER turns are used. voice = the prosody reading of THIS turn (label 'low'/'sad'): with
    life/death words in the turn it adds a 0.5 concern signal (a flat, heavy voice says "jeena" differently
    from a joke). Deterministic."""
    turns = _customer_texts(history) + [text or ""]
    cur = len(turns) - 1
    tok: list[str] = []
    raw: list[str] = []
    turn: list[int] = []
    ev: list[tuple[str, float, int, str]] = []
    hyper: set[int] = set()
    for ti, t in enumerate(turns):
        if any(w in _HYPER_RAW for w in tokens(t)):
            hyper.add(ti)
        sk = skeletons(t)
        if ti:  # turn boundary: a little distance so windows do not glue unrelated turns tightly
            tok += ["-", "-"]
            raw += ["-", "-"]
            turn += [ti, ti]
        tok += sk
        raw += tokens(t)
        turn += [ti] * len(sk)
        ev += _english_evidence(normalize(t), ti)
    ev += _evidence(tok, turn, hyper, raw)
    cur_text = turns[cur]
    ntext = normalize(cur_text)
    for m in list(_LEGACY_RX.finditer(ntext)) + list(_LEGACY_RX.finditer(cur_text.lower())):
        if _EN_NEG.search((ntext[max(0, m.start() - 14):m.start()])) and m.group(0).isascii():
            continue  # "dont want to die"
        ev.append(("ideation", 0.9 * (0.25 if cur in hyper else 1.0), cur, "legacy_phrase"))
        break
    if voice is not None and _voice_low(voice) and _life_death_words(tok, raw, turn, cur):
        ev.append(("voice", 0.5, cur, "voice_low_life_death"))
    # dedupe identical evidence found twice
    seen, uniq = set(), []
    for e in ev:
        if e not in seen:
            seen.add(e)
            uniq.append(e)
    return _combine(uniq, cur)


def is_distress(text: str, history=None, voice=None) -> bool:
    return distress_score(text, history, voice)[0] >= DISTRESS_T


def is_concern(text: str, history=None, voice=None) -> bool:
    """Worrying but not clear enough to call a crisis: check in gently instead of selling."""
    return CONCERN_T <= distress_score(text, history, voice)[0] < DISTRESS_T


def crisis_from_nlu(understanding) -> bool:
    """Second net: the NLU model flagged a possible crisis. The pipeline ORs this in; it can only ADD a
    safety trigger, never suppress one."""
    return bool(getattr(understanding, "crisis", False))


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


def concern_plan() -> TurnPlan:
    """Middle band (CONCERN_T <= score < DISTRESS_T): a gentle check-in instead of selling. Fixed, reviewed
    wording, no LLM. The call stays open; a clearer answer next turn moves to distress_plan."""
    return TurnPlan(
        intent="safety_concern",
        objective="Gently check in on a worrying remark; do not sell this turn",
        decision="safety: pause selling, check in, offer Tele-MANAS",
        facts={"helpline": Fact(value="14416", description="Tele-MANAS helpline")},
        llm_allowed=False, end_call=False,
        fallback={
            "en": "Are you okay? If your heart feels heavy, you can talk to Tele-MANAS on {helpline} any time, it's free. "
                  "I'm here and listening, there's no hurry about anything else.",
            "hi": "क्या आप ठीक हैं? अगर मन भारी है तो Tele-MANAS {helpline} पर कभी भी बात कर सकते हैं, ये मुफ़्त है। "
                  "मैं यहीं हूँ, और किसी बात की जल्दी नहीं है।",
            "hinglish": "Kya aap theek hain? Agar mann bhaari hai to Tele-MANAS {helpline} pe kabhi bhi baat kar sakte hain, "
                        "ye free hai. Main yahin hoon, aur kisi baat ki jaldi nahi hai.",
        },
    )
