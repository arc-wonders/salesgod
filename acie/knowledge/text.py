"""Deterministic text normalisation for retrieval: tokens, synonym canonicalisation, a cross-script phonetic skeleton
(Devanagari <-> Roman, so "पावरलिफ्टिंग" and "powerlifting" share a key) and crc32-hashed char n-grams.
No hash(), no randomness, no external deps."""

from __future__ import annotations

import re
import unicodedata
import zlib

STOP = frozenset("""a an the is are am was were be been do does did of for to in on at and or i me my you your we our us it its this
that these those with can could will would should shall please tell about what how which who whom when where why there any have
has had get got want need like just also so if as by from up out into then than very much more some
hai hain ho hu hun hoon tha thi the ka ki ke ko se me mein mai main par pe liye lie kya kaise kab kyun kyu kaun kahan kahaan
aur ya bhi to toh ye yeh wo woh na nahi nhi mujhe mera meri mere hume humein hamein aap aapka aapke aapki apna apne ji jee
bata batao batana bataiye boliye bolo sakta sakte sakti kar karna karte karti kare karo hota hote hoti hoga hogi honge
think around actually come number recommend suggest really one two three four five six seven eight nine ten maybe majorly
raha rahe rahi wala wali wale vala vali vale bhai sir madam ek koi kuch isme isse iska iski uska
है हैं का की के को से में मैं पर पे लिए क्या कैसे कब क्यों कौन कहाँ कहां और या भी तो ये यह वो वह नहीं मुझे मेरा मेरी मेरे आप आपका
आपके आपकी हम हमें एक कोई कुछ इसमें इससे इसका इसकी उसका बताओ बताइए बताइये बताना सकता सकते सकती कर करना करते करती करें करो होता होते
होती होगा होगी रहा रहे रही वाला वाली वाले जी था थी थे""".split())

# canonical concept -> surface forms (any script). Both docs and queries are mapped through this table.
SYN: dict[str, list[str]] = {
    "price": ["cost", "costs", "fee", "fees", "charge", "charges", "rate", "rates", "pricing", "kitna", "kitne", "kitni", "paisa",
              "paise", "kimat", "keemat", "daam", "dam", "कीमत", "दाम", "फीस", "फ़ीस", "चार्ज", "कितना", "कितने", "कितनी", "पैसा", "पैसे",
              "रेट", "प्राइस", "expensive", "costly", "mehenga", "mehnga", "महंगा", "महँगा"],
    "powerlifting": ["powerlifting", "powerlifter", "powerlifters", "पावरलिफ्टिंग", "पावर", "पॉवरलिफ्टिंग", "पावरलिफ्टर", "powerlift"],
    "weightlifting": ["weightlifting", "वेटलिफ्टिंग"],
    "competition": ["competition", "competitions", "compete", "competing", "competitive", "meet", "meets", "tournament", "tournaments",
                    "contest", "championship", "championships", "event", "pratiyogita", "प्रतियोगिता", "कॉम्पिटिशन", "कम्पटीशन", "कंपटीशन", "टूर्नामेंट",
                    "मुकाबला", "muqabla", "mukabla", "nationals", "state-level", "stage"],
    "bench": ["bench", "benchpress", "बेंच", "बेन्च"],
    "squat": ["squat", "squats", "स्क्वाट", "स्क्वैट"],
    "deadlift": ["deadlift", "deadlifts", "डेडलिफ्ट"],
    "muscle": ["muscle", "muscles", "mass", "bulk", "bulking", "bodybuilding", "bodybuilder", "gain", "gains", "size", "massle", "masl",
               "मसल", "मसल्स", "मांसपेशी", "बॉडी", "body", "bodybuild", "बॉडीबिल्डिंग", "strength", "stronger", "strong", "ताकत", "taqat", "takat", "taakat"],
    "weightloss": ["fat", "slim", "slimming", "obese", "obesity", "thin", "patla", "motapa", "motapa", "mota", "belly", "tummy", "pet",
                   "पेट", "मोटापा", "मोटा", "वजन", "वज़न", "वेट", "weight", "kilos", "kgs", "lose", "losing", "kam", "कम",
                   "cut", "cutting", "diet", "dieting", "shred", "fatloss", "calorie", "calories"],
    "beginner": ["beginner", "beginners", "newbie", "new", "first", "fresh", "starter", "starting", "start", "never", "shuruaat", "shuru",
                 "naya", "nayi", "नया", "नई", "शुरू", "शुरुआत", "बिगिनर", "बिगिनर्स", "learn", "learning", "sikhna", "सीखना", "scratch", "novice"],
    "trainer": ["trainer", "trainers", "coach", "coaches", "coaching", "instructor", "instructors", "guide", "guidance", "ट्रेनर", "कोच", "कोचिंग",
                "teacher", "mentor"],
    "personaltraining": ["pt", "personal", "one-on-one", "1on1", "पर्सनल"],
    "timing": ["timing", "timings", "time", "times", "hours", "open", "close", "closing", "opening", "schedule", "slot", "slots", "batch", "batches",
               "samay", "waqt", "baje", "बजे", "समय", "टाइम", "टाइमिंग", "टाइमिंग्स", "वक्त", "बैच", "khulta", "band"],
    "location": ["location", "branch", "branches", "address", "near", "nearest", "nearby", "area", "kahan", "where", "पता", "ब्रांच", "शाखा",
                 "लोकेशन", "पास", "paas", "map"],
    "membership": ["membership", "memberships", "plan", "plans", "package", "packages", "pass", "subscription", "join", "joining", "admission",
                   "enroll", "enrol", "register", "registration", "मेंबरशिप", "प्लान", "जॉइन", "ज्वाइन", "जोइन"],
    "freeze": ["freeze", "pause", "hold", "holiday", "vacation", "travel", "रोक", "फ्रीज", "फ़्रीज़"],
    "refund": ["refund", "refunds", "money-back", "moneyback", "cancel", "cancellation", "cancelling", "wapas", "वापस", "रिफंड", "कैंसल"],
    "payment": ["payment", "pay", "paying", "upi", "gpay", "phonepe", "paytm", "card", "cash", "emi", "installment", "installments", "instalment",
                "पेमेंट", "भुगतान", "कार्ड", "यूपीआई", "किस्त", "kist", "link"],
    "nutrition": ["nutrition", "nutritionist", "diet", "dietitian", "dietician", "food", "meal", "meals", "protein", "supplement", "supplements",
                  "whey", "creatine", "eating", "khana", "khaana", "खाना", "डाइट", "प्रोटीन", "सप्लीमेंट", "न्यूट्रिशन", "आहार"],
    "yoga": ["yoga", "योग", "योगा", "meditation", "stretching", "flexibility", "pranayama"],
    "zumba": ["zumba", "ज़ुम्बा", "जुम्बा", "dance", "aerobics", "डांस", "cardio-class"],
    "hiit": ["hiit", "circuit", "crossfit", "boot", "bootcamp"],
    "women": ["women", "woman", "ladies", "lady", "girl", "girls", "female", "mahila", "mahilayein", "महिला", "महिलाओं", "लेडीज़", "लेडीज", "लड़की",
              "ladkiyon", "wife", "mother", "mom", "pregnant", "pregnancy", "postpartum", "prenatal"],
    "senior": ["senior", "seniors", "old", "elderly", "aged", "retired", "parents", "papa", "father", "बुजुर्ग", "सीनियर", "बूढ़े", "बुज़ुर्ग", "age", "arthritis"],
    "injury": ["injury", "injured", "pain", "back", "backpain", "knee", "shoulder", "physio", "physiotherapy", "physiotherapist", "rehab", "recovery",
               "doctor", "slipped", "disc", "dard", "दर्द", "चोट", "फिजियो", "फिज़ियो", "घुटना", "कमर", "सर्जरी", "surgery", "medical", "bp", "diabetes",
               "sugar", "thyroid", "heart", "asthma"],
    "locker": ["locker", "lockers", "shower", "showers", "towel", "towels", "changing", "bathroom", "washroom", "toilet", "लॉकर", "शावर", "तौलिया"],
    "equipment": ["equipment", "machine", "machines", "treadmill", "treadmills", "cycle", "cycles", "dumbbell", "dumbbells", "barbell", "barbells",
                  "plates", "plate", "rack", "racks", "platform", "platforms", "weights", "bar", "bars", "cable", "cables", "leg-press", "mashine",
                  "मशीन", "मशीनें", "ट्रेडमिल", "डंबल", "डम्बल", "बारबेल", "प्लेट", "प्लेट्स", "उपकरण", "सामान"],
    "guest": ["guest", "guests", "friend", "friends", "dost", "दोस्त", "bring", "visitor", "visitors", "trial", "demo", "free-trial"],
    "couple": ["couple", "couples", "spouse", "husband", "wife", "partner", "together", "pati", "patni", "family", "jodi", "जोड़ी", "पति", "पत्नी",
               "परिवार"],
    "kids": ["kid", "kids", "child", "children", "teen", "teens", "teenager", "teenagers", "son", "daughter", "bachcha", "bachche", "बच्चा", "बच्चे",
             "age-limit", "minor", "student", "students", "school", "college"],
    "hygiene": ["hygiene", "clean", "cleanliness", "sanitised", "sanitized", "sanitize", "ac", "air-conditioned", "ventilation", "safe", "safety",
                "cctv", "साफ", "सफाई", "सुरक्षित"],
    "discount": ["discount", "discounts", "offer", "offers", "deal", "deals", "cheaper", "cheap", "lower", "reduce", "concession", "sasta", "सस्ता",
                 "डिस्काउंट", "ऑफर", "छूट", "chhoot", "bargain"],
    "parking": ["parking", "park", "scooter", "bike", "car", "पार्किंग"],
    "app": ["app", "application", "attendance", "tracking", "track", "progress", "booking", "book", "online", "ऐप", "एप"],
    "assessment": ["assessment", "assess", "bmi", "measurement", "measurements", "checkup", "check-up", "body-fat", "inbody", "test", "testing", "goal",
                   "goals", "plan-chart", "chart", "जाँच"],
    "technique": ["technique", "form", "posture", "correct", "correction", "clinic", "clinics", "workshop", "seminar", "video", "sahi", "सही", "तरीका",
                  "tareeka", "tarika", "learn-to-lift"],
    "gear": ["belt", "belts", "straps", "sleeves", "wraps", "chalk", "shoes", "singlet", "gear", "kit", "chappal", "लिफ्टिंग-बेल्ट"],
    "federation": ["federation", "federations", "registration", "weigh-in", "weighin", "weight-class", "category", "categories", "raw", "equipped",
                   "attempt", "attempts", "judges", "rules", "doping", "dope", "wada", "nada", "tested", "banned", "steroid", "steroids", "फेडरेशन"],
    "prep": ["prep", "preparation", "prepare", "preparing", "तैयारी", "taiyari", "tayari", "training-block"],
    "frequency": ["frequency", "days", "daily", "weekly", "week", "often", "regularly", "routine", "split", "program", "programme", "programs",
                  "programming", "cycle-plan", "रूटीन", "प्रोग्राम"],
}
_SYN_MAP: dict[str, str] = {}
for _c, _forms in SYN.items():
    for _f in _forms:
        _SYN_MAP.setdefault(_f.lower(), "c:" + _c)
for _c in SYN:
    _SYN_MAP.setdefault(_c, "c:" + _c)

_PHRASES = [("power lifting", "powerlifting"), ("power lifter", "powerlifter"), ("weight lifting", "weightlifting"), ("bench press", "benchpress"),
            ("dead lift", "deadlift"), ("body building", "bodybuilding"), ("back pain", "backpain"), ("kitne ka", "kitna"),
            ("कितने का", "कितना"), ("कितने की", "कितना"), ("personal training", "personaltraining"), ("personal trainer", "personaltraining"),
            ("पर्सनल ट्रेनिंग", "personaltraining"), ("पर्सनल ट्रेनर", "personaltraining"), ("weight loss", "weightloss"),
            ("वजन कम", "weightloss"), ("वज़न कम", "weightloss"), ("fat loss", "fatloss"), ("wazan kam", "weightloss"), ("vajan kam", "weightloss"),
            ("weight kam", "weightloss"), ("पावर लिफ्टिंग", "powerlifting"), ("how much", "kitna"), ("kya milega", "milega"), ("kya hai", "hai"), ("what do", "do")]


def _is_word_char(ch: str) -> bool:
    return ch.isalnum() or unicodedata.category(ch).startswith("M") or ch in "_-"


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKC", text or "").lower()
    t = t.replace("़", "").replace("ँ", "ं")  # nukta off, chandrabindu -> anusvara
    t = re.sub(r"(\d),(\d)", r"\1\2", t)
    out = "".join(ch if _is_word_char(ch) else " " for ch in t)
    out = " ".join(out.split())
    for a, b in _PHRASES:
        if a in out:
            out = out.replace(a, b)
    return out


def raw_tokens(text: str) -> list[str]:
    return normalize(text).split()


_DEV = {"क": "k", "ख": "k", "ग": "g", "घ": "g", "ङ": "n", "च": "c", "छ": "c", "ज": "j", "झ": "j", "ञ": "n", "ट": "t", "ठ": "t", "ड": "d", "ढ": "d",
        "ण": "n", "त": "t", "थ": "t", "द": "d", "ध": "d", "न": "n", "प": "p", "फ": "f", "ब": "b", "भ": "b", "म": "m", "य": "", "र": "r", "ल": "l",
        "व": "v", "श": "s", "ष": "s", "स": "s", "ह": "", "ळ": "l", "ं": "n", "ँ": "n", "ः": ""}
_DEV_FULL = {"ड": "d", "ढ": "d", "फ": "f", "ज": "j"}


def skeleton(tok: str) -> str:
    """Consonant skeleton shared by Devanagari and Roman spellings of the same sound (vowels, h, y dropped)."""
    if not tok:
        return ""
    if any("ऀ" <= ch <= "ॿ" for ch in tok):
        s = "".join(_DEV.get(ch, "") for ch in tok)
        s = s.replace("kz", "ks")
    else:
        s = tok
        for a, b in (("ph", "f"), ("sh", "s"), ("ch", "c"), ("ck", "k"), ("kh", "k"), ("gh", "g"), ("jh", "j"), ("th", "t"), ("dh", "d"),
                     ("bh", "b"), ("ng", "ng")):
            s = s.replace(a, b)
        s = s.translate(str.maketrans({"w": "v", "q": "k", "x": "ks", "z": "j"}))
        s = s.replace("c", "k").replace("kk", "k").replace("tion", "tn").replace("sion", "sn")
        s = re.sub(r"[aeiouyh0-9_\-]", "", s)
        s = s.replace("c", "k")
    return re.sub(r"(.)\1+", r"\1", s)


def canon(tok: str) -> str:
    """The synonym-table concept for a token, else the token itself."""
    return _SYN_MAP.get(tok, tok)


def content_terms(text: str) -> list[str]:
    """Canonical content tokens (stopwords dropped), order kept, duplicates kept."""
    out = []
    for t in raw_tokens(text):
        if t in STOP or len(t) < 2 and not t.isdigit():
            continue
        out.append(canon(t))
    return out


def term_keys(text: str) -> set[str]:
    """All match keys of a text: canonical tokens plus (weaker) skeletons of tokens that are not concepts."""
    keys: set[str] = set()
    for t in content_terms(text):
        keys.add(t)
        if not t.startswith("c:"):
            sk = skeleton(t)
            if len(sk) >= 3:
                keys.add("~" + sk)
    return keys


def ngram_vec(text: str, n: int = 3, buckets: int = 1 << 16) -> dict[int, float]:
    """Sparse crc32-hashed char n-gram counts of the normalised text (stopwords kept out)."""
    vec: dict[int, float] = {}
    for t in raw_tokens(text):
        if t in STOP:
            continue
        s = f"^{t}$"
        for i in range(max(1, len(s) - n + 1)):
            g = s[i:i + n]
            k = zlib.crc32(g.encode("utf-8")) % buckets
            vec[k] = vec.get(k, 0.0) + 1.0
    return vec


def cosine(a: dict[int, float], b: dict[int, float]) -> float:
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    dot = sum(v * b.get(k, 0.0) for k, v in a.items())
    na = sum(v * v for v in a.values()) ** 0.5
    nb = sum(v * v for v in b.values()) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def detect_lang(text: str) -> str:
    """"hi" when mostly Devanagari, "hinglish" when Roman with common Hindi function words, else "en"."""
    letters = [c for c in text if c.isalpha()]
    if letters and sum("ऀ" <= c <= "ॿ" for c in letters) / len(letters) > 0.3:
        return "hi"
    hing = {"kya", "hai", "hain", "ke", "ki", "ka", "liye", "mein", "me", "kaise", "kitna", "kitne", "nahi", "mujhe", "aap", "aapka", "bata", "batao",
            "chahiye", "karna", "karte", "hota", "milta", "milega", "wala", "wali", "kab", "kahan", "ye", "yeh", "ho", "kar", "raha", "rahe", "se", "ko",
            "toh", "bhi", "main", "mera", "meri", "hum", "kitni"}
    toks = raw_tokens(text)
    return "hinglish" if len([t for t in toks if t in hing]) >= 2 else "en"
