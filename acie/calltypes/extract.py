"""Deterministic answer extraction from the caller's words (English, Hindi, Hinglish). No model, no randomness.

Each extractor returns `(found, value)`: found=False means "this did not answer the question". The same text always
gives the same value. Dates are resolved against an explicit `now` (the caller passes the clock), never the wall clock.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from ..text import _AMT_TOKEN, _amount_tokens, _is_digit, _parse_phrase, _starts_number, normalize, parse_amount, tokens
from .schema import Step

DEFAULT_TZ = "Asia/Kolkata"
UNKNOWN = "unknown"  # a real answer: "I don't know / not sure" (counts as answered, never re-asked)

# --------------------------------------------------------------------------- yes / no / don't know

_YES = re.compile(r"(?<![\wऀ-ॿ])(yes|yeah|yep|yup|yess|sure|ok|okay|correct|right|definitely|absolutely|of course|obviously|"
                  r"haan|han|haa|ha|haanji|haan ji|hanji|ji haan|ji han|bilkul|zaroor|jarur|sahi|theek|thik|"
                  r"हाँ|हां|हा|हांजी|हाँजी|जी हाँ|जी हां|बिल्कुल|ज़रूर|जरूर|सही|ठीक|ठीक है|हाँ जी|हां जी)(?![\wऀ-ॿ])", re.I)
_NO = re.compile(r"(?<![\wऀ-ॿ])(no|nope|nah|not|never|nahi|nahin|nhi|nai|naa|mat|"
                 r"नहीं|नही|नहि|ना|मत)(?![\wऀ-ॿ])", re.I)
# "no problem", "koi baat nahi", "why not": a no-word that means yes
_NEG_YES = re.compile(r"no (problem|issue|worries|doubt)|why not|koi (baat|problem|dikkat|issue) nahi|kyun nahi|kyon nahi|"
                      r"mind nahi|हर्ज नहीं|कोई (बात|प्रॉब्लम|दिक्कत) नहीं|क्यों नहीं|कोई हर्ज|harj nahi", re.I)
_DONT_KNOW = re.compile(r"don.?t know|not sure|no idea|can.?t say|cannot say|prefer not|rather not|pata nahi|pata nhi|nahi pata|"
                        r"maloom nahi|bata nahi (sakta|sakti|paunga|paungi)|nahi bata|abhi nahi (pata|soch)|decide nahi|"
                        r"पता नहीं|नहीं पता|मालूम नहीं|बता नहीं (सकता|सकती)|नहीं बता|सोचा नहीं|तय नहीं|"
                        r"depends|dekhna padega|देखना पड़ेगा|देखना पड़|abhi tak nahi socha", re.I)


def is_dont_know(text: str) -> bool:
    return bool(_DONT_KNOW.search(normalize(text)) or _DONT_KNOW.search(text))


def yes_no(text: str) -> tuple[bool, bool | None]:
    low = normalize(text)
    if _DONT_KNOW.search(low):
        return False, None
    masked = _NEG_YES.sub(" yes ", low)
    toks = tokens(masked)
    short = len(toks) <= 3
    ys = [m.start() for m in _YES.finditer(masked)]
    ns = [m.start() for m in _NO.finditer(masked)]
    # "na" alone at the end of a longer sentence is a tag ("theek hai na"), not a no
    ns = [p for p in ns if not (masked[p:p + 3].strip() in ("na", "naa", "ना") and not short)]
    if ys and not ns:
        return True, True
    if ns and not ys:
        return True, False
    if ys and ns:
        # "haan, but not now" / "no, yes": the first clear word decides only when the sentence is short; otherwise unclear
        if short:
            return True, ys[0] < ns[0]
        return False, None
    return False, None


# --------------------------------------------------------------------------- numbers and money

_COUNT_NOUNS = {"log", "logon", "people", "persons", "person", "members", "member", "employees", "employee", "staff", "users",
                "user", "seats", "seat", "bacche", "baccho", "bachche", "kids", "branches", "branch", "shops", "shop", "stores",
                "store", "cars", "car", "gaadi", "gaadiyan", "rooms", "room", "units", "unit", "flats", "flat", "lakh", "hazaar",
                "sau", "crore", "थे", "लोग", "लोगों", "सदस्य", "कर्मचारी", "बच्चे", "ब्रांच", "दुकान", "दुकानें", "गाड़ी", "कमरे"}
_AMBIG_ONLY_OK_NEXT = _COUNT_NOUNS | {"saal", "sal", "mahine", "mahina", "din", "years", "months", "days", "year", "month", "साल", "महीने", "दिन"}


def find_numbers(text: str) -> list[float]:
    """Every number in the text, in order ("das hazaar", "10k", "1.5 lakh", "paanch log", "२५"). Identifiers (long digit
    strings) are skipped. A lone ambiguous word ("do" = give, "ek", "das") only counts in a very short answer or before a counted noun."""
    from ..text import _AMBIGUOUS
    toks = _amount_tokens(text)
    out: list[float] = []
    i = 0
    while i < len(toks):
        if not _starts_number(toks, i):
            i += 1
            continue
        val, j, info = _parse_phrase(toks, i)
        if j == i:
            i += 1
            continue
        nxt = toks[j] if j < len(toks) else ""
        if info.get("identifier"):
            i = j
            continue
        if info["ambiguous_only"] and not info["mult"] and not info["digit"]:
            if not (len(toks) == 1 or nxt in _AMBIG_ONLY_OK_NEXT):
                i = j
                continue
        out.append(val)
        i = j
    return out


def number(text: str) -> tuple[bool, int | float | None]:
    if is_dont_know(text):
        return False, None
    nums = find_numbers(text)
    if not nums:
        return False, None
    v = nums[0]
    return True, int(v) if float(v).is_integer() else round(v, 2)


def money(text: str) -> tuple[bool, int | None]:
    if is_dont_know(text):
        return False, None
    v = parse_amount(text)
    if v is None:  # "budget 5" with no unit: a bare small number in a short money answer is still the amount
        nums = find_numbers(text)
        if nums and len(tokens(text)) <= 3 and nums[0] > 0:
            return True, int(nums[0])
        return False, None
    return True, int(v)


# --------------------------------------------------------------------------- dates and times

_WD = {"monday": 0, "mon": 0, "somvar": 0, "somwar": 0, "सोमवार": 0, "मंडे": 0, "मॉन्डे": 0, "tuesday": 1, "tue": 1, "mangalvar": 1, "mangalwar": 1,
       "मंगलवार": 1, "ट्यूसडे": 1, "ट्यूज़डे": 1, "wednesday": 2, "wed": 2, "budhvar": 2, "budhwar": 2, "बुधवार": 2, "वेडनेसडे": 2, "thursday": 3,
       "thu": 3, "guruvar": 3, "guruwar": 3, "veervar": 3, "वीरवार": 3, "गुरुवार": 3, "थर्सडे": 3, "friday": 4, "fri": 4, "shukravar": 4, "shukrawar": 4,
       "शुक्रवार": 4, "फ्राइडे": 4, "saturday": 5, "sat": 5, "shanivar": 5, "shaniwar": 5, "शनिवार": 5, "सैटरडे": 5, "sunday": 6, "sun": 6,
       "ravivar": 6, "raviwar": 6, "itwar": 6, "रविवार": 6, "संडे": 6}
_TODAY = {"today", "aaj", "आज", "abhi", "now", "right now", "अभी"}
_TOMORROW = {"tomorrow", "tmrw", "tmr", "kal", "कल", "kal tak"}
_DAYAFTER = {"parso", "parson", "परसों", "परसो"}
_PART = {"morning": 9, "subah": 9, "सुबह": 9, "savere": 9, "dopahar": 13, "दोपहर": 13, "afternoon": 14, "lunch": 13, "evening": 18, "shaam": 18,
         "शाम": 18, "sham": 18, "night": 20, "raat": 20, "रात": 20, "tonight": 20}
_PM_WORDS = {"evening", "shaam", "शाम", "sham", "night", "raat", "रात", "tonight", "afternoon", "dopahar", "दोपहर", "pm"}
_AM_WORDS = {"morning", "subah", "सुबह", "savere", "am"}
_BAJE = {"baje", "bajey", "bajke", "बजे", "oclock", "o", "clock"}
_HALF = {"saade", "saadhe", "sadhe", "sade", "साढ़े", "साढे", "साढ़े"}
_NEXT_WEEK = re.compile(r"next week|agle (hafte|week)|agle hafta|अगले (हफ्ते|सप्ताह|हफ़्ते)|अगला हफ्ता|agla hafta", re.I)
_IN_DURATION = re.compile(r"(?:in|after)\s+(\d+|one|two|three|four|five|six|an?)\s*(minutes?|mins?|hours?|hrs?|days?)|"
                          r"(\d+|ek|do|teen|char|paanch|एक|दो|तीन|चार|पांच)\s*(ghante|ghanta|ghnte|din|minute|mins?|घंटे|घंटा|दिन|मिनट)\s*(mein|me|baad|ke baad|में|बाद|के बाद)", re.I)
_CLOCK = re.compile(r"\b(\d{1,2})(?:[:.](\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)\b|\b(\d{1,2})[:.](\d{2})\b")
_DATE_NUM = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s*(?:tarikh|tareekh|तारीख)|\b(\d{1,2})(?:st|nd|rd|th)\s+(?:of\s+)?(?:this month)?\b", re.I)
_DUR_NUM = {"one": 1, "an": 1, "a": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "ek": 1, "do": 2, "teen": 3, "char": 4, "paanch": 5,
            "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पांच": 5}


def _hour_before_baje(toks: list[str], k: int) -> tuple[float | None, int]:
    """Hour (and half) from the words just before 'baje' ('chhe baje', 'saade chaar baje', '6 baje', 'dedh baje')."""
    from ..text import _FRACTION, _SMALL
    prev = toks[k - 1] if k >= 1 else ""
    prev2 = toks[k - 2] if k >= 2 else ""
    if prev in _FRACTION:  # dedh baje 1:30, dhai baje 2:30
        return _FRACTION[prev] , k - 1
    if prev2 in _HALF and (prev in _SMALL or _is_digit(prev)):
        h = _SMALL.get(prev) if prev in _SMALL else float(prev)
        return h + 0.5, k - 2
    if _is_digit(prev):
        return float(prev), k - 1
    if prev in _SMALL and _SMALL[prev] <= 24:
        return float(_SMALL[prev]), k - 1
    return None, k


def date_time(text: str, now: datetime | None = None, tz: str = DEFAULT_TZ) -> tuple[bool, str | None]:
    """A day and/or time the caller named ("kal shaam 6 baje", "tomorrow at 4pm", "Monday morning", "in 2 hours", "5 baje")
    as an ISO-8601 string with offset in `tz`. Needs at least a day word or a time. Day only: part of day, else 11:00."""
    zone = ZoneInfo(tz)
    now = (now or datetime.now(zone)).astimezone(zone)
    low = normalize(text)
    toks = tokens(low)
    if not toks:
        return False, None
    # relative: "in 2 hours", "do ghante baad"
    m = _IN_DURATION.search(low)
    if m:
        qty = m.group(1) or m.group(3)
        unit = (m.group(2) or m.group(4) or "").lower()
        n = int(qty) if str(qty).isdigit() else _DUR_NUM.get(str(qty).lower(), 1)
        delta = timedelta(days=n) if unit.startswith(("day", "din", "दिन")) else timedelta(minutes=n) if unit.startswith(("min", "मिन")) else timedelta(hours=n)
        due = (now + delta).replace(second=0, microsecond=0)
        return True, due.isoformat()
    day: datetime | None = None
    explicit_day = False
    tset = set(toks)
    if tset & _DAYAFTER:
        day, explicit_day = now + timedelta(days=2), True
    elif tset & _TOMORROW and not re.search(r"\bkal (tha|thi|the|gaya|gayi)\b", low):
        day, explicit_day = now + timedelta(days=1), True
    elif tset & {"today", "aaj", "आज", "tonight"}:
        day, explicit_day = now, True
    elif _NEXT_WEEK.search(low):
        day, explicit_day = now + timedelta(days=7), True
    else:
        for t in toks:
            if t in _WD:
                ahead = (_WD[t] - now.weekday()) % 7 or 7
                if "next" in tset or "agle" in tset or "अगले" in tset:
                    ahead = ahead if ahead > 0 else 7
                day, explicit_day = now + timedelta(days=ahead), True
                break
    dm = _DATE_NUM.search(low)
    if day is None and dm:
        dd = int(dm.group(1) or dm.group(2))
        if 1 <= dd <= 31:
            cand = now
            for _ in range(2):
                try:
                    cand = now.replace(day=dd)
                except ValueError:
                    cand = None
                if cand is not None and cand.date() >= now.date():
                    break
                nm = (now.replace(day=1) + timedelta(days=32)).replace(day=1)
                now2 = nm
                try:
                    cand = now2.replace(day=dd)
                except ValueError:
                    cand = None
                break
            if cand is not None:
                day, explicit_day = cand, True
    # time of day
    hour: float | None = None
    pm = bool(tset & _PM_WORDS)
    am = bool(tset & _AM_WORDS)
    cm = _CLOCK.search(low)
    if cm:
        if cm.group(1) is not None:
            hh, mm, mer = int(cm.group(1)), int(cm.group(2) or 0), cm.group(3).replace(".", "")
            if 1 <= hh <= 12:
                hour = (hh % 12 + (12 if mer == "pm" else 0)) + mm / 60
                pm, am = mer == "pm", mer == "am"
        else:
            hh, mm = int(cm.group(4)), int(cm.group(5))
            if 0 <= hh <= 23 and mm < 60:
                hour = hh + mm / 60
    if hour is None:
        for k, t in enumerate(toks):
            if t in _BAJE:
                h, _ = _hour_before_baje(toks, k)
                if h is not None:
                    hour = h
                    break
        if hour is None and ("at" in tset or "by" in tset):
            for k, t in enumerate(toks):
                if t in ("at", "by") and k + 1 < len(toks) and _is_digit(toks[k + 1]) and 1 <= int(float(toks[k + 1])) <= 12:
                    hour = float(toks[k + 1])
                    break
        if hour is not None and hour < 24 and not (cm and cm.group(1) is None):
            h0 = hour
            whole = int(h0)
            if pm and whole < 12:
                hour = h0 + 12
            elif am:
                hour = h0 % 12 if whole == 12 else h0
            elif 1 <= whole <= 7:
                hour = h0 + 12  # "5 baje" on a business call means 5 pm
            elif whole == 12:
                hour = h0
    part_hour = next((h for w, h in _PART.items() if w in tset), None)
    if hour is None and part_hour is None and not explicit_day:
        return False, None
    if hour is None:
        hour = float(part_hour if part_hour is not None else 11)
    if hour >= 24:
        return False, None
    if day is None:  # a time with no day: today if it is still ahead, else tomorrow
        day = now
        cand = day.replace(hour=int(hour), minute=int(round((hour % 1) * 60)), second=0, microsecond=0)
        if cand <= now:
            day = now + timedelta(days=1)
    due = day.replace(hour=int(hour), minute=int(round((hour % 1) * 60)), second=0, microsecond=0)
    return True, due.isoformat()


def has_time_words(text: str) -> bool:
    return date_time(text, datetime(2026, 1, 5, 9, 0, tzinfo=ZoneInfo(DEFAULT_TZ)))[0]


# --------------------------------------------------------------------------- choices and free text

def _choice_words(step: Step, c) -> list[str]:
    words = [c.id.replace("_", " ")] + list(c.keywords) + [v for v in c.label.values() if v]
    return [w for w in dict.fromkeys(w.strip().lower() for w in words if w and w.strip())]


def choice(step: Step, text: str) -> tuple[bool, str | None]:
    """The choice whose keyword/label appears in the caller's words. Several matches: the longest keyword wins ("not interested"
    over "interested"), then the one said first. A keyword written "~word" is weak: used only when no strong keyword matched
    (a bare "haan" or "chahiye" must not beat "sochna padega")."""
    low = " " + " ".join(tokens(text)) + " "
    for weak in (False, True):
        best: tuple[tuple[int, int], str] | None = None
        for c in step.choices:
            for w in _choice_words(step, c):
                if w.startswith("~") != weak:
                    continue
                w = w.lstrip("~")
                tw = " ".join(tokens(w))
                if not tw:
                    continue
                pat = (" " + tw[:-1]) if w.endswith("*") else (" " + tw + " ")
                pos = low.find(pat)
                if pos >= 0 and (best is None or (-len(tw), pos) < best[0]):
                    best = ((-len(tw), pos), c.id)
        if best is not None:
            return True, best[1]
    if is_dont_know(text):
        return True, UNKNOWN
    return False, None


_QWORDS = re.compile(r"\?|\b(what|how|why|when|where|which|who|kya|kaise|kab|kyun|kyon|kaun|kahan|kitna|kitne|kitni)\b|क्या|कैसे|कब|क्यों|कौन|कहां|कहाँ|कितना|कितने|कितनी", re.I)


def free_text(text: str) -> tuple[bool, str | None]:
    t = " ".join(text.split())
    if is_dont_know(t):
        return True, UNKNOWN
    toks = tokens(t)
    if len(toks) < 1 or _QWORDS.search(t) and len(toks) <= 8:
        return False, None
    if all(w in _FILLER for w in toks):
        return False, None
    return True, t[:200]


_FILLER = {"haan", "han", "ha", "hmm", "hm", "ok", "okay", "ji", "yes", "yeah", "acha", "accha", "achha", "um", "uh", "हाँ", "हां", "जी", "हम्म", "अच्छा", "ठीक", "theek",
           "hai", "है", "thik", "sir", "madam", "bolo", "boliye", "bataiye", "बोलिए", "बताइए", "bol", "ji haan", "हांजी"}


def extract(step: Step, text: str, now: datetime | None = None, tz: str = DEFAULT_TZ) -> tuple[bool, object]:
    """One entry point: the value this text gives for `step`, per its answer type."""
    ok, val = _extract(step, text, now, tz)
    if not ok and step.answer_type not in ("text", "choice", "date_time") and is_dont_know(text):
        return True, UNKNOWN  # "pata nahi": a real answer (nothing to record), never asked again
    return ok, val


def _extract(step: Step, text: str, now: datetime | None, tz: str) -> tuple[bool, object]:
    at = step.answer_type
    if at == "yes_no":
        return yes_no(text)
    if at == "number":
        return number(text)
    if at == "money":
        return money(text)
    if at == "date_time":
        ok, v = date_time(text, now, tz)
        if ok:
            return True, v
        yn_ok, yn = yes_no(text)
        if yn_ok and yn is False:  # "not now": a real answer, nothing is booked
            return True, "declined"
        if is_dont_know(text):
            return True, UNKNOWN
        return False, None
    if at == "choice":
        return choice(step, text)
    return free_text(text)


def step_keyword_hit(step: Step, text: str) -> bool:
    """The caller mentioned the topic of this step ("budget", "kitne log"): lets one sentence answer a later question too."""
    low = " " + " ".join(tokens(text)) + " "
    for k in step.keywords:
        tk = " ".join(tokens(k))
        if tk and (" " + tk + " ") in low:
            return True
        if k.endswith("*") and (" " + " ".join(tokens(k[:-1]))) in low:
            return True
    return False
