"""Pronunciation lexicon: every agent line is respelled right before TTS so names, brands, acronyms, units and rupee
amounts come out right in the voice that speaks them (owner 15 Nov: "Aujla", "FitZone", "Claude", VIP/UPI, ₹ prices).

Layers (later wins):  global defaults (GLOBAL below, then data/voice/pronunciations.json "global")
                      -> tenant ("tenants"[tenant_id]) -> product (Product.pronunciations = Hindi/Devanagari values,
                      then "products"[product_id]) -> agent ("agents"[agent_id]).
An entry is  word -> {"hi": Devanagari, "hinglish": Roman, "en": English respelling}  (a bare string = "hi" only, the
old Product.pronunciations shape). Which value is used depends on the voice that speaks the line:
  - lang "en" (English voice)                        -> "en"
  - a line with Devanagari in it (Hindi voice)       -> "hi"
  - a Roman Hinglish line (Hindi voice)              -> "hinglish"
An "en" value of the form "ph:AW1 JH L AA0" is sent as a Fish Audio phoneme tag (CMU Arpabet,
<|phoneme_start|>...<|phoneme_end|>), which s2.1-pro-free honours for English; Hindi has no phoneme format there,
so Hindi uses Devanagari respelling.

Matching is whole-word (letters/digits of the same script may not touch the match), longest key first, one pass (a
replacement is never re-replaced). Keys written in ALL CAPS (VIP, UPI, AI) match case-sensitively, so the Hinglish
word "ai" is left alone; every other key matches in any case. Digits are never respelled by the lexicon; rupee
amounts are only re-ordered / spelled out exactly (₹1,499 -> "चौदह सौ निन्यानवे रुपये"), never rounded.

"suggestions"[product_id] holds auto-detected candidates (acie.pronounce_suggest) that are NOT applied until accepted.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from pathlib import Path

log = logging.getLogger("acie.pronounce")

LANGS = ("hi", "hinglish", "en")
DEVA = re.compile(r"[ऀ-ॿ]")
PH_OPEN, PH_CLOSE = "<|phoneme_start|>", "<|phoneme_end|>"

# Global defaults: common brand/tech words, abbreviations and units the Fish voices misread.
GLOBAL: dict[str, dict[str, str]] = {
    "AI": {"hi": "ए-आई", "hinglish": "A.I."},
    "SMS": {"hi": "एस-एम-एस", "hinglish": "S.M.S."},
    "UPI": {"hi": "यू-पी-आई", "hinglish": "U.P.I."},
    "VIP": {"hi": "वी-आई-पी", "hinglish": "V.I.P."},
    "OTP": {"hi": "ओ-टी-पी", "hinglish": "O.T.P."},
    "EMI": {"hi": "ई-एम-आई", "hinglish": "E.M.I."},
    "GST": {"hi": "जी-एस-टी", "hinglish": "G.S.T."},
    "PIN": {"hi": "पिन", "hinglish": "pin", "en": "pin"},
    "KYC": {"hi": "के-वाई-सी", "hinglish": "K.Y.C."},
    "ID": {"hi": "आई-डी", "hinglish": "I.D."},
    "PDF": {"hi": "पी-डी-एफ", "hinglish": "P.D.F."},
    "WhatsApp": {"hi": "व्हाट्सऐप"},
    "Google Pay": {"hi": "गूगल पे"},
    "PhonePe": {"hi": "फ़ोनपे", "hinglish": "phone pe", "en": "phone pay"},
    "Paytm": {"hi": "पेटीएम", "hinglish": "pay-tee-em", "en": "pay-tee-em"},
    "online": {"hi": "ऑनलाइन"},
    "app": {"hi": "ऐप"},
    "link": {"hi": "लिंक"},
    "offer": {"hi": "ऑफ़र"},
    # units / symbols (whole word, so "km" inside a word is untouched)
    "km": {"hi": "किलोमीटर", "hinglish": "kilometre", "en": "kilometres"},
    "kg": {"hi": "किलो", "hinglish": "kilo", "en": "kilos"},
    "GB": {"hi": "जी-बी", "hinglish": "G.B.", "en": "gigabytes"},
    "hrs": {"hi": "घंटे", "hinglish": "ghante", "en": "hours"},
    "mins": {"hi": "मिनट", "hinglish": "minute", "en": "minutes"},
    "/month": {"hi": " महीना", "hinglish": " mahina", "en": " a month"},
    "/mo": {"hi": " महीना", "hinglish": " mahina", "en": " a month"},
    "/year": {"hi": " साल", "hinglish": " saal", "en": " a year"},
}

# ------------------------------------------------------------------ numbers

_HI_0_99 = ("शून्य एक दो तीन चार पाँच छह सात आठ नौ दस ग्यारह बारह तेरह चौदह पंद्रह सोलह सत्रह अठारह उन्नीस बीस "
            "इक्कीस बाईस तेईस चौबीस पच्चीस छब्बीस सत्ताईस अट्ठाईस उनतीस तीस इकतीस बत्तीस तैंतीस चौंतीस पैंतीस छत्तीस "
            "सैंतीस अड़तीस उनतालीस चालीस इकतालीस बयालीस तैंतालीस चवालीस पैंतालीस छियालीस सैंतालीस अड़तालीस उनचास पचास "
            "इक्यावन बावन तिरपन चौवन पचपन छप्पन सत्तावन अट्ठावन उनसठ साठ इकसठ बासठ तिरसठ चौंसठ पैंसठ छियासठ सड़सठ "
            "अड़सठ उनहत्तर सत्तर इकहत्तर बहत्तर तिहत्तर चौहत्तर पचहत्तर छिहत्तर सतहत्तर अठहत्तर उन्यासी अस्सी इक्यासी "
            "बयासी तिरासी चौरासी पचासी छियासी सत्तासी अट्ठासी नवासी नब्बे इक्यानवे बानवे तिरानवे चौरानवे पंचानवे छियानवे "
            "सत्तानवे अट्ठानवे निन्यानवे").split()
_HG_0_99 = ("shunya ek do teen chaar paanch chhah saat aath nau das gyaarah baarah terah chaudah pandrah solah satrah "
            "athaarah unnees bees ikkees baaees teyees chaubees pachchees chhabbees sattaaees atthaaees untees tees "
            "iktees battees taintees chauntees paintees chhattees saintees adtees untaalees chaalees iktaalees bayaalees "
            "taintaalees chavaalees paintaalees chhiyaalees saintaalees adtaalees unchaas pachaas ikyaavan baavan tirpan "
            "chauvan pachpan chhappan sattaavan atthaavan unsath saath iksath baasath tirsath chaunsath painsath "
            "chhiyaasath sadsath adsath unhattar sattar ikhattar bahattar tihattar chauhattar pachhattar chhihattar "
            "sathattar athhattar unyaasi assi ikyaasi bayaasi tiraasi chauraasi pachaasi chhiyaasi sattaasi atthaasi "
            "navaasi nabbe ikyaanve baanve tiraanve chauraanve pachaanve chhiyaanve sattaanve atthaanve ninyaanve").split()
assert len(_HI_0_99) == 100 and len(_HG_0_99) == 100


def number_words(n: int, lang: str) -> str:
    """Exact Indian-system words for 0 <= n < 1,00,00,00,000 (hi = Devanagari, hinglish = Roman). Larger -> digits."""
    t = _HI_0_99 if lang == "hi" else _HG_0_99
    big = (("करोड़", "crore", 10**7), ("लाख", "lakh", 10**5), ("हज़ार", "hazaar", 1000), ("सौ", "sau", 100))
    if n < 0 or n >= 10**9:
        return f"{n:,}"
    if n < 100:
        return t[n]
    parts = []
    for hi_w, hg_w, unit in big:
        q, n = divmod(n, unit)
        if q:
            parts.append(f"{number_words(q, lang)} {hi_w if lang == 'hi' else hg_w}")
    if n:
        parts.append(t[n])
    return " ".join(parts)


_RUPEE = re.compile(r"(?:₹|\bRs\.?|\bINR)\s?(\d{1,3}(?:,\d{2,3})+|\d+)(?:\.(\d{1,2}))?(?![\d,]*\d)")
_RUPEE_WORD = {"hi": "रुपये", "hinglish": "rupaye", "en": "rupees"}


def speak_rupees(text: str, lang: str) -> str:
    """₹1,499 -> 'चौदह सौ निन्यानवे रुपये' (hi) / 'chaudah sau ninyaanve rupaye' (hinglish) / '1,499 rupees' (en).
    The value never changes: paise are kept, an amount the words table can't hold stays in digits."""
    def sub(m: re.Match) -> str:
        whole, paise = m.group(1), m.group(2)
        n = int(whole.replace(",", ""))
        if lang == "en":
            s = f"{n:,}" + (f".{paise}" if paise else "") + (" rupee" if n == 1 and not paise else " rupees")
            return s
        if n >= 10**9:
            words = whole
        else:
            words = number_words(n, lang)
        if paise and int(paise):
            p = int(paise.ljust(2, "0"))
            words += f" {_RUPEE_WORD[lang]} {number_words(p, lang)} {'पैसे' if lang == 'hi' else 'paise'}"
            return words
        return f"{words} {_RUPEE_WORD[lang]}"
    return _RUPEE.sub(sub, text)


# ------------------------------------------------------------------ lexicon store

def _norm_entry(v) -> dict[str, str]:
    if isinstance(v, str):
        return {"hi": v} if v.strip() else {}
    if isinstance(v, dict):
        return {k: str(s) for k, s in v.items() if k in LANGS and isinstance(s, str) and s.strip()}
    return {}


def norm_lexicon(d) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for k, v in (d or {}).items() if isinstance(d, dict) else ():
        k = str(k).strip()
        e = _norm_entry(v)
        if k and len(k) <= 60 and e:
            out[k] = e
    return out


class PronunciationStore:
    """data/voice/pronunciations.json: {"global": lex, "tenants": {id: lex}, "products": {id: lex},
    "agents": {id: lex}, "suggestions": {product_id: [{word, values, score?, source}]}}. Re-read when the file changes."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._mtime = -1.0
        self._data: dict = {}

    def data(self) -> dict:
        try:
            m = self.path.stat().st_mtime
        except OSError:
            return {"global": {}, "tenants": {}, "products": {}, "agents": {}, "suggestions": {}}
        if m != self._mtime:
            try:
                raw = json.loads(self.path.read_text())
                self._data = raw if isinstance(raw, dict) else {}
            except Exception:  # noqa: BLE001 - a broken file never stops a call
                log.exception("pronunciations.json unreadable")
                self._data = {}
            self._mtime = m
        for k in ("global", "tenants", "products", "agents", "suggestions"):
            self._data.setdefault(k, {})
        return self._data

    def save(self, data: dict) -> None:
        with self._lock:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1))
            tmp.replace(self.path)
            self._mtime = -1.0

    def scope(self, kind: str, key: str = "") -> dict[str, dict[str, str]]:
        d = self.data()
        if kind == "global":
            return norm_lexicon(d.get("global"))
        return norm_lexicon((d.get(kind) or {}).get(key))

    def put(self, kind: str, key: str, lex: dict) -> dict:
        d = json.loads(json.dumps(self.data()))
        clean = norm_lexicon(lex)
        if kind == "global":
            d["global"] = clean
        else:
            d.setdefault(kind, {})[key] = clean
        self.save(d)
        return clean


# ------------------------------------------------------------------ resolver

def merged(store: PronunciationStore | None = None, tenant: str = "", product=None, agent_id: str = "",
           include_global: bool = True) -> dict[str, dict[str, str]]:
    """The layered lexicon for one call: global -> tenant -> product -> agent (later layers win, per language)."""
    layers: list[dict] = []
    if include_global:
        layers.append(GLOBAL)
    if store is not None:
        layers += [store.scope("global"), store.scope("tenants", tenant) if tenant else {}]
    if product is not None:
        layers.append(norm_lexicon(getattr(product, "pronunciations", None) or {}))
        if store is not None:
            layers.append(store.scope("products", getattr(product, "id", "") or ""))
    if store is not None and agent_id:
        layers.append(store.scope("agents", agent_id))
    out: dict[str, dict[str, str]] = {}
    for lay in layers:
        for k, e in lay.items():
            # an entry keyed in another case ("Fitzone" vs "FitZone") is the same word: the later layer replaces it
            for old in [o for o in out if o != k and o.lower() == k.lower() and not (o.isupper() or k.isupper())]:
                out.pop(old)
            out[k] = {**out.get(k, {}), **e}
    return out


def voice_key(text: str, lang: str) -> str:
    if lang == "en":
        return "en"
    return "hi" if DEVA.search(text) else "hinglish"


def _wrap(say: str) -> str:
    if say.startswith("ph:"):
        return f"{PH_OPEN}{say[3:].strip()}{PH_CLOSE}"
    return say


def _pattern(key: str) -> str:
    body = r"\s+".join(re.escape(p) for p in key.split())
    if DEVA.search(key):
        return r"(?<![ऀ-ॿ])" + body + r"(?![ऀ-ॿ])"
    lead = r"(?<![A-Za-z0-9])" if key[:1].isalnum() else ""
    tail = r"(?![A-Za-z0-9])" if key[-1:].isalnum() else ""
    if key.isupper() or not any(c.isalpha() for c in key):
        return lead + body + tail
    return lead + "(?i:" + body + ")" + tail


class Resolver:
    """Compiled lexicon for one voice: apply(text, lang) -> the text sent to TTS."""

    def __init__(self, lexicon: dict[str, dict[str, str]]) -> None:
        self.lexicon = lexicon
        self._rx: dict[str, tuple[re.Pattern | None, list[str]]] = {}
        for key in LANGS:
            pairs = sorted(((k, e[key]) for k, e in lexicon.items() if e.get(key)), key=lambda kv: len(kv[0]), reverse=True)
            rx = re.compile("|".join(f"({_pattern(k)})" for k, _ in pairs)) if pairs else None
            self._rx[key] = (rx, [v for _, v in pairs])

    def apply(self, text: str, lang: str) -> str:
        if not text:
            return text
        key = voice_key(text, lang)
        rx, values = self._rx[key]
        out = text
        if rx is not None:
            out = rx.sub(lambda m: _wrap(values[m.lastindex - 1]), out)
        out = speak_rupees(out, key)
        return re.sub(r"[ \t]{2,}", " ", out)


def resolve(text: str, lang: str, lexicon: dict[str, dict[str, str]]) -> str:
    return Resolver(lexicon).apply(text, lang)


_STORE: PronunciationStore | None = None


def default_store(data_dir: Path | str | None = None) -> PronunciationStore:
    global _STORE
    p = Path(data_dir) / "pronunciations.json" if data_dir else Path("data/voice/pronunciations.json")
    if _STORE is None or _STORE.path != p:
        _STORE = PronunciationStore(p)
    return _STORE


def resolver_for(product=None, agent_id: str = "", data_dir: Path | str | None = None) -> Resolver:
    """The call path's resolver: never raises (a broken lexicon falls back to the global defaults)."""
    try:
        store = default_store(data_dir)
        return Resolver(merged(store, getattr(product, "tenant", "") or "", product, agent_id))
    except Exception:  # noqa: BLE001
        log.exception("pronunciation lexicon failed; global defaults only")
        return Resolver(dict(GLOBAL))
