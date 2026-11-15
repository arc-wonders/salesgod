"""Pronunciation suggestions and round-trip checks (offline / on demand only, never in the live call path).

1. candidates(product): proper nouns, brand names, acronyms and code-mixed English words found in the product's own text
   (name, seller, one-liner, benefits, pitch, FAQs, knowledge, offer terms) that the lexicon does not cover yet.
2. llm_respellings(words): the productgen LLM (Gemini, JSON mode) proposes 2-3 respellings per word and language.
3. score(heard, word, entry, lang): did the transcript of the synthesised line bring the word back? 0..1, fuzzy and
   phonetic (vowel length / w~v / z~j folded), compared with the word and every respelling of it.
   The voice service runs synth -> its own Whisper -> score and keeps the best candidate (voice/pronounce_api.py).
"""

from __future__ import annotations

import difflib
import re
import unicodedata

from .pronounce import DEVA, GLOBAL, LANGS, norm_lexicon

_STOP = set("""a an the and or but if of to in on at by for with from as is are was were be been it this that these those
i you we they he she my your our their me us them his her its not no yes do does did can will would should could may might
must have has had so very more most less just only also than then there here what which who whom whose when where why how
all any each every some such own same other new free per one two three four five six seven eight nine ten month months
year years day days week weeks hour hours minute minutes time times get got make made use used want need like hello hi
please thank thanks sorry ok okay sir madam ji price offer plan plans team call phone today now link sms payment
classes class tickets ticket membership members member gym trainer trainers session sessions concert
product facts todo note notes hindi english""".split())
_HI_FIELDS = ("pitch_hi", "benefits_hi", "answer_hi", "hi")
_WORD = re.compile(r"[A-Za-z][A-Za-z'’\-]*[A-Za-z]|[A-Z]{2,}")
_FIELDS = ("product_name", "seller_name", "agent_name", "one_liner", "description", "knowledge", "pitch_en", "pitch_hi",
           "pitch_hinglish", "benefits", "benefits_hi", "benefits_hinglish", "faqs", "terms", "offer_validity",
           "price_note", "renewal_note", "annual_note", "info_source", "call_to_action", "topic_words")


def _texts(obj) -> list[str]:
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [t for v in obj.values() for t in _texts(v)]
    if isinstance(obj, (list, tuple)):
        return [t for v in obj for t in _texts(v)]
    return []


def _hindi_texts(obj, key: str = "") -> list[str]:
    """Devanagari-language fields only (pitch_hi, benefits_hi, answer_hi, {hi: ...}): English words inside them are
    what the Hindi voice reads."""
    if isinstance(obj, dict):
        return [t for k, v in obj.items() for t in _hindi_texts(v, k)]
    if isinstance(obj, list):
        return [t for v in obj for t in _hindi_texts(v, key)]
    return [obj] if isinstance(obj, str) and key in _HI_FIELDS else []


def candidates(product: dict, known: dict | None = None, playbook_text: str = "", limit: int = 30) -> list[dict]:
    """[{word, kind: acronym|name|english_in_hindi, count, example}] most frequent first. `known` = words already in
    the lexicon (skipped)."""
    known_l = {k.lower() for k in (known or {})} | {k.lower() for k in GLOBAL}
    known_l |= {p for k in known_l if " " in k for p in k.split()}  # "Fan Pit" known -> "Fan", "Pit" too
    hits: dict[str, dict] = {}
    texts = [(t, False) for f in _FIELDS for t in _texts(product.get(f))] + ([(playbook_text, False)] if playbook_text else [])
    texts += [(t, True) for t in _hindi_texts(product)]
    for t, hindi_src in texts:
        hindi_line = hindi_src and bool(DEVA.search(t))
        for m in _WORD.finditer(t):
            w = re.sub(r"['’]s$", "", m.group(0)).strip("'’-")
            wl = w.lower()
            if len(w) < 2 or wl in known_l:
                continue
            start_of_sentence = m.start() == 0 or t[:m.start()].rstrip()[-1:] in ".!?।:"
            if wl in _STOP and (w.isupper() or not hindi_line):
                continue
            if w.isupper() and len(w) <= 6:
                kind = "acronym"
            elif any(c.isupper() for c in w[1:]) or (w[0].isupper() and not start_of_sentence and wl not in _STOP):
                kind = "name"
            elif hindi_line and wl not in _STOP and len(w) > 2:
                kind = "english_in_hindi"
            else:
                continue
            h = hits.setdefault(wl, {"word": w, "kind": kind, "count": 0, "example": t[:160]})
            h["count"] += 1
            if kind == "name" and h["kind"] == "english_in_hindi":
                h["kind"], h["word"] = "name", w
    # multi-word brand names that appear verbatim (e.g. "Fan Pit", "Claude Pro") come from the product name / benefits
    for t in _texts([product.get("product_name"), product.get("benefits"), product.get("seller_name")]):
        for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", t):
            w = m.group(1)
            if w.lower() not in known_l and len(w.split()) <= 3:
                hits.setdefault(w.lower(), {"word": w, "kind": "name", "count": 1, "example": t[:160]})
    order = {"name": 0, "acronym": 1, "english_in_hindi": 2}
    return sorted(hits.values(), key=lambda h: (order[h["kind"]], -h["count"]))[:limit]


PROMPT = ("You write PRONUNCIATION RESPELLINGS for a text-to-speech voice on Indian sales calls. For each word give 2-3 "
          "candidate respellings per language so the voice says it the way an Indian customer expects: "
          "hi = Devanagari spelling for a Hindi voice (acronyms letter by letter with hyphens, e.g. VIP -> वी-आई-पी); "
          "hinglish = Roman spelling a Hindi voice reads correctly (e.g. Aujla -> Aujlaa); "
          "en = an English-friendly respelling for an American English voice, hyphenated syllables (e.g. Aujla -> Ouj-laa, "
          "Karan -> Kuh-run); give the plain word first when an English voice already says it right. "
          "Never change digits. Best candidate first.")
SHAPE = '{"words": [{"word": str, "hi": [str], "hinglish": [str], "en": [str], "note": str}]}'


async def llm_respellings(words: list[dict], product_name: str, llm=None) -> dict[str, dict[str, list[str]]]:
    if not words:
        return {}
    if llm is None:
        from .productgen.llm import default_llm
        llm = default_llm()
    out = await llm.json("pronunciations", {"task": PROMPT, "shape": SHAPE, "product": product_name,
                                            "words": [{"word": w["word"], "kind": w["kind"], "example": w["example"]}
                                                      for w in words]})
    res: dict[str, dict[str, list[str]]] = {}
    for item in out.get("words") or []:
        if not isinstance(item, dict) or not str(item.get("word") or "").strip():
            continue
        w0 = str(item["word"]).strip()
        res[w0] = {lang: [str(x).strip() for x in (item.get(lang) or []) if str(x).strip()][:3]
                                          if isinstance(item.get(lang), list) else
                                          ([str(item[lang]).strip()] if str(item.get(lang) or "").strip() else [])
                                          for lang in LANGS}
        # a candidate equal to the word itself is "already fine" for that voice: not a respelling
        res[w0] = {lang: [v for v in vals if v.lower() != w0.lower()] for lang, vals in res[w0].items()}
        if not any(res[w0].values()):
            res.pop(w0)
    return res


# ------------------------------------------------------------------ round-trip score

_FOLD = [("aa", "a"), ("ee", "i"), ("ii", "i"), ("oo", "u"), ("uu", "u"), ("ou", "o"), ("au", "o"), ("w", "v"),
         ("z", "j"), ("ph", "f"), ("q", "k"), ("c", "k"), ("y", "i"), ("h", "")]


def _deva_to_roman(s: str) -> str:
    """Rough Devanagari -> Latin, enough to compare sounds (not a transliteration standard)."""
    cons = dict(zip(list("कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह") + ["क़", "ख़", "ग़", "ज़", "ड़", "ढ़", "फ़"],
                    ["k", "kh", "g", "gh", "n", "ch", "chh", "j", "jh", "n", "t", "th", "d", "dh", "n", "t", "th", "d",
                     "dh", "n", "p", "ph", "b", "bh", "m", "y", "r", "l", "v", "sh", "sh", "s", "h", "k", "kh", "g", "z",
                     "d", "dh", "f"]))
    vow = {"अ": "a", "आ": "aa", "इ": "i", "ई": "ee", "उ": "u", "ऊ": "oo", "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au",
           "ऑ": "o", "ऋ": "ri"}
    mat = {"ा": "aa", "ि": "i", "ी": "ee", "ु": "u", "ू": "oo", "े": "e", "ै": "ai", "ो": "o", "ौ": "au", "ॉ": "o",
           "ृ": "ri", "ं": "n", "ँ": "n", "ः": "h"}
    s = unicodedata.normalize("NFC", s)
    out, i = [], 0
    while i < len(s):
        ch = s[i]
        nxt = s[i + 1] if i + 1 < len(s) else ""
        if nxt == "़" and ch + nxt in cons:
            ch, i = ch + nxt, i + 1
            nxt = s[i + 1] if i + 1 < len(s) else ""
        if ch in cons:
            out.append(cons[ch])
            if nxt and nxt not in mat and nxt != "्" and DEVA.match(nxt):
                out.append("a")
        elif ch in vow:
            out.append(vow[ch])
        elif ch in mat:
            out.append(mat[ch])
        elif ch == "्":
            pass
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def phonetic(s: str) -> str:
    s = _deva_to_roman(s) if DEVA.search(s) else s
    s = re.sub(r"[^a-z ]", "", s.lower().replace("-", ""))
    for a, b in _FOLD:
        s = s.replace(a, b)
    return re.sub(r"(.)\1+", r"\1", s).replace(" ", "")


def score(heard: str, word: str, entry: dict | None = None, lang: str = "en") -> float:
    """0..1: how well a stretch of the transcript matches the word (or any of its respellings)."""
    targets = {word} | {v for v in (entry or {}).values() if v and not v.startswith("ph:")}
    toks = re.findall(r"[\wऀ-ॿ'-]+", heard)
    best = 0.0
    for t in targets:
        n = max(1, len(t.replace("-", " ").split()))
        pt = phonetic(t)
        if not pt:
            continue
        for k in range(max(1, n - 1), n + 2):
            for i in range(0, max(1, len(toks) - k + 1)):
                cand = phonetic(" ".join(toks[i:i + k]))
                if cand:
                    best = max(best, difflib.SequenceMatcher(None, pt, cand).ratio())
    return round(best, 3)


def sentence_for(word: str, say: str, lang: str) -> str:
    """A short carrier line for the round trip (the word in context, as a call would say it)."""
    if lang == "en":
        return f"Let me tell you about {say} today."
    if lang == "hinglish":
        return f"Main aapko {say} ke baare mein batata hoon."
    return f"मैं आपको {say} के बारे में बताता हूँ।"


def merge_suggestions(existing: list, new: list[dict]) -> list[dict]:
    by = {str(s.get("word", "")).lower(): s for s in existing or [] if isinstance(s, dict)}
    for s in new:
        by[s["word"].lower()] = s
    return list(by.values())


__all__ = ["candidates", "llm_respellings", "score", "phonetic", "sentence_for", "merge_suggestions", "norm_lexicon"]
