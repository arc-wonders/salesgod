"""Phone-length rules for everything the agent speaks (FitZone's hand-written product is the reference).

A spoken turn is built from these fields, so a 35-word pitch becomes a 40-word turn. After generation every over-length field gets ONE
model rewrite ("shorten" stage: same meaning, no new facts, numbers and {slots} kept); anything still too long is trimmed at a sentence
(or clause) boundary. Limits are English words; Hindi / Hinglish get HI_FACTOR more (more function words for the same content)."""

from __future__ import annotations

import asyncio
import re
from typing import Any, Callable, Iterator

# kind -> (max English words, target words for the rewrite, phrase?, max sentences, word cap when it uses 2 sentences)
# phrase = a fragment that continues a sentence (FitZone: "a free personal trainer session every week"): no full stop
LIMITS: dict[str, tuple[int, int, bool, int, int]] = {
    "pitch": (25, 18, False, 2, 25),
    "benefit": (10, 7, True, 1, 0),
    "faq": (22, 15, False, 2, 14),       # one sentence; two only when both are very short
    "discovery": (15, 12, False, 2, 12),
    "cta": (12, 9, True, 1, 0),
    "kb_answer": (22, 15, False, 2, 14),
    "point": (12, 8, False, 1, 0),      # a full short sentence: the runtime speaks a point followed by the ask
    "ask": (15, 12, False, 2, 12),
    "ladder": (15, 12, False, 2, 12),
}
HI_FACTOR = 1.15  # FitZone: Hindi pitch 22 words for 20 English
_SENT = re.compile(r"(?<=[.!?।])\s+")
_CLAUSE = re.compile(r"[,;:—–]")
_TAIL = set("and or but with to of for the a an in on at by from that which who aur ya ke ki ka se mein par jo".split())
_JOIN = {"and", "with", "or", "but", "while", "which", "aur", "ya", "lekin", "और", "या", "लेकिन"}
_TOKS = re.compile(r"\{[^}]+\}|[0-9०-९][0-9०-९,.]*")


class Slot:
    __slots__ = ("key", "lang", "kind", "get", "set")

    def __init__(self, key: str, lang: str, kind: str, get: Callable[[], str], set_: Callable[[str], None]):
        self.key, self.lang, self.kind, self.get, self.set = key, lang, kind, get, set_

    def _n(self, n: int) -> int:
        return n if self.lang == "en" else round(n * HI_FACTOR)

    @property
    def limit(self) -> int:
        return self._n(LIMITS[self.kind][0])

    @property
    def target(self) -> int:
        return self._n(LIMITS[self.kind][1])

    @property
    def phrase(self) -> bool:
        return LIMITS[self.kind][2]

    @property
    def max_sentences(self) -> int:
        return LIMITS[self.kind][3]

    @property
    def two_cap(self) -> int:
        return self._n(LIMITS[self.kind][4])

    def fits(self, text: str) -> bool:
        return fits(text, self.limit, self.max_sentences, self.two_cap)


def sentences(t: str) -> list[str]:
    return [x for x in _SENT.split(" ".join(str(t or "").split())) if x]


def fits(text: str, limit: int, max_sent: int = 2, two_cap: int = 0) -> bool:
    n, k = words(text), len(sentences(text))
    return n <= limit and (k <= 1 or (k <= max_sent and n <= (two_cap or limit)))


def words(t: str) -> int:
    return len(str(t or "").split())


def _item(obj: Any, k: Any) -> tuple[Callable[[], str], Callable[[str], None]]:
    def g() -> str:
        return str(obj[k] or "")

    def s(v: str) -> None:
        obj[k] = v
    return g, s


def slots(bundle: dict) -> Iterator[Slot]:
    """Every length-limited spoken text in a bundle ({product, kb, tree})."""
    p = bundle.get("product") or {}
    for k, lang in (("pitch_en", "en"), ("pitch_hi", "hi"), ("pitch_hinglish", "hinglish")):
        if p.get(k):
            yield Slot(f"product.{k}", lang, "pitch", *_item(p, k))
    for k, lang in (("benefits", "en"), ("benefits_hi", "hi"), ("benefits_hinglish", "hinglish")):
        for i, b in enumerate(p.get(k) or []):
            if b:
                yield Slot(f"product.{k}[{i}]", lang, "benefit", *_item(p[k], i))
    for i, f in enumerate(p.get("faqs") or []):
        for key, lang in (("answer", "en"), ("answer_hi", "hi"), ("answer_hinglish", "hinglish")):
            if f.get(key):
                yield Slot(f"product.faqs[{i}].{key}", lang, "faq", *_item(f, key))
    dq = p.get("discovery_question")
    if isinstance(dq, dict):
        for lang in list(dq):
            if dq.get(lang):
                yield Slot(f"product.discovery_question.{lang}", lang, "discovery", *_item(dq, lang))
    if p.get("call_to_action"):
        yield Slot("product.call_to_action", "en", "cta", *_item(p, "call_to_action"))
    for e in (bundle.get("kb") or {}).get("entries", []) or []:
        a = e.get("answer")
        if isinstance(a, dict):
            for lang in list(a):
                if a.get(lang):
                    yield Slot(f"kb.{e.get('id', '?')}.{lang}", lang, "kb_answer", *_item(a, lang))
    tree = bundle.get("tree") or {}

    def walk(nodes: list[dict]) -> Iterator[Slot]:
        for n in nodes or []:
            nid = n.get("id", "?")
            for key, lang in (("points", "en"), ("points_hi", "hi")):
                for i, x in enumerate(n.get(key) or []):
                    if x:
                        yield Slot(f"node.{nid}.{key}[{i}]", lang, "point", *_item(n[key], i))
            for key, lang in (("ask", "en"), ("ask_hi", "hi")):
                if n.get(key):
                    yield Slot(f"node.{nid}.{key}", lang, "ask", *_item(n, key))
            yield from walk(n.get("children") or [])
    yield from walk(tree.get("nodes") or [])
    for r in tree.get("ladder") or []:
        q = r.get("question")
        if isinstance(q, dict):
            for lang in list(q):
                if q.get(lang):
                    yield Slot(f"ladder.{r.get('id', '?')}.{lang}", lang, "ladder", *_item(q, lang))


def over_length(bundle: dict) -> list[Slot]:
    return [s for s in slots(bundle) if not s.fits(s.get())]


def phrase_style(t: str, lang: str) -> str:
    """FitZone style for a fragment: no final full stop; lower-case start unless the first word is a name / acronym."""
    t = t.strip().rstrip(".।").strip()
    if lang != "hi" and t:
        first = t.split()[0]
        if first[:1].isupper() and first[1:2].islower() and first.lower() in _COMMON:
            t = t[0].lower() + t[1:]
    return t


_COMMON = set(("a an the we you there it all multiple available passes tickets entry get enjoy access priority instant guaranteed official free unlimited exclusive easy fast safe secure book "
               "booking open every one two three our your with no up hassle best direct digital live seamless choose pick "
               "have send share select reserve grab talk let").split())


def trim(text: str, limit: int, phrase: bool = False, lang: str = "en", max_sent: int = 2, two_cap: int = 0) -> str:
    """Deterministic fallback: whole sentences while they fit; else the first clause that fits; else the first `limit` words."""
    t = " ".join(str(text or "").split())
    if fits(t, limit, max_sent, two_cap):
        return phrase_style(t, lang) if phrase else t
    out: list[str] = []
    for s in sentences(t):
        if not fits(" ".join(out + [s]), limit, max_sent, two_cap):
            break
        out.append(s)
    if out:
        res = " ".join(out)
    else:
        first = sentences(t)[0]
        cut = [m.start() for m in _CLAUSE.finditer(first) if limit // 2 <= words(first[:m.start()]) <= limit]
        if cut:
            ws = first[:cut[-1]].split()
        else:
            ws = first.split()[:limit]
            joins = [i for i, w in enumerate(ws) if w.lower() in _JOIN and i >= max(3, limit // 2)]
            if joins:  # "... after booking and sent to ..." -> cut before the conjunction, not mid-clause
                ws = ws[:joins[-1]]
            while len(ws) > 3 and ws[-1].lower().strip(",;:") in _TAIL:
                ws.pop()
        res = " ".join(ws).rstrip(",;:—–")
        if not phrase and res and res[-1] not in ".!?।":
            res += "?" if first.rstrip().endswith("?") else ("।" if lang == "hi" else ".")
    return phrase_style(res, lang) if phrase else res


def _keeps_tokens(old: str, new: str) -> bool:
    """A rewrite may drop a number or slot along with its clause, but never introduce one."""
    return set(_TOKS.findall(new)) <= set(_TOKS.findall(old))


async def shorten(bundle: dict, llm: Any, notes: list[str] | None = None, batch: int = 60) -> dict:
    """One model rewrite for every over-length field, then the deterministic trim for anything the rewrite did not fix."""
    todo = over_length(bundle)
    stats = {"over": len(todo), "rewritten": 0, "trimmed": 0}
    if todo and llm is not None:
        groups = [todo[i:i + batch] for i in range(0, len(todo), batch)]

        async def one(g: list[Slot]) -> dict:
            items = [{"key": s.key, "lang": s.lang, "kind": s.kind, "max_words": s.target,
                      "sentences": 2 if s.kind == "pitch" else 1, "text": s.get()} for s in g]
            try:
                return await llm.json("shorten", {"items": items})
            except Exception as e:  # noqa: BLE001 - the trim below still applies
                if notes is not None:
                    notes.append(f"shorten: model rewrite failed ({type(e).__name__}); trimmed instead")
                return {}
        results = await asyncio.gather(*(one(g) for g in groups))
        by_key = {s.key: s for s in todo}
        for r in results:
            for it in (r.get("items") or []) if isinstance(r, dict) else []:
                s = by_key.get(str(it.get("key", ""))) if isinstance(it, dict) else None
                new = " ".join(str(it.get("text", "")).split()) if s else ""
                if s and new and s.fits(new) and _keeps_tokens(s.get(), new):
                    s.set(phrase_style(new, s.lang) if s.phrase else new)
                    stats["rewritten"] += 1
    stats["trimmed"] = len(trim_all(bundle))
    for s in slots(bundle):  # phrase style for every fragment, also those that were already short
        if s.phrase:
            s.set(phrase_style(s.get(), s.lang))
        elif s.kind == "faq":  # FAQ answers are stored without a final full stop: the runtime adds one ("scanning..")
            s.set(s.get().rstrip().rstrip(".।").rstrip())
    if notes is not None and stats["over"]:
        notes.append(f"phone length: {stats['over']} over-length texts, {stats['rewritten']} rewritten, {stats['trimmed']} trimmed")
    return stats


def trim_all(bundle: dict) -> list[str]:
    out = []
    for s in over_length(bundle):
        s.set(trim(s.get(), s.limit, s.phrase, s.lang, s.max_sentences, s.two_cap))
        out.append(s.key)
    return out
