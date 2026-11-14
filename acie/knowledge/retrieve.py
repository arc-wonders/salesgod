"""Deterministic retrieval: question -> ranked KB entries in the reply language.

Score of an entry = best of (question variants, 0.55 x answer text), each = 0.75 x idf-weighted coverage of the query's
terms (exact / synonym concept / cross-script phonetic skeleton) + 0.25 x cosine of crc32-hashed char 3-grams.
Ties: score desc, confidence desc, id asc. Nothing is random; nothing depends on hash() or dict order.
A question nothing matches returns the nearest topic's OVERVIEW (never "contact the owner")."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from . import text as T
from .render import render_answer, render_text
from .schema import Entry, KnowledgeBase, Text3
from .store import load_kb

CONFIDENT = 0.42


@dataclass
class Hit:
    entry: Entry
    score: float
    text: str
    lang: str
    overview: bool = False

    @property
    def id(self) -> str:
        return self.entry.id


@dataclass
class RetrievalResult:
    hits: list[Hit]
    confident: bool
    topic: str = ""
    fallback: bool = False
    lang: str = "en"
    terms: list[str] = field(default_factory=list)

    @property
    def top(self) -> Hit | None:
        return self.hits[0] if self.hits else None


class _Doc:
    __slots__ = ("keys", "vec", "n")

    def __init__(self, text: str):
        self.keys = T.term_keys(text)
        self.vec = T.ngram_vec(text)
        self.n = len(T.content_terms(text))


class Index:
    def __init__(self, kb: KnowledgeBase):
        self.kb = kb
        self.entries = sorted(kb.usable_entries(), key=lambda e: e.id)
        self.topics = kb.topic_map()
        self.variants: dict[str, list[_Doc]] = {}
        self.answer: dict[str, _Doc] = {}
        self.all_keys: dict[str, set[str]] = {}
        df: dict[str, int] = {}
        for e in self.entries:
            vs = [_Doc(q) for q in e.question_variants.all()]
            ans = _Doc(" ".join(e.answer.all()))
            self.variants[e.id] = vs
            self.answer[e.id] = ans
            ks: set[str] = set(ans.keys)
            for d in vs:
                ks |= d.keys
            title = self.topics[e.topic_path].title
            ks |= T.term_keys(" ".join(title.all()))
            self.all_keys[e.id] = ks
            for k in ks:
                df[k] = df.get(k, 0) + 1
        self.n = max(1, len(self.entries))
        self.df = df

    def idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        sk = "~" + T.skeleton(term)
        if not term.startswith("c:") and sk in self.df:
            n = max(n, self.df[sk])
        return math.log(1.0 + self.n / (1.0 + n)) + 0.1

    def query_terms(self, text: str) -> list[tuple[str, str | None]]:
        out, seen = [], set()
        for t in T.content_terms(text):
            if t in seen:
                continue
            seen.add(t)
            sk = None if t.startswith("c:") else T.skeleton(t)
            out.append((t, ("~" + sk) if sk and len(sk) >= 3 else None))
        return out

    def _cover(self, qterms: list[tuple[str, str | None]], doc: _Doc) -> tuple[float, float]:
        tot = got = 0.0
        for t, sk in qterms:
            if self.df.get(t, 0) == 0 and not (sk and self.df.get(sk, 0)):
                continue  # a word nothing in the KB uses is noise, not evidence against an entry
            w = self.idf(t)
            tot += w
            if t in doc.keys:
                got += w
            elif sk and sk in doc.keys:
                got += 0.8 * w
        return (got / tot if tot else 0.0), got

    def score(self, qterms, qvec, e: Entry) -> float:
        best = 0.0
        for d in self.variants[e.id]:
            cov, got = self._cover(qterms, d)
            if cov <= 0:
                continue
            dtot = sum(self.idf(k) for k in d.keys if not k.startswith("~")) or 1.0
            spec = 0.7 + 0.3 * min(1.0, got / dtot)
            sc = (0.75 * cov + 0.25 * T.cosine(qvec, d.vec)) * spec * min(1.0, got / 1.2)
            if "c:price" in d.keys and not any(t == "c:price" for t, _ in qterms):
                sc *= 0.85  # a price question is not the answer to a question that never mentioned price
            best = max(best, sc)
        a = self.answer[e.id]
        cov, cov_got = self._cover(qterms, a)
        best = max(best, 0.55 * (0.75 * cov + 0.25 * T.cosine(qvec, a.vec)) * min(1.0, cov_got / 1.2))
        return best


_INDEX_CACHE: dict[int, Index] = {}


def index_for(kb: KnowledgeBase) -> Index:
    ix = _INDEX_CACHE.get(id(kb))
    if ix is None or ix.kb is not kb:
        if len(_INDEX_CACHE) > 16:
            _INDEX_CACHE.clear()
        ix = _INDEX_CACHE[id(kb)] = Index(kb)
    return ix


def _overview_entry(kb: KnowledgeBase, path: str | None) -> Entry:
    """A virtual entry carrying a topic's (or the product's) overview text."""
    topics = kb.topic_map()
    ents = [e for e in kb.usable_entries() if path is None or e.topic_path == path or e.topic_path.startswith(path + "/")]
    prov = "operator"
    if ents:
        provs = sorted({e.provenance for e in ents})
        prov = provs[0] if len(provs) == 1 else ("public" if "public" in provs else provs[0])
    if path and path in topics:
        ov, title = topics[path].overview, topics[path].title
    else:
        ov = kb.root_overview or Text3(en=kb.description or "Happy to tell you more.")
        title = Text3(en="overview")
    from .schema import Variants
    return Entry(id=f"overview-{(path or 'root').replace('/', '-')}", topic_path=path if path in topics else sorted(topics)[0],
                 question_variants=Variants(en=[title.en]), answer=ov, provenance=prov, confidence=0.5)


def retrieve(kb: KnowledgeBase | Any, question: str, lang: str | None = None, k: int = 3, product: Any = None,
             exclude: set[str] | frozenset[str] = frozenset()) -> RetrievalResult:
    """Top-k entries for a question. `kb` may be a KnowledgeBase or a product (its KB is loaded). `exclude` = entry ids to skip."""
    if not isinstance(kb, KnowledgeBase):
        product = product if product is not None else kb
        kb = load_kb(kb)
    lang = lang if lang in ("en", "hi", "hinglish") else T.detect_lang(question)
    if kb is None:
        return RetrievalResult([], False, lang=lang)
    ix = index_for(kb)
    qterms = ix.query_terms(question)
    qvec = T.ngram_vec(question)
    scored: list[tuple[float, float, str, Entry]] = []
    for e in ix.entries:
        if e.id in exclude:
            continue
        s = ix.score(qterms, qvec, e) if qterms or qvec else 0.0
        scored.append((-round(s, 6), -e.confidence, e.id, e))
    scored.sort(key=lambda x: (x[0], x[1], x[2]))
    top = [(-s, e) for s, _, _, e in scored[:max(1, k)] if -s > 0.0]
    confident = bool(top) and top[0][0] >= CONFIDENT
    if confident:
        floor = max(CONFIDENT * 0.6, top[0][0] * 0.55)
        hits = [Hit(e, s, render_answer(e, lang, kb, product), lang) for s, e in top if s >= floor][:k]
        return RetrievalResult(hits, True, hits[0].entry.topic_path, False, lang, [t for t, _ in qterms])
    # nothing convincing: the nearest topic's overview (never "contact the owner")
    agg: dict[str, float] = {}
    for s, _, _, e in scored[:12]:
        agg[e.topic_path] = agg.get(e.topic_path, 0.0) - s
    q_title = {p: sum(1 for t, _ in qterms if t in T.term_keys(" ".join(tp.title.all()))) for p, tp in ix.topics.items()}
    for p, n in q_title.items():
        agg[p] = agg.get(p, 0.0) + 0.3 * n
    best = None
    if any(v > 0 for v in agg.values()):
        best = sorted(agg.items(), key=lambda kv: (-round(kv[1], 6), kv[0]))[0][0]
    ov = _overview_entry(kb, best)
    hit = Hit(ov, top[0][0] if top else 0.0, render_text(ov.answer.get(lang), lang, kb, product).strip(), lang, overview=True)
    return RetrievalResult([hit], False, best or "", True, lang, [t for t, _ in qterms])
