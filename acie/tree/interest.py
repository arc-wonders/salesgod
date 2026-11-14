"""Layer 2: the caller's INTEREST CLUSTERS, inferred only from what they said (and facts the dialog holds).

Each interest tag in the KB carries keywords. A customer utterance that hits a tag adds 1 (+0.5 for a second distinct
keyword, capped at 1.5); related tags receive 0.3 x of it, never more than their own direct score would be. Pure
function of the utterances: same input, same output."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..knowledge import text as T
from ..knowledge.schema import InterestTag, KnowledgeBase

DIRECT_MIN = 0.8   # a tag counts as "stated" only with a direct hit (not just spread from a related tag)


@dataclass
class Cluster:
    tag: str
    score: float
    direct: float
    evidence: list[str] = field(default_factory=list)


_KW_CACHE: dict[int, dict[str, list[tuple[str, bool]]]] = {}


def _kw_table(kb: KnowledgeBase) -> dict[str, list[tuple[str, bool]]]:
    tab = _KW_CACHE.get(id(kb))
    if tab is None:
        tab = {}
        for tag in kb.interests:
            kws = []
            for k in sorted(set(tag.keywords)):
                n = T.normalize(k)
                if n:
                    kws.append((n, " " in n))
            tab[tag.id] = kws
        _KW_CACHE.clear()
        _KW_CACHE[id(kb)] = tab
    return tab


def _hits(norm: str, toks: set[str], kws: list[tuple[str, bool]]) -> list[str]:
    padded = f" {norm} "
    out = []
    for kw, multi in kws:
        if multi:
            if f" {kw} " in padded:
                out.append(kw)
        elif kw in toks:
            out.append(kw)
        elif len(kw) >= 5:  # cross-script / spelling variants of the same word (powerlifter ~ powerlifting handled by listing both)
            sk = T.skeleton(kw)
            if len(sk) >= 4 and any(len(t) >= 5 and T.skeleton(t) == sk for t in toks):
                out.append(kw)
    return out


def infer(kb: KnowledgeBase, utterances: list[str]) -> dict[str, Cluster]:
    tab = _kw_table(kb)
    direct: dict[str, float] = {t.id: 0.0 for t in kb.interests}
    ev: dict[str, list[str]] = {t.id: [] for t in kb.interests}
    for u in utterances:
        norm = T.normalize(u)
        toks = set(norm.split())
        for tag in kb.interests:
            hits = _hits(norm, toks, tab[tag.id])
            if hits:
                direct[tag.id] += 1.0 + (0.5 if len(hits) > 1 else 0.0)
                for h in hits:
                    if h not in ev[tag.id]:
                        ev[tag.id].append(h)
    spread: dict[str, float] = {t.id: 0.0 for t in kb.interests}
    for tag in kb.interests:
        for r in tag.related:
            if r in spread and direct[tag.id]:
                spread[r] += 0.3 * direct[tag.id]
    total = {t: direct[t] + min(spread[t], 1.0) for t in direct}
    return {t: Cluster(t, round(total[t], 3), round(direct[t], 3), ev[t][:4]) for t in sorted(total) if total[t] > 0}


def ranked(clusters: dict[str, Cluster]) -> list[Cluster]:
    return sorted(clusters.values(), key=lambda c: (-c.score, c.tag))


_DAYS = re.compile(r"\b(\d|one|two|three|four|five|six|seven)(?:\s*(?:to|-|or)\s*(\d|one|two|three|four|five|six|seven))?\s*(?:days?|din)\b", re.I)


def stated_facts(utterances: list[str]) -> list[str]:
    """Small concrete things the caller said that make a recommendation specific (frequency of training / use)."""
    out: list[str] = []
    for u in utterances:
        m = _DAYS.search(u)
        if m:
            a, b = m.group(1).lower(), (m.group(2) or "").lower()
            f = f"{a} to {b} days a week" if b else f"{a} days a week"
            if f not in out:
                out.append(f)
    return out[:2]


def tag_label(kb: KnowledgeBase, tag: str) -> str:
    for t in kb.interests:
        if t.id == tag:
            return t.label.en
    return tag
