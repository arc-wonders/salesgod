"""tree_hint(): one deterministic read of the call -> what stage, what the caller cares about, which KB entries to use now
(not yet given), which offering fits and why, what was already given, and the next best move."""

from __future__ import annotations

import re
import zlib
from dataclasses import dataclass, field
from typing import Any

from ..knowledge import text as T
from ..knowledge.render import render_text
from ..knowledge.retrieve import index_for, retrieve
from ..knowledge.schema import Entry, KnowledgeBase, Offering
from ..knowledge.store import load_kb
from . import state as S
from .interest import DIRECT_MIN, Cluster, infer, ranked, stated_facts, tag_label
from .rules import (EXPLORED_FOR_BRIDGE, QUESTION, STRONG_FIT, Ctx, next_move, stage_for)

MAX_COVERAGE_CHARS = 700


@dataclass
class PickedEntry:
    id: str
    topic: str
    kind: str            # "answer" (retrieved for the question) | "teach" (cluster-driven tip)
    score: float
    text: str            # rendered in the reply language; prices already stripped to slots by enrich
    raw: str = ""


@dataclass
class OfferingPick:
    id: str
    name: str
    fit: float
    why: str
    pitch: list[str]
    price_label: str = ""
    included: bool = False


@dataclass
class TreeHint:
    stage: str
    stage_rule: str
    clusters: list[dict]
    topic: str
    entries: list[PickedEntry]
    offering: OfferingPick | None
    given: list[str]
    move: str
    action: dict
    coverage: dict
    discovery_question: str = ""
    lang: str = "en"
    kb_found: bool = True
    notes: list[str] = field(default_factory=list)

    def coverage_text(self, limit: int = MAX_COVERAGE_CHARS) -> str:
        """Compact plain lines for the prompt (<= 700 chars): stage, interests, what is already told, next move."""
        parts = [f"Call map: stage {self.stage}; next move {self.move.replace('_', ' ')}."]
        if self.clusters:
            parts.append("Caller cares about: " + ", ".join(f"{c['label']} ({', '.join(c['evidence'][:2])})" if c["evidence"] else c["label"]
                                                                for c in self.clusters[:3]) + ".")
        if self.given:
            parts.append("Already told (do not repeat): " + "; ".join(self.given[:6]) + ".")
        cov = self.coverage
        if cov.get("kb_total"):
            parts.append(f"Covered {cov['given']} of {cov['kb_total']} product points.")
        if self.offering is not None and self.move == "bridge_offering":
            parts.append(f"Fits them: {self.offering.name} ({self.offering.why}).")
        s = " ".join(parts)
        if len(s) > limit:
            s = s[:limit - 1].rstrip() + "…"
        return s


_SENT = re.compile(r"(?<=[.?!।])\s+")


def _retrieve_multi(kb: KnowledgeBase, text: str, lang: str, product: Any, gset: set, top: list[Cluster]):
    """Retrieve per question sentence (a long, chatty turn dilutes one combined query), merge by score, prefer entries of the caller's
    interest cluster. Falls back to the words that carried the interest, then to the nearest topic overview."""
    sents = [x.strip() for x in _SENT.split(text) if x.strip()]
    qs = [x for x in sents if QUESTION.search(x)]
    ev = {w for c_ in top[:2] for w in c_.evidence}
    stated = [x for x in sents if x not in qs and ev and any(w in T.normalize(x) for w in ev)]
    queries = qs + stated + ([text] if len(sents) > 1 or not qs else [])
    pool: dict[str, tuple[float, Any]] = {}
    base = None
    for i, q in enumerate(queries):
        r = retrieve(kb, q, lang, k=5, product=product, exclude=gset)
        base = base or r
        if r.fallback:
            continue
        for h in r.hits:
            sc = h.score + (0.1 if q in qs else 0.0)
            if h.id not in pool or sc > pool[h.id][0]:
                pool[h.id] = (sc, h)
    if not pool and top:
        ev = " ".join(w for c_ in top[:2] for w in c_.evidence[:2])
        alt = retrieve(kb, ev, lang, k=5, product=product, exclude=gset) if ev else None
        if alt is not None and not alt.fallback:
            for h in alt.hits:
                pool[h.id] = (h.score, h)
    if not pool:
        return base
    tt = {c_.tag for c_ in top[:2]}
    ranked_hits = sorted(pool.values(), key=lambda x: (-(x[0] + (0.15 if set(x[1].entry.interests) & tt else 0.0)), x[1].id))
    hits = [h for _, h in ranked_hits[:3]]
    from ..knowledge.retrieve import RetrievalResult
    return RetrievalResult(hits, True, hits[0].entry.topic_path, False, lang, base.terms if base else [])


def _customer_texts(session: Any, text: str) -> list[str]:
    hist = [u.text for u in getattr(session, "history", []) if getattr(u, "role", "") == "customer"]
    if text and not (hist and hist[-1] == text):
        hist.append(text)
    dlg = getattr(session, "dialog", {}) or {}
    for k in ("goal", "interest", "need", "use_case"):
        if dlg.get(k):
            hist.append(str(dlg[k]))
    return hist


def _agent_texts(session: Any) -> list[str]:
    return [u.text for u in getattr(session, "history", []) if getattr(u, "role", "") == "agent"]


def given_from_history(kb: KnowledgeBase, session: Any) -> list[str]:
    """Entries the agent has already said, found by how much of an entry's answer an earlier agent line already carries."""
    agents = _agent_texts(session)
    if not agents:
        return []
    said: set[str] = set()
    for a in agents:
        said |= T.term_keys(a)
    out = []
    for e in sorted(kb.usable_entries(), key=lambda e: e.id):
        ks = {k for k in T.term_keys(e.answer.en) if not k.startswith("~")}
        if len(ks) >= 5 and len(ks & said) / len(ks) >= 0.7:
            out.append(e.id)
    return out


def _fit(o: Offering, cl: dict[str, Cluster]) -> tuple[float, list[str]]:
    scores = sorted(((cl[t].score, t) for t in o.who_suits if t in cl), reverse=True)
    if not scores:
        return 0.0, []
    fit = scores[0][0] + 0.5 * (scores[1][0] if len(scores) > 1 else 0.0)
    return round(fit, 3), [t for _, t in scores if cl[t].direct >= DIRECT_MIN or cl[t].score >= DIRECT_MIN]


def _priceful(e: Entry) -> bool:
    a = " ".join(e.answer.all())
    return "{price:" in a or "{offer_price}" in a or "{original_price}" in a or "{discount}" in a


def _pick_offering(kb: KnowledgeBase, cl: dict[str, Cluster], facts: list[str], asked_topics: set[str], st: dict,
                   product: Any, lang: str, ok: bool) -> tuple[OfferingPick | None, bool]:
    if not ok:
        return None, False
    best = None
    for o in sorted(kb.offering_map().values(), key=lambda o: o.id):
        if not kb.allow_synthetic and o.provenance == "synthetic":
            continue
        fit, tags = _fit(o, cl)
        if fit < o.min_fit or not any(cl[t].direct >= DIRECT_MIN for t in tags if t in cl):
            continue  # never pitch something unrelated to what they said
        if st["offered"].get(o.id, 0) >= 2:
            continue
        if best is None or fit > best[0]:
            best = (fit, o, tags)
    if best is None:
        return None, False
    fit, o, tags = best
    wants = [tag_label(kb, t).lower() for t in tags[:3]]
    why = f"fits their interest in {', '.join(wants)}" + (f"; {facts[0]}" if facts else "")
    pitch = list(o.pitch.en if lang not in ("hi", "hinglish") else (getattr(o.pitch, lang) or o.pitch.en))
    pick = OfferingPick(o.id, o.name.get(lang), fit, why, pitch, o.price_label.get(lang) if o.price_label else "", o.included)
    return pick, fit >= STRONG_FIT


def tree_hint(session: Any, signals: Any, text: str, product: Any, lang: str | None = None, commit: bool = True) -> TreeHint | None:
    """The deterministic read of this turn. Returns None when the product has no KB. `commit` folds what the caller asked into
    session.dialog["_kb"] (idempotent for a repeated call on the same turn)."""
    kb = load_kb(product)
    if kb is None:
        return None
    return tree_hint_kb(kb, session, signals, text, product, lang, commit)


def _sig(signals: Any, name: str, default: Any) -> Any:
    return getattr(signals, name, default) if signals is not None else default


def tree_hint_kb(kb: KnowledgeBase, session: Any, signals: Any, text: str, product: Any, lang: str | None = None,
                 commit: bool = True) -> TreeHint:
    lang = lang if lang in ("en", "hi", "hinglish") else (getattr(getattr(session, "expression", None), "language", None) or T.detect_lang(text))
    if lang not in ("en", "hi", "hinglish"):
        lang = "en"
    if signals is None:  # the pipeline keeps the fused TurnSignals in dialog["_signals"]
        raw = (getattr(session, "dialog", {}) or {}).get("_signals", "")
        if raw:
            try:
                from ..signals import TurnSignals
                signals = TurnSignals.model_validate_json(raw)
            except Exception:  # noqa: BLE001
                signals = None
    st = S.load(session)
    texts = _customer_texts(session, text)
    clusters = infer(kb, texts)
    top = ranked(clusters)
    ix = index_for(kb)

    given = list(st["given"])
    S.add_unique(given, given_from_history(kb, session))
    gset = set(given)

    ctxo = getattr(session, "context", None)
    decision = getattr(ctxo, "decision", "undecided")
    promised_link = bool(ctxo and ctxo.has("payment_link"))
    objection = _sig(signals, "objection", "none") or "none"
    intent = _sig(signals, "intent", "unknown") or "unknown"
    buying = float(_sig(signals, "buying_interest", 0.0) or 0.0)
    state = _sig(signals, "state", {}) or {}
    frustrated = float(state.get("frustration", 0.0)) >= 0.6
    declining = intent in ("deny", "goodbye", "opt_out", "callback") or decision == "declined"
    is_q = bool(text and QUESTION.search(text)) or _sig(signals, "question_topic", "none") in ("product",)
    customer_turns = len(texts) - (1 if any(k in (getattr(session, "dialog", {}) or {}) for k in ("goal", "interest", "need", "use_case")) else 0)

    # retrieval for the question (entries already given are skipped, so the LLM never gets a repeat)
    res = None
    if text and (is_q or top):
        res = _retrieve_multi(kb, text, lang, product, gset, top)
    picks: list[PickedEntry] = []
    topic_node = ""
    if res is not None and res.hits and not res.fallback and is_q:
        for h in res.hits:
            picks.append(PickedEntry(h.id, h.entry.topic_path, "answer", round(h.score, 3), h.text, h.entry.answer.en))
        topic_node = res.topic
    elif res is not None and res.fallback and is_q and res.hits:
        h = res.hits[0]
        picks.append(PickedEntry(h.id, res.topic or h.entry.topic_path, "answer", round(h.score, 3), h.text, h.entry.answer.en))
        topic_node = res.topic

    # offering fit (Layer 2 -> offering)
    asked_topics = set(st["topics"])
    explored_in_cluster = 0
    top_tags = {c.tag for c in top[:3]}
    pool = {e.id: e for e in kb.usable_entries()}
    for eid in set(st["asked"]) | gset:
        e = pool.get(eid)
        if e and set(e.interests) & top_tags:
            explored_in_cluster += 1
    comfortable = not frustrated and not declining
    offering, strong = _pick_offering(kb, clusters, stated_facts(texts), asked_topics, st, product, lang, comfortable)
    offering_ready = offering is not None and explored_in_cluster >= EXPLORED_FOR_BRIDGE
    pitched = bool(offering and st["offered"].get(offering.id, 0) >= 1)

    c = Ctx(customer_turns, text or "", intent, objection, buying, decision, promised_link, is_q, bool(top), offering_ready or (offering is not None and strong))
    stage, rule = stage_for(c)

    if stage in ("negotiate", "close", "after_close"):
        picks = []   # price talks, closing and after-close are not the moment to pour in product detail
    elif stage == "objection":
        picks = [p for p in picks if p.kind == "answer"]
    # fill with cluster-driven teaching tips when the question did not fill three slots
    if stage not in ("after_close", "close", "objection", "negotiate") and len(picks) < 3 and top:
        have = {p.id for p in picks}
        weights = {c_.tag: c_.score for c_ in top[:3]}
        cand = []
        for e in sorted(kb.usable_entries(), key=lambda e: e.id):
            if e.id in gset or e.id in have or _priceful(e):
                continue
            aff = sum(weights.get(t, 0.0) for t in e.interests)
            if aff <= 0:
                continue
            same = bool(picks) and e.topic_path.split("/")[0] == picks[0].topic.split("/")[0]
            bonus = (0.4 if e.teach else 0.0) + (0.3 if same else 0.0) + (0.2 if e.topic_path.split("/")[0] not in {t.split("/")[0] for t in asked_topics} else 0.0)
            cand.append((-(aff + bonus), -e.confidence, e.id, e))
        cand.sort(key=lambda x: x[:3])
        for neg, _, _, e in cand[:3 - len(picks)]:
            picks.append(PickedEntry(e.id, e.topic_path, "teach", round(-neg, 3), render_text(e.answer.get(lang), lang, kb, product), e.answer.en))
        if not topic_node and picks:
            topic_node = picks[0].topic

    entries_left = bool(picks)
    move = next_move(stage, offering is not None, pitched, strong, explored_in_cluster >= EXPLORED_FOR_BRIDGE, entries_left, comfortable)
    if move == "bridge_offering" and offering is None:
        move = "answer" if picks else "ask_discovery"

    dq = ""
    if move == "ask_discovery":
        pref = [c_.tag for c_ in top[:1]]
        tags_with_q = [t for t in kb.interests if (getattr(t.discovery, lang) or t.discovery.en)]
        pool_tags = [t for t in tags_with_q if t.id in pref] or tags_with_q[:1]
        # ask about an interest we know nothing about yet; once something is known, dig into it
        unknown = [t for t in tags_with_q if t.id not in clusters]
        t = (pool_tags[0] if top and pool_tags else (unknown[0] if unknown else (tags_with_q[0] if tags_with_q else None)))
        if t is not None:
            qs = getattr(t.discovery, lang) or t.discovery.en
            dq = qs[0] if qs else ""

    action: dict = {"type": move}
    if move in ("answer", "teach") and picks:
        action["entry"] = picks[0].id
    elif move == "bridge_offering" and offering:
        action["offering"] = offering.id
    elif move == "ask_discovery":
        action["question"] = dq

    if commit:
        key = f"{getattr(session, 'turn_index', 0)}|{zlib.crc32((text or '').encode('utf-8'))}"
        if st["seen"] != key:
            st["seen"] = key
            if res is not None and res.hits and not res.fallback and is_q:
                S.add_unique(st["asked"], [h.id for h in res.hits[:1]])
                top_path = res.hits[0].entry.topic_path
                st["topics"][top_path] = st["topics"].get(top_path, 0) + 1
        st["total"] = len(pool)
        tt: dict[str, int] = {}
        for e in pool.values():
            tt[e.topic_path] = tt.get(e.topic_path, 0) + 1
        st["topic_total"] = tt
        S.add_unique(st["given"], given_from_history(kb, session))
        S.save(session, st)

    cov = S.coverage_summary(session) if commit else {"kb_total": len(pool), "given": len(given), "asked": len(st["asked"]), "given_pct": 0.0,
                                                      "topics_explored": {}, "offered": dict(st["offered"]), "given_ids": given, "asked_ids": list(st["asked"])}
    tmap = kb.entry_map()
    given_labels = [tmap[g].question_variants.en[0] if tmap[g].question_variants.en else g for g in given if g in tmap]
    cl_out = [{"tag": c_.tag, "label": tag_label(kb, c_.tag), "score": c_.score, "evidence": c_.evidence} for c_ in top[:4]]
    return TreeHint(stage, rule, cl_out, topic_node, picks[:3], offering, given_labels, move, action, cov, dq, lang)
