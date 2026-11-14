"""Helpers for hand-authoring a KB as compact Python data (the JSON under data/voice/knowledge is built from these)."""
from __future__ import annotations

from acie.knowledge.schema import Entry, InterestTag, KnowledgeBase, Offering, Text3, Topic, Variants


def _v(s: str) -> list[str]:
    return [x.strip() for x in s.split("|") if x.strip()]


def E(id, topic, qe, qh, ql, ae, ah, al, tags=(), off=(), facts=None, teach=False, conf=1.0, source="") -> dict:
    return dict(id=id, topic_path=topic, question_variants=dict(en=_v(qe), hi=_v(qh), hinglish=_v(ql)),
                answer=dict(en=ae, hi=ah, hinglish=al), interests=list(tags), offerings=list(off), facts=facts or {},
                teach=teach, confidence=conf, source=source)


def TP(path, te, th, tl, oe, oh, ol, tags=()) -> dict:
    return dict(path=path, title=dict(en=te, hi=th, hinglish=tl), overview=dict(en=oe, hi=oh, hinglish=ol), interests=list(tags))


def TAG(id, le, lh, ll, kws, related=(), dq=("", "", "")) -> dict:
    return dict(id=id, label=dict(en=le, hi=lh, hinglish=ll), keywords=_v(kws), related=list(related),
                discovery=dict(en=[dq[0]] if dq[0] else [], hi=[dq[1]] if dq[1] else [], hinglish=[dq[2]] if dq[2] else []))


def OFF(id, ne, nh, nl, suits, price, label, pe, ph, pl, upsell=(), included=False, min_fit=1.0, floor=None) -> dict:
    return dict(id=id, name=dict(en=ne, hi=nh, hinglish=nl), who_suits=list(suits), price_inr=price,
                price_label=dict(en=label[0], hi=label[1], hinglish=label[2]) if label else None, included=included,
                floor_inr=floor, pitch=dict(en=_v(pe), hi=_v(ph), hinglish=_v(pl)), upsell_from=list(upsell), min_fit=min_fit)


def build(product_id, tenant, allow_synth, prov, description, topics, entries, offerings, tags, root, off_prov=None) -> KnowledgeBase:
    es = [Entry(provenance=prov, **e) for e in entries]
    return KnowledgeBase(product_id=product_id, tenant=tenant, allow_synthetic=allow_synth, description=description,
                         topics=[Topic(**t) for t in topics], entries=es,
                         offerings=[Offering(provenance=off_prov or prov, **o) for o in offerings],
                         interests=[InterestTag(**t) for t in tags], root_overview=Text3(**root))
