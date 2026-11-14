"""Tolerant builders: turn raw LLM JSON into valid KB / playbook pieces, dropping and reporting what does not fit.
Everything here is deterministic. Provenance is forced from the effective spec, never trusted from the model."""

from __future__ import annotations

import re
from typing import Any

from acie import playbook as pbm
from acie.knowledge.schema import Entry, InterestTag, Offering, Topic

from .spec import LANGS, Effective, Spec

REVIEW = "review: public knowledge, verify before publishing"
_ID = re.compile(r"[^a-z0-9_\-]+")


def kebab(s: Any, fallback: str = "x") -> str:
    s = _ID.sub("-", str(s or "").strip().lower().replace(" ", "-")).strip("-_")
    return re.sub(r"-{2,}", "-", s)[:60] or fallback


def path_clean(p: Any) -> str:
    return "/".join(kebab(x) for x in str(p or "").split("/") if kebab(x, ""))


def t3(v: Any) -> dict[str, str]:
    if isinstance(v, str):
        return {"en": v.strip()}
    if isinstance(v, dict):
        return {k: str(v[k]).strip() for k in LANGS if isinstance(v.get(k), (str, int, float)) and str(v.get(k)).strip()}
    return {}


def variants(v: Any) -> dict[str, list[str]]:
    if isinstance(v, list):
        v = {"en": v}
    if isinstance(v, str):
        v = {"en": [v]}
    out: dict[str, list[str]] = {}
    if isinstance(v, dict):
        for k in LANGS:
            xs = v.get(k)
            if isinstance(xs, str):
                xs = [xs]
            if isinstance(xs, list):
                out[k] = [str(x).strip() for x in xs if str(x).strip()]
    return out


def money(n: float, cur: str = "₹") -> str:
    n = int(round(n))
    s = str(abs(n))
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        s = ",".join(parts + [tail])
    return f"{cur}{s}"


def price_label(price: int, unit: str = "") -> dict[str, str]:
    m = money(price)
    plain = m[1:]
    return {"en": f"{m} {unit}".strip(), "hi": f"{plain} रुपये {unit}".strip() if not unit else f"{plain} रुपये", "hinglish": f"{plain} rupaye {unit}".strip() if not unit else f"{plain} rupaye"}


def clean_topic(raw: dict, kb: dict, notes: list[str]) -> dict | None:
    try:
        d = {"path": path_clean(raw.get("path")), "title": t3(raw.get("title")), "overview": t3(raw.get("overview")),
             "interests": [kebab(i) for i in raw.get("interests", []) or []]}
        d["title"].setdefault("en", d["path"].split("/")[-1].replace("-", " ").title())
        d["overview"].setdefault("en", d["title"]["en"])
        paths = {t["path"] for t in kb["topics"]}
        if d["path"] in paths:
            return None
        if "/" in d["path"] and d["path"].rsplit("/", 1)[0] not in paths:
            notes.append(f"topic dropped: {d['path']}: parent missing")
            return None
        return Topic(**d).model_dump(mode="json")
    except Exception as ex:  # noqa: BLE001
        notes.append(f"topic dropped: {raw.get('path')}: {str(ex)[:80]}")
        return None


def add_topics(kb: dict, raws: list[dict], notes: list[str]) -> list[str]:
    added = []
    for r in sorted((x for x in raws if isinstance(x, dict)), key=lambda x: str(x.get("path", "")).count("/")):
        t = clean_topic(r, kb, notes)
        if t:
            kb["topics"].append(t)
            added.append(t["path"])
    return added


def clean_interest(raw: dict) -> dict | None:
    try:
        kw = [str(k).strip() for k in raw.get("keywords", []) or [] if str(k).strip()]
        d = {"id": kebab(raw.get("id")), "label": t3(raw.get("label")) or {"en": kebab(raw.get("id"))}, "keywords": kw or [kebab(raw.get("id")).replace("-", " ")],
             "related": [kebab(x) for x in raw.get("related", []) or []], "discovery": variants(raw.get("discovery"))}
        return InterestTag(**d).model_dump(mode="json")
    except Exception:  # noqa: BLE001
        return None


def add_interests(kb: dict, raws: list[dict]) -> list[str]:
    have = {i["id"] for i in kb["interests"]}
    out = []
    for r in raws or []:
        i = clean_interest(r) if isinstance(r, dict) else None
        if i and i["id"] not in have:
            kb["interests"].append(i)
            have.add(i["id"])
            out.append(i["id"])
    tags = {i["id"] for i in kb["interests"]}
    for i in kb["interests"]:
        i["related"] = [r for r in i["related"] if r in tags and r != i["id"]]
    return out


def clean_offering(raw: dict, kb: dict, spec: Spec, eff: Effective, notes: list[str]) -> dict | None:
    try:
        tags = {i["id"] for i in kb["interests"]}
        paths = {t["path"] for t in kb["topics"]}
        oid = kebab(raw.get("id"))
        price = raw.get("price_inr")
        price = int(round(float(price))) if isinstance(price, (int, float)) and not isinstance(price, bool) else None
        label = t3(raw.get("price_label")) or None
        if eff.real_brand:  # a real company's prices are theirs: never invented
            price, label = None, None
        if price is not None and eff.floor is not None and price < eff.floor:
            notes.append(f"{oid}: price {price} raised to the floor {eff.floor:g}")
            price, label = int(round(eff.floor)), None
        if price is not None and (label is None or "en" not in label):
            label = price_label(price)
        who = [w for w in (kebab(x) for x in raw.get("who_suits", []) or []) if w in tags]
        if not who and tags:
            notes.append(f"offering dropped: {oid}: no valid who_suits")
            return None
        d = {"id": oid, "name": t3(raw.get("name")), "who_suits": who or ["general"], "price_inr": price, "price_label": label,
             "pitch": variants(raw.get("pitch")), "upsell_from": [p for p in (path_clean(x) for x in raw.get("upsell_from", []) or []) if p in paths],
             "provenance": eff.provenance, "min_fit": float(raw.get("min_fit", 1.0) or 1.0), "included": bool(raw.get("included", False))}
        if not d["name"].get("en"):
            return None
        return Offering(**d).model_dump(mode="json")
    except Exception as ex:  # noqa: BLE001
        notes.append(f"offering dropped: {raw.get('id')}: {str(ex)[:80]}")
        return None


def add_offerings(kb: dict, raws: list[dict], spec: Spec, eff: Effective, notes: list[str], limit: int | None = None) -> list[str]:
    have = {o["id"] for o in kb["offerings"]}
    out = []
    for r in raws or []:
        if limit is not None and len(kb["offerings"]) >= limit:
            break
        o = clean_offering(r, kb, spec, eff, notes) if isinstance(r, dict) else None
        if o and o["id"] not in have:
            kb["offerings"].append(o)
            have.add(o["id"])
            out.append(o["id"])
    return out


def clean_entry(raw: dict, kb: dict, eff: Effective, notes: list[str], source: str = "") -> dict | None:
    try:
        tags = {i["id"] for i in kb["interests"]}
        offs = {o["id"] for o in kb["offerings"]}
        paths = {t["path"] for t in kb["topics"]}
        facts = {str(k): (v if isinstance(v, (int, float, str)) and not isinstance(v, bool) else str(v)) for k, v in (raw.get("facts") or {}).items()} \
            if isinstance(raw.get("facts"), dict) else {}
        src = str(raw.get("source", "") or "")
        if source:
            src = (src + " " + source).strip() if src and source not in src else (src or source)
        if eff.real_brand and "review" not in src:
            src = (REVIEW + (" | " + src if src else ""))[:300]
        try:
            conf = max(0.0, min(1.0, float(raw.get("confidence", 0.9))))
        except (TypeError, ValueError):
            conf = 0.9
        if eff.real_brand:
            conf = min(conf, 0.7)
        d = {"id": kebab(raw.get("id")), "topic_path": path_clean(raw.get("topic_path")), "question_variants": variants(raw.get("question_variants")),
             "answer": t3(raw.get("answer")), "facts": facts, "offerings": [o for o in (kebab(x) for x in raw.get("offerings", []) or []) if o in offs],
             "interests": [i for i in (kebab(x) for x in raw.get("interests", []) or []) if i in tags], "provenance": eff.provenance,
             "confidence": conf, "source": src, "teach": bool(raw.get("teach", False))}
        if d["topic_path"] not in paths:
            notes.append(f"entry dropped: {d['id']}: unknown topic {d['topic_path']}")
            return None
        return Entry(**d).model_dump(mode="json")
    except Exception as ex:  # noqa: BLE001
        notes.append(f"entry dropped: {raw.get('id')}: {str(ex)[:90]}")
        return None


def _qkey(e: dict) -> str:
    q = (e["question_variants"].get("en") or e["question_variants"].get("hinglish") or e["question_variants"].get("hi") or [""])[0]
    return re.sub(r"\W+", " ", q.lower()).strip()


def add_entries(kb: dict, raws: list[dict], eff: Effective, notes: list[str], source: str = "") -> list[str]:
    """Append valid, non-duplicate entries; ids that already exist are never overwritten (a new id is not invented either)."""
    have_ids = {e["id"] for e in kb["entries"]}
    have_q = {_qkey(e) for e in kb["entries"]}
    out = []
    for r in raws or []:
        if not isinstance(r, dict):
            continue
        e = clean_entry(r, kb, eff, notes, source)
        if not e:
            continue
        k = _qkey(e)
        if e["id"] in have_ids or (k and k in have_q):
            notes.append(f"entry skipped (duplicate): {e['id']}")
            continue
        kb["entries"].append(e)
        have_ids.add(e["id"])
        have_q.add(k)
        out.append(e["id"])
    return out


def replace_entries(kb: dict, raws: list[dict], eff: Effective, notes: list[str]) -> list[str]:
    """Swap in corrected versions of existing entries (same id). A correction that is itself invalid leaves the old one."""
    idx = {e["id"]: i for i, e in enumerate(kb["entries"])}
    out = []
    for r in raws or []:
        if not isinstance(r, dict) or kebab(r.get("id")) not in idx:
            continue
        old = kb["entries"][idx[kebab(r.get("id"))]]
        e = clean_entry({**old, **{k: v for k, v in r.items() if v}}, kb, eff, notes, "")
        if e:
            kb["entries"][idx[e["id"]]] = e
            out.append(e["id"])
    return out


# ---------------------------------------------------------------- playbook nodes

def clean_node(raw: dict, taken: set[str], notes: list[str]) -> dict | None:
    try:
        w = raw.get("when") or {}
        when = {"keywords": [str(k) for k in w.get("keywords", []) or [] if str(k).strip()],
                "intents": [i for i in w.get("intents", []) or [] if i in pbm.INTENTS],
                "objections": [o for o in w.get("objections", []) or [] if o in pbm.OBJECTIONS],
                "topics": [t for t in w.get("topics", []) or [] if t in pbm.TOPICS]}
        for k in ("min_buying", "max_buying", "min_frustration", "min_confusion"):
            if isinstance(w.get(k), (int, float)) and not isinstance(w.get(k), bool):
                when[k] = max(0.0, min(1.0, float(w[k])))
        nid = kebab(raw.get("id") or raw.get("title"))
        if nid in taken:
            return None
        taken.add(nid)
        kids = []
        for c in raw.get("children", []) or []:
            ck = clean_node(c, taken, notes) if isinstance(c, dict) else None
            if ck:
                kids.append(ck)
        node = pbm.Node(id=nid, title=str(raw.get("title") or nid), when=pbm.When(**when), points=[str(p) for p in raw.get("points", []) or []],
                        ask=str(raw.get("ask", "") or ""), points_hi=[str(p) for p in raw.get("points_hi", []) or []], ask_hi=str(raw.get("ask_hi", "") or ""),
                        goal=str(raw.get("goal", "") or ""), children=[pbm.Node.model_validate(k) for k in kids])
        return node.model_dump(mode="json", exclude_none=True)
    except Exception as ex:  # noqa: BLE001
        notes.append(f"node dropped: {raw.get('id')}: {str(ex)[:80]}")
        return None


def all_node_ids(nodes: list[dict]) -> set[str]:
    out: set[str] = set()
    for n in nodes:
        out.add(n["id"])
        out |= all_node_ids(n.get("children", []))
    return out


def add_nodes(nodes: list[dict], raws: list[dict], notes: list[str], limit: int | None = None) -> list[str]:
    taken = all_node_ids(nodes)
    out = []
    for r in raws or []:
        if limit is not None and len(nodes) >= limit:
            break
        n = clean_node(r, taken, notes) if isinstance(r, dict) else None
        if n:
            nodes.append(n)
            out.append(n["id"])
    return out


def find_node(nodes: list[dict], nid: str) -> dict | None:
    for n in nodes:
        if n["id"] == nid:
            return n
        k = find_node(n.get("children", []), nid)
        if k:
            return k
    return None


def clean_ladder(raws: list[dict]) -> list[dict]:
    out, seen = [], set()
    for i, r in enumerate(raws or []):
        if not isinstance(r, dict):
            continue
        q = t3(r.get("question"))
        rid = kebab(r.get("id") or r.get("level") or f"rung-{i + 1}")
        if not q.get("en") or rid in seen:
            continue
        seen.add(rid)
        out.append({"id": rid, "level": str(r.get("level") or rid), "question": q, "purpose": str(r.get("purpose", "") or ""),
                    "interests": [kebab(x) for x in r.get("interests", []) or []]})
    return out


def clusters(kb: dict) -> list[dict]:
    """Interest clusters = connected components of the tags' `related` graph. Deterministic."""
    tags = {i["id"]: i for i in kb["interests"]}
    seen: set[str] = set()
    out = []
    for tid in sorted(tags):
        if tid in seen:
            continue
        comp, stack = [], [tid]
        while stack:
            x = stack.pop()
            if x in seen or x not in tags:
                continue
            seen.add(x)
            comp.append(x)
            stack += [r for r in tags[x]["related"] if r in tags]
            stack += [o for o, t in tags.items() if x in t["related"]]
        comp.sort()
        out.append({"id": "cluster-" + comp[0], "tags": comp, "keywords": sorted({k for c in comp for k in tags[c]["keywords"]})[:12]})
    return out
