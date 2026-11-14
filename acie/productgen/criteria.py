"""Check a generated bundle against the spec criteria. Pure and deterministic.

A check = {id, label, passed, blocking, detail, items}. `blocking` checks (safety, consistency, languages, schema) stop a Publish;
quantity checks only warn. `items` names the entry / node ids at fault so the repair loop can target them."""

from __future__ import annotations

import re
from typing import Any, Iterator

from .spec import Effective, Spec

_DEV = re.compile(r"[ऀ-ॿ]")
_LETTER = re.compile(r"[^\W\d_]", re.U)

BANNED: dict[str, list[str]] = {
    "health or result guarantee": [
        r"\bcures?\b", r"\bcured\b", r"guarantee[d]?\s+(to\s+)?(lose|results?|weight|cure|fat|growth|gain)", r"100\s?%\s*(safe|natural|effective|guarantee[d]?|results?)",
        r"no side[- ]effects?", r"lose\s+\d+\s*(kg|kilos?|pounds?)\s+in", r"clinically\s+(proven|tested)", r"doctor[- ]recommended",
        r"गारंटी\s*(से|के साथ)?\s*(वजन|रिज़ल्ट|नतीजे)", r"100\s?%\s*गारंटी"],
    "financial guarantee": [r"guaranteed\s+returns?", r"assured\s+returns?", r"risk[- ]free", r"double\s+your\s+money", r"no\s+risk\b", r"पैसे\s*डबल", r"पक्का\s*मुनाफ़ा"],
    "false urgency": [r"\bonly\s+today\b", r"\blast\s+chance\b", r"\bhurry\b", r"limited\s+stock", r"offer\s+ends\s+(tonight|soon|today)", r"\bact\s+now\b",
                      r"\bfew\s+(seats|spots|slots|pieces)\s+left\b", r"सिर्फ़?\s*आज", r"आख़?िरी\s*मौका", r"जल्दी\s*कर(ें|ो)", r"\bsirf\s+aaj\b", r"aakhri\s+mauka", r"\bjaldi\s+kar(o|iye|en)\b"],
    "competitor disparagement": [
        r"(competitors?|other\s+(brands?|companies|apps|gyms|sellers|providers))\s+(are|is)\s+(bad|fake|scams?|frauds?|useless|terrible|worse|cheating)",
        r"\b(worse|inferior)\s+than\s+(other|our\s+competitors)", r"unlike\s+(others|competitors),?\s+who\s+(cheat|lie)", r"दूसरे\s*(ब्रांड|कंपनियाँ|कंपनियां)\s*(फ़र्ज़ी|धोखा|बेकार)",
        r"dusre\s+(brands?|companies)\s+(fake|dhoka|bekaar|bekar)"],
}
CERT = [r"\bISO[\s\-]?\d{3,5}\b", r"\bFSSAI\b", r"\bISI\s+(mark|certified)\b", r"\bBIS\s+(mark|certified)\b", r"\bFDA[\s\-]approved\b", r"\bHACCP\b", r"\bGMP\b",
        r"\bAYUSH\b", r"award[\s\-]winning", r"dermatologically\s+tested", r"clinically\s+(proven|tested)", r"\bcertified\s+by\b"]
CERT_REAL_EXTRA = [r"\bcertified\b", r"\baccredited\b", r"\bapproved\s+by\b", r"\blicen[sc]ed\b"]


def _comp(p: str) -> re.Pattern:
    return re.compile(p, re.I)


def _t3(v: Any) -> list[tuple[str, str]]:
    if isinstance(v, dict):
        return [(k, x) for k, x in v.items() if isinstance(x, str)]
    return []


def texts(bundle: dict) -> Iterator[tuple[str, str, str]]:
    """(where, lang, text) for every customer-facing string."""
    kb = bundle.get("kb") or {}
    for e in kb.get("entries", []):
        for lang, t in _t3(e.get("answer")):
            yield e.get("id", "?"), lang, t
        for lang, vs in (e.get("question_variants") or {}).items():
            for v in vs or []:
                yield e.get("id", "?"), lang, v
    for t in kb.get("topics", []):
        for key in ("title", "overview"):
            for lang, x in _t3(t.get(key)):
                yield "topic:" + t.get("path", "?"), lang, x
    for o in kb.get("offerings", []):
        for lang, x in _t3(o.get("name")):
            yield "offering:" + o.get("id", "?"), lang, x
        for lang, vs in (o.get("pitch") or {}).items():
            for v in vs or []:
                yield "offering:" + o.get("id", "?"), lang, v
    p = bundle.get("product") or {}
    for k, lang in (("pitch_en", "en"), ("pitch_hi", "hi"), ("pitch_hinglish", "hinglish"), ("knowledge", "en"), ("one_liner", "en")):
        if p.get(k):
            yield "product:" + k, lang, p[k]
    for k, lang in (("benefits", "en"), ("benefits_hi", "hi"), ("benefits_hinglish", "hinglish")):
        for b in p.get(k, []) or []:
            yield "product:" + k, lang, b
    for i, f in enumerate(p.get("faqs", []) or []):
        for lang, key in (("en", "answer"), ("hi", "answer_hi"), ("hinglish", "answer_hinglish")):
            if f.get(key):
                yield f"faq:{i}", lang, f[key]

    def walk(nodes: list[dict]):
        for n in nodes or []:
            nid = "node:" + n.get("id", "?")
            for x in n.get("points", []) or []:
                yield nid, "en", x
            for x in n.get("points_hi", []) or []:
                yield nid, "hi", x
            if n.get("ask"):
                yield nid, "en", n["ask"]
            if n.get("ask_hi"):
                yield nid, "hi", n["ask_hi"]
            yield from walk(n.get("children", []))
    yield from walk((bundle.get("tree") or {}).get("nodes", []))
    for r in (bundle.get("tree") or {}).get("ladder", []):
        for lang, x in _t3(r.get("question")):
            yield "ladder:" + r.get("id", "?"), lang, x


def count_nodes(nodes: list[dict]) -> int:
    return sum(1 + count_nodes(n.get("children", [])) for n in nodes or [])


def _check(id_: str, label: str, passed: bool, detail: str = "", items: list[str] | None = None, blocking: bool = True) -> dict:
    return {"id": id_, "label": label, "passed": bool(passed), "blocking": blocking, "detail": detail, "items": sorted(set(items or []))}


def _lang_ok(lang: str, text: str) -> bool:
    letters = _LETTER.findall(text)
    if not letters:
        return True
    dev = len(_DEV.findall(text))
    if lang == "hi":
        return dev / len(letters) >= 0.3
    if lang == "hinglish":
        return dev == 0
    return dev / len(letters) < 0.3


def banned_hits(text: str, spec: Spec, eff: Effective) -> list[str]:
    operator_text = " ".join([spec.notes, spec.price.validity, spec.compliance_notes, spec.website, " ".join(spec.economics.perks)]).lower()
    hits = []
    for label, pats in BANNED.items():
        for p in pats:
            m = _comp(p).search(text)
            if m and not (label == "false urgency" and m.group(0).lower() in operator_text):
                hits.append(f"{label}: '{m.group(0)}'")
    certs = CERT + (CERT_REAL_EXTRA if eff.real_brand else [])
    for p in certs:
        m = _comp(p).search(text)
        if m and m.group(0).lower() not in operator_text:
            hits.append(f"invented certification: '{m.group(0)}'")
    for ph in spec.banned_phrases:
        if ph and ph.lower() in text.lower():
            hits.append(f"operator-banned phrase: '{ph}'")
    return hits


def check(bundle: dict, spec: Spec, eff: Effective) -> dict:
    c = eff.criteria
    kb = bundle.get("kb") or {}
    tree = bundle.get("tree") or {}
    entries, topics, offs = kb.get("entries", []), kb.get("topics", []), kb.get("offerings", [])
    nodes = tree.get("nodes", [])
    ladder = tree.get("ladder", [])
    checks: list[dict] = []
    top = [t for t in topics if "/" not in t.get("path", "")]

    checks.append(_check("topics", f"Topics: {c.min_topics}-{c.max_topics} top-level", c.min_topics <= len(top) <= c.max_topics,
                         f"{len(top)} top-level, {len(topics)} in all", blocking=False))
    checks.append(_check("entries", f"Q&A entries: at least {c.min_entries}", len(entries) >= c.min_entries, f"{len(entries)} entries", blocking=False))
    checks.append(_check("offerings", f"Offerings: {c.min_offerings}-{c.max_offerings}", c.min_offerings <= len(offs) <= c.max_offerings,
                         f"{len(offs)} offerings", blocking=False))
    nn = count_nodes(nodes)
    top_n = len(nodes)
    checks.append(_check("situations", f"Playbook situations: {c.min_situations}-{c.max_situations}", c.min_situations <= top_n <= c.max_situations,
                         f"{top_n} top-level, {nn} with children", blocking=False))
    checks.append(_check("ladder", f"Discovery ladder depth: at least {c.min_ladder_depth}", len(ladder) >= c.min_ladder_depth, f"{len(ladder)} rungs", blocking=False))

    # languages
    miss: list[str] = []
    wrong: list[str] = []
    for lang_req in [x for x in spec.languages if c.require_all_languages or x == "en"]:
        for e in entries:
            a = (e.get("answer") or {}).get(lang_req, "")
            qv = (e.get("question_variants") or {}).get(lang_req) or []
            if not a.strip() or not qv:
                miss.append(e.get("id", "?"))
        for t in topics:
            if not ((t.get("title") or {}).get(lang_req) and (t.get("overview") or {}).get(lang_req)):
                miss.append("topic:" + t.get("path", "?"))
        for o in offs:
            if not (o.get("name") or {}).get(lang_req):
                miss.append("offering:" + o.get("id", "?"))
        for r in ladder:
            if not (r.get("question") or {}).get(lang_req):
                miss.append("ladder:" + r.get("id", "?"))
        if lang_req == "hi":
            for n in nodes:
                if not n.get("points_hi") or not n.get("ask_hi"):
                    miss.append("node:" + n.get("id", "?"))
    for where, lang, t in texts(bundle):
        if lang in spec.languages and not _lang_ok(lang, t):
            wrong.append(where)
    checks.append(_check("languages", "Every entry in " + " / ".join(spec.languages), not miss and not wrong,
                         (f"missing: {len(set(miss))}" if miss else "") + ("; wrong script in: " + str(len(set(wrong))) if wrong else "") or "all present",
                         miss + wrong))

    # provenance
    bad_prov: list[str] = []
    want = {"public"} if eff.real_brand else {"synthetic", "operator"}
    for e in entries:
        if e.get("provenance") not in want:
            bad_prov.append(e.get("id", "?"))
    for o in offs:
        if o.get("provenance") not in want:
            bad_prov.append("offering:" + o.get("id", "?"))
    checks.append(_check("provenance", "Provenance label on every entry" + (" (public only)" if eff.real_brand else ""), not bad_prov,
                         f"{len(bad_prov)} missing or not allowed" if bad_prov else "ok", bad_prov))

    # real brand
    if eff.real_brand:
        fails = []
        if kb.get("allow_synthetic") is not False:
            fails.append("allow_synthetic must be false")
        unreviewed = [e.get("id", "?") for e in entries if not str(e.get("source", "")).startswith("review:") and "review" not in str(e.get("source", ""))]
        if unreviewed:
            fails.append(f"{len(unreviewed)} claims not marked for review")
        invented = [o.get("id", "?") for o in offs if o.get("price_inr") is not None]
        if invented:
            fails.append("offerings with invented prices: " + ", ".join(invented))
        checks.append(_check("real_brand", "Real-brand rule: public facts only, marked for review", not fails, "; ".join(fails) or "ok", unreviewed + invented))

    # numeric consistency
    if c.check_numeric_consistency:
        probs, items = [], []
        floor = eff.floor
        op, lp = spec.price.offer_price, spec.price.list_price
        if op is not None and lp is not None and op >= lp:
            probs.append(f"offer {op} is not below list {lp}")
        if floor is not None:
            if op is not None and op < floor:
                probs.append(f"offer {op} below floor {floor}")
            for o in offs:
                pr = o.get("price_inr")
                if pr is not None and not o.get("included") and pr < floor:
                    probs.append(f"{o.get('id')}: {pr} below floor {floor}")
                    items.append("offering:" + o.get("id", "?"))
            for s in ((bundle.get("negotiation") or {}).get("economics") or {}).get("ladder", []):
                if s.get("kind") == "price" and float(s.get("price", 0)) < floor:
                    probs.append(f"ladder step {s.get('id')} below floor")
        for o in offs:
            if o.get("price_inr") is not None and o.get("floor_inr") is not None and o["price_inr"] < o["floor_inr"]:
                probs.append(f"{o.get('id')}: below its own floor")
                items.append("offering:" + o.get("id", "?"))
        for o in offs:
            if o.get("price_inr") is not None and not o.get("price_label"):
                probs.append(f"{o.get('id')}: price without a label")
                items.append("offering:" + o.get("id", "?"))
        checks.append(_check("numeric", "Prices consistent: every price >= floor, offer < list", not probs, "; ".join(probs) or "ok", items))

    # banned content
    if c.check_banned_content:
        bad: dict[str, list[str]] = {}
        for where, lang, t in texts(bundle):
            for h in banned_hits(t, spec, eff):
                bad.setdefault(where, []).append(h)
        checks.append(_check("banned", "No banned content (guarantees, false urgency, disparagement, invented certifications)", not bad,
                             "; ".join(f"{k}: {v[0]}" for k, v in sorted(bad.items())[:4]) or "clean", list(bad)))

    # schema and KB rules (slots, price literals, provenance, unknown refs)
    try:
        from acie.knowledge.schema import KnowledgeBase
        from acie.knowledge.store import validate
        kbm = KnowledgeBase.model_validate(kb)
        issues = validate(kbm, {"profit_floor": eff.floor} if eff.floor is not None else None)
    except Exception as ex:  # noqa: BLE001
        issues = [str(ex)[:300]]
    checks.append(_check("schema", "Knowledge base passes the KB schema and slot rules", not issues, "; ".join(issues[:3]) or "ok"))

    # coverage / consistency
    tags = {i.get("id") for i in kb.get("interests", [])}
    used = {t for e in entries for t in e.get("interests", [])} | {t for o in offs for t in o.get("who_suits", [])}
    unused = sorted(tags - used)
    topic_paths = {t.get("path") for t in topics}
    populated = {e.get("topic_path") for e in entries}
    populated |= {p.rsplit("/", 1)[0] for p in list(populated) if "/" in p}
    empty = sorted(p for p in topic_paths if p not in populated and not any(q.startswith(p + "/") for q in topic_paths))
    checks.append(_check("coverage", "Every topic has entries and every interest tag is used", not unused and not empty,
                         (f"empty topics: {len(empty)}; " if empty else "") + (f"unused tags: {len(unused)}" if unused else "") or "ok",
                         ["topic:" + p for p in empty] + ["tag:" + t for t in unused], blocking=False))
    dq: dict[str, list[str]] = {}
    for e in entries:
        k = re.sub(r"\W+", " ", ((e.get("question_variants") or {}).get("en") or [""])[0].lower()).strip()
        dq.setdefault(k, []).append(e.get("id", "?"))
    dup = [v for k, v in dq.items() if k and len(v) > 1]
    checks.append(_check("duplicates", "No duplicate questions", not dup, f"{len(dup)} duplicated questions" if dup else "ok", [i for v in dup for i in v[1:]], blocking=False))

    failed = [x for x in checks if not x["passed"]]
    return {"passed": not failed, "publishable": not any(x["blocking"] for x in failed), "checks": checks,
            "counts": {"topics_top": len(top), "topics": len(topics), "entries": len(entries), "offerings": len(offs), "situations": top_n,
                       "nodes": nn, "ladder": len(ladder), "interests": len(tags)}}
