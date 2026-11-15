"""Product studio: one editable DRAFT per (tenant, product) on top of the productgen version store.

data/voice/productgen/<tenant>/<pid>/draft.json   the working copy the Studio edits (facts, product, tree, kb, strategy, personas)
data/voice/productgen/<tenant>/<pid>/v<N>.json    immutable versions (store.py); Publish = save the draft as v<N> + store.publish(N)

Facts: every sellable claim is a Fact {id, kind, label, field, value, critical, status, source}. Critical facts (price, offer, validity,
terms, guarantees, anything with a number) start "needs_confirmation" and Publish is refused until a human confirms each one. Confirming
or editing a fact writes its value back into the product field it came from (`field`), so the runtime files stay the same shape.
Live calls are never touched until Publish."""

from __future__ import annotations

import copy
import json
import re
import time
from pathlib import Path
from typing import Any

from . import store as S

CRITICAL_KINDS = {"price", "offer", "validity", "terms", "guarantee", "floor", "link"}
STATUSES = ("ai_draft", "needs_confirmation", "confirmed", "rejected")
_RISKY = re.compile(r"[0-9०-९]|guarantee|refund|money.?back|certified|100\s?%|गारंटी|रिफंड", re.I)
_NUM = re.compile(r"(?<![\w.])(?:₹|rs\.?\s?|\$)?\s?([0-9][0-9,]*(?:\.[0-9]+)?)", re.I)

# product field -> (kind, label, always critical)
SCALARS = [
    ("product_name", "identity", "Product name", False), ("seller_name", "identity", "Company", False),
    ("one_liner", "what", "What it is", False), ("original_price", "price", "Normal price", True),
    ("offer_price", "offer", "Offer price", True), ("discount", "offer", "Discount", True),
    ("offer_validity", "validity", "Offer validity", True), ("terms", "terms", "Full offer terms", True),
    ("renewal_note", "terms", "After the offer", True), ("call_to_action", "close", "Call to action", False),
    ("payment_link", "link", "Payment / joining link", True), ("knowledge", "what", "Background the agent may use", False),
]
ECON = [("list_price", "Negotiation: opening price"), ("cost", "Negotiation: cost"), ("min_margin_pct", "Negotiation: minimum margin %"),
        ("floor_price", "Negotiation: explicit floor")]


def now() -> float:
    return round(time.time(), 3)


def _empty(v: Any) -> bool:
    return v in (None, "", [], {}, 0, 0.0) or (isinstance(v, dict) and not any(str(x).strip() for x in v.values()))


def _text(v: Any) -> str:
    if isinstance(v, dict):
        return " ".join(str(x) for x in v.values())
    if isinstance(v, list):
        return " ".join(_text(x) for x in v)
    return "" if v is None else str(v)


def fact(fid: str, kind: str, label: str, field: str, value: Any, source: dict, status: str | None = None, critical: bool | None = None) -> dict:
    crit = critical if critical is not None else (kind in CRITICAL_KINDS or bool(_RISKY.search(_text(value))))
    if status is None:
        status = "needs_confirmation" if crit else "confirmed"
    return {"id": fid, "kind": kind, "label": label, "field": field, "value": value, "critical": crit, "status": status,
            "source": source, "updated_at": now()}


def facts_from_product(p: dict, source: dict, noncritical_status: str = "confirmed", critical_status: str = "needs_confirmation",
                       given: set[str] | frozenset[str] = frozenset()) -> list[dict]:
    """Deterministic: every sellable field of a product dict becomes a fact. `given` = fields the human typed (confirmed)."""
    out: list[dict] = []

    def add(fid, kind, label, field, value, crit_always=False):
        crit = crit_always or kind in CRITICAL_KINDS or bool(_RISKY.search(_text(value)))
        st = "confirmed" if field in given else (critical_status if crit else noncritical_status)
        src = {**source, "type": "user"} if field in given else source
        out.append(fact(fid, kind, label, field, value, src, st, crit))

    for k, kind, label, crit in SCALARS:
        v = p.get(k)
        if k in ("original_price", "offer_price", "offer_validity") or not _empty(v):  # price slots always show (empty = agent won't say one)
            add("f-" + k.replace("_", "-"), kind, label, k, v if v is not None else "", crit)
    for i, b in enumerate(p.get("benefits") or []):
        add(f"f-usp-{i}", "usp", f"USP {i + 1}", f"benefits[{i}]", {"en": b, "hi": (p.get("benefits_hi") or [""] * (i + 1))[i] if i < len(p.get("benefits_hi") or []) else "",
                                                              "hinglish": (p.get("benefits_hinglish") or [])[i] if i < len(p.get("benefits_hinglish") or []) else ""})
    for i, f in enumerate(p.get("faqs") or []):
        add(f"f-faq-{i}", "faq", "FAQ: " + ", ".join((f.get("keywords") or [])[:3]), f"faqs[{i}]",
            {"en": f.get("answer", ""), "hi": f.get("answer_hi", ""), "hinglish": f.get("answer_hinglish", "")})
    for w, say in (p.get("pronunciations") or {}).items():
        out.append(fact("f-say-" + re.sub(r"[^a-z0-9]+", "-", w.lower()).strip("-"), "pronunciation", "Say " + w, f"pronunciations.{w}", say, source, "confirmed", False))
    eco = p.get("economics") or {}
    for k, label in ECON:
        if not _empty(eco.get(k)):
            add("f-eco-" + k.replace("_", "-"), "floor", label, f"economics.{k}", eco.get(k), True)
    return out


def apply_fact(product: dict, f: dict) -> None:
    """Write a fact's value back into the product field it came from. A rejected fact is emptied / removed."""
    field, v = f.get("field", ""), f.get("value")
    gone = f.get("status") == "rejected"
    if not field:
        return
    m = re.match(r"^(benefits|faqs)\[(\d+)\]$", field)
    if m:
        name, i = m.group(1), int(m.group(2))
        if name == "benefits":
            for key, lang in (("benefits", "en"), ("benefits_hi", "hi"), ("benefits_hinglish", "hinglish")):
                lst = list(product.get(key) or [])
                while len(lst) <= i:
                    lst.append("")
                lst[i] = "" if gone else (v.get(lang, "") if isinstance(v, dict) else (str(v) if lang == "en" else lst[i]))
                product[key] = lst
        else:
            lst = list(product.get("faqs") or [])
            if i < len(lst):
                if gone:
                    lst[i] = dict(lst[i], answer="", answer_hi="", answer_hinglish="")
                elif isinstance(v, dict):
                    lst[i] = dict(lst[i], answer=v.get("en", ""), answer_hi=v.get("hi", ""), answer_hinglish=v.get("hinglish", ""))
                else:
                    lst[i] = dict(lst[i], answer=str(v))
            product["faqs"] = lst
        return
    if field.startswith("pronunciations."):
        pr = dict(product.get("pronunciations") or {})
        w = field.split(".", 1)[1]
        if gone:
            pr.pop(w, None)
        else:
            pr[w] = str(v)
        product["pronunciations"] = pr
        return
    if field.startswith("economics."):
        eco = dict(product.get("economics") or {})
        try:
            eco[field.split(".", 1)[1]] = 0.0 if gone or v in ("", None) else float(v)
        except (TypeError, ValueError):
            pass
        product["economics"] = eco
        return
    if gone:
        product[field] = {} if isinstance(product.get(field), dict) else ""
    else:
        product[field] = v


def compile_product(draft: dict) -> dict:
    p = copy.deepcopy(draft.get("product") or {})
    for f in draft.get("facts") or []:
        apply_fact(p, f)
    p["benefits"] = [b for b in p.get("benefits", []) if str(b).strip()]
    for k in ("benefits_hi", "benefits_hinglish"):
        p[k] = [b for b in p.get(k, []) if str(b).strip()]
    p["faqs"] = [f for f in p.get("faqs", []) if str(f.get("answer", "")).strip()]
    strat = draft.get("strategy") or {}
    if strat.get("agent_id"):
        p["agent_id"] = strat["agent_id"]
    if strat.get("tone_level") in ("professional", "mid", "casual"):
        p["tone_level"] = strat["tone_level"]
    return p


# ---------------------------------------------------------------- links: which facts a node may cite

_STOP = set("the a an and or to of for in on with is are you your it this that we our be can will at as by from per".split())


def _words(t: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9ऀ-ॿ]+", t.lower()) if len(w) > 2 and w not in _STOP}


def auto_cites(nodes: list[dict], facts: list[dict]) -> int:
    """Fill node["cites"] (only when the node has none) with facts whose words overlap its talking points. Returns nodes linked."""
    fw = [(f["id"], _words(_text(f.get("value")) + " " + f.get("label", ""))) for f in facts if f.get("status") != "rejected" and f["kind"] not in ("identity", "pronunciation")]
    linked = 0
    for n in walk(nodes):
        if n.get("cites"):
            continue
        nw = _words(" ".join(n.get("points") or []) + " " + n.get("title", ""))
        cites = [fid for fid, w in fw if len(nw & w) >= 2]
        n["cites"] = cites[:8]
        linked += bool(cites)
    return linked


def walk(nodes: list[dict]):
    for n in nodes or []:
        yield n
        yield from walk(n.get("children") or [])


def find_node(nodes: list[dict], nid: str) -> dict | None:
    return next((n for n in walk(nodes) if n.get("id") == nid), None)


# ---------------------------------------------------------------- numbers (grounding)

def numbers(text: str) -> set[str]:
    t = (text or "").translate(str.maketrans("०१२३४५६७८९", "0123456789"))
    out = set()
    for m in _NUM.finditer(t):
        n = m.group(1).replace(",", "").rstrip(".")
        if n:
            try:
                f = float(n)
            except ValueError:
                continue
            out.add(str(int(f)) if f == int(f) else str(f))
    return out


def allowed_numbers(facts: list[dict], confirmed_only: bool = True) -> set[str]:
    out: set[str] = set()
    for f in facts:
        if f.get("status") == "rejected" or (confirmed_only and f.get("status") != "confirmed"):
            continue
        out |= numbers(_text(f.get("value")))
    return out


def ungrounded(draft: dict) -> list[dict]:
    """Numbers in talking points / asks that no confirmed fact states."""
    ok = allowed_numbers(draft.get("facts") or [])
    bad = []
    for n in walk((draft.get("tree") or {}).get("nodes") or []):
        txt = " ".join((n.get("points") or []) + (n.get("points_hi") or []) + [n.get("ask", ""), n.get("ask_hi", "")])
        miss = sorted(x for x in numbers(txt) if x not in ok)
        if miss:
            bad.append({"node": n.get("id"), "numbers": miss})
    return bad


# ---------------------------------------------------------------- readiness

def readiness(draft: dict) -> dict:
    facts = draft.get("facts") or []
    nodes = list(walk((draft.get("tree") or {}).get("nodes") or []))
    open_crit = [f["id"] for f in facts if f.get("critical") and f.get("status") not in ("confirmed", "rejected")]
    offer = next((f for f in facts if f.get("field") in ("offer_price", "original_price") and f.get("status") == "confirmed" and not _empty(f.get("value"))), None)
    no_ask = [n["id"] for n in nodes if not str(n.get("ask", "")).strip()]
    no_hi = [n["id"] for n in nodes if not n.get("points_hi") or not str(n.get("ask_hi", "")).strip()]
    no_cite = [n["id"] for n in nodes if not n.get("cites")]
    ung = ungrounded(draft)
    tests = draft.get("last_test") or {}
    items = [
        {"id": "facts", "label": "Every critical fact confirmed by a person", "ok": not open_crit, "blocking": True,
         "detail": f"{len(open_crit)} waiting" if open_crit else "all confirmed", "items": open_crit},
        {"id": "price", "label": "A confirmed price the agent can quote", "ok": bool(offer), "blocking": False,
         "detail": "ok" if offer else "no confirmed price: the agent will say it does not have the price"},
        {"id": "tree", "label": "At least 5 situations in the decision tree", "ok": len(nodes) >= 5, "blocking": False, "detail": f"{len(nodes)} situations"},
        {"id": "asks", "label": "Every situation ends with a question", "ok": not no_ask, "blocking": False, "detail": f"{len(no_ask)} without" if no_ask else "ok", "items": no_ask},
        {"id": "hindi", "label": "Every situation has Hindi lines", "ok": not no_hi, "blocking": False, "detail": f"{len(no_hi)} without" if no_hi else "ok", "items": no_hi},
        {"id": "cites", "label": "Every situation is linked to facts it may cite", "ok": not no_cite, "blocking": False, "detail": f"{len(no_cite)} unlinked" if no_cite else "ok", "items": no_cite},
        {"id": "grounded", "label": "Every number in the tree is a confirmed fact", "ok": not ung, "blocking": False,
         "detail": "; ".join(f"{u['node']}: {', '.join(u['numbers'])}" for u in ung[:4]) or "ok", "items": [u["node"] for u in ung]},
        {"id": "tests", "label": "Simulated customers tested", "ok": bool(tests) and tests.get("pass_rate", 0) >= 0.75, "blocking": False,
         "detail": (f"{round(tests.get('pass_rate', 0) * 100)}% passed (run {tests.get('run_id')})" if tests else "not run yet")},
    ]
    return {"ready": not any(i["blocking"] and not i["ok"] for i in items), "score": sum(i["ok"] for i in items), "total": len(items), "items": items}


# ---------------------------------------------------------------- storage

def draft_path(data: Path, tenant: str, pid: str) -> Path:
    return S.pdir(data, tenant, pid) / "draft.json"


def load_draft(data: Path, tenant: str, pid: str) -> dict | None:
    return S._read(draft_path(data, tenant, pid))


def save_draft(data: Path, d: dict) -> dict:
    d["updated_at"] = now()
    S._write(draft_path(data, d["tenant"], d["product_id"]), d)
    return d


def _live_files(data: Path, tenant: str, pid: str, kb_root: Path | None) -> tuple[dict, dict, dict]:
    from acie import playbook as pbm
    from acie.knowledge.store import kb_path
    products = S._read(Path(data) / "products.json", {}) or {}
    p = products.get(pid)
    if p is None:
        raise KeyError(f"unknown product {pid}")
    kb = {}
    for t in dict.fromkeys([tenant, "default"]):
        kp = kb_path(t, pid, kb_root)
        if kp.exists():
            kb = S._read(kp, {}) or {}
            break
    pb = pbm.load(Path(data), tenant, pid)
    tree = {"nodes": [n.model_dump() for n in pb.nodes] if pb else [], "ladder": [], "fit_rules": [], "clusters": []}
    return p, kb, tree


def import_live(data: Path, tenant: str, pid: str, kb_root: Path | None = None, force: bool = False) -> dict:
    """Live product + KB + playbook -> v1 (marked as the published base, since it IS what is live) and a draft.
    Critical facts start 'needs_confirmation' (owner decision): live calls are unaffected until Publish."""
    if not force and (d := load_draft(data, tenant, pid)):
        return d
    p, kb, tree = _live_files(data, tenant, pid, kb_root)
    if kb:
        kb = dict(kb, tenant=tenant)
    bundle = {"product": p, "kb": kb, "tree": tree}
    pub = S.state(data, tenant, pid).get("published")
    if not pub:
        doc = {"kind": "studio_import", "status": "published", "model": "", "spec": {"name": p.get("product_name", pid)}, "effective": {"real_brand": False},
               "bundle": bundle, "report": {"publishable": True, "passed": True, "checks": []}, "note": "imported from the live files"}
        S.save_version(data, tenant, pid, doc)
        st = S.state(data, tenant, pid)
        st.update({"published": doc["version"], "published_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
        S._write(S.pdir(data, tenant, pid) / "state.json", st)
        pub = doc["version"]
    facts = facts_from_product(p, {"type": "import", "ref": "live product sheet"})
    tree = copy.deepcopy(tree)
    auto_cites(tree["nodes"], facts)
    d = {"tenant": tenant, "product_id": pid, "name": p.get("product_name", pid), "base_version": pub, "created_at": now(),
         "facts": facts, "product": p, "kb": kb, "tree": tree, "strategy": {"agent_id": p.get("agent_id", ""), "tone_level": p.get("tone_level", "mid")},
         "personas": [], "last_test": None, "source_text": ""}
    return save_draft(data, d)


def draft_from_version(data: Path, tenant: str, pid: str, v: dict, given: set[str], source_text: str = "") -> dict:
    """A generated productgen version -> draft. AI-written non-critical facts are 'ai_draft'; critical ones need confirmation."""
    b = v["bundle"]
    p = dict(b["product"], id=pid, tenant=tenant)
    neg = (b.get("negotiation") or {}).get("economics")
    if neg:
        p["economics"] = {**(p.get("economics") or {}), **neg}
    facts = facts_from_product(p, {"type": "ai", "ref": v.get("model", "") or "Gemini"}, noncritical_status="ai_draft", given=given)
    if source_text:
        low = source_text.lower()
        for f in facts:  # a fact whose text appears in what the user pasted keeps that quote as its source
            t = _text(f.get("value")).strip()
            if f["source"]["type"] == "ai" and t and len(t) > 3 and t.lower()[:40] in low:
                f["source"] = {"type": "user_text", "ref": "pasted text", "quote": t[:120]}
    tree = copy.deepcopy(b.get("tree") or {"nodes": []})
    auto_cites(tree.get("nodes", []), facts)
    d = {"tenant": tenant, "product_id": pid, "name": p.get("product_name", pid), "base_version": None, "generated_version": v.get("version"),
         "created_at": now(), "facts": facts, "product": p, "kb": b.get("kb") or {}, "tree": tree,
         "strategy": {"agent_id": "", "tone_level": p.get("tone_level", "mid") or "mid"}, "personas": [], "last_test": None, "source_text": source_text[:6000]}
    return save_draft(data, d)


def bundle_of(draft: dict) -> dict:
    tree = copy.deepcopy(draft.get("tree") or {})
    kb = copy.deepcopy(draft.get("kb") or {})
    if kb:
        kb["tenant"], kb["product_id"] = draft["tenant"], draft["product_id"]
    return {"product": dict(compile_product(draft), id=draft["product_id"], tenant=draft["tenant"]), "kb": kb, "tree": tree}


def save_version_from_draft(data: Path, draft: dict, note: str = "") -> dict:
    r = readiness(draft)
    doc = {"kind": "studio", "status": "draft", "model": "", "spec": {"name": draft.get("name")}, "effective": {"real_brand": False},
           "bundle": bundle_of(draft), "facts": draft.get("facts"), "report": {"publishable": r["ready"], "passed": r["score"] == r["total"],
                                                                                  "checks": r["items"]},
           "base_version": draft.get("base_version"), "note": note or "saved from the studio"}
    return S.save_version(data, draft["tenant"], draft["product_id"], doc)


class NotReady(Exception):
    def __init__(self, items: list[dict]):
        super().__init__("; ".join(f"{i['label']} ({i['detail']})" for i in items))
        self.items = items


def publish(data: Path, tenant: str, pid: str, kb_root: Path | None = None) -> dict:
    d = load_draft(data, tenant, pid)
    if not d:
        raise KeyError("no draft")
    r = readiness(d)
    block = [i for i in r["items"] if i["blocking"] and not i["ok"]]
    if block:
        raise NotReady(block)
    v = save_version_from_draft(data, d, "published from the studio")
    out = write_live(data, tenant, pid, v["version"], kb_root=kb_root)
    d["base_version"] = v["version"]
    save_draft(data, d)
    return {**out, "warnings": [i for i in r["items"] if not i["ok"] and not i["blocking"]]}


def write_live(data: Path, tenant: str, pid: str, n: int, kb_root: Path | None = None) -> dict:
    """store.publish semantics (three-way merge against the last published version, backups, state.json), with two differences:
    an unchanged KB is not rewritten, and add-on offerings are checked against their own floors only (the product's floor is the
    main plan's: store.publish applied it to every add-on and refused FitZone's live KB)."""
    import shutil
    from acie import playbook as pbm
    from acie.decision.domains.product_sales import Product
    from acie.knowledge.schema import KnowledgeBase
    from acie.knowledge.store import kb_dir, kb_path, save_kb, validate

    v = S.load_version(data, tenant, pid, n)
    bundle = v["bundle"]
    st = S.state(data, tenant, pid)
    base = ((S.load_version(data, tenant, pid, st["published"]) or {}).get("bundle") or {}) if st.get("published") else {}
    pfile = Path(data) / "products.json"
    products = S._read(pfile, {}) or {}
    newp = dict(bundle["product"], id=pid, tenant=tenant)
    cur_p = products.get(pid)
    merged_p = S._norm_product(newp) if cur_p is None else S._norm_product(
        S.merge_fields(S._norm_product(base["product"]) if base.get("product") else {}, S._norm_product(cur_p), S._norm_product(newp)))
    Product.model_validate(merged_p)
    kroot = kb_root or kb_dir()
    kpath = kb_path(tenant, pid, kroot)
    kb_changed = bundle.get("kb") != base.get("kb") or not any(kb_path(t, pid, kroot).exists() for t in (tenant, "default"))
    kb = None
    if kb_changed and bundle.get("kb"):
        cur_kb = S._read(kpath, {}) or {}
        merged_kb = S._kb_merge(base.get("kb") or {}, cur_kb, bundle["kb"]) if cur_kb else dict(bundle["kb"])
        try:
            kb = KnowledgeBase.model_validate(merged_kb)
        except Exception as ex:  # noqa: BLE001
            raise S.PublishError("knowledge base is invalid: " + str(ex)[:300])
        issues = validate(kb, None)
        if issues:
            raise S.PublishError("knowledge base rules: " + "; ".join(issues[:3]))
    pbf = pbm.playbook_dir(Path(data)) / tenant / f"{pid}.json"
    pb_cur = S._read(pbf, {}) or {}
    tree = bundle.get("tree") or {}
    clean = lambda ns: [{k: x for k, x in nd.items() if k not in ("cites", "origin")} for nd in ns]  # noqa: E731
    nodes = S.merge_list(clean((base.get("tree") or {}).get("nodes", [])), pb_cur.get("nodes", []), clean(tree.get("nodes", [])), lambda x: x.get("id", "")) \
        if pb_cur else clean(tree.get("nodes", []))
    pb = pbm.Playbook(tenant=tenant, product_id=pid, generated_by=f"studio v{n}", nodes=nodes)
    bdir = S.pdir(data, tenant, pid) / "backups" / time.strftime("%Y%m%d-%H%M%S")
    for src in (pfile, kpath, pbf):
        if src.exists():
            bdir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, bdir / (src.parent.name + "__" + src.name))
    products = S._read(pfile, {}) or {}
    products[pid] = merged_p
    S._write_plain(pfile, products)
    if kb is not None:
        save_kb(kb, kpath)
    pbm.save(Path(data), pb)
    st.update({"published": n, "published_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    S._write(S.pdir(data, tenant, pid) / "state.json", st)
    v["status"] = "published"
    S._write(S.pdir(data, tenant, pid) / f"v{n}.json", v)
    return {"ok": True, "version": n, "kb_written": kb is not None, "nodes": len(nodes), "backup": str(bdir) if bdir.exists() else None}
