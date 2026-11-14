"""Self-extension: (a) expand one node on demand, (b) mine finished calls for questions the agent could not answer and
queue proposed knowledge as a PENDING version. Runtime stays deterministic: nothing here touches a live call; new knowledge
only appears after Publish (or auto-approve for synthetic demo products), between calls."""

from __future__ import annotations

import copy
import json
import os
import re
import time
import zlib
from pathlib import Path

from . import criteria as C
from . import kbops as K
from . import store as S
from .generate import spec_payload
from .llm import LLM
from .spec import Effective, Spec

FALLBACK_INTENTS = {"unknown", "fallback", "dont_know", "no_info", "honest_unknown", "info_source", "unanswerable", "kb_fallback", "clarify_unknown"}
FALLBACK_TEXT = re.compile(
    r"(i don'?t have (that|the|this|any)|i do not have (that|the|this)|don'?t have (the )?(exact|that|this|more)|not sure about that|"
    r"check (the|our) (official )?website|official website (will|should)|mere paas (wo|ye|yeh|uski|iski)?.{0,25}nahi|mere paas nahi|"
    r"मेरे पास.{0,30}नहीं|वेबसाइट पर (दिख|मिल)|website par (dikh|mil))", re.I)


def _spec_eff(v: dict) -> tuple[Spec, Effective]:
    return Spec.model_validate(v["spec"]), Effective.model_validate(v["effective"])


def _snapshot(bundle: dict) -> dict:
    kb = bundle["kb"]
    return {"entries": {e["id"]: e for e in kb["entries"]}, "topics": {t["path"]: t for t in kb["topics"]}, "offerings": {o["id"]: o for o in kb["offerings"]},
            "interests": {i["id"]: i for i in kb["interests"]}, "nodes": S._flat_nodes(bundle["tree"]["nodes"])}


def _assert_untouched(before: dict, after: dict, allowed_nodes: set[str] = frozenset()) -> None:
    """Expansion may only ADD. Every pre-existing item must be byte-identical (a parent node may gain children)."""
    for sec in ("entries", "topics", "offerings", "interests"):
        for k, v in before[sec].items():
            if after[sec].get(k) != v:
                raise AssertionError(f"expansion changed existing {sec[:-1]} {k}")
    for k, v in before["nodes"].items():
        if after["nodes"].get(k) != v:
            raise AssertionError(f"expansion changed existing node {k}")


def _drop_banned(bundle: dict, spec: Spec, eff: Effective, new_entries: list[str], new_nodes: list[str]) -> int:
    rep = C.check(bundle, spec, eff)
    bad = set()
    for c in rep["checks"]:
        if c["id"] in ("banned", "languages") and not c["passed"]:
            bad |= set(c["items"])
    ne = {i for i in new_entries if i in bad}
    nn = {i for i in new_nodes if "node:" + i in bad}
    bundle["kb"]["entries"] = [e for e in bundle["kb"]["entries"] if e["id"] not in ne]

    def prune(nodes: list[dict]) -> list[dict]:
        return [dict(n, children=prune(n.get("children", []))) if n.get("children") else n for n in nodes if n["id"] not in nn]
    bundle["tree"]["nodes"] = prune(bundle["tree"]["nodes"])
    return len(ne) + len(nn)


def _base(data: Path, tenant: str, pid: str, version: int | None) -> dict:
    v = S.load_version(data, tenant, pid, version) if version else S.latest_full(data, tenant, pid)
    if not v or not v.get("bundle"):
        raise KeyError("no generated version to expand: generate the product first")
    return v


# ---------------------------------------------------------------- (a) expand a node on demand

async def expand_node(data: Path, tenant: str, pid: str, llm: LLM, node: dict, instruction: str = "", version: int | None = None) -> dict:
    """node = {"kind": "topic"|"entry"|"playbook", "id": <topic path | entry id | node id>}. New children only; saved as a new version."""
    v = _base(data, tenant, pid, version)
    spec, eff = _spec_eff(v)
    bundle = copy.deepcopy(v["bundle"])
    kb, tree = bundle["kb"], bundle["tree"]
    before = _snapshot(bundle)
    kind, nid = node.get("kind"), str(node.get("id", ""))
    notes: list[str] = []
    base = spec_payload(spec, eff)
    new_entries: list[str] = []
    new_nodes: list[str] = []
    note = ""
    if kind in ("topic", "entry"):
        path = K.path_clean(nid)
        if kind == "entry":
            e = next((x for x in kb["entries"] if x["id"] == nid), None)
            if not e:
                raise KeyError("unknown entry " + nid)
            path = e["topic_path"]
        if path not in {t["path"] for t in kb["topics"]}:
            raise KeyError("unknown topic " + path)
        under = [e for e in kb["entries"] if e["topic_path"] == path or e["topic_path"].startswith(path + "/")]
        r = await llm.json("expand_topic", {**base, "topic": path, "focus_entry": nid if kind == "entry" else None, "instruction": instruction,
                                            "existing_subtopics": [t["path"] for t in kb["topics"] if t["path"].startswith(path + "/")],
                                            "existing_questions": [(e["question_variants"].get("en") or [""])[0] for e in under],
                                            "offering_ids": [o["id"] for o in kb["offerings"]], "interest_ids": [i["id"] for i in kb["interests"]]})
        K.add_interests(kb, r.get("interests", []))
        subs = [t for t in r.get("topics", []) or [] if isinstance(t, dict) and K.path_clean(t.get("path")).startswith(path + "/")]
        K.add_topics(kb, subs, notes)
        ents = []
        for e in r.get("entries", []) or []:
            tp = K.path_clean((e or {}).get("topic_path")) if isinstance(e, dict) else ""
            if tp == path or tp.startswith(path + "/"):
                ents.append(e)
            else:
                notes.append(f"entry skipped: outside {path}")
        new_entries = K.add_entries(kb, ents, eff, notes)
        note = f"expanded topic {path}: +{len(new_entries)} entries"
    elif kind == "playbook":
        target = K.find_node(tree["nodes"], nid)
        if not target:
            raise KeyError("unknown node " + nid)
        r = await llm.json("expand_node", {**base, "node": {k: target[k] for k in target if k != "children"}, "existing_children": [c["id"] for c in target.get("children", [])],
                                           "instruction": instruction, "kb": {"topics": [t["path"] for t in kb["topics"]]}})
        target.setdefault("children", [])
        new_nodes = K.add_nodes(target["children"], r.get("children", []), notes, limit=len(target["children"]) + 6)
        # K.add_nodes only checks ids against this list: re-check globally so a child never reuses another node's id
        note = f"expanded situation {nid}: +{len(new_nodes)} branches"
    else:
        raise ValueError("node.kind must be topic, entry or playbook")
    dropped = _drop_banned(bundle, spec, eff, new_entries, new_nodes)
    _assert_untouched(before, _snapshot(bundle))
    report = C.check(bundle, spec, eff)
    bundle["tree"]["clusters"] = K.clusters(kb)
    doc = {"kind": "expand", "status": "draft" if report["publishable"] else "needs_review", "model": getattr(llm, "model", ""), "spec": v["spec"], "effective": v["effective"],
           "bundle": bundle, "report": report, "base_version": v["version"], "note": note + (f" ({dropped} dropped by the checks)" if dropped else ""),
           "expanded": {"kind": kind, "id": nid, "entries": new_entries, "nodes": new_nodes}, "notes": notes[:100]}
    S.save_version(data, tenant, pid, doc)
    return doc


# ---------------------------------------------------------------- (b) gap mining

def _terms(text: str) -> frozenset[str]:
    from acie.knowledge import text as T
    return frozenset(T.term_keys(text))


def _gap_id(rep: str) -> str:
    from acie.knowledge import text as T
    return "gap-%08x" % zlib.crc32(" ".join(sorted(T.term_keys(rep))).encode("utf-8"))


def mine_gaps(calls_dir: Path, product_id: str, kb=None, product: dict | None = None, skip: set[str] | frozenset[str] = frozenset(),
              min_terms: int = 2, jaccard: float = 0.5, max_clusters: int = 40) -> list[dict]:
    """Questions callers asked that got a fallback / 'I don't have that' answer or scored low against the KB, clustered
    deterministically (files sorted, greedy Jaccard on canonical terms, fixed tie-breaks)."""
    from acie.knowledge import text as T
    from acie.knowledge.retrieve import retrieve
    info = [s for s in ((product or {}).get("info_source") or {}).values() if s]
    hits: list[dict] = []
    for f in sorted(Path(calls_dir).glob("*.json")) if Path(calls_dir).exists() else []:
        try:
            call = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(call, dict) or call.get("campaign") != product_id:
            continue
        turns = call.get("turns") or []
        cid = call.get("call_id") or f.stem
        for i, t in enumerate(turns):
            if t.get("role") != "customer" or len(T.content_terms(t.get("text", ""))) < min_terms:
                continue
            nxt = next((x for x in turns[i + 1:i + 3] if x.get("role") == "agent"), None)
            reasons = []
            if nxt is not None and (nxt.get("intent") in FALLBACK_INTENTS or FALLBACK_TEXT.search(nxt.get("text", "")) or any(s and s in nxt.get("text", "") for s in info)):
                reasons.append("fallback_answer")
            score = None
            if kb is not None:
                res = retrieve(kb, t["text"], None, 1)
                score = round(res.top.score if res.top else 0.0, 3)
                if not res.confident:
                    reasons.append("low_kb_score")
            if reasons:
                hits.append({"call_id": cid, "text": t["text"].strip(), "reasons": reasons, "score": score, "terms": _terms(t["text"])})
    clusters: list[dict] = []
    for h in hits:
        for c in clusters:
            u = len(c["terms"] | h["terms"])
            if u and len(c["terms"] & h["terms"]) / u >= jaccard:
                c["items"].append(h)  # the cluster's term set stays the first member's: order-stable
                break
        else:
            clusters.append({"terms": h["terms"], "items": [h]})
    out = []
    for c in clusters:
        qs = sorted({i["text"] for i in c["items"]}, key=lambda q: (len(q), q))
        rep = qs[0]
        gid = _gap_id(rep)
        if gid in skip:
            continue
        if kb is not None and retrieve(kb, rep, None, 1).confident:  # answered well since those calls: not a gap any more
            continue
        scores = [i["score"] for i in c["items"] if i["score"] is not None]
        out.append({"id": gid, "question": rep, "questions": qs[:8], "count": len(c["items"]), "call_ids": sorted({i["call_id"] for i in c["items"]}),
                    "reasons": sorted({r for i in c["items"] for r in i["reasons"]}), "min_score": min(scores) if scores else None})
    out.sort(key=lambda g: (-g["count"], g["id"]))
    return out[:max_clusters]


def _pending_seen(data: Path, tenant: str, pid: str) -> dict:
    return S.state(data, tenant, pid).get("gap_seen", {}) or {}


async def propose_gaps(data: Path, tenant: str, pid: str, llm: LLM, calls_dir: Path, kb_root: Path | None = None, max_clusters: int = 12) -> dict | None:
    """Mine calls, ask the model for proposals per cluster, queue them as a PENDING version. None when there is nothing new."""
    v = _base(data, tenant, pid, None)
    spec, eff = _spec_eff(v)
    kbd = v["bundle"]["kb"]
    from acie.knowledge.schema import KnowledgeBase
    kbm = KnowledgeBase.model_validate(kbd)
    seen = set(_pending_seen(data, tenant, pid))
    gaps = mine_gaps(calls_dir, pid, kbm, v["bundle"]["product"], skip=seen, max_clusters=max_clusters)
    if not gaps:
        return None
    base = spec_payload(spec, eff)
    scratch = copy.deepcopy(v["bundle"])
    proposals: list[dict] = []
    notes: list[str] = []
    for i in range(0, len(gaps), 6):
        chunk = gaps[i:i + 6]
        r = await llm.json("gap_entries", {**base, "clusters": [{"id": g["id"], "questions": g["questions"], "count": g["count"]} for g in chunk],
                                           "topics": [{"path": t["path"], "title": t["title"].get("en")} for t in kbd["topics"]],
                                           "offering_ids": [o["id"] for o in kbd["offerings"]], "interest_ids": [x["id"] for x in kbd["interests"]],
                                           "playbook": [n["id"] for n in v["bundle"]["tree"]["nodes"]]})
        sk = scratch["kb"]
        new_topics = K.add_topics(sk, r.get("topics", []), notes)
        by: dict[str, dict] = {g["id"]: {"id": g["id"], "status": "pending", "questions": g["questions"], "count": g["count"], "call_ids": g["call_ids"],
                                          "reasons": g["reasons"], "topics": [], "entries": [], "nodes": []} for g in chunk}
        for t in sk["topics"]:
            if t["path"] in new_topics:
                owner = next((e.get("cluster") for e in r.get("entries", []) if isinstance(e, dict) and K.path_clean(e.get("topic_path")) == t["path"]), None)
                (by.get(owner) or by[chunk[0]["id"]])["topics"].append(t)
        for e in r.get("entries", []) or []:
            if not isinstance(e, dict):
                continue
            cid = e.get("cluster") if e.get("cluster") in by else next((g["id"] for g in chunk if g["id"] in str(e.get("source", ""))), None)
            if cid is None:
                notes.append("entry skipped: no cluster named")
                continue
            ids = K.add_entries(sk, [e], eff, notes, source="gap:" + ",".join(by[cid]["call_ids"][:5]))
            for eid in ids:
                by[cid]["entries"].append(next(x for x in sk["entries"] if x["id"] == eid))
        for n in r.get("nodes", []) or []:
            cid = n.get("cluster") if isinstance(n, dict) and n.get("cluster") in by else chunk[0]["id"]
            if isinstance(n, dict):
                got = K.add_nodes(scratch["tree"]["nodes"], [n], notes)
                if got:
                    by[cid]["nodes"].append(K.find_node(scratch["tree"]["nodes"], got[0]))
        proposals += [p for p in by.values() if p["entries"] or p["nodes"]]
    # compliance: drop proposed items that fail the banned-content / language checks
    for p in proposals:
        keep = []
        for e in p["entries"]:
            tmp = {"kb": {"entries": [e], "topics": [], "offerings": [], "interests": []}, "tree": {"nodes": []}, "product": {}}
            if all(c["passed"] for c in C.check(tmp, spec, eff)["checks"] if c["id"] in ("banned", "languages")):
                keep.append(e)
        p["entries"] = keep
    proposals = [p for p in proposals if p["entries"] or p["nodes"]]
    if not proposals:
        return None
    doc = {"kind": "gap_proposals", "status": "pending", "model": getattr(llm, "model", ""), "spec": v["spec"], "effective": v["effective"], "bundle": None,
           "report": {"counts": {"proposals": len(proposals), "entries": sum(len(p["entries"]) for p in proposals)}}, "base_version": v["version"],
           "proposals": proposals, "note": f"{len(proposals)} gap proposals from {sum(g['count'] for g in gaps)} unanswered questions", "notes": notes[:100]}
    S.save_version(data, tenant, pid, doc)
    st = S.state(data, tenant, pid)
    st.setdefault("gap_seen", {}).update({p["id"]: "pending" for p in proposals})
    S._write(S.pdir(data, tenant, pid) / "state.json", st)
    if os.getenv("ACIE_PRODUCTGEN_AUTO_APPROVE") == "1" and not eff.real_brand and eff.provenance == "synthetic":
        res = approve_gaps(data, tenant, pid, doc["version"], [p["id"] for p in proposals], [], publish=True, kb_root=kb_root)
        doc["auto_approved"] = res
    return doc


def approve_gaps(data: Path, tenant: str, pid: str, version: int, approve: list[str], reject: list[str], publish: bool = False, kb_root: Path | None = None) -> dict:
    pv = S.load_version(data, tenant, pid, version)
    if not pv or pv.get("kind") != "gap_proposals":
        raise KeyError("not a pending gap version")
    latest = _base(data, tenant, pid, None)
    spec, eff = _spec_eff(latest)
    bundle = copy.deepcopy(latest["bundle"])
    notes: list[str] = []
    approved = []
    for p in pv["proposals"]:
        if p["id"] in set(approve) and p["status"] == "pending":
            K.add_topics(bundle["kb"], p.get("topics", []), notes)
            K.add_entries(bundle["kb"], p.get("entries", []), eff, notes, source="")
            K.add_nodes(bundle["tree"]["nodes"], p.get("nodes", []), notes, limit=eff.criteria.max_situations + 40)
            p["status"] = "approved"
            approved.append(p["id"])
        elif p["id"] in set(reject) and p["status"] == "pending":
            p["status"] = "rejected"
    S._write(S.pdir(data, tenant, pid) / f"v{version}.json", pv)
    st = S.state(data, tenant, pid)
    for p in pv["proposals"]:
        if p["status"] != "pending":
            st.setdefault("gap_seen", {})[p["id"]] = p["status"]
    S._write(S.pdir(data, tenant, pid) / "state.json", st)
    out: dict = {"approved": approved, "rejected": [p["id"] for p in pv["proposals"] if p["status"] == "rejected"], "version": None}
    if approved:
        bundle["tree"]["clusters"] = K.clusters(bundle["kb"])
        report = C.check(bundle, spec, eff)
        doc = {"kind": "gaps_approved", "status": "draft" if report["publishable"] else "needs_review", "model": pv.get("model", ""), "spec": latest["spec"],
               "effective": latest["effective"], "bundle": bundle, "report": report, "base_version": latest["version"],
               "note": f"approved gap proposals {', '.join(approved)} (from version {version})", "notes": notes[:100]}
        S.save_version(data, tenant, pid, doc)
        out["version"] = doc["version"]
        out["report"] = {"passed": report["passed"], "publishable": report["publishable"]}
        if publish:
            out["published"] = S.publish(data, tenant, pid, doc["version"], kb_root)
    return out
