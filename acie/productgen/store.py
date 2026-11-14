"""Versioned files, diffs and the Publish merge.

data/voice/productgen/<tenant>/<product_id>/v<N>.json       one immutable version (spec, criteria, bundle, report)
data/voice/productgen/<tenant>/<product_id>/v<N>.diff.json  what changed vs the previous version
data/voice/productgen/<tenant>/<product_id>/state.json      which version is published (the merge base for the next publish)
Publish merges into products.json, the KB file and the playbook file; fields the operator edited in the console since the last
publish are kept (three-way merge: last published version = base, live files = current, this version = new)."""

from __future__ import annotations

import copy
import json
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Callable

_LOCK = threading.Lock()


class PublishError(Exception):
    pass


def root_dir(data: Path) -> Path:
    return Path(data) / "productgen"


def pdir(data: Path, tenant: str, pid: str) -> Path:
    return root_dir(data) / tenant / pid


def _write(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, p)


def _read(p: Path, default: Any = None) -> Any:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def version_numbers(data: Path, tenant: str, pid: str) -> list[int]:
    d = pdir(data, tenant, pid)
    if not d.exists():
        return []
    out = []
    for f in d.glob("v*.json"):
        s = f.stem[1:]
        if s.isdigit():
            out.append(int(s))
    return sorted(out)


def state(data: Path, tenant: str, pid: str) -> dict:
    return _read(pdir(data, tenant, pid) / "state.json", {}) or {}


def load_version(data: Path, tenant: str, pid: str, n: int) -> dict | None:
    return _read(pdir(data, tenant, pid) / f"v{n}.json")


def latest_full(data: Path, tenant: str, pid: str) -> dict | None:
    """The newest version that carries a full bundle (pending gap versions carry proposals only)."""
    for n in reversed(version_numbers(data, tenant, pid)):
        v = load_version(data, tenant, pid, n)
        if v and v.get("bundle"):
            return v
    return None


def summary(v: dict, published: int | None = None) -> dict:
    b = v.get("bundle") or {}
    rep = v.get("report") or {}
    return {"version": v["version"], "kind": v.get("kind", "generate"), "status": v.get("status", "draft"), "created_at": v.get("created_at"),
            "model": v.get("model", ""), "published": published == v["version"], "counts": rep.get("counts"),
            "passed": rep.get("passed"), "publishable": rep.get("publishable"), "base_version": v.get("base_version"),
            "proposals": len(v.get("proposals", []) or []), "note": v.get("note", "")}


def list_versions(data: Path, tenant: str, pid: str) -> list[dict]:
    pub = state(data, tenant, pid).get("published")
    return [summary(v, pub) for n in version_numbers(data, tenant, pid) if (v := load_version(data, tenant, pid, n))]


def list_products(data: Path) -> list[dict]:
    out = []
    r = root_dir(data)
    if not r.exists():
        return out
    for t in sorted(p for p in r.iterdir() if p.is_dir()):
        for p in sorted(x for x in t.iterdir() if x.is_dir()):
            vs = version_numbers(data, t.name, p.name)
            if vs:
                v = load_version(data, t.name, p.name, vs[-1]) or {}
                out.append({"tenant": t.name, "product_id": p.name, "name": (v.get("spec") or {}).get("name", p.name), "versions": len(vs),
                            "latest": vs[-1], "published": state(data, t.name, p.name).get("published")})
    return out


# ---------------------------------------------------------------- diff

def _flat_nodes(nodes: list[dict], out: dict | None = None) -> dict:
    out = {} if out is None else out
    for n in nodes or []:
        out[n.get("id", "?")] = {k: v for k, v in n.items() if k != "children"}
        _flat_nodes(n.get("children", []), out)
    return out


def _by(items: list[dict], key: str) -> dict:
    return {x.get(key, "?"): x for x in items or []}


def _section(a: dict, b: dict) -> dict:
    added = sorted(k for k in b if k not in a)
    removed = sorted(k for k in a if k not in b)
    changed = sorted(k for k in b if k in a and a[k] != b[k])
    return {"added": added, "removed": removed, "changed": changed}


def diff_bundles(a: dict | None, b: dict | None) -> dict:
    a, b = a or {}, b or {}
    ka, kb = a.get("kb") or {}, b.get("kb") or {}
    ta, tb = a.get("tree") or {}, b.get("tree") or {}
    out = {
        "entries": _section(_by(ka.get("entries"), "id"), _by(kb.get("entries"), "id")),
        "topics": _section(_by(ka.get("topics"), "path"), _by(kb.get("topics"), "path")),
        "offerings": _section(_by(ka.get("offerings"), "id"), _by(kb.get("offerings"), "id")),
        "interests": _section(_by(ka.get("interests"), "id"), _by(kb.get("interests"), "id")),
        "nodes": _section(_flat_nodes(ta.get("nodes")), _flat_nodes(tb.get("nodes"))),
        "ladder": _section(_by(ta.get("ladder"), "id"), _by(tb.get("ladder"), "id")),
    }
    pa, pb = a.get("product") or {}, b.get("product") or {}
    out["product_fields"] = {k: {"old": pa.get(k), "new": pb.get(k)} for k in sorted(set(pa) | set(pb)) if pa.get(k) != pb.get(k)}
    out["total_changes"] = sum(len(v) for s in out.values() if isinstance(s, dict) and "added" in s for v in s.values()) + len(out["product_fields"])
    return out


def save_version(data: Path, tenant: str, pid: str, doc: dict) -> dict:
    """Assign the next version number, write v<N>.json and v<N>.diff.json. `doc` is mutated and returned."""
    with _LOCK:
        nums = version_numbers(data, tenant, pid)
        n = (nums[-1] + 1) if nums else 1
        doc["version"] = n
        doc.setdefault("created_at", time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        doc.setdefault("status", "draft")
        prev = latest_full(data, tenant, pid)
        d = pdir(data, tenant, pid)
        _write(d / f"v{n}.json", doc)
        _write(d / f"v{n}.diff.json", {"version": n, "against": prev["version"] if prev else None,
                                       "diff": diff_bundles(prev.get("bundle") if prev else None, doc.get("bundle"))})
    return doc


def get_diff(data: Path, tenant: str, pid: str, a: int, b: int) -> dict:
    va, vb = load_version(data, tenant, pid, a), load_version(data, tenant, pid, b)
    if not va or not vb:
        raise KeyError("unknown version")
    return {"a": a, "b": b, "diff": diff_bundles(va.get("bundle"), vb.get("bundle"))}


# ---------------------------------------------------------------- three-way merge

def merge_list(base: list[dict], cur: list[dict], new: list[dict], key: Callable[[dict], str]) -> list[dict]:
    bm, cm, nm = ({key(x): x for x in lst} for lst in (base, cur, new))
    out: list[dict] = []
    seen: set[str] = set()
    for x in new:
        k = key(x)
        seen.add(k)
        if k in cm:
            out.append(x if (k in bm and cm[k] == bm[k]) else cm[k])  # console-edited (or not generated by us): keep theirs
        elif k in bm:
            continue  # removed in the console since the last publish: stays removed
        else:
            out.append(x)
    for x in cur:
        k = key(x)
        if k in seen:
            continue
        if k in bm and k not in nm:
            continue  # we generated it, the new version dropped it, the console left it alone: follow the new version
        out.append(x)
    return out


def merge_fields(base: dict, cur: dict, new: dict) -> dict:
    out = dict(cur)
    for k, nv in new.items():
        if k not in cur:
            out[k] = nv
        elif k in base:
            out[k] = nv if cur[k] == base[k] else cur[k]
        else:  # never generated by us: keep what is there unless it is empty
            out[k] = cur[k] if cur[k] not in ("", None, [], {}) else nv
    return out


def _norm_product(d: dict) -> dict:
    from acie.decision.domains.product_sales import Product
    return Product.model_validate(d).model_dump()


def _kb_merge(base: dict, cur: dict, new: dict) -> dict:
    out = {k: v for k, v in new.items() if k not in ("topics", "entries", "offerings", "interests")}
    out["tenant"], out["product_id"] = new.get("tenant"), new.get("product_id")
    for name, key in (("topics", "path"), ("entries", "id"), ("offerings", "id"), ("interests", "id")):
        out[name] = merge_list(base.get(name, []), cur.get(name, []), new.get(name, []), lambda x, k=key: x.get(k, ""))
    return out


def publish(data: Path, tenant: str, pid: str, n: int, kb_root: Path | None = None) -> dict:
    """Write version n into products.json, the KB file and the playbook file. Nothing is written unless everything validates."""
    from acie import playbook as pbm
    from acie.decision.domains.product_sales import Product
    from acie.knowledge.schema import KnowledgeBase
    from acie.knowledge.store import kb_dir, kb_path, save_kb, validate

    v = load_version(data, tenant, pid, n)
    if not v:
        raise PublishError("unknown version")
    if v.get("kind") == "gap_proposals" or not v.get("bundle"):
        raise PublishError("this version only holds pending proposals: approve them first")
    rep = v.get("report") or {}
    if not rep.get("publishable"):
        bad = [c["label"] for c in rep.get("checks", []) if not c["passed"] and c["blocking"]]
        raise PublishError("blocked by failed criteria: " + "; ".join(bad[:4]))
    bundle = v["bundle"]
    eff = v.get("effective") or {}
    st = state(data, tenant, pid)
    base_v = load_version(data, tenant, pid, st["published"]) if st.get("published") else None
    base = (base_v or {}).get("bundle") or {}

    # --- products.json
    pfile = Path(data) / "products.json"
    products = _read(pfile, {}) or {}
    newp = dict(bundle["product"], id=pid, tenant=tenant)
    cur_p = products.get(pid)
    if cur_p is None:
        merged_p = _norm_product(newp)
    else:
        merged_p = _norm_product(merge_fields(_norm_product(base["product"]) if base.get("product") else {}, _norm_product(cur_p), _norm_product(newp)))
    Product.model_validate(merged_p)

    # --- knowledge base
    kroot = kb_root or kb_dir()
    kpath = kb_path(tenant, pid, kroot)
    cur_kb = _read(kpath, {}) or {}
    merged_kb = _kb_merge((base.get("kb") or {}), cur_kb, bundle["kb"]) if cur_kb else dict(bundle["kb"])
    if eff.get("real_brand"):
        merged_kb["allow_synthetic"] = False
    try:
        kb = KnowledgeBase.model_validate(merged_kb)
    except Exception as ex:  # noqa: BLE001
        raise PublishError("merged knowledge base is invalid: " + str(ex)[:300])
    floor = ((merged_p.get("economics") or {}).get("cost") or 0) * (1 + ((merged_p.get("economics") or {}).get("min_margin_pct") or 0) / 100) or eff.get("floor")
    issues = validate(kb, {"profit_floor": floor} if floor else None)
    if issues:
        raise PublishError("knowledge base rules: " + "; ".join(issues[:3]))

    # --- playbook
    pb_cur = _read(pbm.playbook_dir(Path(data)) / tenant / f"{pid}.json", {}) or {}
    tree = bundle.get("tree") or {}
    nodes = merge_list((base.get("tree") or {}).get("nodes", []), pb_cur.get("nodes", []), tree.get("nodes", []), lambda x: x.get("id", "")) \
        if pb_cur else tree.get("nodes", [])
    pb = pbm.Playbook(tenant=tenant, product_id=pid, generated_by="productgen v%d (%s)" % (n, v.get("model", "")), nodes=nodes)

    # --- all good: back up what exists, then write
    stamp = time.strftime("%Y%m%d-%H%M%S")
    bdir = pdir(data, tenant, pid) / "backups" / stamp
    for src in (pfile, kpath, pbm.playbook_dir(Path(data)) / tenant / f"{pid}.json"):
        if src.exists():
            bdir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, bdir / (src.parent.name + "__" + src.name))
    products[pid] = merged_p
    _write_plain(pfile, products)
    save_kb(kb, kpath)
    pbm.save(Path(data), pb)
    if tree.get("ladder") or tree.get("fit_rules"):
        _write(kpath.with_suffix(".ladder.json"), {"ladder": tree.get("ladder", []), "fit_rules": tree.get("fit_rules", []), "version": n})
    st.update({"published": n, "published_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    _write(pdir(data, tenant, pid) / "state.json", st)
    v["status"] = "published"
    _write(pdir(data, tenant, pid) / f"v{n}.json", v)
    return {"ok": True, "version": n, "files": {"products": str(pfile), "kb": str(kpath), "playbook": str(pbm.playbook_dir(Path(data)) / tenant / f"{pid}.json")},
            "entries": len(kb.entries), "nodes": len(nodes), "backup": str(bdir) if bdir.exists() else None}


def _write_plain(p: Path, obj: Any) -> None:
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, p)
