"""The generation pipeline: profile -> information tree -> decision tree -> negotiation -> validate/repair -> save a NEW version.

Everything the model returns is cleaned by kbops (invalid pieces dropped and reported), checked by criteria, repaired
(at most `max_repair_rounds`, error list fed back), and saved as an immutable version. Nothing reaches a live file until Publish."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Awaitable, Callable

from . import criteria as C
from . import kbops as K
from . import store as S
from .llm import LLM
from .spec import Effective, Spec, resolve

STAGES = [("profile", "Product profile"), ("info_tree", "Information tree"), ("decision_tree", "Decision tree"),
          ("negotiation", "Negotiation"), ("validate", "Validate and repair"), ("save", "Save version")]
Progress = Callable[[str, str, str], None]  # stage, status (running|done|skipped|failed), detail


def _noop(stage: str, status: str, detail: str = "") -> None:
    return None


def spec_payload(spec: Spec, eff: Effective) -> dict:
    d = spec.model_dump(exclude={"criteria", "tenant", "product_id"})
    d["product_id"] = eff.product_id
    return {"spec": d, "mode": eff.provenance, "real_brand": eff.real_brand, "floor": eff.floor,
            "criteria": eff.criteria.model_dump(), "languages": spec.languages}


# ---------------------------------------------------------------- stage a: product profile

def _discount(spec: Spec) -> str:
    p = spec.price
    if p.list_price and p.offer_price is not None and p.offer_price < p.list_price:
        return f"{round((1 - p.offer_price / p.list_price) * 100)}% off"
    return ""


def price_strings(spec: Spec) -> dict:
    p = spec.price
    u = (" " + p.unit) if p.unit else ""
    return {"original_price": (K.money(p.list_price, p.currency) + u) if p.list_price else "",
            "offer_price": (K.money(p.offer_price, p.currency) + u) if p.offer_price is not None else (K.money(p.list_price, p.currency) + u if p.list_price else ""),
            "discount": _discount(spec), "offer_validity": p.validity}


def build_product(spec: Spec, eff: Effective, prof: dict) -> dict:
    from acie.knowledge.schema import LANGS
    ben = prof.get("benefits") if isinstance(prof.get("benefits"), dict) else {"en": prof.get("benefits") or []}
    pitch = K.t3(prof.get("pitch"))
    faqs = []
    for f in prof.get("faqs", []) or []:
        a = K.t3(f.get("answer")) if isinstance(f, dict) else {}
        kw = [str(k) for k in (f.get("keywords") or []) if str(k).strip()] if isinstance(f, dict) else []
        if kw and a.get("en"):
            faqs.append({"keywords": kw, "answer": a["en"], "answer_hi": a.get("hi", ""), "answer_hinglish": a.get("hinglish", "")})
    prod = {
        "id": eff.product_id, "tenant": spec.tenant,
        "description": ("productgen: public-knowledge facts only (real brand), claims need operator review" if eff.real_brand
                        else "productgen: synthetic demo facts for an operator-owned / authorised product"),
        "seller_name": spec.seller or spec.name, "product_name": spec.name, "one_liner": str(prof.get("one_liner", "") or ""),
        **price_strings(spec),
        "benefits": [str(x) for x in ben.get("en", [])], "benefits_hi": [str(x) for x in ben.get("hi", [])],
        "benefits_hinglish": [str(x) for x in ben.get("hinglish", [])],
        "call_to_action": str(prof.get("call_to_action") or "send you the details on SMS"),
        "pitch_en": pitch.get("en", ""), "pitch_hi": pitch.get("hi", ""), "pitch_hinglish": pitch.get("hinglish", ""),
        "knowledge": str(prof.get("knowledge", "") or ""),
        "topic_words": [str(w) for w in prof.get("topic_words", []) or []][:60],
        "faqs": faqs, "tone": str(spec.tone or prof.get("tone") or "warm, respectful and conversational, never pushy"),
        "pronunciations": {str(k): str(v) for k, v in (prof.get("pronunciations") or {}).items()} if isinstance(prof.get("pronunciations"), dict) else {},
        "payment_link": "", "sms_text": "", "official_site": spec.website, "style": "natural",
    }
    dq, isrc = K.t3(prof.get("discovery_question")), K.t3(prof.get("info_source"))
    if dq.get("en"):
        prod["discovery_question"] = {k: dq[k] for k in LANGS if k in dq}
    if isrc.get("en"):
        prod["info_source"] = {k: isrc[k] for k in LANGS if k in isrc}
    if eff.real_brand:
        prod["allowed_numbers"] = ["+910000000000"]  # blocks dialing until the operator sets real test phones in the console
    return prod


# ---------------------------------------------------------------- stage b: information tree

def _top(kb: dict) -> list[str]:
    return [t["path"] for t in kb["topics"] if "/" not in t["path"]]


async def build_kb(spec: Spec, eff: Effective, llm: LLM, prof: dict, notes: list[str], progress: Progress = _noop) -> dict:
    base = spec_payload(spec, eff)
    plan = await llm.json("kb_plan", {**base, "profile": {"one_liner": prof.get("one_liner"), "benefits": prof.get("benefits")}})
    kb = {"product_id": eff.product_id, "tenant": spec.tenant, "allow_synthetic": eff.allow_synthetic, "root_overview": None,
          "description": ("Public-knowledge KB (real brand)" if eff.real_brand else "Synthetic KB") + f" for {spec.name}, generated by productgen",
          "topics": [], "entries": [], "offerings": [], "interests": []}
    K.add_topics(kb, plan.get("topics", []), notes)
    K.add_interests(kb, plan.get("interests", []))
    K.add_offerings(kb, plan.get("offerings", []), spec, eff, notes, limit=eff.criteria.max_offerings)
    tops = _top(kb)
    # one top-level topic per call and at most CHUNK entries per call: a whole KB (or even 3 topics x 3 languages) in one
    # reply runs past the model's output limit and comes back cut off (live Gemini run: ~60 KB, invalid JSON)
    per = max(1, int(os.getenv("PRODUCTGEN_TOPICS_PER_CALL", "1")))
    chunk = max(3, int(os.getenv("PRODUCTGEN_ENTRIES_PER_CALL", "6")))
    need = max(6, -(-eff.criteria.min_entries // max(1, len(tops))) + 2)  # entries per top-level topic, with a margin
    groups = []
    for i in range(0, len(tops), per):
        g = tops[i:i + per]
        left, b = need * len(g), 0
        while left > 0:
            groups.append((g, min(chunk, left), b))
            left -= chunk
            b += 1
    sem = asyncio.Semaphore(int(os.getenv("PRODUCTGEN_CONCURRENCY", "4")))
    done = 0

    async def one(item: tuple[list[str], int, int]) -> dict:
        nonlocal done
        g, n, b = item
        async with sem:
            try:
                r = await llm.json("kb_entries", {**base, "topics": [t for t in kb["topics"] if t["path"].split("/")[0] in g],
                                                  "per_topic": n, "max_entries": n, "batch": b,
                                                  "note": f"batch {b} of this topic: cover DIFFERENT questions than other batches "
                                                          f"(angle {b + 1}); ids must be unique (suffix -b{b})",
                                                  "offering_ids": [o["id"] for o in kb["offerings"]],
                                                  "interest_ids": [i["id"] for i in kb["interests"]],
                                                  "existing_questions": []})
            except Exception as e:  # one bad batch must not sink the run: the repair loop tops up short topics
                notes.append(f"entries batch for {g} failed: {str(e)[:120]}")
                r = {}
            done += 1
            progress("info_tree", "running", f"entries: {done}/{len(groups)} batches")
            return r

    results = await asyncio.gather(*(one(g) for g in groups))  # gather keeps group order: deterministic merge
    for r in results:
        K.add_entries(kb, r.get("entries", []), eff, notes)
    return kb


# ---------------------------------------------------------------- stage c: decision tree

async def build_tree(spec: Spec, eff: Effective, llm: LLM, prof: dict, kb: dict, notes: list[str]) -> dict:
    base = spec_payload(spec, eff)
    kbs = {"topics": [{"path": t["path"], "title": t["title"].get("en")} for t in kb["topics"]], "offerings": [{"id": o["id"], "name": o["name"].get("en")} for o in kb["offerings"]],
           "interests": [{"id": i["id"], "keywords": i["keywords"][:5]} for i in kb["interests"]]}
    pb, lad = await asyncio.gather(llm.json("playbook", {**base, "profile": {"one_liner": prof.get("one_liner"), "benefits": prof.get("benefits")}, "kb": kbs}),
                                   llm.json("ladder", {**base, "kb": kbs}))
    nodes: list[dict] = []
    K.add_nodes(nodes, pb.get("nodes", []), notes, limit=eff.criteria.max_situations)
    ladder = K.clean_ladder(lad.get("ladder", []))
    tags = {i["id"] for i in kb["interests"]}
    rules = []
    om = {o["id"]: o for o in kb["offerings"]}
    for r in lad.get("fit_rules", []) or []:
        o = om.get(K.kebab(r.get("offering"))) if isinstance(r, dict) else None
        if not o:
            continue
        who = [w for w in (K.kebab(x) for x in r.get("who_suits", []) or []) if w in tags]
        try:
            mf = max(0.5, min(3.0, float(r.get("min_fit", o["min_fit"]))))
        except (TypeError, ValueError):
            mf = o["min_fit"]
        if who:
            o["who_suits"], o["min_fit"] = who, mf
        rules.append({"offering": o["id"], "who_suits": o["who_suits"], "min_fit": o["min_fit"]})
    for r in ladder:
        r["interests"] = [i for i in r["interests"] if i in tags]
    return {"nodes": nodes, "ladder": ladder, "fit_rules": rules, "clusters": K.clusters(kb)}


# ---------------------------------------------------------------- stage d: negotiation

async def build_negotiation(spec: Spec, eff: Effective, llm: LLM) -> dict:
    eco = spec.economics
    if eff.real_brand:
        return {"suggested": False, "reason": "a real company's prices are theirs: no concessions are invented", "economics": None}
    if eff.floor is None:
        return {"suggested": False, "reason": "no floor or cost + margin was given; the generator never invents a floor. Fill it in to get a suggestion.", "economics": None}
    cur = spec.price.offer_price if spec.price.offer_price is not None else spec.price.list_price
    if cur is None:
        return {"suggested": False, "reason": "no price was given", "economics": None}
    steps: list[dict] = []
    if eco.ladder:
        for i, s in enumerate(eco.ladder):
            if s.get("kind") == "price" and not (eff.floor <= float(s.get("price", 0)) < cur):
                continue
            steps.append({"kind": s.get("kind", "perk"), "id": K.kebab(s.get("id") or f"step-{i + 1}"), "price": float(s.get("price", 0) or 0),
                          "text": s.get("text") or {}, "short": s.get("short") or {}})
        why = "operator-written ladder (steps below the floor were dropped)"
    else:
        r = await llm.json("negotiation", {**spec_payload(spec, eff), "current_price": cur, "operator_perks": eco.perks})
        perks = r.get("perks", []) or []
        if eco.perks:  # only perks the operator allowed
            perks = perks[:len(eco.perks)]
        for i, p in enumerate(perks[:3]):
            tx, sh = K.t3(p.get("text")), K.t3(p.get("short"))
            if tx.get("en"):
                steps.append({"kind": "perk", "id": K.kebab(p.get("id") or f"perk-{i + 1}"), "price": 0.0, "text": tx, "short": sh})
        prices = sorted({float(x.get("price", 0)) for x in r.get("price_steps", []) or [] if isinstance(x, dict)
                         and isinstance(x.get("price"), (int, float)) and eff.floor <= float(x["price"]) < cur}, reverse=True)
        for i, pr in enumerate(prices[:3]):
            steps.append({"kind": "price", "id": f"price-{i + 1}", "price": pr, "text": {}, "short": {}})
        why = "suggested by the model from the operator's floor; review before use"
    cost, margin = (eco.cost, eco.min_margin_pct) if (eco.cost is not None and eco.floor is None) else (eff.floor, 0.0)
    return {"suggested": True, "reason": why, "economics": {"currency": spec.price.currency, "unit": spec.price.unit, "list_price": cur, "cost": cost,
                                                              "min_margin_pct": margin, "ladder": steps}}


# ---------------------------------------------------------------- stage e: validate + repair

def deterministic_fixes(bundle: dict, spec: Spec, eff: Effective) -> list[str]:
    kb, tree = bundle["kb"], bundle["tree"]
    log = []
    if len(kb["offerings"]) > eff.criteria.max_offerings:
        drop = {o["id"] for o in kb["offerings"][eff.criteria.max_offerings:]}
        kb["offerings"] = kb["offerings"][:eff.criteria.max_offerings]
        for e in kb["entries"]:
            e["offerings"] = [o for o in e["offerings"] if o not in drop]
        log.append(f"trimmed {len(drop)} offerings above the maximum")
    if len(tree["nodes"]) > eff.criteria.max_situations:
        del tree["nodes"][eff.criteria.max_situations:]
        log.append("trimmed situations above the maximum")
    have = {e["topic_path"] for e in kb["entries"]}
    gone = {t["path"] for t in kb["topics"] if "/" in t["path"] and t["path"] not in have and not any(q["path"].startswith(t["path"] + "/") for q in kb["topics"])}
    if gone:  # a subtopic nothing is filed under would only ever serve its overview
        kb["topics"] = [t for t in kb["topics"] if t["path"] not in gone]
        for o in kb["offerings"]:
            o["upsell_from"] = [x for x in o["upsell_from"] if x not in gone]
        log.append(f"dropped {len(gone)} empty subtopics")
    used = {t for e in kb["entries"] for t in e["interests"]} | {t for o in kb["offerings"] for t in o["who_suits"]}
    unused = {i["id"] for i in kb["interests"]} - used
    if unused:  # an interest nothing speaks to can never be inferred usefully
        kb["interests"] = [i for i in kb["interests"] if i["id"] not in unused]
        for i in kb["interests"]:
            i["related"] = [r for r in i["related"] if r not in unused]
        for t in kb["topics"]:
            t["interests"] = [x for x in t["interests"] if x not in unused]
        for r in tree["ladder"]:
            r["interests"] = [x for x in r["interests"] if x not in unused]
        log.append(f"dropped {len(unused)} unused interest tags")
    p = bundle["product"]
    kept = [f for f in p.get("faqs", []) if not any(C.banned_hits(x, spec, eff) for x in (f["answer"], f.get("answer_hi", ""), f.get("answer_hinglish", "")))]
    if len(kept) != len(p.get("faqs", [])):
        log.append(f"dropped {len(p['faqs']) - len(kept)} FAQs with banned content")
        p["faqs"] = kept
    for o in kb["offerings"]:  # drop pitch lines with banned content
        for lang, vs in o["pitch"].items():
            good = [v for v in vs if not C.banned_hits(v, spec, eff)]
            if len(good) != len(vs):
                o["pitch"][lang] = good
                log.append(f"dropped pitch lines in {o['id']}")
    return log


def _items(report: dict, *check_ids: str) -> set[str]:
    out: set[str] = set()
    for c in report["checks"]:
        if c["id"] in check_ids and not c["passed"]:
            out |= set(c["items"])
    return out


async def repair(bundle: dict, report: dict, spec: Spec, eff: Effective, llm: LLM, notes: list[str]) -> list[str]:
    """One repair round. The failing checks' errors go back to the model; only what it returns for those items is applied."""
    kb, tree = bundle["kb"], bundle["tree"]
    base = spec_payload(spec, eff)
    log: list[str] = []
    errs = {c["id"]: c["detail"] for c in report["checks"] if not c["passed"]}
    bad_items = _items(report, "languages", "banned", "provenance", "numeric", "duplicates")
    ids = {e["id"] for e in kb["entries"]}
    fix_entries = sorted(i for i in bad_items if i in ids)
    if fix_entries:
        r = await llm.json("kb_fix", {**base, "errors": errs, "fix": [e for e in kb["entries"] if e["id"] in set(fix_entries)],
                                      "topics": [t["path"] for t in kb["topics"]], "offering_ids": [o["id"] for o in kb["offerings"]],
                                      "interest_ids": [i["id"] for i in kb["interests"]]})
        done = K.replace_entries(kb, r.get("entries", []), eff, notes)
        log.append(f"rewrote {len(done)} entries")
        if "banned" in errs or "duplicates" in errs:  # whatever is still banned or duplicated after the rewrite is removed
            still = {i for i in _items(C.check(bundle, spec, eff), "banned", "duplicates") if i in ids}
            if still:
                kb["entries"] = [e for e in kb["entries"] if e["id"] not in still]
                log.append(f"removed {len(still)} entries that stayed non-compliant")
    bad_nodes = {i[5:] for i in bad_items if i.startswith("node:")}
    if bad_nodes or ("situations" in errs and len(tree["nodes"]) < eff.criteria.min_situations):
        need = max(0, eff.criteria.min_situations - (len(tree["nodes"]) - len(bad_nodes)))
        keep = [n for n in tree["nodes"] if n["id"] not in bad_nodes]
        r = await llm.json("playbook", {**base, "errors": errs, "have": [n["id"] for n in keep], "need": need, "fix": [n for n in tree["nodes"] if n["id"] in bad_nodes],
                                        "kb": {"topics": [t["path"] for t in kb["topics"]]}})
        tree["nodes"] = keep
        K.add_nodes(tree["nodes"], r.get("nodes", []), notes, limit=eff.criteria.max_situations)
        log.append("regenerated playbook situations")
    if "ladder" in errs or any(i.startswith("ladder:") for i in bad_items):
        r = await llm.json("ladder", {**base, "errors": errs, "previous": tree["ladder"], "kb": {"interests": [i["id"] for i in kb["interests"]]}})
        lad = K.clean_ladder(r.get("ladder", []))
        if len(lad) >= len(tree["ladder"]):
            tree["ladder"] = lad
        log.append("regenerated the discovery ladder")
    n_have = len(kb["entries"])
    if n_have < eff.criteria.min_entries or _items(report, "coverage"):
        counts = {t: 0 for t in _top(kb)}
        for e in kb["entries"]:
            counts[e["topic_path"].split("/")[0]] = counts.get(e["topic_path"].split("/")[0], 0) + 1
        short = sorted(counts, key=lambda t: (counts[t], t))[:max(3, len(counts) // 3)]
        want = max(0, eff.criteria.min_entries - n_have)
        if want or any(counts[t] == 0 for t in counts):
            r = await llm.json("kb_entries", {**base, "errors": errs, "topics": [t for t in kb["topics"] if t["path"].split("/")[0] in short],
                                              "per_topic": max(4, -(-max(want, 4) // max(1, len(short))) + 1),
                                              "offering_ids": [o["id"] for o in kb["offerings"]], "interest_ids": [i["id"] for i in kb["interests"]],
                                              "existing_questions": [(e["question_variants"].get("en") or [""])[0] for e in kb["entries"]][:400]})
            added = K.add_entries(kb, r.get("entries", []), eff, notes)
            log.append(f"added {len(added)} entries")
    if len(kb["offerings"]) < eff.criteria.min_offerings:
        r = await llm.json("kb_plan", {**base, "errors": errs, "only": "offerings", "have_offerings": [o["id"] for o in kb["offerings"]],
                                       "interests": [{"id": i["id"]} for i in kb["interests"]], "topics": [t["path"] for t in kb["topics"]]})
        added = K.add_offerings(kb, r.get("offerings", []), spec, eff, notes, limit=eff.criteria.max_offerings)
        log.append(f"added {len(added)} offerings")
    t_bad = {i[6:] for i in bad_items if i.startswith("topic:")}
    o_bad = {i[9:] for i in bad_items if i.startswith("offering:")}
    if t_bad or o_bad:
        r = await llm.json("kb_plan", {**base, "errors": errs, "only": "fix_text", "topics": [t for t in kb["topics"] if t["path"] in t_bad],
                                       "offerings": [o for o in kb["offerings"] if o["id"] in o_bad]})
        tm = {path: t for t in (r.get("topics") or []) if isinstance(t, dict) and (path := K.path_clean(t.get("path")))}
        for t in kb["topics"]:
            nt = tm.get(t["path"])
            if nt:
                for k in ("title", "overview"):
                    v = K.t3(nt.get(k))
                    if v.get("en"):
                        t[k] = {**t[k], **v}
        om = {K.kebab(o.get("id")): o for o in (r.get("offerings") or []) if isinstance(o, dict)}
        for o in kb["offerings"]:
            no = om.get(o["id"])
            if no:
                v = K.t3(no.get("name"))
                if v.get("en"):
                    o["name"] = {**o["name"], **v}
        log.append("rewrote topic / offering text")
    prod_bad = {i for i in bad_items if i.startswith("product:")}
    if prod_bad:
        r = await llm.json("profile", {**base, "errors": errs})
        np_ = build_product(spec, eff, r)
        for k in ("pitch_en", "pitch_hi", "pitch_hinglish", "one_liner", "knowledge", "benefits", "benefits_hi", "benefits_hinglish"):
            if np_.get(k):
                bundle["product"][k] = np_[k]
        log.append("rewrote product text")
    return log


def finish_bundle(bundle: dict, spec: Spec, eff: Effective) -> dict:
    bundle["tree"]["clusters"] = K.clusters(bundle["kb"])
    bundle["review"] = sorted(e["id"] for e in bundle["kb"]["entries"] if "review" in str(e.get("source", ""))) if eff.real_brand else []
    return C.check(bundle, spec, eff)


async def validate_and_repair(bundle: dict, spec: Spec, eff: Effective, llm: LLM, notes: list[str], progress: Progress = _noop) -> tuple[dict, list[dict]]:
    history: list[dict] = []
    fixes = deterministic_fixes(bundle, spec, eff)
    report = finish_bundle(bundle, spec, eff)
    history.append({"round": 0, "failed": [c["id"] for c in report["checks"] if not c["passed"]], "fixes": fixes})
    for rnd in range(1, eff.criteria.max_repair_rounds + 1):
        if report["passed"]:
            break
        progress("validate", "running", f"repair round {rnd}: " + ", ".join(history[-1]["failed"]))
        log = await repair(bundle, report, spec, eff, llm, notes)
        log += deterministic_fixes(bundle, spec, eff)
        report = finish_bundle(bundle, spec, eff)
        history.append({"round": rnd, "failed": [c["id"] for c in report["checks"] if not c["passed"]], "fixes": log})
    return report, history


# ---------------------------------------------------------------- the whole run

async def generate(spec: Spec, llm: LLM, data: Path, progress: Progress = _noop) -> dict:
    """Run every stage and save a new version. Returns the saved version document."""
    eff = resolve(spec)
    notes: list[str] = []
    progress("profile", "running", "writing the product profile")
    prof = await llm.json("profile", spec_payload(spec, eff))
    product = build_product(spec, eff, prof)
    progress("profile", "done", f"{len(product['faqs'])} FAQs, {len(product['benefits'])} benefits")

    progress("info_tree", "running", "topics, interests, offerings")
    kb = await build_kb(spec, eff, llm, prof, notes, progress)
    progress("info_tree", "done", f"{len(_top(kb))} topics, {len(kb['entries'])} entries, {len(kb['offerings'])} offerings")

    progress("decision_tree", "running", "situations and discovery ladder")
    tree = await build_tree(spec, eff, llm, prof, kb, notes)
    progress("decision_tree", "done", f"{len(tree['nodes'])} situations, {len(tree['ladder'])} ladder rungs")

    progress("negotiation", "running", "")
    neg = await build_negotiation(spec, eff, llm)
    if neg["economics"]:
        product["economics"] = neg["economics"]
    progress("negotiation", "done" if neg["suggested"] else "skipped", neg["reason"][:140])

    bundle = {"product": product, "kb": kb, "tree": tree, "negotiation": neg, "review": []}
    progress("validate", "running", "checking the criteria")
    report, history = await validate_and_repair(bundle, spec, eff, llm, notes, progress)
    failed = [c["label"] for c in report["checks"] if not c["passed"]]
    progress("validate", "done" if report["passed"] else "failed", "all criteria pass" if report["passed"] else f"{len(failed)} criteria still fail")

    progress("save", "running", "")
    doc = {"kind": "generate", "status": "draft" if report["publishable"] else "needs_review", "model": getattr(llm, "model", ""), "spec": spec.model_dump(),
           "effective": eff.model_dump(), "bundle": bundle, "report": report, "repair_history": history, "notes": notes[:200]}
    S.save_version(data, spec.tenant, eff.product_id, doc)
    progress("save", "done", f"version v{doc['version']}")
    return doc
