"""Live information-explored map for the operator console, plus the negotiation-range checks for the product editor.

snapshot(session, product, ...) reads one call's session (read-only) and returns a plain JSON dict:

  decision path   one node per agent turn: stage -> interest clusters (with evidence) -> KB topic -> leaf action
  information map the product's KB taxonomy (topic -> entries, plus offerings), each node coloured by state:
                  untouched | asked | answered | recommended | closed
  facts           what the caller has said (goal, schedule, budget, objections, decisions)
  negotiation     list price, current offer, floor reached, perks given, agreed price

It works with ACIE_KB off or no KB for the product (stage / intent / facts / ledger / negotiation only). It never mutates the
session (tree_hint is called with commit=False) and never raises into the call: `on_turn` wraps everything and logs once.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

log = logging.getLogger("voice.live_map")

STAGES = ("open", "discover", "educate", "recommend", "objection", "negotiate", "close", "after_close")
STATE_RANK = {"untouched": 0, "asked": 1, "answered": 2, "recommended": 3, "closed": 4}
FACT_KEYS = {"goal": ("goal", "interest", "use_case", "_need", "need"), "sub_goal": ("sub_goal", "subgoal"),
             "schedule": ("schedule", "callback_time"), "budget": ("budget",)}
_warned = False


def _warn_once(where: str, exc: Exception) -> None:
    global _warned
    if not _warned:
        _warned = True
        log.warning("live map (%s) failed, further errors are silent: %r", where, exc)


def _get(o: Any, k: str, default: Any = None) -> Any:
    if o is None:
        return default
    return o.get(k, default) if isinstance(o, dict) else getattr(o, k, default)


def _num(v: Any) -> float | None:
    try:
        return float(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _json(raw: str, default: Any) -> Any:
    try:
        return json.loads(raw) if raw else default
    except ValueError:
        return default


# ------------------------------------------------------------------------------------------------- real-company check

def is_real_company(product: Any) -> bool:
    """A real company's product (its prices are the seller's own): no concessions can be configured."""
    try:
        from acie.knowledge.store import load_kb
        kb = load_kb(product)
        if kb is not None:
            return not kb.allow_synthetic
    except Exception:  # noqa: BLE001
        pass
    return str(_get(product, "description", "")).lower().startswith("real company")


# ------------------------------------------------------------------------------------------------- snapshot

def _customer_turns(session: Any) -> list[str]:
    return [u.text for u in getattr(session, "history", []) if getattr(u, "role", "") == "customer"]


def _agent_count(session: Any) -> int:
    return sum(1 for u in getattr(session, "history", []) if getattr(u, "role", "") == "agent")


def _signals(session: Any) -> Any:
    raw = (getattr(session, "dialog", {}) or {}).get("_signals", "")
    if not raw:
        return None
    try:
        from acie.signals import TurnSignals
        return TurnSignals.model_validate_json(raw)
    except Exception:  # noqa: BLE001
        return None


def _stage_without_kb(session: Any, signals: Any, text: str) -> tuple[str, str]:
    from acie.tree.rules import QUESTION, Ctx, stage_for
    ctx = getattr(session, "context", None)
    objection = getattr(signals, "objection", "none") or "none"
    c = Ctx(len(_customer_turns(session)), text or "", getattr(signals, "intent", "unknown") or "unknown", objection,
            float(getattr(signals, "buying_interest", 0.0) or 0.0), getattr(ctx, "decision", "undecided"),
            bool(ctx and ctx.has("payment_link")), bool(text and QUESTION.search(text)), False, False)
    return stage_for(c)


def _tree_layers(session: Any, product: Any, text: str, signals: Any, use_kb: bool) -> dict:
    """The four layers of this turn: stage, clusters, topic, action (+ coverage when a KB exists)."""
    if use_kb:
        try:
            from acie.tree.hint import tree_hint
            hint = tree_hint(session, signals, text, product, None, commit=False)
            if hint is not None:
                return {"stage": hint.stage, "rule": hint.stage_rule, "clusters": hint.clusters, "topic": hint.topic,
                        "action": dict(hint.action), "entries": [e.id for e in hint.entries],
                        "offering": hint.offering.id if hint.offering else ""}
        except Exception as e:  # noqa: BLE001
            _warn_once("tree_hint", e)
    stage, rule = _stage_without_kb(session, signals, text)
    return {"stage": stage, "rule": rule, "clusters": [], "topic": "", "action": {}, "entries": [], "offering": ""}


def _info_map(session: Any, product: Any, stage: str) -> tuple[dict | None, dict]:
    """KB taxonomy coloured by coverage state, and the coverage numbers. (None, {}) when there is no KB."""
    from acie.knowledge.store import kb_enabled, load_kb
    if not kb_enabled():
        return None, {}
    kb = load_kb(product)
    if kb is None:
        return None, {}
    from acie.tree import state as S
    st = S.load(session)
    asked, given, offered = set(st["asked"]), set(st["given"]), st["offered"]
    ctx = getattr(session, "context", None)
    accepted = getattr(ctx, "decision", "") == "accepted" or stage == "after_close"
    topics: dict[str, dict] = {}
    for t in sorted(kb.topics, key=lambda t: t.path):
        topics[t.path] = {"path": t.path, "parent": t.parent, "title": t.title.en, "state": "untouched", "entries": [], "asked": 0,
                          "answered": 0, "total": 0}
    for e in sorted(kb.usable_entries(), key=lambda e: e.id):
        state = "answered" if e.id in given else "asked" if e.id in asked else "untouched"
        node = topics.get(e.topic_path)
        if node is None:
            continue
        node["entries"].append({"id": e.id, "q": (e.question_variants.en or e.question_variants.all() or [e.id])[0], "state": state})
        node["total"] += 1
        node["asked"] += state in ("asked", "answered")
        node["answered"] += state == "answered"
    for node in topics.values():  # a topic is as far along as its furthest entry; parents inherit from children
        for e in node["entries"]:
            if STATE_RANK[e["state"]] > STATE_RANK[node["state"]]:
                node["state"] = e["state"]
    for node in sorted(topics.values(), key=lambda n: -n["path"].count("/")):
        par = topics.get(node["parent"])
        if par is not None:
            par["asked"] += node["asked"]
            par["answered"] += node["answered"]
            par["total"] += node["total"]
            if STATE_RANK[node["state"]] > STATE_RANK[par["state"]]:
                par["state"] = node["state"]
    offerings = []
    for o in sorted(kb.offerings, key=lambda o: o.id):
        state = "untouched"
        if offered.get(o.id, 0) > 0:
            state = "closed" if accepted else "recommended"
        offerings.append({"id": o.id, "name": o.name.en, "state": state, "pitched": offered.get(o.id, 0), "included": o.included,
                          "price": o.price_label.en if o.price_label else ""})
    total = st["total"] or len(kb.usable_entries())
    cov = {"kb_total": total, "asked": len(asked), "given": len(given), "given_pct": round(100.0 * len(given) / total, 1) if total else 0.0}
    return {"topics": list(topics.values()), "offerings": offerings}, cov


def _facts(session: Any, signals: Any) -> dict:
    d = getattr(session, "dialog", {}) or {}
    ctx = getattr(session, "context", None)
    out: dict = {}
    for name, keys in FACT_KEYS.items():
        v = next((str(d[k]) for k in keys if d.get(k)), "")
        if v:
            out[name] = v
    try:
        from acie.tree.interest import stated_facts
        sched = stated_facts(_customer_turns(session))
        if sched:
            out["schedule"] = "; ".join(([out["schedule"]] if out.get("schedule") else []) + sched)
    except Exception:  # noqa: BLE001
        pass
    for k in ("existing_user", "callback_note"):
        if d.get(k):
            out[k] = str(d[k])
    objections = list(getattr(ctx, "objections", []) or [])
    sig_obj = getattr(signals, "objection", "none")
    if sig_obj and sig_obj != "none" and sig_obj not in objections:
        objections.append(sig_obj)
    out["objections"] = objections
    out["decision"] = getattr(ctx, "decision", "undecided")
    out["outcome"] = getattr(ctx, "outcome", "")
    out["promised"] = list(getattr(ctx, "promised", []) or [])
    out["quoted"] = list(getattr(ctx, "quoted", []) or [])
    out["told"] = list(getattr(ctx, "told", []) or [])
    out["questions"] = [q.get("q", "") if isinstance(q, dict) else str(q) for q in (getattr(ctx, "questions", []) or [])][-6:]
    return out


def _negotiation(session: Any, product: Any, prev: dict | None) -> dict | None:
    eco = _get(product, "economics")
    d = getattr(session, "dialog", {}) or {}
    list_price = _num(_get(eco, "list_price")) or _num(d.get("list_price")) or 0.0
    if not list_price:
        return None
    floor = _num(d.get("profit_floor"))
    if floor is None:
        floor = round((_num(_get(eco, "cost")) or 0.0) * (1 + (_num(_get(eco, "min_margin_pct")) or 0.0) / 100.0), 2)
    neg = _json(d.get("_neg", ""), {})
    cur = _num(neg.get("price")) or list_price
    agreed = _num(d.get("closed_price"))
    old = (prev or {}).get("negotiation") or {}
    perks = list(old.get("perks_given", []))
    offers = list(old.get("offers", []))
    offer = _json(d.get("neg_offer", ""), {})
    if offer:
        key = {"price": _num(offer.get("price")), "perk": offer.get("perk") or ""}
        if key not in offers:
            offers.append(key)
        if key["perk"] and key["perk"] not in perks:
            perks.append(key["perk"])
    ctx = getattr(session, "context", None)
    margin = _num(d.get("margin_pct"))
    return {"currency": _get(eco, "currency", "₹"), "unit": _get(eco, "unit", ""), "list_price": list_price, "floor": floor,
            "current_offer": cur, "floor_reached": bool(floor) and cur <= floor + 0.01 or bool(neg.get("final_said")),
            "final_said": bool(neg.get("final_said")), "pushes": int(neg.get("pushes", 0) or 0), "reason": neg.get("reason", ""),
            "perks_given": perks, "offers": offers, "agreed_price": agreed, "margin_pct": margin,
            "active": bool(neg.get("pushes")) or agreed is not None,
            "accepted": getattr(ctx, "decision", "") == "accepted"}


def snapshot(session: Any, product: Any, call_id: str = "", prev: dict | None = None, agent_intent: str = "") -> dict:
    """One read of the call. `prev` is the previous snapshot of the same call (the path and perks accumulate across turns)."""
    from acie.knowledge.store import kb_enabled
    sig = _signals(session)
    custs = _customer_turns(session)
    text = custs[-1] if custs else ""
    use_kb = kb_enabled()
    layers = _tree_layers(session, product, text, sig, use_kb)
    try:
        info, cov = _info_map(session, product, layers["stage"])
    except Exception as e:  # noqa: BLE001
        _warn_once("info map", e)
        info, cov = None, {}
    n = _agent_count(session)
    node = {"n": n, "turn": len(custs), "stage": layers["stage"], "rule": layers["rule"], "clusters": layers["clusters"][:3],
            "topic": layers["topic"], "action": layers["action"], "intent": agent_intent or str(getattr(sig, "intent", "") or ""),
            "heard": str(getattr(sig, "intent", "") or ""), "objection": str(getattr(sig, "objection", "none") or "none"),
            "said": text[:90]}
    path = [p for p in (prev or {}).get("path", []) if p.get("n") != n]
    path.append(node)
    return {"v": 1, "call_id": call_id, "t": time.time(), "kb": info is not None, "stage": layers["stage"], "stages": list(STAGES),
            "turn": len(custs), "path": path, "clusters": layers["clusters"], "topic": layers["topic"], "action": layers["action"],
            "offering": layers["offering"], "coverage": cov, "tree": info or {"topics": [], "offerings": []},
            "facts": _facts(session, sig), "negotiation": _negotiation(session, product, prev)}


def from_record(record: dict, product: Any = None) -> dict:
    """A degraded map rebuilt from a saved call record (turns + context ledger) when no snapshot was stored."""
    ctx = record.get("context") or {}
    path, n = [], 0
    for i, t in enumerate(record.get("turns", [])):
        if t.get("role") != "agent":
            continue
        n += 1
        nlu = t.get("nlu") or {}
        tr = t.get("trace") or {}
        prev_c = record["turns"][i - 1] if i and record["turns"][i - 1].get("role") == "customer" else {}
        path.append({"n": n, "turn": n - 1, "stage": "", "rule": "", "clusters": [], "topic": tr.get("pb_node") or "", "action": {},
                     "intent": t.get("intent", ""), "heard": nlu.get("intent", ""), "objection": nlu.get("objection", "none"),
                     "said": (prev_c.get("text") or "")[:90]})
    decision = ctx.get("decision", "undecided")
    stage = "after_close" if decision == "accepted" else (path[-1]["stage"] if path else "open")
    return {"v": 1, "call_id": record.get("call_id", ""), "t": record.get("ended_at") or time.time(), "kb": False, "stage": stage,
            "stages": list(STAGES), "turn": sum(1 for t in record.get("turns", []) if t.get("role") == "customer"), "path": path,
            "clusters": [], "topic": "", "action": {}, "offering": "", "coverage": {}, "tree": {"topics": [], "offerings": []},
            "facts": {"objections": ctx.get("objections", []), "decision": decision, "outcome": ctx.get("outcome", ""),
                      "promised": ctx.get("promised", []), "quoted": ctx.get("quoted", []), "told": ctx.get("told", []),
                      "questions": [q.get("q", "") for q in ctx.get("questions", [])][-6:]},
            "negotiation": None, "rebuilt": True}


# ------------------------------------------------------------------------------------------------- server hooks

def _product_for(app: Any, call: Any, cs: Any = None) -> Any:
    p = getattr(cs, "product", None) if cs is not None else None
    if p is not None:
        return p
    try:
        return app.products().get(call.campaign)
    except Exception:  # noqa: BLE001
        return None


def capture(app: Any, call: Any, session: Any, cs: Any = None, agent_intent: str = "") -> dict | None:
    """Compute, remember (app.maps) and return the call's snapshot; None (and a one-time log) when anything goes wrong."""
    try:
        snap = snapshot(session, _product_for(app, call, cs), call.call_id, app.maps.get(call.call_id), agent_intent)
        app.maps[call.call_id] = snap
        return snap
    except Exception as e:  # noqa: BLE001
        _warn_once("snapshot", e)
        return None


def on_turn(app: Any, event: dict) -> None:
    """App.broadcast hook: after the agent's turn, push a {"type": "map"} event to every console listener. Never raises."""
    try:
        cid = event.get("call_id", "")
        cs = app.live_sessions.get(cid)
        call = app.calls.get(cid)
        if cs is None or call is None:
            return
        session = cs.pipeline.sessions.get(cs.session_id)
        if session is None:
            return
        snap = capture(app, call, session, cs, str(event.get("intent", "")))
        if snap is not None:
            app.broadcast({"type": "map", "call_id": cid, "map": snap})
    except Exception as e:  # noqa: BLE001
        _warn_once("on_turn", e)


# ------------------------------------------------------------------------------------------------- economics

PERK_LANGS = ("en", "hi", "hinglish")


def validate_economics(eco: Any, product: Any = None) -> None:
    """Raise ValueError (a clear sentence) when the negotiation range is unsafe. An all-empty economics (no concessions) is valid."""
    ladder = list(_get(eco, "ladder", []) or [])
    lp, cost, margin = (_num(_get(eco, k)) or 0.0 for k in ("list_price", "cost", "min_margin_pct"))
    if not (lp or cost or margin or ladder):
        return
    if product is not None and is_real_company(product):
        raise ValueError("No concessions for a real company's product: these are the seller's own prices, so the negotiation range must stay empty.")
    if lp <= 0:
        raise ValueError("List price must be above 0.")
    if cost < 0 or not 0 <= margin < 100:
        raise ValueError("Cost cannot be negative and the minimum margin must be between 0% and 99%.")
    floor = round(max(cost * (1 + margin / 100.0), _num(_get(eco, "floor_price")) or 0.0), 2)
    if floor > lp:
        raise ValueError(f"The floor ({floor:g}) is above the list price ({lp:g}); lower the cost or margin, or raise the list price.")
    seen: set[str] = set()
    last_price = lp
    for i, s in enumerate(ladder, 1):
        kind = _get(s, "kind", "perk")
        if kind == "price":
            price = _num(_get(s, "price"))
            if price is None or price <= 0:
                raise ValueError(f"Step {i}: a price step needs a price.")
            if price < floor:
                raise ValueError(f"Step {i}: price {price:g} is below the floor {floor:g}.")
            if price > lp:
                raise ValueError(f"Step {i}: price {price:g} is above the list price {lp:g}.")
            if price >= last_price:
                raise ValueError(f"Step {i}: price {price:g} must be lower than the step before it ({last_price:g}).")
            last_price = price
        else:
            pid = str(_get(s, "id", "") or "")
            if not pid:
                raise ValueError(f"Step {i}: a perk needs an id.")
            if pid in seen:
                raise ValueError(f"Step {i}: perk id {pid!r} is used twice.")
            seen.add(pid)
            text = _get(s, "text", {}) or {}
            missing = [l for l in PERK_LANGS if not str(text.get(l, "")).strip()]
            if missing:
                raise ValueError(f"Step {i}: perk {pid!r} needs wording in {', '.join(missing)} (the agent speaks all three).")


def perk_count(eco: Any) -> int:
    return sum(1 for s in (_get(eco, "ladder", []) or []) if _get(s, "kind", "perk") == "perk")


def validate_max_perks(eco: Any, max_perks: Any) -> int:
    """Perks the agent may give in ONE sale (the ladder may list more; the engine picks them in order)."""
    try:
        n = int(max_perks)
    except (TypeError, ValueError):
        raise ValueError("Max perks per sale must be a whole number.")
    if n < 0 or n > 5:
        raise ValueError("Max perks per sale must be between 0 and 5.")
    return n


def extras_path(data_dir: Any):
    from pathlib import Path
    return Path(data_dir) / "economics_extra.json"


def load_extras(data_dir: Any) -> dict:
    p = extras_path(data_dir)
    return _json(p.read_text(), {}) if p.exists() else {}


def save_extra(data_dir: Any, product_id: str, values: dict) -> None:
    ex = load_extras(data_dir)
    ex[product_id] = values
    extras_path(data_dir).write_text(json.dumps(ex, indent=1, sort_keys=True))
