"""Studio tests: AI customers talk to the DRAFT through the same pipeline as /api/sim (in process, no audio), then code checks
and a short Flash verdict grade every situation the call reached. Results warn, never block Publish (owner decision).

Code checks per agent turn: every number said is a confirmed fact or a DealState number; a link is only sent after a consent turn.
Runner (injected by the server): `start(product, nodes, language) -> (handle, opening)`, `turn(handle, text) -> dict(agent, intent,
pb_node, end_call, deal_numbers)`. Ask (injected, default Gemini via productgen settings): `ask(system, payload) -> dict`."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any, Awaitable, Callable

import httpx

from .studio import allowed_numbers, find_node, numbers, walk

Ask = Callable[[str, dict], Awaitable[dict]]
CONSENT_INTENTS = {"confirm_close", "confirm_buy"}
LINK_INTENTS = {"link_sent"}


def default_ask() -> Ask:
    from .llm import OpenAICompatLLM, parse_json
    llm = OpenAICompatLLM(retries=1, timeout_s=60)

    async def ask(system: str, payload: dict) -> dict:
        body: dict[str, Any] = {"model": llm.model, "temperature": 0.7, "max_tokens": 2000, "response_format": {"type": "json_object"},
                                "messages": [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]}
        body.update(llm.extra)
        last = ""
        for attempt in range(2):
            try:
                async with httpx.AsyncClient(timeout=llm.timeout) as c:
                    r = await c.post(f"{llm.base}/chat/completions", json=body, headers={"Authorization": f"Bearer {llm._key}"})
                r.raise_for_status()
                return parse_json(r.json()["choices"][0]["message"]["content"])
            except Exception as e:  # noqa: BLE001 - never leak the key: only the type and status
                last = type(e).__name__ + (f" {e.response.status_code}" if isinstance(e, httpx.HTTPStatusError) else "")
                await asyncio.sleep(1 + attempt)
        raise RuntimeError("model call failed: " + last)
    return ask


PERSONA_SYS = """You design test customers for an Indian phone sales agent. Return JSON {"personas": [...]} with exactly N personas,
each {"id": kebab, "name": first name, "language": "hinglish"|"hi"|"en", "brief": one sentence on who they are and how they behave,
"hidden_objection": str, "target_nodes": [1-3 node ids from the list they will most likely trigger]}. Mix: one easy buyer, one price
haggler, one sceptic who doubts it is genuine, one busy/indifferent person. Use only node ids from the list."""

CUSTOMER_SYS = """You play a customer on a phone call from a sales agent in India. Stay in character (persona below). Speak like a real
person on the phone: one short sentence, in the persona's language (hinglish = Hindi in Roman letters). Push your hidden objection at
least once. Return JSON {"say": str, "done": bool} - done=true only when you would hang up (agreed, refused firmly, or said bye)."""

JUDGE_SYS = """You grade a sales call transcript between an AI agent and a simulated customer. For each listed situation node the call
reached, say whether the agent handled it well: answered the actual question, stayed truthful to the facts, moved the call forward,
was polite. Return JSON {"overall": "pass"|"fail", "reason": one sentence, "nodes": {"<node id>": {"verdict": "pass"|"fail", "reason": short}}}."""

FIX_SYS = """You fix ONE situation node of a phone sales agent's decision tree after it failed a test. Keep the same id and JSON shape
{"id","title","when":{"keywords","intents","objections","topics"},"points":[3-5],"ask","goal","points_hi":[3-5],"ask_hi"}. Use only
the facts given (never invent prices, numbers or guarantees), fix what the failure shows, keep Hindi in Devanagari. Return the node JSON only."""


async def make_personas(ask: Ask, draft: dict, n: int = 4) -> list[dict]:
    nodes = [{"id": x["id"], "title": x.get("title", "")} for x in walk((draft.get("tree") or {}).get("nodes") or [])]
    r = await ask(PERSONA_SYS.replace("N personas", f"{n} personas"), {"product": draft.get("name"), "one_liner": (draft.get("product") or {}).get("one_liner", ""),
                                                                      "nodes": nodes})
    ids = {x["id"] for x in nodes}
    out = []
    for i, p in enumerate((r.get("personas") or [])[:n]):
        if not isinstance(p, dict):
            continue
        p["id"] = str(p.get("id") or f"p{i + 1}")[:30]
        p["language"] = p.get("language") if p.get("language") in ("hi", "hinglish", "en") else "hinglish"
        p["target_nodes"] = [t for t in p.get("target_nodes") or [] if t in ids]
        out.append(p)
    return out


def check_turn(text: str, intent: str, ok_numbers: set[str], deal_numbers: set[str], consented: bool) -> list[str]:
    issues = []
    bad = sorted(x for x in numbers(text) if x not in ok_numbers and x not in deal_numbers and not (len(x) <= 2 and float(x) <= 12))
    if bad:
        issues.append("said numbers that are not confirmed facts: " + ", ".join(bad))
    if intent in LINK_INTENTS and not consented:
        issues.append("sent the link before the customer clearly agreed")
    return issues


async def run_persona(runner, ask: Ask, draft: dict, product: dict, persona: dict, max_turns: int = 6) -> dict:
    ok = allowed_numbers(draft.get("facts") or [])
    nodes = (draft.get("tree") or {}).get("nodes") or []
    handle, opening = await runner.start(product, nodes, persona.get("language", "hinglish"))
    turns = [{"role": "agent", "text": opening, "intent": "opening", "node": None, "issues": check_turn(opening, "opening", ok, set(), False)}]
    consented = False
    for _ in range(max_turns):
        c = await ask(CUSTOMER_SYS, {"persona": persona, "product": draft.get("name"),
                                     "transcript": [{"who": t["role"], "text": t["text"]} for t in turns[-10:]]})
        say = str(c.get("say") or "").strip()[:300]
        if not say:
            break
        turns.append({"role": "customer", "text": say})
        r = await runner.turn(handle, say)
        intent = r.get("intent") or ""
        iss = check_turn(r.get("agent", ""), intent, ok, set(r.get("deal_numbers") or []), consented)
        consented = consented or intent in CONSENT_INTENTS
        turns.append({"role": "agent", "text": r.get("agent", ""), "intent": intent, "node": r.get("pb_node"), "issues": iss})
        if r.get("end_call") or c.get("done"):
            break
    reached = list(dict.fromkeys(t["node"] for t in turns if t.get("node")))
    verdict: dict = {}
    try:
        verdict = await ask(JUDGE_SYS, {"facts": [{"label": f["label"], "value": f["value"]} for f in draft.get("facts") or [] if f.get("status") == "confirmed"][:40],
                                        "nodes": [{"id": i, "title": (find_node(nodes, i) or {}).get("title", "")} for i in reached],
                                        "transcript": [{"who": t["role"], "text": t["text"], "node": t.get("node")} for t in turns]})
    except Exception as e:  # noqa: BLE001 - code checks still stand without the verdict
        verdict = {"overall": "unknown", "reason": f"judge unavailable ({e})", "nodes": {}}
    per_node: dict[str, dict] = {}
    for nid in reached:
        code = [i for t in turns if t.get("node") == nid for i in t.get("issues", [])]
        jv = (verdict.get("nodes") or {}).get(nid) or {}
        passed = not code and jv.get("verdict", "pass") != "fail"
        per_node[nid] = {"verdict": "pass" if passed else "fail", "reasons": code + ([jv["reason"]] if jv.get("reason") and jv.get("verdict") == "fail" else [])}
    for nid in persona.get("target_nodes") or []:
        per_node.setdefault(nid, {"verdict": "not_reached", "reasons": ["the call never reached this situation"]})
    gen = [i for t in turns if not t.get("node") for i in t.get("issues", [])]
    return {"persona": persona, "turns": turns, "nodes": per_node, "general_issues": gen, "judge": {"overall": verdict.get("overall"), "reason": verdict.get("reason")},
            "passed": not gen and all(v["verdict"] != "fail" for v in per_node.values()) and verdict.get("overall") != "fail"}


async def run_tests(runner, ask: Ask, draft: dict, product: dict, n_personas: int = 4, max_turns: int = 6,
                    progress: Callable[[str], None] = lambda s: None) -> dict:
    personas = draft.get("personas") or []
    if len(personas) < n_personas:
        progress("writing test customers")
        personas = await make_personas(ask, draft, n_personas)
    progress(f"calling {len(personas)} test customers")

    async def one(p):
        try:
            return await run_persona(runner, ask, draft, product, p, max_turns)
        except Exception as e:  # noqa: BLE001
            return {"persona": p, "turns": [], "nodes": {}, "general_issues": [f"run failed: {type(e).__name__}: {str(e)[:120]}"], "judge": {}, "passed": False}
    results = await asyncio.gather(*(one(p) for p in personas[:n_personas]))
    grid: dict[str, dict[str, str]] = {}
    for r in results:
        for nid, v in r["nodes"].items():
            grid.setdefault(nid, {})[r["persona"]["id"]] = v["verdict"]
    ran = [r for r in results if r["turns"]]
    return {"run_id": uuid.uuid4().hex[:8], "created_at": time.time(), "personas": personas, "results": results, "grid": grid,
            "pass_rate": round(sum(r["passed"] for r in ran) / len(ran), 2) if ran else 0.0}


def _changes(a: dict, b: dict) -> dict:
    return {k: {"old": a.get(k), "new": b.get(k)} for k in sorted(set(a) | set(b)) if k not in ("children", "cites") and a.get(k) != b.get(k)}


async def fix_node(ask: Ask, draft: dict, node_id: str, failure: dict | None) -> dict:
    nodes = (draft.get("tree") or {}).get("nodes") or []
    n = find_node(nodes, node_id)
    if not n:
        raise KeyError("unknown node " + node_id)
    before = {k: v for k, v in n.items() if k != "children"}
    facts = [{"id": f["id"], "label": f["label"], "value": f["value"]} for f in draft.get("facts") or [] if f.get("status") == "confirmed"][:50]
    r = await ask(FIX_SYS, {"node": before, "failure": failure or {}, "facts": facts, "product": draft.get("name")})
    after = {**before, **{k: r[k] for k in ("title", "when", "points", "ask", "goal", "points_hi", "ask_hi") if k in r}, "id": node_id}
    from acie.playbook import INTENTS, OBJECTIONS, TOPICS, Node
    w = dict(after.get("when") or {})
    w["intents"] = [i for i in w.get("intents") or [] if i in INTENTS]
    w["objections"] = [o for o in w.get("objections") or [] if o in OBJECTIONS]
    w["topics"] = [t for t in w.get("topics") or [] if t in TOPICS]
    after["when"] = w
    Node.model_validate({**after, "children": []})
    return {"node_id": node_id, "before": before, "after": after, "changes": _changes(before, after)}
