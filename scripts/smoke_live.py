#!/usr/bin/env python
"""Live smoke test for a deployed SalesGod server (run ON the server, after a deploy).

    sudo -u salesgod /opt/salesgod/venv/bin/python scripts/smoke_live.py [--base http://127.0.0.1:8080] [--keep]

Reads ADMIN_PASSWORD from the environment (or /opt/salesgod/.env, parsed like systemd; values are never printed).
Creates a throwaway project "smoke-<ts>", exercises every feature end to end, never dials (no campaign call/tick),
and deletes what it created unless --keep. Exit code 1 if any check fails.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import httpx

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((name, bool(ok), detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}{('  -- ' + detail) if detail and not ok else ''}", flush=True)
    return bool(ok)


def load_env(path: str = "/opt/salesgod/.env") -> None:
    if os.getenv("ADMIN_PASSWORD") or not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
            v = v[1:-1]
        os.environ.setdefault(k.strip(), v)


class Client:
    def __init__(self, base: str):
        self.base = base.rstrip("/")
        self.c = httpx.Client(timeout=120)
        self.cookie = ""

    def login(self) -> bool:
        r = self.c.post(self.base + "/api/login", json={"password": os.getenv("ADMIN_PASSWORD", "")})
        sc = r.headers.get("set-cookie", "")
        self.cookie = sc.split(";", 1)[0] if sc else ""  # the cookie is Secure; send it by hand over local http
        return r.status_code == 200 and bool(self.cookie)

    def req(self, method: str, path: str, **kw):
        h = kw.pop("headers", {})
        if self.cookie:
            h["Cookie"] = self.cookie
        return self.c.request(method, self.base + path, headers=h, **kw)


def j(r):
    try:
        return r.json()
    except Exception:
        return {}


def sim(c: Client, campaign: str, turns: list[str], language: str = "hi", call_type: str | None = None) -> tuple[bool, list[dict]]:
    body = {"campaign": campaign, "language": language, "customer_name": "Smoke"}
    if call_type:
        body["call_type"] = call_type
    r = c.req("POST", "/api/sim/start", json=body)
    if r.status_code != 200:
        return False, [{"error": f"start {r.status_code}: {r.text[:120]}"}]
    sid, out = j(r)["sim_id"], [{"agent": j(r).get("agent")}]
    for t in turns:
        r = c.req("POST", f"/api/sim/{sid}", json={"text": t})
        if r.status_code != 200:
            return False, out + [{"error": f"turn {r.status_code}: {r.text[:120]}"}]
        d = j(r)
        out.append({"caller": t, "agent": d.get("agent"), "intent": d.get("intent"), "ms": d.get("latency_ms")})
        if d.get("end_call"):
            break
    return all(o.get("agent") for o in out), out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8080")
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    load_env()
    c = Client(a.base)

    r = c.req("GET", "/")
    check("console page", r.status_code == 200 and "SalesGod" in r.text)
    check("classic page", c.req("GET", "/classic").status_code == 200)
    check("productgen page", c.req("GET", "/productgen").status_code == 200)
    if not check("admin login", c.login()):
        return report()

    prods = j(c.req("GET", "/api/products")).get("products", [])
    pids = [p["id"] for p in prods]
    check("products listed", bool(pids), str(pids))
    gym = next((p for p in prods if "fitzone" in p["id"]), prods[0] if prods else None)

    # --- conversation core (text simulator) ---------------------------------------------------------------
    if gym:
        ok, t = sim(c, gym["id"], ["हाँ बोलिए", "ये कितने का है?", "बहुत महंगा है", "2000 में करो", "ठीक है 2500 में दो तो ले लूँगा"])
        check("sim: sales pitch + haggle (gym)", ok, json.dumps(t, ensure_ascii=False)[:300])
        agent_text = " ".join(x.get("agent") or "" for x in t)
        check("sim: no Claude text in gym call", not any(w in agent_text for w in ("Claude", "Anthropic", "क्लॉड")))
        ok, t = sim(c, gym["id"], ["मेरा गोल है छत से कूदना"])
        check("sim: crisis → helpline", ok and any("14416" in (x.get("agent") or "") for x in t), json.dumps(t, ensure_ascii=False)[:200])
        ok, t = sim(c, gym["id"], ["मैं शाम को ऑफ़िस से आकर जाऊँगा"])
        check("sim: no false crisis", ok and not any("14416" in (x.get("agent") or "") for x in t))
        for ct in ("lead_qualification", "follow_up", "appointment_booking"):
            ok, t = sim(c, gym["id"], ["हाँ बताइए", "मुझे muscle बनाना है", "बजट दस हज़ार है"], call_type=ct)
            check(f"sim: call type {ct}", ok, json.dumps(t, ensure_ascii=False)[:200])

    # --- platform: projects, customers, call types, campaigns, follow-ups ------------------------------------
    ts = str(int(time.time()))
    pr = c.req("POST", "/api/projects", json={"name": f"smoke-{ts}", "id": f"smoke-{ts}", "languages": ["hi", "en"],
                                              "timezone": "Asia/Kolkata", "call_hours": [10, 20]})
    pid = (j(pr).get("project") or {}).get("id")
    if check("create project", pr.status_code in (200, 201) and bool(pid), pr.text[:160]):
        check("custom field", c.req("POST", f"/api/projects/{pid}/fields", json={"key": "goal", "label": "Goal", "type": "text"}).status_code in (200, 201))
        cts = j(c.req("GET", f"/api/calltypes?project={pid}"))
        cts = cts.get("calltypes", cts) if isinstance(cts, dict) else cts
        check("call types listed", isinstance(cts, list) and len(cts) >= 6, str(type(cts)))
        dup = c.req("POST", f"/api/calltypes/lead_qualification/duplicate?project={pid}", json={"name": "Smoke qual"})
        check("duplicate call type", dup.status_code in (200, 201), dup.text[:160])
        csv = "name,phone,city\nSmoke One,+919000000001,Delhi\nSmoke Two,9000000002,Noida\nBad,123,X\n"
        prev = c.req("POST", "/api/customers/import", json={"project": pid, "csv": csv, "commit": False})
        check("CSV import preview", prev.status_code == 200 and j(prev).get("counts", {}).get("new") == 2, prev.text[:200])
        com = c.req("POST", "/api/customers/import", json={"project": pid, "csv": csv, "commit": True})
        check("CSV import commit", com.status_code == 200 and j(com).get("committed"), com.text[:200])
        lst = j(c.req("GET", f"/api/customers?project={pid}"))
        custs = lst.get("customers", [])
        check("customers listed", len(custs) == 2, str(len(custs)))
        if custs:
            cid = custs[0]["id"]
            check("customer timeline", c.req("GET", f"/api/customers/{cid}/timeline").status_code == 200)
            check("customer note", c.req("POST", f"/api/customers/{cid}/notes", json={"text": "smoke"}).status_code in (200, 201))
            check("stage change", c.req("PUT", f"/api/customers/{cid}", json={"stage": "interested"}).status_code == 200)
        tok = (j(pr).get("project") or {}).get("webhook_token", "")
        wh = c.req("POST", f"/api/leads/webhook/{pid}", headers={"X-Webhook-Token": tok}, json={"name": "Web Lead", "phone": "+919000000003"})
        check("lead webhook", wh.status_code in (200, 201) and j(wh).get("ok"), wh.text[:160])
        if gym:
            cp = c.req("POST", "/api/campaigns", json={"project": pid, "name": "smoke", "product_id": gym["id"],
                                                       "call_type_id": "sales_pitch", "dial_mode": "manual"})
            cmp_id = (j(cp).get("campaign") or {}).get("id")
            if check("create campaign", cp.status_code in (200, 201) and bool(cmp_id), cp.text[:160]):
                check("campaign queue", c.req("GET", f"/api/campaigns/{cmp_id}/queue").status_code == 200)
                if not a.keep:
                    c.req("DELETE", f"/api/campaigns/{cmp_id}")
        check("follow-ups list", c.req("GET", f"/api/followups?project={pid}&due=all").status_code == 200)
        if not a.keep:
            for cu in j(c.req("GET", f"/api/customers?project={pid}")).get("customers", []):
                c.req("DELETE", f"/api/customers/{cu['id']}")
            c.req("DELETE", f"/api/projects/{pid}")

    # --- product generation, AI assist, calls/map --------------------------------------------------------
    bc = c.req("POST", "/api/productgen/brand-check", json={"name": "Claude Pro"})
    check("productgen brand check", bc.status_code == 200, bc.text[:160])
    aa = c.req("POST", "/api/ai/assist", json={"kind": "step", "prompt": "ask how many days a week they can train", "context": {}})
    check("AI assist draft (Gemini)", aa.status_code == 200 and bool(j(aa).get("draft")), aa.text[:200])
    calls = j(c.req("GET", "/api/calls"))
    calls = calls.get("calls", calls) if isinstance(calls, dict) else calls
    if isinstance(calls, list) and calls:
        cid = calls[0].get("call_id")
        check("call map endpoint", c.req("GET", f"/api/calls/{cid}/map").status_code == 200)
    return report()


def report() -> int:
    bad = [n for n, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(bad)}/{len(RESULTS)} checks passed" + (f"; FAILED: {', '.join(bad)}" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
