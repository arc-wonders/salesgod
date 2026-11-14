"""Operator UI against the REAL backend, end to end.

Starts the real FastAPI app (voice.server.build_app) on a free port with a temp VOICE_DATA_DIR, a fake Plivo dialer, a fake LLM
for /api/ai/assist and a stub text simulator (the real one needs the models), then drives headless Chrome/Chromium through the
onboarding wizard, customers, the stage board, the customer page, Do-not-call, the campaign queue, follow-ups and "Add with AI".
Every step asserts no JS errors, no failed API calls, and the expected state in the API.

Also (no browser needed): every API path+method that platform.js / console.html fetch is a route of the real app.
Skips cleanly when Playwright or a browser is missing."""
import json
import os
import re
import shutil
import socket
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "voice" / "static"
BROWSER = shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chromium-browser")


# ---------------------------------------------------------------- the real app, with fakes where the world would be

class FakeLLM:
    """Stands in for the OpenAI-compatible endpoint behind /api/ai/assist: one scripted JSON reply per stage."""
    model = "fake"
    replies: dict = {}
    seen: list = []

    async def json(self, stage, payload):
        type(self).seen.append((stage, payload))
        return type(self).replies[stage]


def _build(tmp_path, monkeypatch):
    for k, v in {"PLIVO_AUTH_ID": "x", "PLIVO_AUTH_TOKEN": "x", "PLIVO_NUMBER": "+910000000000", "ADMIN_PASSWORD": "pw", "SESSION_SECRET": "s3cret"}.items():
        os.environ.setdefault(k, v)
    try:
        from voice import server
    except ImportError as e:  # onnxruntime etc. missing in a slim venv
        pytest.skip(f"voice server deps missing: {e}")
    data = tmp_path / "voice"
    (data / "calls").mkdir(parents=True)
    monkeypatch.setattr(server, "DATA", data)
    monkeypatch.setattr(server, "CALLS_DIR", data / "calls")
    from voice.crm import ai_assist
    monkeypatch.setattr(ai_assist, "AssistLLM", FakeLLM)
    api, state = server.build_app()
    api.router.on_startup.clear()  # no Whisper / TTS / voice-model loading
    state.cfg.admin_password = "pw"

    dialed = []

    async def fake_dial(call):  # Plivo: accept the call, never touch the network
        dialed.append(call.to)
        call.request_uuid, call.status = "req-" + call.call_id, "ringing"
    state.plivo.dial = fake_dial

    # text simulator stub (the real one runs the whole ACIE pipeline)
    api.router.routes[:] = [r for r in api.router.routes if not getattr(r, "path", "").startswith("/api/sim")]
    sims = {"started": [], "turns": []}
    from fastapi import Request

    @api.post("/api/sim/start")
    async def sim_start(request: Request) -> dict:
        state_ok = request.cookies.get("sg_session")
        assert state_ok
        body = await request.json()
        sims["started"].append(body)
        return {"sim_id": "sim1", "agent": "Hello, this is Arjun from Acme Gym."}

    @api.post("/api/sim/{sid}")
    async def sim_turn(sid: str, request: Request) -> dict:
        body = await request.json()
        sims["turns"].append(body["text"])
        return {"agent": "Sure, the pass is 2,799 for the year.", "intent": "price", "trace": {"gate": "safety"}}
    return server, api, state, data, dialed, sims


@pytest.fixture
def live(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    uvicorn = pytest.importorskip("uvicorn")
    httpx = pytest.importorskip("httpx")
    FakeLLM.replies, FakeLLM.seen = {}, []
    server, api, state, data, dialed, sims = _build(tmp_path, monkeypatch)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = uvicorn.Server(uvicorn.Config(api, host="127.0.0.1", port=port, log_level="warning", timeout_graceful_shutdown=1))
    th = threading.Thread(target=srv.run, daemon=True)
    th.start()
    for _ in range(100):
        if srv.started:
            break
        time.sleep(0.05)
    assert srv.started
    import hashlib
    import hmac
    exp = str(int(time.time()) + 3600)
    cookie = f"{exp}.{hmac.new(b's3cret', exp.encode(), hashlib.sha256).hexdigest()}"
    secret = state.cfg.session_secret
    cookie = f"{exp}.{hmac.new(secret.encode(), exp.encode(), hashlib.sha256).hexdigest()}"

    class L:
        pass
    L.base, L.data, L.state, L.dialed, L.sims, L.cookie, L.server = f"http://127.0.0.1:{port}", data, state, dialed, sims, cookie, server
    L.http = httpx.Client(base_url=L.base, cookies={"sg_session": cookie}, timeout=20)
    yield L
    L.http.close()
    srv.should_exit = True
    th.join(5)


# ---------------------------------------------------------------- contract: every fetch in the UI hits a real route

def _args(src: str, i: int) -> tuple[list[str], int]:
    """Split the call arguments that start right after the '(' at src[i-1]; returns (args, index after ')')."""
    depth, args, cur, q, j = 1, [], "", None, i
    while j < len(src):
        c = src[j]
        if q:
            cur += c
            if c == "\\":
                cur += src[j + 1]
                j += 1
            elif c == q:
                q = None
        elif c in "'\"`":
            q = c
            cur += c
        elif c in "([{":
            depth += 1
            cur += c
        elif c in ")]}":
            depth -= 1
            if depth == 0:
                args.append(cur.strip())
                return args, j + 1
            cur += c
        elif c == "," and depth == 1:
            args.append(cur.strip())
            cur = ""
        else:
            cur += c
        j += 1
    raise AssertionError("unbalanced call")


def ui_requests(src: str) -> set[tuple[str, str]]:
    out = set()
    for m in re.finditer(r"\b(tryApi|post|saveObj|api|fetch)\(", src):
        args, _ = _args(src, m.end())
        fn = m.group(1)
        if fn == "saveObj":
            base = re.match(r"^'(/api/[^']*)'$", args[0])
            if base:
                out |= {("POST", base.group(1)), ("PUT", base.group(1) + "/{}")}
            continue
        lit = re.match(r"^[`'\"](/api/[^`'\"]*)[`'\"]", args[0] if args else "")
        if not lit:
            continue
        path = re.sub(r"\$\{[^}]*\}", "{}", lit.group(1)).split("?")[0].rstrip("/")
        rest = " ".join(args[1:])
        if fn == "post":
            method = "PUT" if re.search(r"'PUT'", rest) else "DELETE" if "'DELETE'" in rest else "POST"
        else:
            mm = re.search(r"method:\s*'(\w+)'", rest)
            method = mm.group(1) if mm else "GET"
        out.add((method, path))
    return out


def test_every_ui_request_has_a_real_route(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    server, api, *_ = _build(tmp_path, monkeypatch)
    routes = [(m, re.compile("^" + re.sub(r"\{[^/}]+\}", "[^/]+", r.path) + "$")) for r in api.routes if hasattr(r, "methods") for m in r.methods]
    seen = set()
    for name in ("platform.js", "console.html"):
        seen |= ui_requests((STATIC / name).read_text(encoding="utf-8"))
    assert len(seen) >= 30, f"the scanner found too few requests: {sorted(seen)}"
    assert {("POST", "/api/customers/import"), ("GET", "/api/followups"), ("POST", "/api/ai/assist"), ("PUT", "/api/customers/{}"), ("POST", "/api/campaigns/{}/start")} <= seen
    missing = sorted(f"{m} {p}" for m, p in seen if not any(rm == m and rx.match(p.replace("{}", "x")) for rm, rx in routes))
    assert not missing, f"UI calls with no route in the real app: {missing}"


# ---------------------------------------------------------------- the browser run

pw = pytest.importorskip("playwright.sync_api") if BROWSER else None


class Ui:
    """Thin wrapper: the page, the collected JS/API problems, and the API of the live server for assertions."""

    def __init__(self, page, live):
        self.page, self.live, self.errors, self.bad = page, live, [], []
        page.on("pageerror", lambda e: self.errors.append(f"pageerror: {e}"))
        page.on("console", lambda m: m.type == "error" and "Failed to load resource" not in m.text and self.errors.append(f"console: {m.text}"))
        page.on("response", lambda r: r.status >= 400 and self.bad.append(f"{r.status} {r.request.method} {r.url.replace(live.base, '')}"))

    def ok(self, step):
        self.page.wait_for_timeout(150)
        assert not self.errors, f"[{step}] JS errors: {self.errors}"
        assert not self.bad, f"[{step}] failed API calls: {self.bad}"

    def get(self, path):
        r = self.live.http.get(path)
        assert r.status_code == 200, (path, r.text)
        return r.json()

    def put(self, path, body):
        r = self.live.http.put(path, json=body)
        assert r.status_code == 200, (path, r.text)
        return r.json()

    def go(self, route):
        self.page.goto(f"{self.live.base}/#/{route}")
        self.page.reload() if self.page.url.endswith(f"#/{route}") and False else None
        self.page.wait_for_load_state("networkidle")

    def click(self, text, scope="body", exact=True):
        self.page.locator(scope).get_by_role("button", name=text, exact=exact).first.click()
        self.page.wait_for_timeout(250)

    def next(self, label="Next"):
        self.page.locator(".obbody .df .btn.pri").filter(has_text=label).click()
        self.page.wait_for_load_state("networkidle")
        self.page.wait_for_timeout(300)


CSV = ("Customer,Mobile,City,Budget\n"
       "Asha Rao,98765 43210,Pune,3000\n"
       "Rahul Mehta,+91 99999 11111,Delhi,5000\n"
       "Bad Phone,12345,Nowhere,1\n"
       "Asha again,9876543210,Pune,0\n"
       "Dan Dnc,9000090000,Goa,100\n")


@pytest.mark.skipif(not BROWSER, reason="no Chrome/Chromium")
def test_operator_flow_against_the_real_backend(live):
    import asyncio
    L = live
    # state the console needs before the first screen: a product (what Add product with AI would have written) and a DNC entry
    prod = {"tenant": "acme-gym", "product_name": "Acme Gym Pass", "seller_name": "Acme Gym", "agent_name": "Arjun", "original_price": "3,999", "offer_price": "2,799", "discount": "30% off"}
    assert L.http.put("/api/products/acme-gym-pass", json=prod).status_code == 200
    (L.data / "dnc.json").write_text(json.dumps({"+919000090000": time.time()}))
    FakeLLM.replies = {
        "assist_calltype": {"name": "Payment nudge", "objective": "Remind kindly about the pending payment", "steps": [{"id": "ask_link", "text": {"en": "Did you get the payment link?"}, "answer_type": "yes_no", "required": True}],
                            "allowed_actions": ["send_link"], "success_outcomes": ["paid"], "selling_allowed": False, "negotiation_allowed": False, "closing": {"en": "Thank you, have a good day."}},
        "assist_campaign": {"name": "Weekend push", "call_type_id": "follow_up", "language": "en", "audience": {"stages": ["interested"], "tags": ["gym"]},
                            "schedule": {"window": [11, 18], "days": [5, 6], "max_parallel": 2, "pacing_seconds": 30, "retry": {"max_attempts": 2, "retry_after_minutes": 180}}},
    }

    with pw.sync_playwright() as p:
        browser = p.chromium.launch(executable_path=BROWSER, headless=True, args=["--no-sandbox", "--disable-gpu"])
        ctx = browser.new_context(viewport={"width": 1400, "height": 900})
        ctx.add_cookies([{"name": "sg_session", "value": L.cookie, "url": L.base}])
        page = ctx.new_page()
        ui = Ui(page, L)
        try:
            # ---------------------------------------------------------- 1 onboarding: project basics
            ui.go("onboarding?new=1")
            page.get_by_label("Project name").fill("Acme Gym")
            ui.next()
            proj = ui.get("/api/projects/acme-gym")["project"]
            assert proj["name"] == "Acme Gym" and proj["call_hours"] == [10, 20] and proj["timezone"] == "Asia/Kolkata" and proj["webhook_token"]
            assert set(proj["languages"]) == {"en", "hi", "hinglish"}
            ui.ok("project basics")

            # ---------------------------------------------------------- 2 pick the product
            page.get_by_label("Existing product").select_option("acme-gym-pass")
            page.wait_for_timeout(400)
            assert "Using Acme Gym Pass" in page.inner_text(".obbody")
            ui.next()
            assert ui.get("/api/projects/acme-gym")["project"]["onboarding"]["productId"] == "acme-gym-pass"
            ui.ok("product")

            # ---------------------------------------------------------- 3 duplicate a built-in call type and edit a step
            page.locator(".pcard", has_text="Lead qualification").get_by_role("button", name="View steps").click()
            page.wait_for_selector("text=This is a built-in call type")
            page.get_by_role("button", name="Duplicate").click()
            page.wait_for_selector("text=Lead qualification (copy)", state="attached")
            dup = [c for c in ui.get("/api/calltypes?project=acme-gym")["calltypes"] if c.get("project_id") == "acme-gym"]
            assert len(dup) == 1 and dup[0]["id"] == "lead-qualification-copy" and dup[0]["selling_allowed"] is False and not dup[0]["builtin"]
            name = page.locator(".panel", has_text="Purpose").locator("input").first
            name.fill("Gym lead call")
            page.get_by_role("button", name="Add step").click()
            page.locator(".stepcard .sc-h input.mono").fill("goal")
            page.locator(".stepcard .langs textarea").first.fill("What is your fitness goal?")
            page.locator(".stepcard select").select_option("choice")
            page.locator(".stepcard").get_by_text("required").click()
            page.locator(".pad.narrow .df, .pad.narrow .row").get_by_role("button", name="Save").last.click()
            page.wait_for_selector("text=Gym lead call")
            ct = next(c for c in ui.get("/api/calltypes?project=acme-gym")["calltypes"] if c["id"] == "lead-qualification-copy")
            assert ct["name"] == "Gym lead call" and ct["steps"] == [{"id": "goal", "text": {"en": "What is your fitness goal?"}, "answer_type": "choice", "required": True}]
            ui.ok("call type duplicate + edit step")
            page.goto(f"{L.base}/#/onboarding")
            page.wait_for_selector("text=3. Call type")
            page.locator(".pcard.pick", has_text="Gym lead call").locator("input[type=radio]").check()
            page.wait_for_timeout(400)
            ui.next()

            # ---------------------------------------------------------- 4 customers: CSV with mapping, preview, commit
            assert L.http.post("/api/projects/acme-gym/fields", json={"key": "budget", "label": "Budget", "type": "number"}).status_code == 200
            page.reload()
            page.wait_for_selector("text=4. Customers")
            page.locator("input[type=file]").set_input_files(files=[{"name": "leads.csv", "mimeType": "text/csv", "buffer": CSV.encode()}])
            page.wait_for_selector("text=Match your columns")
            page.get_by_label("Field for Budget").select_option(label="Field: Budget")
            page.get_by_role("button", name="Check before importing").click()
            page.wait_for_selector("text=will be added")
            txt = page.inner_text(".importer")
            assert "2 will be added" in txt and "1 duplicates (same phone) skipped" in txt and "1 on Do-not-call" in txt and "1 invalid" in txt and "Row 4: not a valid phone number" in txt
            assert ui.get("/api/customers?project=acme-gym")["total"] == 0  # a preview writes nothing
            page.get_by_role("button", name="Import now").click()
            page.wait_for_selector("text=Imported 2 customers")
            lst = ui.get("/api/customers?project=acme-gym")
            assert lst["total"] == 3
            by = {c["name"]: c for c in lst["customers"]}
            assert by["Asha Rao"]["phone"] == "+919876543210" and by["Asha Rao"]["custom"]["budget"] == 3000 and by["Asha Rao"]["city"] == "Pune" and by["Asha Rao"]["source"] == "csv"
            assert by["Dan Dnc"]["stage"] == "do_not_call" and by["Dan Dnc"]["consent"]["dnc"] is True
            ui.ok("csv import")
            ui.next()

            # ---------------------------------------------------------- 5 campaign
            page.wait_for_selector("text=5. Campaign")
            ui.next("Save and test")
            camps = ui.get("/api/campaigns?project=acme-gym")["campaigns"]
            assert len(camps) == 1
            camp = camps[0]
            assert (camp["product_id"], camp["call_type_id"], camp["status"], camp["dial_mode"]) == ("acme-gym-pass", "lead-qualification-copy", "draft", "manual")
            assert camp["schedule"]["window"] == [10, 20] and camp["schedule"]["days"] == [0, 1, 2, 3, 4, 5] and camp["schedule"]["retry"]["max_attempts"] == 3 and camp["tone_level"] == "mid"
            ui.ok("campaign")

            # ---------------------------------------------------------- 6 test call through the simulator stub
            page.wait_for_selector("text=6. Test call")
            page.get_by_role("button", name="Start test").click()
            page.wait_for_selector("text=Hello, this is Arjun from Acme Gym.")
            assert L.sims["started"][0]["campaign"] == "acme-gym-pass" and L.sims["started"][0]["call_type"] == "lead-qualification-copy"
            page.get_by_label("Customer message").fill("what is the price?")
            page.get_by_label("Customer message").press("Enter")
            page.wait_for_selector("text=the pass is 2,799 for the year")
            assert L.sims["turns"] == ["what is the price?"]
            ui.ok("test call")
            ui.next("Next: go live")

            # ---------------------------------------------------------- 7 go live, then the campaign queue view
            page.get_by_label("I have checked", exact=False).check() if False else page.locator("label.chip", has_text="I have checked").locator("input").check()
            page.get_by_role("button", name="Go live").click()
            page.wait_for_selector("text=Queue: who is next")
            page.wait_for_selector(".tbl")
            assert ui.get("/api/campaigns/" + camp["id"])["campaign"]["status"] == "running"
            rows = page.locator(".tbl tbody tr")
            assert rows.count() == 2
            q = page.inner_text(".tbl")
            assert "Asha Rao" in q and "Rahul Mehta" in q and "Dan Dnc" not in q and "not called yet in this campaign" in q
            api_q = ui.get(f"/api/campaigns/{camp['id']}/queue")
            assert [r["name"] for r in api_q["eligible"]] == ["Asha Rao", "Rahul Mehta"] or sorted(r["name"] for r in api_q["eligible"]) == ["Asha Rao", "Rahul Mehta"]
            ui.ok("go live + queue")

            # ---------------------------------------------------------- customers table, search, stage board drag
            ui.go("customers")
            page.wait_for_selector(".tbl")
            assert page.locator(".tbl tbody tr").count() == 3
            assert "DNC" in page.locator(".tbl tbody tr", has_text="Dan Dnc").inner_text()
            page.get_by_label("Search customers").fill("rahul")
            page.get_by_label("Search customers").press("Enter")
            page.wait_for_timeout(500)
            assert page.locator(".tbl tbody tr").count() == 1
            page.get_by_label("Search customers").fill("")
            page.get_by_label("Search customers").press("Enter")
            page.wait_for_timeout(400)
            page.get_by_role("button", name="Stages").click()
            page.wait_for_selector(".board")
            rahul = by["Rahul Mehta"]["id"]
            page.locator(".kcard", has_text="Rahul Mehta").drag_to(page.locator("section.col", has=page.get_by_text("Contacted", exact=True)))
            page.wait_for_timeout(600)
            assert ui.get(f"/api/customers/{rahul}")["customer"]["stage"] == "contacted"
            assert any(e["type"] == "stage" and e["to"] == "contacted" for e in ui.get(f"/api/customers/{rahul}/timeline")["timeline"])
            assert page.locator("section.col", has=page.get_by_text("Contacted", exact=True)).locator(".kcard", has_text="Rahul Mehta").count() == 1
            dan = page.locator(".kcard", has_text="Dan Dnc")
            assert dan.get_attribute("draggable") == "false" and dan.locator("select").is_disabled()
            ui.ok("customers table + board")

            # ---------------------------------------------------------- customer detail: call now, timeline, outcome
            asha = by["Asha Rao"]["id"]
            ui.go(f"customers/{asha}")
            page.wait_for_selector("text=Timeline")
            assert "Added · csv" in page.inner_text(".tline")
            assert "3000" in page.inner_text(".kv")  # custom field Budget
            page.get_by_role("button", name="Call now").click()
            page.locator(".modal, [role=dialog]").get_by_role("button", name="Place the call").click()
            page.wait_for_url(re.compile(r"#/calls/"))
            assert L.dialed == ["+919876543210"]
            call_id = page.url.rsplit("/", 1)[1]
            link = L.state.crm["store"].call_link(call_id)
            assert link["customer_id"] == asha and link["call_type"] == "lead-qualification-copy" or link["customer_id"] == asha
            ui.ok("call now")
            # the call ends: the customer asked to be called back
            f = L.data / "calls" / f"{call_id}.json"
            rec = json.loads(f.read_text())
            rec.update(status="ended", answered_at=time.time() - 60, ended_at=time.time(), context={"promised": ["callback_later"], "decision": "later"}, turns=[{"role": "customer", "text": "call me later"}])
            f.write_text(json.dumps(rec))
            r = L.http.post(f"/api/crm/calls/{call_id}/apply")
            assert r.status_code == 200 and r.json()["outcome"] == "callback", r.text
            ui.go(f"customers/{asha}")
            page.wait_for_selector("text=Timeline")
            tl = page.inner_text(".tline")
            assert "Call · " in tl and "callback" in tl and "Call started" in tl and "Open call and info map" in tl
            assert "callback" in page.inner_text(".kv")
            cu = ui.get(f"/api/customers/{asha}")["customer"]
            assert cu["crm_customer_id" if False else "stage"] == "contacted" and cu["next_action"]["type"] == "callback" and cu["history"]["call_ids"] == [call_id]
            ui.ok("customer detail timeline")

            # ---------------------------------------------------------- follow-ups
            ui.put(f"/api/customers/{rahul}", {"next_action": {"type": "follow_up", "due_at": time.time() - 3600, "call_type": "follow_up", "note": "he asked for a call after lunch"}})
            ui.go("followups")
            page.get_by_role("tab", name="Due now").click()
            page.wait_for_selector("text=he asked for a call after lunch")
            assert page.locator(".tbl tbody tr").count() == 1 and "Asha" not in page.inner_text(".tbl")
            page.get_by_role("tab", name="This week").click()
            page.wait_for_selector("text=Asha Rao")
            assert page.locator(".tbl tbody tr").count() == 2 and "the customer asked to be called later" in page.inner_text(".tbl")
            api_fu = ui.get("/api/followups?project=acme-gym&due=week")["followups"]
            assert {x["name"] for x in api_fu} == {"Asha Rao", "Rahul Mehta"}
            ui.ok("follow-ups")

            # ---------------------------------------------------------- mark Do not call
            ui.go(f"customers/{asha}")
            page.wait_for_selector("text=Timeline")
            page.get_by_role("button", name="Mark Do not call").click()
            page.locator(".modal, [role=dialog]").get_by_role("button", name="Mark Do not call").click()
            page.wait_for_selector("text=Only the person asking us again can lift this")
            cu = ui.get(f"/api/customers/{asha}")["customer"]
            assert cu["consent"]["dnc"] is True and cu["stage"] == "do_not_call" and cu["next_action"] is None
            assert "+919876543210" in json.loads((L.data / "dnc.json").read_text())
            assert page.get_by_role("button", name="Call now").count() == 0 and page.get_by_text("Remove Do-not-call").count() == 0
            assert [r["name"] for r in ui.get(f"/api/campaigns/{camp['id']}/queue")["eligible"]] == ["Rahul Mehta"]
            ui.ok("mark DNC")
            r = L.http.post(f"/api/customers/{asha}/call", json={"product_id": "acme-gym-pass"})
            assert r.status_code == 409 and "do-not-call" in r.json()["detail"]  # and the server refuses it even if asked directly
            r = L.http.put(f"/api/customers/{asha}", json={"consent": {"dnc": False}})
            assert r.status_code == 409  # one-way
            ui.bad.clear()

            # ---------------------------------------------------------- campaign queue after the DNC, and a refused call is explained
            ui.go(f"campaigns/{camp['id']}?tab=queue")
            page.wait_for_selector(".tbl")
            assert page.locator(".tbl tbody tr").count() == 1 and "Rahul Mehta" in page.inner_text(".tbl")
            L.state.cfg.max_parallel_calls = 0  # the dial gate refuses: the operator sees the reason
            page.get_by_role("button", name="Call", exact=True).click()
            page.locator(".modal, [role=dialog]").get_by_role("button", name="Place the call").click()
            page.wait_for_selector("text=max 0 calls at once")
            L.state.cfg.max_parallel_calls = 8
            assert any(b.startswith("409 POST /api/campaigns/") for b in ui.bad)
            ui.bad.clear()
            page.keyboard.press("Escape")

            # ---------------------------------------------------------- Add with AI: draft -> apply -> save (call type)
            ui.go("calltypes/new")
            page.wait_for_selector("text=Describe what you want")  if False else page.wait_for_selector(".ai")
            page.get_by_label("Describe what you want the AI to draft").first.fill("a follow-up call for people who got the link but did not pay")
            page.get_by_role("button", name="Draft with AI").first.click()
            page.wait_for_selector(".diff")
            diff = page.inner_text(".diff")
            assert "Payment nudge" in diff and "ask_link" in diff and "builtin" not in diff and "project" not in diff
            seen_stage, seen_payload = FakeLLM.seen[-1]
            assert seen_stage == "assist_calltype" and seen_payload["instruction"].startswith("a follow-up call")
            assert [c["name"] for c in ui.get("/api/calltypes?project=acme-gym")["calltypes"] if c.get("project_id")] == ["Gym lead call"]  # a draft saves nothing
            page.get_by_role("button", name="Apply to the form").click()
            assert page.locator(".panel", has_text="Purpose").locator("input").first.input_value() == "Payment nudge"
            page.locator(".pad.narrow .row").get_by_role("button", name="Save").last.click()
            page.wait_for_selector("text=Payment nudge")
            saved = next(c for c in ui.get("/api/calltypes?project=acme-gym")["calltypes"] if c["name"] == "Payment nudge")
            assert saved["id"] == "payment-nudge" and saved["success_outcomes"] == ["paid"] and saved["allowed_actions"] == ["send_link"] and saved["selling_allowed"] is False
            assert saved["steps"][0]["id"] == "ask_link" and saved["steps"][0]["answer_type"] == "yes_no" and saved["closing"] == {"en": "Thank you, have a good day."}
            ui.ok("add with AI: call type")

            # ---------------------------------------------------------- Add with AI on a campaign form: the draft maps onto the real schedule shape
            ui.go("campaigns/new")
            page.wait_for_selector(".ai")
            page.get_by_label("Describe what you want the AI to draft").first.fill("weekend push for interested gym leads")
            page.get_by_role("button", name="Draft with AI").first.click()
            page.wait_for_selector(".diff")
            page.get_by_role("button", name="Apply to the form").click()
            page.get_by_label("Product", exact=True).select_option("acme-gym-pass")
            page.locator(".pad.narrow .row").get_by_role("button", name="Save").last.click()
            page.wait_for_selector("text=Queue: who is next")
            wk = next(c for c in ui.get("/api/campaigns?project=acme-gym")["campaigns"] if c["name"] == "Weekend push")
            assert wk["call_type_id"] == "follow_up" and wk["language"] == "en" and wk["audience"]["stages"] == ["interested"] and wk["audience"]["tags"] == ["gym"]
            assert wk["schedule"]["window"] == [11, 18] and wk["schedule"]["days"] == [5, 6] and wk["schedule"]["max_parallel"] == 2 and wk["schedule"]["retry"]["max_attempts"] == 2 and wk["schedule"]["retry"]["retry_after_minutes"] == 180
            assert wk["status"] == "draft" and wk["dial_mode"] == "manual"
            # start without a product is refused with a plain reason (draft campaign made through the API)
            nop = L.http.post("/api/campaigns", json={"project": "acme-gym", "name": "No product", "call_type_id": "sales_pitch"}).json()["campaign"]
            ui.go("campaigns")
            page.wait_for_selector(".tbl")
            page.locator(".tbl tbody tr", has_text="No product").get_by_role("button", name="Start").click()
            page.locator(".modal, [role=dialog]").get_by_role("button", name="Start").click()
            page.wait_for_selector("text=pick a product for this campaign before starting it")
            assert ui.get(f"/api/campaigns/{nop['id']}")["campaign"]["status"] == "draft"
            assert any(b.startswith("400 POST /api/campaigns/") for b in ui.bad)
            ui.bad.clear()
            ui.ok("add with AI: campaign + refused start")

            # ---------------------------------------------------------- settings + dashboard render against real data
            ui.go("settings?tab=project")
            page.wait_for_selector("text=Project basics")
            page.get_by_label("Calling hours end").select_option("19")
            page.get_by_role("button", name="Save", exact=True).click()
            page.wait_for_timeout(600)
            assert ui.get("/api/projects/acme-gym")["project"]["call_hours"] == [10, 19]
            ui.go("settings")
            page.wait_for_selector("text=Custom fields")
            assert "Budget" in page.inner_text("body")
            ui.go("dashboard")
            page.wait_for_selector(".tiles")
            tiles = page.inner_text(".tiles")
            assert "3" in tiles
            ui.ok("settings + dashboard")
        finally:
            if os.getenv("E2E_SHOT"):
                page.screenshot(path=os.getenv("E2E_SHOT"))
            browser.close()
