import json

from crm_helpers import H, Clock, make, project, customer
from voice.crm.models import Customer
from voice.crm.scoring import score
from voice.crm.store import JsonStore, atomic_write


def test_routes_need_admin(tmp_path):
    c, _, _ = make(tmp_path)
    for m, u in [("get", "/api/projects"), ("post", "/api/projects"), ("get", "/api/customers?project=x"), ("post", "/api/customers/import"),
                 ("get", "/api/campaigns"), ("get", "/api/followups?project=x"), ("post", "/api/ai/assist"), ("get", "/api/calltypes")]:
        assert getattr(c, m)(u).status_code == 401, u


def test_project_crud_and_tenants_sync(tmp_path):
    (tmp_path / "tenants.json").write_text(json.dumps([{"id": "anthropic", "name": "Anthropic", "color": "#d97757"}]))
    c, _, _ = make(tmp_path)
    ids = [p["id"] for p in c.get("/api/projects", headers=H).json()["projects"]]
    assert ids == ["anthropic"]  # derived from tenants.json
    p = project(c, "acme", color="#123456", timezone="Asia/Kolkata", call_hours=[9, 18])
    assert p["webhook_token"] and p["call_hours"] == [9, 18]
    assert {t["id"] for t in json.loads((tmp_path / "tenants.json").read_text())} == {"anthropic", "acme"}  # /api/tenants keeps working
    assert c.post("/api/projects", json={"name": "Acme", "id": "acme"}, headers=H).status_code == 409
    assert c.post("/api/projects", json={"name": "X", "id": "Bad Id"}, headers=H).status_code == 422
    assert c.post("/api/projects", json={"name": "X", "timezone": "Mars/Base"}, headers=H).status_code == 422
    r = c.put("/api/projects/acme", json={"name": "Acme 2", "webhook_token": "hack"}, headers=H).json()["project"]
    assert r["name"] == "Acme 2" and r["webhook_token"] == p["webhook_token"]
    customer(c)
    assert c.delete("/api/projects/acme", headers=H).status_code == 409  # has customers


def test_custom_fields_crud_and_validation(tmp_path):
    c, _, _ = make(tmp_path)
    project(c)
    assert c.post("/api/projects/acme/fields", json={"key": "budget", "type": "number"}, headers=H).status_code == 200
    assert c.post("/api/projects/acme/fields", json={"key": "plan", "type": "choice", "choices": ["gold", "silver"], "required": True}, headers=H).status_code == 200
    assert c.post("/api/projects/acme/fields", json={"key": "Bad Key"}, headers=H).status_code == 422
    assert c.post("/api/projects/acme/fields", json={"key": "budget"}, headers=H).status_code == 409
    r = c.post("/api/customers", json={"project_id": "acme", "phone": "9876543210"}, headers=H)
    assert r.status_code == 400 and "required" in r.text
    r = c.post("/api/customers", json={"project_id": "acme", "phone": "9876543210", "custom": {"plan": "bronze"}}, headers=H)
    assert r.status_code == 400
    r = c.post("/api/customers", json={"project_id": "acme", "phone": "9876543210", "custom": {"plan": "gold", "budget": "5000"}}, headers=H)
    assert r.json()["customer"]["custom"] == {"plan": "gold", "budget": 5000.0}
    assert c.post("/api/customers", json={"project_id": "acme", "phone": "9876543211", "custom": {"nope": 1, "plan": "gold"}}, headers=H).status_code == 400
    assert c.put("/api/projects/acme/fields/budget", json={"label": "Budget"}, headers=H).json()["field"]["label"] == "Budget"
    assert c.delete("/api/projects/acme/fields/budget", headers=H).status_code == 200
    assert [f["key"] for f in c.get("/api/projects/acme/fields", headers=H).json()["fields"]] == ["plan"]


def test_customer_crud_search_dedupe_dnc(tmp_path):
    c, out, _ = make(tmp_path)
    project(c)
    a = customer(c, phone="98765 43210", name="Asha Rao", city="Pune", tags=["Vip"])
    assert a["phone"] == "+919876543210" and a["tags"] == ["vip"] and a["stage"] == "new"
    assert c.post("/api/customers", json={"project_id": "acme", "phone": "+91 98765-43210"}, headers=H).status_code == 409
    assert c.post("/api/customers", json={"project_id": "acme", "phone": "12"}, headers=H).status_code == 400
    b = customer(c, phone="9123456780", name="Bala")
    r = c.get("/api/customers?project=acme&q=asha", headers=H).json()
    assert r["total"] == 1 and r["customers"][0]["id"] == a["id"] and "events" not in r["customers"][0]
    assert c.get("/api/customers?project=acme&tag=vip", headers=H).json()["total"] == 1
    assert c.get("/api/customers?project=acme&stage=won", headers=H).json()["total"] == 0
    assert c.get("/api/customers?project=acme&page=2&page_size=1", headers=H).json()["customers"] != []
    r = c.put(f"/api/customers/{a['id']}", json={"stage": "interested", "email": "a@x.in"}, headers=H).json()["customer"]
    assert r["stage"] == "interested" and r["score"] > a["score"]
    tl = c.get(f"/api/customers/{a['id']}/timeline", headers=H).json()["timeline"]
    assert {e["type"] for e in tl} == {"created", "stage"}
    c.post(f"/api/customers/{a['id']}/notes", json={"text": "likes Hindi"}, headers=H)
    assert c.get(f"/api/customers/{a['id']}/timeline", headers=H).json()["timeline"][0]["type"] in ("note", "stage")
    # do not call: marks the global list, cannot be undone from here, cannot be deleted, blocks stage moves
    r = c.put(f"/api/customers/{b['id']}", json={"consent": {"dnc": True, "reason": "asked"}}, headers=H).json()["customer"]
    assert r["stage"] == "do_not_call" and r["score"] == 0 and "+919123456780" in json.loads((tmp_path / "dnc.json").read_text())
    assert c.put(f"/api/customers/{b['id']}", json={"consent": {"dnc": False}}, headers=H).status_code == 409
    assert c.put(f"/api/customers/{b['id']}", json={"stage": "new"}, headers=H).status_code == 409
    assert c.delete(f"/api/customers/{b['id']}", headers=H).status_code == 409
    assert c.delete(f"/api/customers/{a['id']}", headers=H).status_code == 200
    # a new customer on a DNC number is created flagged
    assert c.post("/api/customers", json={"project_id": "acme", "phone": "9123456780"}, headers=H).status_code == 409
    dn = json.loads((tmp_path / "dnc.json").read_text())
    dn["+919555555555"] = 1.0
    (tmp_path / "dnc.json").write_text(json.dumps(dn))
    d = customer(c, phone="9555555555")
    assert d["consent"]["dnc"] is True and d["stage"] == "do_not_call"
    assert c.get("/api/customers/nope", headers=H).status_code == 404


CSV = """Name,Mobile No,Email,City,Tags,Budget
Asha,98765 43210,asha@x.in,Pune,"vip;hot",5000
Bala,+91 91234 56780,,Delhi,,
Bad,12345,,Goa,,
Dup,9876543210,,,,
Chandra,9000000001,,,,
Nophone,,,,,
"""


def test_import_preview_then_commit_with_bad_rows_dupes_dnc(tmp_path):
    c, out, _ = make(tmp_path)
    project(c)
    c.post("/api/projects/acme/fields", json={"key": "budget", "type": "number"}, headers=H)
    (tmp_path / "dnc.json").write_text(json.dumps({"+919000000001": 1.0}))
    prev = c.post("/api/customers/import", json={"project": "acme", "csv": CSV}, headers=H).json()
    assert prev["committed"] is False and prev["mapping"]["phone"] == "Mobile No" and prev["mapping"]["custom.budget"] == "Budget"
    assert prev["counts"] == {"new": 2, "updated": 0, "duplicate": 1, "invalid": 2, "dnc": 1}
    st = {r["row"]: r["status"] for r in prev["rows"]}
    assert st[4] == "invalid" and st[5] == "duplicate_in_file" and st[6] == "dnc" and st[7] == "invalid"
    assert c.get("/api/customers?project=acme", headers=H).json()["total"] == 0  # preview wrote nothing
    done = c.post("/api/customers/import", json={"project": "acme", "csv": CSV, "commit": True}, headers=H).json()
    assert done["counts"] == prev["counts"]
    rows = c.get("/api/customers?project=acme", headers=H).json()["customers"]
    by = {r["phone"]: r for r in rows}
    assert set(by) == {"+919876543210", "+919123456780", "+919000000001"}
    assert by["+919876543210"]["tags"] == ["hot", "vip"] and by["+919876543210"]["custom"]["budget"] == 5000.0
    assert by["+919000000001"]["stage"] == "do_not_call" and by["+919000000001"]["consent"]["dnc"]
    # importing again: everything is a duplicate (skip), the DNC one stays DNC; update mode never touches DNC
    again = c.post("/api/customers/import", json={"project": "acme", "csv": CSV, "commit": True}, headers=H).json()
    assert again["counts"]["new"] == 0 and again["counts"]["dnc"] == 1
    upd = c.post("/api/customers/import", json={"project": "acme", "csv": CSV, "commit": True, "on_duplicate": "update", "tags": ["March"]}, headers=H).json()
    assert upd["counts"]["updated"] == 2 and "march" in {t for r in c.get("/api/customers?project=acme", headers=H).json()["customers"] for t in r["tags"]}
    # explicit mapping, errors
    assert c.post("/api/customers/import", json={"project": "acme", "csv": CSV, "mapping": {"name": "Name"}}, headers=H).status_code == 400
    assert c.post("/api/customers/import", json={"project": "acme", "csv": CSV, "mapping": {"phone": "Nope"}}, headers=H).status_code == 400
    assert c.post("/api/customers/import", json={"project": "acme", "csv": ""}, headers=H).status_code == 400


def test_webhook_auth_create_and_dedupe(tmp_path):
    c, out, _ = make(tmp_path)
    p = project(c)
    url = "/api/leads/webhook/acme"
    body = {"name": "Lead One", "phone": "9876500000", "tags": ["Ad"], "email": "l@x.in"}
    assert c.post(url, json=body).status_code == 401  # no cookie needed, but a token is
    assert c.post(url, json=body, headers={"x-webhook-token": "wrong"}).status_code == 401
    assert c.post("/api/leads/webhook/ghost", json=body, headers={"x-webhook-token": p["webhook_token"]}).status_code == 401
    r = c.post(url + f"?token={p['webhook_token']}", json=body)
    assert r.status_code == 200 and r.json()["created"] is True
    r2 = c.post(url, json={**body, "name": "Other"}, headers={"x-webhook-token": p["webhook_token"]}).json()
    assert r2["created"] is False and r2["customer_id"] == r.json()["customer_id"]
    cu = c.get(f"/api/customers/{r2['customer_id']}", headers=H).json()["customer"]
    assert cu["source"] == "webhook" and cu["name"] == "Lead One" and "webhook" in cu["tags"]
    assert c.post(url, json={"name": "x"}, headers={"x-webhook-token": p["webhook_token"]}).status_code == 400
    # rotate invalidates the old token
    new = c.post("/api/projects/acme/webhook-token/rotate", headers=H).json()["webhook_token"]
    assert c.post(url, json=body, headers={"x-webhook-token": p["webhook_token"]}).status_code == 401
    assert c.post(url, json=body, headers={"x-webhook-token": new}).status_code == 200


def test_score_is_deterministic_and_documented_ranges():
    now = 1_700_000_000.0
    cu = Customer(project_id="p", phone="+919876543210", stage="interested")
    assert score(cu, now) == score(cu, now) == 40
    cu.history.outcomes = [{"outcome": "interested"}, {"outcome": "accepted"}]
    cu.last_contact_at = now - 3600
    assert score(cu, now) == 40 + 10 + 15 + 10  # stage + 2 connected calls + last accepted + contacted within 2 days
    cu.score_delta_total = 99
    assert score(cu, now) == 40 + 10 + 15 + 10 + 20  # call deltas are clamped
    cu.score_delta_total = 0
    cu.history.outcomes = [{"outcome": "no_answer"}] * 4
    assert score(cu, now) == 40 - 9 + 10
    cu.stage = "won"
    assert score(cu, now) == 100
    for s in ("lost", "do_not_call"):
        cu.stage = s
        assert score(cu, now) == 0
    old = Customer(project_id="p", phone="+919", stage="new", last_contact_at=now - 40 * 86400)
    assert score(old, now) == 0


def test_store_atomic_write_leaves_no_temp_and_survives_corrupt(tmp_path):
    atomic_write(tmp_path / "x.json", {"a": 1})
    assert json.loads((tmp_path / "x.json").read_text()) == {"a": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["x.json"]
    (tmp_path / "campaigns.json").write_text("{not json")
    s = JsonStore(tmp_path)
    assert s.campaigns() == [] and (tmp_path / "campaigns.json.corrupt").exists()


def test_calltypes_builtins_custom_duplicate(tmp_path):
    c, _, _ = make(tmp_path)
    project(c)
    ids = {x["id"] for x in c.get("/api/calltypes?project=acme", headers=H).json()["calltypes"]}
    assert {"sales_pitch", "lead_qualification", "follow_up", "payment_reminder", "feedback_survey", "appointment_booking"} <= ids
    d = c.post("/api/calltypes/follow_up/duplicate?project=acme", json={"name": "Link not paid"}, headers=H).json()["calltype"]
    assert d["id"] == "link_not_paid" and d["project_id"] == "acme" and d["builtin"] is False
    assert c.put("/api/calltypes/sales_pitch", json={"name": "x"}, headers=H).status_code == 400
    r = c.put("/api/calltypes/link_not_paid?project=acme", json={"objective": "Ask about the link", "steps": [{"id": "s1", "text": "Did you get it?", "answer_type": "yes_no"}]}, headers=H)
    assert r.json()["calltype"]["steps"][0]["answer_type"] == "yes_no"
    assert "link_not_paid" in {x["id"] for x in c.get("/api/calltypes?project=acme", headers=H).json()["calltypes"]}
    assert "link_not_paid" not in {x["id"] for x in c.get("/api/calltypes?project=other", headers=H).json()["calltypes"]}
    assert c.delete("/api/calltypes/link_not_paid", headers=H).status_code == 200
