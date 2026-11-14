import json

import pytest

from crm_helpers import H, Clock, FakeDialer, FakeLLM, make, project, customer, write_call

OPEN = 1_700_026_200.0  # Wed 15 Nov 2023 11:00 IST
HOUR = 3600


def setup(tmp_path, **kw):
    c, out, clock = make(tmp_path, clock=Clock(OPEN), **kw)
    project(c)
    return c, out, clock


def camp(c, **kw):
    r = c.post("/api/campaigns", json={"project": "acme", "name": "Nov", "product_id": "p1", **kw}, headers=H)
    assert r.status_code == 200, r.text
    return r.json()["campaign"]


def queue(c, cid):
    return c.get(f"/api/campaigns/{cid}/queue", headers=H).json()


def test_campaign_crud_and_validation(tmp_path):
    c, _, _ = setup(tmp_path)
    cp = camp(c, schedule={"max_parallel": 2, "retry": {"max_attempts": 2}}, audience={"tags": ["hot"]})
    assert cp["status"] == "draft" and cp["dial_mode"] == "manual" and cp["schedule"]["max_parallel"] == 2
    assert c.post("/api/campaigns", json={"project": "acme", "name": "x", "call_type_id": "nope"}, headers=H).status_code == 400
    assert c.post("/api/campaigns", json={"project": "ghost", "name": "x"}, headers=H).status_code == 404
    assert c.post("/api/campaigns", json={"project": "acme", "name": "x", "schedule": {"max_parallel": 0}}, headers=H).status_code == 422
    assert c.put(f"/api/campaigns/{cp['id']}", json={"name": "Renamed", "status": "running"}, headers=H).json()["campaign"]["status"] == "draft"
    assert c.post(f"/api/campaigns/{cp['id']}/start", headers=H).json()["campaign"]["status"] == "running"
    assert c.delete(f"/api/campaigns/{cp['id']}", headers=H).status_code == 409
    assert c.post(f"/api/campaigns/{cp['id']}/pause", headers=H).json()["campaign"]["status"] == "paused"
    assert c.delete(f"/api/campaigns/{cp['id']}", headers=H).status_code == 200
    nop = camp(c, product_id="")
    assert c.post(f"/api/campaigns/{nop['id']}/start", headers=H).status_code == 400  # no product


def test_queue_respects_hours_days_dnc_and_order(tmp_path):
    c, out, clock = setup(tmp_path)
    a = customer(c, phone="9000000001", name="A")
    b = customer(c, phone="9000000002", name="B")
    d = customer(c, phone="9000000003", name="D")
    c.put(f"/api/customers/{b['id']}", json={"stage": "interested"}, headers=H)  # higher score first
    c.put(f"/api/customers/{d['id']}", json={"consent": {"dnc": True}}, headers=H)
    cp = camp(c)
    c.post(f"/api/campaigns/{cp['id']}/start", headers=H)
    q = queue(c, cp["id"])
    assert q["window"]["open"] and q["dialable_now"]
    assert [r["name"] for r in q["eligible"]] == ["B", "A"]
    assert any(s["customer_id"] == d["id"] and s["reason"] == "do not call" for s in q["skipped"])
    assert [r["name"] for r in q["next"]] == ["B"]  # max_parallel 1
    # outside the window: nothing dialable, tells when it opens
    clock.t = OPEN - 8 * HOUR  # 03:00 IST
    q = queue(c, cp["id"])
    assert not q["window"]["open"] and not q["dialable_now"] and q["next"] == []
    assert q["window"]["next_open_at"] == OPEN - HOUR  # 10:00 IST
    # Sunday is not in the default days
    clock.t = OPEN + 4 * 86400  # Sun 19 Nov 11:00 IST
    q = queue(c, cp["id"])
    assert not q["window"]["open"] and q["window"]["next_open_at"] == OPEN + 5 * 86400 - HOUR
    # a draft/paused campaign never reports anything dialable
    clock.t = OPEN
    c.post(f"/api/campaigns/{cp['id']}/pause", headers=H)
    assert queue(c, cp["id"])["dialable_now"] is False and queue(c, cp["id"])["next"] == []
    # project hours override + campaign window override
    c.put("/api/projects/acme", json={"call_hours": [12, 20]}, headers=H)
    assert not queue(c, cp["id"])["window"]["open"]
    c.put(f"/api/campaigns/{cp['id']}", json={"schedule": {"window": [9, 18]}}, headers=H)
    assert queue(c, cp["id"])["window"]["open"]


def test_queue_audience_filters(tmp_path):
    c, _, _ = setup(tmp_path)
    customer(c, phone="9000000001", name="A", tags=["hot"], language="en")
    customer(c, phone="9000000002", name="B", tags=["cold"])
    won = customer(c, phone="9000000003", name="W", tags=["hot"])
    c.put(f"/api/customers/{won['id']}", json={"stage": "won"}, headers=H)
    cp = camp(c, audience={"tags": ["hot"]})
    assert [r["name"] for r in queue(c, cp["id"])["eligible"]] == ["A"]
    cp2 = camp(c, audience={"languages": ["hi"]})
    assert [r["name"] for r in queue(c, cp2["id"])["eligible"]] == ["B"]
    cp3 = camp(c, audience={"stages": ["won"]})  # terminal stages are never dialed
    assert queue(c, cp3["id"])["eligible"] == []


def test_queue_retries_and_gives_up(tmp_path):
    c, out, clock = setup(tmp_path)
    cu = customer(c, phone="9000000001", name="A")
    cp = camp(c, schedule={"retry": {"max_attempts": 2, "retry_after_minutes": 60}})
    c.post(f"/api/campaigns/{cp['id']}/start", headers=H)

    def attempt(n, outcome):
        write_call(tmp_path, f"x{n}", status="ended", answered_at=None, end_reason="no-answer", **({"machine": True} if outcome == "voicemail" else {}),
                   ended_at=clock.t)
        out["store"].link_call(f"x{n}", {"project_id": "acme", "customer_id": cu["id"], "campaign_id": cp["id"], "call_type": "sales_pitch"})
        assert c.post(f"/api/crm/calls/x{n}/apply", headers=H).json()["applied"]
    attempt(1, "no_answer")
    q = queue(c, cp["id"])
    assert q["eligible"] == [] and q["skipped"][0]["reason"] == "waiting before the next retry"
    clock.t += HOUR
    q = queue(c, cp["id"])
    assert q["eligible"][0]["reason"] == "retry 2/2 after no_answer"
    attempt(2, "no_answer")
    clock.t += 2 * HOUR
    q = queue(c, cp["id"])
    assert q["eligible"] == [] and "gave up" in q["skipped"][0]["reason"]
    assert c.get(f"/api/customers/{cu['id']}", headers=H).json()["customer"]["next_action"] is None  # retries exhausted


def test_operator_call_goes_through_dialer_and_links(tmp_path):
    d = FakeDialer()
    c, out, _ = setup(tmp_path, dialer=d)
    cu = customer(c, phone="9000000001", name="Asha", language="en")
    cp = camp(c, call_type_id="lead_qualification")
    r = c.post(f"/api/campaigns/{cp['id']}/call/{cu['id']}", json={}, headers=H)
    assert r.status_code == 200 and r.json()["call_id"] == "call0001"
    assert d.calls == [{"to": "+919000000001", "product": "p1", "language": "en",
                        "link": {"project_id": "acme", "customer_id": cu["id"], "campaign_id": cp["id"], "call_type": "lead_qualification"}}]
    assert out["store"].call_link("call0001")["campaign_id"] == cp["id"]
    tl = c.get(f"/api/customers/{cu['id']}/timeline", headers=H).json()["timeline"]
    assert tl[0]["type"] == "call_started" and tl[0]["map_url"] == "/api/calls/call0001/map"
    # per-call override of the call type; unknown call types refused
    assert c.post(f"/api/campaigns/{cp['id']}/call/{cu['id']}", json={"call_type": "follow_up"}, headers=H).status_code == 200
    assert d.calls[-1]["link"]["call_type"] == "follow_up"
    assert c.post(f"/api/campaigns/{cp['id']}/call/{cu['id']}", json={"call_type": "zzz"}, headers=H).status_code == 400
    assert c.post(f"/api/campaigns/{cp['id']}/call/ghost", json={}, headers=H).status_code == 404


def test_operator_call_refuses_dnc_and_gate_blocks(tmp_path):
    d = FakeDialer()
    c, _, _ = setup(tmp_path, dialer=d)
    cu = customer(c, phone="9000000001")
    cp = camp(c)
    c.put(f"/api/customers/{cu['id']}", json={"consent": {"dnc": True}}, headers=H)
    r = c.post(f"/api/campaigns/{cp['id']}/call/{cu['id']}", json={"allow_dnc": True}, headers=H)
    assert r.status_code == 409 and d.calls == []  # allow_dnc is not honoured by the CRM
    d2 = FakeDialer(block="this campaign may only call its test numbers (product setting allowed_numbers)")
    c2, _, _ = make(tmp_path / "b", dialer=d2) if (tmp_path / "b").mkdir() is None else None
    project(c2)
    cu2 = customer(c2, phone="9000000009")
    cp2 = camp(c2)
    r = c2.post(f"/api/campaigns/{cp2['id']}/call/{cu2['id']}", json={}, headers=H)
    assert r.status_code == 409 and "test numbers" in r.json()["detail"]
    assert c2.get(f"/api/customers/{cu2['id']}/timeline", headers=H).json()["timeline"][0]["type"] == "created"  # nothing logged as started


def test_auto_mode_needs_confirm_flag_env_and_running(tmp_path, monkeypatch):
    d = FakeDialer()
    c, _, clock = setup(tmp_path, dialer=d)
    customer(c, phone="9000000001")
    customer(c, phone="9000000002")
    assert c.post("/api/campaigns", json={"project": "acme", "name": "x", "product_id": "p1", "dial_mode": "auto"}, headers=H).status_code == 400
    cp = camp(c, dial_mode="auto", confirm_auto=True, schedule={"max_parallel": 5, "pacing_seconds": 0})
    assert cp["auto_confirmed"] is True
    monkeypatch.delenv("CRM_AUTO_DIAL", raising=False)
    c.post(f"/api/campaigns/{cp['id']}/start", headers=H)
    r = c.post(f"/api/campaigns/{cp['id']}/tick", headers=H).json()
    assert r["armed"] is False and r["dialed"] == [] and d.calls == []  # env flag missing: report only
    monkeypatch.setenv("CRM_AUTO_DIAL", "1")
    clock.t = OPEN - 8 * HOUR  # outside hours: armed but nothing in the queue's next
    r = c.post(f"/api/campaigns/{cp['id']}/tick", headers=H).json()
    assert r["armed"] and r["dialed"] == [] and d.calls == []
    clock.t = OPEN
    r = c.post(f"/api/campaigns/{cp['id']}/tick", headers=H).json()
    assert len(r["dialed"]) == 2 and len(d.calls) == 2
    # switching back to manual drops the confirmation; switching to auto again needs a fresh confirm
    assert c.put(f"/api/campaigns/{cp['id']}", json={"dial_mode": "manual"}, headers=H).json()["campaign"]["auto_confirmed"] is False
    assert c.put(f"/api/campaigns/{cp['id']}", json={"dial_mode": "auto"}, headers=H).status_code == 400
    # paused campaigns are never ticked
    c.post(f"/api/campaigns/{cp['id']}/pause", headers=H)
    c.put(f"/api/campaigns/{cp['id']}", json={"dial_mode": "auto", "confirm_auto": True}, headers=H)
    n = len(d.calls)
    assert c.post(f"/api/campaigns/{cp['id']}/tick", headers=H).json()["armed"] is False and len(d.calls) == n


def test_queue_pacing_and_parallel_cap(tmp_path):
    c, out, clock = setup(tmp_path, dialer=FakeDialer(), active=lambda: (1, set()))
    customer(c, phone="9000000001")
    cp = camp(c, schedule={"max_parallel": 1, "pacing_seconds": 120})
    c.post(f"/api/campaigns/{cp['id']}/start", headers=H)
    q = queue(c, cp["id"])
    assert q["slots"] == 0 and q["next"] == [] and not q["dialable_now"] and len(q["eligible"]) == 1  # one call already live


# ---------------------------------------------------------------- outcomes

def linked(c, out, tmp_path, call_id, cu, cp=None, **rec):
    write_call(tmp_path, call_id, **rec)
    out["store"].link_call(call_id, {"project_id": "acme", "customer_id": cu["id"], "campaign_id": cp["id"] if cp else None, "call_type": "sales_pitch"})
    r = c.post(f"/api/crm/calls/{call_id}/apply", headers=H).json()
    return r, c.get(f"/api/customers/{cu['id']}", headers=H).json()["customer"]


def test_outcome_accepted_link_sent_schedules_followup(tmp_path):
    c, out, clock = setup(tmp_path)
    cu = customer(c)
    r, cust = linked(c, out, tmp_path, "a1", cu, context={"decision": "accepted", "promised": ["payment_link"], "objections": ["price"]})
    assert r["outcome"] == "accepted" and cust["stage"] == "negotiating"  # not "won" until paid
    assert cust["next_action"]["type"] == "follow_up" and cust["next_action"]["call_type"] == "follow_up" and cust["next_action"]["due_estimated"]
    assert cust["next_action"]["due_at"] == OPEN + 86400 and cust["history"]["objections"] == ["price"] and cust["history"]["call_ids"] == ["a1"]
    assert cust["score"] > 40


def test_outcome_declined_lost_with_nurture(tmp_path):
    c, out, _ = setup(tmp_path)
    cu = customer(c)
    r, cust = linked(c, out, tmp_path, "d1", cu, context={"decision": "declined"})
    assert cust["stage"] == "lost" and cust["score"] == 0
    assert cust["next_action"]["type"] == "nurture" and cust["next_action"]["due_at"] == OPEN + 30 * 86400


def test_outcome_opt_out_is_dnc_everywhere(tmp_path):
    c, out, _ = setup(tmp_path)
    cu = customer(c)
    r, cust = linked(c, out, tmp_path, "o1", cu, context={"promised": ["number_removed_no_more_calls"]})
    assert cust["stage"] == "do_not_call" and cust["consent"]["dnc"] and cust["next_action"] is None and cust["score"] == 0
    assert "+919876543210" in json.loads((tmp_path / "dnc.json").read_text())
    cp = camp(c)
    assert all(s["reason"] == "do not call" for s in queue(c, cp["id"])["skipped"]) and queue(c, cp["id"])["eligible"] == []


def test_outcome_callback_promise_creates_followup(tmp_path):
    c, out, clock = setup(tmp_path)
    cu = customer(c)
    r, cust = linked(c, out, tmp_path, "c1", cu, context={"decision": "later", "promised": ["callback_later"]})
    assert r["outcome"] == "callback" and cust["stage"] == "contacted"
    na = cust["next_action"]
    assert na["type"] == "callback" and na["call_type"] == "follow_up" and na["due_at"] == OPEN + 86400
    assert c.get("/api/followups?project=acme&due=now", headers=H).json()["followups"] == []
    assert len(c.get("/api/followups?project=acme&due=week", headers=H).json()["followups"]) == 1
    clock.t = OPEN + 86400 + 5
    fu = c.get("/api/followups?project=acme&due=now", headers=H).json()["followups"]
    assert [f["customer_id"] for f in fu] == [cu["id"]] and fu[0]["overdue"] is True
    # the campaign queue uses the follow-up call type
    cp = camp(c)
    row = queue(c, cp["id"])["eligible"][0]
    assert row["call_type"] == "follow_up"


def test_outcomes_unreached_and_undecided_and_human(tmp_path):
    c, out, _ = setup(tmp_path)
    n = customer(c, phone="9000000001")
    r, cust = linked(c, out, tmp_path, "n1", n, answered_at=None, end_reason="no-answer")
    assert r["outcome"] == "no_answer" and cust["stage"] == "new" and cust["last_contact_at"] is None
    assert cust["next_action"]["type"] == "retry" and cust["next_action"]["due_at"] == OPEN + 240 * 60
    v = customer(c, phone="9000000002")
    r, cust = linked(c, out, tmp_path, "v1", v, machine=True)
    assert r["outcome"] == "voicemail"
    f = customer(c, phone="9000000003")
    r, cust = linked(c, out, tmp_path, "f1", f, status="failed", answered_at=None, end_reason="Plivo refused")
    assert r["outcome"] == "failed"
    u = customer(c, phone="9000000004")
    r, cust = linked(c, out, tmp_path, "u1", u, turns=[{"role": "agent", "text": "hi"}, {"role": "customer", "text": "hmm"}])
    assert r["outcome"] == "undecided" and cust["stage"] == "contacted" and cust["last_contact_at"] == 1_700_000_100.0
    assert cust["next_action"]["type"] == "follow_up"
    h = customer(c, phone="9000000005")
    r, cust = linked(c, out, tmp_path, "h1", h, context={"promised": ["team_member_will_call"]}, turns=[{"role": "customer", "text": "ok"}])
    assert r["outcome"] == "callback" and cust["next_action"]["type"] == "follow_up" and "team member" in cust["next_action"]["note"]


def test_structured_result_wins_over_context(tmp_path):
    c, out, _ = setup(tmp_path)
    cu = customer(c)
    res = {"outcome": "interested", "stage_suggestion": "qualified", "score_delta": 7,
           "next_action": {"type": "callback", "due_at": OPEN + 7200, "call_type": "follow_up"},
           "answers": {"budget": "5000", "agreed_price": 899, "likes": ["monthly plan"]}}
    r, cust = linked(c, out, tmp_path, "r1", cu, result=res, context={"decision": "declined"})
    assert r["outcome"] == "interested" and not r["derived"] and cust["stage"] == "qualified"
    assert cust["next_action"]["due_at"] == OPEN + 7200 and cust["next_action"]["due_estimated"] is False
    assert cust["history"]["agreed_price"] == 899 and cust["history"]["facts"]["budget"] == "5000" and cust["history"]["likes"] == ["monthly plan"]
    assert cust["score_delta_total"] == 7
    # stages never move backwards from a suggestion, and won stays won
    r, cust = linked(c, out, tmp_path, "r2", cu, result={"outcome": "undecided", "stage_suggestion": "new"})
    assert cust["stage"] == "qualified"
    c.put(f"/api/customers/{cu['id']}", json={"stage": "won"}, headers=H)
    r, cust = linked(c, out, tmp_path, "r3", cu, result={"outcome": "declined"})
    assert cust["stage"] == "won" and cust["next_action"] is None
    # aliases from the calltypes lane map to canonical outcomes
    d = customer(c, phone="9000000009")
    r, _ = linked(c, out, tmp_path, "r4", d, result={"outcome": "demo_booked", "stage_suggestion": "demo_booked"})
    assert r["outcome"] == "interested" and r["stage"] == "demo_booked"


def test_apply_is_idempotent_and_ignores_unlinked_or_live_calls(tmp_path):
    c, out, _ = setup(tmp_path)
    cu = customer(c)
    r, cust = linked(c, out, tmp_path, "i1", cu, context={"decision": "declined"})
    again = c.post("/api/crm/calls/i1/apply", headers=H).json()
    assert again == {"applied": False, "reason": "already applied", "customer_id": cu["id"]}
    assert len(c.get(f"/api/customers/{cu['id']}", headers=H).json()["customer"]["history"]["outcomes"]) == 1
    write_call(tmp_path, "unl")
    assert not c.post("/api/crm/calls/unl/apply", headers=H).json()["applied"]
    live = customer(c, phone="9000000001")
    out["store"].link_call("lv", {"customer_id": live["id"], "project_id": "acme"})
    write_call(tmp_path, "lv", status="live")
    assert c.post("/api/crm/calls/lv/apply", headers=H).json()["reason"] == "call not finished"
    assert c.post("/api/crm/calls/ghost/apply", headers=H).status_code == 404


def test_hook_and_sweep_apply_finished_calls(tmp_path):
    c, out, _ = setup(tmp_path)
    cu = customer(c)
    out["store"].link_call("h9", {"customer_id": cu["id"], "project_id": "acme", "call_type": "sales_pitch"})
    rec = write_call(tmp_path, "h9", context={"decision": "accepted", "promised": ["payment_link"]}, crm_customer_id=cu["id"])
    assert out["on_call_end"](rec)["applied"]
    assert out["on_call_end"]({"call_id": "zz", "crm_customer_id": "ghost"})["applied"] is False  # never raises
    # sweep: a finished linked call that the hook missed is applied when follow-ups are read
    cu2 = customer(c, phone="9000000002")
    out["store"].link_call("s1", {"customer_id": cu2["id"], "project_id": "acme"})
    write_call(tmp_path, "s1", context={"decision": "later", "promised": ["callback_later"]})
    c.get("/api/followups?project=acme&due=all", headers=H)
    assert c.get(f"/api/customers/{cu2['id']}", headers=H).json()["customer"]["next_action"]["type"] == "callback"


def test_followups_skip_dnc_and_terminal_and_sort(tmp_path):
    c, out, clock = setup(tmp_path)
    a, b, d = customer(c, phone="9000000001"), customer(c, phone="9000000002"), customer(c, phone="9000000003")
    na = lambda t: {"next_action": {"type": "follow_up", "due_at": OPEN + t}}
    c.put(f"/api/customers/{a['id']}", json=na(-100), headers=H)
    c.put(f"/api/customers/{b['id']}", json=na(-500), headers=H)
    c.put(f"/api/customers/{d['id']}", json=na(-900), headers=H)
    c.put(f"/api/customers/{d['id']}", json={"consent": {"dnc": True}}, headers=H)
    fu = c.get("/api/followups?project=acme", headers=H).json()["followups"]
    assert [f["customer_id"] for f in fu] == [b["id"], a["id"]]
    assert c.get("/api/followups?project=ghost", headers=H).status_code == 404


# ---------------------------------------------------------------- AI assist

GOOD = {"name": "Link not paid", "call_type_id": "follow_up", "language": "hi", "audience": {"tags": ["link-sent"]}}


def assist(c, kind="campaign", prompt="a follow-up for people who got the link but did not pay", **kw):
    return c.post("/api/ai/assist", json={"kind": kind, "prompt": prompt, **kw}, headers=H)


def test_assist_returns_validated_draft_and_saves_nothing(tmp_path):
    llm = FakeLLM(GOOD)
    c, out, _ = setup(tmp_path, llm=llm)
    r = assist(c, context={"project": "acme"})
    assert r.status_code == 200
    j = r.json()
    assert j["saved"] is False and j["repaired"] is False and j["draft"]["name"] == "Link not paid" and j["draft"]["audience"]["tags"] == ["link-sent"]
    assert j["draft"]["schedule"]["max_parallel"] == 1  # defaults filled by the model
    stage, payload = llm.calls[0]
    assert stage == "assist_campaign" and payload["context"] == {"project": "acme"} and "properties" in payload["schema"]
    assert c.get("/api/campaigns", headers=H).json()["campaigns"] == []


def test_assist_repairs_once_then_gives_up(tmp_path):
    llm = FakeLLM({"call_type_id": "follow_up"}, GOOD)  # first draft lacks required name
    c, _, _ = setup(tmp_path, llm=llm)
    j = assist(c).json()
    assert j["repaired"] is True and len(llm.calls) == 2
    assert llm.calls[1][1]["fix"] and "name" in llm.calls[1][1]["fix"][0] and llm.calls[1][1]["previous_draft"] == {"call_type_id": "follow_up"}
    bad = FakeLLM({"nope": 1})
    c2, _, _ = make(tmp_path / "z", llm=bad) if (tmp_path / "z").mkdir() is None else None
    r = assist(c2)
    assert r.status_code == 422 and "name" in r.json()["detail"] and len(bad.calls) == 2


def test_assist_kinds_and_errors(tmp_path):
    llm = FakeLLM({"fields": [{"key": "budget", "type": "number"}]})
    c, _, _ = setup(tmp_path, llm=llm)
    assert assist(c, "customer_fields", "budget and city").json()["draft"]["fields"][0]["key"] == "budget"
    assert assist(c, "bogus").status_code == 400
    assert assist(c, prompt="  ").status_code == 400
    boom = FakeLLM(RuntimeError("upstream down"))
    c2, _, _ = make(tmp_path / "q", llm=boom) if (tmp_path / "q").mkdir() is None else None
    r = assist(c2)
    assert r.status_code == 502 and "upstream down" in r.json()["detail"]
    # no LLM configured at all
    c3, _, _ = make(tmp_path / "w") if (tmp_path / "w").mkdir() is None else None
    for k in ("PRODUCTGEN_BASE_URL", "ACIE_EXPR_BASE_URL"):
        import os
        os.environ.pop(k, None)
    assert assist(c3).status_code == 503


def test_assist_llm_request_shape_has_no_seed_and_no_key_in_errors(monkeypatch):
    import asyncio
    import httpx
    from voice.crm import ai_assist
    monkeypatch.setenv("ACIE_EXPR_BASE_URL", "http://llm.test/v1")
    monkeypatch.setenv("ACIE_EXPR_MODEL", "gemini-x")
    monkeypatch.setenv("ACIE_EXPR_API_KEY", "SECRET-KEY-123")
    monkeypatch.setenv("PRODUCTGEN_SEED", "5")  # even when set, the assist client never sends a seed
    seen = {}

    class FakeResp:
        status_code = 200

        def json(self):
            return {"choices": [{"message": {"content": json.dumps(GOOD)}}]}

    class FakeClient:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, url, json=None, headers=None):
            seen.update(url=url, body=json, headers=headers)
            return FakeResp()

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    llm = ai_assist.AssistLLM()
    out = asyncio.run(ai_assist.assist(llm, "campaign", "x"))
    assert out["draft"]["name"] == "Link not paid"
    b = seen["body"]
    assert seen["url"] == "http://llm.test/v1/chat/completions" and b["temperature"] == 0 and "seed" not in b
    assert b["response_format"] == {"type": "json_object"} and "SECRET-KEY-123" not in json.dumps(b)

    class Err(FakeResp):
        status_code = 401
        text = "bad key SECRET-KEY-123"
    FakeClient.post = lambda self, *a, **k: _err(Err())

    async def _err(r):
        return r
    with pytest.raises(ai_assist.AssistError) as e:
        asyncio.run(ai_assist.assist(llm, "campaign", "x"))
    assert "SECRET-KEY-123" not in e.value.msg
