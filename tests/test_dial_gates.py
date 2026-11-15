"""Owner, 15 Nov 2026: the operator must be able to start calls. A past distress flag is a warning on the call result (the call
still goes out), DNC is dialed when the operator overrides it (allow_dnc), CALL_HOURS stays off by default, and an empty
allowed_numbers means anyone. Only a product with a non-empty allowed_numbers list (claude-pro: a real company) is restricted.
No network: Plivo's dial is replaced by a recorder."""
import asyncio
import json
import os

import pytest

from acie import safety
from acie.personalization.profile_store import JsonProfileStore


def _build(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    for k, v in {"PLIVO_AUTH_ID": "x", "PLIVO_AUTH_TOKEN": "x", "PLIVO_NUMBER": "+910000000000", "ADMIN_PASSWORD": "pw",
                 "SESSION_SECRET": "s3cret"}.items():
        os.environ.setdefault(k, v)
    try:
        from voice import server
    except ImportError as e:
        pytest.skip(f"voice server deps missing: {e}")
    data = tmp_path / "voice"
    (data / "calls").mkdir(parents=True)
    (data / "products.json").write_text(json.dumps({
        "claude-pro": {"id": "claude-pro", "allowed_numbers": ["+919999900001"]},
        "gym": {"id": "gym", "allowed_numbers": []},
    }))
    monkeypatch.setattr(server, "DATA", data)
    monkeypatch.setattr(server, "CALLS_DIR", data / "calls")
    monkeypatch.delenv("CALL_HOURS", raising=False)
    api, state = server.build_app()
    api.router.on_startup.clear()
    state.cfg.admin_password = "pw"
    state.store = JsonProfileStore(data / "profiles")
    state.backends.setdefault("template", object())  # startup (models) is skipped; the dial never builds a pipeline here
    dialed = []

    async def fake_dial(call):
        dialed.append(call.to)
        call.request_uuid, call.status = "req-" + call.call_id, "ringing"
    state.plivo.dial = fake_dial
    from fastapi.testclient import TestClient
    c = TestClient(api, base_url="https://testserver")  # the session cookie is Secure
    assert c.post("/api/login", json={"password": "pw"}).status_code == 200
    return c, state, data, dialed


def _flag(state, number):
    prof = asyncio.run(state.store.load(number))
    prof.outcomes.append("safety_distress")
    asyncio.run(state.store.save(prof))


def test_distress_flag_is_a_warning_not_a_block(tmp_path, monkeypatch):
    c, state, data, dialed = _build(tmp_path, monkeypatch)
    _flag(state, "+919876543210")
    r = c.post("/api/calls", json={"numbers": "+919876543210", "campaign": "gym", "backend": "template", "language": "en"})
    assert r.status_code == 200, r.text
    row = r.json()["calls"][0]
    assert dialed == ["+919876543210"] and row["call_id"] and not row["error"], row
    assert "distress" in row["warning"]


def test_dnc_needs_override_then_dials_with_warning(tmp_path, monkeypatch):
    c, state, data, dialed = _build(tmp_path, monkeypatch)
    state.add_dnc("+919876543210")
    body = {"numbers": "+919876543210", "campaign": "gym", "backend": "template", "language": "en"}
    r = c.post("/api/calls", json=body)
    assert r.status_code == 200, r.text
    row = r.json()["calls"][0]
    assert dialed == [] and "call anyway" in row["error"]
    row = c.post("/api/calls", json={**body, "allow_dnc": True}).json()["calls"][0]  # "call anyway"
    assert dialed == ["+919876543210"] and row["call_id"] and "do-not-call" in row["warning"]


def test_allowed_numbers_only_restricts_products_that_set_them(tmp_path, monkeypatch):
    c, state, data, dialed = _build(tmp_path, monkeypatch)
    assert state.dial_block("+911234567890", "gym", {}) is None  # empty list = anyone
    assert "allowed_numbers" in state.dial_block("+911234567890", "claude-pro", {})  # the real-brand product stays locked
    assert state.dial_block("+919999900001", "claude-pro", {}) is None


def test_call_hours_off_by_default(tmp_path, monkeypatch):
    c, state, data, dialed = _build(tmp_path, monkeypatch)
    state.cfg.call_window = (0, 0)  # every hour is "outside"
    assert state.dial_block("+911234567890", "gym", {}) is None
    monkeypatch.setenv("CALL_HOURS", "1")
    assert "calling hours" in state.dial_block("+911234567890", "gym", {})


def test_discount_line_is_not_distress_even_with_the_guardrail_on(monkeypatch):
    monkeypatch.setenv("ACIE_SAFETY", "1")
    hist = ["Not this offer but about discount offers.", "uh sure so i can just pay 33 rupees per day rather than a full monthly payment"]
    for line in ("So can't you just give me a good discount for me?", "give me a good deal for four months please",
                 "can you do better for me, a good price"):
        assert safety.distress_score(line, hist)[0] < safety.CONCERN_T, line
