"""Call-type schema, built-ins and registry validation."""

import json

import pytest

from acie.calltypes import CallResult, CallType, CallTypeRegistry, Step, load_custom, validate_call_type
from acie.calltypes.builtins import builtin_call_types
from acie.calltypes.registry import parse_custom

BUILTIN_IDS = {"sales_pitch", "lead_qualification", "follow_up", "payment_reminder", "feedback_survey", "appointment_booking"}


def good(**kw):
    base = dict(id="my_call", name="My call", opening={"en": "Hi", "hi": "नमस्ते", "hinglish": "Namaste"},
                steps=[{"id": "q1", "text": {"en": "Ready?", "hi": "तैयार?", "hinglish": "Ready?"}, "answer_type": "yes_no"}],
                success_outcomes=["completed"], allowed_actions=["transfer_human"])
    base.update(kw)
    return base


def test_builtins_present_and_valid():
    b = builtin_call_types()
    assert set(b) == BUILTIN_IDS
    for ct in b.values():
        assert ct.builtin and validate_call_type(ct) == [], ct.id
    assert b["sales_pitch"].pack == "sales_pitch" and b["sales_pitch"].selling_allowed and b["sales_pitch"].negotiation_allowed
    assert not b["lead_qualification"].selling_allowed
    assert [s.id for s in b["lead_qualification"].steps] == ["need", "budget", "authority", "timeline", "book_demo"]


def test_all_three_languages_everywhere_in_builtins():
    for ct in builtin_call_types().values():
        for field in ("opening", "closing"):
            d = getattr(ct, field)
            assert not d or set(d) >= {"en", "hi", "hinglish"}, (ct.id, field)
        for s in ct.steps:
            assert set(s.text) >= {"en", "hi", "hinglish"}, (ct.id, s.id)


def test_plain_strings_are_coerced():
    ct = CallType.model_validate(good(opening="Hello", closing="Bye", compliance_lines=["Recorded."]))
    assert ct.opening == {"en": "Hello", "hi": "Hello", "hinglish": "Hello"}
    assert ct.compliance_lines[0]["hi"] == "Recorded."
    ct2 = CallType.model_validate(good(steps=[{"id": "c", "text": "Pick", "answer_type": "choice", "choices": ["a", "b"]}]))
    assert [c.id for c in ct2.steps[0].choices] == ["a", "b"]


def test_call_result_shape():
    r = CallResult(outcome="demo_booked", stage_suggestion="demo_booked", next_action={"type": "book_demo", "due_at": "2026-11-16T18:00:00+05:30", "call_type": ""},
                   answers={"budget": 10000, "authority": True}, score_delta=70)
    d = r.model_dump()
    assert set(d) >= {"outcome", "stage_suggestion", "next_action", "answers", "score_delta"}
    assert set(d["next_action"]) == {"type", "due_at", "call_type"}
    assert CallResult.model_validate_json(r.model_dump_json()) == r


@pytest.mark.parametrize("patch,needle", [
    (dict(steps=[]), "at least one step"),
    (dict(opening={"en": "Hi"}), "opening has no hi"),
    (dict(allowed_actions=["teleport"]), "unknown allowed action"),
    (dict(success_outcomes=[]), "success_outcomes is empty"),
    (dict(steps=[{"id": "a", "text": "x"}, {"id": "a", "text": "y"}]), "duplicate step id"),
    (dict(steps=[{"id": "c", "text": "x", "answer_type": "choice", "choices": ["only"]}]), "fewer than 2 choices"),
    (dict(steps=[{"id": "q", "text": "x", "action": "book_demo"}], allowed_actions=["transfer_human"]), "not in allowed_actions"),
    (dict(steps=[{"id": "q", "text": "x", "action": "fly"}]), "unknown action"),
    (dict(steps=[{"id": "q", "text": "x", "skip_if": "???"}]), "unreadable skip_if"),
    (dict(languages=["en", "klingon"]), "unknown language"),
])
def test_validation_catches(patch, needle):
    errs = validate_call_type(CallType.model_validate(good(**patch)))
    assert any(needle in e for e in errs), errs


def test_a_good_custom_type_validates():
    assert validate_call_type(CallType.model_validate(good())) == []


def write(tmp_path, items):
    (tmp_path / "calltypes.json").write_text(json.dumps(items, ensure_ascii=False))


def test_registry_loads_custom_and_reports_bad(tmp_path):
    write(tmp_path, [good(), good(id="broken", steps=[]), good(id="lead_qualification"), {"id": "BAD ID", "name": "x"}, good(id="p_only", project_id="acme")])
    ok, bad = load_custom(tmp_path)
    assert {c.id for c in ok} == {"my_call", "p_only"}
    assert "broken" in bad and any("at least one step" in e for e in bad["broken"])
    assert "lead_qualification" in bad and any("built-in" in e for e in bad["lead_qualification"])
    assert "BAD ID" in bad
    reg = CallTypeRegistry(tmp_path)
    ids = {c.id for c in reg.all("")}
    assert ids == BUILTIN_IDS | {"my_call"}
    assert {c.id for c in reg.all("acme")} == BUILTIN_IDS | {"my_call", "p_only"}
    assert reg.get("my_call").builtin is False and reg.get("nope") is None
    assert reg.get("p_only", "other") is None and reg.get("p_only", "acme") is not None  # another project's type is never handed out


def test_registry_file_shapes_and_broken_file(tmp_path):
    assert parse_custom({"calltypes": [good()]})[0]["id"] == "my_call"
    assert parse_custom({"my_call": {k: v for k, v in good().items() if k != "id"}})[0]["id"] == "my_call"
    (tmp_path / "calltypes.json").write_text("{not json")
    ok, bad = load_custom(tmp_path)
    assert ok == [] and "<file>" in bad
    assert CallTypeRegistry(tmp_path / "missing").all()[0].id == "sales_pitch"


def test_project_type_beats_projectless_one(tmp_path):
    write(tmp_path, [good(name="generic"), good(name="acme version", project_id="acme")])
    reg = CallTypeRegistry(tmp_path)
    assert reg.get("my_call", "acme").name == "acme version"
    assert reg.get("my_call", "").name == "generic"


def test_step_helpers():
    ct = builtin_call_types()["lead_qualification"]
    assert ct.step("budget").answer_type == "money" and ct.step("zzz") is None
    assert ct.line("opening", "hi").startswith("नमस्ते")
    assert Step(id="x", text="t").text["hi"] == "t"
