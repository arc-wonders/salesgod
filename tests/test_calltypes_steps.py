"""Step-driven call types (payment reminder, feedback survey, appointment booking, operator-defined custom ones), the factory,
and the voice server hook that picks the domain by call.call_type."""

import json

import pytest

from acie.calltypes import CallType
from acie.calltypes.factory import build_domain
from acie.decision.domains.custom_steps import StepCallDomain, humanize
from acie.decision.domains.follow_up import FollowUpDomain
from acie.decision.domains.lead_qualification import LeadQualificationDomain
from acie.decision.domains.product_sales import Product, ProductSalesDomain
from tests.calltypes_helpers import Call, asked, run


# --------------------------------------------------------------------------- the factory

def test_factory_picks_the_pack(tmp_path):
    p = Product()
    assert type(build_domain(None, p)) is ProductSalesDomain
    assert type(build_domain("sales_pitch", p)) is ProductSalesDomain  # today's flow, unchanged
    assert type(build_domain("lead_qualification", p)) is LeadQualificationDomain
    assert type(build_domain("follow_up", p, data_dir=tmp_path)) is FollowUpDomain
    for cid in ("payment_reminder", "feedback_survey", "appointment_booking"):
        assert type(build_domain(cid, p)) is StepCallDomain
    assert type(build_domain("no_such_type", p)) is ProductSalesDomain  # a bad id never breaks the call


def test_the_domain_is_a_domain_pack():
    d = build_domain("lead_qualification", Product())
    assert d.name == "lead_qualification" and hasattr(d.classifier, "classify") and callable(d.plan)
    assert d.p.product_name == "Claude Pro"  # the pipeline reads .p (facts ladder, knowledge base)


# --------------------------------------------------------------------------- payment reminder

async def test_payment_reminder_will_pay_with_a_date():
    c = await run("payment_reminder", ["haan boliye", "haan main kal shaam 6 baje payment kar dunga"], lang="hinglish")
    assert c.result["answers"]["status"] == "will_pay"
    assert c.last.plan.intent == "step_ask" and c.result["outcome"] == "ended_early"  # asked for the date: it was not in the same sentence
    await c.say("kal shaam 6 baje")
    r = c.result
    assert r["outcome"] == "promised_to_pay" and r["next_action"] == {"type": "payment_followup", "due_at": "2026-11-16T18:00:00+05:30", "call_type": "payment_reminder"}
    assert "noted" in c.last.text or "note kar liya" in c.last.text and c.last.plan.end_call


async def test_payment_reminder_already_paid_skips_the_date():
    c = await run("payment_reminder", ["haan boliye", "payment ho gaya hai pehle hi"], lang="hinglish")
    assert c.result["outcome"] == "paid" and c.result["stage_suggestion"] == "won" and c.last.plan.end_call
    assert asked(c, "payment_reminder", "pay_date") == 0


async def test_payment_reminder_dispute_goes_to_a_person():
    c = await run("payment_reminder", ["haan boliye", "ye galat hai, mera nahi hai ye payment"], lang="hinglish")
    assert c.result["outcome"] == "human_requested" and c.last.plan.end_call


async def test_payment_reminder_never_sells():
    c = await run("payment_reminder", ["haan boliye", "kya mujhe koi offer milega?"], lang="hinglish")
    assert c.last.plan.intent in ("answer", "step_ask") and "pitch" not in c.intents


# --------------------------------------------------------------------------- feedback survey

async def test_feedback_survey_rating_recommend_comment():
    c = await run("feedback_survey", ["haan boliye", "mein isko 4 rating dunga", "haan recommend karunga", "support thoda slow tha"], lang="hinglish")
    r = c.result
    assert r["answers"] == {"rating": 4, "recommend": True, "comment": "support thoda slow tha"}
    assert r["complete"] and r["outcome"] == "completed" and c.last.plan.end_call and r["score_delta"] == 0


async def test_feedback_optional_comment_is_asked_only_once():
    c = await run("feedback_survey", ["haan boliye", "5 stars", "yes", "hmm"], lang="hinglish")
    assert c.result["complete"] and c.result["answers"]["comment"] is None
    assert asked(c, "feedback_survey", "comment") == 1


# --------------------------------------------------------------------------- appointment booking

@pytest.mark.parametrize("lang,lines", [
    ("en", ["yes", "a haircut", "Friday at 3:30 pm"]),
    ("hinglish", ["haan boliye", "mujhe check up karana hai", "kal subah 10 baje"]),
    ("hi", ["हाँ बोलिए", "मुझे चेकअप करवाना है", "कल सुबह 10 बजे"]),
])
async def test_appointment_booking(lang, lines):
    c = await run("appointment_booking", lines, lang=lang)
    r = c.result
    assert r["outcome"] == "appointment_booked" and r["stage_suggestion"] == "demo_booked" and r["complete"]
    assert r["next_action"]["type"] == "book_appointment" and r["next_action"]["due_at"] in ("2026-11-20T15:30:00+05:30", "2026-11-16T10:00:00+05:30")
    assert set(r["answers"]) == {"purpose", "slot"}


# --------------------------------------------------------------------------- operator-defined custom type

CUSTOM = {
    "id": "renewal_check", "name": "Renewal check", "objective": "see if they will renew",
    "opening": {"en": "Hello {name}, this is {agent} from {seller} about your renewal.", "hi": "नमस्ते {name}, {seller} से {agent}, आपके रिन्यूअल के बारे में।",
                "hinglish": "Namaste {name}, {seller} se {agent}, aapke renewal ke baare mein."},
    "steps": [
        {"id": "renew", "answer_type": "yes_no", "text": {"en": "Will you renew?", "hi": "क्या आप रिन्यू करेंगे?", "hinglish": "Kya aap renew karenge?"}, "score": {"yes": 10}},
        {"id": "seats", "answer_type": "number", "skip_if": "renew == no", "keywords": ["seats", "users"],
         "text": {"en": "How many seats?", "hi": "कितनी सीट?", "hinglish": "Kitni seats?"}},
        {"id": "plan", "answer_type": "choice", "skip_if": "renew == no",
         "text": {"en": "Monthly or yearly?", "hi": "मासिक या सालाना?", "hinglish": "Monthly ya yearly?"},
         "choices": [{"id": "monthly", "keywords": ["monthly", "mahine"]}, {"id": "yearly", "keywords": ["yearly", "saalana", "annual"]}]},
        {"id": "referral", "answer_type": "yes_no", "skip_if": "seats < 5", "required": False,
         "text": {"en": "Any referral?", "hi": "कोई रेफ़रल?", "hinglish": "Koi referral?"}},
    ],
    "allowed_actions": ["transfer_human"], "success_outcomes": ["completed"], "closing": "Thanks!",
}


@pytest.fixture
def custom_dir(tmp_path):
    (tmp_path / "calltypes.json").write_text(json.dumps([CUSTOM], ensure_ascii=False))
    return tmp_path


async def test_custom_type_runs_any_steps_and_skip_if(custom_dir):
    c = Call("renewal_check", data_dir=custom_dir, lang="en")
    plan = await c.open()
    assert "Hello Yash, this is Arjun from Anthropic about your renewal." in plan.fallback["en"]
    for line in ["yes go on", "yes", "we have 8 users", "yearly please", "no referral"]:
        await c.say(line)
    r = c.result
    assert r["answers"] == {"renew": True, "seats": 8, "plan": "yearly", "referral": False} and r["complete"] and r["call_type"] == "renewal_check" and r["score_delta"] == 10
    assert asked_text(c, "Will you renew?") == 1


def asked_text(c, needle):
    return sum(needle in t for t in c.agent)


async def test_custom_skip_if_skips_the_branch(custom_dir):
    c = Call("renewal_check", data_dir=custom_dir, lang="en")
    await c.open()
    for line in ["yes go on", "no, not renewing"]:
        await c.say(line)
    assert c.result["answers"] == {"renew": False} and asked_text(c, "How many seats?") == 0 and asked_text(c, "Monthly or yearly?") == 0
    assert "Any referral?" in c.last.text  # (seats unanswered: "seats < 5" cannot be decided, so the optional step is still asked)
    c2 = Call("renewal_check", data_dir=custom_dir, lang="en")
    await c2.open()
    for line in ["yes go on", "yes", "just 3 users", "monthly"]:
        await c2.say(line)
    assert c2.result["complete"] and asked_text(c2, "Any referral?") == 0  # skipped: seats < 5


async def test_custom_early_answer_by_keyword(custom_dir):
    c = Call("renewal_check", data_dir=custom_dir, lang="en")
    await c.open()
    for line in ["yes go on", "yes, with 10 users"]:
        await c.say(line)
    assert c.result["answers"]["seats"] == 10 and asked_text(c, "How many seats?") == 0 and "Monthly or yearly?" in c.last.text


async def test_operator_opening_without_ai_disclosure_gets_it_added(custom_dir):
    c = Call("renewal_check", data_dir=custom_dir, lang="en")
    plan = await c.open()
    assert "AI assistant" in plan.fallback["en"] and "recorded" in plan.fallback["en"]
    assert "AI" in plan.fallback["hi"] and "रिकॉर्ड" in plan.fallback["hi"]
    assert "AI assistant" in plan.fallback["hinglish"] and "record" in plan.fallback["hinglish"]


async def test_compliance_lines_are_spoken_after_the_opening(tmp_path):
    ct = {**CUSTOM, "id": "with_lines", "compliance_lines": [{"en": "You can ask us to stop at any time.", "hi": "आप कभी भी रोक सकते हैं।", "hinglish": "Aap kabhi bhi rok sakte hain."}]}
    (tmp_path / "calltypes.json").write_text(json.dumps([ct], ensure_ascii=False))
    c = Call("with_lines", data_dir=tmp_path, lang="en")
    plan = await c.open()
    assert plan.fallback["en"].endswith("You can ask us to stop at any time.")


async def test_a_broken_custom_type_falls_back_to_the_sales_call(tmp_path):
    (tmp_path / "calltypes.json").write_text(json.dumps([{**CUSTOM, "steps": []}]))
    assert type(build_domain("renewal_check", Product(), data_dir=tmp_path)) is ProductSalesDomain


# --------------------------------------------------------------------------- safety on every pack

@pytest.mark.parametrize("call_type", ["payment_reminder", "feedback_survey", "appointment_booking", "lead_qualification", "follow_up"])
async def test_opt_out_and_crisis_on_every_type(call_type, tmp_path):
    c = Call(call_type, data_dir=tmp_path, lang="en")
    await c.open()
    await c.say("don't call me again")
    assert c.last.plan.intent == "opt_out" and c.last.plan.end_call and c.result["outcome"] == "opt_out"
    c2 = Call(call_type, data_dir=tmp_path, lang="hinglish")
    await c2.open()
    await c2.say("mujhe marna hai")
    assert c2.last.plan.intent == "safety_distress"


async def test_humanize_three_languages():
    iso = "2026-11-16T18:00:00+05:30"
    assert humanize(iso, "en") == "Monday 16 Nov at 6 pm"
    assert humanize(iso, "hi") == "सोमवार 16 नवंबर, शाम 6 बजे"
    assert humanize("2026-11-16T09:30:00+05:30", "hinglish") == "Somvar 16 Nov, subah 9:30 baje"


# --------------------------------------------------------------------------- the voice server hook

def test_make_pipeline_picks_the_domain_by_call_type(monkeypatch, tmp_path):
    import os
    for k, v in {"PLIVO_AUTH_ID": "x", "PLIVO_AUTH_TOKEN": "x", "PLIVO_NUMBER": "+910000000000", "ADMIN_PASSWORD": "pw", "SESSION_SECRET": "s"}.items():
        os.environ.setdefault(k, v)
    pytest.importorskip("fastapi")
    try:
        from voice import server as srv
        from voice.call import CallRecord
    except ImportError as e:
        pytest.skip(f"voice server deps missing: {e}")
    from acie.expression import TemplateBackend
    app = srv.STATE
    monkeypatch.setattr(srv, "DATA", tmp_path)
    monkeypatch.setattr(app, "products", lambda: {"claude-pro": Product()}, raising=False)
    monkeypatch.setattr(app, "backends", {"template": TemplateBackend()})
    rec = CallRecord(to="+910000000001", campaign="claude-pro", backend="template")
    assert type(app.make_pipeline(rec)[0].decision.domain) is ProductSalesDomain  # no call_type: today's behaviour
    rec.call_type = "sales_pitch"
    assert type(app.make_pipeline(rec)[0].decision.domain) is ProductSalesDomain
    rec.call_type = "lead_qualification"
    assert type(app.make_pipeline(rec)[0].decision.domain) is LeadQualificationDomain
    rec.call_type = "follow_up"
    assert type(app.make_pipeline(rec)[0].decision.domain) is FollowUpDomain
    rec.call_type = "garbage"
    assert type(app.make_pipeline(rec)[0].decision.domain) is ProductSalesDomain
