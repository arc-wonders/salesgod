"""SalesGod tenant (calls 1d5573e7dc, d3ac800811, 15 Nov): "sell me a pen" / "pitch yourself" / "give me a demo" / "one line in Hindi"
are done on the spot (enabled by the playbook's pitch_anything node, i.e. by data), never the unclear gate, the free-trial lever or the
product's own pitch; the pilot sign-up is the free pilot with a setup call; a time after the link question is a callback."""

import asyncio
import json
from pathlib import Path

import pytest

from acie import playbook as pbm
from acie.decision import garble
from acie.decision.domains.product_sales import Product, ProductSalesDomain
from acie.decision.engine import DecisionEngine
from acie.pipeline import ACIEPipeline
from acie.schemas import TurnInput, Utterance
from acie.validator import StreamingValidator

ROOT = Path(__file__).resolve().parents[1]
PRODUCTS = ROOT / "data/voice/products.json"
PB = ROOT / "data/voice/playbooks/salesgod/sales-god.json"
pytestmark = pytest.mark.skipif(not (PRODUCTS.exists() and PB.exists() and "sales-god" in json.loads(PRODUCTS.read_text())),
                                reason="needs the sales-god product and playbook")


class EchoLLM:
    async def stream(self, prompt, plan, expr):
        if plan.intent == "pitch_demo" and plan.llm_allowed:
            # a real model's pitch: 4 sentences, two questions (the question budget / 40-word trim must not cut it)
            yield ("Picture this pen: it writes the first time, every time, even on the page you were sure was cursed. When did your last pen "
                   "quit on you mid-signature? Keep this one in your pocket today and sign your next deal in style. That's exactly how I'd sell "
                   "your product, on every call.")
            return
        yield plan.fallback.get(expr.language) or plan.fallback.get("en")


def _call(lines, sid="sg1", lang="en", name="Yash"):
    p = Product.model_validate(json.loads(PRODUCTS.read_text())["sales-god"])
    pb = pbm.load(ROOT / "data/voice", "salesgod", "sales-god")
    pipe = ACIEPipeline(DecisionEngine(ProductSalesDomain(p, customer_name=name, playbook=pb)), backend=EchoLLM(), buffer_reply=True)
    T = lambda t: TurnInput(session_id=sid, customer_id="+910000000000", text=t)  # noqa: E731

    async def go():
        sess = pipe.session(T(""))
        sess.dialog["_lang"] = lang
        plan = await pipe.decision.decide("greet", T(""), sess)
        text = " ".join(StreamingValidator(plan, sess.expression).finish())
        sess.history.append(Utterance(role="agent", text=text))
        sess.context.record_turn(0, "", plan.intent, False)
        out = [("", text, plan)]
        for c in lines:
            r = await pipe.run_turn(T(c))
            out.append((c, r.text, r.plan))
        return out, sess
    return asyncio.run(go())


OPEN = ["Yeah, I'm doing good.", "Uh, yeah, sure."]


def test_rapport_greeting_has_no_double_space():
    out, _ = _call([], sid="g1")
    assert "  " not in out[0][1] and out[0][1].startswith("Hi Yash!")


@pytest.mark.parametrize("line,item", [("How you bitch? Bitch me a pen.", "pen"), ("Pitch me the bell.", "bell"),
                                       ("Buddy, I am saying if you are the sales god, sell me a pen.", "pen"),
                                       ("Buddy, pitch me the pen.", "pen"), ("pitch me this bottle", "bottle"),
                                       ("pitch me a gym membership", "gym membership")])
def test_pitch_anything_is_done_on_the_spot(line, item):
    out, _ = _call(OPEN + [line], sid="p-" + item.replace(" ", ""))
    c, t, p = out[-1]
    assert p.intent == "pitch_demo", (t, p.intent, p.decision)
    assert item in p.decision
    assert "couldn't catch" not in t and "line seems poor" not in t and "Nice tries" not in t
    assert len([x for x in t.split(". ") if x]) >= 3  # not trimmed to one line


def test_pitch_yourself_pitches_salesgod():
    out, _ = _call(OPEN + ["Okay, pitch yourself."], sid="self")
    assert out[-1][2].intent == "pitch_demo" and "yourself" in out[-1][2].decision


def test_demo_request_is_a_live_demo_not_the_free_trial_lever():
    out, _ = _call(OPEN + ["Buddy, give me a demo. I'm asking for a demo."], sid="demo")
    c, t, p = out[-1]
    assert p.intent == "pitch_demo" and "makes sense to want to look first" not in t and "7-day" not in t


def test_one_line_in_hindi_then_back_in_english():
    out, sess = _call(OPEN + ["i want a demo first can you just tell me how you quickly switch to hindi i just want a one line in hindi"], sid="hin")
    c, t, p = out[-1]
    assert p.intent == "pitch_demo" and any("ऀ" <= ch <= "ॿ" for ch in t) and "back in English" in t
    assert sess.dialog.get("_lang") == "en"


def test_transliterated_english_is_not_garbled():
    assert not garble.looks_garbled("ओके थैंक यू")
    assert not garble.looks_garbled("ओके थैंक यू नेक्स्ट")


def test_pilot_signup_is_the_free_pilot_and_a_setup_call():
    out, _ = _call(OPEN + ["Sign me up for the pilot.", "Tomorrow at 11 am"], sid="pilot")
    c, t, p = out[-2]
    assert p.intent == "pilot_signup" and "free" in t.lower() and "link" not in t.lower() and "14,999" not in t
    c, t, p = out[-1]
    assert p.intent == "callback" and "11 am" in t and "pilot" in t


def test_time_after_link_question_is_a_callback_not_consent():
    out, sess = _call(OPEN + ["What's the price?", "okay sounds good, I'll take it", "Tomorrow at 11 am"], sid="time")
    assert out[-2][2].intent == "confirm_close", out[-2]
    c, t, p = out[-1]
    assert p.intent == "callback" and "link_sent" != p.intent and "11 am" in t


def test_trust_question_mid_haggle_is_answered():
    out, _ = _call(OPEN + ["What's the price?", "Can you do 13,000 a month?", "Will it make up prices?"], sid="trust")
    c, t, p = out[-1]
    assert p.intent != "negotiation", (t, p.decision)


def test_second_pitch_uses_a_new_tie_back_and_ok_after_demo_is_not_a_price_yes():
    out, _ = _call(OPEN + ["sell me a pen", "pitch me this bottle", "ओके थैंक यू"], sid="two")
    assert out[-3][2].intent == out[-2][2].intent == "pitch_demo"
    assert "picture that energy" in out[-2][2].objective
    assert out[-1][2].intent != "price", out[-1]


def test_live_ddf5d69905_demo_plus_item_same_turn_and_current_item():
    out, _ = _call(OPEN + ["okay i want a demo uh do one thing pitch me a rubik's cube", "do you want to pitch me a pen", "pitch it again"], sid="ddf5")
    c, t, p = out[-3]
    assert p.intent == "pitch_demo" and "rubik's cube" in p.decision and "makes sense to want to look first" not in t
    c, t, p = out[-2]
    assert p.intent == "pitch_demo" and "'pen'" in p.decision and "rubik" not in p.objective.lower()
    assert len([x for x in t.split(". ") if x]) >= 3  # not trimmed
    assert "'pen'" in out[-1][2].decision
