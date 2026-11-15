"""Conversation-flow fixes from the owner's calls f79a90af02 ("too rude") and c9960ed17d (15 Nov, FitZone, English, Gemini):
price only after interest, a way forward after an apology, no invented situation, a vague positive is not consent, an unheard
opening + a grunt restates the purpose, no fragment / doubled word from trimming, small talk acknowledged in the call's language,
a repeated opener stripped (not the whole reply), a buying signal / "pay offline" routed by the decision tree, decisions visible."""

import asyncio
import json
import re
from pathlib import Path

import pytest

from acie import playbook as pbm
from acie.decision.domains.product_sales import Product, ProductSalesDomain
from acie.decision.engine import DecisionEngine
from acie.pipeline import ACIEPipeline, bare_apology
from acie.schemas import ExpressionParams, Fact, TurnInput, TurnPlan, Utterance
from acie.trace import gate_of, trace
from acie.understanding import Understanding
from acie.validator import StreamingValidator

ROOT = Path(__file__).resolve().parents[1]
PRODUCTS = ROOT / "data/voice/products.json"
pytestmark = pytest.mark.skipif(not PRODUCTS.exists(), reason="needs data/voice/products.json")


def _gym() -> Product:
    return Product.model_validate(json.loads(PRODUCTS.read_text())["fitzone-gym"])


class FakeLLM:
    """Replies with `next` (a scripted model reply) or the plan's template."""
    def __init__(self):
        self.next = None

    async def stream(self, prompt, plan, expr):
        yield self.next if self.next is not None else (plan.fallback.get(expr.language) or plan.fallback.get("en"))


class FakeNLU:
    def __init__(self):
        self.next = None

    async def understand(self, text, history=None, intents=None, about=None):
        return Understanding(**self.next) if self.next else None


def _call(steps, sid="flow1", cut_opening=None):
    """steps: [(customer text, llm reply or None, nlu dict or None)] -> [(text, plan, trace)], session."""
    llm, nlu = FakeLLM(), FakeNLU()
    pb = pbm.load(ROOT / "data/voice", "fitzone", "fitzone-gym")
    pipe = ACIEPipeline(DecisionEngine(ProductSalesDomain(_gym(), customer_name="Asha", playbook=pb)), backend=llm, buffer_reply=True, nlu=nlu)
    T = lambda t: TurnInput(session_id=sid, customer_id="+910000000000", text=t)  # noqa: E731

    async def go():
        sess = pipe.session(T(""))
        plan = await pipe.decision.decide("greet", T(""), sess)
        text = " ".join(StreamingValidator(plan, sess.expression).finish())
        sess.history.append(Utterance(role="agent", text=text))
        sess.context.record_turn(0, "", plan.intent, False)
        if cut_opening is not None:
            n = int(len(text.split()) * cut_opening)
            pipe.apply_heard(sid, " ".join(text.split()[:n]), cut_opening, "greet")
        out = []
        for c, reply, sig in steps:
            llm.next, nlu.next = reply, sig
            r = await pipe.run_turn(T(c))
            out.append((r.text, r.plan, trace(r, sess.dialog)))
        return out, sess
    return asyncio.run(go())


def _questions(text):
    return sum(x.rstrip().endswith("?") for x in re.split(r"(?<=[.!?])\s+", text))


AFFIRM = {"intent": "affirm", "buying_signal": 0.85}
RUDE = ("I think you are being a bit too rude by just suddenly giving me all the price petitions and all. "
        "Did you even ask me if I want to join or not?")


# 1 ------------------------------------------------------------------ price only after interest / permission, one thing per turn
def test_love_it_after_pitch_is_not_a_price_dump_and_price_never_glued_to_discovery():
    out, _ = _call([("Uh, sure.", None, None), ("Love it.", None, AFFIRM), ("Sure.", None, None)], sid="f1")
    assert "₹" not in out[1][0] and _questions(out[1][0]) <= 1, out[1][0]
    price = out[2][0]
    assert "₹2,999" in price and not re.search(r"fitness goal|main goal|losing weight", price, re.I), price
    assert _questions(price) <= 1


# 2 ------------------------------------------------------------------ apology + a way forward; situations only from the caller's words
def test_situation_comes_only_from_what_the_caller_said():
    pb = pbm.load(ROOT / "data/voice", "fitzone", "fitzone-gym")
    node, _ = pb.pick(RUDE, Understanding(intent="unknown"), "price")
    assert node is None or node.id != "weight_loss"           # "petitions" is not "pet" (belly)
    assert pb.pick("I want to lose some weight", None, "")[0].id == "weight_loss"
    assert pb.pick("I see.", Understanding(intent="tell_more"), "tell_more")[0] is None  # an intent label alone is no situation
    assert pb.pick("Is there any benefit for me?", None, "benefits")[0].id == "why_join"
    assert pb.pick("that would close the deal", None, "")[0] is None   # "lose" is not inside "close"


def test_complaint_gets_own_it_and_a_way_forward_never_a_bare_apology():
    out, _ = _call([("Uh, sure.", None, None), ("Love it.", None, AFFIRM), (RUDE, None, None)], sid="f2")
    a = out[2][0]
    assert re.search(r"sorry|you're right", a, re.I) and a.rstrip().endswith("?") and "₹" not in a, a
    # an LLM reply that is only an apology gets a way forward
    out, _ = _call([("Uh, sure.", None, None), ("Hmm, that was a lot to take in at once.", "I am so sorry, I did not mean to sound rude at all.", None)], sid="f2b")
    assert out[1][0].rstrip().endswith("?") and "slow down" in out[1][0], out[1][0]
    assert bare_apology(["I am so sorry, I did not mean to sound rude at all."])
    assert not bare_apology(["Sorry about that. What would you like to know?"])


# 3 ------------------------------------------------------------------ a vague positive right after a complaint is not purchase consent
def test_okay_love_it_after_complaint_is_not_a_close():
    out, sess = _call([("Uh, sure.", None, None), ("Love it.", None, AFFIRM), (RUDE, None, None), ("Okay. Love it.", None, AFFIRM)], sid="f3")
    text, plan, _ = out[3]
    assert plan.intent not in ("confirm_close", "link_sent", "confirm_buy") and "₹" not in text and "link" not in text.lower(), text
    assert sess.dialog.get("terms_confirmed") != "1"


# 4 ------------------------------------------------------------------ an unheard opening + a grunt: say why we called, never "what's holding you back"
def test_grunt_after_cut_opening_restates_the_purpose():
    out, _ = _call([("Ugh.", None, {"intent": "deny", "buying_signal": 0.1})], sid="f4", cut_opening=0.23)
    text, plan, tr = out[0]
    assert plan.intent == "purpose" and "from FitZone" in text and "recorded" in text and "offer" in text and text.rstrip().endswith("?"), text
    assert "holding you back" not in text and tr["gate"] == "scripted" and tr["decision"]


# 5 ------------------------------------------------------------------ trimming never leaves a fragment or a doubled word
def test_slot_edge_word_is_never_doubled():
    plan = TurnPlan(intent="answer", objective="x", llm_allowed=True, fallback={"en": "x"},
                    facts={"product": Fact(value="FitZone membership", description="product", must_mention=False)})
    v = StreamingValidator(plan, ExpressionParams(language="en"))
    out = v.feed("With a {product} membership, you get one free trainer session every week. ") + v.finish()
    assert out[0].startswith("With a FitZone membership, you get") and "membership membership" not in " ".join(out)


def test_clause_cut_never_returns_a_fragment_and_short_mode_keeps_whole_sentences(monkeypatch):
    s = "With a FitZone membership, you get gym floor access, cardio and weights, group classes and a free trainer session every week."
    assert ACIEPipeline._clause_cut(s, 6) != "With a FitZone membership."
    assert ACIEPipeline._fragment("With a FitZone membership") and not ACIEPipeline._fragment("At FitZone, you get yoga classes")
    pipe = ACIEPipeline(DecisionEngine(ProductSalesDomain(_gym())))
    monkeypatch.delenv("ACIE_LENGTH_CAP", raising=False)
    from acie.psyche import style as S
    plan = TurnPlan(intent="answer", objective="x", llm_allowed=True, fallback={"en": "x"})
    confused = {S.STYLE_KEY: S.Style("brief", "simplify", "", False, row=3).to_json()}
    assert pipe._length_cap(plan, "Jim.", confused) is None   # a garbled / confused line is never trimmed (owner: cap off)
    rude = {S.STYLE_KEY: S.Style("calm", "calm_answer_offer_stop", "", False, row=2).to_json()}
    cap = pipe._length_cap(plan, "ok", rude)
    assert cap is not None and cap["whole"]
    out = pipe._fit_length(s + " The gym is open from 5 am to 11 pm. It also has lockers.", cap, {"w": 0, "s": 0})
    assert out.startswith(s)                                   # whole sentences only: no clause cut, no fragment


def test_trim_is_reported_only_when_something_was_trimmed():
    long_q = ("With {product}, you get one free personal trainer session every week to help with your workout form and your plan. "
              "Would you like to hear about the offer we have running for new members this week?")
    out, _ = _call([("Uh, sure.", None, None), ("What do I get?", long_q, None)], sid="f5")
    issues = out[1][2]["issues"]
    assert not any(re.search(r"trimmed reply from (\d+) to \1 ", i) for i in issues), issues


# 6 ------------------------------------------------------------------ small talk: the call's language, and a human acknowledgement as the fallback
def test_small_talk_fallback_acknowledges_the_question_not_a_random_fact():
    roman_hindi = "Main toh bas ek AI assistant hoon aur bilkul theek hoon, aap batao aapka din kaisa chal raha hai."
    out, _ = _call([("Uh, sure.", None, None), ("How are you today? What are you up to?", roman_hindi, None)], sid="f6")
    text = out[1][0]
    assert re.search(r"doing well|thanks for asking", text, re.I) and not re.search(r"gym|trainer|5 am|zumba", text, re.I), text
    dom = ProductSalesDomain(_gym())
    line = dom._small_talk_line("How are you? Is the weather good there? What is the time right now?")["en"]
    assert "doing well" in line and "weather" in line and "time" in line


def test_english_call_prompt_forbids_hindi():
    from acie.compiler import LANGUAGE_RULES
    assert "Hindi" in LANGUAGE_RULES["en"] and "only" in LANGUAGE_RULES["en"].lower()


# 7 ------------------------------------------------------------------ a repeated opener is stripped, the reply kept
def test_repeated_opener_keeps_the_reply():
    steps = [("Uh, sure.", None, None),
             ("I like running in the park.", "I am glad you enjoy running, and the cardio floor has treadmills for rainy days.", None),
             ("Sad man.", "I am really sorry to hear that you are feeling down. We can talk another day if you like.", None)]
    out, _ = _call(steps, sid="f7")
    text = out[2][0]
    assert "didn't quite get that" not in text and "feeling down" in text, text


# 8 ------------------------------------------------------------------ a clear buying signal goes down the close branch, deterministically
def test_take_my_money_pay_at_the_gym_is_a_desk_close():
    buy = "Just do one thing. Take my money. I'm coming to your gym to pay the money. 2999 was it? Yes, I'm open to coming to your gym. Thank you."
    out, sess = _call([("Uh, sure.", None, None), ("Probably building muscle for me.", None, None),
                       (buy, "Yes, that is correct, it is ₹2,999 for three months, walk into the front desk.", {"intent": "affirm", "buying_signal": 0.9})], sid="f8")
    text, plan, tr = out[2]
    assert plan.intent == "pay_at_desk" and "₹2,999" in text and "front desk" in text and "link" not in text.lower(), text
    assert tr["gate"] == "close" and tr["decision"] and sess.dialog.get("outcome") == "accepted"


def test_buying_signal_without_payment_method_confirms_terms():
    out, _ = _call([("Uh, sure.", None, None), ("Okay, sign me up.", None, None)], sid="f8b")
    text, plan, _ = out[1]
    assert plan.intent == "confirm_close" and "₹2,999" in text and "₹6,000" in text, text


def test_caller_checking_our_price_gets_it_from_the_locked_phrase_not_a_random_fact():
    out, _ = _call([("Uh, sure.", None, None), ("Yes tell me the price.", None, None), ("Do you have showers?", None, None),
                    ("2999 was it?", "Yes, that is correct, it is ₹2,999 for three months under this offer.", None)], sid="f8c")
    text, plan, tr = out[3]
    assert text.startswith("Yes, it's ₹2,999 for 3 months") and not tr["fallback"], text


def test_names_confirmed_price_only_for_our_own_numbers():
    dom = ProductSalesDomain(_gym())
    assert dom._names_confirmed_price("2999 was it?") and not dom._names_confirmed_price("would you do 2500?")


# 9 ------------------------------------------------------------------ a no / prefer offline at the confirmation is never a link consent
def test_no_prefer_offline_at_confirmation_is_desk_close_not_link():
    out, sess = _call([("Uh, sure.", None, None), ("Okay, sign me up.", None, None),
                       ("No, I would prefer paying it offline. Thank you.", None, AFFIRM)], sid="f9")
    text, plan, _ = out[2]
    assert plan.intent == "pay_at_desk" and "send you the payment link" not in text and "front desk" in text, text
    assert not sess.context.has("payment_link") and sess.context.decision == "accepted" and sess.context.outcome == "accepted"
    nxt, _ = _call([("Uh, sure.", None, None), ("Okay, sign me up.", None, None), ("No.", None, AFFIRM)], sid="f9b")
    assert nxt[2][1].intent != "link_sent"


# 10 ----------------------------------------------------------------- the tree drives and every reply shows its decision
def test_every_reply_carries_a_decision_and_the_playbook_node():
    steps = [("Uh, sure.", None, None), ("Is there any benefit for me?", "You get group classes and a free trainer session every week.", None),
             ("Probably it would be building muscle for me.", "Then the weights area with squat racks is for you.", None)]
    out, _ = _call(steps, sid="f10")
    assert out[1][2]["pb_node"] == "why_join" and out[1][2]["decision"].startswith("playbook situation")
    assert out[2][2]["pb_node"] == "muscle"
    for text, plan, tr in out[1:]:
        assert tr["decision"], (text, tr)
    assert gate_of("reason_ask") == "exit" and gate_of("pay_at_desk") == "close" and gate_of("purpose") == "scripted"


# ------------------------------------------------------------------ concert call fixes (15 Nov): safety false positive, quantities, booking
def test_metro_station_question_is_not_a_crisis_but_train_method_still_is():
    from acie import safety
    assert safety.distress_score("अच्छा, गेट 4 बजे खुलेगा तो शो शुरू कितने बजे होगा? और पास में कौन सा मेट्रो स्टेशन है?", [], None)[0] == 0
    assert safety.distress_score("metro station kaunsa hai paas mein?", [], None)[0] == 0
    assert safety.distress_score("मैं ट्रेन के आगे कूद जाऊँगा", [], None)[0] >= 0.85
    assert safety.distress_score("main train ke aage aa jaunga", [], None)[0] >= 0.85
    assert safety.distress_score("मैं मरना चाहता हूँ", [], None)[0] >= 0.9


def test_quantities_are_not_money_and_no_discount_is_fine_is_not_a_push():
    from acie.decision import negotiation as n
    assert n.counter_number("par kya hum 10 log ek saath khade ho payenge?") is None
    assert n.detect("Mujhe 10 tickets book karne hain team outing ke liye") != "counter_offer"
    assert n.detect("2500 mein de do") == "counter_offer" and n.counter_number("4 tickets lene hain, ₹3000 per ticket lagao") == 3000
    assert not n.explicit_push("Discount nahi hai toh thik hai, par kya hum 10 log ek saath khade ho payenge?")
    assert n.explicit_push("thoda discount do na")


def test_how_do_i_book_goes_to_the_close_not_the_haggle_line():
    from acie.decision.domains import product_sales as P
    for t in ("Awesome, Gold sounds perfect for my budget! How do I book the Gold tickets for Delhi?", "Gold ka ticket kaise book karun?"):
        assert P._BUY_SIGNAL.search(t) and not P._BUY_NEG.search(t), t
    out, _ = _call([("Uh, sure.", None, None), ("Great, how do I book it?", None, None)], sid="cb1")
    assert out[1][1].intent == "confirm_close" and "difficult" not in out[1][0], out[1][0]
