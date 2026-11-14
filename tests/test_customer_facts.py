"""CustomerFacts ledger + the deterministic "no repeating context" guard (psyche/ledger.py, pipeline._facts_guard)."""

import json

from decision_helpers import T, run

from acie.decision.domains.product_sales import Product
from acie.psyche.ledger import (CustomerFacts, SaidLedger, clear_input, explicit_bye, firm_no, invented_memory, is_repeat_ask, next_unsaid,
                                one_word_fragment, pure_farewell, unsourced_claims)


def test_caller_facts_are_learned_and_survive_json():
    f = CustomerFacts()
    f.learn_caller("I want to focus on building muscle, around four to five days a week, I am a beginner")
    assert f.knows("goal") and f.knows("schedule") and f.knows("experience")
    d: dict = {}
    f.save(d)
    g = CustomerFacts.load(d)
    assert g.caller == f.caller and json.loads(d["_cf"]) == json.loads(json.dumps(json.loads(d["_cf"]), sort_keys=True))


def test_deterministic_serialisation():
    out = []
    for _ in range(2):
        f = CustomerFacts()
        f.learn_caller("muscle aur weight loss, 5 din, budget kam hai")
        d: dict = {}
        f.save(d)
        out.append(d["_cf"])
    assert out[0] == out[1]


def test_restating_the_callers_facts_is_detected():
    f = CustomerFacts()
    f.learn_caller("I want to focus on building muscles and the aesthetics of my body")
    assert f.restates("It will help with muscle and aesthetics.")
    assert not f.restates("You also get a free personal trainer session every week.")
    h = CustomerFacts()
    h.learn_caller("मैं कॉलेज में पढ़ता हूँ और कोडिंग सीखता हूँ")
    assert h.restates("आप कॉलेज में कोडिंग सीखते हैं, तो ये काम आएगा।")


def test_reasking_an_answered_question_is_detected():
    f = CustomerFacts()
    f.learn_caller("mujhe muscle banana hai")
    assert f.reasks("Aapka main goal kya hai?") == ["goal"]
    assert not f.reasks("Kya main price bata doon?")


def test_next_unsaid_skips_said_points_and_never_runs_dry():
    p = Product()
    led, facts = SaidLedger(), CustomerFacts()
    seen = []
    for _ in range(12):
        slot, line = next_unsaid(p, "en", led, facts, "what else")
        assert line and not led.repeats(line)
        seen.append(line)
        led.add_text(line)
        if slot:
            facts.said_slots.append(slot)
    assert len(set(seen)) >= 4  # benefits first, then new forward steps


def test_next_unsaid_is_same_language_and_male():
    p = Product()
    for lang in ("hi", "hinglish"):
        led = SaidLedger()
        for _ in range(8):
            _, line = next_unsaid(p, lang, led, None, "")
            assert "sakti" not in line and "ungi" not in line.lower()
            led.add_text(line)


def test_helpers():
    assert one_word_fragment("मुझे") and one_word_fragment("mujhe") and one_word_fragment("main") and not one_word_fragment("aur?")
    assert is_repeat_ask("sorry, say that again") and not is_repeat_ask("tell me about the price")
    assert explicit_bye("thanks, bye") and firm_no("not interested") and firm_no("nahi nahi") and not firm_no("beta mujhe link kholna nahi aata")
    assert pure_farewell("ok bye") and not pure_farewell("ok bye but what about the price")
    assert clear_input("weight loss karna hai mujhe") and not clear_input("can't hear... go on")
    assert invented_memory("We have your details from your previous inquiry.")
    assert unsourced_claims("You can cancel any time.", "price is 8 dollars") == ["cancel any time"]
    assert not unsourced_claims("You can cancel any time.", "cancel any time is allowed")
    assert unsourced_claims("Panch guna jyada usage milta hai.", "usage") and unsourced_claims("We have the best models.", "usage")


async def test_goal_is_never_asked_again_and_not_echoed():
    out, last, pipe, dom, _ = await run(["yes go on", "I want to focus on building muscles and the aesthetics of my body", "so what will I get?", "ok"],
                                       Product(product_name="FitZone", seller_name="FitZone"))
    sess = pipe.sessions["s"]
    f = CustomerFacts.load(sess.dialog)
    assert f.knows("goal")
    agent = [u.text for u in sess.history if u.role == "agent"]
    assert not any("help with muscle" in a.lower() or "aesthetics" in a.lower() for a in agent)
    assert not any("didn't quite get" in a for a in agent)  # clear words are never "I didn't catch that"


async def test_same_sentence_is_never_said_twice_across_turns():
    _, _, pipe, _, _ = await run(["yes", "who made it?", "who made it?", "where is it from?", "and what else?", "what else", "anything else?"])
    agent = [u.text for u in pipe.sessions["s"].history if u.role == "agent"][1:]
    led = SaidLedger()
    for a in agent:
        assert not led.reply_repeats(a), a
        led.add_text(a)


async def test_one_word_fragment_gets_ji_bataiye():
    _, last, _, _, _ = await run(["yes", "mujhe"])
    assert last.plan.intent == "acknowledge" and "bataiye" in last.text.lower() or "please go on" in last.text.lower()
    assert "$" not in last.text and "₹" not in last.text


async def test_link_sent_does_not_end_the_call_on_a_filler():
    out, last, _, _, _ = await run(["yes", "yes", "yes", "yes please send the link", "ok"])
    assert not last.plan.end_call


async def test_goodbye_still_ends_the_call():
    out, last, _, _, _ = await run(["yes", "not interested, bye"])
    assert last.plan.end_call


# ---------------------------------------------------------------- discovery ladder + no same phrasing

def _gym():
    import json
    from pathlib import Path
    raw = json.loads((Path(__file__).resolve().parents[1] / "data/voice/products.json").read_text())
    return Product.model_validate(raw["fitzone-gym"])


def test_ladder_only_goes_deeper_in_every_language():
    p = _gym()
    f = CustomerFacts()
    f.use_ladder(p)
    assert "goal" in (f.next_discovery(p, "en") or "").lower()
    f.learn_caller("I want to build muscle")
    assert f.just_answered == ["goal"]
    q = f.next_discovery(p, "en")
    assert "aesthetic" in q and "goal" not in q.lower()
    assert "look" in f.next_discovery(p, "hinglish") and "लिए" in f.next_discovery(p, "hi")
    f.learn_caller("aesthetics mostly")
    assert "starting out" in f.next_discovery(p, "en")
    f.learn_caller("I am a beginner")
    assert "days" in f.next_discovery(p, "en")
    f.learn_caller("4 days a week, evenings")
    assert f.next_discovery(p, "en") is None


def test_ladder_question_never_asked_twice_and_survives_reload():
    p = _gym()
    f = CustomerFacts()
    f.use_ladder(p)
    f.learn_caller("weight loss")
    f.learn_agent(f.next_discovery(p, "en"))
    d: dict = {}
    f.save(d)
    g = CustomerFacts.load(d)
    g.use_ladder(p)
    assert g.knows("goal") and g.asked.get("lvl:subgoal") == 1
    assert g.reasks("What's your fitness goal: losing weight, building muscle, or just staying fit?")


def test_default_ladder_from_discovery_question_and_none():
    class P:
        discovery_question = {"en": "What will you use it for?", "hi": "किसलिए?", "hinglish": "Kis liye?"}
    f = CustomerFacts()
    assert f.next_discovery(P(), "en") == "What will you use it for?"
    assert CustomerFacts().next_discovery(object(), "en") is None


def test_filler_is_rotated_not_repeated():
    f = CustomerFacts()
    f.learn_agent("Bilkul, main bata deta hoon.")
    out = f.rotate_opener("Bilkul, ye free hai.", "hinglish")
    assert not out.lower().startswith("bilkul") and "free hai" in out
    g = CustomerFacts()
    g.learn_agent("Sure, here it is.")
    assert not g.rotate_opener("Sure, it is free.", "en").startswith("Sure")
    h = CustomerFacts()
    h.learn_agent("बिल्कुल, बताता हूँ।")
    assert not h.rotate_opener("बिल्कुल, ये फ्री है।", "hi").startswith("बिल्कुल")
    assert CustomerFacts().rotate_opener("Bilkul, ye free hai.", "hinglish").startswith("Bilkul")  # first use is allowed


def test_prompt_block_lists_used_phrases():
    f = CustomerFacts()
    f.learn_agent("Koi baat nahi. Ye hai.")
    assert "koi baat nahi" in f.prompt_block()


async def test_gym_call_asks_the_next_level_not_the_goal_again():
    _, _, pipe, _, _ = await run(["yes", "I want to build muscle", "ok", "what else", "hmm"], _gym())
    agent = [u.text.lower() for u in pipe.sessions["s"].history if u.role == "agent"][2:]
    assert not any("fitness goal" in a for a in agent)
