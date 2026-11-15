"""The live FitZone haggle call (team/transcript_fitzone_haggle.txt), replayed line for line through the whole pipeline with a fake NLU
that hears what the live model heard. Each bug from that call is impossible now:
  1. a counter ("Harvard, 2,800" = "How about 2,800") is never consent to the list price;
  2. an agreed price is confirmed and the call moves to close (never "Done, X. If I make it X, will you join?");
  3. concessions are monotonic, asked for, never below the customer's counter / an agreed price / the floor, and they shrink;
  4. "where were we at the negotiation price?" restates the deal, never concedes;
  5. the confirm line and the link carry the agreed price, and the offer terms never repeat "if you join this week";
  6. with SMS off, the agent never says it is sending an SMS.
Plus: Hindi / Hinglish counters, the agreed price surviving a context restore (serialise + reload), and a property sweep of the rules."""
import asyncio
import json
import random
import re

import pytest

from acie.decision import deal as dl
from acie.decision import negotiation
from acie.decision.domains.product_sales import Product, ProductSalesDomain
from acie.decision.engine import DecisionEngine
from acie.pipeline import ACIEPipeline
from acie.schemas import ExpressionParams, SessionState, TurnInput

RAW = json.load(open("data/voice/products.json"))
GYM = Product.model_validate(RAW["fitzone-gym"])
FLOOR = dl.config_for(GYM).floor
PRICE = re.compile(r"₹\s?([\d,]+)")


class FakeNLU:
    """Hears what the live model heard on that call (the counter "Harvard, 2,800." came back as a confident 'affirm')."""
    def __init__(self, table):
        self.table = table

    async def understand(self, text, history, intents, about=""):
        from acie.understanding import Understanding
        return Understanding(**self.table[text]) if text in self.table else None


# the customer's lines, exactly as transcribed on the live call
C_GOAL = "So what is my benefit of joining the gym?"
C_2700 = ("the price point is a little bit more for me as i want to test the gym out first, i want to negotiate the price a bit, "
          "can you make it roughly around 2700")
C_2800 = "Harvard, 2,800."
C_BARGAIN = "I thought we were just bargaining right now. I wanted the price to be at 2800."
C_100 = "Sure, but are you allowed to make it to 100?"
C_WHERE = "But where were we at the negotiation price? I just lost for a bit."
C_VAGUE = "Maybe make it more or less."
C_NAME = "Okay. That's a nice name."
C_DONE = "Yeah, yeah. Done, done, done deal."
C_BYE = "Okay, Sean. Bye-bye."

NLU = FakeNLU({
    C_2800: dict(intent="affirm", buying_signal=0.9, question_topic="offer_terms", gist="agrees to the price"),
    C_BARGAIN: dict(intent="affirm", buying_signal=0.8, objection="price"),
    C_100: dict(intent="affirm", buying_signal=0.8),
    C_VAGUE: dict(intent="affirm", buying_signal=0.6),
    C_DONE: dict(intent="affirm", buying_signal=0.95),
})


def _pipe(product=None, nlu=NLU, lang="en"):
    p = product or GYM
    pipe = ACIEPipeline(DecisionEngine(ProductSalesDomain(p, customer_name="Lakshay")), nlu=nlu)
    return pipe


def T(t):
    return TurnInput(session_id="s", customer_id="+910000000000", text=t)


async def _start(pipe, lang="en"):
    sess = pipe.session(T(""))
    sess.expression = ExpressionParams(language=lang)
    op = await pipe.decision.decide("greet", T(""), sess)
    sess.context.record_turn(0, "", op.intent, False)
    return sess


def talk(lines, product=None, nlu=NLU, lang="en"):
    async def go():
        pipe = _pipe(product, nlu)
        sess = await _start(pipe, lang)
        out = []
        for line in lines:
            r = await pipe.run_turn(T(line))
            out.append((r.plan.intent, r.text))
        return out, sess, pipe
    return asyncio.run(go())


def prices(text):
    return [int(x.replace(",", "")) for x in PRICE.findall(text)]


def deal_of(sess):
    return dl.load(sess.context, sess.dialog)


LIVE = ["Uh, sure, I have no minutes.", "yes tell me the price", C_GOAL, C_2700, C_2800, C_BARGAIN, C_100, C_WHERE, C_VAGUE, C_NAME, C_DONE, C_BYE]


def test_transcript_bug1_counter_2800_is_never_consent_to_the_list_price():
    out, sess, _ = talk(LIVE[:5])
    intent, text = out[4]
    assert "2,999" not in text, text                              # never reverts to the full price
    assert "₹2,800" in text and intent == "confirm_close"           # 2,800 clears the floor and our next step: agreed at THEIR number
    d = deal_of(sess)
    assert d.status == "agreed" and d.agreed_price == 2800 and d.last_customer_counter == 2800


def test_transcript_bug2_agreed_means_confirm_and_close_never_if_i_make_it():
    out, sess, _ = talk(LIVE[:6])
    for _, text in out[4:6]:
        assert not re.search(r"if i make it|kar doon to", text, re.I), text
    assert out[4][1].lower().startswith(("done", "alright")) and "just to confirm" in out[4][1].lower()
    assert out[5][0] in ("negotiation", "confirm_close") and "₹2,800" in out[5][1] and "2,999" not in out[5][1]


def test_transcript_bug3_no_unasked_or_lower_concessions_after_the_customer_said_sure():
    out, sess, _ = talk(LIVE)
    after = out[5:]
    for intent, text in after:
        for x in prices(text):
            assert x in (2800, 100, 6000), (intent, text)          # 100 = their joke number said back; 6,000 = the price after the offer
    joke = out[6][1]
    assert "₹100" in joke and "₹2,800" in joke and not re.search(r"2,799|2,649|2,549", joke)
    d = deal_of(sess)
    assert d.agreed_price == 2800 and d.concessions_made == 0


def test_transcript_bug4_where_were_we_restates_the_deal():
    out, _, _ = talk(LIVE[:8])
    intent, text = out[7]
    assert intent == "negotiation" and "₹2,800" in text and re.search(r"agreed|settled", text, re.I)
    assert prices(text) == [2800, 2800] or set(prices(text)) == {2800}


def test_transcript_bug4b_where_were_we_mid_bargain_says_both_last_numbers():
    out, sess, _ = talk(["yes tell me the price", "too costly", "can you do better", "how about 2,500", C_WHERE])
    text = out[-1][1]
    d = deal_of(sess)
    assert d.status == "countering" and d.last_customer_counter == 2500
    assert f"₹{int(d.last_agent_offer):,}" in text and "₹2,500" in text
    assert d.last_agent_offer == min(prices(out[3][1]) or [d.last_agent_offer])   # nothing new conceded by the restate


def test_transcript_vague_more_or_less_is_not_a_request():
    out, sess, _ = talk(["yes tell me the price", "too costly", "can you do better", C_VAGUE])
    before = prices(out[2][1])
    assert out[-1][0] == "negotiation" and prices(out[-1][1]) and min(prices(out[-1][1])) >= min(before)
    assert deal_of(sess).concessions_made == 1


def test_transcript_bug5_link_and_confirm_carry_the_agreed_price_once():
    out, sess, _ = talk(LIVE)
    link = [t for i, t in out if i == "link_sent"]
    assert link and "₹2,800" in link[0]
    assert sess.dialog.get("payment_link_url", "").endswith("amount=2800")
    assert float(sess.dialog["closed_price"]) == 2800.0
    confirm = out[4][1]
    assert confirm.lower().count("if you join this week") == 1 and "₹2,800 for 3 months" in confirm
    assert deal_of(sess).status == "closed"


def test_transcript_bug5_offer_terms_fact_never_duplicates_the_validity_phrase():
    async def go():
        pipe = _pipe(nlu=None)
        sess = await _start(pipe)
        await pipe.run_turn(T("yes tell me the price"))
        r = await pipe.run_turn(T(C_GOAL))
        return r
    r = asyncio.run(go())
    assert "validity" not in r.plan.facts
    assert r.text.lower().count("if you join this week") <= 1


def test_transcript_bug6_sms_off_never_claims_an_sms():
    assert not GYM.sms_text  # the owner turned SMS off for FitZone
    out, _, _ = talk(LIVE)
    for intent, text in out:
        assert not re.search(r"\bby SMS\b|SMS (the|you|kar)|sending .{0,30}SMS|SMS se", text, re.I), (intent, text)
    link = [t for i, t in out if i == "link_sent"][0]
    assert "our team will send you the payment link" in link.lower()


def test_sms_on_still_says_sms():
    p = Product.model_validate({**RAW["fitzone-gym"], "sms_text": "Join {product} at {offer_price}: {link}"})
    out, _, _ = talk(LIVE, product=p)
    link = [t for i, t in out if i == "link_sent"][0]
    assert "SMS" in link and "₹2,800" in link


# ---------------------------------------------------------------- Hindi / Hinglish counters and amount parsing

@pytest.mark.parametrize("line,amt", [
    ("Harvard, 2,800.", 2800), ("How about 2,800", 2800), ("2.8k chalega?", 2800), ("do hazaar aath sau mein de do", 2800),
    ("अट्ठाईस सौ में दे दो", 2800), ("₹2,800 final", 2800), ("2800 rupees", 2800), ("twenty eight hundred?", 2800),
    ("दो हज़ार आठ सौ चलेगा", 2800), ("28 sau kar do", 2800), ("dhai hazaar", 2500), ("2,999 nahi, 2,500 chahiye", 2500),
    ("3 months ke liye 2500 de do", 2500), ("are you allowed to make it to 100?", 100),
])
def test_amounts_in_english_hindi_hinglish_and_stt_noise(line, amt):
    assert dl.amount(line) == amt
    assert negotiation.counter_number(line) == amt


@pytest.mark.parametrize("line", ["do hazaar aath sau mein de do", "अट्ठाईस सौ में दे दो", "28 sau kar do", "2800 pe done?"])
def test_hindi_hinglish_counters_after_the_price_are_counters_never_consent(line):
    out, sess, _ = talk(["haan batao", "kitne ka hai", "bahut mehenga hai", line], nlu=None, lang="hinglish")
    intent, text = out[-1]
    assert intent != "link_sent" and "2,999" not in text
    d = deal_of(sess)
    assert d.last_customer_counter == 2800
    assert "₹2,800" in text  # 2,800 >= floor but below our offer: first push says no to it, or agreed at it - never our list price


def test_hinglish_full_haggle_agrees_and_confirms_in_hinglish():
    out, sess, _ = talk(["haan batao", "kitne ka hai", "bahut mehenga hai", "aur kam karo", "2650 chalega?", "haan theek hai"], nlu=None, lang="hinglish")
    assert out[4][0] == "confirm_close" and "₹2,650" in out[4][1] and "Bas confirm" in out[4][1]
    assert out[5][0] == "link_sent" and "₹2,650" in out[5][1] and "SMS" not in out[5][1]


def test_accept_words():
    for w in ("ok", "done deal", "theek hai", "chalega", "Done, done, done deal.", "पक्का", "haan ji"):
        assert dl.is_accept(w), w
    for w in ("maybe make it more or less", "ok but 2500?", "Okay. That's a nice name."):
        assert not dl.is_accept(w), w


# ---------------------------------------------------------------- the rules, at the negotiation level

def play(lines, p=GYM, tone="mid"):
    st, out = negotiation.NegState(), []
    for t in lines:
        r = negotiation.step(p, t, st, "", None, tone=tone)
        if r is None:  # not price talk: nothing moves
            continue
        st = r.state
        out.append(r)
    return out


def test_ladder_steps_shrink_strictly_and_respect_max_concessions():
    seq = negotiation.ladder_prices(2999, FLOOR, max_steps=GYM.negotiation.max_concessions)
    cuts = [a - b for a, b in zip([2999] + seq, seq)]
    assert all(a > b for a, b in zip(cuts, cuts[1:])), cuts
    assert seq[-1] >= FLOOR and len(seq) <= GYM.negotiation.max_concessions


def test_floor_never_breached_and_concessions_monotonic_for_any_counter_sequence():
    rng = random.Random(7)
    for _ in range(150):
        lines = [rng.choice(["kam karo", "too costly", "can you do better", f"{rng.randint(50, 3200)} mein do", "where were we?",
                             "maybe more or less", "aur discount do"]) for _ in range(10)]
        lowest, best_counter = 2999, 0
        for r in play(lines):
            if r.offer and r.offer.get("price"):
                pr = r.offer["price"]
                assert pr >= FLOOR - 1e-9, (lines, pr)
                assert pr <= lowest + 1e-9, (lines, pr, lowest)
                assert pr >= min(best_counter, lowest) - 1e-9, (lines, pr, best_counter)
                lowest = min(lowest, pr)
            best_counter = max(best_counter, r.state.best_counter)
            assert r.state.concessions <= GYM.negotiation.max_concessions


def test_concessions_need_a_request_and_shrink():
    out = play(["too costly", "can you do better", "can you do better", "can you do better", "can you do better"])
    got = [r.offer["price"] for r in out if r.offer]
    cuts = [a - b for a, b in zip([2999] + got, got)]
    assert got == sorted(got, reverse=True) and all(a > b for a, b in zip(cuts, cuts[1:])), got
    st = out[-1].state
    for line in ("Maybe make it more or less.", "But where were we at the negotiation price?", "Okay. That's a nice name."):
        r = negotiation.step(GYM, line, st, "", None)
        assert r is None or not r.offer or r.offer["price"] >= st.price


def test_never_below_their_own_counter():
    out = play(["2800 mein do", "kam karo"])   # first push says no to 2,800; the next cut (2,799) would go under their 2,800: agree at 2,800
    assert out[1].agreed and out[1].offer["price"] == 2800


def test_agreed_price_never_moves_on_a_push_or_a_lower_counter():
    out = play(["too costly", "2700 chalega"])
    assert out[1].agreed is False or out[1].offer["price"] >= 2700
    st = out[1].state
    st.agreed = st.price = 2700.0
    for line in ("aur kam karo", "2500 mein do", "100 mein do", "can you do better"):
        r = negotiation.step(GYM, line, st, "", None)
        assert r.offer and r.offer["price"] == 2700.0 and r.kind in ("hold", "joke", "restate"), (line, r.text["en"])


def test_absurd_counter_gets_humour_and_no_movement():
    out = play(["too costly", "can you do better", "100 mein de do"])
    r = out[-1]
    assert r.kind == "joke" and "₹100" in r.text["en"] and r.offer["price"] == out[1].offer["price"]


# ---------------------------------------------------------------- the agreed price survives a context restore

def test_agreed_price_survives_serialise_and_reload_of_the_session():
    async def go():
        pipe = _pipe()
        sess = await _start(pipe)
        for line in LIVE[:5]:
            await pipe.run_turn(T(line))
        blob = sess.model_dump_json()                     # what "ctx" persists (whole session incl. context + dialog)
        # a new process: fresh pipeline + fresh domain (starts from the configured product at ₹2,999)
        pipe2 = _pipe()
        restored = SessionState.model_validate_json(blob)
        pipe2.sessions["s"] = restored
        r1 = await pipe2.run_turn(T("where were we on the price?"))
        r2 = await pipe2.run_turn(T("ok done, send it"))
        return r1, r2, restored
    r1, r2, sess = asyncio.run(go())
    assert "₹2,800" in r1.text and "2,999" not in r1.text
    assert r2.plan.intent == "link_sent" and "₹2,800" in r2.text
    assert dl.load(sess.context, sess.dialog).agreed_price == 2800


def test_deal_state_round_trips_through_json():
    d = dl.new_deal(dl.config_for(GYM))
    d.customer_counter(2700, 3)
    d.agent_offer(2799, 4)
    d.agree(2750, 5)
    back = dl.DealState.model_validate_json(d.model_dump_json())
    assert back == d and back.status == "agreed" and back.agreed_price == 2750
    assert back.agent_offer(2500, 6) == 2750          # an agreed price never moves
    assert "floor" not in json.dumps(back.facts())


# ---------------------------------------------------------------- a trust / charges question is never a price request (studio sceptic run)

@pytest.mark.parametrize("lines,lang", [
    (["yes tell me the price", "hidden charges? are the trainers certified?"], "en"),
    (["yes tell me the price", "too costly", "can you do better", "hidden charges? are the trainers certified?"], "en"),
    (["haan batao", "kitne ka hai", "koi hidden charge to nahi hai?"], "hinglish"),
    (["Haan bolo, par mujhe aise cheap offers pe bilkul bharosa nahi hota.",
      "Budget ki baat nahi hai, mujhe darr hai ki baad mein hidden charges honge aur trainers bhi certified nahi honge.",
      "Aap budget pe kyu ja rahe ho, pehle ye batao ki trainers certified hain aur koi hidden charges toh nahi honge?"], "hinglish"),
    (["haan बताइए", "कितने का है", "कोई छुपा चार्ज तो नहीं है?"], "hi"),
])
def test_hidden_charges_or_certification_question_is_answered_never_negotiated(lines, lang):
    out, sess, _ = talk(lines, nlu=None, lang=lang)
    before = deal_of(sess)
    for intent, text in out[-2:] if len(lines) == 3 and "bharosa" in lines[0] else out[-1:]:
        assert intent != "negotiation" and "budget" not in text.lower(), (intent, text)
        assert not re.search(r"2,799|2,649|2,549|2,499", text) or "2,799" in out[-2][1], text
    assert before.concessions_made <= (1 if "can you do better" in lines else 0)


def test_no_sms_left_in_calls_only_answers():
    from acie.decision import honest
    for kind in ("whatsapp", "trial", "branch", "free_signup"):
        for l in ("en", "hi", "hinglish"):
            assert "SMS" not in honest.info_request_variants(GYM, kind, l)[0]
    # (FitZone's FAQ wording lives in data/voice/products.json on the server - operator data, not in git - so it is not asserted here)


# ---------------------------------------------------------------- live call cc6594e5d1: garbage "need" quoted; the deal abandoned after agreement

CC = ["Uh, yeah, sure, I have.", "Why should I even join your gym? I'm already training somewhere else.", "Uh, I didn't get you.",
      "I'm more into more like power lifting and all. Do you have any special things for deadlifts and all?",
      "I have indeed lifted before. Do you have any personal trainers that you can refer to? I would be willing to negotiate for the prize if possible.",
      "So basically,", "I was thinking here about...", "2000 rupees.", "uh maybe can you just go a little bit lower for me like 2000", "600 maybe.",
      "Sure, I'll just...", "pay at the gym, but I'm not comfortable at paying on a link.", "God.",
      "I just said sure. I'm not sure why I was just repeating the same thing again and again."]


def test_cc6594_never_quotes_a_mishearing_as_the_customers_need():
    out, sess, _ = talk(CC[:5], nlu=None)
    for _, text in out:
        assert "didn't get you" not in text and "You mentioned" not in text or "lift" in text.lower(), text
    assert not negotiation.real_need("Uh, I didn't get you.") and negotiation.real_need("I'm more into power lifting")


def test_cc6594_after_agreement_a_pay_at_the_gym_concern_closes_at_the_agreed_price():
    out, sess, _ = talk(CC, nlu=None)
    agreed = deal_of(sess).agreed_price
    assert agreed and agreed >= FLOOR
    p = f"₹{int(agreed):,}"
    i = CC.index("pay at the gym, but I'm not comfortable at paying on a link.")
    if i < len(out):
        intent, text = out[i]
        assert "front desk" in text and p in text and "a day" not in text and "feel like a lot" not in text, text
    for intent, text in out[CC.index("Sure, I'll just..."):]:
        assert "a day" not in text and "feel like a lot" not in text.lower(), text
        assert all(x in (int(agreed), 6000) for x in prices(text)), text


@pytest.mark.parametrize("line", ["I just said sure. I'm not sure why I was just repeating the same thing again and again.", "haan bola na", "Sure."])
def test_after_agreement_sure_is_consent_to_the_agreed_price(line):
    out, sess, _ = talk(["yes tell me the price", "too costly", "how about 2,800", "God.", line], nlu=None)
    assert out[2][0] == "confirm_close" and "₹2,800" in out[2][1]
    assert "₹2,800" in out[3][1] and "a day" not in out[3][1]
    assert out[4][0] == "link_sent" and "₹2,800" in out[4][1]


@pytest.mark.parametrize("line", ["I'm training for a 21 kilometer marathon", "21 km ki marathon hai", "hafte mein 4 din aa sakta hoon",
                                  "5 baje aaunga", "I do 5 sets of 10 reps", "मैं 21 किलोमीटर दौड़ता हूँ", "I lift 100 kg"])
def test_numbers_with_units_are_never_price_counters(line):
    assert dl.amount(line) is None
    out, sess, _ = talk(["yes tell me the price", "too costly", "can you do better", line], nlu=None)
    intent, text = out[-1]
    assert "Nice try" not in text and "koshish" not in text
    d = deal_of(sess)
    assert d.last_customer_counter == 0 and d.last_agent_offer == 2799


def test_bare_or_rupee_amounts_still_count():
    for line, n in (("2500 mein do", 2500), ("₹2,600", 2600), ("do hazaar aath sau", 2800), ("how about 2,700", 2700)):
        assert dl.amount(line) == n


# ---------------------------------------------------------------- live call f79a90af02: an ack right after a complaint is not consent

F79 = ["Uh, sure.", "Love it.",
       "I think you are being a bit too rude by just suddenly giving me all the price petitions and all. Did you even ask me if I want to join or not?",
       "Okay. Love it."]


def test_f79_ack_after_a_complaint_never_jumps_to_confirm_close():
    nlu = FakeNLU({"Love it.": dict(intent="affirm", buying_signal=0.85), "Okay. Love it.": dict(intent="affirm", buying_signal=0.9),
                   F79[2]: dict(intent="slow_down", frustration=0.6, objection="other")})
    out, sess, _ = talk(F79, nlu=nlu)
    assert out[-1][0] not in ("confirm_close", "link_sent"), out[-1]
    assert "payment link" not in out[-1][1]


def test_love_it_without_a_buy_question_is_not_consent_but_a_real_yes_still_is():
    out, _, _ = talk(["yes tell me the price", "Love it."], nlu=None)
    assert out[-1][0] not in ("confirm_close", "link_sent")
    out, _, _ = talk(["yes tell me the price", "yes, send me the link"], nlu=None)
    assert out[-1][0] in ("confirm_close", "link_sent")
