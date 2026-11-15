"""The context records only what the caller HEARD (agent ctx): a barge-in cuts the agent turn to the heard part."""
import json
from types import SimpleNamespace

from acie.decision.domains.product_sales import Product, ProductSalesDomain
from acie.decision.engine import DecisionEngine
from acie.pipeline import ACIEPipeline
from acie.schemas import TurnInput, Utterance
from acie.validator import StreamingValidator
from voice.heard import PlayLog

PRODS = {k: Product.model_validate(v) for k, v in json.load(open("data/voice/products.json")).items()}
GYM = PRODS.get("fitzone-gym") or Product()
SID = "call-heard1"


def T(text):
    return TurnInput(session_id=SID, customer_id="+910000000002", text=text)


async def setup():
    pipe = ACIEPipeline(DecisionEngine(ProductSalesDomain(GYM, customer_name="Lakshay")))
    sess = pipe.session(T(""))
    plan = await pipe.decision.decide("greet", T(""), sess)
    text = " ".join(StreamingValidator(plan, sess.expression).finish())
    sess.history.append(Utterance(role="agent", text=text))
    sess.context.record_turn(0, "", plan.intent, False)
    return pipe, sess, text


# ---------------------------------------------------------------- playback estimate

def test_playlog_only_fully_played_sentences_count():
    log = PlayLog()
    e = log.begin(3, "It's ₹2,999 for 3 months. That's half price. Shall I send the link?")
    PlayLog.sent(e, 10.0, 16.0)
    e["done"] = True
    assert log.heard(3, 10.5)[0] == ""                                    # cut in the first sentence
    h, full, r = log.heard(3, 10.0 + 6.0 * 26 / 64 + 0.4)
    assert h == "It's ₹2,999 for 3 months." and 0 < r < 1
    assert log.heard(3, 16.5) == (full, full, 1.0)                        # full playback: unchanged
    assert log.heard(4, 12.0) == ("", "", 1.0)                            # another epoch: nothing spoken


def test_playlog_still_streaming_uses_a_speech_rate_estimate():
    log = PlayLog()
    e = log.begin(1, "Hi, I'm Arjun from FitZone. I'm calling about a membership offer.")
    PlayLog.sent(e, 0.0, 1.0)  # only 1 s sent so far
    assert log.heard(1, 1.0)[0] == ""


def test_call_session_cuts_the_turn_record_and_tells_the_pipeline():
    from voice.call import CallSession
    log = PlayLog()
    e = log.begin(7, "Sure. It's ₹2,999 for 3 months.")
    PlayLog.sent(e, 0.0, 4.0)
    e["done"] = True
    rec = {"role": "agent", "text": e["text"], "intent": "price", "t": 1.0}
    got, events = [], []
    me = SimpleNamespace(epoch_turn={7: rec}, cut_at={7: 1.2}, playlog=log, session_id=SID,
                         pipeline=SimpleNamespace(apply_heard=lambda *a: got.append(a)),
                         event=lambda k, d: events.append((k, d)), call=SimpleNamespace(call_id="x"))
    CallSession.apply_heard(me, 7)
    assert rec["interrupted"] and rec["heard_text"] == "Sure." and rec["text"] == "Sure. It's ₹2,999 for 3 months."
    assert got and got[0][1] == "Sure." and events[0][0] == "heard"
    CallSession.apply_heard(me, 7)  # once only
    assert len(got) == 1


# ---------------------------------------------------------------- context: heard-only

async def test_barge_in_mid_price_line_price_not_quoted_and_said_again_next():
    pipe, sess, greet = await setup()
    r = await pipe.run_turn(T("yes, what is the price?"))
    assert "offer_price" in sess.context.quoted
    assert pipe.apply_heard(SID, "", 0.0)
    ctx = sess.context
    assert "offer_price" not in ctx.quoted and not ctx.checklist.price_quoted
    assert ctx.last_agent_line != r.text and sess.history[-1].text.startswith("(cut off")
    r2 = await pipe.run_turn(T("what is the price?"))
    assert "2,999" in r2.text or "₹" in r2.text
    assert "offer_price" in ctx.quoted


async def test_partial_heard_keeps_only_the_heard_part():
    pipe, sess, _ = await setup()
    r = await pipe.run_turn(T("yes, what is the price?"))
    first = r.text.split(". ")[0] + "."
    pipe.apply_heard(SID, first, len(first) / len(r.text))
    assert sess.context.last_agent_line == first
    assert sess.history[-1].text == first + " —"


async def test_unheard_counter_offer_is_not_the_deal_last_agent_offer():
    pipe, sess, _ = await setup()
    await pipe.run_turn(T("yes, what is the price?"))
    await pipe.run_turn(T("too costly, can you make it 2500?"))
    before = sess.context.deal_view.get("last_agent_offer")
    r = await pipe.run_turn(T("no, 2500 only, final"))
    import re
    offered = [float(x.replace(",", "")) for x in re.findall(r"₹([\d,]+)", r.text)]
    pipe.apply_heard(SID, "", 0.0)
    ctx = sess.context
    after = ctx.deal_view.get("last_agent_offer")
    assert offered and after == before and after not in offered
    from acie.pipeline import _deal_offer
    assert _deal_offer(ctx.deal) not in offered


async def test_barge_in_mid_opening_purpose_is_stated_next_turn():
    pipe, sess, greet = await setup()
    assert "who_is_calling" in sess.context.told
    pipe.apply_heard(SID, "", 0.0, "greet")  # cut before the caller heard who is calling
    assert "who_is_calling" not in sess.context.told and sess.dialog.get("_purpose_due") == "1"
    r = await pipe.run_turn(T("hello? who is this?"))
    assert GYM.seller_name in r.text and "offer" in r.text.lower()
    r2 = await pipe.run_turn(T("okay"))
    assert "Just so you know" not in r2.text  # once


async def test_full_playback_changes_nothing():
    pipe, sess, _ = await setup()
    r = await pipe.run_turn(T("yes, what is the price?"))
    snap = sess.context.model_dump()
    assert sess.context.last_agent_line == r.text and snap["checklist"]["price_quoted"]
    assert "_purpose_due" not in sess.dialog
