"""Validator: invented CRM memory, unsourced claims, repeated / restated / re-asked sentences."""

from acie.psyche.ledger import CustomerFacts, SaidLedger
from acie.schemas import ExpressionParams, Fact, TurnPlan
from acie.validator import StreamingValidator


def plan(**kw):
    return TurnPlan(intent="answer", objective="answer", **kw)


def run(text, pl=None, known=None, customer="what is it", trusted=False):
    v = StreamingValidator(pl or plan(), ExpressionParams(language="en"), trusted=trusted, customer_text=customer, known=known)
    out = v.feed(text) + v.finish()
    return v, out


def test_invented_memory_is_rejected_even_in_trusted_text():
    for trusted in (False, True):
        v, _ = run("We have your details from your previous inquiry.", trusted=trusted)
        assert "invented_memory" in v.issue_codes and v.used_fallback


def test_unsourced_claims_are_rejected():
    for claim in ("You can cancel any time.", "You get five times more usage.", "It has the best models.", "Unlimited usage for everyone."):
        v, _ = run(claim)
        assert "unsourced_claim" in v.issue_codes, claim


def test_sourced_claims_pass():
    pl = plan(context="You can cancel any time. Unlimited group classes.")
    v, out = run("You can cancel any time.", pl)
    assert "unsourced_claim" not in v.issue_codes and out


def test_repeated_sentence_fails_and_falls_back():
    led = SaidLedger()
    led.add_text("A free personal trainer session comes with it every week.")
    v, _ = run("A free personal trainer session comes with it every week.", known=(led, CustomerFacts()))
    assert "repeated_sentence" in v.issue_codes and v.used_fallback


def test_restated_fact_and_reasked_question_fail():
    f = CustomerFacts()
    f.learn_caller("I want to build muscle")
    v, _ = run("It will help with muscle.", known=(SaidLedger(), f))
    assert "restated_fact" in v.issue_codes
    v, _ = run("What is your main goal?", known=(SaidLedger(), f))
    assert "reasked_question" in v.issue_codes


def test_new_sentence_passes_with_known_block():
    led, f = SaidLedger(), CustomerFacts()
    led.add_text("It opens at five in the morning.")
    f.learn_caller("muscle")
    v, out = run("There is a free trainer session every week.", known=(led, f))
    assert out and not v.used_fallback


def test_repeated_opener_fails_and_new_opener_passes():
    led, f = SaidLedger(), CustomerFacts()
    f.learn_agent("Great news, the offer is on.")
    v, out = run("Great news, there is also a trainer.", known=(led, f))
    assert "repeated_opener" in v.issue_codes
    # the repeated opener is stripped and the rest kept (call c9960ed17d: never the whole reply swapped for "didn't get that")
    assert out == ["There is also a trainer."] and not v.used_fallback
    v, out = run("There is also a trainer session every week.", known=(led, f))
    assert out and not v.used_fallback
