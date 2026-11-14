

def test_customer_numbers_and_rupee_note_may_be_repeated():
    from acie.schemas import ExpressionParams, TurnPlan
    from acie.validator import StreamingValidator
    plan = TurnPlan(intent="answer", objective="x", context="about ₹700 a month, then about ₹1,700", fallback={"en": "fb"})
    v = StreamingValidator(plan, ExpressionParams(language="en"), customer_text="call me tomorrow at 7")
    out = v.feed("Sure, tomorrow at 7 then. It's about ₹700 a month, later ₹1,700. ") + v.finish()
    assert not v.used_fallback, v.issues
    v = StreamingValidator(plan, ExpressionParams(language="en"))
    v.feed("It costs 999 a month. "); v.finish()
    assert v.used_fallback
