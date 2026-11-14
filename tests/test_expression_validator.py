from acie.rules import BankingPack, HindiRegisterPack, SalesCompliancePack, load_packs
from acie.schemas import ExpressionParams, Fact, TurnPlan
from acie.validator import StreamingValidator

PLAN = TurnPlan(intent="x", objective="o", facts={"bal": Fact(value="₹5", description="d")},
                fallback={"en": "Balance {bal}."})
E_ = ExpressionParams()


def run(text, plan=PLAN, expr=E_, **kw):
    v = StreamingValidator(plan, expr, **kw)
    out = v.feed(text)
    out += v.finish()
    return v, out


def test_issue_codes_each_kind():
    v, _ = run("Your balance is {bal}. Also {nope} here.")
    assert "invented_slot" in v.report().issue_codes
    v, _ = run("You have 500 rupees.")
    assert v.report().issue_codes[0] == "raw_number"
    v, _ = run("आपका बैलेंस {bal} है।")
    r = v.report()
    assert "wrong_script" in r.issue_codes
    v, _ = run("Hello there.")
    r = v.report()
    assert r.issue_codes == ["missing_fact", "fallback"] and r.issues[0].startswith("missing facts")
    assert r.signals == {"missing_fact": 1, "fallback": 1}
    v, _ = run("Balance {bal} is fine. This part was cut off and never")
    assert "unfinished" in v.report().issue_codes


def test_clean_reply_has_no_codes():
    v, out = run("Your balance is {bal}.")
    r = v.report()
    assert r.ok and r.issues == [] and r.issue_codes == [] and r.signals == {}


def test_sales_compliance_pack():
    p = SalesCompliancePack()
    for bad in ("This is guaranteed to work.", "Limited time only, grab it!", "You must buy today.",
                "It will cure your back pain.", "Other apps are a scam.", "Zero risk for you.",
                "यह बिल्कुल गारंटी है।", "Aapko lena hi hoga ji."):
        assert p.check(bad, PLAN, E_), bad
    for good in ("It can help you track spending.", "You can decide at your own pace.", "Balance {bal}."):
        assert p.check(good, PLAN, E_) is None, good


def test_banking_pack():
    p = BankingPack()
    assert p.check("Please share your OTP.", PLAN, E_)
    assert p.check("अपना पिन बताइए।", PLAN, E_)
    assert p.check("Tell me your account number.", PLAN, E_)
    assert p.check("Your account is 123456789012.", PLAN, E_)
    assert p.check("Never share your OTP with anyone.", PLAN, E_) is None
    assert p.check("Your balance is {bal}.", PLAN, E_) is None


def test_hindi_register_pack_configurable():
    casual = ExpressionParams(language="hinglish", formality="low")
    p = HindiRegisterPack()
    assert "Kripya" in p.check("Kripya ruk jaiye.", PLAN, casual)
    assert p.check("कृपया रुकिए।", PLAN, ExpressionParams(language="hi", formality="low"))
    assert p.check("Kripya ruk jaiye.", PLAN, ExpressionParams(language="hinglish", formality="high")) is None
    assert p.check("Kripya ruk jaiye.", PLAN, ExpressionParams(language="en", formality="low")) is None
    mine = HindiRegisterPack(formal_words=["zaroori"], casual_levels=("low", "med"))
    assert mine.check("Yeh zaroori hai.", PLAN, ExpressionParams(language="hinglish", formality="med"))
    assert mine.check("Kripya ruk jaiye.", PLAN, casual) is None


def test_packs_off_by_default_and_selected_by_plan():
    v, out = run("This is guaranteed. Your balance is {bal}.")
    assert v.report().ok and "guaranteed" in out[0]
    plan = PLAN.model_copy(update={"rule_packs": ["sales_compliance"]})
    v, out = run("This is guaranteed. Your balance is {bal}.", plan=plan)
    r = v.report()
    assert r.used_fallback and out == ["Balance ₹5."]
    assert "rulepack:sales_compliance" in r.issue_codes and r.signals["rulepack:sales_compliance"] == 1
    assert any("guarantee" in i for i in r.issues)


def test_explicit_pack_arg_and_trusted_skips():
    v, _ = run("Share your OTP now. {bal}.", rule_packs=["banking"])
    assert "rulepack:banking" in v.report().issue_codes
    v, out = run("Limited time only {bal}.", plan=PLAN.model_copy(update={"rule_packs": ["sales_compliance"]}), trusted=True)
    assert v.report().ok


def test_load_packs_ignores_unknown():
    assert load_packs(["nope", "banking"])[0].name == "banking"
    obj = HindiRegisterPack()
    assert load_packs([obj]) == [obj]
