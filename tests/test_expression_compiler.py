import subprocess
import sys

from acie.compiler import SYSTEM_TEMPLATE, LANGUAGE_RULES, LEVEL_RULES, PromptCompiler
from acie.schemas import CustomerProfile, ExpressionParams, Fact, SessionState, TurnPlan

PLAN = TurnPlan(intent="x", objective="tell balance", facts={"balance": Fact(value="₹5", description="bal")},
                fallback={"en": "Balance {balance}."})


def S():
    return SessionState(session_id="s", customer_id="c")


def legacy_system(expr):
    ack = ("\n- Start with one brief, sincere acknowledgement of the customer's frustration, then get to the point."
           if expr.acknowledge_emotion else "")
    return SYSTEM_TEMPLATE.format(
        language=LANGUAGE_RULES[expr.language], verbosity=LEVEL_RULES["verbosity"][expr.verbosity],
        complexity=LEVEL_RULES["complexity"][expr.complexity], formality=LEVEL_RULES["formality"][expr.formality],
        warmth=LEVEL_RULES["warmth"][expr.warmth], proactivity=LEVEL_RULES["proactivity"][expr.proactivity], ack=ack)


def test_default_prompt_byte_identical():
    for e in (ExpressionParams(), ExpressionParams(language="hinglish", acknowledge_emotion=True, verbosity="low")):
        p = PromptCompiler().compile(PLAN, e, S(), "hello")
        assert p.system == legacy_system(e)
        assert "PROFILE" not in p.messages[-1]["content"] and "MEMORIES" not in p.messages[-1]["content"]


class Knobs:
    """ExpressionParams stand-in carrying the new persona-lane knobs (works before the schema merge)."""

    def __init__(self, **kw):
        self._e = ExpressionParams(language=kw.pop("language", "hinglish"))
        self.__dict__.update(kw)

    def __getattr__(self, k):
        return getattr(self._e, k)


def test_new_knobs_add_style_lines():
    e = Knobs(regional_variant="mumbai", explanation_style="step_by_step", info_density="low", reassurance="high")
    sys_ = PromptCompiler().compile(PLAN, e, S(), "hi").system
    assert "Mumbai-style Hinglish" in sys_
    assert "one short step at a time" in sys_ and "only the one key point" in sys_ and "no hurry" in sys_
    de = PromptCompiler().compile(PLAN, Knobs(language="hi", regional_variant="delhi"), S(), "hi").system
    assert "Delhi-style spoken Hindi" in de and "बिल्कुल" in de
    en = PromptCompiler().compile(PLAN, Knobs(language="en", regional_variant="south_indian_english"), S(), "hi").system
    assert "South Indian English" in en
    other = PromptCompiler().compile(PLAN, Knobs(regional_variant="goa"), S(), "hi").system
    assert "goa regional flavour" in other
    # defaults of the new knobs add nothing
    d = Knobs(explanation_style="direct", info_density="med", reassurance="low", regional_variant=None)
    assert PromptCompiler().compile(PLAN, d, S(), "hi").system == legacy_system(ExpressionParams(language="hinglish"))


def test_profile_and_memories_blocks():
    class P:
        likes = ["cricket", "tea {x}"]
        topics = ["upi"]
        dislikes = []

    c = PromptCompiler()
    m = c.compile(PLAN, ExpressionParams(), S(), "hi", profile=P(), memories=["asked about fees last week", "  "]).messages[-1]["content"]
    assert "CUSTOMER PROFILE" in m and "likes: cricket, tea x" in m and "often asks about: upi" in m
    assert "RELEVANT MEMORIES" in m and "- asked about fees last week" in m
    plain = c.compile(PLAN, ExpressionParams(), S(), "hi", profile=CustomerProfile(customer_id="c"), memories=[])
    assert "PROFILE" not in plain.messages[-1]["content"] and "MEMORIES" not in plain.messages[-1]["content"]
