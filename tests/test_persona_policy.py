from acie.personalization.policy import AdaptationPolicy, detect_explicit_requests, make_default_policy
from acie.schemas import CustomerProfile, ExpressionParams, PreferenceField, SessionState
from acie.schemas import HumanState as _Base


class HumanState(_Base):
    """The 14-dim state (state lane adds these to schemas.HumanState; extended here so the test runs before the merge)."""
    satisfaction: float = 0.5
    anxiety: float = 0.0
    excitement: float = 0.0
    calmness: float = 0.5
    trust: float = 0.5
    certainty: float = 0.5
    cognitive_load: float = 0.0
    attention: float = 0.5
    participation: float = 0.5
    receptiveness: float = 0.5


def run(state, explicit=None, text="ok batao", profile=None, session=None):
    s = session or SessionState(session_id="s", customer_id="c")
    p = profile or CustomerProfile(customer_id="c")
    return AdaptationPolicy().decide(text, state, p, s, explicit or {}), s


def test_defaults_identical_for_neutral_inputs():
    a, _ = run(HumanState())
    assert a == ExpressionParams(language=a.language)  # all new knobs at defaults
    assert (a.regional_variant, a.explanation_style, a.info_density, a.reassurance) == (None, "direct", "med", "low")


def test_anxiety_gives_reassurance_and_steps():
    a, _ = run(HumanState(anxiety=0.8))
    assert a.reassurance == "high" and a.explanation_style == "step_by_step"


def test_low_trust_gives_reassurance():
    a, _ = run(HumanState(trust=0.1))
    assert a.reassurance == "high" and a.explanation_style == "step_by_step"


def test_cognitive_load_lowers_density_and_complexity():
    a, _ = run(HumanState(cognitive_load=0.8))
    assert a.info_density == "low" and a.complexity == "low"


def test_certain_and_engaged_raises_density():
    a, _ = run(HumanState(certainty=0.9, engagement=0.9))
    assert a.info_density == "high"
    b, _ = run(HumanState(certainty=0.9, engagement=0.5))
    assert b.info_density == "med"


def test_hysteresis_new_flags():
    s = SessionState(session_id="s", customer_id="c")
    assert run(HumanState(anxiety=0.6), session=s)[0].reassurance == "high"
    assert run(HumanState(anxiety=0.4), session=s)[0].reassurance == "high"  # between exit and enter: stays on
    assert run(HumanState(anxiety=0.2), session=s)[0].reassurance == "low"
    assert run(HumanState(cognitive_load=0.55), session=s)[0].info_density == "low"
    assert run(HumanState(cognitive_load=0.4), session=s)[0].info_density == "low"
    assert run(HumanState(cognitive_load=0.1), session=s)[0].info_density == "med"
    s2 = SessionState(session_id="s", customer_id="c")
    run(HumanState(certainty=0.9, engagement=0.9), session=s2)
    assert run(HumanState(certainty=0.6, engagement=0.6), session=s2)[0].info_density == "high"
    assert run(HumanState(certainty=0.4, engagement=0.6), session=s2)[0].info_density == "med"


def test_explicit_beats_state_rules():
    ex = detect_explicit_requests("sirf zaroori baat batao, example dekar samjhao")
    a, _ = run(HumanState(anxiety=0.9, cognitive_load=0.9, certainty=0.9, engagement=0.9), ex)
    assert a.explanation_style == "example" and a.info_density == "low"
    ex = detect_explicit_requests("poori jaankari do")
    a, _ = run(HumanState(cognitive_load=0.9), ex)
    assert a.info_density == "high"


def test_explicit_regional_and_reassurance():
    ex = detect_explicit_requests("Delhi wali Hindi, mujhe dar lag raha hai")
    a, _ = run(HumanState(), ex)
    assert a.regional_variant == "delhi" and a.reassurance == "high"
    a, _ = run(HumanState(), {"regional_variant": "standard"})
    assert a.regional_variant is None


def test_profile_prefs_used_when_confident_or_explicit():
    p = CustomerProfile(customer_id="c", preferences={
        "regional_variant": PreferenceField(value="mumbai", confidence=0.7),
        "explanation_style": PreferenceField(value="example", confidence=0.3),
        "info_density": PreferenceField(value="low", confidence=0.2, explicit=True)})
    a, _ = run(HumanState(), profile=p)
    assert a.regional_variant == "mumbai" and a.explanation_style == "direct" and a.info_density == "low"


def test_invalid_pref_values_ignored():
    p = CustomerProfile(customer_id="c", preferences={"info_density": PreferenceField(value="weird", confidence=0.9)})
    assert run(HumanState(), profile=p)[0].info_density == "med"


def test_frustration_protects_warmth_in_trace():
    pol = AdaptationPolicy()
    s = SessionState(session_id="s", customer_id="c")
    a, prot = pol.decide_traced("x", HumanState(frustration=0.9), CustomerProfile(customer_id="c"), s, {})
    assert a.warmth == "high" and {"warmth", "verbosity"} <= prot


def test_default_policy_env(monkeypatch):
    assert type(make_default_policy()) is AdaptationPolicy
    monkeypatch.setenv("ACIE_POLICY", "bandit")
    monkeypatch.setenv("ACIE_BANDIT_STATE", "/nonexistent-dir-x/b.json")
    from acie.personalization.bandit import BanditPolicy
    assert isinstance(make_default_policy(), BanditPolicy)
