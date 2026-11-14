import pytest

from acie.personalization.learner import LIST_CAP, learn_from_turn
from acie.personalization.profile_store import update_preference
from acie.schemas import CustomerProfile, HumanState


def prof():
    return CustomerProfile(customer_id="c1")


def conf(p, key):
    return p.preferences[key].confidence if key in p.preferences else 0.0


def run(p, text, n, **kw):
    for _ in range(n):
        learn_from_turn(p, text, HumanState(), None, kw.get("explicit", {}))


def test_formality_curve_is_gradual_and_reaches_policy_threshold():
    p = prof()
    learn_from_turn(p, "aap bataiye sir", HumanState(), None, {})
    assert p.preferences["formality"].value == "high" and conf(p, "formality") < 0.2  # one turn = weak
    run(p, "ji aap kya kar sakte hain sir", 15)
    assert conf(p, "formality") >= 0.6


def test_one_odd_turn_does_not_flip_preference():
    p = prof()
    run(p, "aap kaise hain sir", 14)
    learn_from_turn(p, "arre bhai tu bata yaar", HumanState(), None, {})
    pf = p.preferences["formality"]
    assert pf.value == "high" and pf.confidence > 0.5


def test_informal_learns_low_formality():
    p = prof()
    run(p, "bhai tu bata na yaar", 12)
    assert p.preferences["formality"].value == "low" and conf(p, "formality") >= 0.5


def test_verbosity_from_reply_length():
    p = prof()
    run(p, "haan theek", 20)
    assert p.preferences["verbosity"].value == "low" and conf(p, "verbosity") >= 0.5
    q = prof()
    run(q, " ".join(["word"] * 30), 20)
    assert q.preferences["verbosity"].value == "high"


def test_explicit_keys_are_skipped():
    p = prof()
    run(p, "aap sir ji", 5, explicit={"formality": "low"})
    assert "formality" not in p.preferences


def test_explicit_locks_field():
    p = prof()
    update_preference(p, "formality", "low", explicit=True)
    run(p, "aap sir ji", 20)
    assert p.preferences["formality"].value == "low" and p.preferences["formality"].explicit


@pytest.mark.parametrize("text,key,val", [
    ("can you give an example?", "explanation_style", "example"),
    ("process kya hai step by step", "explanation_style", "step_by_step"),
    ("seedha bata do", "explanation_style", "direct"),
    ("aur batao poori details", "info_density", "high"),
    ("bas itna enough", "info_density", "low"),
    ("thank you so much, great", "warmth", "high"),
    ("just business, no small talk", "warmth", "low"),
    ("samajh nahi aaya simple me bolo", "complexity", "low"),
    ("what is the api latency and sla", "complexity", "high"),
])
def test_signal_keys(text, key, val):
    p = prof()
    learn_from_turn(p, text, HumanState(), None, {})
    assert p.preferences[key].value == val


@pytest.mark.parametrize("text,variant", [
    ("apun ko bhi chahiye bantai", "mumbai"),
    ("hamra ke batawa ka ho", "bihar"),
    ("hum jayenge kal", "bihar"),
    ("janab farmaiye kijiyega", "lucknow"),
    ("bhai yaar scene kya hai", "delhi"),
    ("I will come tomorrow only", "south_indian_english"),
])
def test_regional_markers(text, variant):
    p = prof()
    learn_from_turn(p, text, HumanState(), None, {})
    assert p.preferences["regional_variant"].value == variant


def test_hum_as_we_is_not_bihar():
    p = prof()
    learn_from_turn(p, "hum log aapke saath hain", HumanState(), None, {})
    assert "regional_variant" not in p.preferences


def test_regional_confidence_builds():
    p = prof()
    run(p, "apun ko samajh nahi aaya bantai", 20)
    assert p.preferences["regional_variant"].value == "mumbai" and conf(p, "regional_variant") >= 0.6


def test_likes_dislikes_topics():
    p = prof()
    learn_from_turn(p, "mujhe cricket bahut pasand hai", HumanState(), None, {})
    learn_from_turn(p, "I hate long calls", HumanState(), None, {})
    learn_from_turn(p, "I use it for my coding and study", HumanState(), None, {})
    assert "cricket" in p.likes and "long calls" in p.dislikes
    assert {"coding", "study"} <= set(p.topics)


def test_latest_statement_wins_between_likes_and_dislikes():
    p = prof()
    learn_from_turn(p, "I love spicy food", HumanState(), None, {})
    learn_from_turn(p, "I hate spicy food", HumanState(), None, {})
    assert "spicy food" in p.dislikes and "spicy food" not in p.likes


def test_lists_are_capped_and_deduped():
    p = prof()
    for i in range(40):
        learn_from_turn(p, f"I like thing{i}x", HumanState(), None, {})
    learn_from_turn(p, "I like thing39x", HumanState(), None, {})
    assert len(p.likes) == LIST_CAP and p.likes[-1] == "thing39x" and len(set(p.likes)) == LIST_CAP


def test_empty_text_is_noop():
    p = prof()
    assert learn_from_turn(p, "", HumanState(), None, {}) == []
