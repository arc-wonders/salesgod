import sys
import types

import pytest

from acie.schemas import HumanState, Prosody, SessionState, TurnInput, TurnTiming, Utterance
from acie.state import estimator as E
from acie.state.estimator import HumanStateEstimator, RuleScorer, apply_prosody, state_dims
from acie.understanding import Understanding


def T(text, **kw):
    return TurnInput(session_id="s", customer_id="c", text=text, **kw)


def raw(text, **kw):
    r, _ = RuleScorer().score(T(text, **kw), SessionState(session_id="s", customer_id="c"))
    return r


def test_defaults_and_state_dims():
    h = HumanState()
    d = state_dims(h)
    assert len(d) == 14 and "signals" not in d
    assert (h.satisfaction, h.anxiety, h.excitement, h.calmness, h.trust, h.certainty, h.cognitive_load,
            h.attention, h.participation, h.receptiveness) == (0.5, 0, 0, 0.5, 0.5, 0.5, 0, 0.5, 0.5, 0.5)


@pytest.mark.parametrize("text", ["accha hai, bahut badhiya", "this is great, I like it", "बहुत बढ़िया है, पसंद आया"])
def test_satisfaction_up(text):
    assert raw(text)["satisfaction"] > 0.75


@pytest.mark.parametrize("text", ["bekaar service", "this is terrible, I am disappointed", "ये बकवास है"])
def test_satisfaction_down(text):
    assert raw(text)["satisfaction"] < 0.2


@pytest.mark.parametrize("text", ["accha nahi hai", "not good", "अच्छा नहीं है"])
def test_satisfaction_negated_like_is_not_like(text):
    assert raw(text)["satisfaction"] < 0.5


@pytest.mark.parametrize("text", ["mujhe dar lag raha hai", "I am worried about this", "डर लग रहा है", "safe hai na?"])
def test_anxiety(text):
    assert raw(text)["anxiety"] >= 0.6


@pytest.mark.parametrize("text", ["no worries at all", "koi tension nahi", "टेंशन नहीं है"])
def test_anxiety_masks(text):
    assert raw(text)["anxiety"] == 0


@pytest.mark.parametrize("text", ["wow that is amazing!", "arre wah zabardast", "अरे वाह"])
def test_excitement(text):
    assert raw(text)["excitement"] >= 0.6


def test_excitement_mask():
    assert raw("not excited")["excitement"] == 0


@pytest.mark.parametrize("text", ["koi jaldi nahi aaram se", "take your time no rush", "कोई बात नहीं आराम से"])
def test_calmness_up(text):
    assert raw(text)["calmness"] > 0.7


def test_calmness_down_on_anger():
    assert raw("bakwas hai, kitni baar bolu!!", timing=TurnTiming(barge_in=True))["calmness"] < 0.2


@pytest.mark.parametrize("text", ["is it a fraud? kaise bharosa karun", "this looks like a scam", "ये फ्रॉड है, भरोसा नहीं"])
def test_trust_down(text):
    assert raw(text)["trust"] <= 0.15


@pytest.mark.parametrize("text", ["I trust you", "mujhe bharosa hai", "भरोसा है आप पर", "this is not a scam, genuine"])
def test_trust_up(text):
    assert raw(text)["trust"] > 0.7


def test_not_a_scam_is_not_distrust():
    assert raw("scam nahi hai")["trust"] > 0.5


@pytest.mark.parametrize("text", ["shayad, pata nahi", "maybe, I am not sure", "शायद पता नहीं"])
def test_certainty_down(text):
    assert raw(text)["certainty"] < 0.2


@pytest.mark.parametrize("text", ["bilkul pakka hai", "definitely, I am sure", "पक्का है बिल्कुल"])
def test_certainty_up(text):
    assert raw(text)["certainty"] > 0.75


def test_certainty_mask():
    assert raw("pakka nahi")["certainty"] < 0.2


@pytest.mark.parametrize("text", ["ek minute ruko", "give me a minute, too much", "एक मिनट रुको"])
def test_cognitive_load_phrases(text):
    assert raw(text)["cognitive_load"] >= 0.6


def test_cognitive_load_signals():
    r = raw("um uh hmm... theek", timing=TurnTiming(response_latency_ms=8000))
    assert r["cognitive_load"] >= 0.5
    assert raw("ok")["cognitive_load"] == 0


def test_cognitive_load_repeat_request():
    assert raw("dobara bolo")["cognitive_load"] >= 0.6


def test_participation_attention_levels():
    short = raw("ok")
    long = raw("haan mujhe ye plan samajh aa raha hai, aur kya kya benefits milenge isme?")
    assert long["participation"] > short["participation"] + 0.4
    assert long["attention"] > short["attention"] + 0.4
    q = raw("kya price hai?")
    assert q["participation"] > raw("price")["participation"]


def test_barge_in_raises_participation():
    base = raw("nahi nahi")["participation"]
    assert raw("nahi nahi", timing=TurnTiming(barge_in=True))["participation"] > base


@pytest.mark.parametrize("text", ["haan bhejo", "tell me more", "हां भेजो", "I am interested"])
def test_receptive(text):
    assert raw(text)["receptiveness"] >= 0.8


@pytest.mark.parametrize("text", ["nahi chahiye", "not interested", "नहीं चाहिए", "stop calling"])
def test_not_receptive(text):
    assert raw(text)["receptiveness"] <= 0.1


def test_receptive_mask():
    assert raw("no problem")["receptiveness"] >= 0.5


def test_hinglish_mixed():
    r = raw("bahut badhiya lag raha hai but mujhe thoda dar lag raha hai, safe hai na?")
    assert r["satisfaction"] > 0.6 and r["anxiety"] >= 0.8


def test_core_dims_unchanged():
    r = raw("kitni baar bolna padega, bakwas service!!")
    assert r["frustration"] >= 0.8 and r["engagement"] == 0.8


def test_decay_to_baseline_not_zero():
    est = HumanStateEstimator()
    sess = SessionState(session_id="s", customer_id="c")
    sess.state = est.estimate(T("mujhe bharosa nahi, fraud hai"), sess)
    assert sess.state.trust < 0.4
    for _ in range(12):
        sess.state = est.estimate(T("aap kal ka program mujhe ek line mein samjhaiye zara"), sess)
    assert 0.4 <= sess.state.trust <= 0.6
    assert 0.4 <= sess.state.calmness <= 0.6
    assert sess.state.anxiety < 0.05


def test_decay_zero_baseline_dim():
    est = HumanStateEstimator()
    sess = SessionState(session_id="s", customer_id="c")
    sess.state = est.estimate(T("dar lag raha hai"), sess)
    a1 = sess.state.anxiety
    assert a1 > 0.5
    sess.state = est.estimate(T("aap kal ka program mujhe ek line mein samjhaiye zara"), sess)
    assert 0 < sess.state.anxiety < a1


def test_asymmetric_rise_fast_decay_slow():
    est = HumanStateEstimator()
    sess = SessionState(session_id="s", customer_id="c")
    sess.state = HumanState(anxiety=0.0)
    up = est.estimate(T("I am scared"), sess).anxiety
    assert up == pytest.approx(0.7 * 0.9, abs=0.01)
    sess.state = HumanState(anxiety=0.9)
    down = est.estimate(T("aap kal ka program mujhe ek line mein samjhaiye zara"), sess).anxiety
    assert down == pytest.approx(0.7 * 0.9, abs=0.01)


def test_nlu_fusion_with_new_dims():
    est = HumanStateEstimator()
    sess = SessionState(session_id="s", customer_id="c")
    nlu = Understanding(trust=0.1, anxiety=0.8, gist="worried")
    s = est.estimate(T("aap kal ka program mujhe ek line mein samjhaiye zara"), sess, nlu)
    assert s.trust < 0.5 and s.anxiety > 0.25


def test_nlu_missing_new_fields_parse_and_do_not_fuse():
    u = Understanding.model_validate_json('{"intent":"x","buying_signal":0.5,"frustration":0.2}')
    assert u.trust is None and u.anxiety is None
    est = HumanStateEstimator()
    sess = SessionState(session_id="s", customer_id="c")
    a = est.estimate(T("I trust you"), sess, u)
    b = est.estimate(T("I trust you"), SessionState(session_id="s", customer_id="c"))
    assert a.trust == b.trust


def test_nlu_schema_flag(monkeypatch):
    from acie.understanding import _schema
    assert "trust" not in _schema(["a"])["properties"]
    monkeypatch.setenv("ACIE_NLU_STATE_DIMS", "1")
    sch = _schema(["a"])
    assert "trust" in sch["properties"] and "receptiveness" in sch["required"]


# ---- prosody (synthetic Prosody objects / arousal)

def _p(st=0.0):
    return Prosody(f0_median_hz=200, f0_median_st=st, f0_std_st=2, f0_range_st=6, loudness_db=-20, voiced_ratio=0.5)


def _sess_with_baseline():
    s = SessionState(session_id="s", customer_id="c")
    s.prosody_history = [_p(10.0), _p(10.1), _p(9.9)]
    return s


def test_apply_prosody_high_arousal_negative_gives_anxiety():
    r = {"frustration": 0.3, "confusion": 0, "urgency": 0, "engagement": 0.5, "anxiety": 0.4, "calmness": 0.5,
         "trust": 0.5, "excitement": 0.0}
    o = apply_prosody(r, 0.8, True)
    assert o["anxiety"] > 0.4 + 0.25 and o["calmness"] < 0.3 and o["frustration"] > 0.3


def test_apply_prosody_high_arousal_positive_gives_excitement():
    r = {"frustration": 0, "confusion": 0, "urgency": 0, "engagement": 0.5, "anxiety": 0, "calmness": 0.5,
         "trust": 0.5, "excitement": 0.0}
    o = apply_prosody(r, 0.8, False)
    assert o["excitement"] >= 0.4 and o["anxiety"] == 0


def test_apply_prosody_low_arousal_calmness():
    r = {"frustration": 0, "confusion": 0, "urgency": 0, "engagement": 0.5, "anxiety": 0, "calmness": 0.5,
         "trust": 0.5, "excitement": 0.3}
    o = apply_prosody(r, -0.8, False)
    assert o["calmness"] > 0.75 and o["excitement"] < 0.3


def test_apply_prosody_core_only_dict_still_works():
    o = apply_prosody({"frustration": 0, "confusion": 0, "urgency": 0, "engagement": 0.5}, 0.8, False)
    assert o["engagement"] > 0.5


def test_prosody_off_by_default(monkeypatch):
    monkeypatch.delenv("ACIE_PROSODY", raising=False)
    est = HumanStateEstimator()
    sess = _sess_with_baseline()
    s = est.estimate(T("wow amazing", prosody=_p(14.0)), sess)
    assert not any("voice arousal" in x for x in s.signals)


def test_prosody_on_excited_end_to_end(monkeypatch):
    monkeypatch.setenv("ACIE_PROSODY", "1")
    est = HumanStateEstimator()
    sess = _sess_with_baseline()
    loud = _p(14.0).model_copy(update={"loudness_db": -10, "f0_std_st": 4})
    s = est.estimate(T("haan batao", prosody=loud), sess)
    plain = HumanStateEstimator().estimate(T("haan batao"), _sess_with_baseline())
    assert s.excitement > plain.excitement


def test_prosody_on_flat_voice_calmer(monkeypatch):
    monkeypatch.setenv("ACIE_PROSODY", "1")
    sess = _sess_with_baseline()
    flat = _p(6.0).model_copy(update={"loudness_db": -30, "f0_std_st": 0.5})
    s = HumanStateEstimator().estimate(T("theek hai", prosody=flat), sess)
    plain = HumanStateEstimator().estimate(T("theek hai"), _sess_with_baseline())
    assert s.calmness > plain.calmness


# ---- trained model seam

class _FakeModel:
    def score(self, turn, session):
        return {"trust": 0.0, "satisfaction": 1.0}, ["fake"]


def test_model_fused_5050():
    from acie.state.estimator import _FusedScorer
    sc = _FusedScorer(RuleScorer(), _FakeModel())
    r, ev = sc.score(T("theek hai"), SessionState(session_id="s", customer_id="c"))
    assert r["trust"] == 0.25 and r["satisfaction"] == 0.75 and "turn model" in ev


def test_model_seam_loaded_from_env(monkeypatch, tmp_path):
    mod = types.ModuleType("training.state_classifier.predict")

    class M(_FakeModel):
        @classmethod
        def load(cls, p):
            assert p == str(tmp_path)
            return cls()
    mod.TrainedTurnModel = M
    for n in ("training", "training.state_classifier"):
        monkeypatch.setitem(sys.modules, n, types.ModuleType(n))
    monkeypatch.setitem(sys.modules, "training.state_classifier.predict", mod)
    monkeypatch.setenv("ACIE_TURN_MODEL", str(tmp_path))
    monkeypatch.setattr(E, "_MODEL_FAILED", False)
    est = HumanStateEstimator()
    assert isinstance(est.scorer, E._FusedScorer)
    s = est.estimate(T("theek hai"), SessionState(session_id="s", customer_id="c"))
    assert s.trust < 0.5


def test_model_seam_falls_back_and_logs_once(monkeypatch, caplog):
    monkeypatch.setenv("ACIE_TURN_MODEL", "/nonexistent/model")
    monkeypatch.setitem(sys.modules, "training.state_classifier.predict", None)  # import error
    monkeypatch.setattr(E, "_MODEL_FAILED", False)
    with caplog.at_level("WARNING", logger="acie.state"):
        e1, e2 = HumanStateEstimator(), HumanStateEstimator()
    assert isinstance(e1.scorer, RuleScorer) and isinstance(e2.scorer, RuleScorer)
    assert sum("could not be loaded" in r.message for r in caplog.records) == 1


def test_model_runtime_error_falls_back():
    class Bad:
        def score(self, t, s):
            raise RuntimeError("boom")
    sc = E._FusedScorer(RuleScorer(), Bad())
    r, _ = sc.score(T("fraud"), SessionState(session_id="s", customer_id="c"))
    assert r["trust"] < 0.2


def test_turnresult_state_dump_has_all_dims():
    d = HumanState().model_dump()
    assert {"satisfaction", "receptiveness", "cognitive_load"} <= set(d)
