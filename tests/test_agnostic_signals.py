"""Layer 4 is model-agnostic: it reads one TurnSignals, whichever provider produced the parts."""

import json
import os
import subprocess
import sys
import textwrap
import types

import httpx
import pytest

from acie.decision.domains.product_sales import Product, ProductSalesDomain
from acie.decision.engine import DecisionEngine
from acie.pipeline import ACIEPipeline
from acie.providers import SignalHub
from acie.providers.fake import FakeProvider
from acie.providers.http import HttpProvider
from acie.providers.llm import LLMProvider
from acie.providers.registry import _rules
from acie.schemas import SessionState, TurnInput
from acie.signals import IntentVocabulary, PartialSignals, TurnSignals, as_signals, build_vocabulary
from acie.signals_fusion import fuse
from acie.understanding import Understanding

Q = "मेरे पास पहले से Claude है, क्या मैं ये offer अभी भी ले सकता हूँ?"
TRY = "कोशिश तो करना चाहते हैं"
GARBLE = "ठीक है भाई वो जो था ना"
TABLE = {Q: dict(intent="affirm", buying_signal=0.8, question_topic="offer_terms"),
         TRY: dict(intent="affirm", buying_signal=0.85, gist="wants to try it"),
         "मुझे अकेलापन लगता है": dict(intent="personal", gist="feels lonely"),
         GARBLE: dict(intent="unknown", gist="unclear speech, garbled")}
LINES = ["हाँ बोलिए", "कितने का है", Q, TRY, "मुझे अकेलापन लगता है", GARBLE]


def T(text):
    return TurnInput(session_id="s", customer_id="+910000000000", text=text)


class FakeNLU:
    timeout_s = 2.5

    def __init__(self, table):
        self.table = table

    async def understand(self, text, history, intents, about=""):
        return Understanding(**self.table[text]) if text in self.table else None


def partial_for(d: dict, name="llm") -> PartialSignals:
    u = Understanding(**d)
    return PartialSignals(
        provider=name, intent=u.intent, buying_interest=u.buying_signal, question_topic=u.question_topic,
        objection=u.objection, gist=u.gist,
        garbled=any(w in u.gist for w in ("unclear", "garbled")),
        state={"frustration": u.frustration, "confusion": u.confusion, "urgency": u.urgency, "engagement": u.engagement})


async def play(pipe, lines=LINES):
    sess = pipe.session(T(""))
    opening = await pipe.decision.decide("greet", T(""), sess)
    sess.context.record_turn(0, "", opening.intent, False)
    out = []
    for line in lines:
        r = await pipe.run_turn(T(line))
        out.append((r.plan.intent, r.text, round(r.state.frustration, 3), round(r.state.confusion, 3)))
    return out


def new_pipe(nlu=None):
    return ACIEPipeline(DecisionEngine(ProductSalesDomain(Product(), customer_name="Yash")), nlu=nlu)


# ---------------------------------------------------------------- (a) swap the LLM provider, same plans

async def test_fake_provider_with_same_outputs_gives_identical_plans():
    base = await play(new_pipe(FakeNLU(TABLE)))
    pipe = new_pipe()
    pipe._signal_hub = SignalHub([_rules(pipe), FakeProvider({k: partial_for(v) for k, v in TABLE.items()}, name="llm")])
    assert await play(pipe) == base


async def test_provider_name_does_not_change_the_plan():
    pipe = new_pipe()
    pipe._signal_hub = SignalHub([_rules(pipe), FakeProvider({k: partial_for(v) for k, v in TABLE.items()}, name="gpt")])
    assert [x[:2] for x in await play(pipe)] == [x[:2] for x in await play(new_pipe(FakeNLU(TABLE)))]


# ---------------------------------------------------------------- (b) no LLM provider = rules only, as when NLU is down

async def test_removing_llm_provider_equals_nlu_down(monkeypatch):
    down = await play(new_pipe(FakeNLU({})))  # understand() returns None for every turn
    monkeypatch.setenv("ACIE_SIGNAL_PROVIDERS", "rules")
    assert await play(new_pipe(FakeNLU(TABLE))) == down  # the LLM is configured away
    monkeypatch.delenv("ACIE_SIGNAL_PROVIDERS")
    assert await play(new_pipe()) == down  # never had one


async def test_failing_or_slow_provider_is_skipped():
    down = await play(new_pipe())
    for kw in (dict(fail=True), dict(delay_s=0.5, budget_s=0.05)):
        pipe = new_pipe()
        pipe._signal_hub = SignalHub([_rules(pipe), FakeProvider(default=PartialSignals(intent="price"), name="llm", **kw)])
        assert await play(pipe) == down


# ---------------------------------------------------------------- (c) plug in through env only

def _mock_http(monkeypatch, handler):
    real = httpx.AsyncClient

    def factory(*a, **kw):
        kw["transport"] = httpx.MockTransport(handler)
        return real(*a, **kw)
    monkeypatch.setattr("acie.providers.http.httpx.AsyncClient", factory)


async def test_http_provider_plugs_in_via_env(monkeypatch):
    seen = []

    def handler(req):
        body = json.loads(req.content)
        seen.append(body)
        return httpx.Response(200, json={"signals": {"intent": "Yes", "confidence": 0.9, "buying_signal": 0.9,
                                                     "gist": "wants it", "state": {"frustration": 0.2}}})
    _mock_http(monkeypatch, handler)
    monkeypatch.setenv("ACIE_SIGNAL_PROVIDERS", "rules,http")
    monkeypatch.setenv("ACIE_SIGNAL_HTTP_URL", "http://signals.local/v1/turn")
    pipe = new_pipe()
    await play(pipe, ["हाँ बोलिए", "कितने का है", "okay then"])
    sig = TurnSignals.model_validate_json(pipe.sessions["s"].dialog["_signals"])
    assert seen and seen[-1]["text"] == "okay then" and "affirm" in seen[-1]["intents"]
    assert sig.learned and sig.sources["buying_interest"] == "http" and sig.votes["http"] == "affirm"
    assert pipe.sessions["s"].dialog["_nlu"]  # console compat


async def test_http_down_means_rules_only(monkeypatch):
    _mock_http(monkeypatch, lambda req: httpx.Response(500))
    monkeypatch.setenv("ACIE_SIGNAL_PROVIDERS", "rules,http")
    monkeypatch.setenv("ACIE_SIGNAL_HTTP_URL", "http://signals.local/")
    down = await play(new_pipe())
    assert await play(new_pipe()) == down


async def test_model_loader_plugs_in_via_env(monkeypatch, tmp_path):
    from acie.providers import loaders
    (tmp_path / "feature_spec.json").write_text("{}")
    (tmp_path / "acie_meta.json").write_text("{}")

    class FakeXGB:
        @classmethod
        def load(cls, path):
            m = cls()
            m.path = str(path)
            return m

        def classify_with_confidence(self, text, session):
            return "price", 0.95

        def score(self, turn, session):
            return {"anxiety": 0.8}, ["model: anxiety=2"]
    mod = types.ModuleType("training.state_classifier.predict")
    mod.TrainedTurnModel = FakeXGB
    monkeypatch.setitem(sys.modules, "training.state_classifier.predict", mod)
    loaders._CACHE.clear()
    monkeypatch.setenv("ACIE_TURN_MODEL", str(tmp_path))
    monkeypatch.setenv("ACIE_SIGNAL_PROVIDERS", "rules,model")
    pipe = new_pipe()
    out = await play(pipe, ["hmm interesting stuff"])
    sig = TurnSignals.model_validate_json(pipe.sessions["s"].dialog["_signals"])
    assert sig.intent == "price" and sig.sources["intent"] == "model" and out
    assert sig.state["anxiety"] > 0.3 and "turn model" in sig.evidence
    loaders._CACHE.clear()


def test_custom_loader_registers_without_touching_core(monkeypatch, tmp_path):
    from acie.providers import loaders
    monkeypatch.setattr(loaders, "LOADERS", list(loaders.LOADERS))
    loaders.register_loader("onnx", lambda p: p.endswith(".onnx"), lambda p: "ONNX:" + p, first=True)
    loaders._CACHE.clear()
    assert loaders.load_model("m.onnx") == "ONNX:m.onnx"
    loaders._CACHE.clear()


# ---------------------------------------------------------------- (d) determinism

def _parts():
    return [PartialSignals(provider="rules", intent="unknown", state={"frustration": 0.1, "engagement": 0.5}, evidence=["r"]),
            PartialSignals(provider="llm", intent="affirm", buying_interest=0.9, state={"frustration": 0.5, "engagement": 0.7}, gist="g"),
            PartialSignals(provider="model", intent="price", confidence=0.7, state={"frustration": 0.3})]


def test_fusion_is_deterministic_in_process_and_across_processes():
    voc = build_vocabulary(ProductSalesDomain(Product()))
    a = fuse(_parts(), voc).model_dump_json()
    assert a == fuse(_parts(), voc).model_dump_json()
    code = textwrap.dedent("""
        from acie.decision.domains.product_sales import Product, ProductSalesDomain
        from acie.signals import PartialSignals, build_vocabulary
        from acie.signals_fusion import fuse
        import tests.test_agnostic_signals as t
        print(fuse(t._parts(), build_vocabulary(ProductSalesDomain(Product()))).model_dump_json())""")
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=os.getcwd(),
                           env={**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": os.getcwd()}).stdout.strip()
            for seed in ("1", "2", "99")}
    assert outs == {a}


async def test_two_runs_give_identical_plans():
    assert await play(new_pipe(FakeNLU(TABLE))) == await play(new_pipe(FakeNLU(TABLE)))


# ---------------------------------------------------------------- (e) foreign labels

def test_foreign_labels_are_mapped_or_ignored():
    voc = build_vocabulary(ProductSalesDomain(Product()))
    assert voc.map("Yes") == "affirm" and voc.map("CALL-BACK") == "callback"
    assert voc.map("BUY_NOW") == "unknown" and voc.map("") == "unknown" and voc.map(None) == "unknown"
    rules = PartialSignals(provider="rules", intent="unknown")
    s = fuse([rules, PartialSignals(provider="http", intent="BUY_NOW", confidence=1.0,
                                    intent_scores={"BUY_NOW": 0.9, "Yes": 0.4})], voc)
    assert s.intent == "unknown" and s.intent_scores == {"affirm": 0.4}
    s = fuse([rules, PartialSignals(provider="http", intent="agree")], voc)
    assert s.intent == "affirm"


def test_domain_alias_table_extends_the_mapping():
    class D(ProductSalesDomain):
        intent_aliases = {"purchase_intent": "affirm"}
    assert build_vocabulary(D(Product())).map("Purchase-Intent") == "affirm"


async def test_provider_label_never_leaks_into_the_domain():
    pipe = new_pipe()
    pipe._signal_hub = SignalHub([_rules(pipe), FakeProvider(default=PartialSignals(intent="XYZ_SPECIAL"), name="llm")])
    await play(pipe, ["hmm interesting stuff"])
    sig = TurnSignals.model_validate_json(pipe.sessions["s"].dialog["_signals"])
    assert sig.intent != "XYZ_SPECIAL" and "XYZ_SPECIAL" not in pipe.sessions["s"].dialog["last_plan"]


# ---------------------------------------------------------------- the fusion rules

def vocab():
    return build_vocabulary(ProductSalesDomain(Product()))


def test_rules_keep_priority_intents_and_llm_fills_gaps():
    llm = PartialSignals(provider="llm", intent="affirm", buying_interest=0.9)
    assert fuse([PartialSignals(provider="rules", intent="callback"), llm], vocab()).intent == "callback"
    assert fuse([PartialSignals(provider="rules", intent="clarify"), llm], vocab()).intent == "clarify"
    assert fuse([PartialSignals(provider="rules", intent="price"), llm], vocab()).intent == "price"
    for fill in ("unknown", "tell_more", "benefits", "goodbye", "affirm", "deny"):
        assert fuse([PartialSignals(provider="rules", intent=fill), llm], vocab()).intent == "affirm"


def test_trained_model_needs_confidence_and_gates():
    r = PartialSignals(provider="rules", intent="benefits")
    assert fuse([r, PartialSignals(provider="model", intent="price", confidence=0.4)], vocab()).intent == "benefits"
    assert fuse([r, PartialSignals(provider="model", intent="price", confidence=0.9)], vocab()).intent == "price"
    gated = PartialSignals(provider="model", intent="affirm", confidence=0.99)
    assert fuse([r, gated], vocab(), {}).intent == "benefits"
    assert fuse([r, gated], vocab(), {"pending": "close"}).intent == "affirm"
    assert fuse([PartialSignals(provider="rules", intent="callback"),
                 PartialSignals(provider="model", intent="price", confidence=0.99)], vocab()).intent == "callback"


def test_state_dims_fused_half_half_and_buying_priority():
    s = fuse(_parts(), vocab())
    assert s.state["frustration"] == round(0.5 * (0.5 * 0.1 + 0.5 * 0.5) + 0.5 * 0.3, 3)
    assert s.learned_state["frustration"] == round(0.5 * 0.5 + 0.5 * 0.3, 3)
    assert s.buying_interest == 0.9 and s.sources["buying_interest"] == "llm" and s.buying_signal == 0.9


def test_legacy_understanding_still_coerces():
    sg = as_signals(Understanding(intent="deny", buying_signal=0.2, frustration=0.6, gist="unclear"))
    assert sg.heard("deny") and sg.frustration == 0.6 and sg.garbled and sg.buying_interest == 0.2


async def test_default_providers_match_today(monkeypatch):
    monkeypatch.delenv("ACIE_SIGNAL_PROVIDERS", raising=False)
    assert new_pipe()._hub().names == ["rules"]
    assert new_pipe(FakeNLU({}))._hub().names == ["rules", "llm"]
    monkeypatch.setenv("ACIE_SIGNAL_PROVIDERS", "llm,rules,nonsense,http")
    assert new_pipe(FakeNLU({}))._hub().names == ["rules", "llm"]  # rules first, unknown / unconfigured skipped
