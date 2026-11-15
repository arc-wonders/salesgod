"""Agents (personas): name + gender + voice per language tied together (owner 15 Nov: "Arjun" spoke with a woman's voice)."""

import asyncio
import json
import os
from pathlib import Path

import httpx
import numpy as np
import pytest

from acie.agents import (Agent, AgentError, AgentStore, VoiceRef, gendered, grammar_gender, stock_agents,
                         validation_errors, catalogue, voice_chain)
from acie.decision.domains.product_sales import Product, ProductSalesDomain
from acie.decision.engine import DecisionEngine
from acie.expression import TemplateBackend
from acie.pipeline import ACIEPipeline
from acie.schemas import ExpressionParams, TurnInput
from acie.validator import StreamingValidator

ROOT = Path(__file__).resolve().parents[1]
NEEL, IND_EN, VIRAJ = "6209a5682085409fa935f901f0bce950", "ea47f73b0b9c458cb121997ac6e3f619", "015e4f0be89c4c7b834ac30876597fc0"
GIRL_HI, FEMALE_EN = "4d7609058bd34213b1378b29efbde1f1", "2a9605eeafe84974b5b20628d42c0060"


@pytest.fixture
def store(tmp_path):
    return AgentStore(tmp_path / "agents.json")


# ---------------------------------------------------------------- validation

def test_stock_agents_are_valid_and_arjun_is_male_with_male_voices():
    cat = catalogue()
    arjun, priya = stock_agents()
    assert validation_errors(arjun, cat) == [] and validation_errors(priya, cat) == []
    assert arjun.display_name == "Arjun" and arjun.gender == "male"
    assert all(cat[v.voice_id]["gender"] == "male" for v in [*arjun.voices.values(), arjun.backup_voice])
    assert all(cat[v.voice_id]["gender"] == "female" for v in [*priya.voices.values(), priya.backup_voice])


def test_gender_mismatch_is_rejected(store):
    bad = Agent(id="arjun2", display_name="Arjun", gender="male",
                voices={"hi": VoiceRef(voice_id=GIRL_HI), "hinglish": VoiceRef(voice_id=NEEL), "en": VoiceRef(voice_id=IND_EN)})
    errs = validation_errors(bad, catalogue())
    assert any("voices.hi" in e and "female" in e for e in errs)
    with pytest.raises(AgentError, match="female"):
        store.save(bad, create=True)
    with pytest.raises(AgentError, match="backup_voice"):
        store.save(bad.model_copy(update={"voices": {k: VoiceRef(voice_id=NEEL if k != "en" else IND_EN) for k in ("hi", "hinglish", "en")},
                                          "backup_voice": VoiceRef(voice_id=FEMALE_EN)}), create=True)


def test_missing_language_voice_unknown_voice_and_wrong_language_are_rejected():
    cat = catalogue()
    a = Agent(id="x", display_name="X", gender="male", voices={"hi": VoiceRef(voice_id=NEEL)})
    assert any("voices.en" in e for e in validation_errors(a, cat))
    a = Agent(id="x", display_name="X", gender="male", languages=["hi"], voices={"hi": VoiceRef(voice_id="ffffffffffffffff")})
    assert any("not in the voice catalogue" in e for e in validation_errors(a, cat))
    a = Agent(id="x", display_name="X", gender="male", languages=["en"], voices={"en": VoiceRef(voice_id=NEEL)})
    assert any("does not speak en" in e for e in validation_errors(a, cat))
    with pytest.raises(Exception):
        Agent(id="x", display_name="X", gender="robot")


def test_neutral_agent_takes_any_voice_and_speaks_with_the_hindi_voice_gender():
    a = Agent(id="n", display_name="Sam", gender="neutral", languages=["hi"], voices={"hi": VoiceRef(voice_id=GIRL_HI)})
    assert validation_errors(a, catalogue()) == []
    assert grammar_gender(a) == "female"


def test_store_crud_defaults_and_delete_guards(store):
    assert {a.id for a in store.list("fitzone")} == {"arjun", "priya"}  # seeded stock personas
    own = Agent(id="kabir", tenant="fitzone", display_name="Kabir", gender="male",
                voices={"hi": VoiceRef(voice_id=NEEL), "hinglish": VoiceRef(voice_id=NEEL), "en": VoiceRef(voice_id=IND_EN)})
    store.save(own, create=True)
    with pytest.raises(AgentError):
        store.save(own, create=True)
    assert "kabir" in {a.id for a in store.list("fitzone")} and "kabir" not in {a.id for a in store.list("anthropic")}
    store.set_default("fitzone", "kabir")
    with pytest.raises(AgentError):
        store.set_default("anthropic", "kabir")  # another tenant's persona
    with pytest.raises(AgentError):
        store.delete("kabir", [])  # a tenant default
    with pytest.raises(AgentError):
        store.delete("priya", ["fitzone-gym"])  # used by a product
    with pytest.raises(AgentError):
        store.delete("arjun", [])  # the last fallback
    store.delete("priya", [])
    assert store.get("priya") is None


def test_broken_file_falls_back_to_stock(store):
    store.path.write_text("{not json")
    assert store.resolve("whatever", "fitzone").id == "arjun"


# ---------------------------------------------------------------- resolution: product -> agent -> voice

def test_product_agent_voice_resolution_per_language(store):
    store.save(Agent(id="kabir", tenant="fitzone", display_name="Kabir", gender="male",
                     voices={"hi": VoiceRef(voice_id=NEEL), "hinglish": VoiceRef(voice_id=VIRAJ), "en": VoiceRef(voice_id=IND_EN)},
                     backup_voice=VoiceRef(voice_id=VIRAJ)), create=True)
    a = store.resolve("kabir", "fitzone")
    assert [v.voice_id for v in voice_chain(a, "hi")] == [NEEL, VIRAJ]
    assert [v.voice_id for v in voice_chain(a, "hinglish")] == [VIRAJ]  # its own voice is the backup too: once
    assert [v.voice_id for v in voice_chain(a, "en")] == [IND_EN, VIRAJ]
    # a persona that speaks only Hindi still has a voice for an English reply (nearest language)
    only_hi = Agent(id="h", display_name="H", gender="male", languages=["hi"], voices={"hi": VoiceRef(voice_id=NEEL)})
    assert [v.voice_id for v in voice_chain(only_hi, "en")] == [NEEL]


def test_fallback_when_agent_missing_or_another_tenants(store):
    assert store.resolve("ghost", "fitzone").id == "arjun"
    store.save(Agent(id="riya", tenant="anthropic", display_name="Riya", gender="female",
                     voices={"hi": VoiceRef(voice_id=GIRL_HI), "hinglish": VoiceRef(voice_id=GIRL_HI), "en": VoiceRef(voice_id=FEMALE_EN)}),
               create=True)
    assert store.resolve("riya", "fitzone").id == "arjun"  # not fitzone's
    store.set_default("fitzone", "priya")
    assert store.resolve("", "fitzone").id == "priya" and store.resolve("ghost", "fitzone").id == "priya"
    assert store.resolve("riya", "anthropic").display_name == "Riya"


# ---------------------------------------------------------------- Hindi first-person gender

@pytest.mark.parametrize("male,female", [
    ("नमस्ते, मैं अर्जुन बोल रहा हूँ।", "नमस्ते, मैं अर्जुन बोल रही हूँ।"),
    ("मैं आपको लिंक भेजूँगा, माफ़ी चाहता हूँ।", "मैं आपको लिंक भेजूँगी, माफ़ी चाहती हूँ।"),
    ("मैं समझ सकता हूँ, पर मैं नहीं जानता।", "मैं समझ सकती हूँ, पर मैं नहीं जानती।"),
    ("मैं बता चुका हूँ। पर बताऊँगा।", "मैं बता चुकी हूँ। पर बताऊँगी।"),
    ("Main Arjun bol raha hoon, main samajhta hoon.", "Main Arjun bol rahi hoon, main samajhti hoon."),
    ("Link bhej dunga, main karunga.", "Link bhej dungi, main karungi."),
])
def test_gendered_both_ways(male, female):
    assert gendered(male, "female") == female
    assert gendered(female, "male") == male


def test_gendered_leaves_other_people_and_english_alone():
    for t in ("आप चाहते हैं? वो कर सकता है। मैं ठीक हूँ।", "Aap karenge? I am calling about the offer.", "Main {agent} hoon."):
        assert gendered(t, "female") == t
    assert gendered("मैं बोल रहा हूँ", "") == "मैं बोल रहा हूँ"  # no persona: unchanged


def _gym(**kw) -> Product:
    raw = json.loads((ROOT / "data/voice/products.json").read_text())
    return Product.model_validate({**raw["fitzone-gym"], **kw})


async def _greet_and_reply(product: Product, lang: str, text: str):
    pipe = ACIEPipeline(DecisionEngine(ProductSalesDomain(product)), backend=TemplateBackend())
    out = []
    T = lambda t: TurnInput(session_id="s", customer_id="c", text=t)
    sess = pipe.session(T(""))
    sess.expression = ExpressionParams(language=lang)
    plan = await pipe.decision.decide("greet", T(""), sess)
    assert ("agent_female" if product.agent_gender == "female" else "agent_male") in plan.rule_packs
    v = StreamingValidator(plan, sess.expression)
    out.append(" ".join(v.finish()))
    r = await pipe.run_turn(T(text))
    out.append(r.text if hasattr(r, "text") else str(r))
    return out


def test_template_turns_follow_a_female_agent(store):
    """A female persona on the gym product: the greeting names her and template lines use feminine first person."""
    p = _gym(agent_id="priya")
    a = store.resolve(p.agent_id, p.tenant)
    p = p.model_copy(update={"agent_name": a.display_name, "agent_gender": grammar_gender(a)})
    lines = asyncio.run(_greet_and_reply(p, "hi", "धीरे बोलिए, आप कौन हैं?"))
    assert "Priya" in lines[0]
    joined = " ".join(lines)
    assert not any(m in joined for m in ("रहा हूँ", "चाहता हूँ", "सकता हूँ", "ता हूँ")), joined
    # the same product with Arjun keeps the masculine forms
    lines = asyncio.run(_greet_and_reply(_gym(), "hinglish", "dheere boliye, aap kaun hain?"))
    assert "Arjun" in lines[0] and not any(m in " ".join(lines) for m in ("rahi hoon", "ti hoon"))


# ---------------------------------------------------------------- TTS: the configured voice, safe fallback

def _fish(monkeypatch, fail: set[str], seen: list):
    monkeypatch.setenv("FISH_API_KEY", "k")
    monkeypatch.setenv("FISH_VOICE_HI", GIRL_HI)
    monkeypatch.setenv("FISH_VOICE_EN", FEMALE_EN)
    from voice.tts import FishTTS
    raw = np.round(np.full(2400, 0.3) * 32767).astype("<i2").tobytes()

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append((body.get("reference_id"), request.headers.get("model"), body.get("prosody")))
        if body.get("reference_id") in fail:
            return httpx.Response(404, content=b'{"message":"model not found"}')
        return httpx.Response(200, content=raw)
    t = FishTTS()
    t.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return t


def _say(t, lang, **kw) -> bytes:
    async def go():
        return b"".join([c async for c in t.stream("नमस्ते", lang, **kw)])
    return asyncio.run(go())


def test_tts_uses_the_agents_voice_and_rate(monkeypatch):
    seen: list = []
    t = _fish(monkeypatch, set(), seen)
    arjun = stock_agents()[0].model_copy(update={"speaking_rate": 1.1})
    assert _say(t, "hi", voices=voice_chain(arjun, "hi"), speed=arjun.speaking_rate)
    assert seen == [(NEEL, "s2.1-pro-free", {"speed": 1.1, "volume": 0})]
    seen.clear()
    _say(t, "en", voices=voice_chain(arjun, "en"), speed=1.0)
    assert seen == [(IND_EN, "s2.1-pro-free", None)]
    seen.clear()
    _say(t, "hi")  # no agent: the server's default voice, as before
    assert seen == [(GIRL_HI, "s2.1-pro-free", None)]


def test_tts_falls_back_when_a_voice_id_fails(monkeypatch):
    seen: list = []
    t = _fish(monkeypatch, {NEEL}, seen)
    assert _say(t, "hi", voices=voice_chain(stock_agents()[0], "hi"))
    assert [s[0] for s in seen] == [NEEL, VIRAJ]  # the backup (male) voice, not the female env default
    seen.clear()
    t = _fish(monkeypatch, {NEEL, VIRAJ, GIRL_HI}, seen)
    assert _say(t, "hi", voices=voice_chain(stock_agents()[0], "hi"))
    assert [s[0] for s in seen] == [NEEL, VIRAJ, GIRL_HI, None]  # last resort: Fish's own default, never silence


# ---------------------------------------------------------------- API

@pytest.fixture
def client(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    for k, v in {"PLIVO_AUTH_ID": "x", "PLIVO_AUTH_TOKEN": "x", "PLIVO_NUMBER": "+910000000000", "ADMIN_PASSWORD": "pw", "SESSION_SECRET": "s"}.items():
        os.environ.setdefault(k, v)
    from fastapi.testclient import TestClient
    try:
        from voice import server
    except ImportError as e:
        pytest.skip(f"voice server deps missing: {e}")
    data = tmp_path / "voice"
    (data / "calls").mkdir(parents=True)
    (data / "products.json").write_text((ROOT / "data/voice/products.json").read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(server, "DATA", data)
    monkeypatch.setattr(server, "CALLS_DIR", data / "calls")
    monkeypatch.setattr(server.STATE.cfg, "admin_password", "pw")
    c = TestClient(server.app, base_url="https://t")
    assert c.get("/api/agents").status_code == 401
    assert c.post("/api/login", json={"password": "pw"}).status_code == 200
    c.server, c.data = server, data
    yield c


def test_agents_api_crud_and_products_follow_the_agent(client):
    r = client.get("/api/agents?tenant=fitzone").json()
    assert {a["id"] for a in r["agents"]} >= {"arjun", "priya"} and any(v["gender"] == "male" for v in r["voices"])
    body = {"id": "kabir", "tenant": "fitzone", "display_name": "Kabir", "gender": "male",
            "voices": {"hi": {"voice_id": GIRL_HI}, "hinglish": {"voice_id": NEEL}, "en": {"voice_id": IND_EN}}}
    r = client.post("/api/agents", json=body)
    assert r.status_code == 422 and "female" in r.json()["detail"]
    body["voices"]["hi"] = {"voice_id": NEEL}
    assert client.post("/api/agents", json=body).status_code == 200
    body["display_name"] = "Kabir S"
    assert client.put("/api/agents/kabir", json=body).json()["agent"]["display_name"] == "Kabir S"
    # products: default persona = Arjun; point the gym at Priya -> name and gender follow
    st = client.server.STATE
    assert st.products()["fitzone-gym"].agent_name == "Arjun"
    raw = json.loads((client.data / "products.json").read_text())
    raw["fitzone-gym"]["agent_id"] = "priya"
    (client.data / "products.json").write_text(json.dumps(raw))
    gym = st.products()["fitzone-gym"]
    assert (gym.agent_name, gym.agent_gender) == ("Priya", "female")
    assert client.delete("/api/agents/priya").status_code == 422  # used by fitzone-gym
    assert client.delete("/api/agents/kabir").status_code == 200
    assert client.get("/api/agents/kabir").status_code == 404
    # a product cannot name another tenant's agent
    client.post("/api/agents", json={**body, "id": "anna", "tenant": "anthropic", "display_name": "Anna"})
    prod = client.get("/api/products").json()["products"]
    gymp = next(p for p in prod if p["id"] == "fitzone-gym")
    gymp["agent_id"] = "anna"
    assert client.put("/api/products/fitzone-gym", json=gymp).status_code == 400


def test_preview_endpoint_speaks_with_the_agents_voice(client, monkeypatch):
    seen = []

    class FakeTTS:
        accepts_voice = True

        async def stream(self, text, language="en", voices=None, speed=None):
            seen.append((text, language, [v.voice_id for v in voices or []]))
            yield b"\x00\x01" * 800

    monkeypatch.setattr(client.server.STATE, "tts", FakeTTS())
    r = client.post("/api/agents/preview", json={"agent_id": "priya", "language": "hi"})
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav" and r.content[:4] == b"RIFF"
    assert seen[-1][0] == "नमस्ते, मैं Priya बोल रही हूँ। क्या आपके पास दो मिनट हैं?" and seen[-1][2][0] == GIRL_HI
    unsaved = {"display_name": "Dev", "gender": "male", "languages": ["en"], "voices": {"en": {"voice_id": IND_EN}}}
    assert client.post("/api/agents/preview", json={"agent": unsaved, "language": "en"}).status_code == 200
    assert seen[-1][2][0] == IND_EN and "Dev" in seen[-1][0]
    unsaved["voices"]["en"] = {"voice_id": FEMALE_EN}
    assert client.post("/api/agents/preview", json={"agent": unsaved, "language": "en"}).status_code == 422


def test_every_template_line_can_be_spoken_by_a_female_agent():
    """Every first-person masculine form in the hand-written banks (sales, negotiation, psyche moves, call types, call.py) is
    rewritten for a female agent: the gender rule pack finds nothing left (and the male agent's lines come back unchanged)."""
    import ast
    from acie.rules.claims import _FEM, _MASC
    files = ["acie/decision/domains/product_sales.py", "acie/decision/negotiation_banks.py", "acie/decision/negotiation.py",
             "acie/psyche/moves.py", "acie/decision/honest.py", "acie/decision/domains/custom_steps.py", "voice/call.py",
             "acie/safety.py", "acie/decision/constraints.py"]
    left, checked = [], 0
    for f in files:
        for node in ast.walk(ast.parse((ROOT / f).read_text(encoding="utf-8"))):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and _MASC.search(node.value):
                if "\\b" in node.value or "(?" in node.value or "|" in node.value:
                    continue  # a regex, not a spoken line
                checked += 1
                fem = gendered(node.value, "female")
                if _MASC.search(fem):
                    left.append((f, _MASC.search(fem).group(0), node.value[:80]))
                assert not _FEM.search(gendered(fem, "male")), node.value
    assert checked > 50
    assert not left, left
