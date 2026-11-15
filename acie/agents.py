"""Agents (personas): who the caller hears. One record ties the spoken name, the gender (voice AND Hindi grammar) and the
TTS voice per language together, so "Arjun" never speaks with a woman's voice again (owner 15 Nov).

  data/voice/agents.json = {"version": 1,
                            "tenant_defaults": {tenant_id: agent_id},
                            "voices": [catalogue entries added by the operator, same shape as VOICE_CATALOG],
                            "agents": [Agent, ...]}

- An agent belongs to a tenant ("" = a stock persona any tenant may use).
- A product names its agent (Product.agent_id); empty = the tenant's default; unknown / other tenant's = fallback to the
  tenant default, then the stock "arjun". Resolution never raises: a call always has a persona.
- A voice's gender must match the agent's gender (catalogue metadata). A "neutral" agent may use any voice; its Hindi
  first-person grammar then follows the Hindi voice's gender so what it says matches how it sounds.
- `gendered(text, gender)` rewrites Hindi first-person verb forms (Devanagari and Roman) to the agent's gender:
  "मैं बता रहा हूँ" <-> "मैं बता रही हूँ", "karunga" <-> "karungi". Customer-facing forms ("आप चाहते हैं") are untouched.
Concepts from the owner's Rust voice_platform (agents.rs): tenant-owned config, tolerant load + strict save, separate
"invalid" (rejected) from "problems" (warned), delete guards, preview of unsaved settings. Gender is new here.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

log = logging.getLogger("acie.agents")

LANGUAGES = ("hi", "hinglish", "en")
GENDERS = ("male", "female", "neutral")
DEFAULT_MODEL = "s2.1-pro-free"
STOCK_DEFAULT = "arjun"

# Fish Audio public voices, checked 15 Nov 2026 through Fish's own API (tags) and by synthesising a test line on the
# live model (median pitch measured on the reply: male < ~160 Hz, female > ~170 Hz).
VOICE_CATALOG: list[dict] = [
    {"id": "6209a5682085409fa935f901f0bce950", "gain": 0.26, "name": "Neel (Hindi, calm male)", "gender": "male",
     "languages": ["hi", "hinglish"], "model": DEFAULT_MODEL, "f0_hz": 90,
     "notes": "calm, clear, professional, warm; ~450 ms to first audio"},
    {"id": "ca1c77bf09ee4497ac5cd97c5db1f6d5", "gain": 0.49, "name": "Indian English (male)", "gender": "male",
     "languages": ["en", "hinglish"], "model": DEFAULT_MODEL, "f0_hz": 118,
     "notes": "Indian accent, calm, measured, warm; ~350 ms to first audio"},
    {"id": "ea47f73b0b9c458cb121997ac6e3f619", "gain": 0.6, "name": "Finance (American English male)", "gender": "male",
     "languages": ["en"], "model": DEFAULT_MODEL, "f0_hz": 140,
     "notes": "American, clear, professional; Arjun's English voice since 15 Nov (owner: no Indian accent in English)"},
    {"id": "4746f1b456494387993af9041d0516c4", "gain": 0.28, "name": "w2w (English male, warm)", "gender": "male",
     "languages": ["en"], "model": DEFAULT_MODEL, "f0_hz": 119, "notes": "warm, calm, smooth; very loud raw (RMS ~6600)"},
    {"id": "015e4f0be89c4c7b834ac30876597fc0", "gain": 0.39, "name": "Viraj (Hindi/English male)", "gender": "male",
     "languages": ["hi", "hinglish", "en"], "model": DEFAULT_MODEL, "f0_hz": 152,
     "notes": "deep, smooth, professional; backup for male agents"},
    {"id": "6366ea69a97d41ada9eb855a64e56fba", "gain": 0.54, "name": "Bunty (Hinglish male)", "gender": "male",
     "languages": ["hinglish", "hi", "en"], "model": DEFAULT_MODEL, "f0_hz": 125,
     "notes": "conversational, friendly, Indian accent"},
    {"id": "fa3528ac39d546429da5f65612df7b4c", "gain": 0.75, "name": "Indian English 2 (male)", "gender": "male",
     "languages": ["en"], "model": DEFAULT_MODEL, "f0_hz": 119, "notes": "clear, confident; slower first audio (~760 ms)"},
    {"id": "4d7609058bd34213b1378b29efbde1f1", "name": "Girl Hindi (female)", "gender": "female",
     "languages": ["hi", "hinglish"], "model": DEFAULT_MODEL, "f0_hz": 273,
     "notes": "the voice every call used until 15 Nov (FISH_VOICE_HI)"},
    {"id": "2a9605eeafe84974b5b20628d42c0060", "name": "Female Voice (English)", "gender": "female",
     "languages": ["en", "hinglish"], "model": DEFAULT_MODEL, "f0_hz": 175,
     "notes": "calm, smooth (FISH_VOICE_EN until 15 Nov)"},
    {"id": "b1a1d760d9604bdb957a56eec3460f1b", "name": "Indian Lady (English)", "gender": "female",
     "languages": ["en", "hinglish"], "model": DEFAULT_MODEL, "notes": "friendly, clear, Indian accent; backup for female agents"},
]


class VoiceRef(BaseModel):
    voice_id: str = Field(min_length=8, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    model: str = Field(DEFAULT_MODEL, min_length=1, max_length=40)


class Agent(BaseModel):
    id: str = Field(min_length=1, max_length=40, pattern=r"^[a-z0-9][a-z0-9_-]*$")
    tenant: str = Field("", max_length=40)  # "" = stock persona, any tenant may use it
    display_name: str = Field(min_length=1, max_length=40)  # what it says: "Arjun"
    gender: Literal["male", "female", "neutral"] = "male"
    languages: list[Literal["hi", "hinglish", "en"]] = Field(default_factory=lambda: ["hi", "hinglish", "en"])
    voices: dict[str, VoiceRef] = Field(default_factory=dict)  # language -> voice
    backup_voice: VoiceRef | None = None
    speaking_rate: float = Field(1.0, ge=0.8, le=1.25)  # Fish prosody speed; 1.0 = sent as nothing
    tone_level: Literal["professional", "mid", "casual"] = "mid"  # used when the product sets none
    greeting_style: Literal["standard", "warm", "brief"] = "standard"
    description: str = Field("", max_length=200)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)

    @field_validator("display_name", "description")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip()

    @field_validator("languages")
    @classmethod
    def _langs(cls, v: list[str]) -> list[str]:
        out = list(dict.fromkeys(v))
        if not out:
            raise ValueError("at least one language")
        return out

    @field_validator("voices")
    @classmethod
    def _voice_langs(cls, v: dict) -> dict:
        bad = [k for k in v if k not in LANGUAGES]
        if bad:
            raise ValueError(f"unknown language(s) {bad}; use {list(LANGUAGES)}")
        return v


def stock_agents() -> list[Agent]:
    now = time.time()
    return [
        Agent(id="arjun", display_name="Arjun", gender="male", description="Stock male persona (Hindi / Hinglish / English)",
              voices={"hi": VoiceRef(voice_id="6209a5682085409fa935f901f0bce950"),
                      "hinglish": VoiceRef(voice_id="6209a5682085409fa935f901f0bce950"),
                      "en": VoiceRef(voice_id="ea47f73b0b9c458cb121997ac6e3f619")},
              backup_voice=VoiceRef(voice_id="015e4f0be89c4c7b834ac30876597fc0"), created_at=now, updated_at=now),
        Agent(id="priya", display_name="Priya", gender="female", description="Stock female persona (Hindi / Hinglish / English)",
              voices={"hi": VoiceRef(voice_id="4d7609058bd34213b1378b29efbde1f1"),
                      "hinglish": VoiceRef(voice_id="4d7609058bd34213b1378b29efbde1f1"),
                      "en": VoiceRef(voice_id="2a9605eeafe84974b5b20628d42c0060")},
              backup_voice=VoiceRef(voice_id="b1a1d760d9604bdb957a56eec3460f1b"), created_at=now, updated_at=now),
    ]


# ------------------------------------------------------------------ validation

# Output gain per voice so every voice reaches the line at the old voices' level (RMS ~1800 of 32768 at 8 kHz, measured
# on the real call path 15 Nov: Neel came out at RMS 7008 / peak 30337, 3.6x louder than the old voices -> distorted on calls).
TARGET_RMS = 1850


def voice_gain(voice_id: str | None) -> float:
    if not voice_id:
        return 1.0
    for v in VOICE_CATALOG:
        if v["id"] == voice_id:
            return float(v.get("gain", 1.0))
    return 1.0


def catalogue(extra: list[dict] | None = None) -> dict[str, dict]:
    out = {v["id"]: v for v in VOICE_CATALOG}
    for v in extra or []:
        if isinstance(v, dict) and v.get("id") and v.get("gender") in GENDERS:
            out[v["id"]] = {**v, "languages": list(v.get("languages") or LANGUAGES)}
    return out


def validate_catalogue_entry(v: dict) -> dict:
    errs = []
    vid = str(v.get("id", "")).strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{8,64}", vid):
        errs.append("id: the Fish voice (reference) id, 8-64 letters/digits")
    if v.get("gender") not in GENDERS:
        errs.append(f"gender: one of {list(GENDERS)}")
    langs = [x for x in (v.get("languages") or []) if x in LANGUAGES]
    if not langs:
        errs.append(f"languages: one or more of {list(LANGUAGES)}")
    name = str(v.get("name", "")).strip()[:60]
    if not name:
        errs.append("name: required")
    if errs:
        raise ValueError("; ".join(errs))
    return {"id": vid, "name": name, "gender": v["gender"], "languages": langs,
            "model": str(v.get("model") or DEFAULT_MODEL)[:40], "notes": str(v.get("notes", ""))[:200]}


def validation_errors(agent: Agent, cat: dict[str, dict]) -> list[str]:
    """Hard errors: the agent cannot be saved. Gender mismatch, a language with no voice, a voice for the wrong language."""
    errs: list[str] = []
    for lang in agent.languages:
        if lang not in agent.voices:
            errs.append(f"voices.{lang}: a voice is required for every language the agent speaks")
    refs = [(f"voices.{k}", k, v) for k, v in agent.voices.items()]
    if agent.backup_voice:
        refs.append(("backup_voice", None, agent.backup_voice))
    for field, lang, ref in refs:
        meta = cat.get(ref.voice_id)
        if meta is None:
            errs.append(f"{field}: voice {ref.voice_id} is not in the voice catalogue (add it with its gender first)")
            continue
        if agent.gender != "neutral" and meta["gender"] != agent.gender:
            errs.append(f"{field}: voice '{meta['name']}' is {meta['gender']} but {agent.display_name} is {agent.gender}")
        if lang and lang not in meta.get("languages", LANGUAGES):
            errs.append(f"{field}: voice '{meta['name']}' does not speak {lang} (speaks {meta.get('languages')})")
    return errs


def grammar_gender(agent: Agent | None, cat: dict[str, dict] | None = None) -> str:
    """male | female: the form Hindi verbs take when the agent speaks of itself."""
    if agent is None:
        return "male"
    if agent.gender in ("male", "female"):
        return agent.gender
    cat = cat or catalogue()
    for lang in ("hi", "hinglish", "en"):
        ref = agent.voices.get(lang)
        g = cat.get(ref.voice_id, {}).get("gender") if ref else None
        if g in ("male", "female"):
            return g
    return "male"


# ------------------------------------------------------------------ store

class AgentError(ValueError):
    pass


class AgentStore:
    """data/voice/agents.json. Tolerant load (a broken record is skipped and logged), strict save."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()

    def _read(self) -> dict:
        try:
            raw = json.loads(self.path.read_text())
        except FileNotFoundError:
            raw = None
        except Exception:  # noqa: BLE001 - a corrupt file must not stop calls
            log.exception("agents.json unreadable; using the stock agents")
            return self._seed(write=False)
        if not isinstance(raw, dict) or not isinstance(raw.get("agents"), list):
            return self._seed(write=raw is None)
        return raw

    def _seed(self, write: bool) -> dict:
        data = {"version": 1, "tenant_defaults": {}, "voices": [], "agents": [a.model_dump() for a in stock_agents()]}
        if write:
            self._write(data)
        return data

    def _write(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1))
        os.replace(tmp, self.path)

    # reads
    def data(self) -> dict:
        return self._read()

    def catalogue(self) -> dict[str, dict]:
        return catalogue(self._read().get("voices"))

    def agents(self) -> dict[str, Agent]:
        out: dict[str, Agent] = {}
        for a in self._read().get("agents", []):
            try:
                ag = Agent.model_validate(a)
                out[ag.id] = ag
            except Exception as e:  # noqa: BLE001
                log.warning("agents.json: skipped a broken agent %r: %s", (a or {}).get("id") if isinstance(a, dict) else a, e)
        if not out:
            out = {a.id: a for a in stock_agents()}
        return out

    def list(self, tenant: str | None = None) -> list[Agent]:
        return [a for a in self.agents().values() if tenant is None or a.tenant in ("", tenant)]

    def get(self, agent_id: str) -> Agent | None:
        return self.agents().get(agent_id or "")

    def tenant_default(self, tenant: str) -> str:
        d = self._read().get("tenant_defaults", {}) or {}
        return d.get(tenant or "", "") or ""

    # resolution (never raises)
    def resolve(self, agent_id: str = "", tenant: str = "") -> Agent:
        agents = self.agents()
        for cand in (agent_id, self.tenant_default(tenant), STOCK_DEFAULT):
            a = agents.get(cand or "")
            if a is not None and a.tenant in ("", tenant or ""):
                if cand != agent_id and agent_id:
                    log.warning("agent %r missing or not tenant %r's: using %r", agent_id, tenant, a.id)
                return a
        return next(iter(agents.values()), None) or stock_agents()[0]

    def problems(self, agent: Agent) -> list[str]:
        return validation_errors(agent, self.catalogue())

    # writes
    def save(self, agent: Agent, create: bool) -> Agent:
        with self._lock:
            data = self._read()
            errs = validation_errors(agent, catalogue(data.get("voices")))
            if errs:
                raise AgentError("; ".join(errs))
            rows = [a for a in data["agents"] if isinstance(a, dict)]
            exists = any(a.get("id") == agent.id for a in rows)
            if create and exists:
                raise AgentError(f"agent {agent.id} already exists")
            if not create and not exists:
                raise KeyError(agent.id)
            old = next((a for a in rows if a.get("id") == agent.id), None)
            agent = agent.model_copy(update={"updated_at": time.time(),
                                             "created_at": (old or {}).get("created_at", time.time())})
            data["agents"] = [a for a in rows if a.get("id") != agent.id] + [agent.model_dump()]
            self._write(data)
            return agent

    def delete(self, agent_id: str, used_by: list[str]) -> None:
        with self._lock:
            data = self._read()
            rows = [a for a in data["agents"] if isinstance(a, dict)]
            if not any(a.get("id") == agent_id for a in rows):
                raise KeyError(agent_id)
            if agent_id == STOCK_DEFAULT:
                raise AgentError("the stock agent 'arjun' is every call's last fallback and cannot be deleted")
            if used_by:
                raise AgentError(f"used by product(s): {', '.join(used_by)}; point them at another agent first")
            if agent_id in (data.get("tenant_defaults") or {}).values():
                raise AgentError("it is a tenant's default agent; choose another default first")
            data["agents"] = [a for a in rows if a.get("id") != agent_id]
            self._write(data)

    def set_default(self, tenant: str, agent_id: str) -> None:
        with self._lock:
            data = self._read()
            a = next((x for x in data["agents"] if isinstance(x, dict) and x.get("id") == agent_id), None)
            if a is None or a.get("tenant", "") not in ("", tenant):
                raise AgentError(f"agent {agent_id} is not available to tenant {tenant}")
            data.setdefault("tenant_defaults", {})[tenant] = agent_id
            self._write(data)

    def add_voice(self, entry: dict) -> dict:
        v = validate_catalogue_entry(entry)
        with self._lock:
            data = self._read()
            data["voices"] = [x for x in data.get("voices", []) if isinstance(x, dict) and x.get("id") != v["id"]] + [v]
            self._write(data)
        return v


# ------------------------------------------------------------------ voices for a call

def voice_chain(agent: Agent | None, language: str) -> list[VoiceRef]:
    """Voices to try, in order, for one reply: the language's own, the agent's other-language voice of the same family
    (hinglish <-> hi, then en), then the backup. The env default is TTS's own last resort."""
    if agent is None:
        return []
    order = {"hi": ("hi", "hinglish", "en"), "hinglish": ("hinglish", "hi", "en"), "en": ("en", "hinglish", "hi")}
    seen, out = set(), []
    for lang in order.get(language, ("hi", "hinglish", "en")):
        ref = agent.voices.get(lang)
        if ref and ref.voice_id not in seen:
            seen.add(ref.voice_id)
            out.append(ref)
            break  # the language's own voice (or the nearest), then straight to the backup
    if agent.backup_voice and agent.backup_voice.voice_id not in seen:
        out.append(agent.backup_voice)
    return out


# ------------------------------------------------------------------ Hindi first-person gender

_D = "[\u0900-\u0963\u0966-\u097F]"  # Devanagari letters and signs, not the danda (U+0964/5)
_HOON = r"(?:हूँ|हूं|हु|हूँ)"
_RHOON = r"(?:hoon|hun|hu|hoo)"
# (masculine, feminine) pairs. Only first-person forms: "हूँ"/"hoon" and the -ूँगा/-unga future are always "I".
_DEV_PAIRS = [
    (re.compile(r"(%s+)ता(\s+%s)" % (_D, _HOON)), r"\1ता\2", r"\1ती\2"),
    (re.compile(r"(%s+)ती(\s+%s)" % (_D, _HOON)), r"\1ता\2", r"\1ती\2"),
    (re.compile(r"(?<!%s)(रहा|गया|चुका|सका|आया|हुआ|बैठा|लगा)(\s+%s)" % (_D, _HOON)), None, None),
    (re.compile(r"(?<!%s)(रही|गयी|गई|चुकी|सकी|आई|आयी|हुई|बैठी|लगी)(\s+%s)" % (_D, _HOON)), None, None),
    (re.compile(r"(%s+?)(ूँगा|ूंगा|ऊँगा|ऊंगा)(?!%s)" % (_D, _D)), None, None),
    (re.compile(r"(%s+?)(ूँगी|ूंगी|ऊँगी|ऊंगी)(?!%s)" % (_D, _D)), None, None),
]
_PERF_M2F = {"रहा": "रही", "गया": "गई", "चुका": "चुकी", "सका": "सकी", "आया": "आई", "हुआ": "हुई", "बैठा": "बैठी", "लगा": "लगी"}
_PERF_F2M = {"रही": "रहा", "गयी": "गया", "गई": "गया", "चुकी": "चुका", "सकी": "सका", "आई": "आया", "आयी": "आया",
             "हुई": "हुआ", "बैठी": "बैठा", "लगी": "लगा"}
_FUT_M2F = {"ूँगा": "ूँगी", "ूंगा": "ूंगी", "ऊँगा": "ऊँगी", "ऊंगा": "ऊंगी"}
_FUT_F2M = {v: k for k, v in _FUT_M2F.items()}
# "माफ़ी चाहता", "मैं नहीं जानता" (no हूँ in the negative): only right after मैं ... नहीं
_DEV_NEG = re.compile(r"(मैं\s+(?:%s+\s+){0,3}?नहीं\s+(?:%s+\s+)?)(%s+)(ता|ती)(?=[\s,.!?।]|$)(?!\s+(?:है|हैं|था|थे|थी))"
                      % (_D, _D, _D))
_ROM_TA = re.compile(r"\b([a-z]+)(ta|ti)(\s+%s)\b" % _RHOON, re.I)
_ROM_PERF = re.compile(r"\b(raha|rahi|gaya|gayi|gai|chuka|chuki|saka|saki)(\s+%s)\b" % _RHOON, re.I)
_ROM_PERF_M2F = {"raha": "rahi", "gaya": "gayi", "chuka": "chuki", "saka": "saki"}
_ROM_PERF_F2M = {"rahi": "raha", "gayi": "gaya", "gai": "gaya", "chuki": "chuka", "saki": "saka"}
# -unga / -oonga future, first person: karunga, bataunga, bhejunga, dunga, lunga, chahunga, karoonga, samjhaunga
_ROM_FUT = re.compile(r"\b([a-z]*?(?:u|oo|uu))(nga|ngi)\b", re.I)
_ROM_FUT_SKIP: set[str] = set()  # words that only look like a future verb


def _case(src: str, dst: str) -> str:
    return dst.upper() if src.isupper() else (dst[0].upper() + dst[1:] if src[:1].isupper() else dst)


def gendered(text: str, gender: str) -> str:
    """Hindi first-person verb forms -> the agent's gender. "male" / "female"; anything else returns text unchanged."""
    if not text or gender not in ("male", "female"):
        return text
    fem = gender == "female"
    s = text
    if re.search(_D, s):
        s = _DEV_PAIRS[0 if fem else 1][0].sub(lambda m: m.group(1) + ("ती" if fem else "ता") + m.group(2), s)
        if fem:
            s = _DEV_PAIRS[2][0].sub(lambda m: _PERF_M2F[m.group(1)] + m.group(2), s)
            s = _DEV_PAIRS[4][0].sub(lambda m: m.group(1) + _FUT_M2F[m.group(2)], s)
        else:
            s = _DEV_PAIRS[3][0].sub(lambda m: _PERF_F2M[m.group(1)] + m.group(2), s)
            s = _DEV_PAIRS[5][0].sub(lambda m: m.group(1) + _FUT_F2M[m.group(2)], s)
        s = _DEV_NEG.sub(lambda m: m.group(1) + m.group(2) + ("ती" if fem else "ता"), s)
        s = re.sub(r"माफ़ी चाह(?:ता|ती)(?!%s)" % _D, "माफ़ी चाह" + ("ती" if fem else "ता"), s)
    if re.search(r"[A-Za-z]", s):
        s = _ROM_TA.sub(lambda m: m.group(1) + _case(m.group(2), "ti" if fem else "ta") + m.group(3), s)
        s = _ROM_PERF.sub(lambda m: _case(m.group(1), (_ROM_PERF_M2F if fem else _ROM_PERF_F2M).get(m.group(1).lower(),
                                                                                                m.group(1).lower())) + m.group(2), s)

        def fut(m: re.Match) -> str:
            word = m.group(0)
            if word.lower() in _ROM_FUT_SKIP or len(m.group(1)) < 2:
                return word
            return m.group(1) + _case(m.group(2), "ngi" if fem else "nga")
        s = _ROM_FUT.sub(fut, s)
    return s


def gender_from_packs(rule_packs) -> str:
    """The agent gender a TurnPlan carries as its 'agent_male' / 'agent_female' rule pack ("" = not a persona plan)."""
    for p in rule_packs or ():
        name = p if isinstance(p, str) else getattr(p, "name", "")
        if name in ("agent_male", "agent_female"):
            return name[6:]
    return ""
