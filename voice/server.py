"""salesgod voice server: web console + Plivo webhooks + live call sessions.

  uvicorn voice.server:app --host 127.0.0.1 --port 8080     (Caddy terminates TLS in front)
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import os
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import parse_qs

from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response, StreamingResponse

from acie.trace import GATES, trace
from acie.decision.domains.bank_demo import BankDemoDomain
from acie.decision.domains.product_sales import Product, ProductSalesDomain
from acie.decision.engine import DecisionEngine
from acie.expression import OpenAICompatibleBackend, TemplateBackend, make_backend_from_env
from acie.personalization.profile_store import JsonProfileStore
from acie.pipeline import ACIEPipeline

from .call import CallRecord, CallSession
from .plivo import Plivo, e164

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("voice.server")

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.getenv("VOICE_DATA_DIR", ROOT / "data" / "voice"))
CALLS_DIR = DATA / "calls"
STATIC = Path(__file__).resolve().parent / "static"
COOKIE = "sg_session"



def _age(v) -> int | None:
    try:
        a = int(v)
    except (TypeError, ValueError):
        return None
    return a if 10 <= a <= 110 else None


@dataclass
class Config:
    public_url: str = os.getenv("PUBLIC_URL", "https://voice.arkinkansra.in")
    admin_password: str = os.getenv("ADMIN_PASSWORD", "")
    session_secret: str = os.getenv("SESSION_SECRET", "") or secrets.token_hex(16)
    vad_end_silence_ms: int = int(os.getenv("VAD_END_SILENCE_MS", "700"))
    vad_threshold: float = float(os.getenv("VAD_THRESHOLD", "0.5"))
    max_parallel_calls: int = int(os.getenv("MAX_PARALLEL_CALLS", "8"))
    barge_in_ms: int = int(os.getenv("BARGE_IN_MS", "800"))
    barge_in_rms: float = float(os.getenv("BARGE_IN_RMS", "0.04"))
    # TRAI: promotional calls only 09:00-21:00 local time (voice_platform dial gate: calling window)
    call_window: tuple[int, int] = (int(os.getenv("CALL_START_HOUR", "9")), int(os.getenv("CALL_END_HOUR", "21")))
    timezone: str = os.getenv("CALL_TIMEZONE", "Asia/Kolkata")


@dataclass
class App:
    cfg: Config
    plivo: Plivo
    stt: object = None
    tts: object = None
    pipelines: dict = field(default_factory=dict)
    calls: dict = field(default_factory=dict)
    listeners: set = field(default_factory=set)
    backends: dict = field(default_factory=dict)
    store: object = None
    world: object = None
    nlu: object = None
    sims: dict = field(default_factory=dict)
    session_store: object = None  # acie.store (ACIE_SESSION_STORE); None = in-process sessions
    memory: object = None
    sinks: list = field(default_factory=list)
    live_sessions: dict = field(default_factory=dict)  # call_id -> CallSession while the call is up (live map)
    maps: dict = field(default_factory=dict)  # call_id -> latest live-map snapshot (voice/live_map.py)

    # ------------------------------------------------------------------ products

    def products(self) -> dict[str, Product]:
        path = DATA / "products.json"
        if not path.exists():
            default = Product()  # Claude Pro at 60% off, editable from the console
            path.write_text(json.dumps({default.id: default.model_dump()}, ensure_ascii=False, indent=1))
        products = {k: Product.model_validate(v) for k, v in json.loads(path.read_text()).items()}
        return {k: self.with_agent(p) for k, p in products.items()}

    # ------------------------------------------------------------------ agents (personas: name + gender + voices)

    @property
    def agents(self):
        from acie.agents import AgentStore
        store = getattr(self, "_agents", None)
        if store is None or store.path != DATA / "agents.json":
            store = self._agents = AgentStore(DATA / "agents.json")
        return store

    def with_agent(self, product: Product, agent_id: str = "") -> Product:
        """The product as its agent speaks it: the persona's display name and Hindi grammatical gender replace the
        product's agent_name / agent_gender (so templates, LLM persona and gender checks all follow the agent)."""
        try:
            from acie.agents import grammar_gender
            a = self.agents.resolve(agent_id or getattr(product, "agent_id", ""), getattr(product, "tenant", ""))
            return product.model_copy(update={"agent_name": a.display_name,
                                              "agent_gender": grammar_gender(a, self.agents.catalogue())})
        except Exception:  # noqa: BLE001 - a broken agents.json never stops a call; the product's own fields stay
            log.exception("agent resolution failed for product %s", getattr(product, "id", "?"))
            return product

    def agent_for(self, call, product: Product | None = None):
        """The call's persona: the call's own agent_id, else the product's, else the tenant default, else 'arjun'."""
        try:
            return self.agents.resolve(getattr(call, "agent_id", "") or (getattr(product, "agent_id", "") if product else ""),
                                       getattr(product, "tenant", "") if product else "")
        except Exception:  # noqa: BLE001
            log.exception("agent_for failed")
            return None

    def save_products(self, products: dict[str, Product]) -> None:
        (DATA / "products.json").write_text(
            json.dumps({k: v.model_dump() for k, v in products.items()}, ensure_ascii=False, indent=1))

    def make_pipeline(self, call: CallRecord) -> tuple[ACIEPipeline, str]:
        """A fresh pipeline per call: the campaign decides the domain (and its config snapshot)."""
        if call.campaign == "bank_demo":
            domain, prompt = BankDemoDomain(), "बैलेंस, ट्रांसफर, लिमिट, क्रेडिट कार्ड, हाँ, नहीं, ठीक है"
        else:
            product = self.products()[call.campaign]
            if getattr(call, "agent_id", ""):
                product = self.with_agent(product, call.agent_id)  # a call/campaign that names its own agent
            tone = getattr(call, "tone_level", "") or ""
            if tone in ("professional", "mid", "casual") and tone != getattr(product, "tone_level", "mid"):
                product = product.model_copy(update={"tone_level": tone})  # a campaign/sim tone overrides the product's
            from acie import playbook as pbm
            ct = getattr(call, "call_type", "") or ""  # call types (acie/calltypes): "" / sales_pitch = today's flow
            if ct and ct != "sales_pitch":
                from acie.calltypes.factory import build_domain
                domain = build_domain(ct, product, data_dir=DATA, project_id=getattr(call, "project_id", "") or product.tenant,
                                      customer_name=call.customer_name, customer_age=call.customer_age, world=self.world,
                                      playbook=pbm.load(DATA, product.tenant, product.id), customer_id=call.customer_id,
                                      phone=call.to, call_id=call.call_id)
            else:
                domain = ProductSalesDomain(product, customer_name=call.customer_name, world=self.world,
                                            customer_age=call.customer_age,
                                            playbook=pbm.load(DATA, product.tenant, product.id))
            prompt = f"{product.product_name}, {product.seller_name}, subscription, offer, हाँ, नहीं, ठीक है, बताइए"
        pipe = ACIEPipeline(DecisionEngine(domain), backend=self.backends[call.backend], store=self.store,
                            buffer_reply=call.backend != "template", nlu=self.nlu if call.backend != "template" else None,
                            session_store=getattr(self, "session_store", None))
        for sink in getattr(self, "sinks", ()):
            pipe.context_sinks.append(sink.for_call(call.call_id) if hasattr(sink, "for_call") else sink)
        pipe.caller_memory = self.caller_memory_store()
        return pipe, prompt

    def caller_memory_store(self):
        from acie.caller_memory import CallerMemoryStore
        st = getattr(self, "session_store", None)
        return CallerMemoryStore(DATA / "profiles" / "caller_memory", st if hasattr(st, "record_turn") else None)

    # ------------------------------------------------------------------ do-not-call list

    def dnc(self) -> dict[str, float]:
        path = DATA / "dnc.json"
        return json.loads(path.read_text()) if path.exists() else {}

    def add_dnc(self, number: str) -> None:
        d = self.dnc()
        d[number] = time.time()
        (DATA / "dnc.json").write_text(json.dumps(d, indent=1))

    def dial_block(self, to: str, campaign: str, body: dict) -> str | None:
        """One gate for every dial (voice_platform gate.rs, the parts that apply here)."""
        if campaign != "bank_demo":
            allowed = [a for a in (self.products()[campaign].allowed_numbers or []) if str(a).strip()]
            if allowed and to not in allowed:  # not overridable: e.g. a real brand's name on a demo. Empty list = anyone
                return "this product may only call the numbers in its allowed_numbers list (add the number in the product settings)"
        if to in self.dnc() and not body.get("allow_dnc"):  # the operator decides: allow_dnc ("call anyway") dials it
            return "on the do-not-call list (the person asked not to be called); tick 'call anyway' to dial it regardless"
        from datetime import datetime
        from zoneinfo import ZoneInfo
        h = datetime.now(ZoneInfo(self.cfg.timezone)).hour
        a, b = self.cfg.call_window
        if os.getenv("CALL_HOURS", "0") == "1" and not (a <= h < b) and not body.get("allow_after_hours"):  # off by default
            return f"outside calling hours ({a}:00-{b}:00 {self.cfg.timezone})"
        return None

    async def dial_warning(self, to: str) -> str:
        """Things the operator should know before/while the call runs. Never blocks a dial (owner, 15 Nov)."""
        notes = []
        try:
            if "safety_distress" in (await self.store.load(to)).outcomes[-10:]:
                notes.append("a previous call flagged this person as possibly in distress; check the last transcript")
        except Exception:  # noqa: BLE001 - a warning must never stop the call
            pass
        if to in self.dnc():
            notes.append("on the do-not-call list; dialed because the operator chose 'call anyway'")
        return "; ".join(notes)

    def broadcast(self, event: dict) -> None:
        for q in list(self.listeners):
            q.put_nowait(event)
        if event.get("type") == "turn" and event.get("role") == "agent":
            from . import live_map
            live_map.on_turn(self, event)  # pushes a {"type": "map"} event; never raises

    def save_audio(self, call_id: str, n: int, audio) -> None:
        import wave
        import numpy as np
        d = CALLS_DIR / call_id
        d.mkdir(exist_ok=True)
        with wave.open(str(d / f"customer_{n:02d}.wav"), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())

    def save_recording(self, call_id: str, inbound: bytes, out: list, clears: list) -> None:
        """Stereo 8 kHz WAV: left = customer line, right = agent (as played, cut at barge-ins)."""
        import wave
        import numpy as np
        from .audio import ulaw_to_float
        cust = ulaw_to_float(inbound) if inbound else np.zeros(0, dtype=np.float32)
        n = len(cust)
        for off, part, _ in out:
            n = max(n, int(off * 8000) + len(part))
        left, right = np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)
        left[:len(cust)] = cust
        for off, part, queued in out:
            a, seg = int(off * 8000), ulaw_to_float(part)
            # audio queued before a barge-in clear and not yet played by then was dropped by Plivo
            cut = next((int(c * 8000) for c in clears if queued <= c and int(c * 8000) < a + len(seg)), None)
            if cut is not None:
                seg = seg[:max(0, cut - a)]
            right[a:a + len(seg)] = seg
        d = CALLS_DIR / call_id
        d.mkdir(exist_ok=True)
        stereo = np.stack([left, right], axis=1)
        with wave.open(str(d / "call.wav"), "wb") as w:
            w.setnchannels(2)
            w.setsampwidth(2)
            w.setframerate(8000)
            w.writeframes((np.clip(stereo, -1, 1) * 32767).astype("<i2").tobytes())

    def save(self, call: CallRecord) -> None:
        rec = call.public()
        if self.maps.get(call.call_id):
            rec["map"] = self.maps[call.call_id]  # last live-map snapshot, for GET /api/calls/{id}/map
        rec.update(getattr(call, "crm", None) or {})  # CRM links (project_id, crm_customer_id, campaign_id, call_type), set by voice/crm
        (CALLS_DIR / f"{call.call_id}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1))

    async def finish(self, call: CallRecord, pipeline: ACIEPipeline, session_id: str) -> None:
        call.status, call.ended_at = "ended", time.time()
        if any(t.get("intent") == "opt_out" for t in call.turns if t["role"] == "agent"):
            self.add_dnc(call.to)
            call.events.append({"type": "dnc_added", "t": time.time(), "after_turn": len(call.turns)})
        sess = pipeline.sessions.get(session_id)
        st = getattr(self, "session_store", None)
        if sess is not None and hasattr(st, "record_turn"):  # durable call log: the final state + a call_end event
            try:
                st.record_turn(sess, {"type": "call_end", "v": 1, "session_id": session_id, "turn": sess.turn_index, "ts": time.time(),
                                      "status": call.status, "end_reason": getattr(call, "end_reason", ""),
                                      "outcome": sess.context.outcome, "checklist": sess.context.checklist.model_dump(),
                                      "deal_view": sess.context.deal_view})
            except Exception as e:
                log.warning("[%s] call_end event not queued: %s", call.call_id, e)
        sess = await pipeline.forget_session(session_id)
        if sess is not None and not str(call.customer_id).startswith("sim-") and any(u.role == "customer" for u in sess.history):
            try:  # long-term caller memory (how they talk, what happened), read at their next call's first turn
                from acie import caller_memory as cm
                cms = self.caller_memory_store()
                prev = await cms.get(call.customer_id)
                prof_ = await pipeline.store.load(call.customer_id)
                await cms.put(cm.build(prev, sess, call.turns, getattr(self.products().get(call.campaign), "product_name", call.campaign), prof_))
            except Exception as e:
                log.warning("[%s] caller memory not saved: %s", call.call_id, e)
        if sess is not None:
            call.context = sess.context.model_dump(exclude={"events"})
            from . import live_map
            live_map.capture(self, call, sess, self.live_sessions.get(call.call_id))  # final map, saved with the call
        if sess is not None:  # end of call -> long-term memory update
            prof = await pipeline.store.load(call.customer_id)
            pipeline.feedback.end_session(prof, sess)
            await pipeline.feedback.remember(sess, prof)
            await pipeline.store.save(prof)
        self.save(call)
        if getattr(self, "on_call_end", None):  # CRM: apply the outcome to the customer (never raises)
            self.on_call_end({**call.public(), **(getattr(call, "crm", None) or {}), "result": getattr(call, "result", None)})
        self.live_sessions.pop(call.call_id, None)
        self.maps.pop(call.call_id, None)
        self.broadcast({"type": "status", "call": call.public()})


def _register_agent_routes(api: FastAPI, state: App, _require) -> None:
    """/api/agents: the personas (acie/agents.py). Auth like every console route."""
    from acie.agents import (LANGUAGES, Agent, AgentError, grammar_gender, gendered, validation_errors,
                             voice_chain)

    def _used_by(agent_id: str) -> list[str]:
        try:
            raw = json.loads((DATA / "products.json").read_text())
        except Exception:  # noqa: BLE001
            return []
        return [k for k, v in raw.items() if isinstance(v, dict) and v.get("agent_id") == agent_id]

    def _view(a: Agent) -> dict:
        cat = state.agents.catalogue()
        return {**a.model_dump(), "grammar_gender": grammar_gender(a, cat), "used_by": _used_by(a.id),
                "voice_names": {k: cat.get(v.voice_id, {}).get("name", v.voice_id) for k, v in a.voices.items()},
                "problems": validation_errors(a, cat)}

    def _parse(body: dict, agent_id: str | None = None) -> Agent:
        if not isinstance(body, dict):
            raise HTTPException(422, "a JSON object")
        body = {k: v for k, v in body.items() if k not in ("created_at", "updated_at", "grammar_gender", "used_by",
                                                            "voice_names", "problems")}
        if agent_id is not None:
            body["id"] = agent_id
        try:
            return Agent.model_validate(body)
        except Exception as e:  # pydantic: every field error at once
            errs = getattr(e, "errors", None)
            msg = "; ".join(f"{'.'.join(str(x) for x in er['loc'])}: {er['msg']}" for er in errs()) if callable(errs) else str(e)
            raise HTTPException(422, msg[:600])

    @api.get("/api/agents")
    async def agents_list(request: Request, tenant: str | None = None) -> dict:
        _require(request)
        data = state.agents.data()
        cat = state.agents.catalogue()
        return {"agents": [_view(a) for a in state.agents.list(tenant)],
                "tenant_defaults": data.get("tenant_defaults", {}),
                "voices": sorted(cat.values(), key=lambda v: (v["gender"], v["name"])),
                "languages": list(LANGUAGES), "genders": ["male", "female", "neutral"],
                "tone_levels": ["professional", "mid", "casual"], "greeting_styles": ["standard", "warm", "brief"]}

    @api.get("/api/agents/{agent_id}")
    async def agents_get(agent_id: str, request: Request) -> dict:
        _require(request)
        a = state.agents.get(agent_id)
        if a is None:
            raise HTTPException(404, "unknown agent")
        return {"agent": _view(a)}

    @api.post("/api/agents")
    async def agents_create(request: Request) -> dict:
        _require(request)
        a = _parse(await request.json())
        try:
            a = state.agents.save(a, create=True)
        except AgentError as e:
            raise HTTPException(422, str(e))
        return {"agent": _view(a)}

    @api.put("/api/agents/{agent_id}")
    async def agents_update(agent_id: str, request: Request) -> dict:
        _require(request)
        a = _parse(await request.json(), agent_id)
        try:
            a = state.agents.save(a, create=False)
        except KeyError:
            raise HTTPException(404, "unknown agent")
        except AgentError as e:
            raise HTTPException(422, str(e))
        return {"agent": _view(a)}

    @api.delete("/api/agents/{agent_id}")
    async def agents_delete(agent_id: str, request: Request) -> dict:
        _require(request)
        try:
            state.agents.delete(agent_id, _used_by(agent_id))
        except KeyError:
            raise HTTPException(404, "unknown agent")
        except AgentError as e:
            raise HTTPException(422, str(e))
        return {"ok": True}

    @api.put("/api/agents-default/{tenant_id}")
    async def agents_set_default(tenant_id: str, request: Request) -> dict:
        _require(request)
        body = await request.json()
        try:
            state.agents.set_default(tenant_id, str(body.get("agent_id", "")))
        except AgentError as e:
            raise HTTPException(422, str(e))
        return {"ok": True, "tenant_defaults": state.agents.data().get("tenant_defaults", {})}

    @api.post("/api/agents-voices")
    async def agents_add_voice(request: Request) -> dict:
        """Add a Fish voice to the catalogue with its gender and languages (needed before an agent may use it)."""
        _require(request)
        try:
            return {"voice": state.agents.add_voice(await request.json())}
        except ValueError as e:
            raise HTTPException(422, str(e))

    _PREVIEW = {"hi": "नमस्ते, मैं {name} बोल रहा हूँ। क्या आपके पास दो मिनट हैं?",
                "hinglish": "Namaste, main {name} bol raha hoon. Kya aapke paas do minute hain?",
                "en": "Hi, this is {name}. Do you have two minutes?"}

    @api.post("/api/agents/preview")
    async def agents_preview(request: Request) -> Response:
        """Speak a short line with an agent's settings (saved, or unsaved from the editor) so the operator can hear it.
        Body: {agent_id} or {agent: {...}}, language, optional text (<= 200 chars). Returns audio/wav (8 kHz)."""
        _require(request)
        body = await request.json()
        lang = body.get("language", "hi")
        if lang not in LANGUAGES:
            raise HTTPException(422, f"language: one of {list(LANGUAGES)}")
        if isinstance(body.get("agent"), dict):
            a = _parse(body["agent"], str(body["agent"].get("id") or "preview"))
            errs = validation_errors(a, state.agents.catalogue())
            if errs:
                raise HTTPException(422, "; ".join(errs))
        else:
            a = state.agents.get(str(body.get("agent_id", "")))
            if a is None:
                raise HTTPException(404, "unknown agent")
        text = str(body.get("text") or "").strip()[:200] or _PREVIEW[lang]
        text = gendered(text.replace("{name}", a.display_name), grammar_gender(a, state.agents.catalogue()))
        if state.tts is None:
            raise HTTPException(503, "TTS is still loading")
        import io
        import wave
        pcm = bytearray()
        try:
            kw = {"voices": voice_chain(a, lang), "speed": a.speaking_rate} if getattr(state.tts, "accepts_voice", False) else {}
            async for chunk in state.tts.stream(text, lang, **kw):
                pcm += chunk
        except Exception as e:  # noqa: BLE001
            raise HTTPException(502, f"TTS failed: {str(e)[:200]}")
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(8000)
            w.writeframes(bytes(pcm))
        from urllib.parse import quote
        return Response(buf.getvalue(), media_type="audio/wav",
                        headers={"Cache-Control": "no-store", "X-Spoken": quote(text)})


def build_app() -> tuple[FastAPI, App]:
    CALLS_DIR.mkdir(parents=True, exist_ok=True)
    cfg = Config()
    plivo = Plivo(os.environ["PLIVO_AUTH_ID"], os.environ["PLIVO_AUTH_TOKEN"], os.environ["PLIVO_NUMBER"],
                  cfg.public_url, verify=os.getenv("PLIVO_VERIFY_SIGNATURE", "1") != "0")
    state = App(cfg=cfg, plivo=plivo)
    api = FastAPI(title="salesgod voice", docs_url=None, redoc_url=None)

    @api.on_event("startup")
    async def _startup() -> None:
        from .stt import WhisperSTT
        from .tts import FishTTS
        state.stt = await asyncio.to_thread(WhisperSTT)
        from .speaker import embed
        import numpy as np
        await asyncio.to_thread(embed, np.zeros(16000, dtype=np.float32))  # load the voice model now, not mid-call
        state.tts = FishTTS()
        state.store = JsonProfileStore(DATA / "profiles")
        if any(os.getenv(k) for k in ("ACIE_SESSION_STORE", "ACIE_PROFILE_STORE", "ACIE_MEMORY_STORE", "ACIE_NATS_URL")):
            from acie.store import build_stores_from_env  # flags unset = exactly the stores above
            state.session_store, state.store, state.memory, state.sinks = build_stores_from_env(profile_dir=DATA / "profiles")
            if hasattr(state.session_store, "start"):
                await state.session_store.start()  # NATS JetStream call context; never raises (memory fallback)
        from acie.world import WorldKnowledge
        state.world = WorldKnowledge()
        from acie.understanding import LLMUnderstanding
        state.nlu = LLMUnderstanding()
        # qwen: the local vLLM (ACIE_LLM_*); llm: whichever ACIE_EXPR_* points at (the tuned 8B);
        # gemini: real Gemini (GEMINI_* or the Google URL/key in ACIE_EXPR_* / ACIE_PLANNER_*), else the same as llm
        import json as _json
        qwen = OpenAICompatibleBackend(base_url=os.getenv("ACIE_LLM_BASE_URL", "http://localhost:8000/v1"),
                                       model=os.getenv("ACIE_LLM_MODEL", "acie-expression"),
                                       api_key=os.getenv("ACIE_LLM_API_KEY", "EMPTY"),
                                       max_tokens=int(os.getenv("ACIE_LLM_MAX_TOKENS", "320")),
                                       extra_body=_json.loads(os.getenv("ACIE_LLM_EXTRA_BODY", "{}") or "{}"))
        state.backends = {"llm": make_backend_from_env(), "qwen": qwen, "template": TemplateBackend()}
        from acie.expression import make_gemini_backend
        state.backends["gemini"] = make_gemini_backend() or state.backends["llm"]
        state.products()  # writes the default product on first start
        state.pipelines = {"llm": ACIEPipeline(DecisionEngine(BankDemoDomain()), backend=state.backends["llm"],
                                               store=state.store, buffer_reply=True)}
        log.info("ready: whisper=%s fish=%s replies=%s gemini=%s nlu=%s backends=%s", state.stt.model_name, state.tts.model,
                 getattr(state.backends["llm"], "model", "?"), getattr(state.backends["gemini"], "model", "?"),
                 os.getenv("ACIE_LLM_MODEL"), sorted(state.backends))
        asyncio.create_task(_warmup())

    async def _warmup() -> None:
        """First LLM request after a restart takes ~3 s (CUDA graphs, prefix cache); pay it now."""
        from acie.schemas import TurnInput
        pipe = state.pipelines["llm"]
        for text in ("what is my balance", "मेरा बैलेंस बताओ"):
            try:
                await pipe.run_turn(TurnInput(session_id="warmup", customer_id="c1", text=text))
            except Exception as e:
                log.warning("LLM warmup failed: %s", e)
        pipe.sessions.pop("warmup", None)
        try:  # first guided-JSON request compiles the grammar (~2 s): pay it now, not on a caller's first turn
            from acie.decision.domains.product_sales import ProductSalesDomain
            await state.nlu.understand("हाँ बताइए", [("agent", "hello")], ProductSalesDomain.intent_guide, "warmup")
        except Exception as e:
            log.warning("NLU warmup failed: %s", e)

    # ------------------------------------------------------------------ auth

    def _sign(value: str) -> str:
        return hmac.new(cfg.session_secret.encode(), value.encode(), hashlib.sha256).hexdigest()

    def _authed(request: Request) -> bool:
        raw = request.cookies.get(COOKIE, "")
        exp, _, sig = raw.partition(".")
        return bool(exp.isdigit() and int(exp) > time.time() and hmac.compare_digest(sig, _sign(exp)))

    def _require(request: Request) -> None:
        if not _authed(request):
            raise HTTPException(401, "sign in")

    @api.post("/api/login")
    async def login(request: Request) -> Response:
        body = await request.json()
        if not cfg.admin_password or not hmac.compare_digest(str(body.get("password", "")), cfg.admin_password):
            await asyncio.sleep(1.0)
            raise HTTPException(401, "wrong password")
        exp = str(int(time.time()) + 7 * 86400)
        r = JSONResponse({"ok": True})
        r.set_cookie(COOKIE, f"{exp}.{_sign(exp)}", httponly=True, secure=True, samesite="strict", max_age=7 * 86400)
        return r

    # ------------------------------------------------------------------ console

    @api.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC / "console.html", headers={"Cache-Control": "no-cache"})

    @api.get("/ui/{name}")
    async def ui_asset(name: str) -> FileResponse:
        # console scripts/styles (static, no secrets); plain file names only
        path = STATIC / name
        if "/" in name or "\\" in name or name.startswith(".") or path.suffix not in (".js", ".css") or not path.is_file():
            raise HTTPException(404, "not found")
        return FileResponse(path, headers={"Cache-Control": "no-cache"})

    @api.get("/m")
    async def mobile() -> FileResponse:  # phone view (added on the VM 14 Nov, captured into git)
        return FileResponse(STATIC / "mobile.html", headers={"Cache-Control": "no-cache"})

    @api.get("/classic")
    async def classic() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @api.get("/api/me")
    async def me(request: Request) -> dict:
        return {"authed": _authed(request), "from_number": plivo.number if _authed(request) else None}

    @api.post("/api/calls")
    async def start_calls(request: Request) -> dict:
        _require(request)
        body = await request.json()
        numbers = [n for n in str(body.get("numbers", "")).replace(",", "\n").splitlines() if n.strip()]
        live = sum(1 for c in state.calls.values() if c.status in ("dialing", "ringing", "live"))
        if live + len(numbers) > cfg.max_parallel_calls:
            raise HTTPException(429, f"max {cfg.max_parallel_calls} calls at once ({live} running)")
        out = []
        for raw in numbers:
            to = e164(raw)
            if not to:
                out.append({"number": raw, "error": "not a phone number (use +91XXXXXXXXXX)"})
                continue
            campaign = body.get("campaign", "bank_demo")
            if campaign != "bank_demo" and campaign not in state.products():
                raise HTTPException(400, f"unknown product {campaign!r}")
            if why := state.dial_block(to, campaign, body):
                out.append({"number": to, "error": why})
                continue
            warning = await state.dial_warning(to)  # information for the operator, never a blocker
            call = CallRecord(to=to, language=body.get("language", "hi"), backend=body.get("backend", "llm"),
                              campaign=campaign, customer_name=str(body.get("customer_name", ""))[:40],
                              customer_age=_age(body.get("customer_age")),
                              customer_id=body.get("customer_id", "c1") if campaign == "bank_demo" else to)
            call.agent_id = str(body.get("agent_id", "") or "")[:40]  # optional persona override (acie/agents.py)
            if call.backend not in state.backends or call.language not in ("hi", "en", "hinglish"):
                raise HTTPException(400, "bad backend/language")
            state.calls[call.call_id] = call
            try:
                await plivo.dial(call)
            except Exception as e:
                call.status, call.end_reason = "failed", str(e)[:200]
            state.save(call)
            state.broadcast({"type": "status", "call": call.public()})
            out.append({"number": to, "call_id": call.call_id, "status": call.status, "error": call.end_reason,
                        **({"warning": warning} if warning else {})})
        return {"calls": out}

    @api.get("/api/dnc")
    async def list_dnc(request: Request) -> dict:
        _require(request)
        return {"numbers": state.dnc()}

    @api.delete("/api/dnc/{number}")
    async def remove_dnc(number: str, request: Request) -> dict:
        _require(request)
        d = state.dnc()
        d.pop(e164(number) or number, None)
        (DATA / "dnc.json").write_text(json.dumps(d, indent=1))
        return {"ok": True}

    # ------------------------------------------------------------------ text simulator (persona testing)
    # The same pipeline a phone call uses (NLU, decision, natural expression, world lookups, context
    # ledger), minus audio: lets test personas talk to the agent by text. Transcripts in data/voice/sims/.

    @api.post("/api/sim/start")
    async def sim_start(request: Request) -> dict:
        _require(request)
        body = await request.json()
        from acie.schemas import ExpressionParams, TurnInput, Utterance
        from acie.validator import StreamingValidator
        campaign = body.get("campaign", "claude-pro")
        if campaign not in state.products():
            raise HTTPException(400, "unknown campaign")
        call = CallRecord(to="sim", campaign=campaign, language=body.get("language", "hi"),
                          backend=body.get("backend", "llm") if body.get("backend", "llm") in state.backends else "llm",
                          customer_name=str(body.get("customer_name", ""))[:40],
                          customer_age=_age(body.get("customer_age")))
        call.customer_id = f"sim-{call.call_id}"
        call.call_type = str(body.get("call_type") or "")[:40]  # the wizard's test call runs the chosen call type
        call.tone_level = str(body.get("tone_level") or "")[:12]
        call.project_id = str(body.get("project_id") or "")[:40]
        pipe, _ = state.make_pipeline(call)
        sid = call.call_id
        ti = TurnInput(session_id=sid, customer_id=call.customer_id, text="")
        sess = pipe.session(ti)
        sess.expression = ExpressionParams(language=call.language)  # type: ignore[arg-type]
        plan = await pipe.decision.decide("greet", ti, sess)
        text = " ".join(StreamingValidator(plan, sess.expression).finish())
        sess.history.append(Utterance(role="agent", text=text))
        sess.context.record_turn(0, "", plan.intent, False)
        rec = {"sim_id": sid, "persona": str(body.get("persona", ""))[:80], "campaign": campaign,
               "turns": [{"role": "agent", "text": text, "intent": "opening"}]}
        state.sims[sid] = (pipe, rec)
        return {"sim_id": sid, "agent": text}

    @api.post("/api/sim/{sid}")
    async def sim_turn(sid: str, request: Request) -> dict:
        _require(request)
        if sid not in state.sims:
            raise HTTPException(404, "unknown sim (start a new one)")
        pipe, rec = state.sims[sid]
        body = await request.json()
        text = str(body.get("text", "")).strip()[:500]
        from acie.schemas import TurnInput
        spoken: list[str] = []
        t0 = time.perf_counter()
        r = await pipe.run_turn(TurnInput(session_id=sid, customer_id=f"sim-{sid}", text=text), on_sentence=spoken.append)
        ms = round((time.perf_counter() - t0) * 1000)
        sess = pipe.sessions.get(sid)
        nlu = json.loads(sess.dialog["_nlu"]) if sess and sess.dialog.get("_nlu") else None
        interim = [x for x in spoken if x not in r.text]
        out = {"agent": r.text, "interim": interim, "intent": r.plan.intent, "end_call": r.plan.end_call,
               "latency_ms": ms, "nlu": nlu, "issues": r.validation.issues, "state": r.state.model_dump(exclude={"signals"}),
               "language": r.expression.language, "fallback": r.validation.used_fallback,
               "trace": trace(r, sess.dialog if sess else None)}
        rec["turns"] += [{"role": "customer", "text": text}, {"role": "agent", **out}]
        (DATA / "sims").mkdir(exist_ok=True)
        (DATA / "sims" / f"{sid}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1))
        return out

    @api.get("/api/products")
    async def list_products(request: Request) -> dict:
        _require(request)
        from . import live_map
        out = []
        for p in state.products().values():
            d = p.model_dump()  # economics.max_perks is the engine's own field (acie Economics)
            d["real_company"] = live_map.is_real_company(p)
            out.append(d)
        return {"products": out}

    @api.put("/api/products/{product_id}")
    async def put_product(product_id: str, request: Request) -> dict:
        _require(request)
        body = await request.json()
        body["id"] = product_id
        if not product_id.replace("-", "").replace("_", "").isalnum() or len(product_id) > 40:
            raise HTTPException(400, "id: letters, digits, - and _ only")
        from . import live_map
        raw_eco = body.get("economics") if isinstance(body.get("economics"), dict) else None
        body.pop("real_company", None)
        try:
            product = Product.model_validate(body)
        except Exception as e:
            raise HTTPException(400, str(e)[:300])
        if product.agent_id:  # the persona must exist and be this tenant's (or a stock one)
            ag = state.agents.get(product.agent_id)
            if ag is None or ag.tenant not in ("", product.tenant):
                raise HTTPException(400, f"agent_id: unknown agent {product.agent_id!r} for tenant {product.tenant!r}")
        try:  # negotiation range: floor <= every step <= list, no concessions on a real company's product
            # validate what the operator SENT: the Economics model drops out-of-range steps silently on load
            live_map.validate_economics(raw_eco if raw_eco is not None else product.economics, product)
            if raw_eco is not None and "max_perks" in raw_eco:
                live_map.validate_max_perks(product.economics, raw_eco["max_perks"])
        except ValueError as e:
            raise HTTPException(400, str(e))
        products = state.products()
        products[product_id] = product
        state.save_products(products)
        return {"product": product.model_dump()}

    @api.get("/api/playbooks/{product_id}")
    async def get_playbook(product_id: str, request: Request):
        _require(request)
        from acie import playbook as pbm
        p = state.products().get(product_id)
        if p is None:
            raise HTTPException(404, "unknown product")
        pb = pbm.load(DATA, p.tenant, product_id)
        return {"playbook": pb.model_dump() if pb else None}

    @api.put("/api/playbooks/{product_id}")
    async def put_playbook(product_id: str, request: Request):
        _require(request)
        from acie import playbook as pbm
        p = state.products().get(product_id)
        if p is None:
            raise HTTPException(404, "unknown product")
        body = await request.json()
        pb = pbm.Playbook(tenant=p.tenant, product_id=product_id, generated_by=body.get("generated_by", "manual"),
                          nodes=body.get("nodes", []))
        pbm.save(DATA, pb)
        return {"ok": True, "nodes": len(pb.nodes)}

    @api.post("/api/playbooks/{product_id}/generate")
    async def gen_playbook(product_id: str, request: Request):
        _require(request)
        from acie import playbook as pbm
        p = state.products().get(product_id)
        if p is None:
            raise HTTPException(404, "unknown product")
        try:
            pb = await pbm.generate(p, p.tenant)
        except Exception as e:
            raise HTTPException(502, f"planner failed: {type(e).__name__}: {e}")
        pbm.save(DATA, pb)
        return {"ok": True, "nodes": len(pb.nodes), "generated_by": pb.generated_by}

    # ------------------------------------------------------------------ tenants, tree, progress (console)

    def _tenants() -> list[dict]:
        path = DATA / "tenants.json"
        if not path.exists():
            path.write_text(json.dumps([
                {"id": "anthropic", "name": "Anthropic", "color": "#d97757"},
                {"id": "fitzone", "name": "FitZone Gyms", "color": "#2f9e6b"}], indent=1))
        return json.loads(path.read_text())

    @api.get("/api/tenants")
    async def get_tenants(request: Request) -> dict:
        _require(request)
        prods = state.products()
        return {"tenants": [{**t, "products": [p.id for p in prods.values() if p.tenant == t["id"]]} for t in _tenants()]}

    @api.put("/api/tenants/{tenant_id}")
    async def put_tenant(tenant_id: str, request: Request) -> dict:
        _require(request)
        if not tenant_id.replace("-", "").replace("_", "").isalnum() or len(tenant_id) > 40:
            raise HTTPException(400, "id: letters, digits, - and _ only")
        body = await request.json()
        ts = [t for t in _tenants() if t["id"] != tenant_id]
        ts.append({"id": tenant_id, "name": str(body.get("name", tenant_id))[:60], "color": str(body.get("color", "#888"))[:20]})
        (DATA / "tenants.json").write_text(json.dumps(ts, indent=1))
        return {"ok": True}

    @api.get("/api/tree/gates")
    async def tree_gates(request: Request) -> dict:
        _require(request)
        return {"gates": GATES}

    @api.post("/api/playbooks/{product_id}/suggest")
    async def suggest_node(product_id: str, request: Request) -> dict:
        _require(request)
        from acie import playbook as pbm
        p = state.products().get(product_id)
        if p is None:
            raise HTTPException(404, "unknown product")
        desc = str((await request.json()).get("description", "")).strip()[:600]
        if not desc:
            raise HTTPException(400, "describe the case")
        try:
            node = await pbm.generate_node(p, pbm.load(DATA, p.tenant, product_id), desc)
        except Exception as e:
            raise HTTPException(502, f"planner failed: {type(e).__name__}: {e}")
        return {"node": node.model_dump()}

    @api.get("/api/progress")
    async def progress(request: Request) -> dict:
        _require(request)
        roadmap = json.loads((Path(__file__).resolve().parent.parent / "progress.json").read_text())
        calls = []
        for f in CALLS_DIR.glob("*.json"):
            try:
                calls.append(json.loads(f.read_text()))
            except json.JSONDecodeError:
                pass
        seen = {c.get("call_id") for c in calls}
        calls += [c.public() for cid, c in state.calls.items() if cid not in seen]
        intents = lambda c: [t.get("intent") for t in c.get("turns", []) if t.get("role") == "agent"]
        lat = [t["latency_ms"]["first_audio_after_speech_end"] for c in calls for t in c.get("turns", [])
               if isinstance(t.get("latency_ms"), dict) and t["latency_ms"].get("first_audio_after_speech_end")]
        sims = list((DATA / "sims").glob("*.json")) if (DATA / "sims").exists() else []
        by_campaign: dict[str, int] = {}
        for c in calls:
            by_campaign[c.get("campaign") or "bank_demo"] = by_campaign.get(c.get("campaign") or "bank_demo", 0) + 1
        stats = {
            "calls": len(calls),
            "connected": sum(1 for c in calls if any(t.get("role") == "customer" for t in c.get("turns", []))),
            "sold": sum(1 for c in calls if "link_sent" in intents(c)),
            "opt_outs": sum(1 for c in calls if "opt_out" in intents(c)),
            "customer_turns": sum(1 for c in calls for t in c.get("turns", []) if t.get("role") == "customer"),
            "avg_first_audio_ms": round(sum(lat) / len(lat)) if lat else None,
            "p50_first_audio_ms": sorted(lat)[len(lat) // 2] if lat else None,
            "sim_conversations": len(sims),
            "by_campaign": by_campaign,
            "first_call": min((c.get("created_at") or 0 for c in calls), default=None),
        }
        items = [i for a in roadmap["areas"] for i in a["items"]]
        roadmap["overall_pct"] = round(sum(i["pct"] for i in items) / len(items)) if items else 0
        return {"roadmap": roadmap, "stats": stats}

    @api.delete("/api/products/{product_id}")
    async def delete_product(product_id: str, request: Request) -> dict:
        _require(request)
        products = state.products()
        products.pop(product_id, None)
        state.save_products(products)
        return {"ok": True}

    @api.get("/api/calls")
    async def list_calls(request: Request) -> dict:
        _require(request)
        saved = {}
        for p in sorted(CALLS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:200]:
            try:
                saved[p.stem] = json.loads(p.read_text())
                saved[p.stem].pop("map", None)  # heavy, fetched per call from /api/calls/{id}/map
            except json.JSONDecodeError:
                pass
        for cid, c in state.calls.items():
            saved[cid] = c.public()
        return {"calls": sorted(saved.values(), key=lambda c: c["created_at"], reverse=True)}

    @api.get("/api/calls/{call_id}/map")
    async def call_map(call_id: str, request: Request) -> dict:
        _require(request)
        from . import live_map
        cid = "".join(c for c in call_id if c.isalnum())
        if cid in state.maps:
            return {"map": state.maps[cid], "live": cid in state.live_sessions}
        path = CALLS_DIR / f"{cid}.json"
        if not path.exists():
            call = state.calls.get(cid)
            if call is None:
                raise HTTPException(404, "no such call")
            return {"map": live_map.from_record(call.public()), "live": cid in state.live_sessions}
        rec = json.loads(path.read_text())
        if rec.get("map"):
            return {"map": rec["map"], "live": False}
        return {"map": live_map.from_record(rec, state.products().get(rec.get("campaign"))), "live": False}

    @api.get("/api/calls/{call_id}/context")
    async def call_context(call_id: str, request: Request) -> dict:
        """Where the call is: sale checklist + deal + the compact summary the agent works from (live: memory; after a
        restart: the NATS snapshot; ended: the call record). The deal floor is never included."""
        _require(request)
        from acie.context import CallContext
        from acie.schemas import SessionState
        cid = "".join(c for c in call_id if c.isalnum())
        sid = f"call-{cid}"
        sess, source = None, "memory"
        cs = state.live_sessions.get(cid)
        if cs is not None:
            sess = getattr(cs, "pipeline", None) and cs.pipeline.sessions.get(sid)
        if sess is None and cid in state.sims:
            sess = state.sims[cid][0].sessions.get(cid)
            sid = cid
        st = getattr(state, "session_store", None)
        if not sess and hasattr(st, "load_snapshot"):
            snap = await st.load_snapshot(sid) or await st.load_snapshot(cid)
            if snap:
                try:
                    sess, source = SessionState.model_validate(snap), "nats"
                except Exception:
                    sess = None
        if sess:
            view, turn = sess.context.public_view(), sess.turn_index
        else:
            path = CALLS_DIR / f"{cid}.json"
            if not path.exists():
                raise HTTPException(404, "no such call")
            rec = json.loads(path.read_text())
            try:
                ctx = CallContext.model_validate(rec.get("context") or {})
            except Exception:
                ctx = CallContext()
            view, turn, source = ctx.public_view(), sum(1 for t in rec.get("turns", []) if t.get("role") == "customer"), "record"
        return {"call_id": cid, "live": cid in state.live_sessions, "source": source, "turn": turn, **view,
                "nats": st.status() if hasattr(st, "status") else None}

    @api.get("/api/customers/{phone}/memory")
    async def customer_memory(phone: str, request: Request) -> dict:
        """Long-term caller memory (acie/caller_memory.py) + the block the agent gets at the next call."""
        _require(request)
        from acie.caller_memory import about_block
        mem = await state.caller_memory_store().get(phone.strip())
        if not mem:
            raise HTTPException(404, "no memory for this caller yet")
        return {"customer_id": phone, "memory": mem, "about": about_block(mem)}

    @api.get("/api/calls/{call_id}/recording")
    async def recording(call_id: str, request: Request) -> FileResponse:
        _require(request)
        path = CALLS_DIR / "".join(c for c in call_id if c.isalnum()) / "call.wav"
        if not path.exists():
            raise HTTPException(404, "no recording")
        return FileResponse(path, media_type="audio/wav")

    @api.post("/api/calls/{call_id}/hangup")
    async def hangup(call_id: str, request: Request) -> dict:
        _require(request)
        call = state.calls.get(call_id)
        if not call:
            raise HTTPException(404)
        call.end_reason = "operator_hangup"
        await plivo.hangup(call)
        return {"ok": True}

    @api.get("/api/events")
    async def events(request: Request) -> StreamingResponse:
        _require(request)
        q: asyncio.Queue = asyncio.Queue()
        state.listeners.add(q)

        async def gen():
            try:
                yield "retry: 2000\n\n"
                while not await request.is_disconnected():
                    try:
                        ev = await asyncio.wait_for(q.get(), 15)
                        yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
                    except asyncio.TimeoutError:
                        yield ": ping\n\n"
            finally:
                state.listeners.discard(q)

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # ------------------------------------------------------------------ plivo webhooks

    async def _form(request: Request) -> dict[str, list[str]]:
        params = parse_qs((await request.body()).decode("utf-8", "replace"), keep_blank_values=True)
        if not plivo.valid(f"{cfg.public_url}{request.url.path}", params, request.headers):
            log.warning("bad Plivo signature on %s", request.url.path)
            raise HTTPException(403)
        return params

    @api.post("/plivo/answer")
    async def answer(request: Request) -> Response:
        p = {k: v[0] for k, v in (await _form(request)).items()}
        call = next((c for c in state.calls.values() if c.request_uuid and c.request_uuid == p.get("RequestUUID")), None)
        if call is None:
            log.warning("answer for unknown request %s", p.get("RequestUUID"))
            return Response(plivo.hangup_xml(), media_type="application/xml")
        call.call_uuid = p.get("CallUUID")
        call.token = secrets.token_urlsafe(24)
        return Response(plivo.stream_xml(call.token), media_type="application/xml")

    @api.post("/plivo/hangup")
    async def plivo_hangup(request: Request) -> PlainTextResponse:
        p = {k: v[0] for k, v in (await _form(request)).items()}
        call = next((c for c in state.calls.values() if c.request_uuid and c.request_uuid == p.get("RequestUUID")), None)
        if call and call.status in ("dialing", "ringing"):
            call.status, call.end_reason, call.ended_at = "ended", p.get("HangupCause", "not_answered"), time.time()
            state.save(call)
            state.broadcast({"type": "status", "call": call.public()})
        return PlainTextResponse("ok")

    @api.post("/plivo/machine")
    async def plivo_machine(request: Request) -> PlainTextResponse:
        p = {k: v[0] for k, v in (await _form(request)).items()}
        call = next((c for c in state.calls.values()
                     if (c.call_uuid and c.call_uuid == p.get("CallUUID"))
                     or (c.request_uuid and c.request_uuid == p.get("RequestUUID"))), None)
        if call is not None:
            call.machine = str(p.get("Machine", "")).lower() == "true"
            call.events.append({"type": "machine_detection", "t": time.time(), "machine": call.machine,
                                "after_turn": len(call.turns)})
        return PlainTextResponse("ok")

    @api.websocket("/plivo/stream/{token}")
    async def stream(ws: WebSocket, token: str) -> None:
        call = next((c for c in state.calls.values() if c.token and hmac.compare_digest(c.token, token)), None)
        if call is None:
            await ws.close(code=4404)
            return
        call.token = None
        await ws.accept()
        start = None
        while start is None:
            msg = json.loads(await ws.receive_text())
            if msg.get("event") == "start":
                start = msg["start"]
        cs = CallSession(state, call, ws, start["streamId"])
        state.live_sessions[call.call_id] = cs
        try:
            await cs.run()
        finally:
            state.live_sessions.pop(call.call_id, None)

    # ------------------------------------------------------------------ productgen (add a product by name): one self-contained block
    from acie.productgen.api import register as _productgen_register
    _productgen_register(api, _require, DATA, CALLS_DIR)
    # product studio (drafts, facts, recommendations, tests): agent "studio"
    from acie.productgen.studio_api import register as _studio_register
    from .studio_runner import StudioRunner
    _studio_register(api, _require, DATA, runner=StudioRunner(state))

    # ------------------------------------------------------------------ CRM (projects, customers, campaigns, follow-ups): one self-contained block
    from .crm.api import register as _crm_register

    async def _crm_dial(to: str, product_id: str, language: str, customer_name: str, link: dict) -> dict:
        """Operator/auto dial for the CRM: the same gates as POST /api/calls (allowed_numbers, DNC unless link["allow_dnc"], call hours
        only with CALL_HOURS=1, max parallel). A past distress flag is a warning, never a block."""
        live = sum(1 for c in state.calls.values() if c.status in ("dialing", "ringing", "live"))
        if live >= cfg.max_parallel_calls:
            return {"error": f"max {cfg.max_parallel_calls} calls at once ({live} running)"}
        if product_id not in state.products():
            return {"error": f"unknown product {product_id!r}"}
        if why := state.dial_block(to, product_id, {"allow_dnc": bool(link.get("allow_dnc"))}):
            return {"error": why}
        warning = await state.dial_warning(to)
        call = CallRecord(to=to, language=language if language in ("hi", "en", "hinglish") else "hi", backend="llm", campaign=product_id,
                          customer_name=customer_name[:40], customer_id=to)
        call.crm = {"project_id": link["project_id"], "crm_customer_id": link["customer_id"], "campaign_id": link["campaign_id"],
                    "call_type": link["call_type"]}
        call.call_type = link["call_type"]  # read by make_pipeline (acie.calltypes.factory.build_domain)
        call.tone_level = link.get("tone_level", "")  # campaign tone: make_pipeline overrides the product's tone_level
        call.agent_id = str(link.get("agent_id", "") or "")[:40]  # campaign persona, if the campaign names one
        call.project_id = link["project_id"]  # the call-type registry looks up this project's own call types
        state.calls[call.call_id] = call
        try:
            await plivo.dial(call)
        except Exception as e:
            call.status, call.end_reason = "failed", str(e)[:200]
        state.save(call)
        state.broadcast({"type": "status", "call": call.public()})
        return {"call_id": call.call_id, "status": call.status, "error": call.end_reason, **({"warning": warning} if warning else {})}

    def _crm_active() -> tuple[int, set[str]]:
        live = [c for c in state.calls.values() if c.status in ("dialing", "ringing", "live")]
        return len(live), {getattr(c, "crm", {}).get("crm_customer_id", "") for c in live}

    state.crm = _crm_register(api, _require, DATA, CALLS_DIR, dialer=_crm_dial, active=_crm_active)
    state.on_call_end = state.crm["on_call_end"]

    _register_agent_routes(api, state, _require)
    from .pronounce_api import register as _pronounce_register  # TTS pronunciation lexicon: agent "phonetics"
    _pronounce_register(api, state, _require, DATA)

    @api.get("/healthz")
    async def health() -> dict:
        return {"ok": state.stt is not None}

    return api, state


app, STATE = build_app()
