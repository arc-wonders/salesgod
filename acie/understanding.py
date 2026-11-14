"""Layer 1.5 - neural understanding (NLU): what the customer MEANS, as structured signals.

The keyword classifier and lexicon scorer only see words they were given, so a live caller who
said "कोशिश तो करना चाहते हैं" (we do want to try) got "sorry, didn't catch that", and "हाँ भेज
दीजिए लिंक" over a noisy line was not a yes. This layer asks the LLM (Qwen3-14B on vLLM, the
same server as the expression layer) to read the turn IN CONTEXT and return JSON constrained by a
schema (vLLM guided decoding), so the output is always parseable:

  intent           one of the domain's intents (the domain describes them)
  buying_signal    0..1  how ready they are to buy right now
  question_topic   product | offer_terms | general | conversation | none
  objection        trust | price | need | timing | language | other | none
  frustration, confusion, urgency, engagement   0..1  (fused with lexicon + prosody in layer 2)
  gist             one short English line: what they mean (repairs garbled speech recognition)

It runs with a hard time budget; on timeout or error the rule layers decide alone, exactly as
before. The decision layer still owns WHAT happens; this only improves what it hears.
"""

from __future__ import annotations

import json
import logging
import os

import httpx
from pydantic import BaseModel, Field

log = logging.getLogger("acie.nlu")


class Understanding(BaseModel):
    intent: str = "unknown"
    buying_signal: float = Field(0.0, ge=0, le=1)
    question_topic: str = "none"
    objection: str = "none"
    frustration: float = Field(0.0, ge=0, le=1)
    confusion: float = Field(0.0, ge=0, le=1)
    urgency: float = Field(0.0, ge=0, le=1)
    engagement: float = Field(0.5, ge=0, le=1)
    gist: str = ""
    # extended human-state dims: None = the server did not return it (old servers / flag off), so it
    # is simply not fused and the lexicon alone decides
    satisfaction: float | None = Field(None, ge=0, le=1)
    anxiety: float | None = Field(None, ge=0, le=1)
    excitement: float | None = Field(None, ge=0, le=1)
    calmness: float | None = Field(None, ge=0, le=1)
    trust: float | None = Field(None, ge=0, le=1)
    certainty: float | None = Field(None, ge=0, le=1)
    cognitive_load: float | None = Field(None, ge=0, le=1)
    attention: float | None = Field(None, ge=0, le=1)
    participation: float | None = Field(None, ge=0, le=1)
    receptiveness: float | None = Field(None, ge=0, le=1)


STATE_DIMS_PROMPT = (
    "\n- satisfaction (0 dislikes, 1 likes what is offered/said), anxiety, excitement, calmness, trust (0 suspects a "
    "scam), certainty (0 hedging), cognitive_load (needs time / overwhelmed), attention, participation (reply "
    "effort), receptiveness (open to the agent's pitch): 0..1 each for THIS turn; 0.5 when neutral."
)
_NEW = ["satisfaction", "anxiety", "excitement", "calmness", "trust", "certainty", "cognitive_load",
        "attention", "participation", "receptiveness"]


def state_dims_enabled() -> bool:
    return os.getenv("ACIE_NLU_STATE_DIMS", "0") == "1"


SYSTEM = """You analyse one turn of a phone conversation for a sales agent. The customer's words come from
speech recognition over a phone line and may be garbled or mis-spelled (Hindi in Devanagari, English, or
Hinglish). Work out what the customer most likely MEANT, using the conversation so far.

Output JSON only, following the schema.
- intent: the single best label from INTENTS (use "unknown" only if nothing fits).
- buying_signal: 0 = clearly not buying, 0.5 = undecided or just asking, 0.8+ ONLY if they clearly say yes to buying /
  "send it" / "I'll take it". A question is never above 0.5 ("UPI se ho jayega?", "can I get it too?"). A filler
  ("haan haan", "ok", "hmm") or agreeing to LISTEN is about 0.5, not a purchase. Annoyance or suspicion lowers it.
- A question about the offer or product (price, duration, payment, privacy, sharing, trust) is NOT "deny" even if
  sceptical. Talking about their own life/feelings is "personal"; chit-chat or unrelated questions are "small_talk".
- question_topic: what they ask about, if they ask something: "product" (what it is / does), "offer_terms"
  (price, payment, link, cancellation, trial), "general" (anything unrelated to the product: weather, news,
  facts, small talk), "conversation" (who are you, why calling, what did you say), or "none".
- objection: their concern if any.
- frustration / confusion / urgency / engagement: how the customer sounds in THIS turn, 0..1.
- gist: what they mean, at most 12 words (e.g. "wants to try it", "asks how to pay")."""


def _schema(intents: list[str]) -> dict:
    lvl = {"type": "number", "minimum": 0, "maximum": 1}
    sch = _base_schema(intents, lvl)
    if state_dims_enabled():
        for d in _NEW:
            sch["properties"][d] = lvl
            sch["required"].append(d)
    return sch


def _base_schema(intents: list[str], lvl: dict) -> dict:
    return {
        "type": "object",
        "properties": {
            "intent": {"type": "string", "enum": intents + ["unknown"]},
            "buying_signal": lvl,
            "question_topic": {"type": "string", "enum": ["product", "offer_terms", "general", "conversation", "none"]},
            "objection": {"type": "string", "enum": ["trust", "price", "need", "timing", "language", "other", "none"]},
            "frustration": lvl, "confusion": lvl, "urgency": lvl, "engagement": lvl,
            "gist": {"type": "string", "maxLength": 90},
        },
        "required": ["intent", "buying_signal", "question_topic", "objection", "frustration", "confusion",
                     "urgency", "engagement", "gist"],
    }


class LLMUnderstanding:
    def __init__(self, base_url: str | None = None, model: str | None = None, timeout_s: float | None = None):
        self.base_url = (base_url or os.getenv("ACIE_LLM_BASE_URL", "http://localhost:8000/v1")).rstrip("/")
        self.model = model or os.getenv("ACIE_LLM_MODEL", "acie-expression")
        self.timeout_s = timeout_s or float(os.getenv("ACIE_NLU_TIMEOUT_S", "2.5"))
        self.extra_body = json.loads(os.getenv("ACIE_LLM_EXTRA_BODY", "{}"))
        self.client = httpx.AsyncClient(timeout=self.timeout_s)

    async def understand(self, text: str, history: list[tuple[str, str]], intents: dict[str, str],
                         about: str = "") -> Understanding | None:
        convo = "\n".join(f"{'AGENT' if r == 'agent' else 'CUSTOMER'}: {t}" for r, t in history[-6:])
        guide = "\n".join(f"- {k}: {v}" for k, v in intents.items())
        user = (f"WHAT THE AGENT IS CALLING ABOUT: {about}\n\nINTENTS:\n{guide}\n\nCONVERSATION SO FAR:\n{convo}\n\n"
                f"CUSTOMER NOW SAYS: {text}")
        body = {"model": self.model, "temperature": 0, "max_tokens": 230 if state_dims_enabled() else 110,
                "messages": [{"role": "system", "content": SYSTEM + (STATE_DIMS_PROMPT if state_dims_enabled() else "")}, {"role": "user", "content": user}],
                "guided_json": _schema(list(intents)), **self.extra_body}
        try:
            r = await self.client.post(f"{self.base_url}/chat/completions", json=body)
            r.raise_for_status()
            return Understanding.model_validate_json(r.json()["choices"][0]["message"]["content"])
        except Exception as e:  # timeout, server down, bad JSON: the rule layers decide alone
            log.info("NLU unavailable: %s", type(e).__name__)
            return None
