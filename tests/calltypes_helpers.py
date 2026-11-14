"""Shared helpers for the call-type tests: scripted conversations through the real pipeline (template backend, no LLM)."""

import json
from datetime import datetime
from zoneinfo import ZoneInfo

from acie.calltypes.factory import build_domain
from acie.decision.domains.product_sales import Product
from acie.decision.engine import DecisionEngine
from acie.pipeline import ACIEPipeline
from acie.schemas import ExpressionParams, TurnInput

NOW = datetime(2026, 11, 15, 10, 0, tzinfo=ZoneInfo("Asia/Kolkata"))  # a Sunday


def T(text, cid="+910000000001"):
    return TurnInput(session_id="s", customer_id=cid, text=text)


class Call:
    """One scripted call: .say(text) -> TurnResult; .agent = every agent line; .result = the CallResult dict so far."""

    def __init__(self, call_type, product=None, lang="hinglish", **kw):
        self.domain = build_domain(call_type, product or Product(), customer_name=kw.pop("customer_name", "Yash"), now_fn=lambda: NOW, **kw)
        self.pipe = ACIEPipeline(DecisionEngine(self.domain))
        self.sess = self.pipe.session(T(""))
        self.sess.expression = ExpressionParams(language=lang)
        self.lang = lang
        self.lines: list[tuple[str, str, str]] = []  # (role, intent, text)
        self.last = None

    async def open(self):
        plan = await self.pipe.decision.decide("greet", T(""), self.sess)
        self.sess.context.record_turn(0, "", plan.intent, False)
        text = plan.fallback.get(self.lang) or plan.fallback["en"]
        self.lines.append(("agent", plan.intent, text))
        self.opening = plan
        return plan

    async def say(self, text):
        r = await self.pipe.run_turn(T(text))
        self.last = r
        self.lines += [("customer", "", text), ("agent", r.plan.intent, r.text)]
        return r

    @property
    def agent(self):
        return [t for role, _, t in self.lines if role == "agent"]

    @property
    def intents(self):
        return [i for role, i, _ in self.lines if role == "agent"]

    @property
    def result(self):
        raw = self.sess.dialog.get("_result")
        return json.loads(raw) if raw else None


async def run(call_type, lines, **kw):
    c = Call(call_type, **kw)
    await c.open()
    for line in lines:
        await c.say(line)
    return c


def step_text(ct_id, step_id, lang):
    from acie.calltypes import CallTypeRegistry
    return CallTypeRegistry().get(ct_id).step(step_id).text[lang]


def asked(c, ct_id, step_id):
    """How many agent lines contain this step's question (in any language: the policy picks the reply language per turn)."""
    from acie.calltypes import CallTypeRegistry
    texts = list(CallTypeRegistry().get(ct_id).step(step_id).text.values())
    return sum(any(t in line for t in texts) for line in c.agent)
