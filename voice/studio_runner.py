"""In-process runner for studio tests: the same pipeline as /api/sim (NLU, gates, decision tree, reply model, checker), but built
from a DRAFT product + DRAFT tree instead of the live files. No audio, no call record, nothing persisted except the usual sim profile."""

from __future__ import annotations

import uuid

from acie.trace import trace


class StudioRunner:
    def __init__(self, state):
        self.state = state

    def _pipe(self, product: dict, nodes: list[dict]):
        from acie import playbook as pbm
        from acie.decision.domains.product_sales import Product, ProductSalesDomain
        from acie.decision.engine import DecisionEngine
        from acie.pipeline import ACIEPipeline
        st = self.state
        prod = Product.model_validate(product)
        if hasattr(st, "with_agent"):
            prod = st.with_agent(prod)  # the persona's name and gender, as on a real call
        clean = [{k: v for k, v in n.items() if k not in ("cites", "origin")} for n in nodes]
        pb = pbm.Playbook(tenant=prod.tenant, product_id=prod.id, generated_by="studio draft", nodes=clean)
        domain = ProductSalesDomain(prod, customer_name="", world=st.world, customer_age=None, playbook=pb)
        backend = st.backends.get("gemini") or st.backends["llm"]
        return ACIEPipeline(DecisionEngine(domain), backend=backend, store=st.store, buffer_reply=True, nlu=st.nlu,
                            session_store=getattr(st, "session_store", None))

    async def start(self, product: dict, nodes: list[dict], language: str):
        from acie.schemas import ExpressionParams, TurnInput, Utterance
        from acie.validator import StreamingValidator
        pipe = self._pipe(product, nodes)
        sid = "studio-" + uuid.uuid4().hex[:10]
        ti = TurnInput(session_id=sid, customer_id=f"sim-{sid}", text="")
        sess = pipe.session(ti)
        sess.expression = ExpressionParams(language=language if language in ("hi", "en", "hinglish") else "hinglish")  # type: ignore[arg-type]
        plan = await pipe.decision.decide("greet", ti, sess)
        text = " ".join(StreamingValidator(plan, sess.expression).finish())
        sess.history.append(Utterance(role="agent", text=text))
        sess.context.record_turn(0, "", plan.intent, False)
        return (pipe, sid), text

    async def turn(self, handle, text: str) -> dict:
        from acie.schemas import TurnInput
        pipe, sid = handle
        r = await pipe.run_turn(TurnInput(session_id=sid, customer_id=f"sim-{sid}", text=text))
        sess = pipe.sessions.get(sid)
        tr = trace(r, sess.dialog if sess else None)
        deal_nums: set[str] = set()
        deal = getattr(getattr(sess, "context", None), "deal", None) if sess else None
        if deal is not None:
            for o in getattr(deal, "offers", []) or []:
                a = getattr(o, "amount", 0) or 0
                deal_nums.add(str(int(a)) if a == int(a) else str(a))
            for k in ("offer_price", "list_price", "agreed_price", "last_agent_offer"):
                a = getattr(deal, k, 0) or 0
                if a:
                    deal_nums.add(str(int(a)) if a == int(a) else str(a))
        return {"agent": r.text, "intent": r.plan.intent, "pb_node": tr.get("pb_node"), "end_call": bool(r.plan.end_call),
                "deal_numbers": sorted(deal_nums)}
