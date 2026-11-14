"""Follow-up call: continues from the customer's previous call(s) (data/voice/calls/*.json, matched by customer id / phone).

It opens with continuity ("last time you asked me to call you around ...", "I had sent you the payment link", "we had talked
about Rs 2,999"), answers the questions that were left open FIRST, honours the agreed price, then moves to the decision step
(go ahead / more time / not interested), confirms the link or reschedules. It never repeats the pitch or anything the previous
call already said, and never negotiates. With no previous call it falls back to the call type's own opening."""

from __future__ import annotations

from pathlib import Path

from ...calltypes.memory import PreviousCall, load_previous_calls
from ...calltypes.schema import LANGS, CallType, Step
from ...schemas import SessionState, TurnPlan
from .custom_steps import StepCallDomain, _clean, humanize
from .product_sales import _tpl

_OBJ = {"price": ("the price", "कीमत", "price"), "trust": ("trust", "भरोसे", "bharose"), "need": ("whether you need it", "ज़रूरत", "zaroorat"),
        "timing": ("the timing", "समय", "timing"), "language": ("language", "भाषा", "bhasha"), "other": ("a concern", "एक चिंता", "ek chinta")}


class FollowUpDomain(StepCallDomain):
    name = "follow_up"

    def __init__(self, call_type: CallType, *args, memory: list[PreviousCall] | None = None, data_dir: Path | str | None = None,
                 customer_id: str = "", phone: str = "", call_id: str = "", **kw):
        super().__init__(call_type, *args, **kw)
        if memory is None and data_dir is not None:
            try:
                memory = load_previous_calls(data_dir, customer_id=customer_id, phone=phone, exclude_call_id=call_id, tz=self.tz)
            except Exception:  # noqa: BLE001 - no memory is never a failed call
                memory = []
        self.mem = memory or []
        self.prev: PreviousCall | None = self.mem[0] if self.mem else None

    # ------------------------------------------------------------------ continuity

    def _bits(self) -> list[dict[str, str]]:
        p = self.prev
        if p is None:
            return []
        out: list[dict[str, str]] = []
        if p.callback_at or p.callback_text:
            w = {l: (humanize(p.callback_at, l) if p.callback_at else f"“{p.callback_text}”") for l in LANGS}
            out.append({"en": f"Last time you asked me to call you around {w['en']}.", "hi": f"पिछली बार आपने कहा था कि मैं {w['hi']} कॉल करूँ।",
                        "hinglish": f"Pichhli baar aapne kaha tha ki main {w['hinglish']} call karoon."})
        if p.link_sent:
            out.append(_tpl("I had sent you the payment link on SMS.", "मैंने आपको SMS पर पेमेंट लिंक भेजा था।", "Maine aapko SMS par payment link bheja tha."))
        if p.agreed_price:
            out.append(_tpl(f"We had talked about {p.agreed_price}.", f"हमने {p.agreed_price} पर बात की थी।", f"Humne {p.agreed_price} par baat ki thi."))
        for ob in p.objections[:1]:
            lab = _OBJ.get(ob, _OBJ["other"])
            out.append(_tpl(f"You had a question about {lab[0]}.", f"आपके मन में {lab[1]} को लेकर सवाल था।", f"Aapke mann mein {lab[2]} ko lekar sawal tha."))
        if p.open_questions:
            out.append(_tpl("You also had a question I'd like to answer.", "आपका एक सवाल भी था जिसका जवाब देना है।", "Aapka ek sawal bhi tha jiska jawab dena hai."))
        return out[:2 if not p.open_questions else 3]

    def _opening_texts(self, session: SessionState) -> dict[str, str]:
        bits = self._bits()
        if not bits:
            return super()._opening_texts(session)
        base = super()._opening_texts(session)  # the operator's opening (AI disclosure + recording + compliance lines ensured)
        head = {"en": "I'm following up on our last call about {product}.", "hi": "{product} के बारे में पिछली कॉल का फ़ॉलो-अप है।",
                "hinglish": "{product} ke baare mein pichhli call ka follow-up hai."}
        ask = _tpl("Is this a good time for two minutes?", "क्या अभी दो मिनट बात हो सकती है?", "Kya abhi do minute baat ho sakti hai?")
        out = {}
        for l in LANGS:
            intro = {"en": "Hi {name}, I'm {agent}, an AI assistant calling for {seller}. This call is recorded.",
                     "hi": "नमस्ते {name}, मैं {agent} हूँ, {seller} की तरफ़ से एक AI असिस्टेंट। ये कॉल रिकॉर्ड हो रही है।",
                     "hinglish": "Namaste {name}, main {agent} hoon, {seller} ki taraf se ek AI assistant. Ye call record ho rahi hai."}[l]
            extra = " ".join(self._fmt(x.get(l) or x.get("en", ""), l) for x in self.ct.compliance_lines)
            out[l] = _clean(self._fmt(intro + " " + head[l], l) + " " + " ".join(b[l] for b in bits) + " " + ask[l] + (" " + extra if extra else ""))
        return out

    def _memory_context(self) -> str:
        p = self.prev
        if p is None:
            return ""
        lines = ["LAST CALL with this customer (do NOT repeat what was already said; no pitch):"]
        if p.told:
            lines.append("already told: " + ", ".join(p.told))
        if p.agreed_price:
            lines.append(f"agreed price: {p.agreed_price}; it stays")
        if p.objections:
            lines.append("their concerns: " + ", ".join(p.objections))
        if p.link_sent:
            lines.append("a payment link was sent")
        return "\n".join(lines)

    def _persona_extra(self) -> str:
        return "Never repeat what the last call already covered, never pitch again, never offer another price. " if self.prev else ""

    def _price_answer(self, text: str, lang: str, d: dict) -> str | None:
        base = super()._price_answer(text, lang, d)
        if base is None:
            return None
        if self.prev is not None and self.prev.agreed_price:
            pr = self.prev.agreed_price
            return {"en": f"The price we talked about, {pr}, stays as it is.", "hi": f"जो कीमत हमने तय की थी, {pr}, वही रहेगी।",
                    "hinglish": f"Jo price humne tay kiya tha, {pr}, wahi rahega."}[lang]
        return base

    def _step_text_for(self, step: Step, d: dict) -> dict[str, str]:
        t = super()._step_text_for(step, d)
        if step.action == "send_link" and self.prev is not None and self.prev.agreed_price:
            pr = self.prev.agreed_price
            pre = {"en": f"At the price we talked about, {pr}. ", "hi": f"{pr} की उसी कीमत पर। ", "hinglish": f"{pr} ke usi price par. "}
            return {l: pre[l] + t[l] for l in LANGS}
        return t

    # ------------------------------------------------------------------ open questions first

    async def _after_permission(self, session: SessionState, text: str) -> TurnPlan | None:
        d = session.dialog
        if "ct_oq" not in d:
            import json
            d["ct_oq"] = json.dumps(list(self.prev.open_questions) if self.prev else [], ensure_ascii=False)
        if not self._j(d, "ct_oq", []):
            return None
        d["ct_phase"] = "open_q"
        return await self._open_questions(session, "", "", None)

    async def _open_questions(self, session: SessionState, text: str, intent: str, u) -> TurnPlan:
        import json

        from ...context import is_question
        d = session.dialog
        if text and (is_question(text) or (u is not None and u.question_topic != "none")):
            return self._answer_plan(session, text, None)  # something new: answer it first; the old ones wait
        oq = self._j(d, "ct_oq", [])
        said: list[dict[str, str]] = []
        while oq and len(said) < 2:
            q = oq[0]
            det = self._det_answers(q, d)
            if det is None:
                break
            oq.pop(0)
            quote = q if len(q) <= 50 else q[:47].rstrip() + "..."
            said.append({"en": f"About your earlier question, “{quote}”: {det['en']}", "hi": f"आपके पिछले सवाल, “{quote}” के बारे में: {det['hi']}",
                         "hinglish": f"Aapke pichhle sawal, “{quote}” ke baare mein: {det['hinglish']}"})
        d["ct_oq"] = json.dumps(oq, ensure_ascii=False)
        if said:
            d["ct_phase"] = "steps" if not oq else "open_q"
            prefix = self._join(*said)
            if oq:  # one needs the model: next turn
                return self._say("answer", prefix)
            return self._ask_next(session, prefix=prefix)
        if oq:  # the model answers it from the facts (honest "I don't have that" when it is not there)
            q = oq.pop(0)
            d["ct_oq"] = json.dumps(oq, ensure_ascii=False)
            if not oq:
                d["ct_phase"] = "steps"
                step, ask = self._prep_ask(d)
                return self._answer_plan(session, q, ask if step is not None else None)
            return self._answer_plan(session, q, None)
        d["ct_phase"] = "steps"
        return self._ask_next(session)

    def _fallback_outcome(self, d: dict) -> str:
        return "undecided"
