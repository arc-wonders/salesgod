"""Step-driven call domain: runs any CallType's ordered steps as a deterministic machine (custom calls, payment reminder,
feedback survey, appointment booking; lead_qualification and follow_up extend it).

It sits on the same layers as the sales domain and REUSES the sales pack by composition (`self.sales`): the compliance
floor (opt-out, number source, AI disclosure, never take OTP/card digits), the product FAQ / knowledge-base answers, the
locked price lines, the garbled-line reply. The crisis row is handled before any domain by the pipeline.

Per turn, in this order (nothing here is sampled, same input -> same output):
  1. safety / compliance (word for word)            4. human / callback requests
  2. firm "no" / "stop" / goodbye -> a polite close  5. permission -> the steps in order -> closing
  3. a sad caller -> no pushing                      6. questions are answered first (FAQ/KB/price, else the LLM from the facts)

Answers come from the caller's words by acie/calltypes/extract.py (yes/no, numbers, money, dates, choices, text). A step is
asked at most twice and never again once answered; one sentence can answer later steps too (their `keywords`).
The CallResult is rebuilt after every turn into session.dialog["_result"] and session.context.result.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from ...calltypes.extract import DEFAULT_TZ, UNKNOWN, extract, is_dont_know, step_keyword_hit
from ...calltypes.extract import date_time as parse_date_time
from ...calltypes.schema import LANGS, STAGE_FOR_OUTCOME, CallResult, CallType, NextAction, Step
from ...context import is_question
from ...schemas import Fact, SessionState, TurnInput, TurnPlan
from .. import honest
from .product_sales import (_CB_WORD, _CUT_CALL, _FIRM_NO, _GOODBYE_NOW, _PRICE_CORE, _STRONG_NO, _TIME_WORDS, Product, ProductSalesDomain, _tpl)

log = logging.getLogger("acie.calltypes")

_DISCLOSURE = _tpl("I'm {agent}, an AI assistant calling for {seller}.", "मैं {agent} हूँ, {seller} की तरफ़ से एक AI असिस्टेंट।",
                   "Main {agent} hoon, {seller} ki taraf se ek AI assistant.")
_RECORDED = _tpl("This call is recorded.", "ये कॉल रिकॉर्ड हो रही है।", "Ye call record ho rahi hai.")
_HAS_AI = re.compile(r"\bai\b|artificial|एआई|ए आई|असिस्टेंट|assistant", re.I)
_HAS_REC = re.compile(r"record|रिकॉर्ड", re.I)
_UNSAFE_DIGITS = re.compile(r"\d{9,}")
_SENSITIVE_WORDS = re.compile(r"\botp\b|ओटीपी|\bcvv\b|card|कार्ड|\bpin\b|पिन|password|पासवर्ड|aadhaar|aadhar|आधार|account number|अकाउंट नंबर", re.I)
_BUSY = re.compile(r"\bbusy\b|बिजी|abhi (time|free) nahi|अभी (टाइम|समय|फ्री) नहीं|\bdriving\b|\bmeeting\b|मीटिंग|not (a )?good time|bad time|wrong time|later|baad mein|baad me|बाद में", re.I)
_NO_SELL_PERSONA = "Do not sell, push, pressure or create urgency. Be brief and warm. "
_OPEN_Q_END = {"en": "Does that answer it?", "hi": "क्या इससे बात साफ़ हुई?", "hinglish": "Kya isse baat saaf hui?"}

_ACTION_NEXT = {"book_callback": ("follow_up_call", "follow_up"), "book_demo": ("book_demo", ""), "book_appointment": ("book_appointment", ""),
                "take_payment_promise": ("payment_followup", "payment_reminder"), "send_link": ("follow_up_call", "follow_up"),
                "transfer_human": ("human_call", ""), "send_sms": ("", "")}
_MONTHS = {"en": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
           "hi": ["जनवरी", "फ़रवरी", "मार्च", "अप्रैल", "मई", "जून", "जुलाई", "अगस्त", "सितंबर", "अक्टूबर", "नवंबर", "दिसंबर"],
           "hinglish": ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]}
_WEEKDAYS = {"en": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
             "hi": ["सोमवार", "मंगलवार", "बुधवार", "गुरुवार", "शुक्रवार", "शनिवार", "रविवार"],
             "hinglish": ["Somvar", "Mangalvar", "Budhvar", "Guruvar", "Shukravar", "Shanivar", "Ravivar"]}


def humanize(iso: str, lang: str) -> str:
    """'2026-11-16T18:00:00+05:30' -> 'Monday 16 Nov at 6 pm' / 'सोमवार 16 नवंबर, शाम 6 बजे' / 'Somvar 16 Nov, shaam 6 baje'."""
    try:
        dt = datetime.fromisoformat(iso)
    except (ValueError, TypeError):
        return str(iso)
    h, m = dt.hour, dt.minute
    h12 = h % 12 or 12
    mm = f":{m:02d}" if m else ""
    day = f"{_WEEKDAYS[lang][dt.weekday()]} {dt.day} {_MONTHS[lang][dt.month - 1]}"
    if lang == "en":
        return f"{day} at {h12}{mm} {'am' if h < 12 else 'pm'}"
    part = ("सुबह" if lang == "hi" else "subah") if h < 12 else ("दोपहर" if lang == "hi" else "dopahar") if h < 16 else \
           ("शाम" if lang == "hi" else "shaam") if h < 20 else ("रात" if lang == "hi" else "raat")
    return f"{day}, {part} {h12}{mm} {'बजे' if lang == 'hi' else 'baje'}"


def _clean(s: str) -> str:
    s = re.sub(r"\{[^}]*\}", "", s)
    s = re.sub(r"\s+([,.!?।])", r"\1", s)
    return re.sub(r"\s{2,}", " ", s).strip()


class StepCallDomain:
    """A DomainPack for step-driven call types. `ct` is the CallType, `product` the product being discussed."""

    name = "call_steps"  # (the sales-only pipeline guards, keyed on name == "product_sales", do not apply here)
    score_qualified, score_interested = 60, 40

    def __init__(self, call_type: CallType, product: Product | None = None, customer_name: str = "", world=None, customer_age: int | None = None,
                 playbook=None, business=None, customer=None, turn_model=None, now_fn=None, dnc_lookup=None, tz: str = DEFAULT_TZ,
                 sales: ProductSalesDomain | None = None):
        self.ct = call_type
        self.sales = sales or ProductSalesDomain(product or Product(), customer_name=customer_name, world=world, customer_age=customer_age,
                                                 playbook=playbook, business=business, customer=customer, turn_model=turn_model,
                                                 now_fn=now_fn, dnc_lookup=dnc_lookup)
        self.classifier = self.sales.classifier
        self.customer_name = customer_name.strip()
        self.customer_age = customer_age
        self.interim = None
        self._now_fn = now_fn
        self.tz = tz

    @property
    def p(self) -> Product:
        return self.sales.p

    # ------------------------------------------------------------------ small helpers

    def _now(self) -> datetime:
        z = ZoneInfo(self.tz)
        return (self._now_fn() if self._now_fn else datetime.now(z)).astimezone(z)

    @staticmethod
    def _j(d: dict, key: str, default):
        try:
            return json.loads(d.get(key) or "") if d.get(key) else default
        except ValueError:
            return default

    def _ans(self, d: dict) -> dict:
        return self._j(d, "ct_ans", {})

    def _put_ans(self, d: dict, a: dict) -> None:
        d["ct_ans"] = json.dumps(a, ensure_ascii=False, sort_keys=True)

    def _fmt(self, text: str, lang: str = "en") -> str:
        p = self.p
        out = (text.replace("{agent}", p.agent_name).replace("{seller}", p.seller_name).replace("{product}", p.product_name)
                   .replace("{name}", self.customer_name))
        return _clean(out)

    def _lines(self, field: str) -> dict[str, str]:
        return {l: self._fmt(self.ct.line(field, l), l) for l in LANGS}

    def _step_texts(self, step: Step, d: dict) -> dict[str, str]:
        return {l: self._fmt(step.text.get(l) or step.text.get("hinglish") or step.text.get("en", ""), l) for l in LANGS}

    def _say(self, intent: str, texts: dict[str, str], *, end: bool = False, decision: str = "", objective: str = "") -> TurnPlan:
        return TurnPlan(intent=intent, llm_allowed=False, end_call=end, decision=decision or f"{self.ct.id}: {intent}",
                        objective=objective or "Say exactly the reviewed line", fallback={l: _clean(texts.get(l) or texts.get("en", "")) for l in LANGS})

    @staticmethod
    def _join(*parts: dict[str, str] | None) -> dict[str, str]:
        out = {}
        for l in LANGS:
            bits = []
            for p in parts:
                t = ((p or {}).get(l) or "").strip()
                if t:
                    bits.append(t if t[-1] in ".!?।…\"”" else t + ("।" if l == "hi" else "."))
            out[l] = " ".join(bits)
        return out

    # ------------------------------------------------------------------ state

    def _init(self, session: SessionState) -> None:
        d = session.dialog
        d["pending"] = "intro"  # (the sales identify / number-source lines read it: "intro" = ask for two minutes)
        d["ct_id"] = self.ct.id
        d["ct_phase"] = "perm"
        d.setdefault("ct_t0", f"{self._now().timestamp():.0f}")
        d.setdefault("ct_score", "0")
        session.context.call_type = self.ct.id

    def _skipped(self, step: Step, d: dict) -> bool:
        if step.id in self._j(d, "ct_skip", []):
            return True
        a = self._ans(d)
        if step.action and self.ct.allowed_actions and step.action not in self.ct.allowed_actions:
            return True
        if step.action == "send_link" and not (self.p.payment_link or "").strip():
            return True
        return bool(step.skip_if) and self._eval_skip(step.skip_if.strip(), a, int(d.get("ct_score", "0") or 0))

    @staticmethod
    def _eval_skip(expr: str, a: dict, score: int) -> bool:
        m = re.match(r"^(answered|unanswered):([a-z][a-z0-9_]*)$", expr)
        if m:
            real = a.get(m.group(2)) not in (None, UNKNOWN, "declined")
            return real if m.group(1) == "answered" else not real
        m = re.match(r"^([a-z][a-z0-9_]*)\s*(==|!=|>=|<=|>|<)\s*(\S.*)$", expr)
        if not m:
            return False
        var, op, right = m.group(1), m.group(2), m.group(3).strip().lower()
        left = score if var == "score" else a.get(var)
        if isinstance(left, bool):
            left = "yes" if left else "no"
        try:
            rn = float(right)
            ln = float(left)
            return {"==": ln == rn, "!=": ln != rn, ">=": ln >= rn, "<=": ln <= rn, ">": ln > rn, "<": ln < rn}[op]
        except (TypeError, ValueError):
            pass
        ls = "" if left is None else str(left).lower()
        rs = {"true": "yes", "false": "no"}.get(right, right)
        return ls == rs if op == "==" else ls != rs if op == "!=" else False

    def _next_step(self, d: dict) -> Step | None:
        a = self._ans(d)
        if d.get("ct_end") == "1":  # a choice that ends the call was picked: no more questions
            return None
        for s in self.ct.steps:
            if s.id in a or self._skipped(s, d):
                continue
            return s
        return None

    def _cur(self, d: dict) -> Step | None:
        sid = d.get("ct_cur")
        return self.ct.step(sid) if sid else None

    # ------------------------------------------------------------------ result

    def _points(self, step: Step, value) -> int:
        if value is None or value == UNKNOWN:
            return 0
        keys = ["yes" if value else "no", str(value).lower()] if isinstance(value, bool) else [str(value).lower()]
        if step.answer_type == "date_time" and value != "declined":
            keys.append("booked")
        pts = next((step.score[k] for k in keys if k in step.score), 0)
        return pts + (step.score_answered if value not in ("declined",) else 0)

    def _build_result(self, d: dict, complete: bool) -> CallResult:
        a = self._ans(d)
        outcome = d.get("ct_out") or (self._fallback_outcome(d) if complete else "ended_early")
        act = self._j(d, "ct_act", {})
        typ, ctype = _ACTION_NEXT.get(act.get("action", ""), ("", ""))
        due = act.get("due_at")
        if act.get("action") == "send_link" and not due:
            due = (self._now() + timedelta(days=2)).replace(hour=11, minute=0, second=0, microsecond=0).isoformat()
        if not typ and outcome in ("callback_requested", "undecided"):
            typ, ctype = "follow_up_call", "follow_up"
        facts = {k: str(v) for k, v in self._j(d, "ct_facts", {}).items()}
        un = [s.id for s in self.ct.steps if s.required and s.id not in a and not self._skipped(s, d)]
        if complete and un:
            facts["unanswered"] = ",".join(un)
        return CallResult(outcome=outcome, stage_suggestion=STAGE_FOR_OUTCOME.get(outcome, "contacted"),
                          next_action=NextAction(type=typ, due_at=due, call_type=ctype), answers=a,
                          score_delta=max(0, min(100, int(d.get("ct_score", "0") or 0))), call_type=self.ct.id, facts=facts, complete=complete)

    def _fallback_outcome(self, d: dict) -> str:
        return "completed"

    def _publish(self, session: SessionState) -> None:
        d = session.dialog
        if "ct_phase" not in d:
            return
        res = self._build_result(d, complete=d["ct_phase"] == "done")
        d["_result"] = res.model_dump_json()
        session.context.result = res.model_dump()

    def _fact(self, d: dict, key: str, val: str) -> None:
        f = self._j(d, "ct_facts", {})
        f[key] = val
        d["ct_facts"] = json.dumps(f, ensure_ascii=False, sort_keys=True)

    def _set_action(self, d: dict, action: str, due: str | None) -> None:
        d["ct_act"] = json.dumps({"action": action, "due_at": due})

    # ------------------------------------------------------------------ entry

    async def plan(self, intent: str, turn: TurnInput, session: SessionState) -> TurnPlan:
        try:
            plan = await self._plan(intent, turn, session)
        except Exception:  # noqa: BLE001 - a silent line on a live call is worse than a generic one
            log.exception("call-type plan failed (%s, intent=%s)", self.ct.id, intent)
            cur = self._cur(session.dialog)
            ask = self._step_texts(cur, session.dialog) if cur else _tpl("Sorry, could you say that again?", "माफ़ कीजिए, क्या आप दोबारा बोलेंगे?", "Sorry, kya aap dobara bolenge?")
            plan = self._say("clarify", self._join(_tpl("Sorry, could you say that again?", "माफ़ कीजिए, क्या आप दोबारा बोलेंगे?", "Sorry, kya aap dobara bolenge?"), ask if cur else None))
        self._publish(session)
        d = session.dialog
        d["last_intent"] = plan.intent
        d["last_customer"] = turn.text
        return plan

    # ------------------------------------------------------------------ opening

    def _opening_texts(self, session: SessionState) -> dict[str, str]:
        base = self._lines("opening")
        out = {}
        for l in LANGS:
            t = base[l]
            if not _HAS_AI.search(t):  # the AI disclosure and the recording notice are never optional
                t = self._fmt(_DISCLOSURE[l], l) + " " + t
            if not _HAS_REC.search(t):
                t = t + " " + _RECORDED[l]
            extra = " ".join(self._fmt(line.get(l) or line.get("en", ""), l) for line in self.ct.compliance_lines)
            out[l] = _clean(t + (" " + extra if extra else ""))
        return out

    def _opening(self, session: SessionState) -> TurnPlan:
        self._init(session)
        return self._say("opening", self._opening_texts(session), decision=f"{self.ct.id}: opening, who is calling and why, ask permission",
                         objective="Say who you are (an AI assistant), that the call is recorded, why you are calling, and ask permission to continue")

    # ------------------------------------------------------------------ the turn

    async def _plan(self, intent: str, turn: TurnInput, session: SessionState) -> TurnPlan:
        d = session.dialog
        text = (turn.text or "").strip()
        lang = d.get("_lang", "hi")
        if intent == "greet":
            return self._opening(session)
        if "ct_phase" not in d:
            self._init(session)
        phase = d["ct_phase"]
        if phase == "done":
            return self._say("closing", self._lines("closing"), end=True, decision="the call is already over")
        if self._over_time(d):
            return self._close(session, "ok", why="max_call_s reached")
        cur = self._cur(d)
        sad = None
        # -------- 1. safety and compliance (the sales floor, word for word)
        side = self.sales._compliance(self._mask_numbers(text, cur), intent, d, session.context)
        if side is not None:
            if side.intent == "opt_out":
                return self._finish_with(session, side, "opt_out", stage_text=None)
            if side.intent in ("identify", "are_you_ai"):
                return self._resume(side, session)
            return side
        u = self.sales._nlu(d)
        # -------- 2. a firm no, a cut-off or goodbye: honour it
        short_yn = cur is not None and cur.answer_type == "yes_no" and len(text.split()) <= 3
        if _STRONG_NO.search(text) or _CUT_CALL.search(text) or (_FIRM_NO.search(text) and not short_yn):
            return self._close(session, "declined", why="the caller said no")
        if _GOODBYE_NOW.search(text):
            return self._close(session, "ok" if self._ans(d) else "declined", why="the caller is ending the call")
        # -------- 3. a sad caller: warm and short, no pushing
        if d.get("_row") == "sad" or honest.SAD.search(text):
            sad = self.sales._sad_step(text, intent, d, lang)
            if sad is not None:
                return sad
        # -------- 4. callback in progress / requested, a person requested
        if d.get("ct_cb") == "asking":
            return self._callback_answer(session, text)
        if intent == "human_agent":
            return self._human(session)
        wants_cb = intent == "callback" or _CB_WORD.search(text) or (phase == "perm" and _BUSY.search(text))
        if wants_cb and not (phase == "steps" and cur is not None and cur.answer_type == "date_time"):
            if not (phase == "steps" and cur is not None and cur.answer_type in ("choice", "yes_no", "money", "number") and self._try(cur, text)[0]):
                return self._callback(session, text)
        # -------- 5. phases
        if phase == "perm":
            return await self._permission(session, text, intent, u)
        if phase == "open_q":
            return await self._open_questions(session, text, intent, u)
        return await self._steps(session, text, intent, u)

    # ---- hooks for the specialised packs
    async def _after_permission(self, session: SessionState, text: str) -> TurnPlan | None:
        return None

    def _price_answer(self, text: str, lang: str, d: dict) -> str | None:
        if self.p.offer_price and _PRICE_CORE.search(text):
            return self.sales._price_lead(text, lang)
        return None

    def _memory_context(self) -> str:
        return ""

    def _persona_extra(self) -> str:
        return ""

    def _step_text_for(self, step: Step, d: dict) -> dict[str, str]:
        return self._step_texts(step, d)

    # ------------------------------------------------------------------ pieces

    def _over_time(self, d: dict) -> bool:
        try:
            t0 = float(d.get("ct_t0", "0"))
        except ValueError:
            return False
        return t0 > 0 and self.ct.max_call_s > 0 and self._now().timestamp() - t0 > self.ct.max_call_s

    def _mask_numbers(self, text: str, cur: Step | None) -> str:
        """A budget / count said in digits ("500000") is not a card or OTP: hide 6-8 digit runs from the sensitive-digit rule
        when a numeric question is open and no card/OTP word is present. 9+ digit runs (phones, accounts) always stay."""
        if cur is not None and cur.answer_type in ("money", "number") and not _SENSITIVE_WORDS.search(text):
            return re.sub(r"(?<!\d)\d[\d,]{5,8}(?!\d)", " number ", text)
        return text

    def _try(self, step: Step, text: str) -> tuple[bool, object]:
        return extract(step, text, self._now(), self.tz)

    def _resume(self, plan: TurnPlan, session: SessionState) -> TurnPlan:
        """After an identity / AI-disclosure answer, come back to the open question."""
        d = session.dialog
        cur = self._cur(d)
        ask = self._step_text_for(cur, d) if (cur is not None and d.get("ct_phase") == "steps") else None
        if ask:
            plan = plan.model_copy(update={"fallback": {l: (plan.fallback.get(l, "") + " " + ask[l]).strip() for l in LANGS}})
        return plan

    def _facts_ident(self) -> dict[str, Fact]:
        return {"product": Fact(value=self.p.product_name, description="product name", must_mention=False)}

    def _knowledge(self, text: str, lang: str) -> str:
        p = self.p
        kn = [p.knowledge] if p.knowledge else []
        approved = [f"- {f.answer}" for f in p.faqs if f.answer]
        if approved:
            kn.append("Approved answers you can use:\n" + "\n".join(approved))
        kn.append(f"The product: {self.sales.about}")
        kn.append("If you do not have a detail, say so honestly and where to look: " + honest.source(p, "en") + "; never guess or invent a value, a yes or a no.")
        try:
            from ...tree import kb_knowledge
            if (extra := kb_knowledge(p, text, lang)):
                kn.append(extra)
        except Exception:  # noqa: BLE001
            log.exception("kb_knowledge failed")
        if (m := self._memory_context()):
            kn.append(m)
        return "\n\n".join(kn)

    def _det_answers(self, text: str, d: dict) -> dict[str, str] | None:
        out = {}
        kb = os.environ.get("ACIE_KB") == "1"
        for l in LANGS:
            a = self._price_answer(text, l, d) or self.sales._answer_text(text, l) or (self.sales._kb_line(text, l, d) if kb else "")
            if not a:
                return None
            out[l] = a.strip()
        return out

    def _answer_plan(self, session: SessionState, text: str, ask: dict[str, str] | None, intent: str = "answer") -> TurnPlan:
        """A question: answer it first (the approved FAQ / KB / locked price line, word for word; else the LLM from the facts), then
        come back to the step. Never a pitch."""
        d = session.dialog
        det = self._det_answers(text, d)
        ask = ask or {l: "" for l in LANGS}
        if det is not None:
            return self._say(intent, self._join(det, ask), decision="answered from the approved facts, then back to the questions",
                             objective="Answer their question from the approved answer, then ask the next question")
        lang = d.get("_lang", "hi")
        n = int(d.get("ct_unk", "0"))
        d["ct_unk"] = str(n + 1)
        unk = {l: self.sales._unknown_answer(text, d, l, n) for l in LANGS}
        male = self.p.agent_gender != "female"
        persona = ((self.ct.tone + ". ") if self.ct.tone else "") + _NO_SELL_PERSONA * (not self.ct.selling_allowed) + self._persona_extra() + (
            f"You ({self.p.agent_name}) are {'a man' if male else 'a woman'}: in Hindi use the {'masculine' if male else 'feminine'} form for yourself. ")
        return TurnPlan(intent=intent, facts=self._facts_ident(), context=self._knowledge(text, lang), persona=persona,
                        rule_packs=["sales_compliance", "claims", "respectful_register", "agent_male" if male else "agent_female"],
                        objective=("Answer exactly what they asked in one or two short sentences, from the background only; never invent. "
                                   "Do not sell, push or add urgency. " + ("Then ask: " + ask["en"] if ask.get("en") else "")),
                        ask=ask.get("en") or None, decision="a question: answer from the facts, then back to the questions",
                        must_not_say=["made-up features, prices or policies", "'limited time' or false urgency", "claims not in the background"],
                        fallback=self._join(unk, ask))

    # ------------------------------------------------------------------ permission -> first step

    async def _permission(self, session: SessionState, text: str, intent: str, u) -> TurnPlan:
        d = session.dialog
        from ...calltypes.extract import yes_no
        ok, yn = yes_no(text)
        if ok and yn is False and len(text.split()) <= 4:
            return self._callback(session, text)  # "no" to "do you have two minutes?" = not now
        asks = (is_question(text) or (u is not None and u.question_topic != "none")) and not (ok and yn)
        if asks:
            resume = _tpl("Is this a good time for two minutes?", "क्या अभी दो मिनट बात हो सकती है?", "Kya abhi do minute baat ho sakti hai?")
            return self._answer_plan(session, text, resume)
        d["ct_phase"] = "steps"
        if (hook := await self._after_permission(session, text)) is not None:
            return hook
        pure_yes = ok and yn and len(text.split()) <= 4
        if pure_yes or not text:
            return self._ask_next(session)
        return await self._steps(session, text, intent, u)  # they started answering already

    def _prep_ask(self, d: dict) -> tuple[Step | None, dict[str, str]]:
        """Move to the next unanswered step: (step, its texts), or (None, {}) when there is none."""
        step = self._next_step(d)
        if step is None:
            return None, {}
        d["ct_cur"] = step.id
        tried = self._j(d, "ct_tried", {})
        tried.setdefault(step.id, 0)
        d["ct_tried"] = json.dumps(tried)
        d["pending"] = "ct_" + step.id
        return step, self._step_text_for(step, d)

    def _ask_next(self, session: SessionState, prefix: dict[str, str] | None = None, again: bool = False) -> TurnPlan:
        d = session.dialog
        step, ask = self._prep_ask(d)
        if step is None:
            return self._finish_steps(session, prefix)
        if again:
            pre = self._reask_prefix(step)
            ask = {l: pre[l] + " " + ask[l] for l in LANGS}
        return TurnPlan(intent="step_ask", llm_allowed=False, ask=ask["en"], decision=f"{self.ct.id}: ask step {step.id}",
                        objective=f"Ask the question: {ask['en']}", fallback={l: _clean((prefix or {}).get(l, "") + " " + ask[l]) for l in LANGS})

    def _reask_prefix(self, step: Step) -> dict[str, str]:
        if step.answer_type == "yes_no":
            return _tpl("Sorry, I didn't catch that, a simple yes or no is fine.", "माफ़ कीजिए, समझ नहीं आया, बस हाँ या ना बता दीजिए।", "Sorry, samajh nahi aaya, bas haan ya na bata dijiye.")
        if step.answer_type == "choice":
            return {l: {"en": "Sorry, I didn't catch that. You can say: ", "hi": "माफ़ कीजिए, समझ नहीं आया। आप बता सकते हैं: ", "hinglish": "Sorry, samajh nahi aaya. Aap bata sakte hain: "}[l]
                    + ", ".join((c.label.get(l) or c.label.get("en") or c.id.replace("_", " ")) for c in step.choices) + "." for l in LANGS}
        return _tpl("Sorry, I didn't catch that.", "माफ़ कीजिए, समझ नहीं आया।", "Sorry, samajh nahi aaya.")

    # ------------------------------------------------------------------ the steps

    async def _steps(self, session: SessionState, text: str, intent: str, u) -> TurnPlan:
        d = session.dialog
        cur = self._cur(d) or self._next_step(d)
        if cur is None:
            return self._finish_steps(session, None)
        d["ct_cur"] = cur.id
        a = self._ans(d)
        asked = is_question(text) or (u is not None and u.question_topic not in ("none",))
        garbled = bool(u is not None and u.garbled)
        ok, val = self._try(cur, text)
        if ok and cur.answer_type == "text" and asked and not self._substantive_text(text):
            ok = False  # a question is not the answer to an open question
        if ok and cur.answer_type == "yes_no" and asked and len(text.split()) > 5:
            ok = False
        # one sentence can answer later steps too ("mera budget 10 hazaar hai")
        for s in self.ct.steps:
            if s.id != cur.id and s.id not in a and s.keywords and step_keyword_hit(s, text):
                ok2, v2 = self._try(s, text)
                if ok2 and v2 != UNKNOWN:
                    self._record(session, s, v2)
        if ok:
            self._record(session, cur, val)
            return self._ask_next(session, prefix=self._side_answer(session, text, asked))
        if asked and not garbled:
            return self._answer_plan(session, text, self._step_text_for(cur, d))
        if garbled:
            g = int(d.get("garbled_n", "0")) + 1
            d["garbled_n"] = str(g)
            return self.sales._garbled_plan(g, int(d.get("garbled_total", "0")))
        tried = self._j(d, "ct_tried", {})
        n = int(tried.get(cur.id, 0)) + 1
        tried[cur.id] = n
        d["ct_tried"] = json.dumps(tried)
        if n >= (2 if cur.required else 1):  # never ask a third time (and an optional one only once)
            self._record(session, cur, None)
            return self._ask_next(session)
        return self._ask_next(session, again=True)

    @staticmethod
    def _substantive_text(text: str) -> bool:
        return not text.strip().endswith("?") and len(text.split()) >= 4

    def _side_answer(self, session: SessionState, text: str, asked: bool) -> dict[str, str] | None:
        """A step answer that also carried a question ("10 hazaar. aur price kya hai?"): answer it before the next question."""
        d = session.dialog
        if not asked:
            return None
        det = self._det_answers(text, d)
        if det is None:
            return None
        return det

    def _record(self, session: SessionState, step: Step, value) -> None:
        """Store an answer (None = asked, never answered), score it, apply what the step earns."""
        d = session.dialog
        a = self._ans(d)
        if step.id in a:
            return
        a[step.id] = value
        self._put_ans(d, a)
        d["ct_score"] = str(int(d.get("ct_score", "0") or 0) + self._points(step, value))
        real = value not in (None, UNKNOWN, "declined", False)
        if step.answer_type == "choice" and real:
            ch = next((c for c in step.choices if c.id == value), None)
            if ch is not None:
                if ch.outcome:
                    d["ct_out"] = ch.outcome
                if ch.ends_call:
                    d["ct_end"] = "1"
        elif real and step.outcome:
            d["ct_out"] = step.outcome
        if real and step.action:
            if step.answer_type == "date_time":
                self._set_action(d, step.action, str(value))
                self._fact(d, step.action + "_at", str(value))
            elif step.answer_type == "yes_no" and value is True:
                self._set_action(d, step.action, None)
        if value is True and step.answer_type == "yes_no" and step.action == "send_link":
            d["ct_send_link"] = "1"
        if step.answer_type == "date_time" and value == "declined" and step.action:
            d["ct_decl_" + step.id] = "1"
        self._after_record(session, step, value)

    def _after_record(self, session: SessionState, step: Step, value) -> None:
        pass

    # ------------------------------------------------------------------ finishing

    def _finish_steps(self, session: SessionState, prefix: dict[str, str] | None) -> TurnPlan:
        d = session.dialog
        if d.get("ct_send_link") == "1":
            return self._link(session, prefix)
        return self._close(session, "ok", prefix=prefix, why="all steps done")

    def _link(self, session: SessionState, prefix: dict[str, str] | None) -> TurnPlan:
        d = session.dialog
        d["ct_phase"] = "done"
        d["ct_out"] = "link_sent"
        d["payment_link_url"] = (self.p.payment_link or "").strip()
        self._fact(d, "link_sent", "1")
        site = self.p.official_site
        txt = _tpl("Great, I'm sending the payment link by SMS to this number now. Thank you" + (f"; the offer is also on {site}" if site else "") + ".",
                   "बढ़िया, मैं अभी इसी नंबर पर SMS से पेमेंट लिंक भेज रहा हूँ। धन्यवाद।",
                   "Badhiya, main abhi isi number pe SMS se payment link bhej raha hoon. Thank you.")
        plan = self._say("link_sent", self._join(prefix, txt), end=True, decision="payment link sent after their yes")
        if (self.p.payment_link or "").strip():
            plan.facts["payment_link"] = Fact(value=self.p.payment_link, description="the payment link (sent by SMS, never spoken)", must_mention=False, hidden=True)
        return plan

    def _confirm_line(self, d: dict) -> dict[str, str] | None:
        act = self._j(d, "ct_act", {})
        due = act.get("due_at")
        if not due or act.get("action") not in ("book_demo", "book_callback", "book_appointment", "take_payment_promise"):
            return None
        w = {l: humanize(due, l) for l in LANGS}
        kind = act["action"]
        if kind == "take_payment_promise":
            return {"en": f"Thank you, I've noted that you'll pay by {w['en']}.", "hi": f"धन्यवाद, मैंने नोट कर लिया कि आप {w['hi']} तक पेमेंट करेंगे।",
                    "hinglish": f"Thank you, maine note kar liya ki aap {w['hinglish']} tak payment karenge."}
        word = {"book_demo": ("a demo", "डेमो", "demo"), "book_callback": ("a call", "कॉल", "call"), "book_appointment": ("your appointment", "आपका अपॉइंटमेंट", "aapka appointment")}[kind]
        return {"en": f"Done, I've noted {word[0]} for {w['en']}.", "hi": f"ठीक है, मैंने {w['hi']} के लिए {word[1]} नोट कर लिया है।",
                "hinglish": f"Theek hai, maine {w['hinglish']} ke liye {word[2]} note kar liya hai."}

    def _close(self, session: SessionState, kind: str, prefix: dict[str, str] | None = None, why: str = "") -> TurnPlan:
        """kind: 'ok' (normal end) or 'declined' (they said no / stop: a polite goodbye, never a pitch)."""
        d = session.dialog
        d["ct_phase"] = "done"
        d.pop("pending", None)
        if kind == "declined":
            if d.get("ct_out") not in ("opt_out", "human_requested", "callback_requested"):
                d["ct_out"] = "declined"
            d["ct_act"] = ""
            texts = self._lines("closing_declined") if self.ct.closing_declined else self._lines("closing")
            intent = "declined"
        else:
            if d.get("ct_end") == "1" and d.get("ct_out") in ("declined", "not_interested"):
                texts = self._lines("closing_declined") if self.ct.closing_declined else self._lines("closing")
                intent = "declined"
            else:
                conf = self._confirm_line(d)
                texts = self._join(conf, self._lines("closing")) if conf else self._lines("closing")
                intent = "callback" if d.get("ct_out") in ("callback_booked", "callback_requested") else "closing"
            if not d.get("ct_out"):
                d["ct_out"] = self._fallback_outcome(d)
        return self._say(intent, self._join(prefix, texts), end=True, decision=f"{self.ct.id}: close ({why})",
                         objective="Thank them and say goodbye; no pitch")

    def _finish_with(self, session: SessionState, plan: TurnPlan, outcome: str, stage_text=None) -> TurnPlan:
        d = session.dialog
        d["ct_phase"] = "done"
        d["ct_out"] = outcome
        d["ct_act"] = ""
        return plan

    # ------------------------------------------------------------------ steps answered by choice that ends the call
    # (handled in _ask_next via _next_step: after an ending choice every later step is skipped)

    # ------------------------------------------------------------------ callback / human

    def _callback(self, session: SessionState, text: str) -> TurnPlan:
        d = session.dialog
        ok, iso = parse_date_time(text, self._now(), self.tz)
        if "book_callback" not in (self.ct.allowed_actions or ["book_callback"]):
            return self._close(session, "declined", why="a callback is not allowed for this call type")
        if ok and iso:
            return self._book_callback(session, iso)
        d["ct_cb"] = "asking"
        return TurnPlan(intent="callback_time", llm_allowed=False, decision="busy: ask when to call back",
                        objective="Kindly accept; ask what time suits for a call back; no pitch",
                        fallback=_tpl("Of course, no problem. When should I call you back, say tomorrow evening?", "बिल्कुल, कोई बात नहीं। मैं आपको कब वापस कॉल करूँ, जैसे कल शाम?",
                                      "Bilkul, koi baat nahi. Main aapko kab wapas call karoon, jaise kal shaam?"))

    def _callback_answer(self, session: SessionState, text: str) -> TurnPlan:
        d = session.dialog
        ok, iso = parse_date_time(text, self._now(), self.tz)
        if ok and iso:
            d.pop("ct_cb", None)
            return self._book_callback(session, iso)
        n = int(d.get("ct_cb_n", "0")) + 1
        d["ct_cb_n"] = str(n)
        if n >= 2 or _FIRM_NO.search(text) or _GOODBYE_NOW.search(text):
            d.pop("ct_cb", None)
            d["ct_out"] = "callback_requested"
            d["ct_phase"] = "done"
            self._set_action(d, "book_callback", None)
            return self._say("callback", _tpl("No problem, I'll try you again another time. Thank you, have a good day!", "कोई बात नहीं, मैं फिर किसी और समय कोशिश करूँगा। धन्यवाद, आपका दिन शुभ हो!",
                                              "Koi baat nahi, main phir kisi aur time try karunga. Thank you, aapka din accha ho!"), end=True, decision="no time given: close politely")
        return self._say("callback_time", _tpl("Sorry, which day and time would suit you?", "माफ़ कीजिए, कौन सा दिन और समय ठीक रहेगा?", "Sorry, kaun sa din aur time theek rahega?"))

    def _book_callback(self, session: SessionState, iso: str) -> TurnPlan:
        d = session.dialog
        d["ct_phase"] = "done"
        d["ct_out"] = "callback_booked"
        self._set_action(d, "book_callback", iso)
        self._fact(d, "callback_at", iso)
        w = {l: humanize(iso, l) for l in LANGS}
        return self._say("callback", {"en": f"Sure, I'll call you back on {w['en']}. Thank you, have a good day!", "hi": f"ज़रूर, मैं आपको {w['hi']} को वापस कॉल करूँगा। धन्यवाद, आपका दिन शुभ हो!",
                                      "hinglish": f"Zaroor, main aapko {w['hinglish']} ko wapas call karunga. Thank you, aapka din accha ho!"}, end=True, decision="callback booked")

    def _human(self, session: SessionState) -> TurnPlan:
        d = session.dialog
        d["ct_phase"] = "done"
        d["ct_out"] = "human_requested"
        self._set_action(d, "transfer_human", None)
        return self._say("human_agent", _tpl("Of course. I'll note that you'd like to speak to a person, and someone from the team can call you. Thank you!",
                                             "ज़रूर। मैं नोट कर लेता हूँ कि आप किसी इंसान से बात करना चाहते हैं, टीम का कोई व्यक्ति आपको कॉल कर सकता है। धन्यवाद!",
                                             "Zaroor. Main note kar leta hoon ki aap kisi insaan se baat karna chahte hain, team ka koi vyakti aapko call kar sakta hai. Thank you!"),
                         end=True, decision="human requested")

    # ------------------------------------------------------------------ open questions (follow_up)

    async def _open_questions(self, session: SessionState, text: str, intent: str, u) -> TurnPlan:
        session.dialog["ct_phase"] = "steps"
        return await self._steps(session, text, intent, u)
