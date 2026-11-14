"""Orchestrator: one customer turn through all layers.

  Stream A (intent + human state)  ┐
                                   ├─> policy -> decision -> compiler -> LLM -> validator -> TTS
  Stream B (profile retrieval)     ┘                                                 └─> feedback

Validated sentences are pushed to `on_sentence` as soon as they pass, so TTS
can start speaking before the LLM finishes.
"""

from __future__ import annotations

import asyncio
import difflib
import re
import inspect
import os
import json
import os
import time
from collections.abc import Awaitable, Callable

from .compiler import PromptCompiler
from .decision.engine import DecisionEngine
from .expression import SESSION_KEY, ExpressionBackend, TemplateBackend, render_fallback
from .feedback import FeedbackEngine
from .personalization.policy import AdaptationPolicy, detect_explicit_requests, make_default_policy
from .personalization.profile_store import MemoryProfileStore, ProfileStore
from .context import is_question
from .safety import CONCERN_T, DISTRESS_T, concern_plan, distress_followup_plan, distress_plan, distress_score
from . import psyche
from .psyche import emotion_policy
from .psyche.ledger import SaidLedger
from .psyche.moves import Move, MoveContext, choose_move, repeat_move
from .psyche.needs import NeedInput, estimate_need
from .schemas import HumanState, SessionState, TurnInput, TurnResult, Utterance
from .state.estimator import HumanStateEstimator
from .validator import StreamingValidator

_FILLER_OPEN = re.compile(r"^(aapke sawal ka jawab (yeh |ye )?(hai )?(ki |hai ki )?|आपके सवाल का जवाब (यह |ये )?(है )?(कि )?|"
                          r"आपके सवाल का जवाब देते हुए,? |to answer your question,? |the answer to your question is,? )", re.I)
_ABOUT_IT = re.compile(r"(karta|karti|kar sakta|kar sakti|sakta|sakti|samajhta|samjhta) (hai|h)\b|करता है|करती है|कर सकता|"
                       r"सकता है|समझता है|\b(does|can|will) (it|claude)\b|support|\b(usse|isse|usme|isme|use|ise)\b|उससे|इससे|इसमें|उसमें", re.IGNORECASE)

# plan intents whose wording must never be a near copy of an earlier reply (compliance lines such as the
# confirmation with full terms and the opt-out confirmation are required wording and stay as they are)
_REPEATABLE = frozenset({"pitch", "price", "answer", "faq", "not_understood", "off_topic", "clarify", "hello", "reoffer", "recover",
                         "small_talk", "personal", "goodbye", "link_sent", "open_question", "price_floor", "acknowledge"})

# the discovery question ("what do you use it for / what is your goal?") and the goal words that answer it
_DISC_Q = re.compile(r"\b(what|which)\b.{0,40}\b(use|using|goal|work|study|need|looking for|spend)\b|\b(goal|use it for|mainly use)\b|"
                     r"किस (काम|चीज़|चीज)|आपका (main )?(goal|लक्ष्य)|इस्तेमाल करेंगे|\bkis (kaam|cheez)\b|aapka (main )?goal|use karenge|काम या पढ़ाई|kaam ya padhai|"
                     r"लक्ष्य", re.I)
_GOAL_WORDS = re.compile(r"muscle|bulk|मसल|बॉडी|body ?build|weight ?(loss|lose)|lose weight|वज़न|वजन|fat loss|stay fit|fitness|fit rehna|फिट रहना|"
                         r"coding|code|कोडिंग|writing|study|padhai|पढ़ाई|resume|assignment|report", re.I)

_FILLER = {"haan", "han", "ha", "ji", "ok", "okay", "yes", "yeah", "hmm", "hm", "acha", "achha", "accha", "theek", "thik", "hai", "hello", "हाँ", "हां",
           "जी", "ओके", "ठीक", "है", "हम्म", "अच्छा", "हेलो", "bolo", "बोलो", "bataiye", "बताइए", "please", "sure", "nahi", "नहीं", "no"}

SentenceSink = Callable[[str], Awaitable[None] | None]


class ACIEPipeline:
    def __init__(self, decision: DecisionEngine, backend: ExpressionBackend | None = None,
                 store: ProfileStore | None = None, estimator: HumanStateEstimator | None = None,
                 policy: AdaptationPolicy | None = None, compiler: PromptCompiler | None = None,
                 feedback: FeedbackEngine | None = None, buffer_reply: bool = False, nlu=None,
                 session_store=None, signal_providers: list | None = None):
        # buffer_reply: validate the whole reply before releasing any of it. On a phone call this
        # avoids speaking half an LLM reply and then the safe template (the customer would hear
        # the same facts twice). Replies are 1-2 sentences, so the extra wait is small.
        self.buffer_reply = buffer_reply
        self.max_reply_words = 40
        self.nlu = nlu  # acie.understanding.LLMUnderstanding (optional neural NLU layer)
        self.context_sinks: list = []  # called with every new CallContext event (console, NATS, ...)
        self.decision = decision
        self.backend = backend or TemplateBackend()
        self.store = store or MemoryProfileStore()
        self.estimator = estimator or HumanStateEstimator()
        self.policy = policy or make_default_policy()
        self.compiler = compiler or PromptCompiler()
        self.feedback = feedback or FeedbackEngine()
        if getattr(self.feedback, "policy", 1) is None:
            self.feedback.policy = self.policy  # learned policies get observe(...) rewards
        self.sessions: dict[str, SessionState] = {}  # live-call cache
        # optional durable copy (acie.store.sessions): loaded at turn start if this process has not seen the
        # session yet (restart mid-call), saved at turn end. None = today's in-process behaviour.
        self.session_store = session_store
        # acie.providers: None = built from ACIE_SIGNAL_PROVIDERS (default: rules, plus the LLM NLU when `nlu` is given)
        self._signal_hub = None
        self._hub_cache = None
        if signal_providers is not None:
            from .providers.hub import SignalHub
            self._signal_hub = SignalHub(signal_providers)

    async def forget_session(self, session_id: str) -> SessionState | None:
        sess = self.sessions.pop(session_id, None)
        if self.session_store is not None:
            await self.session_store.delete(session_id)
        return sess

    def session(self, turn: TurnInput) -> SessionState:
        if turn.session_id not in self.sessions:
            self.sessions[turn.session_id] = SessionState(session_id=turn.session_id, customer_id=turn.customer_id)
        return self.sessions[turn.session_id]

    def _hub(self):
        """The signal providers for this pipeline (acie.providers), rebuilt when their configuration changes."""
        if self._signal_hub is not None:
            return self._signal_hub
        key = (id(self.nlu), os.getenv("ACIE_SIGNAL_PROVIDERS", ""), os.getenv("ACIE_SIGNAL_HTTP_URL", ""),
               os.getenv("ACIE_TURN_MODEL", ""))
        if self._hub_cache is None or self._hub_cache[0] != key:
            from .providers import build_hub
            self._hub_cache = (key, build_hub(self))
        return self._hub_cache[1]

    async def _stream_a(self, turn: TurnInput, session: SessionState) -> tuple[str, HumanState]:
        """Layers 1-2: every provider reads the turn, the fusion (acie/signals_fusion.py) makes one TurnSignals, and
        the decision layer sees only that. Rules keep the last word on priority intents; the other providers fill gaps."""
        dom = self.decision.domain
        session.dialog.pop("_nlu", None)
        session.dialog.pop("_signals", None)
        sig = await self._hub().run(turn, session, dom, getattr(dom, "classifier", None))
        if sig.learned:
            session.dialog["_nlu"] = sig.to_understanding().model_dump_json()  # console / logs compat
            session.dialog["_rule_intent"] = sig.rule_intent
        session.dialog["_signals"] = sig.model_dump_json()
        if hasattr(self.estimator, "scorer") and sig.state:
            return sig.intent, self.estimator.estimate(turn, session, signals=sig)
        return sig.intent, self.estimator.estimate(turn, session, sig.to_understanding() if sig.learned else None)

    # ------------------------------------------------------------------ psyche (need -> move) and the repeat guard

    def _move_context(self, session: SessionState, need, ledger: SaidLedger, lang: str, state, text: str):
        dom = self.decision.domain
        return MoveContext(need=need, lang=lang, product=getattr(dom, "p", None), ledger=ledger, text=text,
                           turn_index=session.turn_index, frustration=state.frustration, engagement=state.engagement,
                           decided=session.context.decision, has_link=session.context.has("payment_link"),
                           refused=session.context.decision in ("declined", "later"),
                           hold=session.dialog.get("sad_hold") == "1")

    def _psyche_move(self, turn: TurnInput, session: SessionState, state: HumanState, intent: str, strat,
                     ledger: SaidLedger, lang: str):
        from .state.dims import state_dims
        ctx = session.context
        need = estimate_need(NeedInput(text=turn.text, state=state_dims(state), intent=intent,
                                       has_link=ctx.has("payment_link"), decided=ctx.decision,
                                       turn_index=session.turn_index,
                                       prev_agent=next((u.text for u in reversed(session.history) if u.role == "agent"), "")))
        if need.need == "information" and not strat.continue_business:  # the emotion row outranks plain curiosity
            need.need = {"suspicious": "reassurance", "anxious": "reassurance", "sad": "connection", "busy": "space"}.get(strat.row, "be_heard")
            need.rule += "+" + strat.row
        ledger.add_move("need:" + need.need)
        mv = choose_move(self._move_context(session, need, ledger, lang, state, turn.text))
        session.dialog["_need"] = need.need
        return mv

    def _no_repeat(self, plan, expr, session: SessionState, ledger: SaidLedger, state: HumanState, intent: str):
        """Plain bug fix, always on: a deterministic reply that is a near copy (ratio > 0.6) of an earlier whole
        reply is replaced by something that helps (live calls: the same price line four times, "link sent" twice)."""
        if os.getenv("ACIE_REPEAT_GUARD", "1") == "0" or getattr(self.decision.domain, "name", "") != "product_sales" or plan.intent not in _REPEATABLE or plan.intent in ("opening", "language_switch") or plan.reference or not ledger.replies:
            return plan
        template_only = not plan.llm_allowed or isinstance(self.backend, TemplateBackend)
        if not template_only:
            return plan
        lang = expr.language
        parts = []
        if plan.lead:
            parts.append(plan.lead.get(lang) or plan.lead.get("en") or "")
        parts.append(plan.fallback.get(lang) or plan.fallback.get("en") or "")
        probe = StreamingValidator(plan, expr, trusted=True, customer_text="")
        cand = " ".join(probe.fill(x) for x in parts if x)
        if not ledger.reply_repeats(cand):
            return plan
        from .state.dims import state_dims
        need = estimate_need(NeedInput(text=session.dialog.get("last_customer", ""), state=state_dims(state)), use_model=False)
        mv = repeat_move(self._move_context(session, need, ledger, lang, state, session.dialog.get("last_customer", "")), plan.intent)
        session.dialog["_repeat_guard"] = str(int(session.dialog.get("_repeat_guard", "0")) + 1)
        if mv.end_call:
            intent_out = "goodbye"
        elif mv.name.split(":")[0] in ("answer_fact", "answer_unknown", "answer_health", "no_detail"):
            intent_out = "answer"  # the caller asked again: it is answered, not "that's all I have"
        elif mv.name in ("ask_repeat", "offer_stop", "simplify", "ask_unclear"):
            intent_out = "clarify"
        else:
            intent_out = "nothing_new"
        session.dialog["_pm"] = mv.to_json()
        return plan.model_copy(update={"intent": intent_out, "llm_allowed": False, "end_call": mv.end_call or plan.end_call and (plan.intent in ("goodbye", "declined") or str(plan.decision).startswith("honest answer")),
                                       "decision": "psyche_move", "fallback": dict(mv.text), "lead": None, "facts": {},
                                       "objective": mv.note or "Say something new"})

    # ------------------------------------------------------------------ what the caller already told us (never re-ask it)

    @staticmethod
    def _track_need(session: SessionState, text: str) -> None:
        """The discovery question was asked and they answered (or said their goal on their own): remember it, so it is never
        asked again ("muscle" was asked for three more times in the persona test)."""
        d = session.dialog
        if d.get("need_known"):
            return
        said_goal = bool(_GOAL_WORDS.search(text))
        toks = re.findall(r"[\w\u0900-\u097f]+", text.lower())
        filler = not toks or all(t in _FILLER for t in toks)
        if said_goal or (d.get("need_asked") == "1" and not filler and not is_question(text) and len(toks) <= 12):
            d["need_known"] = text.strip()[:80]

    @staticmethod
    def _note_asked(session: SessionState, agent_text: str) -> None:
        if any(_DISC_Q.search(x) for x in re.split(r"(?<=[.!?।])\s+", agent_text) if x.rstrip().endswith("?")):
            session.dialog["need_asked"] = "1"

    @staticmethod
    def _drop_repeated_questions(reply: list[str], ledger: SaidLedger, session: SessionState) -> list[str]:
        """Questions the reply asks that the call already asked, or that the caller already answered (the discovery
        question once their need is known), are dropped. Only ever the question: the answer stays."""
        known = bool(session.dialog.get("need_known"))
        keep: list[str] = []
        for sent in reply:
            q = sent.rstrip().endswith("?")
            n = re.sub(r"[\s.,!?;:\"'()\-।]+", " ", sent.lower()).strip()
            again = q and any(difflib.SequenceMatcher(None, n, a).ratio() > 0.6 for a in ledger.asks)
            if q and (again or (known and _DISC_Q.search(sent))) and (keep or len(reply) > 1):
                continue
            keep.append(sent)
        return keep or reply

    def _learn_said(self, ledger: SaidLedger, session: SessionState, plan, move, text: str) -> None:
        self._note_asked(session, text)
        ledger.add_text(text)
        ledger.add_intent(plan.intent)
        raw = session.dialog.pop("_pm", None)
        mv = Move.from_json(raw) if raw else None
        if plan.decision == "psyche_move":
            m = mv or move
            if m is not None:
                ledger.add_slot(m.slot)
                ledger.add_move(m.name)
                if m.sells:
                    ledger.offers += 1
        if plan.intent in ("declined", "callback"):
            ledger.refused = True
        ledger.save(session.dialog)

    async def _regenerate(self, plan, expr, session, text: str, note: str) -> list[str] | None:
        """One more generation with a correction note; None if it also fails validation."""
        retry = self.compiler.compile(plan, expr, session, text)
        retry.messages[-1]["content"] += "\n\n" + note
        v2 = StreamingValidator(plan, expr, trusted=False, customer_text=text)
        out: list[str] = []
        try:
            async for delta in self.backend.stream(retry, plan, expr):
                out.extend(v2.feed(delta))
                if v2.failed:
                    break
        except Exception:
            return None
        out += v2.finish()
        return None if (v2.failed or v2.used_fallback or not out) else out

    async def run_turn(self, turn: TurnInput, on_sentence: SentenceSink | None = None) -> TurnResult:
        t: dict[str, float] = {}
        t0 = time.perf_counter()

        def mark(name: str) -> None:
            t[name] = round((time.perf_counter() - t0) * 1000, 2)

        if self.session_store is not None and turn.session_id not in self.sessions:
            saved = await self.session_store.get(turn.session_id)
            if saved is not None:
                self.sessions[turn.session_id] = saved
        session = self.session(turn)
        SESSION_KEY.set(session.session_id)  # sticky A/B variant per session (acie/expression.py)
        if session.turn_index == 0 and hasattr(self.feedback, "recall"):
            await self.feedback.recall(session, turn.text)  # semantic memory (ACIE_MEMORY=1), else a no-op
        (intent, state), profile = await asyncio.gather(self._stream_a(turn, session), self.store.load(turn.customer_id))
        mark("understand")

        explicit = detect_explicit_requests(turn.text)
        if "language" in explicit and _ABOUT_IT.search(turn.text):
            # "Hindi mein bhi baat karta hai kya?" asks what the PRODUCT can do, not to switch language
            explicit.pop("language")
        prev_expr = session.expression if session.turn_index > 0 else None
        expr = self.policy.decide(turn.text, state, profile, session, explicit, intent)
        # the emotion table (acie/psyche/emotion_policy.py): one deterministic row per turn from the 14 state dims.
        # The crisis row is always active (it is the old distress override, now a row); the others need ACIE_PSYCHE=1
        # crisis evidence: phonetic + 2-turn window + voice (acie/safety.distress_score), OR the NLU's crisis flag
        hist = [u.text for u in session.history if u.role == "customer"][-2:]
        d_score, d_ev = distress_score(turn.text, hist, getattr(turn, "voice", None))
        nlu_crisis = session.dialog.pop("_nlu_crisis", None)
        flagged = bool(session.dialog.get("safety_flag")) or bool(nlu_crisis) or d_score >= DISTRESS_T
        if d_ev or nlu_crisis:
            session.dialog["_distress"] = json.dumps({"score": round(d_score, 2), "evidence": d_ev[:4], "nlu": nlu_crisis})
        strat = emotion_policy.decide(state, turn.text, turn.voice, flagged=flagged)
        session.dialog["_row"] = strat.row  # the domain reads it (a sad caller pauses the selling in both configs)
        if psyche.enabled() and strat.row != "crisis":
            expr = emotion_policy.apply_to_expression(expr, strat)
        mark("adapt")
        ledger = SaidLedger.load(session.dialog)
        self._track_need(session, turn.text)
        move = None

        if strat.row == "crisis":  # sticky: once in crisis, never back to business
            if session.dialog.get("safety_flag"):
                plan = distress_followup_plan(turn.text)
            else:  # never sell to someone in crisis
                plan = distress_plan()
                session.dialog["safety_flag"] = "1"
                session.dialog.pop("pending", None)
                profile.outcomes.append("safety_distress")
        elif CONCERN_T <= d_score < DISTRESS_T and not session.dialog.get("safety_flag") \
                and not session.dialog.get("concern_asked"):
            plan = concern_plan()  # middle band: a gentle check-in instead of selling; a clearer answer reaches crisis
            session.dialog["concern_asked"] = "1"
        else:
            # the domain may react to how the caller feels right now (e.g. recover when annoyed)
            session.dialog["_frustration"] = f"{state.frustration:.2f}"
            session.dialog["_language_request"] = explicit.get("language", "")
            session.dialog["_lang"] = expr.language  # the reply language, for language-specific facts
            session.dialog.pop("_move", None)
            if psyche.enabled():
                move = self._psyche_move(turn, session, state, intent, strat, ledger, expr.language)
                session.dialog["_move"] = move.to_json()
            dom = self.decision.domain
            if hasattr(dom, "interim"):
                # a slow step (world lookup) can say "one second, let me check" right away; spoken at once,
                # outside the reply buffer
                async def interim(texts: dict[str, str]) -> None:
                    if on_sentence and (t := texts.get(expr.language) or texts.get("en")):
                        r = on_sentence(t)
                        if inspect.isawaitable(r):
                            await r
                dom.interim = interim
            plan = await self.decision.decide(intent, turn, session)
            plan = self._no_repeat(plan, expr, session, ledger, state, intent)
        mark("decide")
        # whole-call memory: what was promised / quoted / asked survives beyond the prompt window
        session.context.record_turn(session.turn_index, turn.text, plan.intent, is_question(turn.text),
                                    objection=session.dialog.pop("_objection", None), sinks=self.context_sinks)

        extra = {}
        if os.getenv("ACIE_MEMORY") == "1":  # semantic memory + profile reach the prompt only when memory is on
            extra = {"profile": profile, "memories": json.loads(session.dialog.get("_memories", "[]"))}
        prompt = self.compiler.compile(plan, expr, session, turn.text, **extra)
        validator = StreamingValidator(plan, expr, trusted=not plan.llm_allowed, customer_text=turn.text)
        sentences: list[str] = []

        spoken_parts: list[str] = []  # every sentence already spoken this turn (lead, interim, reply)

        async def emit(batch: list[str]) -> None:
            for s in batch:
                # live call 7abf60f9bd: the lead and the safe template carried the same answer, so one reply said
                # "अच्छा सवाल है। ये इस हफ्ते का लॉन्च ऑफर है…" twice. Never speak a sentence twice in one turn.
                parts = [x for x in re.split(r"(?<=[.!?।])\s+", s.strip()) if x]
                fresh = [x for x in parts if not any(
                    difflib.SequenceMatcher(None, x.strip(" ।.!?,").lower(), y.strip(" ।.!?,").lower()).ratio() > 0.85
                    for y in spoken_parts)]
                if parts and not fresh:
                    continue
                if len(fresh) < len(parts):
                    s = " ".join(fresh)
                spoken_parts.extend(fresh)
                if not sentences:
                    mark("first_sentence")
                sentences.append(s)
                if on_sentence:
                    r = on_sentence(s)
                    if inspect.isawaitable(r):
                        await r

        held: list[str] = []
        if plan.lead:
            lead = plan.lead.get(expr.language) or plan.lead.get("en")
            if lead:
                await emit([validator.fill(lead)])

        async def release(batch: list[str]) -> None:
            if self.buffer_reply:
                held.extend(batch)
            else:
                await emit(batch)

        llm_down = False
        try:
            backend = self.backend if plan.llm_allowed else TemplateBackend()
            async for delta in backend.stream(prompt, plan, expr):
                await release(validator.feed(delta))
                if validator.failed:
                    break
        except Exception as e:  # LLM down / timeout -> safe template, call continues
            validator.add_issue(f"llm error: {type(e).__name__}: {e}", "llm_error")
            validator.failed = True
            llm_down = True
        tail = validator.finish()
        if self.buffer_reply:
            # a generation that failed half-way is replaced by the whole safe template: speaking the
            # first sentence alone ("Fair question.") left the caller waiting for an answer
            reply = tail if validator.failed else held + tail
            if validator.used_fallback and plan.llm_allowed and not llm_down and self.buffer_reply:
                # persona reviews: "sorry, didn't catch that" for questions the model had understood
                # (a rupee amount, a callback time, Devanagari in a Hinglish reply). Tell the model what
                # was wrong and try once more before falling back to the template.
                why = "; ".join(i.split(":")[0] for i in validator.issues[-2:])
                again = await self._regenerate(plan, expr, session, turn.text,
                                               f"Your previous reply was rejected ({why}). Write it again: answer what the "
                                               "customer asked, in the reply language, and use the {slots} for the offer price.")
                if again:
                    validator.add_issue(f"regenerated after: {why}", "regenerated")
                    validator.used_fallback = False
                    reply = again
            if plan.llm_allowed and len(" ".join(reply).split()) > self.max_reply_words:
                # the LLM ignores "one short sentence" (41-word answers on live call 8): keep the
                # leading sentences that fit; the ask usually comes last and is re-asked next turn
                kept, n = [], 0
                ask = reply[-1] if reply[-1].rstrip().endswith("?") else None  # the closing question is kept
                budget = self.max_reply_words - (len(ask.split()) if ask else 0)
                for s in reply[:-1] if ask else reply:
                    if kept and n + len(s.split()) > budget:
                        break
                    kept.append(s)
                    n += len(s.split())
                if ask:
                    kept.append(ask)
                validator.add_issue(f"trimmed reply from {len(reply)} to {len(kept)} sentences", "trimmed")
                reply = kept
            if plan.llm_allowed and reply and not validator.used_fallback:
                # "Aapke sawal ka jawab hai ki ..." on every turn sounded robotic (student persona)
                reply[0] = _FILLER_OPEN.sub("", reply[0]).strip()
                reply = [re.sub(r"क्या आपको (कुछ )?और (कुछ )?पता है\?|क्या आपको कुछ और पता करना है\?", "और कुछ जानना चाहेंगे?", r)
                         for r in reply]
                reply = [r for r in reply if r]
            if plan.llm_allowed and len(reply) > 1 and not validator.used_fallback:
                # Qwen likes to open by echoing the customer ("हाँ बताइए।", "ठीक है, कोशिश करके देखते हैं।"). After a
                # question only a near-verbatim echo goes ("yes, it writes reports" is an answer, not an echo)
                first = reply[0].strip(" ।.!?,")
                bar = 0.88 if is_question(turn.text) else 0.75
                if first and difflib.SequenceMatcher(None, first.lower(), turn.text.strip(" ।.!?,").lower()).ratio() > bar:
                    validator.add_issue(f"dropped echo of the customer: {first!r}", "echo")
                    reply = reply[1:]
            if plan.lead and plan.llm_allowed and not validator.used_fallback:
                # the model often restates the operator line it was told was already said: drop those sentences
                said = validator.fill(plan.lead.get(expr.language) or plan.lead.get("en") or "")
                parts = [x for x in re.split(r"(?<=[.!?।])\s+", said) if x]
                reply = [r for r in reply if not any(
                    difflib.SequenceMatcher(None, r.lower(), x.lower()).ratio() > 0.55 for x in parts)]
            last_agent = next((u.text for u in reversed(session.history) if u.role == "agent"), "")
            if (not validator.used_fallback and plan.llm_allowed and last_agent
                    and difflib.SequenceMatcher(None, " ".join(reply), last_agent).ratio() > 0.7):
                # live call 4: the LLM gave the same answer three times in a row. Ask once more for a
                # different reply (persona reviews: the "say that again?" fallback felt like not listening)
                validator.add_issue("llm repeated its previous reply; regenerated", "repeat")
                again = await self._regenerate(plan, expr, session, turn.text,
                                               f"You already said this last time: \"{last_agent}\". Do NOT repeat it; "
                                               "respond to what the customer just said in different words.")
                if again and difflib.SequenceMatcher(None, " ".join(again), last_agent).ratio() <= 0.8:
                    reply = again
                else:
                    # keep only the sentences that are new; "sorry, didn't catch that" for a clear question
                    # sounded like not listening (live sim: "ChatGPT से अलग कैसे?")
                    prev = [x for x in re.split(r"(?<=[.!?।])\s+", last_agent) if x]
                    fresh = [r for r in (again or reply) if not any(
                        difflib.SequenceMatcher(None, r, x).ratio() > 0.75 for x in prev)]
                    if fresh:
                        reply = fresh
                    else:
                        validator.used_fallback = True
            if plan.llm_allowed and not validator.used_fallback and reply:
                kept = self._drop_repeated_questions(reply, ledger, session)
                if kept != reply:
                    validator.add_issue("dropped a question the call already asked or answered", "question_repeated")
                    reply = kept
            # any problem anywhere in the reply -> speak only the safe template
            await emit([validator.fill(render_fallback(plan, expr))] if validator.used_fallback else reply)
        else:
            await emit(tail)
        mark("total")

        text = " ".join(sentences)
        if plan.llm_allowed and (plan.intent == "answer" and plan.lead or "offer" in plan.facts and plan.facts["offer"].value in text) \
                and "offer_price" not in session.context.quoted:
            session.context.quoted.append("offer_price")  # the price was told inside an answer: don't offer it again
        self._learn_said(ledger, session, plan, move, text)
        session.history += [Utterance(role="customer", text=turn.text), Utterance(role="agent", text=text)]
        session.state, session.expression = state, expr
        if turn.prosody is not None:  # this turn becomes part of the caller's voice baseline
            session.prosody_history = (session.prosody_history + [turn.prosody])[-20:]
        session.turn_index += 1

        self.feedback.after_turn(profile, turn.text, explicit, prev_expr, state, plan, session=session, expr=expr,
                                 validation=validator.report(), barge_in=turn.timing.barge_in)
        if plan.end_call:
            self.feedback.end_session(profile, session)
            await self.feedback.remember(session, profile)
        await self.store.save(profile)
        if self.session_store is not None:
            try:
                await self.session_store.put(session)
            except Exception:  # the durable copy must never break a live call
                pass

        return TurnResult(text=text, plan=plan, state=state, expression=expr,
                          validation=validator.report(), timings_ms=t)
