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
import time
from collections.abc import Awaitable, Callable

from .compiler import PromptCompiler
from .decision.engine import DecisionEngine
from .expression import ExpressionBackend, TemplateBackend, render_fallback
from .feedback import FeedbackEngine
from .personalization.policy import AdaptationPolicy, detect_explicit_requests
from .personalization.profile_store import MemoryProfileStore, ProfileStore
from .context import is_question
from .safety import distress_followup_plan, distress_plan, is_distress
from .schemas import HumanState, SessionState, TurnInput, TurnResult, Utterance
from .state.estimator import HumanStateEstimator
from .validator import StreamingValidator

_FILLER_OPEN = re.compile(r"^(aapke sawal ka jawab (yeh |ye )?(hai )?(ki |hai ki )?|आपके सवाल का जवाब (यह |ये )?(है )?(कि )?|"
                          r"आपके सवाल का जवाब देते हुए,? |to answer your question,? |the answer to your question is,? )", re.I)
_ABOUT_IT = re.compile(r"(karta|karti|kar sakta|kar sakti|sakta|sakti|samajhta|samjhta) (hai|h)\b|करता है|करती है|कर सकता|"
                       r"सकता है|समझता है|\b(does|can|will) (it|claude)\b|support|\b(usse|isse|usme|isme|use|ise)\b|उससे|इससे|इसमें|उसमें", re.IGNORECASE)

SentenceSink = Callable[[str], Awaitable[None] | None]


class ACIEPipeline:
    def __init__(self, decision: DecisionEngine, backend: ExpressionBackend | None = None,
                 store: ProfileStore | None = None, estimator: HumanStateEstimator | None = None,
                 policy: AdaptationPolicy | None = None, compiler: PromptCompiler | None = None,
                 feedback: FeedbackEngine | None = None, buffer_reply: bool = False, nlu=None,
                 session_store=None):
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
        self.policy = policy or AdaptationPolicy()
        self.compiler = compiler or PromptCompiler()
        self.feedback = feedback or FeedbackEngine()
        self.sessions: dict[str, SessionState] = {}  # live-call cache
        # optional durable copy (acie.store.sessions): loaded at turn start if this process has not seen the
        # session yet (restart mid-call), saved at turn end. None = today's in-process behaviour.
        self.session_store = session_store

    async def forget_session(self, session_id: str) -> SessionState | None:
        sess = self.sessions.pop(session_id, None)
        if self.session_store is not None:
            await self.session_store.delete(session_id)
        return sess

    def session(self, turn: TurnInput) -> SessionState:
        if turn.session_id not in self.sessions:
            self.sessions[turn.session_id] = SessionState(session_id=turn.session_id, customer_id=turn.customer_id)
        return self.sessions[turn.session_id]

    async def _stream_a(self, turn: TurnInput, session: SessionState) -> tuple[str, HumanState]:
        rule_intent = self.decision.classify(turn, session)
        nlu = None
        dom = self.decision.domain
        guide = getattr(dom, "intent_guide", None)
        session.dialog.pop("_nlu", None)
        sure = rule_intent in getattr(dom, "confident_intents", ()) and len(turn.text.split()) <= 6
        if self.nlu is not None and guide and turn.text.strip() and not sure:  # skip (~0.6 s) when the rules are sure
            history = [(u.role, u.text) for u in session.history]
            nlu = await self.nlu.understand(turn.text, history, guide, getattr(dom, "about", ""))
        intent = rule_intent
        if nlu is not None:
            session.dialog["_nlu"] = nlu.model_dump_json()
            session.dialog["_rule_intent"] = rule_intent
            # the rules keep the last word on what they are sure about; the model fills the gaps
            if nlu.intent != "unknown" and rule_intent in ("unknown", "tell_more", "benefits", "goodbye", "affirm", "deny"):
                intent = nlu.intent
        return intent, self.estimator.estimate(turn, session, nlu)

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
        (intent, state), profile = await asyncio.gather(self._stream_a(turn, session), self.store.load(turn.customer_id))
        mark("understand")

        explicit = detect_explicit_requests(turn.text)
        if "language" in explicit and _ABOUT_IT.search(turn.text):
            # "Hindi mein bhi baat karta hai kya?" asks what the PRODUCT can do, not to switch language
            explicit.pop("language")
        prev_expr = session.expression if session.turn_index > 0 else None
        expr = self.policy.decide(turn.text, state, profile, session, explicit, intent)
        mark("adapt")

        if session.dialog.get("safety_flag"):  # sticky: once in crisis, never back to business
            plan = distress_followup_plan(turn.text)
        elif is_distress(turn.text):  # safety override: never sell to someone in crisis
            plan = distress_plan()
            session.dialog["safety_flag"] = "1"
            session.dialog.pop("pending", None)
            profile.outcomes.append("safety_distress")
        else:
            # the domain may react to how the caller feels right now (e.g. recover when annoyed)
            session.dialog["_frustration"] = f"{state.frustration:.2f}"
            session.dialog["_language_request"] = explicit.get("language", "")
            session.dialog["_lang"] = expr.language  # the reply language, for language-specific facts
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
        mark("decide")
        # whole-call memory: what was promised / quoted / asked survives beyond the prompt window
        session.context.record_turn(session.turn_index, turn.text, plan.intent, is_question(turn.text),
                                    objection=session.dialog.pop("_objection", None), sinks=self.context_sinks)

        prompt = self.compiler.compile(plan, expr, session, turn.text)
        validator = StreamingValidator(plan, expr, trusted=not plan.llm_allowed, customer_text=turn.text)
        sentences: list[str] = []

        async def emit(batch: list[str]) -> None:
            for s in batch:
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
            validator.issues.append(f"llm error: {type(e).__name__}: {e}")
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
                    validator.issues.append(f"regenerated after: {why}")
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
                validator.issues.append(f"trimmed reply from {len(reply)} to {len(kept)} sentences")
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
                    validator.issues.append(f"dropped echo of the customer: {first!r}")
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
                validator.issues.append("llm repeated its previous reply; regenerated")
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
            # any problem anywhere in the reply -> speak only the safe template
            await emit([validator.fill(render_fallback(plan, expr))] if validator.used_fallback else reply)
        else:
            await emit(tail)
        mark("total")

        text = " ".join(sentences)
        if plan.llm_allowed and (plan.intent == "answer" and plan.lead or "offer" in plan.facts and plan.facts["offer"].value in text) \
                and "offer_price" not in session.context.quoted:
            session.context.quoted.append("offer_price")  # the price was told inside an answer: don't offer it again
        session.history += [Utterance(role="customer", text=turn.text), Utterance(role="agent", text=text)]
        session.state, session.expression = state, expr
        if turn.prosody is not None:  # this turn becomes part of the caller's voice baseline
            session.prosody_history = (session.prosody_history + [turn.prosody])[-20:]
        session.turn_index += 1

        self.feedback.after_turn(profile, turn.text, explicit, prev_expr, state, plan)
        if plan.end_call:
            self.feedback.end_session(profile)
        await self.store.save(profile)
        if self.session_store is not None:
            try:
                await self.session_store.put(session)
            except Exception:  # the durable copy must never break a live call
                pass

        return TurnResult(text=text, plan=plan, state=state, expression=expr,
                          validation=validator.report(), timings_ms=t)
