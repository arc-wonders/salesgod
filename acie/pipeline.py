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
import logging
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
from . import safety
from . import psyche
from .psyche import emotion_policy
from .psyche import style as style_mod
from .psyche.ledger import (CustomerFacts, SaidLedger, clear_input, explicit_bye, firm_no, invented_memory, is_repeat_ask, next_unsaid,
                            norm as _lnorm, one_word_fragment, pure_farewell, sentences as _lsentences, unsourced_claims, _FORWARD)
from .psyche.moves import Move, MoveContext, choose_move, repeat_move
from .psyche.needs import NeedInput, estimate_need
from .schemas import ExpressionParams, HumanState, SessionState, TurnInput, TurnPlan, TurnResult, Utterance
from .state.estimator import HumanStateEstimator
from .state import social as social_mod
from .tree import enrich_plan
from .validator import StreamingValidator

log = logging.getLogger("acie.pipeline")
_HANGUP_TXT = re.compile(r"(फ़ोन|फोन|phone|fon) (रख|rakh|काट|kaat)|रख दीजिए|rakh dijiye|(कुछ|कोई) (नहीं|नही) (खरीदना|लेना|चाहिए)|"
                         r"(kuch|koi) (nahi|nahin) (kharidna|lena|chahiye)|राम[- ]राम|\bram[- ]ram\b", re.I)

_FILLER_OPEN = re.compile(r"^(aapke sawal ka jawab (yeh |ye )?(hai )?(ki |hai ki )?|आपके सवाल का जवाब (यह |ये )?(है )?(कि )?|"
                          r"आपके सवाल का जवाब देते हुए,? |to answer your question,? |the answer to your question is,? )", re.I)
_ASKS_HINDI = re.compile(r"(hindi|हिंदी|हिन्दी)\s*(mein|me|में|मे)\b|(hindi|हिंदी)\s+(bol|baat|बोल|बात)|speak (in )?hindi|talk (in )?hindi", re.I)
_ASKS_ENGLISH = re.compile(r"(english|इंग्लिश|अंग्रेज़ी|अंग्रेजी)\s*(mein|me|में|मे)\s*(baat|bol|बात|बोल|samjha|समझा)|(speak|talk|reply|continue) (in )?english|in english please", re.I)
_ONE_LINE_DEMO = re.compile(r"\b(?:one|ek|1|a|single|just a)\s+(?:line|sentence|lines)\b[^.?!]{0,15}\b(?:hindi|हिंदी)|\b(?:hindi|हिंदी)\b[^.?!]{0,12}\b(?:one|ek|a) (?:line|sentence)|"
                            r"\bsay (?:something|it|that|one line|a line) in hindi|हिंदी में (?:एक )?लाइन", re.I)
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
                         r"marathon|मैराथन|\brun(ning)?\b|\b\d+\s*(km|kilomet)|stamina|endurance|\blean\b|physique|power ?lift|lifting|strength|"
                         r"coding|code|कोडिंग|writing|study|padhai|पढ़ाई|resume|assignment|report", re.I)

_FILLER = {"haan", "han", "ha", "ji", "ok", "okay", "yes", "yeah", "hmm", "hm", "acha", "achha", "accha", "theek", "thik", "hai", "hello", "हाँ", "हां",
           "जी", "ओके", "ठीक", "है", "हम्म", "अच्छा", "हेलो", "bolo", "बोलो", "bataiye", "बताइए", "please", "sure", "nahi", "नहीं", "no"}

_RECORDED = re.compile(r"call is (being )?recorded|रिकॉर्ड हो रही|record ho rahi|कॉल की रिकॉर्डिंग|call ki recording", re.I)
_RECORDED_LINE = {"en": "This call is recorded.", "hi": "ये कॉल रिकॉर्ड हो रही है।", "hinglish": "Ye call record ho rahi hai."}
_LANG_NAME = {"en": "English", "hi": "Hindi in Devanagari script", "hinglish": "Hinglish in Roman script"}
# a reply that is only an apology ("I am so sorry, I did not mean to sound rude at all.") leaves the caller hanging: offer a way forward
_APOLOGY = re.compile(r"\b(sorry|apologi[sz]e|apologies|my (mistake|bad)|did ?n.?t mean|didn.?t intend|maaf|galti)\b|माफ़|माफ|गलती|ग़लती", re.I)
_WAY_FORWARD = {"en": "I'll slow down. What would you like to know?",
                "hi": {"male": "मैं आराम से बताता हूँ। आप क्या जानना चाहेंगे?", "female": "मैं आराम से बताती हूँ। आप क्या जानना चाहेंगे?"},
                "hinglish": {"male": "Main aaram se batata hoon. Aap kya jaanna chahenge?", "female": "Main aaram se batati hoon. Aap kya jaanna chahenge?"}}


def bare_apology(reply: list[str]) -> bool:
    sents = [x for r in reply for x in re.split(r"(?<=[.!?।])\s+", (r or "").strip()) if x]
    return bool(sents) and all(_APOLOGY.search(x) and not _DIDNT.search(x) for x in sents) and not any(x.rstrip().endswith("?") for x in sents) \
        and len(" ".join(sents).split()) <= 30


def way_forward(lang: str, gender: str = "male") -> str:
    v = _WAY_FORWARD.get(lang) or _WAY_FORWARD["en"]
    return v if isinstance(v, str) else v.get(gender or "male", v["male"])


SentenceSink = Callable[[str], Awaitable[None] | None]

# deterministic "no repeating context" guard (see _facts_guard). Plans that are compliance wording or a safety reply never change.
# content the caller explicitly asked for (a live pitch of any item, a one-line demo): never trimmed by the question budget, the
# repeated-question filter or the product facts guard (call 1d5573e7dc: "sell me a pen" was cut to "Nice tries on the classic pen challenge!")
_ASKED_FOR = frozenset({"pitch_demo"})
_GUARD_KEEP = frozenset({"opening", "order_slot", "pitch_demo", "language_switch", "opt_out", "opt_out_confirmed", "declined", "callback", "callback_time", "human_agent", "identify",
                         "are_you_ai", "number_source", "sensitive_stop", "slow_down", "link_cancelled", "confirm_buy", "constraint_stop",
                         "purpose", "pay_at_desk", "goodbye_after"})
_FIRM_END = frozenset({"declined", "opt_out", "opt_out_confirmed", "callback", "human_agent"})  # intents that may end the call by themselves
_DIDNT = re.compile(r"didn.?t (quite )?(get|catch)|say (that )?(it )?(once more|again)|दोबारा|phir se bol|samajh nahi paya|सुन नहीं पाया|समझ नहीं पाया|"
                    r"dobara bol|ek baar phir|एक बार फिर|सुनाई नहीं|sunai nahi|ठीक से सुन|theek se sun", re.I)
_UNKNOWN = re.compile(r"don.?t have (that|any)|won.?t guess|not sure about that|can.?t answer for sure|can.?t say anything sure|that is all i have|"
                      r"nothing more on that|that is really all|rather not guess|jaankari mere paas nahi|pakka nahi|nahi bata sakta|bas itna hi|"
                      r"जानकारी मेरे पास नहीं|पक्का नहीं|नहीं बता सकता|बस इतना ही|अंदाज़ा नहीं", re.I)
_FAREWELL = re.compile(r"धन्यवाद|शुक्रिया|दिन शुभ|thank you|thanks for|have a (nice|good|great) day|take care|goodbye|bye\b|dhanyavaad|din accha|din achha|alvida", re.I)
_FALSE_ALARM = re.compile(r"\b(i.?m|i am) (totally |completely |perfectly |absolutely )?(fine|ok(ay)?|alright|good)\b|i was (just |only )?asking|what are you (saying|talking)|"
                          r"wrong (number|person)|your system is (broken|wrong)|(ठीक|theek|thik) (हूँ|हूं|hoon|hun|hu)|(मैं|main|mai) (तो )?(बस|bas|sirf|सिर्फ़|सिर्फ) .{0,60}(पूछ|pooch|puch)|"
                          r"(ये|yeh|ye|aap) (क्या|kya) (कह|keh|bol|बोल) (रहे|rahe)|सिस्टम खराब|system kharab", re.I)
_REASSURED = re.compile(
    r"ठीक\s*(हूँ|हूं|हु|है)|theek\s*(hoon|hun|hu|hai)|thik\s*(hoon|hun|hu|hai)|\bi\s*'?a?m\s+(fine|ok|okay|good|alright|all right)\b|"
    r"\b(just|only)\s+(asking|wanted|want|checking)|\bbas\s+\w+\s+(puch|pooch|bata|jaan)|\bbas\s+(fees|price|rate|kimat)|बस\s+\S+\s+(पूछ|बता|जान)|"
    r"बस\s+(फ़ीस|फीस|कीमत|रेट)|कोई\s+(दिक्कत|परेशानी)\s+नहीं|no\s+problem|nothing\s+(wrong|like that)|koi\s+(dikkat|problem|baat)\s+nahi", re.I)
_NO_SELL_ROWS = ("sad", "anxious", "angry", "crisis")


# the domain's "what I just said / asked" keys: rolled back when the end of a turn was not heard
_DOMAIN_TURN_KEYS = ("pending", "last_intent", "last_plan", "repeat_n", "sms_offered", "need_asked")


def _amounts_in(text: str) -> set[float]:
    from .context import amounts
    return set(amounts(text or ""))


def _deal_offer(deal) -> float | None:
    if deal is None:
        return None
    v = deal.get("last_agent_offer") if isinstance(deal, dict) else getattr(deal, "last_agent_offer", None)
    try:
        return float(v) if v else None
    except (TypeError, ValueError):
        return None


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
        self._undo: dict[str, dict] = {}  # session id -> the last turn's undo point (apply_heard after a barge-in)
        self.caller_memory = None  # acie.caller_memory.CallerMemoryStore: long-term caller memory, loaded at the first turn
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

    @staticmethod
    def _empty(s: SessionState | None) -> bool:
        """No finished turn in this copy (missing, fresh, or only the greeting)."""
        return s is None or s.turn_index == 0

    def session(self, turn: TurnInput) -> SessionState:
        """The call's session. Missing or empty here (pipeline recreated, session evicted) -> transparently restored from
        the session store's in-process mirror; `restore_session` (run at every turn start) also asks the durable store."""
        sid = turn.session_id
        cur = self.sessions.get(sid)
        if self._empty(cur) and self.session_store is not None and hasattr(self.session_store, "get_cached"):
            saved = self.session_store.get_cached(sid)
            if saved is not None and not self._empty(saved):
                if cur is not None and cur.history:
                    saved.history += [u for u in cur.history if u not in saved.history]
                self.sessions[sid] = cur = saved
        if cur is None:
            self.sessions[sid] = cur = SessionState(session_id=sid, customer_id=turn.customer_id)
        return cur

    async def restore_session(self, turn: TurnInput) -> SessionState:
        """Turn start, the normal path: if this process has no (or an empty) copy of the session, load the latest
        snapshot (NATS KV / stream, Redis) within a tight budget (ACIE_RESTORE_TIMEOUT_MS, default 50); else memory."""
        sid = turn.session_id
        if self._empty(self.sessions.get(sid)) and self.session_store is not None:
            budget = float(os.getenv("ACIE_RESTORE_TIMEOUT_MS", "50")) / 1000
            try:
                saved = await asyncio.wait_for(self.session_store.get(sid), budget)
            except asyncio.TimeoutError:
                saved = None
                log.warning("session %s: restore timed out after %.0f ms, continuing from memory", sid, budget * 1000)
            except Exception as e:  # the durable copy must never break a live call
                saved = None
                log.warning("session %s: restore failed (%s), continuing from memory", sid, e)
            if saved is not None and not self._empty(saved):
                cur = self.sessions.get(sid)
                if cur is not None and cur.history:  # e.g. a line spoken into the fresh copy meanwhile: keep it
                    saved.history += [u for u in cur.history if u not in saved.history]
                self.sessions[sid] = saved
                log.info("session %s restored at turn %d (deal %s, checklist turn %d)", sid, saved.turn_index,
                         (saved.context.deal_view or {}).get("status", "-"), saved.context.checklist.updated_turn)
        sess = self.session(turn)
        if self.caller_memory is not None and not sess.context.caller and not sess.dialog.get("_caller_tried"):
            sess.dialog["_caller_tried"] = "1"
            try:
                mem = await asyncio.wait_for(self.caller_memory.get(sess.customer_id or turn.customer_id), 0.3)
            except Exception:
                mem = None
            if mem:
                sess.context.caller = mem
        return sess

    # ------------------------------------------------------------------ only what the caller HEARD counts (barge-in)

    _AGENT_SIDE = ("told", "quoted", "promised", "last_agent_line", "last_agent_intent")

    def _undo_point(self, session: SessionState, facts) -> dict:
        import copy
        ctx = session.context
        return {"ctx": {k: copy.deepcopy(getattr(ctx, k)) for k in self._AGENT_SIDE},
                "checklist": ctx.checklist.model_copy(deep=True), "deal_view": dict(ctx.deal_view),
                "deal": copy.deepcopy(ctx.deal), "dialog": {k: session.dialog.get(k) for k in ("_deal", "_neg", "_said", "_cf") + _DOMAIN_TURN_KEYS},
                "facts": copy.deepcopy(facts)}

    def _product_names(self) -> tuple[str, str, str]:
        p = getattr(self.decision.domain, "p", None)
        return (getattr(p, "agent_name", "") or "", getattr(p, "seller_name", "") or "", getattr(p, "product_name", "") or "")

    def purpose_texts(self) -> dict[str, str] | None:
        agent, seller, product = self._product_names()
        if not (seller and product):
            return None
        fem = getattr(getattr(self.decision.domain, "p", None), "agent_gender", "male") == "female"
        rh, rg = ("रही", "rahi") if fem else ("रहा", "raha")
        return {"en": f"Just so you know, I'm {agent} calling from {seller} about an offer on {product}.",
                "hi": f"बता दूँ, मैं {seller} से {agent} बोल {rh} हूँ, {product} के एक ऑफ़र के बारे में।",
                "hinglish": f"Bata doon, main {seller} se {agent} bol {rg} hoon, {product} ke ek offer ke baare mein."}

    def _purpose_first(self, plan, session: SessionState, lang: str):
        """The caller must know who is calling and that it is a sales call about the product. If a barge-in cut the
        opening/pitch before they heard it, the next agent turn says it first, briefly (once per cut)."""
        ctx = session.context
        if plan.intent == "purpose":  # the opening's purpose step says who / why / recorded itself
            session.dialog.pop("_purpose_due", None)
            return plan
        if session.dialog.get("_purpose_due") != "1" or ("who_is_calling" in ctx.told and "what_the_product_is" in ctx.told):
            session.dialog.pop("_purpose_due", None)
            return plan
        if plan.end_call or plan.intent in ("goodbye", "opt_out", "opt_out_confirmed", "declined", "annoyed_exit", "safety_distress",
                                            "safety_followup", "constraint_stop", "repeat_last", "pitch", "opening", "identify"):
            return plan
        texts = self.purpose_texts()
        if not texts:
            return plan
        session.dialog.pop("_purpose_due", None)
        if "call_recorded" not in ctx.told and os.getenv("ACIE_RAPPORT_OPENING", "1") != "0":  # the recording notice was never heard either
            texts = {l: t + " " + _RECORDED_LINE.get(l, _RECORDED_LINE["en"]) for l, t in texts.items()}
            ctx.told.append("call_recorded")
        lead = {l: (t + " " + ((plan.lead or {}).get(l) or "")).strip() for l, t in texts.items()}
        for what in ("who_is_calling", "what_the_product_is"):
            if what not in ctx.told:
                ctx.told.append(what)
        return plan.model_copy(update={"lead": lead})

    def apply_heard(self, session_id: str, heard_text: str, heard_ratio: float, intent: str = "") -> bool:
        """Barge-in: the caller heard only `heard_text` of the last agent turn. Roll the agent side of the context back to
        before that turn and re-apply only what was heard: last line, told/quoted/promised, checklist, SaidLedger slots,
        the deal's last agent offer. Unheard facts stay unsaid, so the next turns state them again. -> applied?"""
        sess = self.sessions.get(session_id)
        if sess is None:
            return False
        ctx, d = sess.context, sess.dialog
        undo = self._undo.get(session_id)
        heard = (heard_text or "").strip()
        full = undo.get("text", "") if undo else ""
        if undo is not None and undo.get("turn") == sess.turn_index - 1:
            for k, v in undo["ctx"].items():
                setattr(ctx, k, v)
            ctx.checklist, ctx.deal_view = undo["checklist"], undo["deal_view"]
            plan = undo["plan"]
            intent = plan.intent
            # the deal: an offer the caller never heard was never made
            new_offer = _deal_offer(ctx.deal)
            if new_offer and new_offer != _deal_offer(undo["deal"]) and new_offer not in _amounts_in(heard):
                ctx.deal = undo["deal"]
                for k in ("_deal", "_neg"):
                    if undo["dialog"].get(k) is None:
                        d.pop(k, None)
                    else:
                        d[k] = undo["dialog"][k]
            for k in ("_said", "_cf") + (_DOMAIN_TURN_KEYS if heard_ratio < 0.9 else ()):  # the unheard question isn't pending
                if undo["dialog"].get(k) is None:
                    d.pop(k, None)
                else:
                    d[k] = undo["dialog"][k]
            if heard:
                self._learn_said(SaidLedger.load(d), sess, plan, undo.get("move"), heard, undo["facts"])
            agent, seller, product = self._product_names()
            low = heard.lower()
            from .context import _AGENT_EVENTS
            if intent in _AGENT_EVENTS:
                kind, what = _AGENT_EVENTS[intent]
                ok = {"offer_price": bool(_amounts_in(heard)) or "₹" in heard,
                      "what_the_product_is": bool(product) and product.lower().split()[0] in low,
                      "who_is_calling": any(x and x.lower() in low for x in (agent, seller)),
                      "payment_link": "link" in low or "लिंक" in heard}.get(what, heard_ratio >= 0.9)
                if ok and what not in getattr(ctx, kind):
                    getattr(ctx, kind).append(what)
            if _RECORDED.search(heard) and "call_recorded" not in ctx.told:
                ctx.told.append("call_recorded")
            if any(t and t in heard for t in (self.purpose_texts() or {}).values()):  # the purpose line itself was heard
                for what in ("who_is_calling", "what_the_product_is"):
                    if what not in ctx.told:
                        ctx.told.append(what)
            p_ = getattr(self.decision.domain, "p", None)
            eco = getattr(p_, "economics", None)
            ctx.after_reply(undo["turn"], undo.get("customer_text", ""), intent if heard_ratio >= 0.9 else "partial", heard,
                            undo.get("lang", ""), d, float(getattr(eco, "list_price", 0) or 0), getattr(eco, "currency", "") or "")
            if not heard:
                ctx.last_agent_line = undo["ctx"]["last_agent_line"]
        else:  # e.g. the opening (spoken outside run_turn): only drop the facts it did not get across
            agent, seller, product = self._product_names()
            low = heard.lower()
            if "who_is_calling" in ctx.told and not any(x and x.lower() in low for x in (agent, seller)):
                ctx.told.remove("who_is_calling")
            if "what_the_product_is" in ctx.told and not (product and product.lower().split()[0] in low):
                ctx.told.remove("what_the_product_is")
            if "call_recorded" in ctx.told and not _RECORDED.search(heard):
                ctx.told.remove("call_recorded")
            if heard:
                ctx.last_agent_line = heard
        for u in reversed(sess.history):  # the transcript the LLM sees: only the heard words
            if u.role == "agent":
                u.text = (heard + " —") if heard else "(cut off before saying anything)"
                break
        if not ("who_is_calling" in ctx.told and "what_the_product_is" in ctx.told) and self.purpose_texts():
            d["_purpose_due"] = "1"
        ctx.add(__import__("acie.context", fromlist=["ContextEvent"]).ContextEvent(
            turn=max(sess.turn_index - 1, 0), who="system", kind="interrupted", detail=f"heard {heard_ratio:.0%}"), self.context_sinks)
        st = self.session_store
        if st is not None and hasattr(st, "record_turn"):
            st.record_turn(sess, {"type": "heard", "v": 1, "session_id": session_id, "turn": sess.turn_index - 1, "ts": time.time(),
                                  "heard_text": heard, "heard_ratio": round(heard_ratio, 2), "checklist": ctx.checklist.model_dump(),
                                  "deal_view": ctx.deal_view})
        return True

    def _recall_plan(self, text: str, session: SessionState) -> TurnPlan | None:
        """'Can you repeat that?' -> the last agent line, word for word. 'Where were we?' -> the state restated from the
        ledger (checklist + deal). Both are answered from state, before any domain logic, so neither turns into a new
        pitch, a guess or a fresh concession."""
        from .context import recall_kind, repeat_texts, where_texts
        if os.getenv("ACIE_RECALL", "1") == "0" or session.turn_index == 0 and not session.history:
            return None
        kind = recall_kind(text)
        if kind is None:
            return None
        ctx = session.context
        fallback_line = next((u.text for u in reversed(session.history) if u.role == "agent"), "")
        if kind == "repeat":
            texts = repeat_texts(ctx, fallback_line)
            if not texts:
                return None
            return TurnPlan(intent="repeat_last", llm_allowed=False, decision="repeat request: replay the last agent line from state",
                            objective="They asked to hear the last thing again: say it again, word for word", fallback=texts)
        dom = self.decision.domain
        deal = ctx.deal
        if deal is None and session.dialog.get("_deal"):
            try:
                deal = json.loads(session.dialog["_deal"])
            except (ValueError, TypeError):
                deal = None
        status = (deal.get("status") if isinstance(deal, dict) else getattr(deal, "status", "")) if deal is not None else ""
        if status in ("open", "countering", "agreed", "closed", "walked") and getattr(dom, "name", "") == "product_sales":
            return None  # a live deal: the negotiation domain (agent haggle) restates it from DealState, no new concession
        p = getattr(dom, "p", None)
        recap = None
        if hasattr(dom, "deal_recap"):  # the negotiation domain (agent haggle) phrases the deal part
            try:
                recap = dom.deal_recap(session)
            except Exception as e:
                log.warning("deal_recap failed: %s", e)
        texts = where_texts(ctx, getattr(p, "product_name", "") if p is not None else "", recap)
        return TurnPlan(intent="where_were_we", llm_allowed=False, decision="where were we: restate the state from the ledger",
                        objective="They lost the thread: restate where we are (deal, what is agreed, the open question); no new offer",
                        fallback=texts)

    def _turn_event(self, turn: TurnInput, session: SessionState, plan, text: str, expr, delta: dict, t: dict) -> dict:
        ctx = session.context
        return {"v": 1, "session_id": session.session_id, "turn": session.turn_index, "ts": time.time(),
                "customer_text": turn.text, "agent_text": text, "intent": plan.intent, "decision": str(plan.decision or "")[:200],
                "end_call": bool(plan.end_call), "language": expr.language, "checklist_delta": delta,
                "checklist": ctx.checklist.model_dump(), "deal_view": ctx.deal_view, "outcome": ctx.outcome,
                "ctx_decision": ctx.decision, "timings_ms": t}

    async def _persist(self, turn: TurnInput, session: SessionState, plan, text: str, expr, delta: dict, t: dict) -> None:
        """After every turn: the snapshot + one event. NATS: queued, sent in the background (never waits). Other stores
        (Redis): awaited with a small budget."""
        st = self.session_store
        if st is None:
            return
        if hasattr(st, "record_turn"):
            try:
                st.record_turn(session, self._turn_event(turn, session, plan, text, expr, delta, t))
            except Exception as e:
                log.warning("session %s: snapshot not queued (%s)", session.session_id, e)
            return
        try:
            await asyncio.wait_for(st.put(session), float(os.getenv("ACIE_PERSIST_BUDGET_MS", "250")) / 1000)
        except Exception:  # the durable copy must never break a live call
            pass

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
        session.dialog.pop(social_mod.NLU_KEY, None)
        sig = await self._hub().run(turn, session, dom, getattr(dom, "classifier", None))
        try:  # social signals (sarcasm, humour, hostility, interest, impatience, scepticism, rapport): lexicon + NLU, smoothed
            raw_soc = session.dialog.get(social_mod.NLU_KEY)
            social_mod.update(session.dialog, turn.text, json.loads(raw_soc) if raw_soc else None)
        except Exception as e:  # never let a signal break a turn
            log.info("social signals failed: %s", type(e).__name__)
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
                           hold=session.dialog.get("sad_hold") == "1", signals=session.dialog)

    def _psyche_move(self, turn: TurnInput, session: SessionState, state: HumanState, intent: str, strat,
                     ledger: SaidLedger, lang: str):
        from .state.dims import state_dims
        ctx = session.context
        need = estimate_need(NeedInput(text=turn.text, state=state_dims(state), intent=intent,
                                       has_link=ctx.has("payment_link"), decided=ctx.decision,
                                       turn_index=session.turn_index, dialog=session.dialog,
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

    # ------------------------------------------------------------------ no repeating context (CustomerFacts, deterministic)

    @staticmethod
    def _plan_text(plan, lang: str) -> str:
        probe = StreamingValidator(plan, ExpressionParams(language=lang), trusted=True, customer_text="")
        parts = []
        if plan.lead:
            parts.append(plan.lead.get(lang) or plan.lead.get("en") or "")
        parts.append(plan.fallback.get(lang) or plan.fallback.get("en") or "")
        return " ".join(probe.fill(x) for x in parts if x)

    @staticmethod
    def _claim_sources(plan, customer_text: str) -> str:
        bits = [customer_text, plan.context or "", plan.reference or "", plan.objective or ""]
        bits += [f.value + " " + (f.description or "") for f in plan.facts.values()]
        bits += list((plan.lead or {}).values()) + list(plan.fallback.values())
        return " ".join(bits)

    def reply_issues(self, text: str, ledger: SaidLedger, facts: CustomerFacts, customer_text: str = "", plan=None) -> list[tuple[str, str]]:
        """What is wrong with a reply, as (code, detail): invented memory, a repeated sentence, a restated caller fact, a re-asked
        question, an unsourced claim. Empty for a caller who asked us to repeat ('repeat_ask')."""
        out: list[tuple[str, str]] = []
        m = invented_memory(text)
        if m:
            out.append(("invented_memory", m))
        if is_repeat_ask(customer_text):
            return out
        thr = 0.9 if is_question(customer_text) else 0.6  # a direct question may get the same fact again, reworded
        for s in _lsentences(text):
            if len(_lnorm(s).split()) >= 3 and ledger.similar_to(s, thr):
                out.append(("repeated_sentence", s))
        if ledger.replies and ledger.reply_repeats(text, 0.85 if is_question(customer_text) else 0.6) and not any(c == "repeated_sentence" for c, _ in out):
            out.append(("repeated_sentence", text[:80]))
        out += [("restated_fact", s) for s in facts.restates(text)]
        out += [("reasked_question", t) for t in facts.reasks(text)]
        if plan is not None:
            out += [("unsourced_claim", c) for c in unsourced_claims(text, self._claim_sources(plan, customer_text))]
        return out

    def _new_point(self, session: SessionState, ledger: SaidLedger, facts: CustomerFacts, lang_text: str, intent: str,
                   hold: bool) -> tuple[dict[str, str], str | None]:
        """The next thing not yet said, in all three languages: (text per lang, slot in the call's language)."""
        p = getattr(self.decision.domain, "p", None)
        sell_ok = intent in ("confirm_close", "price", "reoffer", "pitch", "price_floor") and not (ledger.refused or hold)
        link_ok = intent == "confirm_close" and sell_ok
        text, slot = {}, None
        cur = session.dialog.get("_lang") or "en"
        for lang in ("en", "hi", "hinglish"):
            sl, line = next_unsaid(None if (hold or ledger.refused or link_ok) else p, lang, ledger, facts, lang_text, allow_sell=sell_ok, allow_link=link_ok)
            if not (hold or ledger.refused or link_ok):
                q = facts.next_discovery(p, lang)  # the NEXT unanswered level of the discovery ladder, never one already answered
                if q and not ledger.repeats(q) and not ledger.reply_repeats(q) and style_mod.discovery_ok(session.dialog):
                    if sl is None:
                        line = q
                    elif facts.just_answered:  # they just answered a level: a new point, then the next level down
                        line = f"{line} {q}"
            text[lang] = line
            if lang == cur:
                slot = sl
        return text, slot

    _PLACEHOLDER = re.compile(r"\{[A-Za-z_][\w\[\]\.]*\}|\{\{|\}\}")
    _SMS_CLAIM = re.compile(r"\bsms\b|एसएमएस|text message", re.I)

    def _drop_broken(self, reply: list[str]) -> list[str]:
        """LLM reply: a sentence with a raw template slot ("{talking_points[0]}") is never spoken; nor an SMS claim when the product
        sends no SMS (the domain's own link lines handle that honestly)."""
        dom = self.decision.domain
        sms_on = bool(getattr(dom, "_sms_on", lambda: True)()) if getattr(dom, "name", "") == "product_sales" else True
        out, changed = [], False
        for r in reply:
            sents = [x for x in re.split(r"(?<=[.!?।])\s+", (r or "").strip()) if x]
            keep = [x for x in sents if not self._PLACEHOLDER.search(x) and (sms_on or not self._SMS_CLAIM.search(x))]
            changed = changed or len(keep) != len(sents)
            if keep:
                out.append(" ".join(keep))
        return out if changed else reply

    def _facts_guard(self, plan, expr, session: SessionState, ledger: SaidLedger, facts: CustomerFacts, state, text: str):
        """_facts_guard_core, then the same filler is never used twice: a used leading filler becomes the next unused synonym."""
        plan = self._facts_guard_core(plan, expr, session, ledger, facts, state, text)
        if os.getenv("ACIE_FACTS_GUARD", "1") == "0" or getattr(self.decision.domain, "name", "") != "product_sales" or plan.intent in _GUARD_KEEP \
                or plan.intent.startswith(("safety", "distress", "concern")) or explicit_bye(text) or not (not plan.llm_allowed or isinstance(self.backend, TemplateBackend)):
            return plan
        lang = expr.language
        cand = self._plan_text(plan, lang)
        new = facts.rotate_opener(cand, lang) if cand else cand
        if new and new != cand:
            return plan.model_copy(update={"lead": None, "facts": {}, "llm_allowed": False, "fallback": {**plan.fallback, lang: new}})
        return plan

    def _facts_guard_core(self, plan, expr, session: SessionState, ledger: SaidLedger, facts: CustomerFacts, state, text: str):
        """Final deterministic gate on a plan whose words are fixed (no LLM, or the offline template backend): never repeat a sentence we
        said, never say the caller's own facts back, never re-ask what was answered, never "I didn't catch that" for clear words, never
        invented CRM memory. Repeated/restated parts are dropped; if nothing new is left, the next unsaid benefit/fact (or a short new
        forward step) is said instead. 'repeat_ask' (the caller asked us to repeat) is exempt. ACIE_FACTS_GUARD=0 turns it off."""
        dom = self.decision.domain
        if os.getenv("ACIE_FACTS_GUARD", "1") == "0" or getattr(dom, "name", "") != "product_sales":
            return plan
        if plan.intent in _GUARD_KEEP or plan.intent.startswith(("safety", "distress", "concern")) or plan.reference \
                or (plan.decision or "").startswith(("garbled input", "audio check", "negotiation")) \
                or plan.intent in ("confirm_close", "price_floor") \
                or (plan.intent == "link_sent" and not plan.lead):  # a plain link confirmation (at the agreed price) is never swapped for a fact; the negotiation, floor and confirmation lines carry the price on purpose
            return plan  # (a garbled-line "say it again" and the audio answer are the point of the turn, never rewritten)
        if not (not plan.llm_allowed or isinstance(self.backend, TemplateBackend)):
            return plan
        if explicit_bye(text):
            return plan  # their goodbye / opt-out: the closing words are required, never rewritten
        lang = expr.language
        cand = self._plan_text(plan, lang)
        if not cand.strip():
            return plan
        hold = session.dialog.get("sad_hold") == "1" or session.dialog.get("_row") in _NO_SELL_ROWS
        issues = self.reply_issues(cand, ledger, facts, text, plan)
        codes = {c for c, _ in issues}
        clear = clear_input(text)
        generic_didnt = bool(_DIDNT.search(cand)) and clear and not is_repeat_ask(text)
        unknown = bool(_UNKNOWN.search(cand))
        asks_more = bool(re.search(r"what (will|do|can) i get|what.?s (in|inside|included)|what does it (do|include|give)|kya milega|kya milta|क्या मिलेगा|क्या मिलता|"
                                   r"benefits?|features?|\baur\b|और (कुछ|क्या)|anything else|what else|kuch aur|कुछ और|\bmore\b|kya kya|क्या क्या|tell me about", text, re.I))
        unknown_swap = unknown and not is_repeat_ask(text) and (asks_more or (clear and not is_question(text)))
        if not (codes or generic_didnt or unknown_swap):
            return plan
        keep: list[str] = []
        if codes and not (generic_didnt or unknown_swap) and codes <= {"repeated_sentence", "restated_fact", "reasked_question"} and plan.intent not in ("link_sent",):
            bad = {s for c, s in issues if c in ("repeated_sentence", "restated_fact")}
            keep = [s for s in _lsentences(cand) if s not in bad and not (s.rstrip().endswith("?") and "reasked_question" in codes and facts.reasks(s))]
            if any(not s.rstrip().endswith("?") for s in keep):
                session.dialog["_facts_guard"] = str(int(session.dialog.get("_facts_guard", "0")) + 1)
                facts.guards["trimmed"] = facts.guards.get("trimmed", 0) + 1
                return plan.model_copy(update={"decision": "facts_guard:trim", "llm_allowed": False, "lead": None, "facts": {},
                                               "fallback": {l: " ".join(keep) for l in ("en", "hi", "hinglish")}})
        why = "didnt_catch" if generic_didnt else ("unknown" if unknown_swap else "+".join(sorted(codes)))
        texts, slot = self._new_point(session, ledger, facts, text, plan.intent, hold)
        facts.guards["forward"] = facts.guards.get("forward", 0) + (1 if slot is None else 0)
        session.dialog["_facts_guard"] = str(int(session.dialog.get("_facts_guard", "0")) + 1)
        session.dialog["_cf_slot"] = slot or ""
        keep_end = plan.end_call and plan.intent in _FIRM_END
        return plan.model_copy(update={"intent": "answer" if plan.intent in ("not_understood", "clarify", "small_talk", "off_topic", "answer") else plan.intent,
                                       "llm_allowed": False, "end_call": keep_end, "decision": f"facts_guard:{why}", "fallback": texts,
                                       "lead": None, "facts": {}, "objective": "Say the next thing not said yet"})

    def _fragment_plan(self, session: SessionState, expr, ledger: SaidLedger) -> TurnPlan:
        """A one-word fragment ("मुझे", "mujhe", "main"): the caller is about to say it. A short 'ji, bataiye', no answer, no price."""
        n = ledger.count_move("fragment")
        ledger.add_move("fragment")
        lines = {"en": ("Yes, please go on.", "Go ahead, I'm listening."), "hi": ("जी, बताइए।", "जी, बोलिए।"),
                 "hinglish": ("Ji, bataiye.", "Ji, boliye.")}
        return TurnPlan(intent="acknowledge", objective="Invite the caller to finish", llm_allowed=False, decision="fragment",
                        fallback={l: v[n % 2] for l, v in lines.items()})

    # ------------------------------------------------------------------ conversational style (acie/psyche/style.py, ACIE_STYLE)

    def _decide_style(self, session: SessionState, state, strat, text: str, intent: str, facts: CustomerFacts, before: dict):
        """This turn's style row, stored in session.dialog['_style'] (the compiler reads it for the LLM). None when off."""
        session.dialog.pop(style_mod.STYLE_KEY, None)
        session.dialog.pop("_style_used", None)
        if not style_mod.enabled() or getattr(self.decision.domain, "name", "") != "product_sales":
            return None
        new = [h for k, v in facts.caller.items() if k != "name" for h in v if h not in before.get(k, [])]
        dom = self.decision.domain
        elder = session.dialog.get("_elder") == "1" or (getattr(dom, "customer_age", None) or 0) >= 55
        st = style_mod.decide_style(row=strat.row, state=state, text=text, intent=intent, turn_index=session.turn_index,
                                    dialog=session.dialog, facts=facts, elder=elder, garbled=not clear_input(text), new_detail=new)
        session.dialog[style_mod.STYLE_KEY] = st.to_json()
        return st

    def _backchannel_plan(self, session: SessionState, expr, ledger: SaidLedger) -> TurnPlan:
        """The caller is mid-sentence ("So basically,"): a short backchannel, no fact, no price, no question."""
        n = ledger.count_move("backchannel")
        ledger.add_move("backchannel")
        g = self._agent_gender()
        return TurnPlan(intent="acknowledge", objective="Let the caller finish their sentence", llm_allowed=False, decision="style:backchannel",
                        fallback={l: style_mod.pick("GO_ON", l, n, g) for l in ("en", "hi", "hinglish")})

    def _handback_plan(self, session: SessionState, expr, ledger: SaidLedger) -> TurnPlan:
        """The caller says we pushed ("rude", "did you even ask me?"): own it and hand control back with one question."""
        n = ledger.count_move("handback")
        ledger.add_move("handback")
        g = self._agent_gender()
        session.dialog["pending"] = "pitch"  # their next yes means "tell me what's included", never a confirmation
        session.dialog["handback_open"] = "1"  # the domain answers that yes with what's included (no price)
        return TurnPlan(intent="apology", objective="Own that you rushed, then ask whether they want to hear what's included or stop",
                        llm_allowed=False, decision="style:own_and_hand_back",
                        fallback={l: style_mod.pick("HANDBACK", l, n, g) for l in ("en", "hi", "hinglish")})

    def _agent_gender(self, plan=None) -> str:
        from .agents import gender_from_packs
        g = gender_from_packs(getattr(plan, "rule_packs", None)) if plan is not None else ""
        if not g:
            g = getattr(getattr(self.decision.domain, "p", None), "agent_gender", "") or ""
        return g if g in ("male", "female") else ""

    _STYLE_KEEP_ASK = frozenset({"negotiation", "order_slot", "pitch_demo", "confirm_close", "link_sent", "price_floor", "callback", "callback_time", "opening", "clarify",
                                 "reason_ask", "language_switch", "opt_out", "declined", "human_agent", "identify", "are_you_ai", "purpose", "pitch",
                                 "pay_at_desk"})
    _ASKS_PRICE = re.compile(r"price|cost|kitne|कितन|कीमत|rate|how much|fees?|फ़ीस|फीस|charge|paisa|पैसे|₹|\d", re.I)

    def _style_keeps_asks(self, plan) -> bool:
        return plan.intent in self._STYLE_KEEP_ASK or str(plan.decision or "").startswith(("negotiation", "garbled")) \
            or plan.intent.startswith(("safety", "distress", "concern"))

    def _permission_first(self, session: SessionState, text: str) -> bool:
        """A price quote the caller did not ask for, after appreciation ("love it", "nice", "okay") instead of a clear yes:
        ask permission first (once). A plain "ok / yes / sure" right after our own "shall I tell you the price?" is consent."""
        d = session.dialog
        if not style_mod.enabled() or self._ASKS_PRICE.search(text or "") or int(d.get("_style_perm", "0") or 0) >= 1:
            return False
        if not style_mod.APPRECIATE.search(text or ""):
            return False
        last = next((u.text for u in reversed(session.history) if u.role == "agent"), "")
        asked_price = bool(re.search(r"(price|offer|कीमत|ऑफर|daam|keemat|kimat)[^?]*\?", last, re.I))
        consent = bool(re.fullmatch(r"\W*(ok(ay)?|yes|yeah|yep|sure|haan|haan ji|ji|हाँ|हां|जी|हाँ जी|theek hai|ठीक है|go ahead|bataiye|बताइए)\W*",
                                    text or "", re.I))
        return not (asked_price and consent)

    _PRICE_TURN = frozenset({"price", "confirm_close", "negotiation", "price_floor", "reoffer", "link_sent"})

    def _price_turn(self, plan, text: str) -> bool:
        return plan.intent in self._PRICE_TURN or str(plan.decision or "").startswith("negotiation") \
            or bool(self._ASKS_PRICE.search(text or "")) or bool(re.search(r"discount|offer|deal|sasta|mehnga|महंग|सस्ता|छूट", text or "", re.I))

    def _offer_off_topic(self, plan, text: str) -> bool:
        return "offer" in plan.facts and not plan.facts["offer"].hidden and not self._price_turn(plan, text) \
            and "{offer}" not in " ".join(list((plan.lead or {}).values()) + [plan.objective or ""])

    def _style_drop_recent_price(self, reply: list[str], plan, session: SessionState, text: str) -> list[str]:
        """LLM reply: a sentence restating a price said in the last 2 agent turns goes, unless the caller asked about price or
        this is a price / negotiation / close turn (only when a sentence without it is left)."""
        if not style_mod.enabled() or not reply or self._price_turn(plan, text):
            return reply
        b = style_mod.budgets(session.dialog)
        recent = {p for p, n in b["prices"].items() if b["turns"] - n < 2}
        if not recent:
            return reply
        sents = [x for r in reply for x in re.split(r"(?<=[.!?।])\s+", r.strip()) if x]
        keep = [x for x in sents if not any(m.group(0).replace(" ", "") in recent for m in style_mod.PRICE_RX.finditer(x))]
        if len(keep) == len(sents) or not keep:
            return reply
        return [" ".join(keep)]

    def _style_drop_asks(self, reply: list[str], plan, session: SessionState) -> list[str]:
        """LLM reply: a turn that may not ask loses its questions (only when a statement is left); a discovery question
        outside the discovery budget goes too. Negotiation / close / safety lines keep theirs."""
        st = style_mod.Style.from_json(session.dialog.get(style_mod.STYLE_KEY))
        if st is None or not reply or self._style_keeps_asks(plan):
            return reply
        disc_ok = style_mod.discovery_ok(session.dialog)
        sents = [x for r in reply for x in re.split(r"(?<=[.!?।])\s+", r.strip()) if x]
        keep = [x for x in sents if not (x.rstrip().endswith("?") and (not st.ask_ok or (style_mod.is_discovery(x) and not disc_ok)))]
        if not keep or not any(not x.rstrip().endswith("?") for x in keep):
            return reply
        return [" ".join(keep)] if len(keep) != len(sents) else reply

    def _apply_style(self, plan, expr, session: SessionState, st, text: str):
        """Template turns: prepend a rotated lead (label / mirror / backchannel / humour / proof) and drop the ask when this turn
        may not ask; a price line said in the last 2 turns is not said again unless they asked. LLM turns: the plan's ask is
        dropped when the turn may not ask (the compiler adds the style card). Negotiation and close lines keep their numbers and
        their ask; only a lead may be prepended."""
        if st is None or st.style in ("safety", "listening") or getattr(self.decision.domain, "name", "") != "product_sales":
            return plan
        if plan.intent in _GUARD_KEEP and plan.intent not in ("callback",) or plan.intent in ("repeat_last", "where_were_we", "acknowledge") \
                or plan.intent.startswith(("safety", "distress", "concern")) or explicit_bye(text) or plan.end_call:
            return plan
        d = session.dialog
        if st.row in (6, 7):
            d["_style_price_obj"] = str(int(d.get("_style_price_obj", "0") or 0) + 1)
        keeps = self._style_keeps_asks(plan)
        template = not plan.llm_allowed or isinstance(self.backend, TemplateBackend)
        if plan.intent == "price" and self._permission_first(session, text):
            # owner (call f79a90af02): "Love it." after the pitch got the whole price + a goal question -> "too rude". Ask first.
            n = int(d.get("_style_perm", "0") or 0)
            d["_style_perm"] = str(n + 1)
            d["last_intent"] = "pitch"
            d["pending"] = "pitch"
            d["quoted_last"] = "0"
            g = self._agent_gender(plan)
            return plan.model_copy(update={"intent": "pitch", "llm_allowed": False, "lead": None, "ask": None, "facts": {},
                                           "decision": "style:permission_before_price",
                                           "objective": "Glad they like it; ask if they want to hear the offer (no price yet)",
                                           "fallback": {l: style_mod.pick("PERMISSION", l, n, g) for l in ("en", "hi", "hinglish")}})
        if not template:
            upd = {}
            if plan.ask and not st.ask_ok and not keeps:
                upd["ask"] = None
            if self._offer_off_topic(plan, text):
                # the LLM may use the {offer} slot only on a price turn (live sim: "help your ₹2,999 for 3 months... training")
                upd["facts"] = {k: (f.model_copy(update={"hidden": True}) if k == "offer" else f) for k, f in plan.facts.items()}
            return plan.model_copy(update=upd) if upd else plan
        b = style_mod.budgets(d)
        lang = expr.language
        g = self._agent_gender(plan)
        dom = self.decision.domain
        elder = d.get("_elder") == "1" or (getattr(dom, "customer_age", None) or 0) >= 55
        # the lead
        key = st.lead_key
        if key == "HUMOR" and not style_mod.humour_allowed(st, d.get("_row", ""), elder, session.turn_index, text, d):
            key = ""
        if key == "PROOF":
            key = ""  # proof lines come only from an operator-approved product list (none configured yet)
            proof = list(getattr(getattr(dom, "p", None), "proof", None) or [])
            if proof:
                key = "PROOF"
        if key in ("LABEL_UPSET",) and keeps and plan.intent != "negotiation" or plan.intent in ("clarify", "not_understood", "apology") \
                or str(plan.decision or "").startswith(("garbled", "audio")):
            key = ""  # a "say that again" or an apology line carries its own acknowledgement
        neg = str(plan.decision or "").startswith("negotiation") or plan.intent in ("negotiation", "confirm_close", "link_sent", "price_floor")
        if neg and key not in ("LABEL_PRICE", "LABEL_UPSET"):
            key = ""  # negotiation lines: only a label lead, the line itself (numbers, ask) stays as it is
        if neg and key and len(self._plan_text(plan, lang).split()) > 16:
            key = ""  # a long terms line gets no extra sentence in front
        if not key and (plan.intent == "price" or (st.row == 2 and not neg)) \
                and plan.intent not in ("clarify", "not_understood", "apology"):
            key = "SOFT"  # owner: short is not curt; one warm word ("Sure." / "Ji.") in front of price and held-back lines
        out: dict[str, str] = {}
        leads: dict[str, str] = {}
        used = ""
        asked_price = bool(self._ASKS_PRICE.search(text or ""))
        disc_ok = style_mod.discovery_ok(d)
        probe = StreamingValidator(plan, ExpressionParams(language=lang), trusted=True, customer_text="")
        for l in ("en", "hi", "hinglish"):
            raw = plan.fallback.get(l)
            if not raw:
                continue
            # the plan's own words keep their {slots}, lead and facts; only whole sentences are dropped and a lead prepended
            sents = [x for x in re.split(r"(?<=[.!?।])\s+", raw.strip()) if x]
            keep = sents
            if not neg:
                keep = [x for x in keep if not (x.rstrip().endswith("?") and (
                    (not st.ask_ok and not keeps) or (style_mod.is_discovery(x) and (not disc_ok or plan.intent == "price"))))]
                if not asked_price and plan.intent not in ("price", "price_floor"):
                    recent = {p for p, n in b["prices"].items() if b["turns"] - n < 2}
                    keep = [x for x in keep if not any(m.group(0).replace(" ", "") in recent for m in style_mod.PRICE_RX.finditer(probe.fill(x)))]
                stmt = " ".join(probe.fill(x) for x in keep if not x.rstrip().endswith("?"))
                if len(stmt.split()) < 5 or _DIDNT.search(stmt):
                    keep = sents  # a bare filler or "didn't catch that" needs its question
            if keep != sents:
                out[l] = " ".join(keep)
            lead = ""
            if key == "PROOF":
                lead = proof[style_mod.crc_n(session.session_id, str(b["leads"])) % len(proof)]
                lead = lead.get(l, "") if isinstance(lead, dict) else (lead if l == "en" else "")
            elif key == "MIRROR":
                lead = style_mod.pick("MIRROR", l, b["leads"], g, " ".join(st.mirror[:3]))
            elif key:
                lead = style_mod.pick(key, l, b["leads"], g)
            old_lead = (plan.lead or {}).get(l) or ""
            first = probe.fill(old_lead or (keep[0] if keep else ""))
            if lead and re.match(r"(great|perfect|glad|badhiya|बढ़िया|sorry|i.?m sorry|you.?re right|माफ़|माफ|maaf|aap sahi|आप सही|that.?s okay|right|okay|ok|got it|ji|जी|achha|अच्छा|"
                                 r"hmm|हम्म|sure|of course|no problem|fair|ठीक|theek|lagta|लगता|sounds like)\b", first, re.I):
                lead = ""  # the line already opens with an acknowledgement: no stacked fillers
            if lead:
                leads[l] = (lead + " " + old_lead).strip()
                if l == lang:
                    used = key
        if not out and not leads:
            return plan
        if used:
            d["_style_used"] = used
        upd = {"fallback": {**plan.fallback, **out}, "decision": (str(plan.decision or "") + " +style:" + st.style)[:200]}
        if leads:
            upd["lead"] = {**(plan.lead or {}), **leads}
        return plan.model_copy(update=upd)

    # ------------------------------------------------------------------ length budget (owner, 15 Nov: "replies are far too long")

    _LEN_EXEMPT = frozenset({"opening", "pitch_demo", "opt_out", "opt_out_confirmed", "are_you_ai", "number_source", "identify", "human_agent",
                             "sensitive_stop", "constraint_stop", "repeat_last", "where_were_we", "language_switch", "link_sent",
                             "declined", "goodbye", "callback", "callback_time", "personal"})
    _LEN_TERMS = frozenset({"price", "confirm_close", "negotiation", "price_floor", "reoffer"})
    _NUM = re.compile(r"\d|₹|\$")
    _PRICE = re.compile(r"(₹|\$|rs\.?\s?)\s?\d[\d,.]*|\d[\d,]*\s*(rupees|rupaye|रुपये|रुपए)", re.I)

    def _length_cap(self, plan, customer_text: str, dialog: dict | None = None) -> dict | None:
        """The turn's budget: acks <= 6 words, an answer to a direct question <= 25 words, other turns <= 22 words and <= 2
        sentences (a 1-4 word filler such as "Koi baat nahi." does not count as one). Price / terms lines keep everything up to
        their last sentence with a number, then only the closing question. None = exempt (compliance wording, safety, replays,
        link / goodbye / callback lines, a sad caller's row). Off by default (owner, 15 Nov: the older length was right); ACIE_LENGTH_CAP=1 turns it on."""
        short = False
        if dialog is not None and os.getenv("ACIE_LENGTH_CAP", "0") != "1" and getattr(self.decision.domain, "name", "") == "product_sales":
            st = style_mod.Style.from_json(dialog.get(style_mod.STYLE_KEY))
            # a hurried / rude / sarcastic caller: short, always (not the "brief" of a confused / garbled line: that is about simple
            # words, and cutting it left "With a FitZone membership membership." on call c9960ed17d)
            short = st is not None and (st.style in ("calm", "dry-honest") or (st.style == "brief" and st.row == 8))
        if not short and (os.getenv("ACIE_LENGTH_CAP", "0") != "1" or getattr(self.decision.domain, "name", "") != "product_sales"):
            return None
        if plan.intent in self._LEN_EXEMPT or plan.intent.startswith(("safety", "distress", "concern")):
            return None
        dec = str(plan.decision or "")
        if dec.startswith(("style:backchannel", "fragment")):
            return {"w": 6, "s": 1, "terms": False}
        terms = plan.intent in self._LEN_TERMS or dec.startswith("negotiation")
        # with the owner's cap off, the short mode only drops whole extra sentences: it never cuts inside one (no fragments)
        return {"w": 25 if is_question(customer_text) else 22, "s": 2, "terms": terms, "whole": os.getenv("ACIE_LENGTH_CAP", "0") != "1"}

    def _fit_length(self, text: str, cap: dict, spent: dict) -> str:
        """Keep whole sentences while they fit; the first sentence of the turn is always spoken (cut at a clause boundary
        if it alone is too long, never inside a number). A sentence with a number (price, terms) is never dropped. The turn's
        closing question (the ask / permission question) is never dropped: its words are reserved first and the middle is trimmed."""
        parts = [x for x in re.split(r"(?<=[.!?।])\s+", (text or "").strip()) if x]
        if not cap["terms"] and cap["w"] > 6 and len(parts) > 1 and parts[-1].rstrip().endswith("?") and not cap.get("_ask") \
                and len(parts[-1].split()) <= 10 and not self._PRICE.search(parts[-1]):
            ask = parts[-1]
            aw = len(ask.split())
            inner = dict(cap, w=max(cap["w"] - aw, 8), s=max(cap["s"] - 1, 1), _ask=True)
            body = self._fit_length(" ".join(parts[:-1]), inner, spent)
            spent["w"] += aw
            spent["s"] += 1
            return (body + " " + ask).strip()
        out = []
        for part in [x for x in re.split(r"(?<=[.!?।])\s+", (text or "").strip()) if x]:
            w = len(part.split())
            num = bool(self._NUM.search(part)) if cap["terms"] else bool(self._PRICE.search(part))
            if cap["terms"]:
                if spent.get("num") and not num and not part.rstrip().endswith("?"):
                    continue  # nothing extra after the terms except the closing question
                out.append(part)
            elif num and not cap["w"] <= 6:
                if spent["w"] + w > cap["w"]:
                    part = self._price_cut(part)  # the price stays; a question or extra clause glued after it goes
                    w = len(part.split())
                out.append(part)  # a price said inside an answer stays
            elif spent["s"] == 0:  # the first sentence of content (after at most a short filler) is always spoken
                if spent["w"] + w > cap["w"] and not cap.get("whole"):
                    part = self._clause_cut(part, max(cap["w"] - spent["w"], 4))
                    if cap["w"] <= 6 and len(part.split()) > cap["w"] and not self._NUM.search(part):
                        part = " ".join(part.split()[:cap["w"]]).rstrip(",;:") + "."  # an ack: words only, no numbers to break
                    w = len(part.split())
                out.append(part)
            elif spent["s"] < cap["s"] and spent["w"] + w <= cap["w"]:
                out.append(part)
            elif spent["s"] < cap["s"] and cap["w"] - spent["w"] >= 4 and not cap.get("whole") and \
                    len((cut := self._clause_cut(part, cap["w"] - spent["w"])).split()) <= cap["w"] - spent["w"]:
                part, w = cut, len(cut.split())  # the main clause of the second sentence still fits
                out.append(part)
            else:
                continue
            spent["w"] += w
            if w > 4 or cap["w"] <= 6:
                spent["s"] += 1  # a short filler is not a sentence of content
            if cap["terms"] and num:
                spent["num"] = True
        return " ".join(out)

    def _price_cut(self, sent: str) -> str:
        """The sentence up to the first clause boundary after its last price ("₹6,000 hai, kya aapne ...?" -> "₹6,000 hai.")."""
        ms = list(self._PRICE.finditer(sent))
        if not ms:
            return sent
        m = re.compile(r"[,;:]\s|\s[-–]\s").search(sent, ms[-1].end())
        if not m:
            return sent
        pre = sent[:m.start()].rstrip()
        return pre + ("।" if sent.rstrip().endswith("।") else ".")

    @staticmethod
    def _clause_cut(sent: str, limit: int) -> str:
        """Longest clause prefix ("," ";" ":" " - " boundaries followed by a space) within `limit` words; the sentence as is if none."""
        best = ""
        if sent.rstrip().endswith("?"):
            return sent  # a question is asked whole or not at all
        for m in re.finditer(r"[,;:]\s|\s[-–]\s", sent):
            pre = sent[:m.start()].rstrip()
            if len(pre.split()) <= limit and len(pre.split()) >= min(4, max(2, limit // 2)) and not ACIEPipeline._fragment(pre):
                best = pre
        if not best:
            return sent
        end = "।" if re.search(r"[\u0900-\u097f]", best) and not re.search(r"[A-Za-z]{3,}\s*$", best) else "."
        return best + end

    _FRAG_START = re.compile(r"^\W*(with|for|if|when|whenever|because|since|as|while|although|though|unless|until|after|before|so that|like|plus|"
                             r"and|but|or|to|at|in|on|from|by|of|about|instead of|including|especially|which|who|that|where|jab|agar|kyunki|ki|"
                             r"अगर|जब|क्योंकि|कि|और|लेकिन)\b", re.I)
    _FRAG_END = re.compile(r"\b(a|an|the|and|or|but|with|for|to|of|in|on|at|your|my|our|their|is|are|ki|ka|ke|aur|का|की|के|और)\W*$", re.I)

    @classmethod
    def _fragment(cls, clause: str) -> bool:
        """A clause that cannot stand alone as a spoken sentence ("With a FitZone membership", "If you join this week")."""
        # (an opening phrase followed by its main clause - "At FitZone, you get yoga classes" - stands; the phrase alone does not)
        return bool((cls._FRAG_START.search(clause) and not re.search(r"[,;:]\s", clause)) or cls._FRAG_END.search(clause))

    def _keep_call_open(self, plan, expr, session: SessionState, ledger: SaidLedger, facts: CustomerFacts, text: str):
        """The call ends only on an explicit bye / opt-out / firm no (and the crisis and handoff plans). A link that was just sent, or a
        caller still mid-explanation, never ends it; the farewell text is replaced by something that continues the conversation."""
        if not plan.end_call and plan.intent == "link_sent" and pure_farewell(text) and os.getenv("ACIE_KEEP_OPEN", "1") != "0":
            return plan.model_copy(update={"end_call": True})  # "thanks, bye" right after the link: they are done
        if not plan.end_call or os.getenv("ACIE_KEEP_OPEN", "1") == "0" or getattr(self.decision.domain, "name", "") != "product_sales":
            return plan
        if plan.intent in ("opt_out", "opt_out_confirmed", "human_agent", "constraint_stop") or plan.intent.startswith(("safety", "distress")) or explicit_bye(text) \
                or (plan.intent in _FIRM_END and plan.decision != "psyche_move" and (plan.intent != "declined" or firm_no(text))) \
                or (plan.intent in _FIRM_END and (firm_no(text) or re.search(r"call|कॉल|फ़ोन|फोन|phone|बाद|later|कल|शाम|evening|tomorrow|baad|kal|subah|सुबह|morning", text, re.I))) \
                or _HANGUP_TXT.search(text):
            return plan  # "मुझे कुछ नहीं खरीदना, आप फ़ोन रख दीजिए। राम-राम।": they want the call over
        lang = expr.language
        cur = self._plan_text(plan, lang)
        upd: dict = {"end_call": False}
        if _FAREWELL.search(cur) or plan.intent in ("goodbye", "link_sent"):
            hold = session.dialog.get("sad_hold") == "1" or plan.intent == "declined" or ledger.refused
            texts = self._new_point(session, ledger, facts, text, "answer", hold)[0]
            upd.update({"intent": "answer" if plan.intent == "goodbye" else plan.intent, "llm_allowed": False, "lead": None, "facts": {},
                        "fallback": texts, "decision": "keep_call_open"})
        session.dialog["_kept_open"] = str(int(session.dialog.get("_kept_open", "0")) + 1)
        return plan.model_copy(update=upd)

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

    def _learn_said(self, ledger: SaidLedger, session: SessionState, plan, move, text: str, facts: CustomerFacts | None = None) -> None:
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
        if facts is not None:
            slots = [x for x in (session.dialog.pop("_cf_slot", ""),) + tuple(i for i in ledger.slots[-3:] if i.startswith(("benefit_", "offer", "faq_", "what"))) if x]
            facts.learn_agent(text, slots)
            facts.save(session.dialog)
        ledger.save(session.dialog)

    async def _regenerate(self, plan, expr, session, text: str, note: str) -> list[str] | None:
        """One more generation with a correction note; None if it also fails validation."""
        retry = self.compiler.compile(plan, expr, session, text)
        retry.messages[-1]["content"] += "\n\n" + note
        led = SaidLedger.load(session.dialog)
        v2 = StreamingValidator(plan, expr, trusted=False, customer_text=text,
                                known=(led, CustomerFacts.load(session.dialog)) if led.replies and not is_repeat_ask(text) else None)
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

        session = await self.restore_session(turn)
        SESSION_KEY.set(session.session_id)  # sticky A/B variant per session (acie/expression.py)
        if session.turn_index == 0 and hasattr(self.feedback, "recall"):
            await self.feedback.recall(session, turn.text)  # semantic memory (ACIE_MEMORY=1), else a no-op
        (intent, state), profile = await asyncio.gather(self._stream_a(turn, session), self.store.load(turn.customer_id))
        mark("understand")

        explicit = detect_explicit_requests(turn.text)
        if "language" in explicit and _ONE_LINE_DEMO.search(turn.text):
            # "just one line in Hindi" (call d3ac800811) is a demo request: one Hindi line, the call stays in the caller's language
            explicit.pop("language")
        if "language" in explicit and _ABOUT_IT.search(turn.text):
            # "Hindi mein bhi baat karta hai kya?" asks what the PRODUCT can do, not to switch language
            explicit.pop("language")
        if explicit.get("language") == "en" and _ASKS_HINDI.search(turn.text) and not _ASKS_ENGLISH.search(turn.text):
            explicit["language"] = "hi"  # "Hindi mein baat karo please, English mushkil lagti hai": the request is Hindi
        prev_expr = session.expression if session.turn_index > 0 else None
        expr = self.policy.decide(turn.text, state, profile, session, explicit, intent)
        # the emotion table (acie/psyche/emotion_policy.py): one deterministic row per turn from the 14 state dims.
        # The crisis row is always active (it is the old distress override, now a row); the others need ACIE_PSYCHE=1
        # crisis evidence: phonetic + 2-turn window + voice (acie/safety.distress_score), OR the NLU's crisis flag
        hist = [u.text for u in session.history if u.role == "customer"][-2:]
        d_score, d_ev = distress_score(turn.text, hist, getattr(turn, "voice", None)) if safety.enabled() else (0.0, [])
        nlu_crisis = session.dialog.pop("_nlu_crisis", None)
        if not safety.enabled():  # guardrail off (default): no crisis flag, no check-in, the call stays on the sale
            nlu_crisis = None
        # The NLU flag alone is never a crisis (gate run 4: "मेरे पास पहले से cloud है" and "मैं ठीक हूँ, बस fees पूछ रहा हूँ" ended calls).
        # Full distress needs lexical/phonetic evidence >= DISTRESS_T, or concern-band evidence plus the NLU flag. A caller who
        # reassures ("मैं ठीक हूँ") or asks about the product is released: back to normal, no sticky flag, no re-check.
        reassured = bool(_REASSURED.search(turn.text)) or is_question(turn.text)
        if session.dialog.get("safety_flag") and d_score < CONCERN_T and not nlu_crisis and _FALSE_ALARM.search(turn.text) \
                and float(session.dialog.get("_safety_trigger", "1") or 1) < 0.9 and session.dialog.get("_safety_turns", "1") == "1":
            # one borderline match (not clear ideation) and the caller says it was a normal question / they are fine: release, don't
            # keep reading the helpline (concert call: a metro-station question got Tele-MANAS three times). Clear ideation stays sticky.
            session.dialog.pop("safety_flag", None)
            session.dialog["_safety_released"] = "1"
        flagged = bool(session.dialog.get("safety_flag")) or d_score >= DISTRESS_T \
            or (bool(nlu_crisis) and d_score >= CONCERN_T and not reassured)
        nlu_only_concern = bool(nlu_crisis) and d_score < CONCERN_T and not reassured
        if d_ev or nlu_crisis:
            session.dialog["_distress"] = json.dumps({"score": round(d_score, 2), "evidence": d_ev[:4], "nlu": nlu_crisis})
        strat = emotion_policy.decide(state, turn.text, turn.voice, flagged=flagged, signals=session.dialog)
        session.dialog["_row"] = strat.row  # the domain reads it (a sad caller pauses the selling in both configs)
        if psyche.enabled() and strat.row != "crisis":
            expr = emotion_policy.apply_to_expression(expr, strat)
        mark("adapt")
        ledger = SaidLedger.load(session.dialog)
        facts = CustomerFacts.load(session.dialog)
        facts.use_ladder(getattr(self.decision.domain, "p", None))
        before_facts = {k: list(v) for k, v in facts.caller.items()}
        facts.learn_caller(turn.text, asked_goal=session.dialog.get("need_asked") == "1")
        self._track_need(session, turn.text)
        move = None
        style = self._decide_style(session, state, strat, turn.text, intent, facts, before_facts)

        undo = self._undo_point(session, facts)  # what the context was before this turn (barge-in: keep only what was heard)
        if strat.row == "crisis":  # sticky: once in crisis, never back to business
            if session.dialog.get("safety_flag"):
                session.dialog["_safety_turns"] = str(int(session.dialog.get("_safety_turns", "1") or 1) + 1)
                plan = distress_followup_plan(turn.text)
            else:  # never sell to someone in crisis
                plan = distress_plan()
                session.dialog["safety_flag"] = "1"
                session.dialog["_safety_trigger"] = f"{d_score:.2f}" if d_score else "1"
                session.dialog["_safety_turns"] = "1"
                session.dialog.pop("pending", None)
                profile.outcomes.append("safety_distress")
        elif (CONCERN_T <= d_score < DISTRESS_T or nlu_only_concern) and not session.dialog.get("safety_flag") \
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
            session.dialog["_where"] = session.context.where_we_are()  # the decision layer's view of where the call is
            if session.context.caller:
                from .caller_memory import about_block
                session.dialog["_caller"] = about_block(session.context.caller)
            recall = self._recall_plan(turn.text, session)
            if recall is not None:
                plan = recall  # answered from state: no domain step, no repeat/facts guard may swap it
            elif os.getenv("ACIE_FRAGMENT_GUARD", "1") != "0" and one_word_fragment(turn.text) and getattr(dom, "name", "") == "product_sales" \
                    and session.turn_index > 0:
                plan = self._fragment_plan(session, expr, ledger)  # "मुझे" / "mujhe": they are about to say it; no answer, no price
            elif style is not None and style.partial and getattr(dom, "name", "") == "product_sales":
                plan = self._backchannel_plan(session, expr, ledger)  # "So basically," / "I was thinking about...": let them finish
            elif style is not None and style.move == "own_and_hand_back" and getattr(dom, "name", "") == "product_sales" \
                    and not explicit_bye(turn.text) and not firm_no(turn.text):
                plan = self._handback_plan(session, expr, ledger)  # "you're being rude, did you even ask me?": own it, give control back
            else:
                plan = await self.decision.decide(intent, turn, session)
                plan = enrich_plan(plan, session, None, turn.text, getattr(self.decision.domain, "p", None))  # knowledge base + tree (ACIE_KB=1), else a no-op
                plan = self._no_repeat(plan, expr, session, ledger, state, intent)
                plan = self._facts_guard(plan, expr, session, ledger, facts, state, turn.text)
                plan = self._keep_call_open(plan, expr, session, ledger, facts, turn.text)
                plan = self._apply_style(plan, expr, session, style, turn.text)
        mark("decide")
        plan = self._purpose_first(plan, session, expr.language)
        # whole-call memory: what was promised / quoted / asked survives beyond the prompt window
        session.context.record_turn(session.turn_index, turn.text, plan.intent, is_question(turn.text),
                                    objection=session.dialog.pop("_objection", None), sinks=self.context_sinks)

        extra = {}
        if os.getenv("ACIE_MEMORY") == "1":  # semantic memory + profile reach the prompt only when memory is on
            extra = {"profile": profile, "memories": json.loads(session.dialog.get("_memories", "[]"))}
        prompt = self.compiler.compile(plan, expr, session, turn.text, **extra)
        known = (ledger, facts) if (plan.llm_allowed and not is_repeat_ask(turn.text) and ledger.replies
                                    and os.getenv("ACIE_FACTS_GUARD", "1") != "0") else None
        validator = StreamingValidator(plan, expr, trusted=not plan.llm_allowed, customer_text=turn.text, known=known)
        sentences: list[str] = []

        spoken_parts: list[str] = []  # every sentence already spoken this turn (lead, interim, reply)
        cap = self._length_cap(plan, turn.text, session.dialog)  # owner: phone-call length, enforced on every agent turn after wording
        spent = {"w": 0, "s": 0}

        async def emit(batch: list[str]) -> None:
            if cap is not None:
                batch = [x for x in (self._fit_length(b_, cap, spent) for b_ in batch) if x]
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
                                               f"customer asked, ONLY in {_LANG_NAME.get(expr.language, 'the reply language')} (even if they "
                                               "mixed languages), and use the {slots} for the offer price.")
                if again:
                    validator.add_issue(f"regenerated after: {why}", "regenerated")
                    validator.used_fallback = False
                    reply = again
            max_words = getattr(plan, "max_words", None) or self.max_reply_words
            if plan.llm_allowed and len(" ".join(reply).split()) > max_words:
                # the LLM ignores "one short sentence" (41-word answers on live call 8): keep the
                # leading sentences that fit; the ask usually comes last and is re-asked next turn
                kept, n = [], 0
                ask = reply[-1] if reply[-1].rstrip().endswith("?") else None  # the closing question is kept
                budget = max_words - (len(ask.split()) if ask else 0)
                for s in reply[:-1] if ask else reply:
                    if kept and n + len(s.split()) > budget:
                        break
                    kept.append(s)
                    n += len(s.split())
                if ask:
                    kept.append(ask)
                if len(kept) < len(reply):
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
                kept = self._drop_broken(reply)
                if kept != reply:
                    validator.add_issue("dropped a sentence with a template placeholder or an SMS claim", "broken_sentence")
                    reply = kept
                    if not reply:
                        validator.used_fallback = True
            if plan.llm_allowed and not validator.used_fallback and reply and plan.intent not in _ASKED_FOR:
                kept = self._drop_repeated_questions(reply, ledger, session)
                if kept != reply:
                    validator.add_issue("dropped a question the call already asked or answered", "question_repeated")
                    reply = kept
                kept = self._style_drop_recent_price(reply, plan, session, turn.text)
                if kept != reply:
                    validator.add_issue("style: dropped a price said in the last 2 turns", "style_price")
                    reply = kept
                kept = self._style_drop_asks(reply, plan, session)
                if kept != reply:
                    validator.add_issue("style: dropped a question (question budget)", "style_ask")
                    reply = kept
            if plan.llm_allowed and reply and os.getenv("ACIE_FACTS_GUARD", "1") != "0" and getattr(self.decision.domain, "name", "") == "product_sales" \
                    and plan.intent not in _ASKED_FOR:
                # the model's reply, or the safe template that replaces it, still repeats / restates / re-asks: regenerate once with the
                # known-block note, then the next unsaid point as a deterministic template (never the same words twice)
                spoken = validator.fill(render_fallback(plan, expr)) if validator.used_fallback else " ".join(reply)
                bad = self.reply_issues(spoken, ledger, facts, turn.text, None if validator.used_fallback else plan)
                if bad and not validator.used_fallback:
                    kinds = ", ".join(sorted({c for c, _ in bad}))
                    again = await self._regenerate(plan, expr, session, turn.text,
                                                   f"Your reply was rejected ({kinds}). It repeated something already said, said back what the caller told you, "
                                                   "or asked what they already answered. Say something NEW and relevant, in different words.")
                    if again and not self.reply_issues(" ".join(again), ledger, facts, turn.text, plan):
                        validator.add_issue(f"regenerated after: {kinds}", "regenerated")
                        reply, bad = again, []
                if bad:
                    validator.add_issue("replaced a repeating reply with the next new point", "repeat_replaced")
                    hold = session.dialog.get("sad_hold") == "1" or session.dialog.get("_row") in _NO_SELL_ROWS
                    texts, slot = self._new_point(session, ledger, facts, turn.text, plan.intent, hold)
                    session.dialog["_cf_slot"] = slot or ""
                    reply = [texts.get(expr.language) or texts["en"]]
                    validator.used_fallback = False
                    plan = plan.model_copy(update={"fallback": texts, "lead": None, "facts": {}})
            # any problem anywhere in the reply -> speak only the safe template
            out_reply = [validator.fill(render_fallback(plan, expr))] if validator.used_fallback else reply
            if out_reply and plan.llm_allowed and getattr(self.decision.domain, "name", "") == "product_sales" and not explicit_bye(turn.text):
                out_reply = [facts.rotate_opener(out_reply[0], expr.language)] + out_reply[1:]
                if bare_apology(out_reply) and not plan.end_call:
                    out_reply = out_reply + [way_forward(expr.language, self._agent_gender(plan) or "male")]
                    validator.add_issue("bare apology: added a way forward", "way_forward")
            await emit(out_reply)
        else:
            await emit(tail)
        mark("total")

        text = " ".join(sentences)
        if plan.llm_allowed and (plan.intent == "answer" and plan.lead or "offer" in plan.facts and plan.facts["offer"].value in text) \
                and "offer_price" not in session.context.quoted:
            session.context.quoted.append("offer_price")  # the price was told inside an answer: don't offer it again
        if plan.intent not in ("repeat_last", "where_were_we"):  # a replay says nothing new
            self._learn_said(ledger, session, plan, move, text, facts)
        style_mod.note_reply(session.dialog, text)
        if _RECORDED.search(text) and "call_recorded" not in session.context.told:
            session.context.told.append("call_recorded")
        p_ = getattr(self.decision.domain, "p", None)
        eco = getattr(p_, "economics", None)
        delta = session.context.after_reply(session.turn_index, turn.text, plan.intent, text, expr.language, session.dialog,
                                            float(getattr(eco, "list_price", 0) or 0), getattr(eco, "currency", "") or "")
        session.history += [Utterance(role="customer", text=turn.text), Utterance(role="agent", text=text)]
        undo.update(plan=plan, move=move, turn=session.turn_index, customer_text=turn.text, lang=expr.language, text=text)
        self._undo[session.session_id] = undo
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
        await self._persist(turn, session, plan, text, expr, delta, t)

        return TurnResult(text=text, plan=plan, state=state, expression=expr,
                          validation=validator.report(), timings_ms=t, result=session.context.result or None)
