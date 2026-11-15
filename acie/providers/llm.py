"""LLM NLU as a provider. Works with any model behind an OpenAI-compatible endpoint (Qwen, Gemini, GPT, Claude ...):
the model and URL come from ACIE_LLM_BASE_URL / ACIE_LLM_MODEL (see acie/understanding.py); nothing here names one."""

from __future__ import annotations

from ..schemas import SessionState, TurnInput
from ..signals import GARBLED_GIST, STATE_DIMS, PartialSignals
from .base import ProviderBase, SignalGuide


class LLMProvider(ProviderBase):
    name = "llm"
    needs_guide = True
    skip_when_rules_sure = True   # the ~0.6 s call is skipped when the rules are sure of a short utterance

    def __init__(self, nlu):
        self.nlu = nlu            # any object with: async understand(text, history, intents, about) -> Understanding | None
        self.budget_s = float(getattr(nlu, "timeout_s", 2.5)) + 0.5

    async def provide(self, turn: TurnInput, session: SessionState, guide: SignalGuide) -> PartialSignals | None:
        u = await self.nlu.understand(turn.text, guide.history, guide.intents, guide.about)
        if u is not None and getattr(u, "crisis", False):
            # safety second net (acie/safety.crisis_from_nlu): may only ADD a crisis flag, never suppress one
            session.dialog["_nlu_crisis"] = (getattr(u, "crisis_reason", "") or "1")[:120]
        if u is None:
            return None
        soc = {k: float(v) for k in ("sarcasm", "humour", "hostility", "interest", "impatience", "scepticism")
               if (v := getattr(u, k, None)) is not None}
        if soc:  # social signals (acie/state/social.py): fused with the lexicon in the pipeline, read by the style policy
            import json as _json
            session.dialog["_nlu_social"] = _json.dumps(soc, sort_keys=True)
        gist = getattr(u, "gist", "") or ""
        state = {d: float(getattr(u, d)) for d in STATE_DIMS[:4]}
        for d in STATE_DIMS[4:]:
            v = getattr(u, d, None)
            if v is not None:
                state[d] = float(v)
        intent = getattr(u, "intent", "unknown") or "unknown"
        garbled = bool(GARBLED_GIST.search(gist))
        buying = float(u.buying_signal)
        return PartialSignals(provider="llm", intent=intent, intent_scores={intent: 1.0} if intent != "unknown" else {},
                              objection=u.objection, question_topic=u.question_topic, buying_interest=buying,
                              garbled=garbled, garbled_reason=gist if garbled else "", gist=gist, state=state,
                              evidence=[f"nlu: {gist}" if gist else "nlu", f"buying {buying:.2f}"])
