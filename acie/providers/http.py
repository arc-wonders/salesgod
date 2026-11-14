"""Any service that returns the signals as JSON: ACIE_SIGNAL_HTTP_URL (+ ACIE_SIGNAL_HTTP_TOKEN, ACIE_SIGNAL_HTTP_TIMEOUT_S).

Request:  POST {"text", "session_id", "history": [[role, text], ...], "intents": {label: meaning}, "about",
                "rule_intent", "pending"}
Response: a TurnSignals-shaped JSON object (or {"signals": {...}}); every field is optional, `buying_signal`
          is accepted for `buying_interest`. Labels are mapped onto the domain's vocabulary by the fusion."""

from __future__ import annotations

import logging
import os

import httpx

from ..schemas import SessionState, TurnInput
from ..signals import PartialSignals
from .base import ProviderBase, SignalGuide

log = logging.getLogger("acie.providers")


class HttpProvider(ProviderBase):
    name = "http"

    def __init__(self, url: str | None = None, timeout_s: float | None = None, token: str | None = None, transport=None):
        self.url = url or os.getenv("ACIE_SIGNAL_HTTP_URL", "")
        self.timeout_s = timeout_s or float(os.getenv("ACIE_SIGNAL_HTTP_TIMEOUT_S", "1.5"))
        self.budget_s = self.timeout_s + 0.5
        token = token if token is not None else os.getenv("ACIE_SIGNAL_HTTP_TOKEN", "")
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}
        self.client = httpx.AsyncClient(timeout=self.timeout_s, transport=transport)

    async def provide(self, turn: TurnInput, session: SessionState, guide: SignalGuide) -> PartialSignals | None:
        body = {"text": turn.text, "session_id": turn.session_id, "history": [list(h) for h in guide.history[-6:]],
                "intents": guide.intents, "about": guide.about, "rule_intent": guide.rule_intent,
                "pending": session.dialog.get("pending")}
        try:
            r = await self.client.post(self.url, json=body, headers=self.headers)
            r.raise_for_status()
            data = r.json()
            data = data.get("signals", data) if isinstance(data, dict) else {}
            part = PartialSignals.model_validate({**data, "provider": "http"})
            if part.intent is None and part.intent_scores:  # a bare distribution: take its top label
                part.intent = max(sorted(part.intent_scores), key=part.intent_scores.get)
            if part.confidence is None and part.intent in part.intent_scores:
                part.confidence = part.intent_scores[part.intent]
            return part
        except Exception as e:
            log.info("signal service unavailable: %s", type(e).__name__)
            return None
