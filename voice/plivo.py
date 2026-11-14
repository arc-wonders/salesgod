"""Plivo outbound calls + audio streaming (pattern from uno_pipeline/backend/app/telephony/plivo.py).

  outbound   POST {Plivo}/Call/  (from PLIVO_NUMBER, answer_url, hangup_url)
  answer     Plivo -> POST /plivo/answer   (signed)  -> <Stream bidirectional> XML
  audio      Plivo -> WS   /plivo/stream/{token}
  hangup     Plivo -> POST /plivo/hangup   (signed)

The answer URL is passed per call, so the Plivo number's own application (used by other
systems for inbound calls) is never modified.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
from urllib.parse import parse_qs, urlsplit, urlunsplit
from xml.sax.saxutils import escape

import httpx

log = logging.getLogger("voice.plivo")
API = "https://api.plivo.com/v1/Account/{}/"


def e164(raw: str) -> str | None:
    raw = (raw or "").strip()
    digits = "".join(c for c in raw if c.isdigit())
    if raw.startswith("+") and 8 <= len(digits) <= 15:
        return "+" + digits
    if len(digits) == 10 and digits[0] in "6789":  # Indian mobile without country code
        return "+91" + digits
    if len(digits) == 12 and digits.startswith("91"):
        return "+" + digits
    return None


def _sorted_query(params: dict[str, list[str]]) -> str:
    return "&".join("&".join(f"{k}={v}" for v in sorted(params[k])) for k in sorted(params))


def signature_v3(auth_token: str, method: str, url: str, params: dict[str, list[str]], nonce: str) -> str:
    parts = urlsplit(url)
    base = urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
    query = parse_qs(parts.query, keep_blank_values=True)
    if method.upper() == "GET":
        query = {**params, **query}
        signed = base + ("?" + _sorted_query(query) if query else "")
    else:
        q = _sorted_query(query)
        signed = base + ("?" + q if q or params else "") + ("." if q and params else "")
        signed += "".join("".join(f"{k}{v}" for v in sorted(params[k])) for k in sorted(params))
    digest = hmac.new(auth_token.encode(), f"{signed}.{nonce}".encode(), hashlib.sha256).digest()
    return base64.b64encode(digest).decode()


class Plivo:
    def __init__(self, auth_id: str, auth_token: str, number: str, public_url: str, verify: bool = True):
        self.auth_id, self.auth_token, self.number = auth_id, auth_token, number
        self.public_url = public_url.rstrip("/")
        self.verify_signatures = verify
        self.client = httpx.AsyncClient(base_url=API.format(auth_id), auth=(auth_id, auth_token), timeout=15.0)

    def valid(self, url: str, params: dict[str, list[str]], headers) -> bool:
        if not self.verify_signatures:
            return True
        nonce, received = headers.get("x-plivo-signature-v3-nonce"), headers.get("x-plivo-signature-v3")
        if not nonce or not received:
            return False
        expected = signature_v3(self.auth_token, "POST", url, params, nonce)
        return any(hmac.compare_digest(expected, s.strip()) for s in received.split(","))

    async def dial(self, call) -> None:
        body = {"from": self.number.lstrip("+"), "to": call.to.lstrip("+"),
                "answer_url": f"{self.public_url}/plivo/answer", "answer_method": "POST",
                "hangup_url": f"{self.public_url}/plivo/hangup", "hangup_method": "POST",
                "ring_timeout": 45, "time_limit": 420,
                # asynchronous answering-machine detection (voice_platform F6): the agent starts
                # talking at once; /plivo/machine tells us if a machine picked up
                "machine_detection": "true", "machine_detection_time": 5000,
                "machine_detection_url": f"{self.public_url}/plivo/machine", "machine_detection_method": "POST"}
        r = await self.client.post("Call/", json=body)
        if r.status_code >= 300:
            raise RuntimeError(f"Plivo refused the call ({r.status_code}): {r.text[:300]}")
        call.request_uuid = r.json().get("request_uuid")
        call.status = "ringing"

    async def hangup(self, call) -> None:
        uuid = call.call_uuid
        try:
            if uuid:
                await self.client.delete(f"Call/{uuid}/")
            elif call.request_uuid:
                await self.client.delete(f"Request/{call.request_uuid}/")
        except httpx.HTTPError as e:
            log.warning("hangup failed: %s", e)

    async def send_sms(self, to: str, text: str) -> dict:
        """POST Message/ from the call's line to the customer's own number only (voice_platform F4)."""
        try:
            r = await self.client.post("Message/", json={"src": self.number.lstrip("+"), "dst": to.lstrip("+"), "text": text})
        except httpx.HTTPError as e:
            return {"status": "failed", "reason": f"{type(e).__name__}"}
        if r.status_code >= 300:
            return {"status": "failed", "reason": r.text[:200]}
        return {"status": "sent", "message_id": (r.json().get("message_uuid") or [None])[0]}

    def stream_xml(self, token: str) -> str:
        ws = self.public_url.replace("https://", "wss://", 1).replace("http://", "ws://", 1)
        return ('<?xml version="1.0" encoding="UTF-8"?><Response>'
                '<Stream bidirectional="true" keepCallAlive="true" contentType="audio/x-mulaw;rate=8000">'
                f'{escape(ws)}/plivo/stream/{token}</Stream></Response>')

    @staticmethod
    def hangup_xml() -> str:
        return '<?xml version="1.0" encoding="UTF-8"?><Response><Hangup/></Response>'
