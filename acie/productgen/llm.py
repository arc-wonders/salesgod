"""The LLM seam. `LLM.json(stage, payload)` returns a dict. The real client talks to any OpenAI-compatible endpoint
(default: the production Gemini settings ACIE_EXPR_*, overridable by PRODUCTGEN_*); tests pass a fake with canned JSON.
Keys are never logged or returned."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any, Protocol

import httpx

log = logging.getLogger("productgen.llm")


class LLM(Protocol):
    model: str

    async def json(self, stage: str, payload: dict) -> dict: ...


class LLMError(RuntimeError):
    pass


def _env(*names: str, default: str = "") -> str:
    for n in names:
        v = os.getenv(n)
        if v:
            return v
    return default


def configured() -> bool:
    return bool(_env("PRODUCTGEN_BASE_URL", "ACIE_EXPR_BASE_URL") and _env("PRODUCTGEN_MODEL", "ACIE_EXPR_MODEL"))


SYSTEM = """You write product knowledge and sales decision trees for a phone sales assistant that calls customers in India.
Output ONLY one JSON object, no prose, no markdown fences. Rules that always apply:
- Every customer-facing text is given in three languages: en (English), hi (Devanagari Hindi, natural spoken Hindi with common English words)
  and hinglish (Hindi in Roman script). Never leave a language empty and never put English in the hi field.
- Answers are 1-3 short spoken sentences. Never write a price as digits in an answer: use the slots {offer_price}, {original_price},
  {discount}, {offer_validity} or {price:OFFERING_ID}.
- Never promise health results or financial returns, never claim a guarantee, never create false urgency (no "only today", "last chance",
  "limited stock" unless the operator gave that exact offer validity), never run down a competitor, never invent certifications,
  awards, licences or lab results.
- Never invent people's names; name roles. Ids are lower-case kebab-case.
- MODE "synthetic": the operator owns the product; write specific, realistic, internally consistent detail (set provenance facts as invented demo data).
- MODE "public": the product belongs to a real company. Write ONLY facts that are publicly well known and certain; no prices, plans, limits,
  policies or numbers you are not sure of; if unsure, leave it out. Never invent offers. Fewer, safe entries are better than many guesses.
- The payload's `stage` field says what to return; follow its `shape` exactly."""

SHAPES: dict[str, str] = {
    "profile": '{"one_liner": str, "pitch": {en,hi,hinglish}, "benefits": {"en": [3-6 str], "hi": [same count], "hinglish": [same count]}, '
               '"faqs": [{"keywords": [any-script words a caller would say], "answer": {en,hi,hinglish}}] (8-12), "knowledge": "a factual background paragraph", '
               '"topic_words": [words in en/hi/hinglish that mean the caller talks about this product], "discovery_question": {en,hi,hinglish}, '
               '"pronunciations": {"Word": "spoken Devanagari respelling"}, "tone": str, "info_source": {en,hi,hinglish}, "call_to_action": str}',
    "kb_plan": '{"topics": [{"path": "a" or "a/b", "title": {en,hi,hinglish}, "overview": {en,hi,hinglish}, "interests": [tag ids]}] (15-30 top-level topics plus subtopics), '
               '"interests": [{"id", "label": {en,hi,hinglish}, "keywords": [what callers say, any script], "related": [ids], "discovery": {"en": [q], "hi": [q], "hinglish": [q]}}] (8-15), '
               '"offerings": [{"id", "name": {en,hi,hinglish}, "who_suits": [tag ids], "price_inr": int or null, "price_label": {en,hi,hinglish} or null, '
               '"pitch": {"en": [..], "hi": [..], "hinglish": [..]}, "upsell_from": [topic paths], "min_fit": 1.0}] (5-10)}',
    "kb_entries": '{"entries": [{"id": "kebab-id", "topic_path": "...", "question_variants": {"en": [2-3], "hi": [1-2], "hinglish": [1-2]}, "answer": {en,hi,hinglish}, '
                  '"facts": {}, "offerings": [offering ids], "interests": [tag ids], "teach": bool, "confidence": 0..1, "source": str}]}',
    "kb_fix": 'same as kb_entries; return a corrected version of every entry in `fix` (same id), applying the listed errors',
    "playbook": '{"nodes": [{"id", "title", "when": {"keywords": [short regex fragments in English, Roman Hindi AND Devanagari], "intents": [affirm|deny|tell_more|price|benefits|callback|clarify|goodbye|personal|small_talk], '
                '"objections": [trust|price|need|timing|language|other], "topics": [product|offer_terms|general|conversation], "min_buying"?, "max_buying"?, "min_frustration"?, "min_confusion"?}, '
                '"points": [3-5 str], "ask": str, "goal": str, "points_hi": [3-5 str], "ask_hi": str, "children": [same shape, optional]}] (14-20 situations)}',
    "ladder": '{"ladder": [{"id", "level": "goal|sub_goal|experience|schedule|preference|...", "question": {en,hi,hinglish}, "purpose": str, "interests": [tag ids it tells apart]}] '
              '(at least 4 rungs, ordered from broad goal to narrow preference), "fit_rules": [{"offering": id, "who_suits": [tag ids], "min_fit": 1.0-2.5}]}',
    "negotiation": '{"perks": [{"id", "text": {en,hi,hinglish}, "short": {en,hi,hinglish}}] (0-3 value perks that cost little), "price_steps": [{"id", "price": number}] (descending, each above the floor and below the list price)}',
    "expand_topic": '{"topics": [{"path", "title", "overview", "interests"}] (new subtopics under the given topic), "entries": [kb_entries shape] (6-10 per new subtopic), "interests": [optional new tags]}',
    "expand_node": '{"children": [playbook node shape] (2-5 more specific situations under the given node)}',
    "gap_entries": '{"entries": [kb_entries shape plus "cluster": the cluster id it answers; one or more per cluster], "topics": [new topics only if no existing topic fits], "nodes": [playbook nodes only for objections / situations]}',
}


def build_messages(stage: str, payload: dict) -> list[dict]:
    body = {"stage": stage, "shape": SHAPES.get(stage, "{}"), **payload}
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": json.dumps(body, ensure_ascii=False)}]


def parse_json(s: str) -> dict:
    s = (s or "").strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", s, re.S)
    if m:
        s = m.group(1).strip()
    a, b = s.find("{"), s.rfind("}")
    if a < 0 or b < a:
        raise ValueError("no JSON object in the reply")
    out = json.loads(s[a:b + 1])
    if not isinstance(out, dict):
        raise ValueError("the reply is not a JSON object")
    return out


class OpenAICompatLLM:
    """temperature 0, JSON mode, retries (network / 429 / 5xx / unparsable JSON) with backoff."""

    def __init__(self, base_url: str | None = None, model: str | None = None, api_key: str | None = None,
                 extra_body: dict | None = None, timeout_s: float | None = None, retries: int | None = None):
        self.base = (base_url or _env("PRODUCTGEN_BASE_URL", "ACIE_EXPR_BASE_URL")).rstrip("/")
        self.model = model or _env("PRODUCTGEN_MODEL", "ACIE_EXPR_MODEL")
        self._key = api_key if api_key is not None else _env("PRODUCTGEN_API_KEY", "ACIE_EXPR_API_KEY", default="EMPTY")
        raw = _env("PRODUCTGEN_EXTRA_BODY", "ACIE_EXPR_EXTRA_BODY", default="{}")
        self.extra = extra_body if extra_body is not None else json.loads(raw or "{}")
        self.timeout = timeout_s or float(_env("PRODUCTGEN_TIMEOUT_S", default="300"))
        self.retries = retries if retries is not None else int(_env("PRODUCTGEN_RETRIES", default="3"))
        self.max_tokens = int(_env("PRODUCTGEN_MAX_TOKENS", default="16000"))
        self.json_mode = _env("PRODUCTGEN_JSON_MODE", default="1") != "0"
        if not self.base or not self.model:
            raise LLMError("set PRODUCTGEN_BASE_URL and PRODUCTGEN_MODEL (or ACIE_EXPR_BASE_URL and ACIE_EXPR_MODEL)")

    def __repr__(self) -> str:  # never show the key
        return f"OpenAICompatLLM(model={self.model!r})"

    async def json(self, stage: str, payload: dict) -> dict:
        body: dict[str, Any] = {"model": self.model, "temperature": 0, "max_tokens": self.max_tokens, "seed": 7,
                                "messages": build_messages(stage, payload)}
        if self.json_mode:
            body["response_format"] = {"type": "json_object"}
        body.update(self.extra)
        last = ""
        for attempt in range(self.retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as c:
                    r = await c.post(f"{self.base}/chat/completions", json=body, headers={"Authorization": f"Bearer {self._key}"})
                if r.status_code in (429, 500, 502, 503, 504):
                    last = f"HTTP {r.status_code}"
                else:
                    r.raise_for_status()
                    return parse_json(r.json()["choices"][0]["message"]["content"])
            except (httpx.TransportError, ValueError, KeyError) as e:  # no key in these messages
                last = f"{type(e).__name__}: {str(e)[:120]}"
            except httpx.HTTPStatusError as e:
                raise LLMError(f"{stage}: HTTP {e.response.status_code}") from None
            log.warning("productgen %s attempt %d failed: %s", stage, attempt + 1, last)
            await asyncio.sleep(min(2 ** attempt, 8))
        raise LLMError(f"{stage}: gave up after {self.retries + 1} tries ({last})")


def default_llm() -> LLM:
    return OpenAICompatLLM()
