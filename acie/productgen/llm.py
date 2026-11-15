"""The LLM seam. `LLM.json(stage, payload)` returns a dict. The real client talks to any OpenAI-compatible endpoint
(PRODUCTGEN_* first, then Gemini (acie.expression.gemini_settings), then ACIE_EXPR_*: the reply model may be a small local
model with a 4k context, which cannot write a product); tests pass a fake with canned JSON.
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


def _settings() -> dict:
    """base/model/key/extra from PRODUCTGEN_*, else Gemini, else ACIE_EXPR_*. ACIE_EXPR_EXTRA_BODY is only used with ACIE_EXPR_*'s
    own endpoint (e.g. Qwen's chat_template_kwargs would be rejected by Gemini)."""
    from acie.expression import gemini_settings
    if os.getenv("PRODUCTGEN_BASE_URL"):
        base, model, key, extra = os.getenv("PRODUCTGEN_BASE_URL", ""), _env("PRODUCTGEN_MODEL", "ACIE_EXPR_MODEL"), \
            _env("PRODUCTGEN_API_KEY", "ACIE_EXPR_API_KEY", default="EMPTY"), _env("PRODUCTGEN_EXTRA_BODY", default="{}")
    elif (g := gemini_settings()):
        base, model, key = g["base_url"], _env("PRODUCTGEN_MODEL", default=g["model"]), _env("PRODUCTGEN_API_KEY", default=g["api_key"])
        extra = _env("PRODUCTGEN_EXTRA_BODY", default="{}")
    else:
        base, model = _env("ACIE_EXPR_BASE_URL"), _env("PRODUCTGEN_MODEL", "ACIE_EXPR_MODEL")
        key = _env("PRODUCTGEN_API_KEY", "ACIE_EXPR_API_KEY", default="EMPTY")
        extra = _env("PRODUCTGEN_EXTRA_BODY", "ACIE_EXPR_EXTRA_BODY", default="{}")
    return {"base": base, "model": model, "key": key, "extra": extra}


def configured() -> bool:
    s = _settings()
    return bool(s["base"] and s["model"])


SYSTEM = """You write product knowledge and sales decision trees for a phone sales assistant that calls customers in India.
Output ONLY one JSON object, no prose, no markdown fences. Rules that always apply:
- Every customer-facing text is given in three languages: en (English), hi (Devanagari Hindi, natural spoken Hindi with common English words)
  and hinglish (Hindi in Roman script). Never leave a language empty and never put English in the hi field.
- Everything is SPOKEN on a phone call, so keep it short like a person talking: an answer is ONE sentence of at most 22 words (two
  very short ones at most), a pitch at most 25 words, a benefit a short phrase of at most 10 words, a question at most 15 words.
  Hindi / Hinglish versions say the same thing just as briefly. Never write a price as digits in an answer: use the slots {offer_price}, {original_price},
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
    "profile": '{"one_liner": str, "pitch": {en,hi,hinglish} (one or two short sentences, at most 25 words), "benefits": {"en": [3-6 short phrases of at most 10 words, lower-case start, no full stop, e.g. "a free personal trainer session every week"], "hi": [same count], "hinglish": [same count]}, '
               '"faqs": [{"keywords": [any-script words a caller would say], "answer": {en,hi,hinglish} (one sentence, at most 22 words)}] (8-12; only questions the given facts can answer: no invented venues, dates or times), "knowledge": "a factual background paragraph", '
               '"topic_words": [words in en/hi/hinglish that mean the caller talks about this product], "discovery_question": {en,hi,hinglish} (at most 15 words), '
               '"pronunciations": {"Word": "spoken Devanagari respelling"}, "tone": str, "info_source": {en,hi,hinglish}, "call_to_action": str (a short phrase that follows "Shall I ...", e.g. "have our team send you the joining link")}',
    "kb_plan": '{"topics": [{"path": "a" or "a/b", "title": {en,hi,hinglish}, "overview": {en,hi,hinglish}, "interests": [tag ids]}] (5-20 top-level topics, 15-40 topics in all with subtopics; for a focused product fewer top-level topics with subtopics under them is right), '
               '"interests": [{"id", "label": {en,hi,hinglish}, "keywords": [what callers say, any script], "related": [ids], "discovery": {"en": [q], "hi": [q], "hinglish": [q]}}] (8-15), '
               '"offerings": [{"id", "name": {en,hi,hinglish}, "who_suits": [tag ids], "price_inr": int or null, "price_label": {en,hi,hinglish} or null, '
               '"pitch": {"en": [..], "hi": [..], "hinglish": [..]}, "upsell_from": [topic paths], "min_fit": 1.0}] (5-10)}',
    "kb_entries": '{"entries": [{"id": "kebab-id", "topic_path": "...", "question_variants": {"en": [2-3], "hi": [1-2], "hinglish": [1-2]}, "answer": {en,hi,hinglish} (one spoken sentence, at most 22 words), '
                  '"facts": {}, "offerings": [offering ids], "interests": [tag ids], "teach": bool, "confidence": 0..1, "source": str}]}',
    "kb_fix": '{"entries": [kb_entries entry shape]}: ONE object whose "entries" list holds a corrected version of every entry in `fix` '
              '(same id), applying the listed errors',
    "playbook": '{"nodes": [{"id", "title", "when": {"keywords": [short regex fragments in English, Roman Hindi AND Devanagari], "intents": [affirm|deny|tell_more|price|benefits|callback|clarify|goodbye|personal|small_talk], '
                '"objections": [trust|price|need|timing|language|other], "topics": [product|offer_terms|general|conversation], "min_buying"?, "max_buying"?, "min_frustration"?, "min_confusion"?}, '
                '"points": [3-5 short sentences, at most 12 words each], "ask": str (at most 15 words), "goal": str, "points_hi": [3-5 short sentences], "ask_hi": str, "children": [same shape, optional]}] (14-20 situations)}',
    "ladder": '{"ladder": [{"id", "level": "goal|sub_goal|experience|schedule|preference|...", "question": {en,hi,hinglish} (at most 15 words), "purpose": str, "interests": [tag ids it tells apart]}] '
              '(at least 4 rungs, ordered from broad goal to narrow preference), "fit_rules": [{"offering": id, "who_suits": [tag ids], "min_fit": 1.0-2.5}]}',
    "negotiation": '{"perks": [{"id", "text": {en,hi,hinglish}, "short": {en,hi,hinglish}}] (0-3 value perks that cost little), "price_steps": [{"id", "price": number}] (descending, each above the floor and below the list price)}',
    "expand_topic": '{"topics": [{"path", "title", "overview", "interests"}] (new subtopics under the given topic), "entries": [kb_entries shape] (6-10 per new subtopic), "interests": [optional new tags]}',
    "expand_node": '{"children": [playbook node shape] (2-5 more specific situations under the given node)}',
    "shorten": '{"items": [{"key": same key, "text": the shortened text}]}: rewrite every item in `items` to at most its max_words words, '
               'and its `sentences` count, same language and script, same meaning, keep every number and {slot} it needs, add NO new fact, '
               'keep names and acronyms as written (VIP, Fan Pit); kind benefit / cta = a short phrase (no full stop), otherwise '
               'short spoken sentences like a person on the phone',
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
    try:
        out = json.loads(s[a:b + 1])
    except json.JSONDecodeError as e:
        if not e.msg.startswith("Extra data"):
            raise
        out = _merge_objects(s[a:b + 1])  # the model sometimes writes several objects in a row ("{..}\n{..}")
    if not isinstance(out, dict):
        raise ValueError("the reply is not a JSON object")
    return out


def _merge_objects(s: str) -> dict:
    """Several top-level JSON objects -> one: bare entries ({"id", "answer", ...}) become {"entries": [...]}; otherwise list fields
    are concatenated and the first value of any other field wins."""
    dec, objs, i = json.JSONDecoder(), [], 0
    while i < len(s):
        j = s.find("{", i)
        if j < 0:
            break
        o, i = dec.raw_decode(s, j)
        if isinstance(o, dict):
            objs.append(o)
    if objs and all("id" in o and "answer" in o for o in objs):
        return {"entries": objs}
    out: dict = {}
    for o in objs:
        for k, v in o.items():
            if isinstance(v, list) and isinstance(out.get(k), list):
                out[k] = out[k] + v
            else:
                out.setdefault(k, v)
    return out


class OpenAICompatLLM:
    """temperature 0, JSON mode, retries (network / 429 / 5xx / unparsable JSON) with backoff."""

    def __init__(self, base_url: str | None = None, model: str | None = None, api_key: str | None = None,
                 extra_body: dict | None = None, timeout_s: float | None = None, retries: int | None = None):
        st = _settings()
        self.base = (base_url or st["base"]).rstrip("/")
        self.model = model or st["model"]
        self._key = api_key if api_key is not None else st["key"]
        raw = st["extra"]
        self.extra = extra_body if extra_body is not None else json.loads(raw or "{}")
        self.timeout = timeout_s or float(_env("PRODUCTGEN_TIMEOUT_S", default="300"))
        self.retries = retries if retries is not None else int(_env("PRODUCTGEN_RETRIES", default="3"))
        self.max_tokens = int(_env("PRODUCTGEN_MAX_TOKENS", default="16000"))
        self.json_mode = _env("PRODUCTGEN_JSON_MODE", default="1") != "0"
        seed = _env("PRODUCTGEN_SEED", default="")
        self.seed = int(seed) if seed.strip() else None
        if not self.base or not self.model:
            raise LLMError("set PRODUCTGEN_BASE_URL and PRODUCTGEN_MODEL (or ACIE_EXPR_BASE_URL and ACIE_EXPR_MODEL)")

    def __repr__(self) -> str:  # never show the key
        return f"OpenAICompatLLM(model={self.model!r})"

    async def json(self, stage: str, payload: dict) -> dict:
        body: dict[str, Any] = {"model": self.model, "temperature": 0, "max_tokens": self.max_tokens,
                                "messages": build_messages(stage, payload)}
        if self.seed is not None:  # Gemini's OpenAI endpoint rejects "seed" (HTTP 400): only sent when PRODUCTGEN_SEED is set
            body["seed"] = self.seed
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
            except httpx.HTTPStatusError as e:  # include the provider's message (never contains the key; scrubbed anyway)
                msg = e.response.text[:200].replace(self._key, "<key>") if self._key else e.response.text[:200]
                raise LLMError(f"{stage}: HTTP {e.response.status_code}: {msg}") from None
            log.warning("productgen %s attempt %d failed: %s", stage, attempt + 1, last)
            await asyncio.sleep(min(2 ** attempt, 8))
        raise LLMError(f"{stage}: gave up after {self.retries + 1} tries ({last})")


def default_llm() -> LLM:
    return OpenAICompatLLM()
