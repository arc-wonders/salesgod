"""POST /api/ai/assist backend: ask the LLM for a DRAFT of one object kind, validate it against that kind's pydantic model,
repair once on failure. Never saves anything. The LLM is injected (`json(stage, payload) -> dict`), so tests use a fake;
the real one is acie.productgen.llm.OpenAICompatLLM (temperature 0, JSON mode, no `seed`, key never logged)."""

from __future__ import annotations

import json
from typing import Protocol

import httpx
from pydantic import BaseModel, ValidationError

from .models import ASSIST_MODELS

SYSTEM = """You help an operator configure a phone sales/CRM tool used in India. Reply with ONE JSON object only, matching the given
JSON schema, no prose and no markdown. Customer-facing text per language uses keys en, hi (Devanagari), hinglish (Roman Hindi).
Never invent prices, guarantees or false urgency; never create ways to call people who opted out. Ids are lower-case kebab-case."""


class LLM(Protocol):
    async def json(self, stage: str, payload: dict) -> dict: ...


class AssistError(Exception):
    def __init__(self, status: int, msg: str):
        super().__init__(msg)
        self.status, self.msg = status, msg


def _errors(e: ValidationError) -> list[str]:
    return [f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}" for err in e.errors()[:8]]


async def assist(llm: LLM, kind: str, prompt: str, context: dict | None = None) -> dict:
    model: type[BaseModel] | None = ASSIST_MODELS.get(kind)
    if model is None:
        raise AssistError(400, f"kind must be one of {sorted(ASSIST_MODELS)}")
    prompt = (prompt or "").strip()
    if not prompt:
        raise AssistError(400, "write what you want in the prompt")
    payload = {"kind": kind, "instruction": prompt[:2000], "context": context or {}, "schema": model.model_json_schema(), "system": SYSTEM}
    problems: list[str] = []
    raw: dict = {}
    for attempt in (0, 1):  # one repair round
        if attempt:
            payload = {**payload, "previous_draft": raw, "fix": problems}
        try:
            raw = await llm.json(f"assist_{kind}", payload)
        except Exception as e:  # noqa: BLE001  (messages never contain keys: the client scrubs them)
            raise AssistError(502, f"the AI service failed: {type(e).__name__}: {str(e)[:200]}")
        try:
            return {"kind": kind, "draft": model.model_validate(raw).model_dump(), "repaired": bool(attempt), "saved": False}
        except ValidationError as e:
            problems = _errors(e)
    raise AssistError(422, "the AI draft did not fit the form even after a retry: " + "; ".join(problems))


class AssistLLM:
    """Chat-completions client for ACIE_EXPR_* / PRODUCTGEN_* (see acie/productgen/llm.py): temperature 0, JSON mode, no seed."""

    def __init__(self):
        from acie.productgen.llm import OpenAICompatLLM
        self._c = OpenAICompatLLM(retries=1, timeout_s=60)
        self._c.seed = None
        self.model = self._c.model

    async def json(self, stage: str, payload: dict) -> dict:
        from acie.productgen.llm import parse_json
        system = payload.get("system", SYSTEM)
        user = {k: v for k, v in payload.items() if k != "system"}
        body = {"model": self._c.model, "temperature": 0, "max_tokens": 4000, "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": json.dumps(user, ensure_ascii=False)}]}
        body.update(self._c.extra)
        body.pop("seed", None)
        async with httpx.AsyncClient(timeout=60) as c:
            r = await c.post(f"{self._c.base}/chat/completions", json=body, headers={"Authorization": f"Bearer {self._c._key}"})
        if r.status_code >= 300:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:160].replace(self._c._key, '<key>') if self._c._key else r.text[:160]}")
        return parse_json(r.json()["choices"][0]["message"]["content"])
