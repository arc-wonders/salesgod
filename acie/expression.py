"""Layer 6 - LLM Expression Engine.

Backends stream text deltas. `OpenAICompatibleBackend` talks to vLLM (our
fine-tuned model on the 4xA100 box) or any OpenAI-compatible server.
`TemplateBackend` renders the plan's fallback template: zero latency, always
correct, used offline, in tests, and as the validator's safety net.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from typing import Protocol

import httpx

from .schemas import CompiledPrompt, ExpressionParams, TurnPlan


class ExpressionBackend(Protocol):
    def stream(self, prompt: CompiledPrompt, plan: TurnPlan, expr: ExpressionParams) -> AsyncIterator[str]: ...


def render_fallback(plan: TurnPlan, expr: ExpressionParams) -> str:
    """Template text with {slots} still in place (validator fills them)."""
    return plan.fallback.get(expr.language) or plan.fallback.get("en") or plan.objective


class TemplateBackend:
    async def stream(self, prompt: CompiledPrompt, plan: TurnPlan, expr: ExpressionParams) -> AsyncIterator[str]:
        yield render_fallback(plan, expr)


class OpenAICompatibleBackend:
    def __init__(self, base_url: str | None = None, model: str | None = None, api_key: str | None = None,
                 temperature: float = 0.3, max_tokens: int | None = None, timeout_s: float = 5.0):
        # ACIE_EXPR_* point only the reply writer at another model (e.g. Gemini through its OpenAI-compatible
        # endpoint) while the understanding layer stays on the local Qwen (ACIE_LLM_*, guided JSON, fast)
        env = lambda k, d: os.getenv(f"ACIE_EXPR_{k}") or os.getenv(f"ACIE_LLM_{k}", d)
        self.base_url = (base_url or env("BASE_URL", "http://localhost:8000/v1")).rstrip("/")
        self.model = model or env("MODEL", "acie-expression")
        self.api_key = api_key or env("API_KEY", "EMPTY")
        self.temperature = temperature
        self.max_tokens = max_tokens or int(env("MAX_TOKENS", "320"))  # Devanagari is token-heavy
        # e.g. '{"chat_template_kwargs": {"enable_thinking": false}}' for Qwen3 on vLLM,
        # '{"reasoning_effort": "low"}' for Gemini
        self.extra_body = json.loads(os.getenv("ACIE_EXPR_EXTRA_BODY") if os.getenv("ACIE_EXPR_BASE_URL")
                                     else os.getenv("ACIE_LLM_EXTRA_BODY", "{}") or "{}")
        self.client = httpx.AsyncClient(timeout=timeout_s)

    async def stream(self, prompt: CompiledPrompt, plan: TurnPlan, expr: ExpressionParams) -> AsyncIterator[str]:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": prompt.system}, *prompt.messages],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
            **self.extra_body,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}
        async with self.client.stream("POST", f"{self.base_url}/chat/completions", json=body, headers=headers) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:].strip()
                if data == "[DONE]":
                    break
                choices = json.loads(data).get("choices") or [{}]
                delta = choices[0].get("delta", {}).get("content")
                if delta:
                    yield delta

    async def aclose(self) -> None:
        await self.client.aclose()
