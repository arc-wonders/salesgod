"""Layer 6 - LLM Expression Engine.

Backends stream text deltas. `OpenAICompatibleBackend` talks to vLLM (our
fine-tuned model on the 4xA100 box) or any OpenAI-compatible server.
`TemplateBackend` renders the plan's fallback template: zero latency, always
correct, used offline, in tests, and as the validator's safety net.
"""

from __future__ import annotations

import asyncio
import contextvars
import hashlib
import json
import os
import random
import time
from collections import defaultdict
from collections.abc import AsyncIterator
from typing import Protocol

import httpx

from .schemas import CompiledPrompt, ExpressionParams, TurnPlan


# set by the pipeline per turn: keeps an A/B variant sticky for the whole call
SESSION_KEY: contextvars.ContextVar[str | None] = contextvars.ContextVar("acie_session_key", default=None)


class FirstTokenTimeout(TimeoutError):
    pass


class ExpressionMetrics:
    """Per-backend/variant counters and latencies (ms). Cheap, in-process; read with snapshot()."""

    KEEP = 500

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.turns: dict[str, int] = defaultdict(int)
        self.errors: dict[str, int] = defaultdict(int)
        self.timeouts: dict[str, int] = defaultdict(int)
        self.failovers = 0
        self.first_token_ms: dict[str, list[float]] = defaultdict(list)
        self.total_ms: dict[str, list[float]] = defaultdict(list)

    def record(self, label: str, first_token_ms: float | None, total_ms: float, error: bool = False,
               timeout: bool = False) -> None:
        self.turns[label] += 1
        self.errors[label] += error
        self.timeouts[label] += timeout
        if first_token_ms is not None:
            self.first_token_ms[label] = (self.first_token_ms[label] + [first_token_ms])[-self.KEEP:]
        self.total_ms[label] = (self.total_ms[label] + [total_ms])[-self.KEEP:]

    @staticmethod
    def _pct(xs: list[float], q: float) -> float | None:
        if not xs:
            return None
        xs = sorted(xs)
        return round(xs[min(len(xs) - 1, int(q * len(xs)))], 1)

    def snapshot(self) -> dict:
        return {"failovers": self.failovers, "backends": {
            k: {"turns": self.turns[k], "errors": self.errors[k], "timeouts": self.timeouts[k],
                "first_token_p50_ms": self._pct(self.first_token_ms[k], 0.5),
                "first_token_p95_ms": self._pct(self.first_token_ms[k], 0.95),
                "total_p50_ms": self._pct(self.total_ms[k], 0.5), "total_p95_ms": self._pct(self.total_ms[k], 0.95)}
            for k in self.turns}}


METRICS = ExpressionMetrics()


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
                 temperature: float = 0.3, max_tokens: int | None = None, timeout_s: float = 5.0,
                 first_token_s: float | None = None, label: str | None = None, extra_body: dict | None = None):
        # ACIE_EXPR_* point only the reply writer at another model (e.g. Gemini through its OpenAI-compatible
        # endpoint) while the understanding layer stays on the local Qwen (ACIE_LLM_*, guided JSON, fast)
        env = lambda k, d: os.getenv(f"ACIE_EXPR_{k}") or os.getenv(f"ACIE_LLM_{k}", d)
        self.base_url = (base_url or env("BASE_URL", "http://localhost:8000/v1")).rstrip("/")
        self.model = model or env("MODEL", "acie-expression")
        self.api_key = api_key or env("API_KEY", "EMPTY")
        # ACIE_PSYCHE=1: temperature 0, so the same plan gets the same words as far as the server allows
        self.temperature = 0.0 if os.getenv("ACIE_PSYCHE", "") in ("1", "true", "yes", "on") else temperature
        self.max_tokens = max_tokens or int(env("MAX_TOKENS", "320"))  # Devanagari is token-heavy
        # e.g. '{"chat_template_kwargs": {"enable_thinking": false}}' for Qwen3 on vLLM,
        # '{"reasoning_effort": "low"}' for Gemini
        self.extra_body = extra_body if extra_body is not None else json.loads(
            os.getenv("ACIE_EXPR_EXTRA_BODY") if os.getenv("ACIE_EXPR_BASE_URL") else os.getenv("ACIE_LLM_EXTRA_BODY", "{}") or "{}")
        self.client = httpx.AsyncClient(timeout=timeout_s)
        # ACIE_EXPR_FIRST_TOKEN_S (default off): no first token in time -> speak the safe template instead
        env_ft = os.getenv("ACIE_EXPR_FIRST_TOKEN_S")
        self.first_token_s = first_token_s if first_token_s is not None else (float(env_ft) if env_ft else None)
        self.fallback_to_template = True  # FailoverBackend turns this off on its primary to catch the timeout
        self.label = label or self.model

    async def stream(self, prompt: CompiledPrompt, plan: TurnPlan, expr: ExpressionParams) -> AsyncIterator[str]:
        t0 = time.perf_counter()
        first_ms: float | None = None
        error = timeout = False
        try:
            async for delta in self._timed(self._stream_raw(prompt, plan, expr)):
                if first_ms is None:
                    first_ms = round((time.perf_counter() - t0) * 1000, 1)
                yield delta
        except FirstTokenTimeout:
            timeout = True
            if not self.fallback_to_template:
                raise
            yield render_fallback(plan, expr)
        except BaseException as e:
            error = not isinstance(e, GeneratorExit | asyncio.CancelledError)
            raise
        finally:
            METRICS.record(self.label, first_ms, round((time.perf_counter() - t0) * 1000, 1), error, timeout)

    async def _timed(self, agen: AsyncIterator[str]) -> AsyncIterator[str]:
        """Pass chunks through; the first one must arrive within first_token_s (when set)."""
        it = agen.__aiter__()
        first = True
        try:
            while True:
                try:
                    if first and self.first_token_s:
                        delta = await asyncio.wait_for(it.__anext__(), self.first_token_s)
                    else:
                        delta = await it.__anext__()
                except StopAsyncIteration:
                    return
                except asyncio.TimeoutError:
                    raise FirstTokenTimeout(f"no first token in {self.first_token_s}s") from None
                first = False
                yield delta
        finally:
            aclose = getattr(it, "aclose", None)
            if aclose:
                await aclose()

    async def _stream_raw(self, prompt: CompiledPrompt, plan: TurnPlan, expr: ExpressionParams) -> AsyncIterator[str]:
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


class FailoverBackend:
    """Primary backend; if it fails before producing any text, the secondary answers instead."""

    def __init__(self, primary: ExpressionBackend, secondary: ExpressionBackend):
        self.primary, self.secondary = primary, secondary
        # a first-token timeout must reach us, not become the template
        for b in [v[1] for v in getattr(primary, "variants", [])] or [primary]:
            if hasattr(b, "fallback_to_template"):
                b.fallback_to_template = False

    async def stream(self, prompt: CompiledPrompt, plan: TurnPlan, expr: ExpressionParams) -> AsyncIterator[str]:
        started = False
        try:
            async for delta in self.primary.stream(prompt, plan, expr):
                started = True
                yield delta
            return
        except Exception:
            if started:
                raise  # half a reply was already released: let the pipeline fall back to the template
        METRICS.failovers += 1
        async for delta in self.secondary.stream(prompt, plan, expr):
            yield delta

    async def aclose(self) -> None:
        for b in (self.primary, self.secondary):
            if hasattr(b, "aclose"):
                await b.aclose()


def parse_ab(spec: str) -> list[tuple[str, float]]:
    """"modelA:0.5,modelB:0.5" -> [("modelA", .5), ("modelB", .5)]. A bare name has weight 1."""
    out = []
    for part in filter(None, (p.strip() for p in spec.split(","))):
        name, _, w = part.rpartition(":")
        try:
            out.append((name, float(w)) if name else (part, 1.0))
        except ValueError:
            out.append((part, 1.0))  # a model name containing ':' with no weight
    return [(n, w) for n, w in out if w > 0]


class ABBackend:
    """Split sessions between several backends by weight. Sticky: a session always gets the same variant."""

    def __init__(self, variants: list[tuple[str, ExpressionBackend, float]], salt: str = ""):
        if not variants:
            raise ValueError("ABBackend needs at least one variant")
        self.variants, self.salt = variants, salt
        self.assignments: dict[str, str] = {}

    def pick(self, key: str | None = None) -> tuple[str, ExpressionBackend]:
        key = key if key is not None else SESSION_KEY.get()
        u = (int(hashlib.sha256(f"{self.salt}|{key}".encode()).hexdigest()[:12], 16) / 16 ** 12
             if key is not None else random.random())
        total = sum(w for _, _, w in self.variants)
        acc = 0.0
        for label, backend, w in self.variants:
            acc += w / total
            if u < acc:
                break
        if key is not None:
            self.assignments[key] = label
        return label, backend

    async def stream(self, prompt: CompiledPrompt, plan: TurnPlan, expr: ExpressionParams) -> AsyncIterator[str]:
        _, backend = self.pick()
        async for delta in backend.stream(prompt, plan, expr):
            yield delta

    async def aclose(self) -> None:
        for _, b, _ in self.variants:
            if hasattr(b, "aclose"):
                await b.aclose()


def make_backend_from_env() -> ExpressionBackend:
    """Plain OpenAICompatibleBackend unless ACIE_EXPR_AB and/or ACIE_EXPR_FALLBACK_BASE_URL are set."""
    ab = os.getenv("ACIE_EXPR_AB")
    if ab:
        variants = [(m, OpenAICompatibleBackend(model=m), w) for m, w in parse_ab(ab)]
        backend: ExpressionBackend = ABBackend(variants, salt=os.getenv("ACIE_EXPR_AB_SALT", "")) if variants \
            else OpenAICompatibleBackend()
    else:
        backend = OpenAICompatibleBackend()
    fb_url = os.getenv("ACIE_EXPR_FALLBACK_BASE_URL")
    if fb_url:
        fb = OpenAICompatibleBackend(base_url=fb_url, model=os.getenv("ACIE_EXPR_FALLBACK_MODEL") or None,
                                     api_key=os.getenv("ACIE_EXPR_FALLBACK_API_KEY") or None,
                                     label="fallback:" + (os.getenv("ACIE_EXPR_FALLBACK_MODEL") or "default"))
        backend = FailoverBackend(backend, fb)
    return backend
