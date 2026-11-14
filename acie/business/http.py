"""HTTP business API: ACIE_BUSINESS_API_URL + ACIE_BUSINESS_API_TOKEN (bearer).

Endpoints (JSON, POST): /eligibility, /quote, /payment-link, /outcome. Every call has a hard time
budget (default 1.5 s in total, retries included) so it can never stall a live turn; on a timeout,
an error or a malformed answer the mock's value is returned marked `verified=False,
source="fallback"`. Nothing here raises.
"""

from __future__ import annotations

import asyncio
import os
import time

import httpx

from .base import Customer, Eligibility, PaymentLink, Quote
from .mock import MockBusinessAPI


class HTTPBusinessAPI:
    def __init__(self, base_url: str | None = None, token: str | None = None, budget_s: float = 1.5,
                 retries: int = 1, client: httpx.AsyncClient | None = None, fallback: MockBusinessAPI | None = None):
        self.base = (base_url if base_url is not None else os.getenv("ACIE_BUSINESS_API_URL", "")).rstrip("/")
        token = token if token is not None else os.getenv("ACIE_BUSINESS_API_TOKEN", "")
        self.headers = {"Authorization": f"Bearer {token}"} if token else {}
        self.budget_s = min(budget_s, 1.5)  # never block a turn longer than this
        self.retries = retries
        self.client = client
        self.mock = fallback or MockBusinessAPI()

    async def _post(self, path: str, body: dict) -> dict | None:
        if not self.base:
            return None
        deadline = time.monotonic() + self.budget_s
        for _ in range(self.retries + 1):
            left = deadline - time.monotonic()
            if left <= 0.05:
                return None
            try:
                async def go() -> dict:
                    c = self.client or httpx.AsyncClient()
                    try:
                        r = await c.post(f"{self.base}{path}", json=body, headers=self.headers, timeout=left)
                    finally:
                        if self.client is None:
                            await c.aclose()
                    r.raise_for_status()
                    j = r.json()
                    if not isinstance(j, dict):
                        raise ValueError("not an object")
                    return j
                return await asyncio.wait_for(go(), left)
            except Exception:
                continue
        return None

    @staticmethod
    def _product_ref(product) -> dict:
        return {"id": getattr(product, "id", ""), "tenant": getattr(product, "tenant", "")}

    async def eligibility(self, customer: Customer) -> Eligibility:
        j = await self._post("/eligibility", {"customer": customer.model_dump()})
        try:
            if j is not None:
                return Eligibility(eligible=bool(j["eligible"]), reason=str(j.get("reason", "")),
                                   verified=True, source="http")
        except Exception:
            pass
        e = await self.mock.eligibility(customer)
        return e.model_copy(update={"verified": False, "source": "fallback"})

    async def quote(self, product, customer: Customer) -> Quote:
        j = await self._post("/quote", {"product": self._product_ref(product), "customer": customer.model_dump()})
        try:
            if j is not None:
                return Quote(product_id=getattr(product, "id", ""), price=str(j["price"]),
                             original_price=str(j.get("original_price", "")), validity=str(j.get("validity", "")),
                             terms={str(k): str(v) for k, v in (j.get("terms") or {}).items()},
                             verified=True, source="http")
        except Exception:
            pass
        q = await self.mock.quote(product, customer)
        return q.model_copy(update={"verified": False, "source": "fallback"})

    async def create_payment_link(self, product, customer: Customer) -> PaymentLink:
        j = await self._post("/payment-link", {"product": self._product_ref(product), "customer": customer.model_dump()})
        try:
            if j is not None and str(j["url"]).startswith("https://"):
                return PaymentLink(url=str(j["url"]), verified=True, source="http")
        except Exception:
            pass
        link = await self.mock.create_payment_link(product, customer)
        return link.model_copy(update={"verified": False, "source": "fallback"})

    async def record_outcome(self, call_id: str, outcome: str) -> None:
        await self._post("/outcome", {"call_id": call_id, "outcome": outcome})


def from_env():
    """HTTP API when ACIE_BUSINESS_API_URL is set, else the mock."""
    return HTTPBusinessAPI() if os.getenv("ACIE_BUSINESS_API_URL") else MockBusinessAPI()
