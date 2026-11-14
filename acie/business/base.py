"""Business API seam for the decision layer.

The decision engine never invents business facts (eligibility, price, payment link). It asks a
`BusinessAPI`. The mock returns what the product config already says (today's behaviour); the HTTP
implementation talks to the operator's backend and, when that is slow or down, falls back to the
mock's values marked `verified=False`. Whatever comes back is turned into `Fact` slots by the
domain: the LLM never writes these values.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field


class Customer(BaseModel):
    """What the call knows about the person being called (all optional)."""
    id: str = ""  # phone number / CRM id
    name: str = ""
    age: int | None = None
    region: str = ""
    existing_subscriber: bool | None = None
    extra: dict = Field(default_factory=dict)


class Eligibility(BaseModel):
    eligible: bool = True
    reason: str = ""
    verified: bool = False  # True only when the backend itself answered
    source: str = "mock"


class Quote(BaseModel):
    product_id: str = ""
    price: str = ""
    original_price: str = ""
    validity: str = ""
    terms: dict[str, str] = Field(default_factory=dict)  # optional per-language phrase of the full offer
    verified: bool = False
    source: str = "mock"


class PaymentLink(BaseModel):
    url: str = ""
    verified: bool = False
    source: str = "mock"


@runtime_checkable
class BusinessAPI(Protocol):
    async def eligibility(self, customer: Customer) -> Eligibility: ...
    async def quote(self, product, customer: Customer) -> Quote: ...
    async def create_payment_link(self, product, customer: Customer) -> PaymentLink: ...
    async def record_outcome(self, call_id: str, outcome: str) -> None: ...
