"""Mock business API: exactly today's behaviour (everything comes from the product config)."""

from __future__ import annotations

from .base import Customer, Eligibility, PaymentLink, Quote


class MockBusinessAPI:
    def __init__(self) -> None:
        self.outcomes: list[tuple[str, str]] = []  # (call_id, outcome), for tests and the console

    async def eligibility(self, customer: Customer) -> Eligibility:
        if customer.existing_subscriber:
            return Eligibility(eligible=False, reason="existing_subscriber", verified=False, source="mock")
        return Eligibility(eligible=True, verified=False, source="mock")

    async def quote(self, product, customer: Customer) -> Quote:
        return Quote(product_id=getattr(product, "id", ""), price=getattr(product, "offer_price", ""),
                     original_price=getattr(product, "original_price", ""),
                     validity=getattr(product, "offer_validity", ""), verified=False, source="mock")

    async def create_payment_link(self, product, customer: Customer) -> PaymentLink:
        return PaymentLink(url=getattr(product, "payment_link", ""), verified=False, source="mock")

    async def record_outcome(self, call_id: str, outcome: str) -> None:
        self.outcomes.append((call_id, outcome))
