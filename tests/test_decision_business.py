import asyncio

import httpx
from decision_helpers import run

from acie.business import Customer, HTTPBusinessAPI, MockBusinessAPI
from acie.business.base import BusinessAPI, Eligibility, PaymentLink, Quote
from acie.decision.domains.product_sales import Product


class FakeAPI:
    """A verified backend that changes the offer and the link."""
    def __init__(self, eligible=True, slow=0.0):
        self.eligible, self.slow, self.outcomes = eligible, slow, []

    async def eligibility(self, customer):
        await asyncio.sleep(self.slow)
        return Eligibility(eligible=self.eligible, reason="" if self.eligible else "blocked_by_crm", verified=True, source="fake")

    async def quote(self, product, customer):
        await asyncio.sleep(self.slow)
        return Quote(price="$5 per month", original_price="$25 per month", validity="for the first 2 months", verified=True)

    async def create_payment_link(self, product, customer):
        return PaymentLink(url="https://pay.example.com/abc", verified=True)

    async def record_outcome(self, call_id, outcome):
        self.outcomes.append((call_id, outcome))


async def test_mock_is_default_and_changes_nothing():
    out, last, pipe, dom, _ = await run(["yes tell me", "how much", "yes send the link"])
    assert isinstance(dom.business, MockBusinessAPI) and isinstance(dom.business, BusinessAPI)
    assert out[-1] == "link_sent"
    s = pipe.sessions["s"]
    assert s.dialog["payment_link_url"] == "https://claude.ai/upgrade"
    await asyncio.sleep(0)
    assert dom.business.outcomes == [("s", "accepted")]


async def test_verified_api_facts_flow_through_fact_slots():
    api = FakeAPI()
    out, last, pipe, dom, _ = await run(["kitne ka hai ye"], business=api)
    assert out[-1] == "price" and dom.p.offer_price == "$5 per month"
    assert "$5" in last.plan.facts["offer"].value and "$25" in last.plan.facts["offer"].value
    out, last, pipe, dom, _ = await run(["yes tell me", "how much", "yes send the link"], business=api)
    assert out[-1] == "link_sent"
    assert last.plan.facts["payment_link"].value == "https://pay.example.com/abc"
    assert last.plan.facts["payment_link"].hidden
    assert pipe.sessions["s"].dialog["payment_link_url"] == "https://pay.example.com/abc"
    await asyncio.sleep(0)
    assert ("s", "accepted") in api.outcomes


async def test_verified_ineligible_ends_call_politely():
    out, last, _, _, opening = await run([], business=FakeAPI(eligible=False))
    assert opening.intent == "constraint_stop" and opening.end_call and not opening.llm_allowed


async def test_slow_api_never_blocks_a_turn_past_budget():
    import time
    t = time.monotonic()
    out, _, _, dom, _ = await run(["kitne ka hai ye"], business=FakeAPI(slow=5))
    assert time.monotonic() - t < 4 and out[-1] == "price" and dom.p.offer_price == "$8 per month"


async def test_api_exception_is_swallowed():
    class Broken(FakeAPI):
        async def eligibility(self, customer):
            raise RuntimeError("x")

        async def create_payment_link(self, product, customer):
            raise RuntimeError("x")
    out, last, pipe, _, _ = await run(["yes tell me", "how much", "yes send the link"], business=Broken())
    assert out[-1] == "link_sent" and pipe.sessions["s"].dialog["payment_link_url"] == "https://claude.ai/upgrade"


# ------------------------------------------------------------------ HTTP implementation

def client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_http_api_success_sends_bearer_and_is_verified():
    seen = []

    def h(req):
        seen.append((req.url.path, req.headers.get("authorization")))
        if req.url.path == "/eligibility":
            return httpx.Response(200, json={"eligible": False, "reason": "existing_subscriber"})
        if req.url.path == "/quote":
            return httpx.Response(200, json={"price": "Rs 99", "original_price": "Rs 199", "validity": "for 1 month"})
        if req.url.path == "/payment-link":
            return httpx.Response(200, json={"url": "https://pay.example.com/x"})
        return httpx.Response(200, json={})
    api = HTTPBusinessAPI("https://biz.example.com", "tok", client=client(h))
    e = await api.eligibility(Customer(id="+91"))
    q = await api.quote(Product(), Customer())
    link = await api.create_payment_link(Product(), Customer())
    await api.record_outcome("c1", "accepted")
    assert not e.eligible and e.verified and q.price == "Rs 99" and q.verified and link.url.endswith("/x")
    assert all(a == "Bearer tok" for _, a in seen) and [p for p, _ in seen][-1] == "/outcome"


async def test_http_api_falls_back_unverified_on_error_timeout_and_bad_payload():
    calls = []

    def boom(req):
        calls.append(1)
        return httpx.Response(500)
    api = HTTPBusinessAPI("https://biz.example.com", "t", client=client(boom), retries=2)
    q = await api.quote(Product(), Customer())
    assert not q.verified and q.source == "fallback" and q.price == "$8 per month" and len(calls) == 3
    link = await HTTPBusinessAPI("https://b", "t", client=client(lambda r: httpx.Response(200, json={"url": "http://insecure"}))).create_payment_link(Product(), Customer())
    assert not link.verified and link.url == "https://claude.ai/upgrade"

    async def slow(req):
        await asyncio.sleep(3)
    import time
    t = time.monotonic()
    e = await HTTPBusinessAPI("https://b", "t", budget_s=0.2, client=httpx.AsyncClient(transport=httpx.MockTransport(slow))).eligibility(Customer())
    assert time.monotonic() - t < 1.0 and not e.verified and e.eligible


async def test_http_api_budget_is_capped_at_1_5s_and_no_url_means_mock(monkeypatch):
    assert HTTPBusinessAPI("https://b", "t", budget_s=60).budget_s == 1.5
    monkeypatch.delenv("ACIE_BUSINESS_API_URL", raising=False)
    q = await HTTPBusinessAPI().quote(Product(), Customer())
    assert not q.verified and q.source == "fallback"
    from acie.business import from_env
    assert isinstance(from_env(), MockBusinessAPI)
    monkeypatch.setenv("ACIE_BUSINESS_API_URL", "https://x")
    assert isinstance(from_env(), HTTPBusinessAPI)
