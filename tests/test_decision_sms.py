import sys
import types

try:
    import onnxruntime  # noqa: F401
except ImportError:  # voice.call imports the speaker/VAD models at module level; the SMS path does not use them
    sys.modules["onnxruntime"] = types.ModuleType("onnxruntime")

from decision_helpers import run

from acie.decision.domains.product_sales import Product
from voice.call import CallRecord, CallSession


class FakePlivo:
    def __init__(self):
        self.sent = []

    async def send_sms(self, to, text):
        self.sent.append((to, text))
        return {"status": "sent", "message_id": "m1"}


def fake_call(product, pipe, to="+919999999999"):
    rec = CallRecord(to=to, campaign=product.id)
    events = []
    cs = types.SimpleNamespace(product=product, call=rec, app=types.SimpleNamespace(plivo=FakePlivo()),
                               pipeline=pipe, session_id="s", event=lambda k, v: events.append((k, v)))
    return cs, events


SHOP = dict(id="shop", seller_name="FitZone", product_name="Gold pass", offer_price="Rs 999", original_price="Rs 1999",
            payment_link="https://pay.fitzone.example/gold",
            sms_text="{seller}: pay for {product} ({offer_price}) here {link} {unknown}")


async def sms_after(lines, product, **kw):
    out, last, pipe, _, _ = await run(lines, product)
    cs, ev = fake_call(product, pipe, **kw)
    if out[-1] == "link_sent":  # what voice/call.py does after the turn
        await CallSession.send_link(cs)
    return out, cs


async def test_default_product_never_sends_sms():
    out, cs = await sms_after(["yes tell me", "how much", "yes send the link"], Product())
    assert out[-1] == "link_sent" and cs.app.plivo.sent == [] and cs.call.sms == []


async def test_sms_sent_once_after_confirmed_yes_with_link_and_safe_format():
    out, cs = await sms_after(["yes tell me", "how much", "yes send the link"], Product(**SHOP))
    assert out[-1] == "link_sent"
    ((to, text),) = cs.app.plivo.sent
    assert to == "+919999999999"
    assert text == "FitZone: pay for Gold pass (Rs 999) here https://pay.fitzone.example/gold {unknown}"
    await CallSession.send_link(cs)  # asked again: still one SMS per call
    assert len(cs.app.plivo.sent) == 1 and cs.call.sms[0]["status"] == "sent"


async def test_no_sms_before_consent_or_after_decline():
    for lines in (["yes tell me", "how much"], ["yes tell me", "no not interested", "nahi chahiye"],
                  ["yes tell me", "is it fake?"]):
        out, cs = await sms_after(lines, Product(**SHOP))
        assert out[-1] != "link_sent" and cs.app.plivo.sent == []


async def test_api_payment_link_is_the_one_texted():
    from acie.business.base import PaymentLink

    class API:
        async def eligibility(self, c):
            from acie.business.base import Eligibility
            return Eligibility()

        async def quote(self, p, c):
            from acie.business.base import Quote
            return Quote()

        async def create_payment_link(self, p, c):
            return PaymentLink(url="https://pay.fitzone.example/unique-123", verified=True)

        async def record_outcome(self, i, o):
            pass
    p = Product(**SHOP)
    out, last, pipe, _, _ = await run(["yes tell me", "how much", "yes send the link"], p, business=API())
    cs, _ = fake_call(p, pipe)
    await CallSession.send_link(cs)
    assert "unique-123" in cs.app.plivo.sent[0][1]


async def test_restricted_campaign_never_texts_other_numbers():
    p = Product(**SHOP, allowed_numbers=["+911111111111"])
    out, cs = await sms_after(["yes tell me", "how much", "yes send the link"], p, to="+912222222222")
    assert cs.app.plivo.sent == []
    out, cs = await sms_after(["yes tell me", "how much", "yes send the link"], p, to="+911111111111")
    assert len(cs.app.plivo.sent) == 1
