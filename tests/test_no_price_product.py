"""A product with no prices set (live call 2e184aaf90, karan-aujla-concert-tickets): never an empty slot ("Right now it's ."), a price
question gets an honest "it depends on the option" answer, no negotiation, title-cased names, clean benefit insertion."""
import asyncio
import json
import re

import pytest

from acie.decision import deal as dl
from acie.decision import negotiation
from acie.decision.domains.product_sales import Product, ProductSalesDomain
from acie.decision.engine import DecisionEngine
from acie.pipeline import ACIEPipeline
from acie.schemas import ExpressionParams, TurnInput

RAW = json.load(open("data/voice/products.json"))
CONCERT = {"id": "karan-aujla-concert-tickets", "seller_name": "karan aujla concert tickets", "product_name": "karan aujla concert tickets",
           "one_liner": "Official tickets and entry passes for Karan Aujla's live music concert tour across India.",
           "original_price": "", "offer_price": "", "discount": "", "offer_validity": "", "terms": {},
           "benefits": ["Guaranteed genuine digital passes delivered instantly after booking.", "Access to exclusive zones including Fan Pit, VIP, and General Access."],
           "benefits_hi": ["बुकिंग के तुरंत बाद डिजिटल पास की गारंटीड डिलीवरी।", "फैन पिट, वीआईपी और जनरल एक्सेस जैसे खास ज़ोन तक पहुंच।"],
           "benefits_hinglish": ["Booking ke turant baad digital pass ki guaranteed delivery.", "Fan Pit, VIP aur General Access jaise exclusive zones tak access."],
           "pitch_en": "Get ready to experience Karan Aujla live in concert with his high-energy Punjabi hits and massive stage production. Reserve your official passes now to secure your spot at the concert.",
           "pitch_hi": "करन औजला के लाइव कॉन्सर्ट का अनुभव लेने के लिए तैयार हो जाइए।", "pitch_hinglish": "Karan Aujla ke live concert ka experience lene ke liye tayar ho jaiye.",
           "discovery_question": {"en": "Which category of tickets are you interested in, and how many passes do you require?",
                                  "hi": "आप किस कैटेगरी के टिकट खरीदना चाहते हैं और आपको कितने पास की आवश्यकता है?",
                                  "hinglish": "Aap kis category ke ticket lena chahte hain aur kitne pass chahiye?"},
           "topic_words": ["karan aujla", "concert", "tickets"], "sms_text": "", "payment_link": ""}
P = Product.model_validate(CONCERT)  # the in-file no-price example (the live concert entry now has prices)
EMPTY = re.compile(r"\s[.,]|\bis\s*,|it's\s*\.|it's\s*,|for you it's\s+,|,\s*then\s*\.|that's\s*\.|:\s*\.|\{\w+\}")


def talk(lines, lang="en"):
    async def go():
        pipe = ACIEPipeline(DecisionEngine(ProductSalesDomain(P, customer_name="Arkin")))
        T = lambda t: TurnInput(session_id="s", customer_id="+910000000000", text=t)  # noqa: E731
        sess = pipe.session(T(""))
        sess.expression = ExpressionParams(language=lang)
        op = await pipe.decision.decide("greet", T(""), sess)
        sess.context.record_turn(0, "", op.intent, False)
        out = [(op.intent, pipe.decision.domain.p.product_name)]
        for line in lines:
            r = await pipe.run_turn(T(line))
            out.append((r.plan.intent, r.text))
        return out, sess
    return asyncio.run(go())


LIVE = ["I'll be back.", "Uh, sure.", "Hello.", "I just told you please tell me the price", "What is the price?"]


@pytest.mark.parametrize("lang,lines", [("en", LIVE), ("hinglish", ["haan", "haan bolo", "kitne ka hai?", "price kya hai"]),
                                        ("hi", ["हाँ", "हाँ बताइए", "कितने का है?"])])
def test_no_price_product_never_speaks_an_empty_slot(lang, lines):
    out, sess = talk(lines, lang)
    for intent, text in out[1:]:
        assert not EMPTY.search(text), (intent, text)
        assert not re.search(r"₹|\$|\d+%", text), (intent, text)
    assert dl.load(sess.context, sess.dialog).status == "none"


def test_price_question_gets_the_honest_depends_on_the_option_answer():
    out, _ = talk(LIVE)
    for intent, text in out[4:]:
        assert re.search(r"depends? on (the|which) option", text) and "₹" not in text, text


def test_pitch_omits_the_discount_sentence_and_names_are_title_cased():
    out, _ = talk(LIVE)
    assert out[0][1] == "Karan Aujla Concert Tickets"
    pitch = [t for i, t in out if i == "pitch"]
    for t in pitch:
        assert "Right now it's" not in t


def test_no_negotiation_without_a_list_price():
    assert not negotiation.ladder_enabled(P) and dl.config_for(P) is None
    out, sess = talk(["haan", "price?", "too expensive, give me a discount", "2000 mein do"])
    for intent, text in out[1:]:
        assert intent != "negotiation" and not re.search(r"₹\s?\d", text), (intent, text)


def test_inserted_benefit_is_lower_cased_with_one_period():
    from acie.psyche.moves import build_catalog
    items = [i for i in build_catalog(P) if i.kind == "benefit"]
    en = items[0].text["en"]
    assert en == "It also comes with guaranteed genuine digital passes delivered instantly after booking." and ".." not in en
    assert "Fan Pit" in items[1].text["en"] and "VIP" in items[1].text["en"]
    assert not any(i.slot == "offer" for i in build_catalog(P))
