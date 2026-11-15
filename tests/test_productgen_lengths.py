"""Phone-length rules for generated products (FitZone's hand-written lengths are the reference)."""

import json

from acie.productgen import lengths as LN

LONG_PITCH = ("Get ready to experience Karan Aujla live in concert with his high-energy Punjabi hits and massive stage production. "
              "Reserve your official passes now to secure your spot at the concert.")


def bundle():
    return {"product": {
        "pitch_en": LONG_PITCH, "pitch_hi": "छोटा पिच।", "pitch_hinglish": "",
        "benefits": ["Guaranteed genuine digital passes delivered instantly after booking and sent to your email and phone right away."],
        "benefits_hi": [], "benefits_hinglish": [],
        "faqs": [{"keywords": ["refund"], "answer": "Concert passes are non-refundable once issued. However, digital passes are transferable to friends "
                                                   "or family members through the official ticketing app before the event.",
                  "answer_hi": "", "answer_hinglish": ""}],
        "discovery_question": {"en": "Which category of tickets are you interested in, and how many passes do you require for your group today?"},
        "call_to_action": "Select your preferred ticket zone and book your concert passes today.",
    }, "kb": {"entries": [{"id": "e1", "answer": {"en": "Fine short answer about {offer_price}."}}]},
        "tree": {"nodes": [{"id": "n1", "points": ["We have official passes available for Karan Aujla's upcoming India tour across many big cities."],
                            "ask": "Which city?", "children": []}], "ladder": []}}


def test_fitzone_reference_passes():
    fz = {"product": {"pitch_en": "FitZone is a gym with trainers and yoga and zumba classes, open from 5 in the morning to 11 at night.",
                      "benefits": ["a free personal trainer session every week", "unlimited group classes like yoga and zumba", "open from 5 am to 11 pm"],
                      "faqs": [{"answer": "We have branches across the city; our team will tell you the one nearest you"}],
                      "discovery_question": {"en": "What's your fitness goal: losing weight, building muscle, or just staying fit?"},
                      "call_to_action": "have our team send you the joining link"}}
    assert LN.over_length(fz) == []


def test_over_length_finds_every_long_field():
    keys = {s.key for s in LN.over_length(bundle())}
    assert keys == {"product.pitch_en", "product.benefits[0]", "product.faqs[0].answer", "product.discovery_question.en",
                    "node.n1.points[0]"}  # cta is 11 words: fits
    assert LN.Slot("k", "hi", "faq", lambda: "", lambda v: None).limit == round(22 * LN.HI_FACTOR)


def test_trim_keeps_whole_sentences_or_a_clause():
    assert LN.trim(LONG_PITCH, 25) == ("Get ready to experience Karan Aujla live in concert with his high-energy Punjabi hits and massive stage "
                                       "production.")
    t = LN.trim("Concert passes are non-refundable once issued. However, digital passes are transferable to friends.", 10)
    assert t == "Concert passes are non-refundable once issued."
    one = LN.trim("Which category of tickets are you interested in, and how many passes do you require for your group today?", 15)
    assert one == "Which category of tickets are you interested in?" and LN.words(one) <= 15
    assert LN.trim("Guaranteed genuine digital passes delivered instantly after booking and sent to your email.", 10, phrase=True) \
        == "guaranteed genuine digital passes delivered instantly after booking"


def test_phrase_style_like_fitzone():
    assert LN.phrase_style("Guaranteed genuine digital passes.", "en") == "guaranteed genuine digital passes"
    assert LN.phrase_style("VIP lounge access", "en") == "VIP lounge access"
    assert LN.phrase_style("Karan Aujla meet and greet", "en") == "Karan Aujla meet and greet"


class FakeLLM:
    model = "fake"

    def __init__(self, bad_key=None):
        self.calls, self.bad_key = [], bad_key

    async def json(self, stage, payload):
        self.calls.append(stage)
        out = []
        for it in payload["items"]:
            txt = " ".join(it["text"].split()[: it["max_words"]])
            if it["key"] == self.bad_key:
                txt = "Only 99 rupees today"  # a new number: must be rejected and trimmed instead
            out.append({"key": it["key"], "text": txt})
        return {"items": out}


async def test_shorten_rewrites_then_trims_and_never_adds_numbers():
    b = bundle()
    llm = FakeLLM(bad_key="product.faqs[0].answer")
    notes = []
    st = await LN.shorten(b, llm, notes)
    assert llm.calls == ["shorten"] and st["over"] == 5
    assert LN.over_length(b) == []
    assert "99" not in json.dumps(b) and b["product"]["faqs"][0]["answer"] == "Concert passes are non-refundable once issued"  # no final stop
    assert b["product"]["call_to_action"].startswith("select ") and not b["product"]["call_to_action"].endswith(".")
    assert b["product"]["benefits"][0][0].islower()
    assert b["kb"]["entries"][0]["answer"]["en"] == "Fine short answer about {offer_price}."
    assert any("phone length" in n for n in notes)


async def test_shorten_survives_a_failing_model():
    class Boom:
        async def json(self, stage, payload):
            raise RuntimeError("down")
    b = bundle()
    await LN.shorten(b, Boom(), [])
    assert LN.over_length(b) == []


def test_criteria_flags_long_texts():
    from acie.productgen import criteria as C
    from acie.productgen.spec import Spec, resolve
    spec = Spec(name="zz test", authorised=True)
    b = bundle()
    b["kb"] = {"entries": [], "topics": [], "offerings": [], "interests": []}
    b["tree"]["ladder"] = []
    chk = next(c for c in C.check(b, spec, resolve(spec))["checks"] if c["id"] == "lengths")
    assert not chk["passed"] and not chk["blocking"] and "product.pitch_en" in chk["items"]


def test_two_sentences_only_when_both_are_very_short():
    s = LN.Slot("faq", "en", "faq", lambda: "", lambda v: None)
    assert s.fits("Passes are non-refundable once issued.")
    assert s.fits("No refunds. Passes are transferable.")  # 5 words in two sentences: fine
    two = "Concert passes are non-refundable once issued. However, digital passes are transferable to friends or family."
    assert LN.words(two) <= 22 and not s.fits(two)  # 15 words over two sentences: too long for one spoken answer
    assert LN.trim(two, s.limit, False, "en", s.max_sentences, s.two_cap) == "Concert passes are non-refundable once issued."
    p = LN.Slot("pitch", "en", "pitch", lambda: "", lambda v: None)
    assert p.fits("FitZone is a gym with trainers. It is open from 5 am to 11 pm.")
    assert not LN.Slot("point", "en", "point", lambda: "", lambda v: None).phrase  # spoken before the ask: keeps its full stop


async def test_rewrite_must_fit_the_sentence_rule_too():
    class TwoSentences:
        async def json(self, stage, payload):
            return {"items": [{"key": it["key"], "text": "Passes are not refundable at all once issued. You can transfer them to a friend before the gate."}
                              for it in payload["items"]]}
    b = {"product": {"faqs": [{"answer": "Concert passes are non-refundable once issued. However, digital passes are transferable to friends "
                                         "or family members through the official ticketing app before the event."}]}, "kb": {}, "tree": {}}
    st = await LN.shorten(b, TwoSentences(), [])
    assert st["rewritten"] == 0 and b["product"]["faqs"][0]["answer"] == "Concert passes are non-refundable once issued"
