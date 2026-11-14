"""Closing focus: an interested caller is led to a natural ask within 4 turns, asks rotate, a firm no / hardship gets none,
and the move wording is natural (no robotic lead-ins, no feminine first person, no English in a Hindi reply)."""

import json
import re

import pytest

from acie.psyche.ledger import SaidLedger
from acie.psyche.moves import CLOSE_ASKS, LANGS, MoveContext, build_catalog, choose_move
from acie.psyche.needs import NeedInput, mid_explanation, table_need
from acie.decision.domains.product_sales import Product
from acie.signals import TurnSignals

FIT = Product.model_validate(json.load(open("data/voice/products.json"))["fitzone-gym"])
CLAUDE = Product.model_validate(json.load(open("data/voice/products.json"))["claude-pro"])
LINKISH = re.compile(r"\blink\b|\bsms\b|payment|लिंक|पेमेंट", re.I)


def turn(L, text, lang="en", interest=0.8, product=FIT, decided="undecided", has_link=False, **kw):
    sg = TurnSignals(learned=True, buying_interest=interest, state={"engagement": 0.8})
    n = table_need(NeedInput(text=text, state={"engagement": 0.8}, signals=sg, has_link=has_link, prev_agent="?"))
    L.add_move("need:" + n.need)
    mc = MoveContext(need=n, lang=lang, product=product, ledger=L, text=text, engagement=0.8, decided=decided, has_link=has_link,
                     refused=kw.pop("refused", False), **kw)
    mv = choose_move(mc)
    reply = mv.line(lang)
    L.add_text(reply)
    L.add_slot(mv.slot)
    L.add_move(mv.name)
    if mv.sells:
        L.offers += 1
    return mv, reply


@pytest.mark.parametrize("lang", LANGS)
def test_interested_caller_reaches_a_closing_ask_within_four_turns(lang):
    L = SaidLedger()
    asked_at = None
    for i, say in enumerate(["haan bataiye", "achha, aur?", "theek hai, mujhe pasand aaya", "haan"], 1):
        mv, reply = turn(L, say, lang)
        if mv.sells and mv.name.startswith("close_ask"):
            asked_at = i
            break
    assert asked_at is not None and asked_at <= 4
    assert not LINKISH.search(reply)  # the ask is "shall we start?", the link only follows a yes


def test_closing_ask_form_never_repeats_back_to_back():
    L = SaidLedger()
    L.add_slot("what")
    L.add_slot("offer")
    asks = []
    for say in ["achha", "ok", "hmm theek", "sahi hai"]:
        mv, reply = turn(L, say, "hinglish")
        if mv.name.startswith("close_ask"):
            asks.append(reply)
    assert len(asks) >= 2 and len(set(asks)) == len(asks)
    assert len(asks) <= 3  # never more than the cap: more is pushing


def test_at_least_three_forms_per_language_and_none_feminine_or_mixed():
    assert len(CLOSE_ASKS) >= 3
    for form in CLOSE_ASKS:
        assert not re.search(r"karungi|sakti|chahungi|rakhungi|lungi", form["hinglish"], re.I)
        assert not re.search(r"[A-Za-z]{4,}", form["hi"])  # no English sentence in the Hindi form
        assert len({form[l] for l in LANGS}) == 3


@pytest.mark.parametrize("say", ["not interested, please stop calling", "नहीं चाहिए", "mujhe nahi chahiye", "call me later"])
def test_a_firm_no_never_gets_a_close_or_a_link(say):
    L = SaidLedger()
    L.add_slot("what")
    L.add_slot("offer")
    mv, reply = turn(L, say, "en", interest=0.9)
    assert not mv.sells and not mv.name.startswith("close_ask")
    assert not LINKISH.search(reply)
    # and later turns, even with high interest signals, stay quiet once the no was said
    mv2, reply2 = turn(L, "achha ok", "en", interest=0.9, refused=True)
    assert not mv2.name.startswith("close_ask") and not mv2.sells


def test_no_selling_to_someone_in_hardship():
    L = SaidLedger()
    L.add_slot("what")
    L.add_slot("offer")
    mv, reply = turn(L, "I lost my job, I have no money right now", "en", interest=0.9)
    assert not mv.sells and not LINKISH.search(reply)
    mv2, _ = turn(L, "ok tell me", "en", interest=0.9, hold=True)
    assert not mv2.name.startswith("close_ask")


def test_no_close_once_the_link_went_out_or_a_decision_was_made():
    for kw in ({"has_link": True}, {"decided": "accepted"}):
        mv, _ = turn(SaidLedger(), "ok great", "en", **kw)
        assert not mv.name.startswith("close_ask")


def test_benefit_wording_is_natural_no_robotic_lead_in():
    for p in (FIT, CLAUDE):
        for it in build_catalog(p):
            for lang, txt in it.text.items():
                assert "Ek cheez jo milti hai" not in txt and "एक चीज़ जो मिलती है" not in txt and "One thing you get" not in txt
                assert "Once more, step by step" not in txt and "Phir se, dheere-dheere" not in txt


def test_restated_fact_has_no_robotic_lead_in():
    from acie.psyche.moves import _restate
    for it in build_catalog(FIT):
        for lang in LANGS:
            for line in _restate(it, FIT, lang):
                assert not re.search(r"Once more|Phir se|फिर से, धीरे|छोटे में फिर से|Chhote mein phir se", line)


def test_mid_explanation_is_never_ended_and_never_asked_to_repeat():
    assert mid_explanation("I wanted to say that my budget is tight and,")
    assert mid_explanation("मेरा बजट कम है क्योंकि")
    assert not mid_explanation("ok bye")
    L = SaidLedger()
    mv, reply = turn(L, "see the thing is my daughter is in college and also", "en", interest=0.1)
    assert not mv.end_call and reply
    L2 = SaidLedger()
    mv, reply = turn(L2, "ok that's all because", "en", interest=0.1)
    assert not mv.end_call


def test_clear_caller_text_never_gets_a_didnt_catch():
    n = table_need(NeedInput(text="mujhe samajh nahi aaya aap phir se bolo please"))
    L = SaidLedger()
    L.add_move("need:" + n.need)
    mv = choose_move(MoveContext(need=n, lang="hinglish", product=FIT, ledger=L, text="mujhe samajh nahi aaya aap phir se bolo please"))
    assert mv.name != "ask_repeat" and "sun nahi paya" not in mv.line("hinglish") and "catch" not in mv.line("en")
