"""The Indian negotiation ladder: a warm no first, then one shrinking step per push, midpoint for counters, accept at/after the last step, a
profit check on every offer, never a lower-then-higher price, a new line every round, the agreed price carried into the terms."""
import json
import random
import re
from difflib import SequenceMatcher

import pytest

from acie.decision import negotiation
from acie.decision.domains.product_sales import Economics, Product, Step

PRODS = {k: Product.model_validate(v) for k, v in json.load(open("data/voice/products.json")).items()}
GYM, CLAUDE = PRODS["fitzone-gym"], PRODS["claude-pro"]
FLOOR = GYM.economics.profit_floor
LANGS = ("en", "hi", "hinglish")
TONES = ("professional", "mid", "casual")
FINAL_WORDS = re.compile(r"\bfinal\b|antim|अंतिम|फ़ाइनल|aakhri|आख़िरी", re.I)
URGENCY = re.compile(r"abhi tay|अभी तय|last chance|only \d+ left|hurry|expires|jaldi karo|आज ही खत्म|offer (ends|expires)", re.I)


def play(lines, p=GYM, tone="mid", **kw):
    st, out = negotiation.NegState(), []
    for t in lines:
        r = negotiation.step(p, t, st, "", None, tone=tone, **kw)
        st = r.state
        out.append(r)
    return out


def prices_in(text):
    return [int(x.replace(",", "")) for x in re.findall(r"₹([\d,]+)", text)]


def test_ladder_math_comes_from_list_and_floor_only():
    assert FLOOR == 2400.0
    seq = negotiation.ladder_prices(2999, FLOOR)
    assert seq == [2799, 2649, 2549, 2449]                      # 33%, 25%, 17% of the 599 room, friendly ...49/...99, last = floor rounded UP
    cuts = [a - b for a, b in zip([2999] + seq, seq)]
    assert cuts == sorted(cuts, reverse=True) or cuts[-1] <= cuts[0]   # shrinking steps
    assert cuts[0] == 200 and cuts[1] == 150 and cuts[2] == 100
    assert seq[-1] >= FLOOR and all(a > b for a, b in zip(seq, seq[1:]))
    assert negotiation.ladder_prices(2999, 2999) == [] and negotiation.ladder_prices(2999, 2995) == []   # nothing to give


def test_first_push_is_a_warm_no_with_value_and_the_budget_question_no_cut():
    r = play(["bahut mehenga hai"])[0]
    assert r.offer is None and not prices_in(r.text["hinglish"].replace("₹33", ""))   # no price on the table, no cut
    assert "?" in r.text["hinglish"] and "budget" in r.text["hinglish"].lower()
    assert "free personal trainer" in r.text["en"] or "personal trainer" in r.text["hinglish"]   # concrete value from the facts
    assert "₹33" in r.text["hinglish"]                                                           # per-day value


def test_haggler_saying_only_bahut_mehenga_gets_stepwise_prices_asks_and_never_a_repeated_no():
    out = play(["bahut mehenga hai"] * 5)
    got = [r.offer["price"] for r in out[1:]]
    assert got == [2799, 2649, 2549, 2449]
    for r in out[1:]:
        assert "?" in r.text["hinglish"] and r.offer
    texts = [r.text["hinglish"] for r in out]
    for a, b in zip(texts, texts[1:]):
        assert SequenceMatcher(None, a, b).ratio() < 0.75, (a, b)
    assert len(set(texts)) == len(texts)
    for l in LANGS:
        for r in out:
            assert not URGENCY.search(r.text[l])


def test_only_the_true_last_step_says_final_and_only_once():
    out = play(["kam karo"] * 9)
    finals = [i for i, r in enumerate(out) if FINAL_WORDS.search(r.text["hinglish"]) and r.offer and r.offer["price"] == 2449]
    assert finals == [4]                                                                    # the 2,449 step, nowhere else
    for r in out[1:4]:
        assert not FINAL_WORDS.search(r.text["hinglish"])
    assert sum("mushkil" in r.text["hinglish"] for r in out[1:4]) == 1                      # "isse neeche mushkil hai" once per call, never a false "last offer"


def test_counter_below_the_floor_is_a_no_to_that_number_then_the_next_step_never_below():
    out = play(["2000 mein karo"] * 9)
    assert out[0].offer is None and "₹2,000" in out[0].text["hinglish"]
    offered = [r.offer["price"] for r in out if r.offer]
    assert offered[0] == 2799 and all(x >= FLOOR for x in offered) and offered == sorted(offered, reverse=True)
    for r in out:
        for l in LANGS:
            assert all(x >= FLOOR or x in (2000, 33) for x in prices_in(r.text[l]))   # 33 = the per-day figure


def test_counter_between_current_offer_and_floor_is_met_in_the_middle():
    out = play(["bahut mehenga", "2500 mein karo"])
    assert out[1].offer == {"price": 2749.0} and "beech" in out[1].text["hinglish"] and "बीच" in out[1].text["hi"] and "middle" in out[1].text["en"]


def test_counter_at_or_above_our_current_offer_is_accepted_at_once_at_our_offer():
    out = play(["bahut mehenga", "kam karo", "2900 mein do"])
    assert out[2].offer == {"price": 2799.0} and "pakka" in out[2].text["hinglish"] and "?" in out[2].text["hinglish"]


def test_counter_at_or_above_the_next_step_or_after_the_last_step_is_accepted_at_their_price():
    out = play(["bahut mehenga", "kam karo", "2500 mein do", "2500 mein do", "2500 mein do", "2500 mein do"])
    prices = [r.offer["price"] for r in out if r.offer]
    assert prices[-1] == 2500.0 and "pakka" in out[-1].text["hinglish"]
    last = play(["kam karo"] * 6 + ["2430 mein do"])[-1]   # at the floor: a number at/above the floor is accepted
    assert last.offer == {"price": 2430.0}


def test_at_the_floor_final_once_then_one_sweetener_then_a_warm_walk_away_never_a_loss():
    out = play(["kam karo"] * 10)
    assert out[4].offer["price"] == 2449 and out[4].state.final_said
    assert out[5].offer and out[5].offer["price"] == 2449 and not out[5].state.closed        # the sweetener at the floor (a configured lever), with a closing ask
    assert out[6].state.closed and not out[6].offer and "?" in out[6].text["hinglish"]        # the warm walk-away offers a callback / details
    assert not any(r.offer for r in out[7:])
    for r in out:
        if r.offer:
            assert negotiation.margin_ok(GYM.economics, r.offer["price"], [r.offer["perk"]] if r.offer.get("perk") else [])


def test_closing_ask_varies_every_concession_has_one():
    out = play(["kam karo"] * 8)
    asks = [re.split(r"(?<=[.!])\s", r.text["hinglish"])[-1] for r in out[1:5]]
    assert len(set(asks)) == len(asks)


def test_info_and_comparison_questions_get_answers_not_concessions():
    r = play(["Cult 2,500 mein deta hai, aapka kyun lu"])[0]
    assert r.offer is None and "2,799" not in r.text["hinglish"] and "2,999" not in r.text["hinglish"]
    r = play(["ChatGPT bhi free hai, aapka kya better hai"])[0]
    assert r.offer is None and "2,799" not in r.text["hinglish"]


def test_no_cash_ladder_for_a_real_company_product():
    assert not negotiation.ladder_enabled(CLAUDE)
    out = play(["too expensive", "give me a discount", "reduce the price", "$5 do"], CLAUDE)
    for r in out:
        assert r.offer is None


def test_a_yes_after_a_concession_puts_the_agreed_price_in_terms_offer_and_ledger():
    import asyncio
    from tests.decision_helpers import run
    out, last, pipe, dom, _ = asyncio.run(run(["haan bolo", "kitne ka hai", "bahut mehenga hai", "aur kam karo", "haan theek hai, kar do"], GYM))
    d = pipe.sessions["s"].dialog if hasattr(pipe, "sessions") else {}
    assert out[-1] == "confirm_close"
    assert "₹2,799" in dom.p.terms["en"] and "₹2,799" in dom.p.offer_price and "₹2,999" not in dom.p.terms["en"].split(",")[0]
    assert float(d["closed_price"]) == 2799.0 and float(d["margin_pct"]) == pytest.approx((2799 - 2000) / 2799 * 100, abs=0.1)
    ns = negotiation.NegState.load(d)
    assert ns.agreed == 2799.0 and ns.price == 2799.0
    # a later push never retracts or raises it
    r = negotiation.step(dom.p, "aur kam karo", ns, "", None)
    assert r.offer is None and r.state.agreed == 2799.0


def _eco(rng):
    lst = rng.choice([49, 99, 299, 499, 999, 1499, 2999, 7999, 24999, 120000])
    cost = round(lst * rng.uniform(0.2, 0.95), 2)
    margin = rng.choice([0, 5, 10, 20, 35, 60])
    floor_price = rng.choice([0, 0, round(lst * rng.uniform(0.5, 0.99), 2)])
    perk = rng.choice([0, 0, round(lst * 0.02, 2), round(lst * 0.07, 2)])
    ladder = [Step(kind="perk", id="pk", text={l: f"free extra {l}" for l in LANGS}, short={l: "plus extra" for l in LANGS}, cost=perk)] if perk else []
    return Economics(currency="₹", unit="1 month", list_price=lst, cost=cost, min_margin_pct=margin, floor_price=floor_price, ladder=ladder)


@pytest.mark.parametrize("seed", range(60))
def test_profit_check_never_violated_across_random_economics(seed):
    rng = random.Random(seed * 7919 + 13)
    eco = _eco(rng)
    p = Product(id="p", product_name="P", seller_name="S", economics=eco, benefits=["b one", "b two"], benefits_hi=["अ", "ब"], benefits_hinglish=["a", "b"])
    pushes = [["mehenga hai"] * 12, ["kam karo"] * 12, [f"{int(eco.list_price * f)} mein do" for f in (0.4, 0.6, 0.8, 0.9, 0.95, 0.5, 0.7, 0.99, 0.3, 0.85)]]
    for tone in TONES:
        for lines in pushes:
            floor = eco.profit_floor
            seq = negotiation.ladder_prices(eco.list_price, floor)
            assert all(x >= floor for x in seq)
            lowest = eco.list_price
            for r in play(lines, p, tone):
                if r.offer:
                    price = r.offer["price"]
                    assert negotiation.margin_ok(eco, price, r.state.perks), (seed, price)   # price - cost - perk cost >= min margin
                    assert price <= lowest + 1e-9 or r.offer.get("perk") or price <= r.state.price + 1e-9
                    lowest = min(lowest, price)
                    assert r.state.price <= eco.list_price
                if r.state.price:
                    assert r.state.price >= floor - 1e-9
