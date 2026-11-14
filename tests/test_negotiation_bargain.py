"""Indian-household bargaining in the close-in-profit ladder: accept at the floor or above, meet halfway, never below the floor,
a new line every round, one final-price line, one last honest sweetener, then a warm goodbye."""
import json
import re
from difflib import SequenceMatcher

import pytest

from acie.decision import negotiation
from acie.decision.domains.product_sales import Product

PRODS = {k: Product.model_validate(v) for k, v in json.load(open("data/voice/products.json")).items()}
GYM, CLAUDE = PRODS["fitzone-gym"], PRODS["claude-pro"]
FLOOR = GYM.economics.profit_floor
LANGS = ("en", "hi", "hinglish")
FINAL = re.compile(r"final price|आख़िरी कीमत|aakhri keemat", re.I)
BAD = re.compile(r"listening|sun raha|सुन रहा|team (will|se)|call you back soon|टीम", re.I)


def play(lines, p=GYM, **kw):
    st, out = negotiation.NegState(), []
    for t in lines:
        r = negotiation.step(p, t, st, "", None, **kw)
        st = r.state
        out.append(r)
    return out


def prices(r):
    return [int(x.replace(",", "")) for x in re.findall(r"₹([\d,]+)", r.text["en"])]


@pytest.mark.parametrize("line", ["2500 mein de do", "₹2,500 pe done?", "2500 में दे दो", "give it for 2500"])
def test_counter_at_or_above_floor_closes_at_once_on_the_first_push(line):
    r = play([line])[0]
    assert r.offer == {"price": 2500.0}
    for l in LANGS:
        assert "2,500" in r.text[l] and "?" in r.text[l]
    assert r.state.price == 2500.0


@pytest.mark.parametrize("counter", ["1000", "0", "2000", "2399", "500", "100"])
def test_never_below_the_floor_for_any_counter(counter):
    for r in play([f"{counter} mein do"] * 9):
        if r.offer:
            assert r.offer["price"] >= FLOOR
        for l in LANGS:
            assert all(x >= FLOOR for x in [int(y.replace(",", "")) for y in re.findall(r"₹([\d,]+)", r.text[l])])


def test_counter_below_current_meets_halfway_with_friendly_rounding():
    out = play(["2350 mein do"] * 3)  # perk, perk, then the middle of 2999 and 2350 = 2674 -> 2650
    assert out[2].offer == {"price": 2650.0}
    assert "beech" in out[2].text["hinglish"] and "बीच" in out[2].text["hi"] and "middle" in out[2].text["en"]
    assert out[2].offer["price"] % 50 == 0 and out[2].offer["price"] >= FLOOR


def test_vague_push_asks_their_number_once_then_concedes():
    out = play(["thoda kam karo", "aur kam", "aur kam"])
    assert "?" in out[0].text["hinglish"] and not out[0].offer and "mann mein" in out[0].text["hinglish"]
    assert out[1].offer and out[1].offer.get("perk") == "extra_week"
    assert "mann mein kitna" not in out[1].text["hinglish"] + out[2].text["hinglish"]


def test_committed_buyer_is_just_asked_for_their_number():
    r = play(["thoda kam karo"], committed=True)[0]
    assert "kitna" in r.text["hinglish"] and "₹2,999" not in r.text["hinglish"]


def test_value_before_cash_shrinking_steps_and_commitment_wording():
    out = play(["kam karo"] * 6)
    assert out[1].offer["perk"] == "extra_week" and out[2].offer["perk"] == "pt_sessions"
    assert [o.offer["price"] for o in out[3:5]] == [2800, 2700]
    assert "free" in out[1].text["en"] and "फ्री" in out[1].text["hi"] and "free" in out[1].text["hinglish"]
    for l in LANGS:
        assert re.search(r"now|today|अभी|आज|abhi|aaj|ready", out[3].text[l], re.I)
        assert not re.search(r"last chance|only \d+ left|hurry|expires", out[3].text[l], re.I)


@pytest.mark.parametrize("lang", LANGS)
def test_eight_round_haggle_never_repeats_one_final_price_line_then_warm_goodbye(lang):
    lines = {"en": "please reduce the price more", "hi": "और कम करो ना", "hinglish": "aur kam karo na"}[lang]
    out = play([lines] * 9)
    texts = [r.text[lang] for r in out]
    for a, b in zip(texts, texts[1:]):
        assert SequenceMatcher(None, a, b).ratio() <= 0.6, (a, b)
    assert sum(1 for t in texts if FINAL.search(t)) == 1
    assert not any(BAD.search(t) for t in texts)
    assert out[-1].state.closed or out[-2].state.closed
    assert all(prices(r)[0] >= FLOOR for r in out if prices(r))
    assert out[-1].text == negotiation.CLOSE_WARM[0] or out[-1].text == negotiation.CLOSE_WARM[1]
    assert "?" in out[-2].text[lang] or out[-2].state.closed


def test_after_the_floor_one_last_new_offer_then_goodbye():
    out = play(["kam karo"] * 10)
    kinds = [("sweet" if r.state.sweet else "") for r in out]
    first_sweet = kinds.index("sweet")
    assert "referral" in out[first_sweet].state.levers or out[first_sweet].state.levers
    assert not out[first_sweet].state.closed and out[first_sweet + 1].state.closed
    assert "dost" in out[first_sweet].text["hinglish"] or "friend" in out[first_sweet].text["en"] or out[first_sweet].state.levers


def test_number_at_or_above_floor_after_the_floor_is_still_accepted():
    out = play(["kam karo"] * 7 + ["2500 mein de do"])
    assert out[-1].offer == {"price": 2500.0}


def test_closing_ask_varies_never_the_same_form_twice_in_a_row():
    out = play(["kam karo"] * 8)
    asks = [r.state.ask_i for r in out if r.offer]
    assert len(set(asks)) >= 3 and all(a != b for a, b in zip(asks, asks[1:]))
    for l in LANGS:
        assert len({a[l] for a in negotiation.ASKS if False} | {negotiation.ASKS[i][LANGS.index(l)] for i in set(asks)}) >= 3


def test_no_invented_perks_or_history():
    out = play(["kam karo"] * 10)
    allowed = " ".join(x.text.get("en", "") for x in GYM.economics.ladder if x.kind == "perk") + " ".join(v["en"] for v in GYM.negotiation.levers.values())
    for r in out:
        t = r.text["en"]
        assert not re.search(r"loyal|valued customer|regular|as always|last time|previous", t, re.I)
    assert "week" in allowed


def test_real_company_without_a_ladder_keeps_the_no_invented_discount_path():
    r = negotiation.step(CLAUDE, "can you reduce it", negotiation.NegState(), "", None)
    assert r.intent == "price_floor" and not r.offer
    assert negotiation.FLOOR[0]["en"].split("{")[0] in r.text["en"]
