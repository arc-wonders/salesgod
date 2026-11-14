"""Deterministic answer extraction: yes/no, numbers, money, dates and times, choices, free text (en / hi / hinglish)."""

import pytest

from acie.calltypes.extract import UNKNOWN, choice, date_time, extract, find_numbers, free_text, money, number, step_keyword_hit, yes_no
from acie.calltypes.schema import Step
from tests.calltypes_helpers import NOW  # Sunday 15 Nov 2026, 10:00 IST


@pytest.mark.parametrize("text,want", [
    ("yes please", True), ("yeah sure", True), ("haan bilkul", True), ("हाँ बिल्कुल", True), ("जी हाँ", True), ("theek hai", True),
    ("no", False), ("nope not really", False), ("nahi", False), ("नहीं चाहिए", False), ("nahi chahiye", False),
    ("koi baat nahi", True), ("no problem", True), ("why not", True), ("kyun nahi", True),
    ("theek hai na", True),
])
def test_yes_no(text, want):
    assert yes_no(text) == (True, want)


@pytest.mark.parametrize("text", ["hmm", "kya matlab", "haan lekin abhi nahi bata sakta", "pata nahi", "I don't know", "पता नहीं"])
def test_yes_no_unclear(text):
    assert yes_no(text)[0] is False


@pytest.mark.parametrize("text,want", [
    ("das hazaar", 10000), ("mera budget 10 hazaar hai", 10000), ("5 lakh", 500000), ("dedh lakh", 150000), ("50k rupaye", 50000),
    ("fifty thousand", 50000), ("पचास हजार", 50000), ("दस हज़ार रुपये", 10000), ("ek lakh pachas hazaar", 150000), ("₹5,000", 5000), ("1.5 lakh", 150000),
    ("around 20000", 20000),
])
def test_money(text, want):
    assert money(text) == (True, want)


@pytest.mark.parametrize("text", ["pata nahi", "abhi nahi bata sakta", "no budget decided", "hello"])
def test_money_not_found(text):
    assert money(text)[0] is False


@pytest.mark.parametrize("text,want", [("paanch log hain", 5), ("we are 12 people", 12), ("do", 2), ("तीन", 3), ("pachees", 25), ("4 out of 5", 4)])
def test_number(text, want):
    assert number(text) == (True, want)


def test_number_ignores_give_do_and_identifiers():
    assert number("bata do")[0] is False
    assert find_numbers("mera number 9876543210 hai") == []


@pytest.mark.parametrize("text,iso", [
    ("kal shaam 6 baje", "2026-11-16T18:00:00+05:30"),
    ("tomorrow at 4pm", "2026-11-16T16:00:00+05:30"),
    ("कल शाम 6 बजे", "2026-11-16T18:00:00+05:30"),
    ("परसों सुबह 10 बजे", "2026-11-17T10:00:00+05:30"),
    ("Monday morning", "2026-11-16T09:00:00+05:30"),
    ("aaj raat 8 baje", "2026-11-15T20:00:00+05:30"),
    ("in 2 hours", "2026-11-15T12:00:00+05:30"),
    ("do ghante baad", "2026-11-15T12:00:00+05:30"),
    ("saade chaar baje kal", "2026-11-16T16:30:00+05:30"),
    ("dedh baje parso", "2026-11-17T13:30:00+05:30"),
    ("5 baje", "2026-11-15T17:00:00+05:30"),
    ("kal", "2026-11-16T11:00:00+05:30"),
    ("kal dopahar", "2026-11-16T13:00:00+05:30"),
    ("Friday 3:30 pm", "2026-11-20T15:30:00+05:30"),
])
def test_date_time(text, iso):
    assert date_time(text, NOW) == (True, iso)


@pytest.mark.parametrize("text", ["abhi nahi", "hello", "kya hai ye", "maybe"])
def test_date_time_not_found(text):
    assert date_time(text, NOW)[0] is False


def test_extract_dispatch_and_declined_dates():
    s = lambda t, **kw: Step(id="x", text="q", answer_type=t, **kw)
    assert extract(s("yes_no"), "haan") == (True, True)
    assert extract(s("date_time"), "nahi abhi nahi", NOW) == (True, "declined")
    assert extract(s("date_time"), "pata nahi", NOW) == (True, UNKNOWN)
    assert extract(s("money"), "pata nahi") == (True, UNKNOWN)  # "don't know" is an answer: never asked again
    assert extract(s("money"), "hello there") == (False, None)  # not found: asked again


def test_choice_longest_match_and_unknown():
    st = Step(id="c", text="q", answer_type="choice", choices=[
        {"id": "yes", "keywords": ["interested"]}, {"id": "no", "keywords": ["not interested"]}, {"id": "later", "keywords": ["baad mein", "abhi nahi"]}])
    assert choice(st, "I am not interested") == (True, "no")
    assert choice(st, "yes I am interested") == (True, "yes")
    assert choice(st, "abhi nahi, baad mein dekhte hain") == (True, "later")
    assert choice(st, "pata nahi") == (True, UNKNOWN)
    assert choice(st, "kuch aur") == (False, None)


def test_free_text_rules():
    assert free_text("mujhe CRM chahiye sales team ke liye") == (True, "mujhe CRM chahiye sales team ke liye")
    assert free_text("kya aap bata sakte hain") == (False, None)
    assert free_text("haan") == (False, None)
    assert free_text("pata nahi") == (True, UNKNOWN)


def test_step_keywords():
    st = Step(id="b", text="q", answer_type="money", keywords=["budget", "kharcha"])
    assert step_keyword_hit(st, "mera budget 10 hazaar hai")
    assert not step_keyword_hit(st, "10 hazaar")


def test_determinism():
    for t in ["kal shaam 6 baje", "das hazaar", "haan bilkul", "next week"]:
        assert date_time(t, NOW) == date_time(t, NOW) and money(t) == money(t) and yes_no(t) == yes_no(t)


def test_weak_choice_keywords_lose_to_strong_ones():
    st = Step(id="c", text="q", answer_type="choice", choices=[
        {"id": "go", "keywords": ["~haan", "~chahiye", "lena hai"]}, {"id": "wait", "keywords": ["sochna", "thoda time"]}])
    assert choice(st, "haan sochna padega") == (True, "wait")
    assert choice(st, "mujhe thoda time chahiye") == (True, "wait")
    assert choice(st, "haan") == (True, "go")
    assert choice(st, "lena hai") == (True, "go")
