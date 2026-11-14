"""Asking for a lower price / another offer is negotiation, never a decline, even with nahi / नहीं inside."""

import pytest

from tests.test_agnostic_signals import new_pipe, play

ASKS = [
    "और सस्ता नहीं हो सकता", "और कम नहीं होगा", "kam nahi ho sakta", "thoda kam karo na", "discount nahi milega",
    "can't you do it cheaper", "koi aur offer nahi hai?", "सस्ता नहीं हो सकता क्या?", "और सस्ता नहीं मिलेगा?",
    "कोई और ऑफर नहीं है?", "can you not do it any cheaper?", "sasta nahi ho sakta kya", "can't you give a discount",
]
REFUSALS = ["नहीं चाहिए", "nahi chahiye", "I don't want it", "not interested", "मुझे नहीं चाहिए"]


@pytest.mark.parametrize("line", ASKS)
async def test_price_request_is_price_floor_and_call_continues(line):
    out = await play(new_pipe(), ["हाँ बोलिए", "कितने का है", line])
    assert out[-1][0] == "price_floor", out
    assert "bye" not in out[-1][1].lower()


@pytest.mark.parametrize("line", REFUSALS)
async def test_real_no_is_still_a_decline(line):
    pipe = new_pipe()
    out = await play(pipe, ["हाँ बोलिए", "कितने का है", line])
    assert out[-1][0] in ("declined", "reoffer"), out


@pytest.mark.parametrize("line", ASKS[:4])
async def test_needs_table_is_not_autonomy(line, monkeypatch):
    from acie.psyche.needs import NeedInput, table_need
    r = table_need(NeedInput(text=line, intent="deny"))
    assert r.need != "autonomy"
