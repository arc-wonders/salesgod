"""Replay of the real calls (skipped when data/real is absent: the calls are private and git-ignored)."""

import asyncio
import os
import sys

import pytest

sys.path.insert(0, "scripts/real_calls")
if not os.path.isdir("data/real/calls") or not os.listdir("data/real/calls"):
    pytest.skip("no real calls on this machine", allow_module_level=True)

import replay  # noqa: E402
from common import count_repetitions, load_calls  # noqa: E402


def test_replay_is_deterministic_and_has_no_repetition():
    calls, prods = load_calls()[:12], replay.products()
    a = asyncio.run(replay.run_mode(calls, prods, True))
    b = asyncio.run(replay.run_mode(calls, prods, True))
    os.environ["ACIE_PSYCHE"] = ""
    assert a == b
    assert sum(count_repetitions(v) for v in a.values()) == 0
