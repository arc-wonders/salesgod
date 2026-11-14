"""Psyche needs / emotion read the fused TurnSignals (acie/signals.py); raw words are the fallback when there are none."""

import json

from acie.psyche import emotion_policy
from acie.psyche.moves import MoveContext
from acie.psyche.needs import NeedInput, resolve_signals, table_need
from acie.signals import TurnSignals


def sig(**kw):
    kw.setdefault("learned", True)
    return TurnSignals(**kw)


def test_resolve_accepts_signals_dialog_json_and_nothing():
    s = sig(buying_interest=0.7)
    assert resolve_signals(s) is s
    assert resolve_signals(None) is None
    dlg = {"_signals": s.model_dump_json()}
    assert resolve_signals(dlg).buying_interest == 0.7
    assert resolve_signals({}) is None
    assert resolve_signals(json.dumps({"x": 1})) is None  # not a signals payload: rules only


def test_need_input_reads_question_topic_state_and_intent_from_signals():
    s = sig(question_topic="offer_terms", state={"frustration": 0.8}, votes={"llm": "price"})
    n = NeedInput(text="ok", signals=s)
    assert n.question_topic == "offer_terms" and n.state["frustration"] == 0.8 and n.nlu_intent == "price"
    assert table_need(n).need == "be_heard"  # the fused frustration row, though the words are mild


def test_explicit_state_wins_over_signals_state():
    n = NeedInput(text="ok", state={"frustration": 0.1}, signals=sig(state={"frustration": 0.9}))
    assert n.state["frustration"] == 0.1


def test_objection_trust_and_language_move_the_need():
    assert table_need(NeedInput(text="hmm okay then", signals=sig(objection="trust"))).need == "reassurance"
    assert table_need(NeedInput(text="hmm okay then", signals=sig(objection="language"))).need == "clarity"


def test_learned_deny_and_callback_are_heard():
    assert table_need(NeedInput(text="hmm", signals=sig(votes={"llm": "deny"}))).need == "autonomy"
    assert table_need(NeedInput(text="hmm", signals=sig(votes={"model": "callback"}))).need == "autonomy"


def test_signals_absent_falls_back_to_raw_words():
    assert NeedInput(text="नहीं चाहिए").signals is None
    assert table_need(NeedInput(text="नहीं चाहिए")).need == "autonomy"
    assert table_need(NeedInput(text="what do you have?")).need == "answer"


def test_dialog_seam_loads_the_stored_signals():
    dlg = {"_signals": sig(objection="trust").model_dump_json()}
    assert table_need(NeedInput(text="hmm okay then", dialog=dlg)).need == "reassurance"


def test_closing_flag_needs_interest_and_nothing_open():
    ok = table_need(NeedInput(text="yes, sounds good", signals=sig(buying_interest=0.8)))
    assert ok.rule == "9:close" and ok.flags["closing"]
    assert not table_need(NeedInput(text="yes, sounds good", signals=sig(buying_interest=0.3))).flags["closing"]
    assert not table_need(NeedInput(text="yes, sounds good", signals=sig(buying_interest=0.8, objection="price"))).flags["closing"]
    assert not table_need(NeedInput(text="not interested", signals=sig(buying_interest=0.9))).flags["closing"]
    assert not table_need(NeedInput(text="yes, sounds good", state={"frustration": 0.5}, signals=sig(buying_interest=0.9))).flags["closing"]


def test_emotion_table_reads_signals_when_state_lacks_a_dim():
    assert emotion_policy.decide({}, "hello", signals=sig(state={"frustration": 0.8})).row == "angry"
    assert emotion_policy.decide({}, "hello", signals=sig(objection="trust")).row == "suspicious"
    assert emotion_policy.decide({}, "hello", signals=None).row == "calm"
    # a stored session.dialog works too
    assert emotion_policy.decide({}, "hello", signals={"_signals": sig(state={"confusion": 0.7}).model_dump_json()}).row == "confused"
    # explicit state beats the signals
    assert emotion_policy.decide({"frustration": 0.0}, "hello", signals=sig(state={"frustration": 0.8})).row == "calm"


def test_move_context_resolves_signals():
    n = table_need(NeedInput(text="hello"))
    mc = MoveContext(need=n, signals={"_signals": sig(state={"frustration": 0.6}).model_dump_json()})
    assert mc.signals is not None and mc.frustration >= 0.6
    assert MoveContext(need=n).signals is None
