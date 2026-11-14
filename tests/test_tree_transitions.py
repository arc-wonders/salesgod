import json
import subprocess
import sys
from pathlib import Path

import pytest

from acie.schemas import Fact, SessionState, TurnPlan, Utterance
from acie.signals import TurnSignals
from acie.tree import coverage_summary, enrich_plan, tree_hint
from acie.tree.rules import Ctx, STAGE_RULES, stage_for

ROOT = Path(__file__).resolve().parents[1]
P = json.loads((ROOT / "data/voice/products.json").read_text(encoding="utf-8"))
FZ, CP = P["fitzone-gym"], P["claude-pro"]

# the live call cac0d80ca4, caller turns paraphrased
CALL = ["Sure, but make it fast please.",
        "What do I get after joining? I am mostly into muscle building and power lifting.",
        "Mainly building muscle and power lifting. Do you have anything for specific competitions?",
        "Around four to five days a week. How many days should I come in?"]


def sess():
    return SessionState(session_id="s", customer_id="c")


def step(s, text, product=FZ, **sig):
    plan = TurnPlan(intent="answer", objective="x")
    new = enrich_plan(plan, s, TurnSignals(**sig), text, product)
    s.history.append(Utterance(role="customer", text=text))
    s.history.append(Utterance(role="agent", text="ok"))
    s.turn_index += 1
    return new


def test_stage_table_is_ordered_and_total():
    assert [r[0] for r in STAGE_RULES][-1] == "discover"
    base = dict(customer_turns=2, text="", intent="unknown", objection="none", buying=0.0, decision="undecided", promised_link=False,
                is_question=False, has_cluster=False, offering_ready=False)
    assert stage_for(Ctx(**base))[0] == "discover"
    assert stage_for(Ctx(**{**base, "customer_turns": 0}))[0] == "open"
    assert stage_for(Ctx(**{**base, "is_question": True}))[0] == "educate"
    assert stage_for(Ctx(**{**base, "offering_ready": True}))[0] == "recommend"
    assert stage_for(Ctx(**{**base, "objection": "trust"}))[0] == "objection"
    assert stage_for(Ctx(**{**base, "objection": "price", "text": "can you give it for 2500"}))[0] == "negotiate"
    assert stage_for(Ctx(**{**base, "buying": 0.8}))[0] == "close"
    assert stage_for(Ctx(**{**base, "decision": "accepted", "buying": 0.9}))[0] == "after_close"


def test_hint_stages_follow_the_call(monkeypatch):
    monkeypatch.setenv("ACIE_KB", "1")
    s = sess()
    h0 = tree_hint(s, TurnSignals(), CALL[0], FZ)
    assert h0.stage == "open" and h0.move == "ask_discovery" and h0.discovery_question
    step(s, CALL[0])
    h1 = tree_hint(s, TurnSignals(), CALL[1], FZ, commit=False)
    assert h1.stage == "educate"
    tags = [c["tag"] for c in h1.clusters]
    assert tags[0] == "powerlifting" and "muscle" in tags
    assert h1.entries and h1.entries[0].kind == "answer"
    h2 = tree_hint(s, TurnSignals(buying_interest=0.8), "ok send me the link", FZ, commit=False)
    assert h2.stage == "close" and h2.move == "close_ask" and not h2.entries
    h3 = tree_hint(s, TurnSignals(objection="price"), "I want it for 2500, can you lower the price?", FZ, commit=False)
    assert h3.stage == "negotiate" and h3.move == "handle_objection" and not h3.entries


def test_upsell_picks_powerlifting_coaching_for_the_live_call(monkeypatch):
    monkeypatch.setenv("ACIE_KB", "1")
    s = sess()
    for t in CALL[:2]:
        step(s, t)
    h = tree_hint(s, TurnSignals(), CALL[2], FZ, commit=False)
    assert h.offering is not None and h.offering.id == "powerlifting-coaching"
    assert "powerlifting" in h.offering.why and "muscle" in h.offering.why
    h = tree_hint(s, TurnSignals(), CALL[3], FZ, commit=False)
    assert h.offering.id == "powerlifting-coaching" and "four to five days a week" in h.offering.why
    # competition wording pulls in the prep block as the answer, the coaching program as the offering
    h = tree_hint(s, TurnSignals(), CALL[2], FZ, commit=False)
    assert "pl-meet-prep" in [e.id for e in h.entries]


def test_teach_then_pitch_prices_only_through_slots(monkeypatch):
    monkeypatch.setenv("ACIE_KB", "1")
    s = sess()
    step(s, CALL[0])
    plan = step(s, CALL[1])
    assert plan.context and "ANSWER:" in plan.context
    assert "₹" not in (plan.context or "") and "₹" not in (plan.reference or "")
    plan = step(s, CALL[2])
    bridged = any(n.startswith("kb_price_") for n in plan.facts) or "OFFER ONE THING" in (plan.context or "")
    # by the first strong turn the offering has been bridged exactly once
    assert coverage_summary(s)["offered"].get("powerlifting-coaching") == 1 or bridged
    for f in plan.facts.values():
        assert isinstance(f, Fact)


def test_never_pitch_unrelated_offerings(monkeypatch):
    monkeypatch.setenv("ACIE_KB", "1")
    s = sess()
    for t in ("hello", "I just want to lose weight, belly fat mostly", "how fast can I lose weight?", "and what diet should I follow?"):
        h = tree_hint(s, TurnSignals(), t, FZ, commit=False)
        step(s, t)
        if h.offering:
            assert h.offering.id in ("weight-loss-program", "nutrition-plan"), h.offering.id
    for t in ("hi", "it is for my wife", "she wants a women only batch and a lady trainer"):
        h = tree_hint(sess(), TurnSignals(), t, FZ, commit=False)
        assert h.offering is None or h.offering.id in ("womens-batch", "couple-plan")


def test_no_pitch_when_frustrated_or_declining(monkeypatch):
    s = sess()
    for t in CALL[:2]:
        step(s, t)
    h = tree_hint(s, TurnSignals(state={"frustration": 0.8}), CALL[2], FZ, commit=False)
    assert h.offering is None and h.move != "bridge_offering"
    h = tree_hint(s, TurnSignals(intent="deny"), "not interested thanks", FZ, commit=False)
    assert h.offering is None


def test_coverage_never_resuggests_given_entries(monkeypatch):
    monkeypatch.setenv("ACIE_KB", "1")
    s = sess()
    seen: list[str] = []
    for t in CALL:
        h = tree_hint(s, TurnSignals(), t, FZ)
        step_plan = enrich_plan(TurnPlan(intent="answer", objective="x"), s, TurnSignals(), t, FZ)
        picked = [e.id for e in h.entries]
        assert not set(picked) & set(seen), (t, picked, seen)
        used = coverage_summary(s)["given_ids"]
        seen = list(used)
        s.history.append(Utterance(role="customer", text=t))
        s.history.append(Utterance(role="agent", text="ok"))
        s.turn_index += 1
    cov = coverage_summary(s)
    assert cov["given"] >= 5 and cov["kb_total"] == 160 and cov["asked"] >= 2 and cov["topics_explored"]


def test_agent_lines_already_spoken_count_as_given(monkeypatch):
    s = sess()
    s.history.append(Utterance(role="customer", text="what about powerlifting"))
    s.history.append(Utterance(role="agent", text="Yes. Powerlifting is a core use of our strength zone: platforms, racks, calibrated plates and "
                                                  "competition bars, a coach-led powerlifting program, a monthly technique clinic and an 8-week prep block before meets."))
    h = tree_hint(s, TurnSignals(), "tell me about powerlifting", FZ, commit=False)
    assert "pl-overview" not in [e.id for e in h.entries]
    assert any("support powerlifting" in g for g in h.given)


def test_coverage_text_is_compact(monkeypatch):
    monkeypatch.setenv("ACIE_KB", "1")
    s = sess()
    for t in CALL * 3:
        step(s, t)
    h = tree_hint(s, TurnSignals(), CALL[2], FZ)
    assert len(h.coverage_text()) <= 700


def test_flag_off_and_protected_plans_untouched(monkeypatch):
    s = sess()
    plan = TurnPlan(intent="answer", objective="x")
    monkeypatch.delenv("ACIE_KB", raising=False)
    assert enrich_plan(plan, s, TurnSignals(), CALL[1], FZ) is plan
    monkeypatch.setenv("ACIE_KB", "1")
    locked = TurnPlan(intent="price", objective="x", llm_allowed=False)
    assert enrich_plan(locked, s, TurnSignals(), CALL[1], FZ) is locked
    ended = TurnPlan(intent="declined", objective="x", end_call=True)
    assert enrich_plan(ended, s, TurnSignals(), CALL[1], FZ) is ended
    assert enrich_plan(plan, s, TurnSignals(), CALL[1], {"id": "no-such-product"}) is plan


def test_claude_pro_tree_has_no_synthetic_and_no_invented_pitch(monkeypatch):
    monkeypatch.setenv("ACIE_KB", "1")
    s = sess()
    plan = None
    for t in ("hi", "I write code all day, can it help me debug and with DSA?", "and can my team share one account?"):
        plan = step(s, t, CP)
    from acie.knowledge.store import load_kb
    kb = load_kb(CP)
    given = set(coverage_summary(s)["given_ids"])
    ents = kb.entry_map()
    assert given and all(ents[g].provenance == "public" for g in given)
    h = tree_hint(s, TurnSignals(), "can my team share one account", CP, commit=False)
    assert h.offering is None or h.offering.id in ("team-plan", "yearly-billing")


def test_deterministic_across_processes():
    code = ("import json,sys;sys.path.insert(0,'.');from acie.tree import tree_hint,enrich_plan;from acie.schemas import *;from acie.signals import TurnSignals;"
            "import os;os.environ['ACIE_KB']='1';P=json.load(open('data/voice/products.json'))['fitzone-gym'];"
            "s=SessionState(session_id='s',customer_id='c');out=[]\n"
            "for t in ['hi','I am into muscle and power lifting','do you have competition prep','four to five days a week']:\n"
            "    p=enrich_plan(TurnPlan(intent='answer',objective='x'),s,TurnSignals(),t,P);h=tree_hint(s,TurnSignals(),t,P,commit=False)\n"
            "    out.append([h.stage,h.move,[e.id for e in h.entries],h.offering and h.offering.id,p.context,p.reference,sorted(p.facts)])\n"
            "    s.history.append(Utterance(role='customer',text=t));s.turn_index+=1\n"
            "print(json.dumps(out,sort_keys=True))")
    outs = {subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, env={"PYTHONHASHSEED": str(x), "PATH": "/usr/bin"}).stdout
            for x in (11, 22)}
    assert len(outs) == 1 and next(iter(outs)).strip()


def test_signals_default_to_the_session_copy(monkeypatch):
    s = sess()
    s.dialog["_signals"] = TurnSignals(objection="price", buying_interest=0.1).model_dump_json()
    h = tree_hint(s, None, "can you lower the price for me, I want it for 2500", FZ, commit=False)
    assert h.stage == "negotiate"
