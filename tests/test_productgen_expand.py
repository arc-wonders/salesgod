import copy
import json
from pathlib import Path

import pytest

from acie.productgen import expand as X
from acie.productgen import kbops as K
from acie.productgen import store as S
from acie.productgen.generate import generate
from acie.productgen.spec import Spec
from test_productgen_fake import FakeLLM

PID = "zenbrew-coffee-subscription"
SPEC = dict(name="ZenBrew Coffee Subscription", price={"list_price": 1200, "offer_price": 999, "unit": "per month"}, economics={"floor": 600})


async def _v1(tmp_path, **kw):
    return await generate(Spec.model_validate({**SPEC, **kw}), FakeLLM(), tmp_path)


def _call(cid, turns, campaign=PID):
    out = []
    for q, a, intent in turns:
        out.append({"role": "customer", "text": q})
        out.append({"role": "agent", "text": a, "intent": intent})
    return {"call_id": cid, "campaign": campaign, "turns": out}


def _write_calls(d: Path):
    d.mkdir(parents=True, exist_ok=True)
    IDK = "I don't have that detail, the website will have it."
    calls = [
        _call("c1", [("do you deliver to goa pincode", IDK, "price"), ("ok", "Great", "affirm")]),
        _call("c2", [("can you deliver to goa", "Sorry, no idea.", "unknown")]),
        _call("c3", [("goa delivery possible for you", "ok", "benefits")]),          # no fallback signal: only counts if KB scores low
        _call("c4", [("is the packaging recyclable plastic", "mere paas ye detail nahi hai", "benefits")]),
        _call("c5", [("What about item 1 in topic-2?", IDK, "price")]),               # the KB answers this one confidently: not a gap
        _call("c6", [("does it ship to goa", IDK, "price")], campaign="some-other-product"),
    ]
    for c in calls:
        (d / f"{c['call_id']}.json").write_text(json.dumps(c))
    (d / "broken.json").write_text("{not json")


def _kb(doc):
    from acie.knowledge.schema import KnowledgeBase
    return KnowledgeBase.model_validate(doc["bundle"]["kb"])


async def test_gap_mining_clusters_deterministically(tmp_path):
    doc = await _v1(tmp_path)
    calls = tmp_path / "calls"
    _write_calls(calls)
    kb = _kb(doc)
    g1 = X.mine_gaps(calls, PID, kb, doc["bundle"]["product"])
    g2 = X.mine_gaps(calls, PID, kb, doc["bundle"]["product"])
    assert g1 == g2
    qs = [set(g["questions"]) for g in g1]
    goa = next(g for g in g1 if "do you deliver to goa pincode" in g["questions"])
    assert goa["count"] >= 2 and {"c1", "c2"} <= set(goa["call_ids"]) and "c6" not in goa["call_ids"]
    assert any("is the packaging recyclable plastic" in q for q in qs)
    assert not any("What about item 1 in topic-2?" in q for q in qs)  # the KB confidently answers it
    assert g1 == sorted(g1, key=lambda g: (-g["count"], g["id"]))
    assert X.mine_gaps(calls, PID, kb, doc["bundle"]["product"], skip={goa["id"]}) != g1


async def test_gap_proposals_queue_as_pending_version_then_approve_and_publish(tmp_path, monkeypatch):
    monkeypatch.delenv("ACIE_PRODUCTGEN_AUTO_APPROVE", raising=False)
    doc = await _v1(tmp_path)
    calls = tmp_path / "calls"
    _write_calls(calls)
    kb_root = tmp_path / "kb"
    S.publish(tmp_path, "default", PID, 1, kb_root)
    n_entries = len(doc["bundle"]["kb"]["entries"])
    pend = await X.propose_gaps(tmp_path, "default", PID, FakeLLM(), calls, kb_root)
    assert pend["kind"] == "gap_proposals" and pend["status"] == "pending" and pend["bundle"] is None
    assert pend["version"] == 2 and pend["proposals"] and all(p["status"] == "pending" for p in pend["proposals"])
    p0 = pend["proposals"][0]
    assert p0["call_ids"] and p0["entries"][0]["provenance"] == "synthetic" and "gap:" in p0["entries"][0]["source"]
    # nothing is live yet
    assert len(json.loads((kb_root / "default" / f"{PID}.json").read_text())["entries"]) == n_entries
    # a pending version can't be published
    with pytest.raises(S.PublishError):
        S.publish(tmp_path, "default", PID, 2, kb_root)
    # the same questions are not proposed twice
    assert await X.propose_gaps(tmp_path, "default", PID, FakeLLM(), calls, kb_root) is None

    ids = [p["id"] for p in pend["proposals"]]
    out = X.approve_gaps(tmp_path, "default", PID, 2, ids[:1], ids[1:], publish=True, kb_root=kb_root)
    assert out["approved"] == ids[:1] and out["version"] == 3 and out["published"]["ok"]
    live = json.loads((kb_root / "default" / f"{PID}.json").read_text())
    assert len(live["entries"]) == n_entries + len(p0["entries"])
    assert any(e["source"].startswith("gap:") or "gap:" in e["source"] for e in live["entries"])
    st = S.state(tmp_path, "default", PID)["gap_seen"]
    assert st[ids[0]] == "approved" and all(st[i] == "rejected" for i in ids[1:])
    # approving again does nothing
    again = X.approve_gaps(tmp_path, "default", PID, 2, ids[:1], [], publish=False, kb_root=kb_root)
    assert again["approved"] == []


async def test_auto_approve_only_for_synthetic_products(tmp_path, monkeypatch):
    monkeypatch.setenv("ACIE_PRODUCTGEN_AUTO_APPROVE", "1")
    doc = await _v1(tmp_path)
    calls = tmp_path / "calls"
    _write_calls(calls)
    kb_root = tmp_path / "kb"
    pend = await X.propose_gaps(tmp_path, "default", PID, FakeLLM(), calls, kb_root)
    assert pend["auto_approved"]["published"]["ok"]
    assert (kb_root / "default" / f"{PID}.json").exists()

    # a real-brand product is never auto-published
    rid = "claude-pro"
    rd = await generate(Spec.model_validate({"name": "Claude Pro"}), FakeLLM(), tmp_path)
    rc = tmp_path / "rcalls"
    rc.mkdir()
    (rc / "r1.json").write_text(json.dumps(_call("r1", [("does it support voice dictation mode", "I don't have that detail", "price")], campaign=rid)))
    (rc / "r2.json").write_text(json.dumps(_call("r2", [("is dictation voice mode supported", "I don't have that detail", "price")], campaign=rid)))
    rp = await X.propose_gaps(tmp_path, "default", rid, FakeLLM(), rc, kb_root)
    assert rp is not None and "auto_approved" not in rp and rp["status"] == "pending"
    assert not (kb_root / "default" / f"{rid}.json").exists()
    assert all("review" in e["source"] and e["provenance"] == "public" for p in rp["proposals"] for e in p["entries"])


async def test_expand_topic_adds_children_and_touches_nothing_else(tmp_path):
    d1 = await _v1(tmp_path)
    out = await X.expand_node(tmp_path, "default", PID, FakeLLM(variant="x"), {"kind": "topic", "id": "topic-4"})
    assert out["version"] == 2 and out["base_version"] == 1 and out["kind"] == "expand"
    old, new = d1["bundle"]["kb"], out["bundle"]["kb"]
    om = {e["id"]: e for e in old["entries"]}
    nm = {e["id"]: e for e in new["entries"]}
    assert all(nm[k] == v for k, v in om.items())  # every old entry untouched
    added = set(nm) - set(om)
    assert len(added) == 6 and all(nm[i]["topic_path"] == "topic-4/new-a" for i in added)  # the stray entry for topic-9 was refused
    assert "topic-4/new-a" in {t["path"] for t in new["topics"]} and "elsewhere/x" not in {t["path"] for t in new["topics"]}
    assert old["topics"] == [t for t in new["topics"] if t["path"] != "topic-4/new-a"]
    assert out["bundle"]["tree"]["nodes"] == d1["bundle"]["tree"]["nodes"]
    diff = json.loads((tmp_path / f"productgen/default/{PID}/v2.diff.json").read_text())["diff"]
    assert sorted(diff["entries"]["added"]) == sorted(added) and not diff["entries"]["removed"] and not diff["entries"]["changed"]


async def test_expand_playbook_node_adds_branches(tmp_path):
    d1 = await _v1(tmp_path)
    out = await X.expand_node(tmp_path, "default", PID, FakeLLM(), {"kind": "playbook", "id": "sit-2"})
    nodes = {n["id"]: n for n in out["bundle"]["tree"]["nodes"]}
    assert [c["id"] for c in nodes["sit-2"]["children"]] == ["sit-2-kid-0", "sit-2-kid-1", "sit-2-kid-2"]
    for n in d1["bundle"]["tree"]["nodes"]:
        if n["id"] != "sit-2":
            assert nodes[n["id"]] == n
    assert out["bundle"]["kb"] == d1["bundle"]["kb"]
    # a second expansion of the same node adds only new ids
    out2 = await X.expand_node(tmp_path, "default", PID, FakeLLM(), {"kind": "playbook", "id": "sit-2"})
    assert len(next(n for n in out2["bundle"]["tree"]["nodes"] if n["id"] == "sit-2")["children"]) == 3
    with pytest.raises(KeyError):
        await X.expand_node(tmp_path, "default", PID, FakeLLM(), {"kind": "playbook", "id": "nope"})
    with pytest.raises(ValueError):
        await X.expand_node(tmp_path, "default", PID, FakeLLM(), {"kind": "weird", "id": "x"})


def test_assert_untouched_detects_edits():
    kb = {"entries": [{"id": "a", "v": 1}], "topics": [], "offerings": [], "interests": []}
    b = {"kb": kb, "tree": {"nodes": []}}
    before = X._snapshot(b)
    b2 = copy.deepcopy(b)
    b2["kb"]["entries"][0]["v"] = 2
    with pytest.raises(AssertionError):
        X._assert_untouched(before, X._snapshot(b2))


def test_kbops_cleaning():
    kb = {"topics": [], "entries": [], "offerings": [], "interests": []}
    notes = []
    from acie.productgen.spec import Spec, resolve
    eff = resolve(Spec(name="Some Product", economics={"floor": 500}))
    K.add_topics(kb, [{"path": "A/b", "title": "x"}, {"path": "A", "title": {"en": "A"}}, {"path": "orphan/child"}], notes)
    assert [t["path"] for t in kb["topics"]] == ["a", "a/b"] and any("parent missing" in n for n in notes)
    K.add_interests(kb, [{"id": "t1", "keywords": ["k"], "related": ["t1", "zzz"]}])
    assert kb["interests"][0]["related"] == []
    K.add_offerings(kb, [{"id": "o", "name": {"en": "O"}, "who_suits": ["t1"], "price_inr": 100}, {"id": "bad", "name": {"en": "B"}, "who_suits": ["zz"]}], Spec(name="Some Product", economics={"floor": 500}), eff, notes)
    assert kb["offerings"][0]["price_inr"] == 500 and kb["offerings"][0]["price_label"]["en"] and len(kb["offerings"]) == 1
    ids = K.add_entries(kb, [{"id": "E 1", "topic_path": "a", "question_variants": {"en": ["q?"]}, "answer": {"en": "ans"}, "provenance": "operator"},
                             {"id": "e2", "topic_path": "nope", "question_variants": ["q"], "answer": "a"},
                             {"id": "e3", "topic_path": "a", "question_variants": ["q?"], "answer": "dup question"}], eff, notes)
    assert ids == ["e-1"] and kb["entries"][0]["provenance"] == "synthetic"  # provenance is forced, never trusted
    assert K.money(150000) == "₹1,50,000" and K.money(2999) == "₹2,999"
