import json
import types

from decision_helpers import run

from acie import playbook as pbm
from acie.decision import nba
from acie.decision.domains.product_sales import Product
from acie.understanding import Understanding

TREE = {
    "product_id": "claude-pro", "nodes": [
        {"id": "price", "title": "price worry", "when": {"keywords": ["expensive", "mehnga"]},
         "points": ["it is a discount"], "ask": "what do you use?",
         "children": [
             {"id": "price_student", "title": "student price worry", "when": {"keywords": ["student"]},
              "points": ["same offer for all"]}]},
        {"id": "trust", "title": "scam", "when": {"keywords": ["scam"]}, "points": ["secure link"]}]}


def test_tree_child_beats_parent_only_when_it_matches(tmp_path):
    f = tmp_path / "claude-pro.tree.json"
    f.write_text(json.dumps(TREE))
    pb = pbm.load_tree(f)
    assert pb.pick("it is too expensive")[0].id == "price"
    assert pb.pick("I am a student and it is expensive")[0].id == "price_student"
    assert pb.pick("is this a scam")[0].id == "trust"
    assert pb.pick("hello there")[0] is None


def test_load_prefers_tree_file_per_product(tmp_path):
    d = tmp_path / "playbooks" / "default"
    d.mkdir(parents=True)
    (d / "p1.json").write_text(json.dumps({"product_id": "p1", "nodes": [{"id": "flat", "title": "t"}]}))
    assert pbm.load(tmp_path, "default", "p1").nodes[0].id == "flat"
    (d / "p1.tree.json").write_text(json.dumps({"product_id": "p1", "nodes": [{"id": "tree", "title": "t"}]}))
    assert pbm.load(tmp_path, "default", "p1").nodes[0].id == "tree"


def test_seed_playbooks_still_load_and_flat_pick_unchanged():
    pb = pbm.load(pbm.SEED.parent / "data", "default", "claude-pro")
    if pb is not None:
        assert all(isinstance(n.children, list) for n in pb.nodes)


async def test_product_playbook_tree_path_is_used(tmp_path):
    f = tmp_path / "t.json"
    f.write_text(json.dumps(TREE))
    _, _, _, dom, _ = await run([], Product(playbook_tree=str(f)))
    assert dom.playbook is not None and dom.playbook.nodes[0].id == "price"
    _, _, _, dom, _ = await run([], Product(playbook_tree=str(tmp_path / "missing.json")))
    assert dom.playbook is None


def ctx(**kw):
    c = types.SimpleNamespace(quoted=[], told=[], promised=[], objections=[], decision="undecided",
                              has=lambda x: x in c.promised)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def test_nba_formula_cases():
    a, _ = nba.next_best_action("intro", 0.0, [], ctx())
    assert a == "pitch"
    a, _ = nba.next_best_action("pitch", 0.2, [], ctx(told=["what_the_product_is"]))
    assert a == "quote_price"
    # an unresolved objection is answered before anything else
    a, s = nba.next_best_action("close", 0.3, ["price"], ctx(quoted=["offer_price"]))
    assert a == "answer_objection" and s["answer_objection"] > s["confirm_close"]
    # strong buying signal after the price: close
    a, _ = nba.next_best_action("close", 0.95, [], ctx(quoted=["offer_price"]))
    assert a == "confirm_close"
    # never close before the price was quoted
    _, s = nba.next_best_action("close", 1.0, [], ctx())
    assert s["confirm_close"] < 0
    # after a decision nothing but a soft exit
    a, _ = nba.next_best_action("close", 1.0, [], ctx(decision="declined", quoted=["offer_price"]))
    assert a == "soft_exit"


async def test_nba_is_off_by_default_and_hints_when_on(monkeypatch):
    monkeypatch.delenv("ACIE_NBA", raising=False)
    q = "ye bahut mehnga hai, student ke liye kuch hai?"
    nlu = type("N", (), {"understand": staticmethod(lambda *a, **k: None)})
    _, _, pipe, _, _ = await run(["yes tell me", "kitne ka hai", q])
    assert "nba" not in pipe.sessions["s"].dialog
    monkeypatch.setenv("ACIE_NBA", "1")
    _, last, pipe, _, _ = await run(["yes tell me", "kitne ka hai", q])
    d = pipe.sessions["s"].dialog
    assert d["nba"] in nba.ACTIONS and set(json.loads(d["nba_scores"])) == set(nba.ACTIONS)
