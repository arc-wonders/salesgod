import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from acie.knowledge.schema import Entry, KnowledgeBase, Offering, Text3, Topic, Variants
from acie.knowledge.store import load_kb, validate

ROOT = Path(__file__).resolve().parents[1]
PRODUCTS = json.loads((ROOT / "data/voice/products.json").read_text(encoding="utf-8"))


def _entry(**k):
    d = dict(id="e1", topic_path="a", question_variants=Variants(en=["q"]), answer=Text3(en="a"), provenance="operator")
    d.update(k)
    return Entry(**d)


def _kb(entries, allow=False, offerings=()):
    return KnowledgeBase(product_id="p", allow_synthetic=allow, topics=[Topic(path="a", title=Text3(en="A"), overview=Text3(en="ov"))],
                         entries=entries, offerings=list(offerings))


def test_provenance_is_mandatory():
    with pytest.raises(ValidationError):
        Entry(id="e", topic_path="a", question_variants=Variants(en=["q"]), answer=Text3(en="a"))
    with pytest.raises(ValidationError):
        _entry(provenance="made-up")


def test_synthetic_rejected_when_not_allowed():
    with pytest.raises(ValidationError):
        _kb([_entry(provenance="synthetic")], allow=False)
    assert _kb([_entry(provenance="synthetic")], allow=True).entries


def test_price_literals_are_rejected_in_answers():
    with pytest.raises(ValidationError):
        _kb([_entry(answer=Text3(en="It costs ₹3,000 a month"))])
    with pytest.raises(ValidationError):
        _kb([_entry(answer=Text3(en="fine", hi="कीमत ₹500 है"))])


def test_offering_below_floor_and_unknown_refs():
    o = dict(id="o", name=Text3(en="O"), who_suits=["x"], price_inr=100, price_label=Text3(en="100"), provenance="operator")
    with pytest.raises(ValidationError):
        Offering(floor_inr=500, **o)
    with pytest.raises(ValidationError):
        _kb([_entry(offerings=["nope"])])


def test_shipped_kbs_validate_and_have_sizes():
    fz = load_kb(PRODUCTS["fitzone-gym"])
    assert fz is not None and fz.allow_synthetic
    assert len(fz.entries) >= 150 and 15 <= len([t for t in fz.topics if "/" not in t.path]) <= 30 and 5 <= len(fz.offerings) <= 10
    assert all(e.provenance == "synthetic" for e in fz.entries)
    for e in fz.entries:  # all three languages everywhere
        assert e.answer.hi and e.answer.hinglish and e.question_variants.hi and e.question_variants.hinglish, e.id
    assert validate(fz, PRODUCTS["fitzone-gym"]) == []
    # offerings never priced below a floor the config sets
    assert validate(fz, {**PRODUCTS["fitzone-gym"], "profit_floor": 10 ** 6}) != []


def test_claude_pro_is_public_only_and_blocks_synthetic():
    kb = load_kb(PRODUCTS["claude-pro"])
    assert kb is not None and kb.allow_synthetic is False
    assert {e.provenance for e in kb.entries} == {"public"}
    assert all(e.source for e in kb.entries)
    data = kb.model_dump(mode="json")
    data["entries"][0]["provenance"] = "synthetic"
    with pytest.raises(ValidationError):
        KnowledgeBase.model_validate(data)


def test_synthesize_refuses_real_company(capsys):
    from scripts.kb.synthesize import main
    assert main(["--product", "claude-pro", "--allow-synthetic"]) == 2
