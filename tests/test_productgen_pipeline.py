import json
from pathlib import Path

import pytest

from acie.productgen import criteria as C
from acie.productgen import store as S
from acie.productgen.generate import generate
from acie.productgen.spec import Criteria, Spec, looks_like_real_brand, resolve, slugify
from test_productgen_fake import FakeLLM


@pytest.fixture(autouse=True)
def _one_big_batch(monkeypatch):
    # these tests were written for whole-topic batches; the default (1 topic, 6 entries per call) has its own test below
    monkeypatch.setenv("PRODUCTGEN_TOPICS_PER_CALL", "3")
    monkeypatch.setenv("PRODUCTGEN_ENTRIES_PER_CALL", "100")

SPEC = dict(name="ZenBrew Coffee Subscription", category="coffee", notes="A monthly coffee box", price={"list_price": 1200, "offer_price": 999, "unit": "per month"},
            economics={"floor": 600})


def spec(**kw):
    d = {**SPEC, **kw}
    return Spec.model_validate(d)


# ---------------------------------------------------------------- spec
def test_spec_requires_name_and_validates_numbers():
    with pytest.raises(Exception):
        Spec.model_validate({})
    with pytest.raises(Exception):
        Spec.model_validate({"name": "x"})  # too short
    with pytest.raises(Exception):
        spec(price={"list_price": 100, "offer_price": 150})  # offer above list
    with pytest.raises(Exception):
        spec(economics={"floor": 1500})  # quoted price below the floor
    with pytest.raises(Exception):
        spec(criteria={"min_topics": 40, "max_topics": 30})
    s = spec(languages=["hi"])
    assert s.languages[0] == "en" and "hi" in s.languages
    assert slugify("ZenBrew Coffee Subscription!") == "zenbrew-coffee-subscription" and s.pid == "zenbrew-coffee-subscription"
    assert spec(economics={"cost": 500, "min_margin_pct": 20}).economics.profit_floor == 600


def test_real_brand_rule_resolution():
    assert looks_like_real_brand("Claude Pro") == "claude" and looks_like_real_brand("ZenBrew") is None
    e = resolve(spec(name="Netflix Premium"))
    assert e.real_brand and e.provenance == "public" and not e.allow_synthetic and e.criteria.min_entries <= 30
    e = resolve(spec(name="Netflix Premium", authorised=True))
    assert not e.real_brand and e.provenance == "synthetic" and e.allow_synthetic
    assert resolve(spec(real_brand=True)).real_brand  # explicit operator override
    d = resolve(spec())
    assert d.criteria.min_entries == 150 and d.floor == 600


# ---------------------------------------------------------------- generation + criteria
async def _gen(tmp_path, llm=None, **kw):
    s = spec(**kw)
    llm = llm or FakeLLM()
    return await generate(s, llm, tmp_path), llm


async def test_generation_meets_all_criteria_and_saves_a_version(tmp_path):
    doc, llm = await _gen(tmp_path)
    rep = doc["report"]
    assert rep["passed"], [c for c in rep["checks"] if not c["passed"]]
    assert rep["counts"]["entries"] >= 150 and 15 <= rep["counts"]["topics_top"] <= 30 and 5 <= rep["counts"]["offerings"] <= 10
    assert 14 <= rep["counts"]["situations"] <= 20 and rep["counts"]["ladder"] >= 4
    d = tmp_path / "productgen" / "default" / "zenbrew-coffee-subscription"
    assert (d / "v1.json").exists() and (d / "v1.diff.json").exists()
    saved = json.loads((d / "v1.json").read_text())
    assert saved["spec"]["criteria"]["min_entries"] == 150  # the criteria travel with the version
    assert all(e["provenance"] == "synthetic" for e in saved["bundle"]["kb"]["entries"])
    assert saved["bundle"]["kb"]["allow_synthetic"] is True
    p = saved["bundle"]["product"]
    assert p["offer_price"] == "₹999 per month" and p["original_price"] == "₹1,200 per month" and p["discount"] == "17% off"
    # unknown intent labels are dropped, interests scrubbed, bad fit rule ignored
    n0 = saved["bundle"]["tree"]["nodes"][0]
    assert "bogus" not in n0["when"]["intents"]
    assert [r["offering"] for r in saved["bundle"]["tree"]["fit_rules"]] == ["off-1"]
    assert all(i != "nope" for r in saved["bundle"]["tree"]["ladder"] for i in r["interests"])
    # a key is never written
    assert "api_key" not in json.dumps(saved).lower()


async def test_prices_never_below_floor_and_negotiation_filtered(tmp_path):
    doc, _ = await _gen(tmp_path, llm=FakeLLM(flaws={"cheap_offering"}))
    b = doc["bundle"]
    assert all(o["price_inr"] >= 600 for o in b["kb"]["offerings"])
    neg = b["negotiation"]
    assert neg["suggested"]
    prices = [s["price"] for s in neg["economics"]["ladder"] if s["kind"] == "price"]
    assert prices == sorted(prices, reverse=True) and all(600 <= x < 999 for x in prices) and 550 not in prices
    assert b["product"]["economics"]["cost"] == 600


async def test_no_floor_means_no_negotiation_suggestion(tmp_path):
    doc, llm = await _gen(tmp_path, economics={})
    neg = doc["bundle"]["negotiation"]
    assert neg["suggested"] is False and neg["economics"] is None and "floor" in neg["reason"]
    assert not any(st == "negotiation" for st, _ in llm.calls)
    assert "economics" not in doc["bundle"]["product"]


async def test_repair_loop_fixes_flaws_and_records_history(tmp_path):
    llm = FakeLLM(flaws={"missing_hi", "banned", "few_entries", "short_ladder", "few_nodes"})
    doc, _ = await _gen(tmp_path, llm=llm)
    assert doc["report"]["passed"], [c for c in doc["report"]["checks"] if not c["passed"]]
    h = doc["repair_history"]
    assert h[0]["failed"] and 1 <= len(h) <= 4 and h[-1]["failed"] == []
    stages = [s for s, _ in llm.calls]
    assert "kb_fix" in stages and stages.count("kb_entries") > 6


async def test_repair_is_capped_and_unfixable_banned_content_blocks_publish(tmp_path):
    llm = FakeLLM(flaws={"banned"}, fixable=False)
    doc, _ = await _gen(tmp_path, llm=llm)
    assert len(doc["repair_history"]) <= 4  # round 0 + max 3
    # the leftover banned entries are removed rather than shipped
    assert not any("guaranteed to lose" in e["answer"]["en"] for e in doc["bundle"]["kb"]["entries"])


async def test_unfixable_language_gap_is_not_publishable(tmp_path):
    llm = FakeLLM(flaws={"missing_hi"}, fixable=False)
    doc, _ = await _gen(tmp_path, llm=llm)
    lang = next(c for c in doc["report"]["checks"] if c["id"] == "languages")
    assert not lang["passed"] and not doc["report"]["publishable"] and doc["status"] == "needs_review"
    with pytest.raises(S.PublishError):
        S.publish(tmp_path, "default", doc["spec"]["product_id"] or "zenbrew-coffee-subscription", doc["version"], tmp_path / "kb")


async def test_real_brand_forces_public_only(tmp_path):
    doc, llm = await _gen(tmp_path, name="Claude Pro", seller="Anthropic", price={}, economics={})
    eff, b = doc["effective"], doc["bundle"]
    assert eff["real_brand"] and eff["provenance"] == "public" and eff["allow_synthetic"] is False
    assert b["kb"]["allow_synthetic"] is False
    assert all(e["provenance"] == "public" and "review" in e["source"] and e["confidence"] <= 0.7 for e in b["kb"]["entries"])
    assert all(o["price_inr"] is None for o in b["kb"]["offerings"])  # a real company's prices are never invented
    assert b["negotiation"]["suggested"] is False and len(b["review"]) == len(b["kb"]["entries"])
    assert b["product"]["allowed_numbers"], "dialing stays blocked until the operator sets test phones"
    assert doc["report"]["counts"]["entries"] < 150 or doc["report"]["passed"]
    assert next(c for c in doc["report"]["checks"] if c["id"] == "real_brand")["passed"]
    # the model is told the mode
    assert all(p["mode"] == "public" for _, p in llm.calls if "mode" in p)


async def test_authorised_real_brand_name_gets_synthetic(tmp_path):
    doc, _ = await _gen(tmp_path, name="Netflix Clone Box", authorised=True)
    assert doc["effective"]["provenance"] == "synthetic"


# ---------------------------------------------------------------- criteria checks on hand-broken bundles
async def test_criteria_catch_violations(tmp_path):
    doc, _ = await _gen(tmp_path)
    s, eff = Spec.model_validate(doc["spec"]), resolve(Spec.model_validate(doc["spec"]))
    import copy

    def broken(fn):
        b = copy.deepcopy(doc["bundle"])
        fn(b)
        return {c["id"]: c for c in C.check(b, s, eff)["checks"]}

    r = broken(lambda b: b["kb"]["offerings"][0].update(price_inr=100))
    assert not r["numeric"]["passed"] and "below" in r["numeric"]["detail"]
    r = broken(lambda b: b["kb"]["entries"][0]["answer"].update(hi=""))
    assert not r["languages"]["passed"]
    r = broken(lambda b: b["kb"]["entries"][0]["answer"].update(hinglish="यह देवनागरी है"))
    assert not r["languages"]["passed"]  # wrong script
    r = broken(lambda b: b["kb"]["entries"][0]["answer"].update(en="Guaranteed returns, risk-free, hurry!"))
    assert not r["banned"]["passed"]
    r = broken(lambda b: b["kb"]["entries"][1]["answer"].update(en="We are ISO 9001 certified."))
    assert not r["banned"]["passed"] and "certification" in r["banned"]["detail"]
    r = broken(lambda b: b["kb"]["entries"][0]["answer"].update(en="Other brands are fake and useless competitors"))
    assert not r["banned"]["passed"] and "disparagement" in r["banned"]["detail"]
    r = broken(lambda b: b["kb"]["entries"][0].pop("provenance"))
    assert not r["provenance"]["passed"]
    r = broken(lambda b: b["kb"]["entries"][0].update(provenance="public"))
    assert not r["provenance"]["passed"]  # synthetic product: public label not allowed
    r = broken(lambda b: b["kb"]["entries"].__delitem__(slice(0, 30)))
    assert not r["entries"]["passed"] and not r["entries"]["blocking"]
    r = broken(lambda b: b["tree"].update(ladder=b["tree"]["ladder"][:2]))
    assert not r["ladder"]["passed"]
    r = broken(lambda b: b["kb"]["entries"][0]["answer"].update(en="It costs ₹500 only"))
    assert not r["schema"]["passed"]  # price literal
    # operator-banned phrase and operator-supplied urgency
    s2 = Spec.model_validate({**doc["spec"], "banned_phrases": ["item 3 of"]})
    assert any(not c["passed"] for c in C.check(doc["bundle"], s2, resolve(s2))["checks"] if c["id"] == "banned")


def test_urgency_allowed_only_when_operator_gave_it():
    s = spec()
    e = resolve(s)
    assert C.banned_hits("Hurry, only today!", s, e)
    s2 = spec(price={"list_price": 1200, "offer_price": 999, "validity": "only today"})
    assert not [h for h in C.banned_hits("Offer valid only today.", s2, resolve(s2)) if "urgency" in h]
    assert C.banned_hits("We are FSSAI approved", s, e) and not C.banned_hits("fresh beans", s, e)


# ---------------------------------------------------------------- versioning + diff
async def test_versions_and_diff(tmp_path):
    d1, _ = await _gen(tmp_path)
    d2, _ = await _gen(tmp_path, llm=FakeLLM(per_topic=11, variant="b"))
    assert (d1["version"], d2["version"]) == (1, 2)
    diff = json.loads((tmp_path / "productgen/default/zenbrew-coffee-subscription/v2.diff.json").read_text())
    assert diff["against"] == 1 and diff["diff"]["entries"]["added"] and diff["diff"]["entries"]["removed"]
    assert S.get_diff(tmp_path, "default", "zenbrew-coffee-subscription", 1, 2)["diff"]["total_changes"] > 0
    vs = S.list_versions(tmp_path, "default", "zenbrew-coffee-subscription")
    assert [v["version"] for v in vs] == [1, 2] and vs[0]["counts"]["entries"] >= 150


# ---------------------------------------------------------------- publish merge
async def test_publish_writes_files_and_keeps_console_edits(tmp_path):
    kb_root = tmp_path / "kb"
    pid = "zenbrew-coffee-subscription"
    d1, _ = await _gen(tmp_path)
    res = S.publish(tmp_path, "default", pid, 1, kb_root)
    assert res["ok"]
    products = json.loads((tmp_path / "products.json").read_text())
    assert products[pid]["product_name"] == "ZenBrew Coffee Subscription" and products[pid]["offer_price"] == "₹999 per month"
    from acie.decision.domains.product_sales import Product
    Product.model_validate(products[pid])
    kbf = kb_root / "default" / f"{pid}.json"
    kb = json.loads(kbf.read_text())
    assert len(kb["entries"]) >= 150
    assert (tmp_path / "playbooks/default" / f"{pid}.json").exists()

    # the operator edits things in the console
    products[pid]["tone"] = "calm and slow (console edit)"
    products[pid]["one_liner"] = "console one liner"
    (tmp_path / "products.json").write_text(json.dumps(products))
    v1_ids = {e["id"] for e in kb["entries"]}
    kb["entries"][0]["answer"]["en"] = "Console-edited answer."
    edited_id = kb["entries"][0]["id"]
    removed_id = kb["entries"][1]["id"]
    del kb["entries"][1]
    kb["entries"].append({**kb["entries"][2], "id": "console-added"})
    kb["entries"][-1]["question_variants"] = {"en": ["a brand new console question?"], "hi": [], "hinglish": []}
    kbf.write_text(json.dumps(kb))
    pbf = tmp_path / "playbooks/default" / f"{pid}.json"
    pb = json.loads(pbf.read_text())
    pb["nodes"][0]["ask"] = "console ask"
    pbf.write_text(json.dumps(pb))

    d2, _ = await _gen(tmp_path, llm=FakeLLM(per_topic=11), price={"list_price": 1200, "offer_price": 899, "unit": "per month"})
    assert d2["version"] == 2
    res2 = S.publish(tmp_path, "default", pid, 2, kb_root)
    assert res2["backup"]
    p2 = json.loads((tmp_path / "products.json").read_text())[pid]
    assert p2["tone"] == "calm and slow (console edit)" and p2["one_liner"] == "console one liner"  # kept
    assert p2["offer_price"] == "₹899 per month"  # untouched field takes the new version
    kb2 = {e["id"]: e for e in json.loads(kbf.read_text())["entries"]}
    assert kb2[edited_id]["answer"]["en"] == "Console-edited answer."
    assert removed_id not in kb2 and "console-added" in kb2
    assert set(kb2) - v1_ids - {"console-added"}  # new entries from v2 arrived
    pb2 = json.loads(pbf.read_text())
    assert next(n for n in pb2["nodes"] if n["id"] == pb["nodes"][0]["id"])["ask"] == "console ask"
    assert S.state(tmp_path, "default", pid)["published"] == 2
    assert S.load_version(tmp_path, "default", pid, 2)["status"] == "published"


async def test_publish_real_brand_never_writes_synthetic(tmp_path):
    doc, _ = await _gen(tmp_path, name="Claude Pro", price={}, economics={})
    pid = doc["effective"]["product_id"]
    S.publish(tmp_path, "default", pid, 1, tmp_path / "kb")
    kb = json.loads((tmp_path / "kb/default" / f"{pid}.json").read_text())
    assert kb["allow_synthetic"] is False and all(e["provenance"] == "public" for e in kb["entries"])


async def test_default_batching_small_calls(tmp_path, monkeypatch):
    """Live Gemini run: 3 topics x all entries in one reply came back cut off (~60 KB). Default = 1 topic, <= 6 entries per call."""
    monkeypatch.delenv("PRODUCTGEN_TOPICS_PER_CALL", raising=False)
    monkeypatch.delenv("PRODUCTGEN_ENTRIES_PER_CALL", raising=False)
    llm = FakeLLM()
    await _gen(tmp_path, llm=llm)
    sizes = [c.get("max_entries") for st, c in getattr(llm, "calls", []) if st == "kb_entries" and "errors" not in c]
    assert sizes and max(sizes) <= 6
