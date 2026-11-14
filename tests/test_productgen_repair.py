"""Targeted repair of failing locations, last-resort dropping, subtopic top-up, early stop, repair-only pass on a saved version."""
import copy
import json
from pathlib import Path

import pytest

from acie.productgen import criteria as C
from acie.productgen import kbops as K
from acie.productgen import store as S
from acie.productgen.generate import generate, repair_version, validate_and_repair
from acie.productgen.spec import Spec, resolve
from test_productgen_fake import FakeLLM, t3

DEVA = "यह अंग्रेज़ी की जगह हिंदी है"
SPEC = dict(name="ZenBrew Coffee Subscription", category="coffee", notes="A monthly coffee box", price={"list_price": 1200, "offer_price": 999, "unit": "per month"},
            economics={"floor": 600}, criteria={"min_entries": 30})
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _batches(monkeypatch):
    monkeypatch.setenv("PRODUCTGEN_TOPICS_PER_CALL", "3")
    monkeypatch.setenv("PRODUCTGEN_ENTRIES_PER_CALL", "100")


class FixLLM(FakeLLM):
    """Fake whose targeted calls return correct-script text for exactly the items they are given."""

    def _kb_fix(self, p):
        out = []
        for e in p["fix"]:
            e = copy.deepcopy(e)
            e["answer"]["en"] = "A plain English answer."
            out.append(e)
        return {"entries": out}

    def _playbook(self, p):
        if p.get("only") == "fix_nodes":
            out = []
            for n in p["fix"]:
                n = copy.deepcopy(n)
                n["ask"] = "What matters most to you?"
                n["points"] = ["point one", "point two"]
                out.append(n)
            return {"nodes": out}
        return super()._playbook(p)

    def _ladder(self, p):
        if p.get("only") == "fix_text":
            return {"ladder": [{**r, "question": {**r["question"], "en": "A fixed English question?"}} for r in p["fix"]]}
        return super()._ladder(p)

    def _kb_plan(self, p):
        if p.get("only") == "subtopics":
            subs = [{"path": f"{t}/sub-{j}", "title": t3(f"Sub {t} {j}", "s"), "overview": t3("Sub overview", "s"), "interests": []}
                    for t in p["top_level"] for j in range(3)]
            return {"topics": subs + [{"path": "brand-new-top", "title": t3("Nope", "n"), "overview": t3("Nope", "n")}]}
        return super()._kb_plan(p)

    def _kb_entries(self, p):
        subs = [t for t in p["topics"] if "/" in t["path"]]
        if subs:
            return {"entries": [self._entry(t["path"], k + 70) for t in subs for k in range(2)]}
        return super()._kb_entries(p)


async def _base(llm=None, **kw):
    s = Spec.model_validate({**SPEC, **kw})
    llm = llm or FixLLM()
    doc = await generate(s, llm, Path("/nonexistent-never-written") if False else _TMP[0])
    return s, resolve(s), doc["bundle"], llm


_TMP: list = [None]


@pytest.fixture(autouse=True)
def _tmp(tmp_path):
    _TMP[0] = tmp_path


def _wrong(b):
    """Put a Devanagari text into an English field in every location kind."""
    kb, tree = b["kb"], b["tree"]
    kb["entries"][3]["answer"]["en"] = DEVA
    kb["topics"][2]["title"]["en"] = DEVA
    kb["offerings"][1]["name"]["en"] = DEVA
    tree["ladder"][1]["question"]["en"] = DEVA
    tree["nodes"][2]["ask"] = DEVA
    tree["nodes"][4].setdefault("children", []).append({"id": "child-x", "title": "Child", "when": {}, "points": ["fine"], "ask": DEVA, "points_hi": ["अ"], "ask_hi": "क्या?"})
    b["product"]["pitch_en"] = DEVA
    b["product"]["faqs"][2]["answer"] = DEVA
    return {"entry": kb["entries"][3]["id"], "topic": "topic:" + kb["topics"][2]["path"], "offering": "offering:" + kb["offerings"][1]["id"],
            "ladder": "ladder:" + tree["ladder"][1]["id"], "node": "node:" + tree["nodes"][2]["id"], "child": "node:child-x", "product": "product:pitch_en", "faq": "faq:2"}


async def test_wrong_script_in_every_location_is_fixed_by_the_targeted_call():
    s, eff, b, llm = await _base()
    assert C.check(b, s, eff)["publishable"]
    loc = _wrong(b)
    rep0 = C.check(b, s, eff)
    lang = next(c for c in rep0["checks"] if c["id"] == "languages")
    assert set(loc.values()) <= set(lang["items"]) and lang["problems"][loc["node"]]
    n0 = len(llm.calls)
    report, history = await validate_and_repair(b, s, eff, llm, [])
    assert not [c for c in report["checks"] if c["blocking"] and not c["passed"]]
    assert report["dropped"] == []
    used = [(st, p.get("only")) for st, p in llm.calls[n0:]]
    assert ("kb_fix", None) in used and ("playbook", "fix_nodes") in used and ("ladder", "fix_text") in used
    assert ("kb_plan", "fix_text") in used and ("profile", None) in used
    assert "languages" not in history[1]["failed"]  # fixed in the first round
    assert K.find_node(b["tree"]["nodes"], "child-x")["ask"] != DEVA  # nested node fixed in place


class StuckLLM(FixLLM):
    """Returns everything unchanged: nothing it is asked to fix gets fixed."""

    def _kb_fix(self, p):
        return {"entries": p["fix"]}

    def _playbook(self, p):
        if p.get("only") == "fix_nodes":
            return {"nodes": p["fix"]}
        return super()._playbook(p)

    def _ladder(self, p):
        if p.get("only") == "fix_text":
            return {"ladder": p["fix"]}
        return super()._ladder(p)

    def _profile(self, p):
        r = super()._profile(p)
        r["faqs"][2]["answer"] = t3(DEVA)
        return r


async def test_unfixable_optional_items_are_dropped_and_publish_allowed():
    llm = StuckLLM()
    s, eff, b, _ = await _base(llm)
    kb, tree = b["kb"], b["tree"]
    eid, nid, rid = kb["entries"][3]["id"], tree["nodes"][2]["id"], tree["ladder"][1]["id"]
    kb["entries"][3]["answer"]["en"] = DEVA
    tree["nodes"][2]["ask"] = DEVA
    tree["ladder"][1]["question"]["en"] = DEVA
    b["product"]["faqs"][2]["answer"] = DEVA
    report, history = await validate_and_repair(b, s, eff, llm, [])
    assert report["publishable"] and not [c for c in report["checks"] if c["blocking"] and not c["passed"]]
    assert {f"entry {eid}", f"playbook situation {nid}", f"ladder rung {rid}", "FAQ 2"} <= {x for x in report["dropped"] if not x.startswith("(then)")}
    assert all(e["id"] != eid for e in b["kb"]["entries"]) and K.find_node(b["tree"]["nodes"], nid) is None
    assert history[-1]["round"] == "final"


async def test_systemic_failure_is_not_hidden_by_dropping():
    llm = FakeLLM(flaws={"missing_hi"}, fixable=False)
    doc = await generate(Spec.model_validate(SPEC), llm, _TMP[0])
    assert not doc["report"]["publishable"] and doc["report"]["dropped"] == []


async def test_topic_shortfall_adds_subtopics_not_new_top_level_topics():
    llm = FixLLM(topics=6)
    doc = await generate(Spec.model_validate(SPEC), llm, _TMP[0])
    kb = doc["bundle"]["kb"]
    tops = [t for t in kb["topics"] if "/" not in t["path"]]
    assert len(tops) == 6 and "brand-new-top" not in {t["path"] for t in tops}
    assert 15 <= len(kb["topics"]) <= 40
    assert next(c for c in doc["report"]["checks"] if c["id"] == "topics")["passed"]
    assert any(st == "kb_plan" and p.get("only") == "subtopics" for st, p in llm.calls)
    have = {e["topic_path"] for e in kb["entries"]}
    assert all(t["path"] in have for t in kb["topics"] if "/" in t["path"])


def test_topic_criterion_defaults():
    c = Spec.model_validate(SPEC).criteria
    assert (c.min_topics, c.max_topics, c.min_total_topics, c.max_total_topics) == (5, 20, 15, 40)


async def test_no_progress_round_stops_the_loop_early():
    llm = FakeLLM(flaws={"missing_hi"}, fixable=False)
    doc = await generate(Spec.model_validate(SPEC), llm, _TMP[0])
    h = doc["repair_history"]
    assert len(h) == 2 and h[-1]["stopped"] == "no progress"  # round 0 + one round, not three identical ones
    assert [st for st, _ in llm.calls].count("kb_fix") == 1


ZEN = ROOT / "data" / "real" / "pg" / "zenbrew_v1.json"


@pytest.mark.skipif(not ZEN.exists(), reason="the real ZenBrew v1 is not on this machine")
async def test_repair_only_pass_on_the_real_zenbrew_v1(tmp_path):
    d = json.loads(ZEN.read_text(encoding="utf-8"))
    pid, tenant = d["effective"]["product_id"], d["spec"].get("tenant", "default")
    S._write(S.pdir(tmp_path, tenant, pid) / "v1.json", d)

    class OnlyFix(FixLLM):
        pass

    llm = OnlyFix()
    doc = await repair_version(tmp_path, tenant, pid, 1, llm)
    assert doc["version"] == 2 and doc["report"]["publishable"], [c for c in doc["report"]["checks"] if not c["passed"]]
    assert doc["report"]["passed"]
    assert {st for st, _ in llm.calls} == {"playbook"}  # nothing regenerated; only the failing item went back
    assert S.load_version(tmp_path, tenant, pid, 1)["report"]["publishable"] is False
