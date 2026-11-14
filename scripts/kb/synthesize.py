"""Generate (or validate) a knowledge base for ANY product config.

  python -m scripts.kb.synthesize --product fitzone-gym --tenant default            # needs KB_LLM_* env vars
  python -m scripts.kb.synthesize --from-file my_kb.json --product fitzone-gym      # hand-authored: validate + write
  python -m scripts.kb.synthesize --check data/voice/knowledge/default/fitzone-gym.json --product fitzone-gym

Env: KB_LLM_BASE_URL (OpenAI-compatible, e.g. http://host:8000/v1), KB_LLM_MODEL, KB_LLM_API_KEY, optional KB_LLM_SEED (default 7).
Requests use temperature 0 and a fixed seed, so a given model and prompt gives the same KB on every run.

Rules (the same ones the loader enforces):
  * provenance "synthetic" is only produced for a product the operator owns or has approved: pass --allow-synthetic;
    a product config with "allow_synthetic": false (a real company's product) is refused;
  * prices are never written into answer text: the model is told to use {offer_price}, {original_price} and {price:OFFERING_ID};
  * every offering price must be >= the product's profit floor when the config has one ("profit_floor": number or {offering_id: n});
  * entries are validated with the schema, deduplicated, and anything invalid is dropped and reported.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from acie.knowledge import text as T  # noqa: E402
from acie.knowledge.schema import Entry, KnowledgeBase, Offering, Topic  # noqa: E402
from acie.knowledge.store import save_kb, validate  # noqa: E402

SYSTEM = ("You write product knowledge bases for a phone sales assistant. Output ONLY JSON. Be specific and realistic: concrete "
          "equipment, schedules, qualifications, policies, programs. Every answer is 1-3 short spoken sentences. Provide en, hi "
          "(Devanagari) and hinglish (Roman Hindi) for every question variant list and every answer. Never write a price as digits "
          "in an answer: use the slots {offer_price}, {original_price}, {discount}, {offer_validity} or {price:OFFERING_ID}. "
          "Never invent real people's names; name roles instead.")


def llm(messages: list[dict], max_tokens: int = 6000) -> str:
    base = os.environ["KB_LLM_BASE_URL"].rstrip("/")
    body = {"model": os.environ["KB_LLM_MODEL"], "messages": messages, "temperature": 0, "max_tokens": max_tokens,
            "seed": int(os.getenv("KB_LLM_SEED", "7"))}
    req = urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer " + os.getenv("KB_LLM_API_KEY", "none")})
    with urllib.request.urlopen(req, timeout=300) as r:
        out = json.loads(r.read())
    return out["choices"][0]["message"]["content"]


def parse_json(s: str):
    s = s.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", s, re.S)
    if m:
        s = m.group(1)
    return json.loads(s[s.index("{") if "{" in s[:5] else 0:])


def plan_kb(product: dict) -> dict:
    p = json.dumps({k: product.get(k) for k in ("product_name", "seller_name", "one_liner", "original_price", "offer_price", "benefits", "knowledge")},
                   ensure_ascii=False)
    prompt = (f"Product config: {p}\n\nReturn JSON {{\"topics\": [{{\"path\": \"a\" or \"a/b\", \"title\": {{en,hi,hinglish}}, \"overview\": {{en,hi,hinglish}}, "
              "\"interests\": [tag ids]}}], \"interests\": [{\"id\", \"label\": {en,hi,hinglish}, \"keywords\": [\"...\"], \"related\": [ids], "
              "\"discovery\": {en:[q],hi:[q],hinglish:[q]}}], \"offerings\": [{\"id\", \"name\": {en,hi,hinglish}, \"who_suits\": [tag ids], "
              "\"price_inr\": int, \"price_label\": {en,hi,hinglish}, \"pitch\": {en:[...],hi:[...],hinglish:[...]}, \"upsell_from\": [topic paths]}]}. "
              "15-30 topics (with subtopics), 8-15 interest tags, 5-10 offerings/add-ons whose prices are sensible next to the base price.")
    return parse_json(llm([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]))


def entries_for(product: dict, plan: dict, topics: list[str]) -> list[dict]:
    offs = [o["id"] for o in plan.get("offerings", [])]
    tags = [t["id"] for t in plan.get("interests", [])]
    prompt = (f"Product: {product.get('product_name')} ({product.get('one_liner')}). Offering ids: {offs}. Interest tag ids: {tags}.\n"
              f"Write 6-8 Q&A entries for EACH of these topics: {topics}. Return JSON {{\"entries\": [{{\"id\": \"kebab-id\", \"topic_path\": "
              "\"...\", \"question_variants\": {en:[2-3],hi:[1-2],hinglish:[1-2]}, \"answer\": {en,hi,hinglish}, \"facts\": {}, \"offerings\": [ids], "
              "\"interests\": [tags], \"teach\": false, \"confidence\": 0.9}]}. Consistent specifics across entries.")
    return parse_json(llm([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}])).get("entries", [])


def dedupe(entries: list[Entry]) -> tuple[list[Entry], int]:
    seen_q: set[str] = set()
    seen_id: set[str] = set()
    out, dropped = [], 0
    for e in entries:
        key = T.normalize(e.question_variants.en[0] if e.question_variants.en else e.question_variants.all()[0])
        if key in seen_q or e.id in seen_id:
            dropped += 1
            continue
        seen_q.add(key)
        seen_id.add(e.id)
        out.append(e)
    return out, dropped


def assemble(product: dict, tenant: str, raw_plan: dict, raw_entries: list[dict], allow_synth: bool, prov: str = "synthetic") -> tuple[KnowledgeBase, list[str]]:
    notes: list[str] = []
    topics = [Topic(**t) for t in raw_plan["topics"]]
    paths = {t.path for t in topics}
    tags_raw = raw_plan.get("interests", [])
    offs: list[Offering] = []
    floor = product.get("profit_floor")
    for o in raw_plan.get("offerings", []):
        try:
            off = Offering(provenance=prov, **o)
        except Exception as ex:  # noqa: BLE001
            notes.append(f"offering dropped: {o.get('id')}: {str(ex)[:80]}")
            continue
        f = (floor.get(off.id) if isinstance(floor, dict) else floor) if floor is not None else None
        if f is not None and off.price_inr is not None and off.price_inr < f:
            notes.append(f"offering dropped: {off.id} price {off.price_inr} below profit floor {f}")
            continue
        offs.append(off)
    ids = {o.id for o in offs}
    good: list[Entry] = []
    for r in raw_entries:
        try:
            r = dict(r)
            r["offerings"] = [x for x in r.get("offerings", []) if x in ids]
            e = Entry(provenance=prov, **r)
            if e.topic_path not in paths:
                raise ValueError("unknown topic " + e.topic_path)
            good.append(e)
        except Exception as ex:  # noqa: BLE001
            notes.append(f"entry dropped: {r.get('id')}: {str(ex)[:90]}")
    good, dup = dedupe(good)
    if dup:
        notes.append(f"{dup} duplicate entries dropped")
    from acie.knowledge.schema import InterestTag
    tags = [InterestTag(**t) for t in tags_raw]
    tagids = {t.id for t in tags}
    for e in good:
        e.interests = [i for i in e.interests if i in tagids]
    for o in offs:
        o.who_suits = [i for i in o.who_suits if i in tagids] or o.who_suits[:1]
    kb = KnowledgeBase(product_id=product["id"], tenant=tenant, allow_synthetic=allow_synth, description=f"Synthetic KB for {product.get('product_name')}",
                       topics=topics, entries=good, offerings=offs, interests=tags)
    notes.extend(validate(kb, product))
    return kb, notes


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--product", required=True)
    ap.add_argument("--tenant", default="default")
    ap.add_argument("--products-file", default=str(ROOT / "data/voice/products.json"))
    ap.add_argument("--allow-synthetic", action="store_true", help="the operator owns / approved this product")
    ap.add_argument("--from-file", help="a hand-authored KB JSON to validate and install")
    ap.add_argument("--check", help="validate an existing KB file")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    product = json.loads(Path(a.products_file).read_text(encoding="utf-8"))[a.product]
    if a.check or a.from_file:
        kb = KnowledgeBase.model_validate_json(Path(a.check or a.from_file).read_text(encoding="utf-8"))
        issues = validate(kb, product)
        print(len(kb.entries), "entries", len(kb.topics), "topics", len(kb.offerings), "offerings", issues or "ok")
        if issues:
            return 1
        if a.from_file:
            print(save_kb(kb, Path(a.out) if a.out else None))
        return 0
    from acie.knowledge.store import load_kb
    existing = load_kb(product)
    if product.get("allow_synthetic") is False or (existing is not None and not existing.allow_synthetic) or not a.allow_synthetic:
        print("refused: synthetic generation needs --allow-synthetic and a product whose config does not set allow_synthetic=false "
              "(never fabricate claims about a real company's product; an existing KB with allow_synthetic=false also blocks it)", file=sys.stderr)
        return 2
    if not (os.getenv("KB_LLM_BASE_URL") and os.getenv("KB_LLM_MODEL")):
        print("set KB_LLM_BASE_URL and KB_LLM_MODEL (and KB_LLM_API_KEY)", file=sys.stderr)
        return 3
    plan = plan_kb(product)
    top = sorted({t["path"].split("/")[0] for t in plan["topics"]})
    raw: list[dict] = []
    for i in range(0, len(top), 4):
        group = [t["path"] for t in plan["topics"] if t["path"].split("/")[0] in top[i:i + 4]]
        raw += entries_for(product, plan, group)
    kb, notes = assemble(product, a.tenant, plan, raw, True)
    for n in notes:
        print(" -", n)
    print(len(kb.entries), "entries", len(kb.topics), "topics", len(kb.offerings), "offerings")
    print(save_kb(kb, Path(a.out) if a.out else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
