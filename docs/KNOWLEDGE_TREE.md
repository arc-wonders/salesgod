# Knowledge base and multi-layer decision tree

Why: a live caller asked about powerlifting and competitions and the agent said "check the website". The agent had no detail to
give. This adds (1) a product-agnostic knowledge base (KB) with deep, multilingual detail and (2) a deterministic decision tree
that tracks how much of the KB has been explored and given, infers what the caller cares about, teaches what they ask, and only
then offers the one add-on that actually fits. Nothing here is gym-specific: the same code runs any product that has a KB file.

Everything is deterministic (no `hash()`, no randomness, sorted iteration, fixed tie-breaks). Only the final wording stays with the LLM.
Off by default: `ACIE_KB=1` turns it on.

## Files
| path | what |
|---|---|
| `acie/knowledge/schema.py` | pydantic schema: Topic taxonomy, Entry, Offering, InterestTag, KnowledgeBase |
| `acie/knowledge/text.py` | normalisation, synonym concepts, Devanagari<->Roman phonetic skeleton, crc32 char n-grams |
| `acie/knowledge/retrieve.py` | `retrieve(kb_or_product, question, lang, k, product, exclude)` |
| `acie/knowledge/store.py` | `load_kb(product)`, `save_kb`, `validate` (slots, profit floor, provenance) |
| `acie/knowledge/render.py` | fills `{offer_price}`, `{price:OFFERING_ID}` etc. from config; prices live only there |
| `acie/tree/` | `rules.py` (stage table), `interest.py`, `state.py` (coverage), `hint.py` (`tree_hint`), `enrich.py` (`enrich_plan`, `kb_knowledge`) |
| `data/voice/knowledge/<tenant>/<product_id>.json` | the KB files (committed) |
| `scripts/kb/synthesize.py` | generate / validate / install a KB for any product |
| `scripts/kb/build_fitzone.py`, `build_claude_pro.py`, `scripts/kb/authored/*` | the hand-authored sources of the two shipped KBs |

## KB schema
Product -> topics (taxonomy `topic` / `topic/subtopic`, each with an overview in en/hi/hinglish) -> entries.
Entry: `id`, `topic_path`, `question_variants` {en,hi,hinglish}, `answer` {en,hi,hinglish}, `facts`, `offerings` (related ids), `interests`
(tags), `provenance` (**mandatory**: `operator` | `public` | `synthetic`), `confidence`, `source`, `teach` (a tip worth teaching unprompted).
Offering: `id`, `name`, `who_suits` (interest tags), `price_inr` + `price_label` (the ONLY place a price is written), `included`, `floor_inr`,
`pitch` lines per language, `upsell_from` topics, `min_fit`. InterestTag: `keywords` (what callers say, any script), `related`, `discovery` question.

Hard rules enforced by the loader: provenance on every entry; a KB with `allow_synthetic: false` cannot hold or serve a synthetic entry;
answers may not contain price literals (they use slots); offering price >= its floor / the product's `profit_floor` when the config has one.

## Shipped KBs
| product | file | provenance | size |
|---|---|---|---|
| `fitzone-gym` (operator's own demo gym) | `data/voice/knowledge/default/fitzone-gym.json` | synthetic | 160 entries, 35 topic nodes (23 top level), 15 interest tags, 9 offerings |
| `claude-pro` (a REAL company's product) | `data/voice/knowledge/default/claude-pro.json` | public, `allow_synthetic: false` | 33 entries, 7 topics, 9 tags, 2 offerings (yearly billing, Team plan) |

### Synthetic data policy
Synthetic KBs are for products the operator owns (the FitZone demo) or has approved. Inventing equipment, prices, coaches or policies for
a real company's product and telling them to real customers is deception and a legal risk. So: `claude-pro` is public facts only (taken from
Anthropic's public pages and the operator-approved product text), its KB carries `allow_synthetic: false`, the schema refuses a synthetic entry
in it, retrieval filters synthetic entries out, and `scripts/kb/synthesize.py` refuses to generate for it (or for any product whose config or
existing KB says `allow_synthetic: false`, or without `--allow-synthetic`). Its only offerings are real, public ones; no discounts are invented
(the existing config has no concessions). A product with no KB changes nothing.

For FitZone, specifics (strength zone, coach role, powerlifting program, prices of add-ons) are demo data. Add-on prices are set in the KB
(e.g. powerlifting coaching, competition prep) and are not the membership price; if you want other numbers, edit
`scripts/kb/authored/fitzone_meta.py` and rerun `python scripts/kb/build_fitzone.py`.

## Retrieval
Query and every entry text go through: lower-case, NFKC, phrase joins ("power lifting" -> powerlifting), stopword removal (en/hi/hinglish),
synonym concepts (price, competition, powerlifting, nutrition... in any script), and a cross-script phonetic skeleton so "पावरलिफ्टिंग" and
"powerlifting" meet. Score = best over an entry's question variants (and 0.55 x its answer text) of 0.75 x idf-weighted coverage of the query
terms + 0.25 x cosine of crc32-hashed char 3-grams. Words nothing in the KB uses are ignored (noise). Ties: score, confidence, id.
Nothing confident -> the nearest topic's overview (never "contact the owner"). `exclude=` skips entries already given.

## Decision tree (4 layers)
1. **Stage** (first matching rule wins; `acie/tree/rules.py`):

| rule | stage | when |
|---|---|---|
| R1 | after_close | caller accepted or the payment link is promised |
| R2 | negotiate | price objection and asking for lower |
| R3 | objection | any other objection this turn |
| R4 | close | buying interest >= 0.7, or a clear yes after the first turn |
| R5 | open | first customer turn, nothing stated yet |
| R6 | educate | the caller asked something (or intent benefits / tell_more / clarify) |
| R7 | recommend | an offering fits and the cluster was explored |
| R8 | discover | otherwise |

2. **Interest cluster**: tags from the KB scored from the caller's own words (1 per utterance hit, +0.5 for a second keyword; related tags get 0.3x).
3. **Topic node**: the taxonomy node of the best retrieved entry, else the least-explored one that speaks to the top cluster.
4. **Leaf action**: `answer | teach | ask_discovery | bridge_offering | close_ask | handle_objection | acknowledge_close`.

| stage | action |
|---|---|
| after_close | acknowledge_close |
| objection, negotiate | handle_objection (no product detail poured in) |
| close | close_ask |
| any, offering fits + comfortable + not pitched, and (fit >= 2.0 strong, or >= 2 cluster entries explored, or stage recommend) | bridge_offering: answer first, then one natural offer |
| educate | answer |
| recommend | teach |
| open | ask_discovery |
| discover | teach if there is something left to teach, else ask_discovery |

### Upsell rules
* an offering is only considered if a tag in its `who_suits` was stated directly (>= 0.8): never pitched from related-tag spread alone;
* fit = best tag score + 0.5 x second best; must reach `min_fit`; strong at >= 2.0;
* never when the caller is frustrated (>= 0.6), declining (deny / goodbye / opt-out / callback / declined), or in objection / close / after-close;
* at most 2 pitches per offering; prices only through a `Fact` slot (`kb_price_<offering>`), never as text;
* `why` is built from what they said (interests + facts like "four to five days a week").

### Coverage
State in `session.dialog["_kb"]` (JSON): entries asked (explored), entries given, offerings pitched, per-topic asked counts, KB totals.
Entries the agent already said are also recognised from the agent's earlier lines (>= 70% of the answer's terms). `coverage_summary(session)`
returns `{kb_total, asked, given, given_pct, topics_explored{topic:{asked,total}}, offered, given_ids, asked_ids}`.
`TreeHint.coverage_text()` is the prompt line (<= 700 chars). Given entries are excluded from retrieval, so the LLM never gets a repeat.

## API
```python
from acie.tree import tree_hint, enrich_plan, coverage_summary, kb_knowledge
hint = tree_hint(session, signals, text, product)      # TreeHint or None (no KB); signals may be None (reads dialog["_signals"])
plan = enrich_plan(plan, session, signals, text, product)   # no-op unless ACIE_KB=1; leaves llm_allowed=False / end_call / safety plans alone
text = kb_knowledge(product, question, lang)           # talking points or nearest overview, for product_sales' background
```

## Hook lines for the coordinator (3 small edits, nothing else needed)
1. `acie/pipeline.py`, right after `plan = await self.decision.decide(intent, turn, session)` (and before `_no_repeat`):
```python
            from .tree import enrich_plan  # (or at the top of the file)
            plan = enrich_plan(plan, session, None, turn.text, getattr(self.decision.domain, "p", None))
```
   (`None` for signals makes it read the fused `TurnSignals` the pipeline already stored in `session.dialog["_signals"]`.)
2. `acie/decision/domains/product_sales.py`, in the plan builder where `knowledge = [p.knowledge] ...` is assembled (about line 1040), so an unanswerable
   question gets real detail or the nearest overview instead of "I don't know / check the website":
```python
        from ...tree import kb_knowledge
        if (extra := kb_knowledge(p, text, lang)):
            knowledge.append(extra)
```
3. (optional) log `coverage_summary(session)` with the turn trace.
