# Add a product by name

Type a product name in `/productgen`; Gemini (any OpenAI-compatible model) writes the product config, the **information tree** (the
knowledge base, `acie/knowledge` schema) and the **decision tree** (playbook nodes, discovery ladder, interest clusters, offering-fit
rules), checks them against explicit criteria, repairs them, and saves a **new version file**. Nothing reaches a live call until the
operator presses **Publish**. At runtime everything stays deterministic: generation happens offline, between calls.

| path | what |
|---|---|
| `acie/productgen/spec.py` | `Spec` (operator inputs), `Criteria` (what a version must meet), `resolve()` (real-brand rule) |
| `acie/productgen/llm.py` | LLM seam: OpenAI-compatible client (temperature 0, JSON mode, retries); prompts and per-stage JSON shapes |
| `acie/productgen/kbops.py` | tolerant builders: raw model JSON -> valid KB / playbook pieces; drops and reports what does not fit |
| `acie/productgen/criteria.py` | `check(bundle, spec, eff)` -> report of pass/fail checks |
| `acie/productgen/generate.py` | the staged pipeline and the repair loop |
| `acie/productgen/store.py` | versions, diffs, Publish (three-way merge) |
| `acie/productgen/expand.py` | expand a node on demand; mine call records for gaps; approve proposals |
| `acie/productgen/api.py` | `/api/productgen/*`, the job runner, the `/productgen` page route |
| `voice/static/productgen.html` | the page (same look as the console) |
| `scripts/productgen/run.py` | the same pipeline from the command line |
| `tests/test_productgen_*.py` | tests with a fake LLM (`test_productgen_fake.py`) |

## Operator inputs (`Spec`)
Only `name` is required. Optional: category, seller / brand, website, notes (description), city / market (default India), languages
(en always; hi, hinglish), price (list, offer, unit, validity), economics (floor, or cost + min margin %, an operator ladder, allowed
perks), tone, audience, compliance notes, extra banned phrases, `authorised` ("I own / am authorised to sell this"), `real_brand`
(override), tenant, product id, and `criteria`.

## Criteria (shown in the UI, saved in every version)
Quantity checks only warn; the others block Publish. Defaults:

| check | rule | blocks Publish |
|---|---|---|
| topics | 15-30 top-level topics (subtopics unlimited) | no |
| entries | at least 150 Q&A entries | no |
| offerings | 5-10 | no |
| situations | 14-20 top-level playbook situations | no |
| ladder | discovery ladder of at least 4 rungs (goal, sub-goal, experience, schedule, preference ...) | no |
| languages | every entry, topic, offering, ladder rung in en + hi + hinglish; Hindi in Devanagari, Hinglish in Roman; playbook nodes have `points_hi` / `ask_hi` | yes |
| provenance | every entry and offering labelled; only `public` for a real brand, `synthetic` / `operator` otherwise | yes |
| real_brand | real brand: `allow_synthetic=false`, all claims marked for review, no invented prices | yes |
| numeric | every offering price >= floor, offer < list, concession ladder steps >= floor | yes |
| banned | no health / result guarantees, financial guarantees, false urgency (unless the operator's own offer validity says it), competitor disparagement, invented certifications (ISO, FSSAI, FDA ... ; for a real brand also "certified", "licensed" ...), operator-banned phrases | yes |
| schema | the KB passes `acie.knowledge` validation (slots, no price literals in answers, unknown refs) | yes |
| coverage | every topic has entries, every interest tag is used | no |
| duplicates | no duplicate questions | no |

## Real-brand rule
If the name or seller looks like a real company's product (a built-in list of well-known brands, or `real_brand: true`) and the operator
did not tick "I own / am authorised", the run is **public-knowledge only**: provenance `public`, `allow_synthetic=false`, every entry's
`source` starts with `review:` and confidence is capped at 0.7, offerings carry no price (a real company's prices are theirs), no
negotiation is suggested, `allowed_numbers` is set to a placeholder so dialing stays blocked until the operator enters test phones, and
the quantity criteria relax (topics >= 5, Q&A >= 30, offerings >= 0, situations >= 10) so the model is never pushed to invent facts to
hit a number. The effective criteria are saved with the version. A real-brand product is never auto-published.

## Pipeline
1. **profile**: pitch (3 languages), benefits, FAQs, knowledge paragraph, topic words, discovery question, pronunciations, tone. Price strings (`offer_price`, `original_price`, `discount`) are computed from the numbers, never by the model.
2. **info_tree**: `kb_plan` (topics, interest tags, offerings), then `kb_entries` per group of topics (concurrent, merged in a fixed order). Provenance is forced from the spec, never trusted from the model; price below floor is raised to the floor; prices only as slots.
3. **decision_tree**: `playbook` (situations, with labels filtered to the ones the picker knows) and `ladder` (discovery rungs + offering fit rules). Interest clusters are derived deterministically (connected components of the tags' `related` graph).
4. **negotiation**: only with a floor (explicit, or cost + margin) and not for a real brand. Price steps below the floor or at/above the price are dropped; perks come from the operator's allowed list when given. With no floor nothing is suggested: the generator never invents a floor.
5. **validate**: deterministic fixes (trim extras, drop empty subtopics, unused tags, banned FAQs / pitch lines), then up to `max_repair_rounds` (3) repair rounds: the failing checks' error text goes back to the model for only the failing items (`kb_fix`, more `kb_entries`, `playbook`, `ladder`, ...). Entries still non-compliant after the rounds are removed, never shipped.
6. **save**: `data/voice/productgen/<tenant>/<product_id>/v<N>.json` (spec, criteria, bundle, report, repair history) + `v<N>.diff.json`. A version with a failed blocking check is `needs_review` and cannot be published.

## Publish
`POST /api/productgen/publish` writes `products.json`, `data/voice/knowledge/<tenant>/<id>.json` (or `ACIE_KB_DIR`) and
`data/voice/playbooks/<tenant>/<id>.json`, plus `<id>.ladder.json` next to the KB (the discovery ladder and fit rules, for consumers). It is a
**three-way merge**: last published version = base, live file = current, this version = new. Fields, entries, topics, offerings,
tags and nodes the operator changed in the console since the last publish are kept; things removed in the console stay removed;
new generated items arrive; untouched items take the new text. With no earlier publish, whatever already exists wins on conflict. Files that are
overwritten are copied first to `productgen/<tenant>/<id>/backups/<time>/`. Nothing is written unless the merged result validates
(Product model, KB schema and rules, profit floor). A hand-written `<id>.tree.json` in `data/playbooks` still beats the generated playbook (see `acie/playbook.py`).

## Self-extension
* **Expand** (`POST /api/productgen/expand {node: {kind: topic|entry|playbook, id}}`): the model proposes subtopics and entries under a topic, or more specific child situations under a playbook node, given the existing tree (existing subtopics, existing questions). Only items inside that subtree are accepted; the result is checked, banned / non-compliant additions are dropped, and the run **asserts every pre-existing entry, topic, offering, tag and node is byte-identical** before saving a new version.
* **Gap mining** (`GET /api/productgen/gaps` is deterministic and free; `POST /api/productgen/gaps/scan` asks the model): scans `data/voice/calls/*.json` for the product's customer turns whose next agent turn was a fallback (intent in a fixed set, an "I don't have that / check the website" line, or the product's own `info_source` text) or that score below the retrieval threshold against the KB. Terms are canonicalised with `acie.knowledge.text`, clustered greedily by Jaccard >= 0.5 over sorted files, ranked by count then id; a cluster the KB now answers confidently is dropped. Per cluster the model proposes entries (and new topics / situations when needed), each marked with provenance and `source: gap:<call ids>`. They are queued as a **pending version** (`kind: gap_proposals`, no bundle). `POST /api/productgen/approve-gaps {version, approve[], reject[], publish}` builds a new full version from the latest bundle plus the approved proposals and optionally publishes it. Seen / approved / rejected cluster ids are kept in `state.json` so the same question is not proposed twice.
* `ACIE_PRODUCTGEN_AUTO_APPROVE=1` auto-approves and publishes proposals for synthetic (demo / own) products only. Real-brand products are never auto-published.

## API (all admin-auth, same cookie as the console)
| method and path | body / query | returns |
|---|---|---|
| GET `/productgen` | | the page |
| GET `/api/productgen/defaults` | | `{criteria, languages, llm_configured, auto_approve}` |
| POST `/api/productgen/brand-check` | `{name, seller?, authorised?, real_brand?}` | `{real_brand, brand_match, provenance, allow_synthetic, notes, product_id}` |
| POST `/api/productgen/start` | `{spec}` | `{job_id}` (422 invalid spec, 503 no model configured) |
| GET `/api/productgen/jobs`, `/jobs/{id}` | | `{id, kind, status, stages: [{id, label, status, detail}], version, error, result}` |
| POST `/api/productgen/jobs/{id}/cancel` | | `{ok, status}` |
| GET `/api/productgen/products` | | `{products: [{tenant, product_id, name, versions, latest, published}]}` |
| GET `/api/productgen/versions` | `product_id, tenant` | `{versions: [summary], published}` |
| GET `/api/productgen/version` | `product_id, tenant, n` | `{version: {spec, effective, bundle, report, repair_history, ...}, diff, published}` |
| GET `/api/productgen/diff` | `product_id, tenant, a, b` | `{a, b, diff}` |
| POST `/api/productgen/publish` | `{product_id, tenant, version}` | `{ok, version, files, entries, nodes, backup}` (409 when blocked) |
| POST `/api/productgen/expand` | `{product_id, tenant, node: {kind, id}, instruction?, version?}` | `{job_id}` |
| GET `/api/productgen/gaps` | `product_id, tenant` | `{unanswered: [cluster], pending: [{version, proposals}]}` |
| POST `/api/productgen/gaps/scan` | `{product_id, tenant}` | `{job_id}` |
| POST `/api/productgen/approve-gaps` | `{product_id, tenant, version, approve[], reject[], publish}` | `{approved, rejected, version, report, published?}` |

Jobs are background asyncio tasks held in memory (a restart forgets finished job records; the saved versions remain).

## Model configuration
`PRODUCTGEN_BASE_URL`, `PRODUCTGEN_MODEL`, `PRODUCTGEN_API_KEY`, `PRODUCTGEN_EXTRA_BODY` (JSON merged into the request) override the production Gemini
settings `ACIE_EXPR_BASE_URL`, `ACIE_EXPR_MODEL`, `ACIE_EXPR_API_KEY`, `ACIE_EXPR_EXTRA_BODY`. Other knobs: `PRODUCTGEN_MAX_TOKENS` (16000),
`PRODUCTGEN_TIMEOUT_S` (300), `PRODUCTGEN_RETRIES` (3), `PRODUCTGEN_JSON_MODE=0` to drop `response_format`, `PRODUCTGEN_CONCURRENCY` (3),
`PRODUCTGEN_TOPICS_PER_CALL` (3). Temperature is 0 with a fixed seed. Keys are only sent in the Authorization header: never logged, never
in a saved version, never in an error message.

## Running the real thing (needs the server's model settings; they are not on the laptop)
```
cd /opt/salesgod/app && set -a && . /opt/salesgod/.env && set +a
python -m scripts.productgen.run --name "ZenBrew Coffee Subscription" --list 1200 --offer 999 --unit "per month" --floor 600
# review in the page: https://<voice host>/productgen?product=default/zenbrew-coffee-subscription ; add --publish to write it live
```
