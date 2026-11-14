# Status: `docs/CONTEXT.md`, layer by layer

Snapshot: `lane/docs` from `build/acie-complete` @ `c2e6be6` (658 tests collected). Each row maps a requirement of `docs/CONTEXT.md` to the code that implements it.
"Live" means on in the production voice path by default in code; "flag" means built and tested but off until the env var in `docs/FLAGS.md` is set; "not merged" means it exists only on another branch.
Architecture and formulas: `docs/ARCHITECTURE.md`.

Legend: DONE = built and live by default; FLAG = built, off; PARTIAL = part built; LEFT = not built.

---

## 1. Objective and the core split (CONTEXT section 1, 2, 8)

| requirement | status | where |
|---|---|---|
| LLM only realises wording; the response is decided by deterministic systems | DONE | plan is built before the LLM (`acie/pipeline.py:267-277`); `llm_allowed=False` plans skip the LLM (`:318`); validator guards the rest (`acie/validator.py`) |
| deterministic before the LLM (owner rule: no sampling, no `hash()`, fixed tie-breaks) | PARTIAL | everything after hearing is deterministic. The hearing step is deterministic only without the `llm` provider; the live voice server runs the NLU (`voice/server.py:111`). Fix available today: `ACIE_SIGNAL_PROVIDERS=rules` (loses buying signal, garbled flag, learned votes). Expression LLM stays an LLM by design |
| decision intelligence vs interaction intelligence separation | DONE | `acie/decision/` vs `acie/personalization/`, `acie/compiler.py`, `acie/expression.py` |

## 2. Layer 1: STT input (out of scope in CONTEXT, built anyway for the phone path)

| item | status | where |
|---|---|---|
| streaming transcript in | DONE | Whisper large-v3 (`voice/stt.py`), Silero VAD (`voice/vad.py`), caller-voice lock (`voice/speaker.py`), barge-in, noise/hallucination filters (`voice/call.py`) |
| telephony | DONE | Plivo (`voice/plivo.py`, `voice/server.py`) |
| TTS | DONE | Fish Audio (`voice/tts.py`) |
| left | LEFT | nothing in scope; measured latency is stored in call logs only (`voice/call.py:533`) |

## 3. Layer 2: Human State Estimator (CONTEXT 3A)

| CONTEXT group / dims | status | where |
|---|---|---|
| Satisfaction (liking, disliking, satisfaction) | DONE | `satisfaction`, `acie/state/dims.py:242` (one dim covers like and dislike) |
| Emotion: frustration, anxiety, excitement, calmness | DONE | `acie/state/estimator.py:227-236`, `dims.py:249-272` |
| Engagement: attention, participation, receptiveness (+ engagement) | DONE | `dims.py:316-359`, `estimator.py:245-253` |
| Comprehension: confusion, cognitive load | DONE | `estimator.py:216-225`, `dims.py:292-314` |
| Interaction: urgency, trust, certainty | DONE | `estimator.py:238-243`, `dims.py:274-290` |
| all 14 dims in `HumanState`, in [0,1], with evidence strings | DONE | `acie/schemas.py:80-98` |
| smoothing: rise fast, decay slow, per-dim neutral baseline | DONE | `estimator.py:396-446` (alpha 0.7 / 0.3) |
| voice (pitch, loudness, effort, rate vs the same caller's baseline) | PARTIAL | calibrator computes a reading every turn and the emotion table can read it (`voice/calibrator.py`); the estimator uses it only with `ACIE_PROSODY=1` (FLAG, `estimator.py:434`, off) |
| trained classifier replacing the rules | PARTIAL | seam done: `ACIE_TURN_MODEL` + `model` provider + 50/50 fusion (`estimator.py:356-390`, `providers/trained.py`). On this commit the trainer is MuRIL with only 4 state dims (`training/state_classifier/data.py:35`) and `TrainedTurnModel` has no `.load`, so the flag falls back to the rules. The 14-dim deterministic XGBoost classifier is on `lane/ml` (`c7e0562`), NOT MERGED. No trained checkpoint is in the repo |
| LLM reading of the 14 dims | FLAG | `ACIE_NLU_STATE_DIMS=1` (`acie/understanding.py:66`); the live NLU returns the 4 core dims |
| labelled data for the 14 dims | LEFT | the held-out eval set (`data/eval/turns.jsonl`, 369 synthetic turns) labels the 4 core dims only (`docs/DATA_SCHEMA.md`); real calls: 169 caller turns labelled for needs, not for dims |

## 4. Layer 3: Personalization (CONTEXT 3B)

| CONTEXT item | status | where |
|---|---|---|
| language history, regional variant | DONE | `preferences["language"]`, `regional_variant` (delhi, mumbai, lucknow, bihar, south_indian_english), `learner.py:160-175` |
| tone / formality, vocabulary complexity, technical depth | DONE | `formality`, `warmth`, `complexity` (jargon evidence), `learner.py:198-228` |
| preferred explanation style, information preferences | DONE | `explanation_style`, `info_density`, `learner.py:230-244` |
| likes, dislikes, topics, past interactions | DONE | `CustomerProfile.likes/dislikes/topics/outcomes/interaction_count` (`schemas.py:111-120`), lists capped at 20 |
| long-term vector with confidence + explicit overrides | DONE | `update_preference` (`profile_store.py:52-75`), act-on threshold 0.6 (`policy.py:84`) |
| short-term vector: verbosity, tone, reassurance, language adjustment, complexity, proactivity, density | DONE | `ExpressionParams` (`schemas.py:123-136`) |
| `A_t = pi(H_t, P_t, C_t)` rule policy with hysteresis | DONE | `policy.py:123-241` (Schmitt triggers, language streak, acknowledgement cooldown) |
| learned adaptation policy | FLAG | LinUCB posterior-mean (`ACIE_POLICY=bandit`) and offline XGBoost (`ACIE_POLICY=xgb`); no exploration; rule policy is the prior; moves a knob only after 30 observations per arm and a 0.1 gain. No logged training rows exist yet because the live policy is the rule policy |
| continuous (0.5) values in the CONTEXT example | DEVIATION | levels are discrete low/med/high by design (LLMs do not follow 0.4 vs 0.5); documented in `README.md` |
| storage | DONE / FLAG | JSON per customer by default; Postgres with `ACIE_PROFILE_STORE=postgres` (`personalization/pg_store.py`) |

## 5. Layer 4: Response determination (CONTEXT 3C)

| item | status | where |
|---|---|---|
| deterministic engine, structured `TurnPlan` (never prose) | DONE | `acie/decision/engine.py`, `acie/schemas.py:156-184` |
| FSM / rule engine for the sales call | DONE | `acie/decision/domains/product_sales.py:762-985` (gate order in `acie/trace.py`) |
| intent classifier: keyword rules | DONE | `engine.py:36-117` |
| intent classifier: small transformer / learned model | PARTIAL | seam done (`HybridIntentClassifier`, `providers/trained.py`); model not trained/merged (see section 3) |
| model-agnostic input contract | DONE | `TurnSignals`, providers, deterministic fusion (`acie/signals*.py`, `acie/providers/`, `docs/MODEL_AGNOSTIC.md`) |
| business rules / constraints | DONE | `decision/constraints.py` (min_age, region, existing subscriber, price floor, call hours, DNC, re-offer cap) |
| database queries / business APIs | PARTIAL | `acie/business/`: mock API is what runs; `HTTPBusinessAPI` exists but nothing calls `from_env()` (`acie/business/http.py:100`), so a real API is not wired into the voice server |
| compliance and safety | DONE | crisis row always on (`acie/safety.py`, `psyche/emotion_policy.py`); opt-out, AI admission, never take card/OTP, consent + full terms before the link, at most one re-offer |
| whole-call memory (promised / quoted / told / decision) | DONE | `acie/context.py` |
| playbook decision tree | DONE | deterministic picker (`acie/playbook.py:61-107`); authoring needs `ACIE_PLANNER_*` |
| next-best-action scorer | FLAG | `ACIE_NBA=1` (`decision/nba.py`) |
| repetition ("it repeats itself", the top live complaint) | DONE | always-on repeat guard (`pipeline.py:148-173`); full ledger-driven moves with `ACIE_PSYCHE=1` (FLAG). Replay: 56 repeats with the guard off, 0 with it (`docs/REAL_CALLS_FINDINGS.md`) |
| needs, moves, emotion rows (non-crisis) | FLAG | `ACIE_PSYCHE=1`; crisis row is DONE |
| negotiation | PARTIAL | there is NO negotiation state machine. A request for a lower price gets one fixed "the price is fixed, no pressure" plan and the call continues (`product_sales.py:888-895`); it is never read as a decline (`tests/test_negotiate_not_decline.py`). No counter-offers, no discount ladder (and none is wanted under the no-profit rule) |
| no profit optimisation | DONE | sales outcomes have reward weight 0 (`feedback.py:66`); selling is a single gated move (`psyche/moves.py:389-403`). Caveat: `nba.py` scores closing actions, but only when `ACIE_NBA=1`, and only as a hint |

## 6. Layer 5: Structured Prompt Compiler (CONTEXT 4)

| item | status | where |
|---|---|---|
| response spec + expression params -> prompt | DONE | `acie/compiler.py` |
| facts as `{slot}` placeholders only; constraints (preserve facts, no invention, keep decision) | DONE | `compiler.py:121-141,152-165` |
| regional flavour, explanation style, density, reassurance lines | DONE | `compiler.py:37-93` |
| JSON shape in CONTEXT (`response` / `expression` / `constraints`) | DEVIATION | the prompt is a system message of rules plus a `PLAN` JSON (`objective, decision, facts, talking_points, allowed_actions, must_not_say, ask, ...`); the constraints are the HARD RULES text and the validator, not JSON keys |
| profile and memory blocks in the prompt | FLAG | only with `ACIE_MEMORY=1` (`pipeline.py:275`) |

## 7. Layer 6: LLM Expression Engine

| item | status | where |
|---|---|---|
| streaming OpenAI-compatible backend, template backend, failover, A/B | DONE / FLAG | `acie/expression.py` (failover and A/B need env, section FLAGS) |
| live model | DONE | Qwen3-14B-AWQ on vLLM (`deploy/voice/salesgod-llm.service`), temperature 0.3 |
| fine-tuned LoRA (7-8B) | PARTIAL | full pipeline exists: seed data, teacher generation, `train_lora.py`, `merge_lora.py`, `eval_expression.py` (`training/expression/`). No trained adapter is deployed; the live unit serves the base model. Seed set 142 gold rows (`seed_data.py`, `training/expression/README.md`) |
| temperature 0 | FLAG | with `ACIE_PSYCHE=1` (`expression.py:104`) |

## 8. Layer 7: Response Validation

| item | status | where |
|---|---|---|
| per-sentence streaming validation, safe template on failure | DONE | `acie/validator.py` |
| no LLM-written numbers, no invented slots, all facts present | DONE | `validator.py:24-28,144-163,90-96` |
| language/script check | DONE | `validator.py:182-190` |
| domain rule packs | DONE | `acie/rules/` (`sales_compliance`, `banking`, `hindi_register`), selected per plan (`TurnPlan.rule_packs`) |
| one regeneration with the reason before the template | DONE | `pipeline.py:337` |

## 9. Layer 8: Feedback and long-term memory

| item | status | where |
|---|---|---|
| explicit preferences lock immediately; implicit evidence weak | DONE | `feedback.py:157-170`, `learner.py` (LR 0.1 / 0.05) |
| outcomes recorded per call | DONE | `feedback.py:103-115` (safety, opt_out, accepted, declined, later, no_answer, human_agent) |
| per-turn reward from the customer's reaction | DONE (computed) | `feedback.py:128-133`; used only if the policy has `observe`, i.e. FLAG policies |
| call-level credit (no pushing, respected no, no repeats) | DONE (computed) | `feedback.py:64-65,255-286` |
| validator-driven learning (`prefer_templates`) | DONE | `feedback.py:200-210`; whether the decision layer reads `prefer_templates` is not wired in `product_sales.py` (the key is written, not read there) |
| semantic memory | FLAG | `ACIE_MEMORY=1`; hashing embedder or e5-small; pgvector with `ACIE_MEMORY_STORE=pgvector` |
| continuous personalization improvements (CONTEXT roadmap 7) | LEFT | needs logged rows and a switched-on learned policy; nothing is learning in production |

## 10. Runtime architecture (CONTEXT 5)

| CONTEXT item | status | where |
|---|---|---|
| Python asyncio orchestration, stream A (understand) and stream B (profile) in parallel | DONE | `pipeline.py:222` |
| context merger | DONE | `TurnSignals` + fusion (`acie/signals_fusion.py`) |
| session state in Redis | FLAG | `ACIE_SESSION_STORE=redis` (`acie/store/sessions.py`); in-process dict otherwise |
| customer database in PostgreSQL | FLAG | `ACIE_PROFILE_STORE=postgres` |
| semantic memory in pgvector | FLAG | `ACIE_MEMORY_STORE=pgvector` |
| event processing in NATS | FLAG | `ACIE_NATS_URL` (`acie/events.py`) |
| intent classification: small transformer | PARTIAL | MuRIL/XLM-R trainer exists; XGBoost classifier on `lane/ml`; nothing deployed |
| state estimation: lightweight ML classifier | PARTIAL | same |
| Pydantic schema validation | DONE | `acie/schemas.py`, `acie/signals.py` |
| LLM: Qwen / GPT / Claude | DONE | any OpenAI-compatible endpoint (`ACIE_LLM_*`, `ACIE_EXPR_*`); live = Qwen3-14B-AWQ |

## 11. Architectural principles (CONTEXT 6)

| # | principle | status | evidence |
|---|---|---|---|
| 1 | the LLM is not the source of business truth | DONE | slots + validator |
| 2 | determination and generation independent | DONE | plan before compile |
| 3 | long-term preferences and temporary states separate | DONE | `CustomerProfile` vs `HumanState` / `session.expression` |
| 4 | confidence scores and explicit overrides | DONE | `update_preference` |
| 5 | multidimensional personalization | DONE | 14 dims, 11 knobs |
| 6 | no oscillation | DONE | EMA, Schmitt triggers, language streak, acknowledgement cooldown |
| 7 | feedback updates memory | DONE / FLAG | profile learning is on; semantic memory and policy learning are flags |
| 8 | critical facts validated before speech | DONE | `validator.py` |
| 9 | optimise comprehension and satisfaction, not manipulation | DONE | reward has no sale term; `sales_compliance` pack blocks false urgency and pressure |

## 12. Roadmap phases (CONTEXT 7)

| phase | deliverable | status |
|---|---|---|
| 1 | deterministic response engine | DONE |
| 2 | customer profile and personalization vectors | DONE |
| 3 | human state estimation and adaptation policy | DONE (rules); learned model PARTIAL |
| 4 | structured prompt compiler and LLM integration | DONE |
| 5 | response validation | DONE |
| 6 | feedback and long-term memory | DONE; memory FLAG |
| 7 | continuous personalization improvements | LEFT (learned policies built, off, no training data) |

## 13. "Next steps" questions (CONTEXT, last section)

1. Complete state-vector schema: DONE (14 dims, `docs/ARCHITECTURE.md` section 3).
2. Learned vs configured parameters: DONE in code and docs; configured = all formulas and tables; learned (offline, fixed params) = need model (live path only with `ACIE_PSYCHE=1`), style arms (flag), turn classifier (not merged).
3. Which ML model estimates each parameter: PARTIAL; today every dim is estimated by rules (plus the LLM NLU for the 4 core dims); the trained model is the open item.
4. How preferences are stored and updated: DONE (`profile_store.py`, `learner.py`, `feedback.py`).
5. How the policy maps state changes to communication changes: DONE (`policy.py`, tables in `docs/ARCHITECTURE.md`).
6. Evaluating whether personalization helps: PARTIAL. Signal audit and scripted-call eval exist (`docs/EVAL.md`); the real-call replay measures repetition and determinism only, not satisfaction (`docs/REAL_CALLS_FINDINGS.md`). No A/B on real callers yet (`ACIE_EXPR_AB` is a model split, not a style experiment).

---

## 14. What is left, ranked

1. Merge `lane/ml` (XGBoost 14-dim classifier) and train it on real + synthetic rows; then set `ACIE_SIGNAL_PROVIDERS=rules,model` so hearing no longer needs the LLM NLU.
2. Decide on `ACIE_PSYCHE=1` in production (needs, moves, emotion rows, temperature 0); the replay shows 0 repeats with it, but it is untested on live callers.
3. Wire a real business API into the voice server (`acie/business/http.py:from_env()` is never called).
4. Make the decision layer read `prefer_templates` (written by `feedback.py:208`, never read in `acie/decision/`).
5. Log (context, action, reward) rows from live calls so the XGBoost style policy can be trained offline; no row exists today.
6. Deploy a fine-tuned expression adapter (or keep the base model and drop the LoRA pipeline from the plan).
7. Switch on and load-test Redis sessions, Postgres profiles, pgvector memory and NATS events (code and tests exist, nothing runs them in production).
