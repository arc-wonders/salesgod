# SalesGod (ACIE) flags and environment variables

Source: every `os.getenv` / `os.environ` read in `acie/`, `voice/`, `training/` at `lane/docs` (from `build/acie-complete` @ `c2e6be6`).
"Since" = whether the name already existed at the import commit `6e4b64e` (`pre`) or was added by the completion lanes (`new`).

**Live status.** The production `.env` (`/opt/salesgod/.env`, read by `deploy/voice/salesgod-voice.service:8`) is not in the repo, so a live value
can only be stated for what the code or the service units fix. Rule used here: every flag marked `new` is unset in production (per the
lane brief), so the default in the "Default" column is what runs. Rows marked `pre` are the original configuration surface; whether they are set
on the VM cannot be verified from the repo.

"Off" below means: with the variable unset, today's behaviour is unchanged.

---

## 1. Flags that turn on a behaviour (all off by default, since = new unless stated)

| variable | default | turns on | layer | where |
|---|---|---|---|---|
| `ACIE_PSYCHE` | off (`1`, `true`, `yes`, `on` enable) | need table, moves, SaidLedger-driven wording, the 10 non-crisis emotion rows, and expression temperature 0 | 2b / 4 / 6 | `acie/psyche/__init__.py:6`, `acie/expression.py:104`, `acie/pipeline.py:234,254` |
| `ACIE_SIGNAL_PROVIDERS` | unset = `rules`, plus `llm` when the pipeline has an NLU | an explicit ordered provider list, e.g. `rules`, `rules,llm,model`; with an explicit list the rules are the plain rules and models only enter as providers | 1-2 / 4 | `acie/providers/hub.py:19`, `acie/providers/registry.py:354` |
| `ACIE_TURN_MODEL` | unset | loads a trained turn classifier from this directory (XGBoost JSON dir, or a MuRIL dir with `encoder/`) as the `model` provider, the hybrid classifier and the state scorer blend | 2 / 4 | `acie/providers/loaders.py:44`, `acie/state/estimator.py:381`, `acie/providers/registry.py:335` |
| `ACIE_POLICY` | unset = rule policy | `bandit` = deterministic LinUCB posterior-mean style policy; `xgb` = offline-trained XGBoost style policy | 3 | `acie/personalization/policy.py:248` |
| `ACIE_MEMORY` | `0` | semantic memory: store a call summary at call end, recall top 3 at turn 0, put profile and memories in the prompt | 8 / 5 | `acie/memory/base.py:31`, `acie/pipeline.py:275` |
| `ACIE_NBA` | off (`1` enables) | next-best-action scorer, one hint line in the plan objective | 4 | `acie/decision/nba.py:35` |
| `ACIE_NLU_STATE_DIMS` | `0` | the NLU also returns the 10 extended dims (`max_tokens` 110 -> 230) | 1-2 | `acie/understanding.py:66` |
| `ACIE_SESSION_STORE` | `memory` | `redis`: durable session copy with TTL | runtime | `acie/store/__init__.py:23` |
| `ACIE_PROFILE_STORE` | `json` | `postgres`: profile store in Postgres | runtime | `acie/store/__init__.py:28` |
| `ACIE_MEMORY_STORE` | `memory` | `pgvector`: semantic memory in Postgres + pgvector | runtime | `acie/memory/base.py:48`, `acie/store/__init__.py:35` |
| `ACIE_NATS_URL` | unset | publish call-context events to NATS | runtime | `acie/store/__init__.py:47` |
| `ACIE_PROSODY` | `0` (pre) | pitch / loudness arousal feeds the state (`apply_prosody`), and the "background voice" noise check in the call | 2 / voice | `acie/state/estimator.py:434`, `voice/call.py:492` |
| `ACIE_EMBED` | `hashing` | `st`: sentence-transformers embedder for memory (falls back to hashing if it cannot load) | 8 | `acie/memory/base.py:36` |
| `ACIE_EXPR_AB` | unset | split sessions between expression models, `"modelA:0.5,modelB:0.5"`, sticky per session | 6 | `acie/expression.py:264` |
| `ACIE_EXPR_FALLBACK_BASE_URL` | unset | second expression backend used if the primary fails before any text | 6 | `acie/expression.py:271` |
| `ACIE_EXPR_FIRST_TOKEN_S` | unset (off) | no first token within N seconds -> speak the safe template (or fail over) | 6 | `acie/expression.py:112` |

## 2. Flags that are ON by default and can be turned off

| variable | default | effect | where |
|---|---|---|---|
| `ACIE_REPEAT_GUARD` | `1` (on) | `0` disables the always-on repeat guard (a deterministic reply that is a near copy, ratio > 0.6, of an earlier whole reply is replaced). Only for measuring | `acie/pipeline.py:151` |
| `ACIE_NEED_MODEL` | unset = use `acie/psyche/need_model.json` if present | `off` disables the XGBoost need refinement (the table alone decides); any other value is read as a model path | `acie/psyche/needs.py:214-217` |
| `PLIVO_VERIFY_SIGNATURE` | `1` | `0` skips Plivo webhook signature checks | `voice/server.py:210` |

## 3. Tuning values (only read when the flag above is on)

| variable | default | used by | layer | where |
|---|---|---|---|---|
| `ACIE_TURN_MODEL_THRESHOLD` | `0.6` | minimum confidence for a trained model to replace the rules' intent | 4 | `acie/signals_fusion.py:44`, `acie/decision/hybrid.py:34` |
| `ACIE_INTENT_ALIASES` | `{}` | JSON object, foreign intent label -> domain label | 4 | `acie/signals.py:180` |
| `ACIE_SIGNAL_HTTP_URL` | unset | the `http` provider endpoint; the provider exists only when this is set | 4 | `acie/providers/http.py:26`, `acie/providers/registry.py:51` |
| `ACIE_SIGNAL_HTTP_TOKEN` | empty | bearer token for that endpoint | 4 | `acie/providers/http.py:29` |
| `ACIE_SIGNAL_HTTP_TIMEOUT_S` | `1.5` | request timeout; provider budget = this + 0.5 s | 4 | `acie/providers/http.py:27` |
| `ACIE_BANDIT_MIN_OBS` | `30` | observations an arm needs before the LinUCB policy may move a knob | 3 | `acie/personalization/bandit.py:191` |
| `ACIE_BANDIT_MARGIN` | `0.1` | predicted-reward gain required to leave the rule action | 3 | `acie/personalization/bandit.py:192` |
| `ACIE_BANDIT_ALPHA` | `1.0` | optimism weight of the offline-only `ucb` bound; never used to decide | 3 | `acie/personalization/bandit.py:193` |
| `ACIE_BANDIT_STATE` | `data/bandit.json` | LinUCB state file (git-ignored) | 3 | `acie/personalization/bandit.py:197` |
| `ACIE_STYLE_MIN_OBS` | `30` | same, for the XGBoost style policy | 3 | `acie/personalization/xgb_style.py:34` |
| `ACIE_STYLE_MARGIN` | `0.1` | same, for the XGBoost style policy | 3 | `acie/personalization/xgb_style.py:35` |
| `ACIE_STYLE_MODEL` | `data/style_xgb.json` | trained XGBoost style model file | 3 | `acie/personalization/xgb_style.py:36` |
| `ACIE_STYLE_LOG` | `data/style_log.jsonl` | rows logged for the next offline training run | 3 | `acie/personalization/xgb_style.py:37` |
| `ACIE_REWARD_WEIGHTS` | empty | override reward weights, `"frustration=0.9,barge=0.2"` (keys: frustration, confusion, positive, validator, barge) | 8 | `acie/feedback.py:75` |
| `ACIE_EMBED_MODEL` | `intfloat/multilingual-e5-small` | model for `ACIE_EMBED=st` | 8 | `acie/memory/st.py:11` |
| `ACIE_MEMORY_DIM` | `256` | embedding size for the pgvector table | runtime | `acie/store/__init__.py:43` |
| `ACIE_REDIS_URL` | `redis://localhost:6379/0` | Redis for sessions | runtime | `acie/store/__init__.py:24` |
| `ACIE_PG_DSN` | required when postgres/pgvector is chosen | Postgres DSN | runtime | `acie/store/__init__.py:30,44` |
| `ACIE_EXPR_AB_SALT` | empty | salt for the sticky A/B hash | 6 | `acie/expression.py:267` |
| `ACIE_EXPR_FALLBACK_MODEL` | backend default | model name of the fallback backend | 6 | `acie/expression.py:273` |
| `ACIE_EXPR_FALLBACK_API_KEY` | backend default (`EMPTY`) | API key of the fallback backend | 6 | `acie/expression.py:274` |
| `ACIE_BUSINESS_API_URL` | unset = mock API | `HTTPBusinessAPI` base URL; read only by `acie/business/http.py:from_env()`, which the voice server does not call (`ProductSalesDomain` defaults to `MockBusinessAPI`, `product_sales.py:432`) | 4 | `acie/business/http.py:24,102` |
| `ACIE_BUSINESS_API_TOKEN` | empty | bearer token for it | 4 | `acie/business/http.py:25` |

## 4. Model endpoints (original configuration surface, `pre` unless noted)

| variable | default | meaning | where |
|---|---|---|---|
| `ACIE_LLM_BASE_URL` | `http://localhost:8000/v1` | OpenAI-compatible server for the NLU and the expression writer (live: local vLLM serving `acie-expression` = Qwen3-14B-AWQ, `deploy/voice/salesgod-llm.service:9`) | `acie/understanding.py:116`, `acie/expression.py:99`, `voice/server.py:233` |
| `ACIE_LLM_MODEL` | `acie-expression` | served model name | `acie/understanding.py:117`, `voice/server.py:234` |
| `ACIE_LLM_API_KEY` | `EMPTY` | API key | `voice/server.py:235`, `acie/expression.py:102` |
| `ACIE_LLM_MAX_TOKENS` | `320` (new) | expression `max_tokens` (Devanagari is token-heavy) | `voice/server.py:236`, `acie/expression.py:105` |
| `ACIE_LLM_EXTRA_BODY` | `{}` | extra JSON merged into every request body (e.g. disable Qwen3 thinking) | `acie/understanding.py:119`, `acie/expression.py:109`, `voice/server.py:237` |
| `ACIE_NLU_TIMEOUT_S` | `2.5` | NLU request timeout; its provider budget is this + 0.5 s | `acie/understanding.py:118` |
| `ACIE_EXPR_BASE_URL`, `ACIE_EXPR_MODEL`, `ACIE_EXPR_API_KEY`, `ACIE_EXPR_MAX_TOKENS` | fall back to the matching `ACIE_LLM_*` | point ONLY the reply writer at another model (e.g. Gemini) while the NLU stays on the local model | `acie/expression.py:99-105`, `voice/server.py:239` |
| `ACIE_EXPR_EXTRA_BODY` | falls back to `ACIE_LLM_EXTRA_BODY` | extra body for the writer, used only when `ACIE_EXPR_BASE_URL` is set | `acie/expression.py:108-109` |
| `ACIE_PLANNER_BASE_URL`, `ACIE_PLANNER_MODEL`, `ACIE_PLANNER_API_KEY` | none / `EMPTY` | frontier model that writes playbooks offline (`generate`, `generate_node`); required for those calls | `acie/playbook.py:175-182,207-217` |

## 5. Voice server (`voice/`)

| variable | default | meaning | where |
|---|---|---|---|
| `PLIVO_AUTH_ID`, `PLIVO_AUTH_TOKEN`, `PLIVO_NUMBER` | required | telephony credentials and caller number | `voice/server.py:209` |
| `PUBLIC_URL` | `https://voice.arkinkansra.in` | public base URL for webhooks | `voice/server.py:55` |
| `ADMIN_PASSWORD` | empty (login refused) | console password | `voice/server.py:56,281` |
| `SESSION_SECRET` | random per start | signs the console cookie | `voice/server.py:57` |
| `VOICE_DATA_DIR` | `<repo>/data/voice` | calls, products, profiles, playbooks | `voice/server.py:38` |
| `VAD_END_SILENCE_MS` | `700` | silence that ends a caller turn | `voice/server.py:58` |
| `VAD_THRESHOLD` | `0.5` | Silero speech probability threshold | `voice/server.py:59` |
| `BARGE_IN_MS` | `800` | continuous speech needed to cut the agent off (the staged voice check in `BARGE_STAGES` is separate, a constant) | `voice/server.py:61`, `voice/call.py:309` |
| `BARGE_IN_RMS` | `0.04` | energy floor for a barge-in | `voice/server.py:62`, `voice/call.py:310` |
| `MAX_PARALLEL_CALLS` | `8` | concurrent calls; more get HTTP 429 | `voice/server.py:60,309` |
| `CALL_START_HOUR`, `CALL_END_HOUR`, `CALL_TIMEZONE` | `9`, `21`, `Asia/Kolkata` | calling window | `voice/server.py:64-65` |
| `CALL_HOURS` | `0` | `1` enforces the calling window when dialling (off by default) | `voice/server.py:140` |
| `PLIVO_AMD_HANGUP` | `0` | `1` hangs up on answering machines detected by Plivo before the caller speaks | `voice/call.py:52` |
| `STT_MODEL` | `large-v3` | Whisper model name | `voice/stt.py:17` |
| `STT_CONCURRENCY` | `4` | parallel transcriptions (semaphore) | `voice/stt.py:19` |
| `FISH_API_KEY` | required | Fish Audio key | `voice/tts.py:15` |
| `FISH_MODEL` | `s2.1-pro-free` | TTS model header | `voice/tts.py:16` |
| `FISH_VOICE_EN`, `FISH_VOICE_HI` | none | Fish reference voice ids per language | `voice/tts.py:17` |
| `VOICE_CALIB_RELOAD` | unset | `1` re-reads `voice/assets/calib_stats.json` for every new call (instead of once per process) | `voice/calibrator.py:291` |

## 6. Training scripts (`training/`, not read at call time)

| variable | default | meaning | where |
|---|---|---|---|
| `TEACHER_BASE_URL` | `http://localhost:8001/v1` | teacher model server for SFT data | `training/expression/teacher_generate.py:200` |
| `TEACHER_MODEL` | `teacher` | teacher model name | `training/expression/teacher_generate.py:201` |
| `TEACHER_API_KEY` | `EMPTY` | key | `training/expression/teacher_generate.py:202` |
| `ACIE_LLM_BASE_URL`, `ACIE_LLM_API_KEY`, `ACIE_LLM_MODEL` | `http://localhost:8000/v1`, `EMPTY`, `acie-expression` | defaults for `eval_expression.py` | `training/expression/eval_expression.py:171-173` |

---

## Safe combinations

* Strictly deterministic hearing: `ACIE_SIGNAL_PROVIDERS=rules` (drops the LLM NLU; the domain then gets no learned `garbled` flag, no `buying_signal`, no `heard(...)` votes).
* Repetition fix plus emotion/need handling: `ACIE_PSYCHE=1` (the repeat guard itself is already on).
* A trained turn model without changing the rules' protected intents: `ACIE_SIGNAL_PROVIDERS=rules,model` and `ACIE_TURN_MODEL=<dir>`; threshold `ACIE_TURN_MODEL_THRESHOLD`.
* Learned style, with nothing explored live: `ACIE_POLICY=xgb` (train offline from `ACIE_STYLE_LOG`) or `ACIE_POLICY=bandit`; both fall back to the rule policy until an arm has 30 observations.
