# salesgod — ACIE

Adaptive Conversational Intelligence Engine: the layer of a voice agent between "transcript in" and "text to TTS out".

**Decision intelligence** (what to say) is deterministic. **Interaction intelligence** (how to say it, for this person, right now) is adaptive. The LLM only does the wording, and it never writes a number itself.

See [docs/CONTEXT.md](docs/CONTEXT.md) for the original concept.

Current documentation: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) (every model, formula and threshold, with file:line; open `docs/ARCHITECTURE.html` locally for the browsable version), [docs/FLAGS.md](docs/FLAGS.md) (every env var and flag), [docs/STATUS.md](docs/STATUS.md) (what is done and left against CONTEXT.md). The layer table below is the original v1 summary.

## Layers

| # | Layer | Module | v1 implementation | Upgrade path |
|---|-------|--------|-------------------|--------------|
| 2 | Human State Estimator | `acie/state/estimator.py` | Phrase/timing signals + asymmetric EMA (rise fast, decay slow) | Fine-tuned XLM-R/MuRIL classifier behind `RawScorer` |
| 3 | Personalization | `acie/personalization/` | Long-term profile with confidence + explicit locks; rule policy with hysteresis | Postgres store; contextual bandit over the same discrete params |
| 4 | Decision Engine | `acie/decision/` | Keyword intents + per-domain FSM + mock backend | Small intent classifier; real APIs |
| 5 | Prompt Compiler | `acie/compiler.py` | Plan → system prompt; facts passed as `{slots}` only | — |
| 6 | Expression Engine | `acie/expression.py` | vLLM/OpenAI-compatible streaming, or template backend | Fine-tuned ~8B (LoRA) for Hindi/Hinglish/English |
| 7 | Validator | `acie/validator.py` | Per-sentence streaming checks: no raw numbers, no invented slots, all facts present → else safe template | Domain-specific rule packs |
| 8 | Feedback | `acie/feedback.py` | Explicit prefs lock; weak implicit updates | Outcome-driven policy learning |

Orchestration: `acie/pipeline.py`. Intent + state estimation run in parallel with profile loading. Validated sentences stream out through `on_sentence` so TTS can start early.

### Adaptation parameters (A_t)

`language` (en / hi / hinglish), plus `formality`, `verbosity`, `complexity`, `warmth` and `proactivity`, each one of low / med / high. There is also an `acknowledge_emotion` flag. The levels are discrete on purpose, because LLMs can't reliably follow fine-grained values like 0.4 vs 0.5.

### Human state (H_t)

`frustration`, `confusion`, `urgency`, `engagement` ∈ [0, 1]. Each value comes with the evidence (`signals`) that produced it.

## Run

```bash
python -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/pytest -q
.venv/bin/python -m acie.cli --debug                     # template backend, no GPU
ACIE_LLM_BASE_URL=http://<voice-box>:8000/v1 ACIE_LLM_MODEL=<model> \
  .venv/bin/python -m acie.cli --backend openai --debug  # real LLM via vLLM
```

Demo customers: `c1` (card-eligible, ₹25k of the daily limit left) and `c2` (not eligible).

## Hardware

- **voice**: 4× A100 80 GB. Serves the live pipeline and also runs training and teacher-model data generation.
- **test3-ocr**: 1× A100 40 GB. Extra training capacity.
