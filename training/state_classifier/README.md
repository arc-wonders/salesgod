# ACIE turn classifier (intent + human state + explicit requests)

A single multilingual encoder with these heads:

| head | type | classes |
|---|---|---|
| intent | softmax | the 11 intents in `docs/DATA_SCHEMA.md` |
| frustration / confusion / urgency / engagement | softmax, one per dim | 0 / 1 / 2 (engagement: 0 disengaged, 1 normal, 2 high) |
| explicit | multi-label sigmoid | `language=en/hi/hinglish`, `verbosity=low/high`, `complexity=low` |

The input is a sentence pair: `[CLS] <context> [SEP] <customer text> [SEP]`. The context holds the pending dialog question, the pending slot, bucketed timing (barge-in, slow or fast reply, fast or slow speech), the previous customer turn and the previous agent turn. See `build_context` in `data.py`. Training and inference build the context with the same function.

It replaces the v1 rules through the pipeline's existing seams:
- `predict.TrainedTurnModel.score(turn, session)` uses the same interface as `acie.state.estimator.RawScorer`.
- `predict.TrainedTurnModel.classify(text, session)` uses the same interface as `IntentClassifier`. It masks `affirm`/`deny` when no question is pending, like the keyword classifier's gating.
- `predict.TrainedTurnModel.explicit(text, session)` can stand in for `detect_explicit_requests`.

## Data. Read this before training.

- **Train on `data/train/turns.jsonl`**. It uses the same schema as `data/eval/turns.jsonl`, but it must be a *separate* synthetic set produced by the teacher-model generator, with different seeds, templates and paraphrases.
- **Never train, tune or pick checkpoints on `data/eval/`.** That set is the held-out test that `scripts/eval_signals.py` reports. `train.py` refuses any `--train`/`--dev` path under `data/eval/`. It also drops training rows whose normalized text matches an eval text exactly. Exact-match dedup does not catch near-duplicates, so the generator must not see the eval file either.
- Model selection uses a 10% dev split of the training file, split per (lang, intent), or a separate `--dev` file. `--eval data/eval/turns.jsonl` only produces one held-out report after training ends.
- Target size: at least 3k turns per language (en, hi, hinglish), with every intent × state level represented. Rare positives (frustration=2, confusion, disengaged) need deliberate oversampling at generation time. The loss also uses inverse-sqrt class weights.

Check data loading without torch (stdlib only):

```bash
python3 training/state_classifier/data.py --data data/train/turns.jsonl
python3 training/state_classifier/data.py --data scripts/fixtures/smoke_turns.jsonl   # smoke
```

## Train (GPU box, separate env)

Do not install these into the project `.venv`.

```bash
python3 -m venv ~/venvs/acie-train && source ~/venvs/acie-train/bin/activate
pip install -r training/state_classifier/requirements.txt

# MuRIL (default; strongest on Devanagari + romanized Hindi). Fits 1x A100 40GB with batch 64, len 128, bf16.
python training/state_classifier/train.py \
    --train data/train/turns.jsonl \
    --out runs/muril_v1 \
    --eval data/eval/turns.jsonl          # held-out report only, never used for selection

# XLM-R alternative
python training/state_classifier/train.py --model xlm-roberta-base --out runs/xlmr_v1 --train data/train/turns.jsonl

# xlm-roberta-large on A100 80GB: lower lr, smaller batch
python training/state_classifier/train.py --model xlm-roberta-large --lr 1e-5 --batch 32 --epochs 5 \
    --out runs/xlmrl_v1 --train data/train/turns.jsonl
```

Useful flags: `--epochs 6 --lr 3e-5 --head-lr 1e-3 --max-len 128 --w-intent 1 --w-state 0.5 --w-explicit 0.5 --timing-dropout 0.3`. On a base model, 10k rows take a few minutes per epoch on an A100.

Outputs in `--out`:
- `encoder/` (Hugging Face weights + tokenizer)
- `heads.pt`
- `acie_meta.json` (label maps, base model, best dev metrics)
- `history.json`
- `heldout_eval.json` (only when `--eval` is given)

## Use in the pipeline

```python
import sys; sys.path.insert(0, "training/state_classifier")
from predict import TrainedTurnModel
from acie.decision.domains.bank_demo import BankDemoDomain
from acie.decision.engine import DecisionEngine
from acie.state.estimator import HumanStateEstimator
from acie.pipeline import ACIEPipeline

m = TrainedTurnModel("runs/muril_v1")
domain = BankDemoDomain(); domain.classifier = m
pipe = ACIEPipeline(DecisionEngine(domain), estimator=HumanStateEstimator(scorer=m))
```

`score()` maps class probabilities to one expected level: `0.5*p(mild) + 1.0*p(clear)`. Engagement uses the same scale, where a normal turn is about 0.5. Two things to watch:
- The policy compares `1 - engagement` against `ENTER = 0.5`, so a perfectly "normal" 0.5 counts as disengaged. `docs/EVAL.md` lists this as a known policy issue that also affects the rule scorer.
- The estimator's EMA multiplies a first-turn score by 0.7. Re-tune `ENTER` (or the scorer's calibration) against `scripts/eval_signals.py` before switching scorers.

Amount parsing (`acie.text.parse_amount`) is deliberately **not** a model head. Amounts must be exact, and a deterministic spoken-number parser is the right tool for that. See `docs/EVAL.md` for its current accuracy.
