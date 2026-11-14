"""Fine-tune a multilingual encoder as ACIE's multi-task turn classifier.

Heads: intent (softmax, 11) + frustration/confusion/urgency/engagement (3-class each)
+ explicit preference requests (multi-label, 6). Plain torch loop, bf16 autocast.

  python training/state_classifier/train.py \
      --train data/train/turns.jsonl --out runs/muril_v1            # default google/muril-base-cased
  python training/state_classifier/train.py --model xlm-roberta-base --out runs/xlmr_v1

NEVER pass data/eval/*: that is the held-out test set (the script refuses it, and
drops any training row whose text duplicates an eval text). Evaluate a trained
checkpoint against data/eval with scripts/eval_signals.py-style metrics via
`--eval data/eval/turns.jsonl` (read-only, reported after training, never used for
model selection - model selection uses the dev split carved out of --train).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from transformers import AutoTokenizer, get_linear_schedule_with_warmup  # noqa: E402

from data import (DEFAULT_TRAIN, EXPLICIT_LABELS, INTENTS, STATE_DIMS, assert_not_eval,  # noqa: E402
                  drop_eval_overlap, load, record_inputs, record_targets, split)
from model import TurnClassifier  # noqa: E402


def collate(tokenizer, max_len: int, timing_dropout: float = 0.0):
    def fn(batch: list[dict]) -> dict:
        ctx, cust = zip(*(record_inputs(r, drop_timing=random.random() < timing_dropout) for r in batch))
        # truncate the context first: the customer's own words matter most
        enc = tokenizer(list(ctx), list(cust), truncation="only_first", max_length=max_len,
                        padding=True, return_tensors="pt")
        t = [record_targets(r) for r in batch]
        enc["y_intent"] = torch.tensor([x["intent"] for x in t])
        enc["y_states"] = torch.tensor([x["states"] for x in t])
        enc["y_explicit"] = torch.tensor([x["explicit"] for x in t])
        return enc
    return fn


def class_weights(recs: list[dict], n: int, key) -> torch.Tensor:
    """Inverse-sqrt frequency weights: positives for states are rare, don't let 'absent' dominate."""
    counts = torch.ones(n)
    for r in recs:
        counts[key(r)] += 1
    w = (counts.sum() / counts).sqrt()
    return w / w.mean()


def losses(out, b, weights, lw) -> torch.Tensor:
    loss = lw["intent"] * nn.functional.cross_entropy(out["intent"].float(), b["y_intent"],
                                                      weight=weights["intent"], label_smoothing=0.05)
    for i, d in enumerate(STATE_DIMS):
        loss = loss + lw["state"] * nn.functional.cross_entropy(out["states"][d].float(), b["y_states"][:, i],
                                                                weight=weights[d])
    loss = loss + lw["explicit"] * nn.functional.binary_cross_entropy_with_logits(out["explicit"].float(),
                                                                                   b["y_explicit"])
    return loss


def _f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    gi, pi, gs, ps, ge, pe = [], [], [], [], [], []
    for b in loader:
        b = {k: v.to(device) for k, v in b.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(b["input_ids"], b["attention_mask"], b.get("token_type_ids"))
        gi += b["y_intent"].tolist()
        pi += out["intent"].argmax(-1).tolist()
        gs.append(b["y_states"].cpu())
        ps.append(torch.stack([out["states"][d].argmax(-1) for d in STATE_DIMS], 1).cpu())
        ge.append(b["y_explicit"].cpu())
        pe.append((out["explicit"].float().sigmoid() > 0.5).float().cpu())
    model.train()
    if not gi:
        return {}
    gs, ps, ge, pe = torch.cat(gs), torch.cat(ps), torch.cat(ge), torch.cat(pe)
    res = {"intent_acc": sum(a == b for a, b in zip(gi, pi)) / len(gi)}
    f1s = []
    for lab in range(len(INTENTS)):
        tp = sum(a == lab and b == lab for a, b in zip(gi, pi))
        fp = sum(a != lab and b == lab for a, b in zip(gi, pi))
        fn = sum(a == lab and b != lab for a, b in zip(gi, pi))
        if tp + fn:
            f1s.append(_f1(tp, fp, fn))
    res["intent_macro_f1"] = sum(f1s) / len(f1s)
    for i, d in enumerate(STATE_DIMS):
        # binary "present" F1 the policy cares about: label>=1 (engagement: label==0 i.e. disengaged)
        g = gs[:, i] == 0 if d == "engagement" else gs[:, i] >= 1
        p = ps[:, i] == 0 if d == "engagement" else ps[:, i] >= 1
        res[f"{d}_f1"] = _f1(int((g & p).sum()), int((~g & p).sum()), int((g & ~p).sum()))
    tp, fp, fn = int((ge * pe).sum()), int(((1 - ge) * pe).sum()), int((ge * (1 - pe)).sum())
    res["explicit_f1"] = _f1(tp, fp, fn)
    res["score"] = (res["intent_macro_f1"] + sum(res[f"{d}_f1"] for d in STATE_DIMS) / 4 + res["explicit_f1"]) / 3
    return {k: round(v, 4) for k, v in res.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default=str(DEFAULT_TRAIN))
    ap.add_argument("--dev", help="separate dev jsonl (default: 10%% split of --train)")
    ap.add_argument("--eval", help="held-out test jsonl, reported once at the end, never used for selection")
    ap.add_argument("--model", default="google/muril-base-cased", help="or xlm-roberta-base / xlm-roberta-large")
    ap.add_argument("--out", default="runs/state_classifier")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=64, help="64 fits easily on A100 40GB for base models at len 128")
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.06)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--w-intent", type=float, default=1.0)
    ap.add_argument("--w-state", type=float, default=0.5)
    ap.add_argument("--w-explicit", type=float, default=0.5)
    ap.add_argument("--timing-dropout", type=float, default=0.3,
                    help="fraction of training rows with timing words removed (classify() has no timing)")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    assert_not_eval(args.train)
    if args.dev:
        assert_not_eval(args.dev)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    recs = drop_eval_overlap(load(args.train))
    if args.dev:
        train, dev = recs, load(args.dev)
    else:
        train, dev = split(recs, seed=args.seed)
    print(f"train {len(train)}  dev {len(dev)}  device {device}  model {args.model}")
    if len(train) < 500:
        print("WARNING: <500 training rows; expect an under-fit, noisy model", file=sys.stderr)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = TurnClassifier(args.model).to(device)
    fn = collate(tok, args.max_len)
    tl = DataLoader(train, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                    collate_fn=collate(tok, args.max_len, args.timing_dropout))
    dl = DataLoader(dev, batch_size=args.batch * 2, collate_fn=fn, num_workers=args.workers)

    weights = {"intent": class_weights(train, len(INTENTS), lambda r: INTENTS.index(r["intent"])).to(device)}
    for i, d in enumerate(STATE_DIMS):
        weights[d] = class_weights(train, 3, lambda r, d=d: r["labels"][d]).to(device)
    lw = {"intent": args.w_intent, "state": args.w_state, "explicit": args.w_explicit}

    enc_params = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    opt = torch.optim.AdamW([{"params": enc_params, "lr": args.lr}, {"params": head_params, "lr": args.head_lr}],
                            weight_decay=args.weight_decay, fused=device.type == "cuda")
    steps = args.epochs * math.ceil(len(train) / args.batch)
    sched = get_linear_schedule_with_warmup(opt, int(args.warmup * steps), steps)

    out = Path(args.out)
    best, history = -1.0, []
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        tot, n = 0.0, 0
        for b in tl:
            b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                o = model(b["input_ids"], b["attention_mask"], b.get("token_type_ids"))
            loss = losses(o, b, weights, lw)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            tot, n = tot + loss.item(), n + 1
        m = evaluate(model, dl, device) if dev else {}
        history.append({"epoch": ep, "train_loss": round(tot / max(n, 1), 4), **m})
        print(json.dumps(history[-1]), f"({time.time() - t0:.0f}s)")
        score = m.get("score", float(ep))  # no dev set -> keep the last epoch
        if score > best:
            best = score
            model.save(out, tok, {"base_model": args.model, "max_len": args.max_len, "epoch": ep, "dev": m,
                                  "train_file": str(args.train)})
            print(f"  saved -> {out}")

    (out / "history.json").write_text(json.dumps(history, indent=2))
    if args.eval:
        model = TurnClassifier.load(out).to(device)
        test = load(args.eval)
        res = evaluate(model, DataLoader(test, batch_size=args.batch * 2, collate_fn=fn), device)
        by_lang = {l: evaluate(model, DataLoader([r for r in test if r.get("lang") == l], batch_size=args.batch * 2,
                                                 collate_fn=fn), device) for l in ("en", "hi", "hinglish")}
        report = {"overall": res, "by_lang": by_lang, "n": len(test)}
        (out / "heldout_eval.json").write_text(json.dumps(report, indent=2))
        print("held-out:", json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
