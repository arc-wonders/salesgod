"""Train the optional need classifier (XGBoost, fixed params) -> acie/psyche/need_model.json (+ .meta.json).

Data: data/real/turns_labeled.jsonl (real turns, weight 3) + data/eval/turns.jsonl (synthetic, labelled by the
deterministic table, weight 1). Features are hashed with zlib.crc32 (acie/psyche/needs.features). Prints a
deterministic 5-fold cross-validation on the real turns. The model contains only tree splits on hashed buckets
(no text). Run: python scripts/real_calls/train_need.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from acie.psyche.needs import DIMS, MODEL_PATH, NEEDS, NeedInput, features, table_need  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PARAMS = {"objective": "multi:softprob", "tree_method": "exact", "nthread": 1, "seed": 0, "subsample": 1.0,
          "colsample_bytree": 1.0, "max_depth": 4, "eta": 0.2, "min_child_weight": 1, "lambda": 1.0}
ROUNDS = 40
CLASSES = list(NEEDS)


def real_rows():
    for l in (ROOT / "data/real/turns_labeled.jsonl").read_text().splitlines():
        r = json.loads(l)
        lv = {k: (0.1, 0.5, 0.9)[v] for k, v in r["state14"].items()}
        yield NeedInput(text=r["text"], state=lv, intent=r["intent"], turn_index=0), r["need"], 3.0


def synth_rows():
    for l in (ROOT / "data/eval/turns.jsonl").read_text().splitlines():
        r = json.loads(l)
        lab = r["labels"]
        st = {"frustration": (0.1, 0.5, 0.9)[lab["frustration"]], "confusion": (0.1, 0.5, 0.9)[lab["confusion"]],
              "urgency": (0.1, 0.5, 0.9)[lab["urgency"]], "engagement": (0.2, 0.5, 0.8)[lab["engagement"]]}
        inp = NeedInput(text=r["text"], state=st, intent=r["intent"])
        yield inp, table_need(inp).need, 1.0


def fit(rows):
    X = np.array([features(i) for i, _, _ in rows], dtype="float32")
    y = np.array([CLASSES.index(n) for _, n, _ in rows])
    w = np.array([w for _, _, w in rows])
    d = xgb.DMatrix(X, label=y, weight=w)
    return xgb.train({**PARAMS, "num_class": len(CLASSES)}, d, ROUNDS)


def main() -> None:
    real, synth = list(real_rows()), list(synth_rows())
    # 5-fold CV on the real turns (fold = index mod 5), synthetic always in training
    ok = tot = tb = 0
    for k in range(5):
        tr = [r for i, r in enumerate(real) if i % 5 != k] + synth
        te = [r for i, r in enumerate(real) if i % 5 == k]
        b = fit(tr)
        p = b.predict(xgb.DMatrix(np.array([features(i) for i, _, _ in te], dtype="float32")))
        for (inp, need, _), pr in zip(te, p):
            ok += CLASSES[int(np.argmax(pr))] == need
            tb += table_need(inp).need == need
            tot += 1
    print(f"5-fold CV on {tot} real turns: model {ok / tot:.3f} accuracy vs table {tb / tot:.3f}")
    b = fit(real + synth)
    b.save_model(str(MODEL_PATH))
    MODEL_PATH.with_suffix(".meta.json").write_text(json.dumps({"classes": CLASSES, "params": PARAMS, "rounds": ROUNDS,
                                                                "buckets": 128, "real_rows": len(real), "synthetic_rows": len(synth),
                                                                "cv_accuracy": round(ok / tot, 3), "table_accuracy": round(tb / tot, 3)}, indent=1))
    print("saved", MODEL_PATH)


if __name__ == "__main__":
    main()
