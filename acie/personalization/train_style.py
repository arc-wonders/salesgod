"""Offline trainer for XGBStylePolicy.

    python -m acie.personalization.train_style --log data/style_log.jsonl --out data/style_xgb.json [--min-obs 30]

Reads logged rows {"x": [active context indices], "actions": {knob: action}, "protected": [knobs], "reward": r}
(written by XGBStylePolicy.observe or BanditPolicy.make_row), trains one XGBRegressor per (knob, action) on the
rows where that knob was NOT protected and that action was played, with the fixed parameters below, and writes
the models as XGBoost JSON. Rows are used in file order and nothing is random: the same log gives a
byte-identical model file. Arms with fewer than --min-obs rows get no model (the rule prior is kept there).
Rewards come from the feedback lane (customer comfort/understanding; never sales).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .bandit import DIM, KNOB_ACTIONS, row_context
from .xgb_style import save_model

XGB_PARAMS = dict(tree_method="exact", n_estimators=200, max_depth=4, learning_rate=0.1, subsample=1.0,
                  colsample_bytree=1.0, n_jobs=1, random_state=0)
# the same settings in xgboost's native names (xgb.train; no scikit-learn dependency)
NATIVE_PARAMS = dict(objective="reg:squarederror", tree_method=XGB_PARAMS["tree_method"], max_depth=XGB_PARAMS["max_depth"],
                     eta=XGB_PARAMS["learning_rate"], subsample=XGB_PARAMS["subsample"],
                     colsample_bytree=XGB_PARAMS["colsample_bytree"], nthread=XGB_PARAMS["n_jobs"],
                     seed=XGB_PARAMS["random_state"])


def read_rows(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def train(rows: list[dict], min_obs: int = 30) -> dict:
    import xgboost as xgb
    arms: dict[str, dict[str, dict]] = {}
    for knob, acts in KNOB_ACTIONS.items():
        arms[knob] = {}
        for act in acts:
            sel = [r for r in rows if r["actions"].get(knob) == act and knob not in r.get("protected", ())]
            model = None
            if len(sel) >= min_obs:
                X = np.stack([row_context(r) for r in sel]).astype(np.float32)
                y = np.array([max(-1.0, min(1.0, float(r["reward"]))) for r in sel], dtype=np.float32)
                booster = xgb.train(NATIVE_PARAMS, xgb.DMatrix(X, label=y), num_boost_round=XGB_PARAMS["n_estimators"])
                model = booster.save_raw("json").decode()
            arms[knob][act] = {"n": len(sel), "model": model}
    return {"version": 1, "dim": DIM, "min_obs": min_obs, "params": XGB_PARAMS, "arms": arms}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="data/style_log.jsonl")
    ap.add_argument("--out", default="data/style_xgb.json")
    ap.add_argument("--min-obs", type=int, default=30)
    a = ap.parse_args(argv)
    model = train(read_rows(a.log), a.min_obs)
    save_model(Path(a.out), model)
    print(f"trained {sum(1 for k in model['arms'].values() for v in k.values() if v['model'])} arms -> {a.out}")


if __name__ == "__main__":
    main()
