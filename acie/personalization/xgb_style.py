"""Deterministic XGBoost style policy (ACIE_POLICY=xgb).

One XGBoost regressor per (knob, action) arm predicts the reward of playing that action in a context.
Trained OFFLINE by acie/personalization/train_style.py from logged (context, action, reward) rows with fixed
parameters (tree_method="exact", n_jobs=1, random_state=0, no subsampling); at runtime the policy only
predicts. Selection = argmax predicted reward, but the rule prior wins unless the margin and minimum-
observation rules hold and the knob is not protected (explicit request, safety or state rule), exactly as in
BanditPolicy. With no model file the policy is the rule policy.

Live `observe` never changes the model: it appends the (context, actions, reward) row to the log
(ACIE_STYLE_LOG, JSONL, git-ignored; default data/style_log.jsonl) for the next offline training run.
Model file: ACIE_STYLE_MODEL (default data/style_xgb.json): {"dim", "min_obs", "arms": {knob: {action:
{"n": rows, "model": <xgboost JSON or null>}}}}, keys sorted, so the same log gives a byte-identical model.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from ..schemas import CustomerProfile, ExpressionParams, HumanState
from .bandit import DIM, KNOB_ACTIONS, LearnedStylePolicy, atomic_write
from .policy import AdaptationPolicy


class XGBStylePolicy(LearnedStylePolicy):
    def __init__(self, prior: AdaptationPolicy | None = None, *, model_path: str | Path | None = None,
                 log_path: str | Path | None = None, min_obs: int | None = None, margin: float | None = None,
                 log: bool = True) -> None:
        env = os.environ.get
        super().__init__(prior, int(env("ACIE_STYLE_MIN_OBS", 30)) if min_obs is None else min_obs,
                         float(env("ACIE_STYLE_MARGIN", 0.1)) if margin is None else margin)
        self.model_path = Path(model_path or env("ACIE_STYLE_MODEL", "data/style_xgb.json"))
        self.log_path = Path(log_path or env("ACIE_STYLE_LOG", "data/style_log.jsonl"))
        self.log = log
        self.counts: dict[str, dict[str, int]] = {}
        self.boosters: dict[str, dict[str, object]] = {}
        self.load()

    def load(self) -> bool:
        try:
            import xgboost as xgb
            raw = json.loads(self.model_path.read_text())
            if raw.get("dim") != DIM:
                return False
            counts: dict[str, dict[str, int]] = {}
            boosters: dict[str, dict[str, object]] = {}
            for knob, acts in KNOB_ACTIONS.items():
                counts[knob], boosters[knob] = {}, {}
                for act in acts:
                    d = raw["arms"][knob][act]
                    counts[knob][act] = int(d["n"])
                    if d.get("model"):
                        b = xgb.Booster()
                        b.load_model(bytearray(d["model"].encode()))
                        b.set_param({"nthread": 1})
                        boosters[knob][act] = b
        except (ImportError, OSError, ValueError, KeyError, TypeError, AttributeError):
            self.counts, self.boosters = {}, {}
            return False
        self.counts, self.boosters = counts, boosters
        return True

    def n_obs(self, knob: str, action: str) -> int:
        return self.counts.get(knob, {}).get(action, 0) if action in self.boosters.get(knob, {}) else 0

    def predict(self, knob: str, action: str, x: np.ndarray) -> float:
        return float(self.boosters[knob][action].inplace_predict(x.reshape(1, -1).astype(np.float32))[0])

    def _learn(self, row: dict) -> None:
        if not self.log:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def save_model(path: Path, model: dict) -> None:
    atomic_write(Path(path), json.dumps(model, sort_keys=True))
