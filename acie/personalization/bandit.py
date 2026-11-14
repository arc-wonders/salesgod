"""Learned, DETERMINISTIC style policy: a contextual model per expression knob, rule policy as the prior.

Same input -> same output on every run: no random sampling, no epsilon exploration, no Thompson sampling,
no hash(); fixed iteration order (knob order below, action order per knob) and fixed tie-breaks (first action
in order wins). Models are trained from LOGGED (context, action, reward) rows; at decision time they only
predict. Rewards come from the feedback lane and optimise the customer's comfort and understanding, never sales.

Knobs (each learned independently, 3 discrete actions): verbosity, complexity, warmth, formality,
proactivity, info_density in {low, med, high}; explanation_style in {direct, example, step_by_step}.

Context x (d = 53, binary): the 14 HumanState dims each bucketed low/mid/high (<1/3, <2/3, else) as
one-hot (42), language one-hot (3), turn-index bucket {0, 1-2, 3-5, 6+} (4), mean profile preference
confidence bucket {<0.3, <0.6, >=0.6} (3), and a bias term (1).

BanditPolicy = LinUCB-style ridge regression, one regressor per (knob, action) arm a:
    A_a = lambda*I + sum x x^T,   b_a = sum r x,   theta_a = A_a^-1 b_a,   mean_a(x) = theta_a . x
    A_a^-1 is kept directly and updated by Sherman-Morrison (applied in observation order):
        A^-1 <- A^-1 - (A^-1 x)(A^-1 x)^T / (1 + x^T A^-1 x)
    Rewards are clipped to [-1, 1]. mean_a(x) is the posterior mean reward of playing a in context x
    (lambda = 1, prior N(0, I/lambda)). The optimistic bound ucb_a(x) = mean_a(x) + alpha*sqrt(x^T A_a^-1 x)
    exists for OFFLINE analysis only (`ucb`, alpha = ACIE_BANDIT_ALPHA, default 1.0); decisions use the pure
    posterior mean (zero exploration bonus). Because the live policy never explores, its training data comes
    from the logging policy (the rule prior) and from offline replays through `fit_rows`.

Safe by construction (the rule policy is the prior):
  * a knob that is not protected moves off the rule action only if the best arm a* (argmax predicted reward,
    ties -> earlier action) has pred_a*(x) - pred_rule(x) > margin (default 0.1) AND both arms have
    >= min_obs observations (default 30);
  * protected knobs are never touched: anything the customer asked for explicitly, anything set by a safety or
    state rule (frustration -> warmth high, verbosity low; confusion -> complexity low; urgency; anxiety ->
    step-by-step; overload -> low density) and everything once the call is flagged for safety.
observe() updates only knobs that were not protected on the decision it rewards.

Persistence: JSON at ACIE_BANDIT_STATE (default data/bandit.json, git-ignored), written atomically
(temp file + fsync + os.replace), keys sorted so the same observation log gives a byte-identical file.
Enable with ACIE_POLICY=bandit. The XGBoost alternative (ACIE_POLICY=xgb) lives in xgb_style.py.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path

import numpy as np

from ..schemas import CustomerProfile, ExpressionParams, HumanState, SessionState
from .policy import AdaptationPolicy

LEVELS = ("low", "med", "high")
KNOB_ACTIONS: dict[str, tuple[str, ...]] = {
    "verbosity": LEVELS, "complexity": LEVELS, "warmth": LEVELS, "formality": LEVELS,
    "proactivity": LEVELS, "info_density": LEVELS,
    "explanation_style": ("direct", "example", "step_by_step"),
}
STATE_DIMS = ("frustration", "confusion", "urgency", "engagement", "satisfaction", "anxiety", "excitement",
              "calmness", "trust", "certainty", "cognitive_load", "attention", "participation", "receptiveness")
LANGS = ("en", "hi", "hinglish")
DIM = len(STATE_DIMS) * 3 + len(LANGS) + 4 + 3 + 1
_DEFAULTS = {"engagement": 0.5, "satisfaction": 0.5, "calmness": 0.5, "trust": 0.5, "certainty": 0.5,
             "attention": 0.5, "participation": 0.5, "receptiveness": 0.5}


def _bucket3(v: float, lo: float = 1 / 3, hi: float = 2 / 3) -> int:
    return 0 if v < lo else (1 if v < hi else 2)


def context_vector(state: HumanState, language: str, turn_index: int, profile: CustomerProfile) -> np.ndarray:
    x = np.zeros(DIM)
    for i, name in enumerate(STATE_DIMS):
        x[i * 3 + _bucket3(float(getattr(state, name, _DEFAULTS.get(name, 0.0))))] = 1.0
    o = len(STATE_DIMS) * 3
    x[o + (LANGS.index(language) if language in LANGS else 0)] = 1.0
    o += len(LANGS)
    x[o + (0 if turn_index <= 0 else 1 if turn_index <= 2 else 2 if turn_index <= 5 else 3)] = 1.0
    o += 4
    confs = [p.confidence for p in profile.preferences.values()]
    x[o + _bucket3(sum(confs) / len(confs) if confs else 0.0, 0.3, 0.6)] = 1.0
    x[-1] = 1.0
    return x


class _Arm:
    __slots__ = ("ainv", "b", "n")

    def __init__(self, lam: float) -> None:
        self.ainv = np.eye(DIM) / lam
        self.b = np.zeros(DIM)
        self.n = 0

    def mean(self, x: np.ndarray) -> float:
        return float((self.ainv @ self.b) @ x)

    def bonus(self, x: np.ndarray) -> float:
        return float(np.sqrt(max(0.0, x @ self.ainv @ x)))

    def update(self, x: np.ndarray, r: float) -> None:
        v = self.ainv @ x
        self.ainv -= np.outer(v, v) / (1.0 + float(x @ v))
        self.b += r * x
        self.n += 1


def row_context(row: dict) -> np.ndarray:
    x = np.zeros(DIM)
    x[row["x"]] = 1.0
    return x


class LearnedStylePolicy:
    """Shared deterministic decision logic. Subclasses supply `predict(knob, action, x)`, `n_obs(knob, action)`
    and `_learn(row)`."""

    def __init__(self, prior: AdaptationPolicy | None, min_obs: int, margin: float) -> None:
        self.prior = prior or AdaptationPolicy()
        self.min_obs = min_obs
        self.margin = margin
        self.total_obs = 0
        self._lock = threading.RLock()
        self._pending: dict[str, tuple[np.ndarray, set[str]]] = {}  # customer -> (context, protected) of the last decision
        self.last_trace: dict[str, dict[str, str]] = {}  # knob -> {rule, final, source}

    def predict(self, knob: str, action: str, x: np.ndarray) -> float:
        raise NotImplementedError

    def n_obs(self, knob: str, action: str) -> int:
        raise NotImplementedError

    def _learn(self, row: dict) -> None:
        raise NotImplementedError

    # ------------------------------------------------------------------ decision (pure function of the learned state)
    def decide(self, text: str, state: HumanState, profile: CustomerProfile, session: SessionState,
               explicit: dict[str, str], intent: str | None = None) -> ExpressionParams:
        a, protected = self.prior.decide_traced(text, state, profile, session, explicit, intent)
        x = context_vector(state, a.language, session.turn_index, profile)
        with self._lock:
            self._pending[profile.customer_id] = (x, set(protected))
            if len(self._pending) > 10_000:  # abandoned calls
                self._pending.pop(next(iter(self._pending)))
            trace: dict[str, dict[str, str]] = {}
            for knob, actions in KNOB_ACTIONS.items():
                if knob in protected:
                    continue
                rule_val = getattr(a, knob)
                if rule_val not in actions or self.n_obs(knob, rule_val) < self.min_obs:
                    continue
                pred = {v: self.predict(knob, v, x) for v in actions if self.n_obs(knob, v) >= self.min_obs}
                best = rule_val
                for v in actions:  # fixed order: on a tie the earlier action wins
                    if v in pred and pred[v] > pred[best]:
                        best = v
                if best != rule_val and pred[best] - pred[rule_val] > self.margin:
                    setattr(a, knob, best)
                    trace[knob] = {"rule": rule_val, "final": best, "source": "learned"}
            self.last_trace = trace
        return a

    # ------------------------------------------------------------------ learning
    def make_row(self, state: HumanState, expr: ExpressionParams, reward: float, profile: CustomerProfile) -> dict:
        r = max(-1.0, min(1.0, float(reward)))
        with self._lock:
            pend = self._pending.pop(profile.customer_id, None)
        x, protected = pend if pend else (context_vector(state, expr.language, 0, profile), set())
        return {"x": np.flatnonzero(x).tolist(), "reward": r,
                "actions": {k: getattr(expr, k) for k in KNOB_ACTIONS},
                "protected": sorted(protected)}

    def observe(self, state: HumanState, expr: ExpressionParams, reward: float, profile: CustomerProfile) -> None:
        self.fit_rows([self.make_row(state, expr, reward, profile)])

    def fit_rows(self, rows) -> None:
        """Apply logged rows in the given order (offline replay and live observe share this path)."""
        with self._lock:
            for row in rows:
                self._learn(row)
                self.total_obs += 1
            self._after_fit()

    def _after_fit(self) -> None:
        pass


class BanditPolicy(LearnedStylePolicy):
    def __init__(self, prior: AdaptationPolicy | None = None, *, state_path: str | Path | None = None,
                 min_obs: int | None = None, margin: float | None = None, alpha: float | None = None,
                 lam: float = 1.0, save_every: int = 20, persist: bool = True) -> None:
        env = os.environ.get
        super().__init__(prior, int(env("ACIE_BANDIT_MIN_OBS", 30)) if min_obs is None else min_obs,
                         float(env("ACIE_BANDIT_MARGIN", 0.1)) if margin is None else margin)
        self.alpha = float(env("ACIE_BANDIT_ALPHA", 1.0)) if alpha is None else alpha  # offline analysis only
        self.lam = lam
        self.save_every = save_every
        self.persist = persist
        self.path = Path(state_path or env("ACIE_BANDIT_STATE", "data/bandit.json"))
        self.arms: dict[str, dict[str, _Arm]] = {k: {a: _Arm(lam) for a in acts} for k, acts in KNOB_ACTIONS.items()}
        if persist:
            self.load()

    def predict(self, knob: str, action: str, x: np.ndarray) -> float:
        return self.arms[knob][action].mean(x)

    def ucb(self, knob: str, action: str, x: np.ndarray) -> float:
        """Optimistic bound, for offline analysis and data-collection design only; never used to decide."""
        arm = self.arms[knob][action]
        return arm.mean(x) + self.alpha * arm.bonus(x)

    def n_obs(self, knob: str, action: str) -> int:
        return self.arms[knob][action].n

    def counts(self, knob: str) -> dict[str, int]:
        return {a: arm.n for a, arm in self.arms[knob].items()}

    def _learn(self, row: dict) -> None:
        x, r = row_context(row), float(row["reward"])
        prot = set(row.get("protected", ()))
        for knob, arms in self.arms.items():  # fixed knob order
            if knob in prot:
                continue
            arm = arms.get(row["actions"].get(knob))
            if arm is not None:
                arm.update(x, r)

    def _after_fit(self) -> None:
        if self.persist and self.save_every and self.total_obs % self.save_every == 0:
            self.save()

    # ------------------------------------------------------------------ persistence
    def to_dict(self) -> dict:
        return {"version": 1, "dim": DIM, "lam": self.lam, "total_obs": self.total_obs,
                "arms": {k: {a: {"ainv": arm.ainv.tolist(), "b": arm.b.tolist(), "n": arm.n} for a, arm in arms.items()}
                         for k, arms in self.arms.items()}}

    def load(self) -> bool:
        try:
            raw = json.loads(self.path.read_text())
            if raw.get("dim") != DIM:
                return False
            loaded: dict[str, dict[str, _Arm]] = {}
            for knob, acts in KNOB_ACTIONS.items():
                loaded[knob] = {}
                for act in acts:
                    d = raw["arms"][knob][act]
                    arm = _Arm(self.lam)
                    arm.ainv, arm.b, arm.n = np.array(d["ainv"], dtype=float), np.array(d["b"], dtype=float), int(d["n"])
                    if arm.ainv.shape != (DIM, DIM) or arm.b.shape != (DIM,):
                        return False
                    loaded[knob][act] = arm
        except (OSError, ValueError, KeyError, TypeError):
            return False  # missing or corrupt state: start from the rule policy
        with self._lock:
            self.arms, self.total_obs = loaded, int(raw.get("total_obs", 0))
        return True

    def save(self) -> None:
        with self._lock:
            payload = json.dumps(self.to_dict(), sort_keys=True)
        atomic_write(self.path, payload)


def atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
