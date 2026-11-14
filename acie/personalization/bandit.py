"""Learned adaptation policy: a contextual bandit per expression knob, rule policy as the prior.

Knobs (each learned independently, 3 discrete actions): verbosity, complexity, warmth, formality,
proactivity, info_density in {low, med, high}; explanation_style in {direct, example, step_by_step}.

Context x (d = 53, binary): the 14 HumanState dims each bucketed low/mid/high (<1/3, <2/3, else) as
one-hot (42), language one-hot (3), turn-index bucket {0, 1-2, 3-5, 6+} (4), mean profile preference
confidence bucket {<0.3, <0.6, >=0.6} (3), and a bias term (1).

Model: disjoint LinUCB (Li et al. 2010), one ridge regressor per (knob, action) arm a:
    A_a = lambda*I + sum x x^T,   b_a = sum r x,   theta_a = A_a^-1 b_a,   mean_a(x) = theta_a . x
    A_a^-1 is kept directly and updated by Sherman-Morrison:
        A^-1 <- A^-1 - (A^-1 x)(A^-1 x)^T / (1 + x^T A^-1 x)
    Rewards r are clipped to [-1, 1] (the feedback lane computes them). mean_a(x) is the posterior mean of
    the reward of playing a in context x (lambda = 1, a Gaussian prior N(0, I/lambda)), and
        ucb_a(x) = mean_a(x) + alpha * sqrt(x^T A_a^-1 x)
    is its optimistic bound (alpha = exploration parameter, default 1.0; ACIE_BANDIT_ALPHA).

Safe by construction (the rule policy is the prior):
  * exploit: for a knob that is not protected, the bandit replaces the rule action r only if the best arm a*
    has mean_a*(x) - mean_r(x) > margin (default 0.1) AND both arms have >= min_obs observations (default 30).
  * explore: each unprotected knob, independently with probability epsilon (default 0.05; ACIE_BANDIT_EPSILON), is
    set to the non-rule arm with the highest UCB, so arms other than the rule's collect data at all. UCB
    bonuses shrink as an arm is observed, so exploration fades arm by arm.
  * protected knobs are never touched: anything the customer asked for explicitly, anything set by a safety or
    state rule (frustration -> warmth high, verbosity low; confusion -> complexity low; urgency; anxiety ->
    step-by-step; overload -> low density) and everything once the call is flagged for safety.
observe() updates only knobs that were not protected on the decision it rewards.

Persistence: JSON at ACIE_BANDIT_STATE (default data/bandit.json, git-ignored), written atomically
(temp file + fsync + os.replace) every `save_every` observations and on save(). Enable with ACIE_POLICY=bandit.
"""

from __future__ import annotations

import json
import os
import random
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


class BanditPolicy:
    def __init__(self, prior: AdaptationPolicy | None = None, *, state_path: str | Path | None = None,
                 min_obs: int | None = None, margin: float | None = None, epsilon: float | None = None,
                 alpha: float | None = None, lam: float = 1.0, save_every: int = 20,
                 rng: random.Random | None = None, persist: bool = True) -> None:
        env = os.environ.get
        self.prior = prior or AdaptationPolicy()
        self.min_obs = int(env("ACIE_BANDIT_MIN_OBS", 30)) if min_obs is None else min_obs
        self.margin = float(env("ACIE_BANDIT_MARGIN", 0.1)) if margin is None else margin
        self.epsilon = float(env("ACIE_BANDIT_EPSILON", 0.05)) if epsilon is None else epsilon
        self.alpha = float(env("ACIE_BANDIT_ALPHA", 1.0)) if alpha is None else alpha
        self.lam = lam
        self.save_every = save_every
        self.rng = rng or random.Random()
        self.persist = persist
        self.path = Path(state_path or env("ACIE_BANDIT_STATE", "data/bandit.json"))
        self.arms: dict[str, dict[str, _Arm]] = {k: {a: _Arm(lam) for a in acts} for k, acts in KNOB_ACTIONS.items()}
        self.total_obs = 0
        self._lock = threading.RLock()
        self._pending: dict[str, tuple[np.ndarray, set[str]]] = {}  # customer -> (context, protected) of the last decision
        self.last_trace: dict[str, dict[str, str]] = {}  # knob -> {rule, final, source}
        if persist:
            self.load()

    # ------------------------------------------------------------------ decision
    def decide(self, text: str, state: HumanState, profile: CustomerProfile, session: SessionState,
               explicit: dict[str, str], intent: str | None = None) -> ExpressionParams:
        a, protected = self.prior.decide_traced(text, state, profile, session, explicit, intent)
        x = context_vector(state, a.language, session.turn_index, profile)
        with self._lock:
            self._pending[profile.customer_id] = (x, set(protected))
            if len(self._pending) > 10_000:  # abandoned calls
                self._pending.pop(next(iter(self._pending)))
            trace: dict[str, dict[str, str]] = {}
            free = [k for k in KNOB_ACTIONS if k not in protected]
            for knob in free:
                rule_val = getattr(a, knob)
                arms = self.arms[knob]
                if rule_val not in arms:
                    continue
                final, source = rule_val, "rule"
                if self.rng.random() < self.epsilon:
                    alts = [v for v in arms if v != rule_val]
                    final = max(alts, key=lambda v: (arms[v].mean(x) + self.alpha * arms[v].bonus(x), -arms[v].n))
                    source = "explore"
                else:
                    best = max(arms, key=lambda v: arms[v].mean(x))
                    if (best != rule_val and arms[best].n >= self.min_obs and arms[rule_val].n >= self.min_obs
                            and arms[best].mean(x) - arms[rule_val].mean(x) > self.margin):
                        final, source = best, "bandit"
                if final != rule_val:
                    setattr(a, knob, final)
                    trace[knob] = {"rule": rule_val, "final": final, "source": source}
            self.last_trace = trace
        return a

    # ------------------------------------------------------------------ learning
    def observe(self, state: HumanState, expr: ExpressionParams, reward: float, profile: CustomerProfile) -> None:
        r = max(-1.0, min(1.0, float(reward)))
        with self._lock:
            pend = self._pending.pop(profile.customer_id, None)
            if pend:
                x, protected = pend
            else:
                x, protected = context_vector(state, expr.language, 0, profile), set()
            for knob, arms in self.arms.items():
                if knob in protected:
                    continue
                arm = arms.get(getattr(expr, knob, None))
                if arm is not None:
                    arm.update(x, r)
            self.total_obs += 1
            if self.persist and self.save_every and self.total_obs % self.save_every == 0:
                self.save()

    def counts(self, knob: str) -> dict[str, int]:
        return {a: arm.n for a, arm in self.arms[knob].items()}

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
            payload = json.dumps(self.to_dict())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=self.path.name + ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
