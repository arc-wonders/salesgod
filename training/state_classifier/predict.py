"""Drop-in inference wrapper for a trained ACIE turn classifier.

`TrainedTurnModel` implements both seams of the pipeline:
  - acie.state.estimator.RawScorer:      score(turn, session) -> (dict[str, float], list[str])
  - acie.decision.engine.IntentClassifier: classify(text, session) -> str
plus explicit(text, session) -> dict for preference requests.

  from training.state_classifier.predict import TrainedTurnModel
  m = TrainedTurnModel("runs/muril_v1")
  domain = BankDemoDomain(); domain.classifier = m
  pipe = ACIEPipeline(DecisionEngine(domain), estimator=HumanStateEstimator(scorer=m))

classify()/explicit() get no timing (the IntentClassifier seam only passes text), so they
share one cached forward pass; score() includes bucketed timing words when the turn has
them, which costs one more pass (~5-10 ms each for a base encoder on GPU). Training uses
timing dropout so both input variants are in-distribution.
"""

from __future__ import annotations

import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from data import EXPLICIT_LABELS, GATED_INTENTS, INTENTS, STATE_DIMS, build_context  # noqa: E402
from model import TurnClassifier  # noqa: E402


def _session_context(session: Any, timing: dict | None) -> str:
    hist = list(getattr(session, "history", []) or [])
    prev_agent = next((u.text for u in reversed(hist) if u.role == "agent"), None)
    prev_customer = next((u.text for u in reversed(hist) if u.role == "customer"), None)
    dialog = getattr(session, "dialog", {}) or {}
    return build_context(prev_agent, prev_customer, dialog.get("pending"), dialog.get("pending_slot"), timing)


class TrainedTurnModel:
    def __init__(self, model_dir: str | Path, device: str | None = None, explicit_threshold: float = 0.5,
                 cache_size: int = 256):
        d = Path(model_dir)
        self.meta = json.loads((d / "acie_meta.json").read_text())
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tok = AutoTokenizer.from_pretrained(d / "encoder")
        self.model = TurnClassifier.load(d).to(self.device).eval()
        if self.device.type == "cuda":
            self.model.to(torch.bfloat16)
        self.max_len = int(self.meta.get("max_len", 128))
        self.explicit_threshold = explicit_threshold
        self._cache: OrderedDict[tuple[str, str], dict] = OrderedDict()
        self._cache_size = cache_size

    @torch.no_grad()
    def predict(self, text: str, context: str) -> dict:
        key = (context, text)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        enc = self.tok([context], [text], truncation="only_first", max_length=self.max_len,
                       return_tensors="pt").to(self.device)
        out = self.model(enc["input_ids"], enc["attention_mask"], enc.get("token_type_ids"))
        res = {"intent_probs": out["intent"].float().softmax(-1)[0].tolist(),
               "state_probs": {dd: out["states"][dd].float().softmax(-1)[0].tolist() for dd in STATE_DIMS},
               "explicit_probs": out["explicit"].float().sigmoid()[0].tolist()}
        self._cache[key] = res
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return res

    # ------------------------------------------------------------------ RawScorer

    def score(self, turn: Any, session: Any) -> tuple[dict[str, float], list[str]]:
        timing = turn.timing.model_dump() if hasattr(turn.timing, "model_dump") else dict(turn.timing or {})
        p = self.predict(turn.text, _session_context(session, timing))
        scores, ev = {}, []
        for dd in STATE_DIMS:
            p0, p1, p2 = p["state_probs"][dd]
            # expected level in [0,1]: absent 0 / mild 0.5 / clear 1.0.
            # engagement uses the same scale (disengaged 0 / normal 0.5 / high 1.0), matching RuleScorer's 0.5 prior.
            scores[dd] = round(0.5 * p1 + 1.0 * p2, 4)
            lvl = max(range(3), key=lambda i: (p0, p1, p2)[i])
            if (dd != "engagement" and lvl > 0) or (dd == "engagement" and lvl == 0):
                ev.append(f"model: {dd}={lvl} (p={max(p0, p1, p2):.2f})")
        return scores, ev

    # ------------------------------------------------------------------ IntentClassifier

    def classify(self, text: str, session: Any) -> str:
        p = self.predict(text, _session_context(session, None))
        allowed_gated = "pending" in (getattr(session, "dialog", {}) or {})
        probs = [0.0 if (i in GATED_INTENTS and not allowed_gated) else pr for i, pr in zip(INTENTS, p["intent_probs"])]
        return INTENTS[max(range(len(INTENTS)), key=probs.__getitem__)]

    def explicit(self, text: str, session: Any) -> dict[str, str]:
        p = self.predict(text, _session_context(session, None))
        out: dict[str, tuple[str, float]] = {}
        for lbl, pr in zip(EXPLICIT_LABELS, p["explicit_probs"]):
            if pr >= self.explicit_threshold:
                k, v = lbl.split("=")
                if k not in out or pr > out[k][1]:
                    out[k] = (v, pr)
        return {k: v for k, (v, _) in out.items()}
