"""Fit the calibrator on real recorded calls (the "training" step; CPU only, ~1 min).

For every call: embed each customer utterance (data/voice/calls/<id>/customer_NN.wav), take the
largest group of mutually-similar turns as THE caller, measure their voice features, and record how
far each turn sits from that caller's own median. The robust spread of those deviations, pooled over
all callers, is "normal turn-to-turn variation" = 1 z unit in voice/calibrator.py. Emotional turns
are a minority, so the median-based spread is not inflated by them.

Also reports how well the voice print separates people on real phone audio: caller-vs-caller
similarity within a call versus caller-vs-other-caller across calls.

  python -m voice.calibrate_fit [calls_dir] [--write]
"""

from __future__ import annotations

import json
import sys
import time
import wave
from pathlib import Path

import numpy as np

from .calibrator import DEFAULT_SPREAD, FEATURES, STATS_PATH, measure
from .speaker import SpeakerLock, embed


def load(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768
        assert w.getframerate() == 16000
    return x


def main(calls_dir: Path, write: bool) -> None:
    devs: dict[str, list[float]] = {k: [] for k in FEATURES}
    pickup: dict[str, list[float]] = {k: [] for k in FEATURES}
    same, cross, centroids = [], [], []
    n_calls = n_turns = 0
    t_feat = []
    for d in sorted(p for p in calls_dir.iterdir() if p.is_dir()):
        wavs = sorted(d.glob("customer_*.wav"))
        items = []
        for w in wavs:
            x = load(w)
            first = w.name == "customer_01.wav"
            if len(x) < (8000 if first else 16000):  # < 1 s: too short to trust (the pickup is often short)
                continue
            t0 = time.perf_counter()
            _, f = measure(x)
            t_feat.append((time.perf_counter() - t0) * 1000 / (len(x) / 16000))
            e = embed(x)
            if f is not None and e is not None:
                items.append(({**f, "first": first}, e))
        if len(items) < 3:
            continue
        lock = SpeakerLock()
        lock.embs = [e for _, e in items]
        caller = lock._caller()
        if caller is None:
            continue
        mine = [(f, e) for f, e in items if float(e @ caller) >= (0.4 if f["first"] else lock.SAME)]
        if len(mine) < 3:
            continue
        n_calls += 1
        n_turns += len(mine)
        E = np.stack([e for _, e in mine])
        sims = E @ E.T
        same += list(sims[np.triu_indices(len(E), 1)])
        centroids.append(caller)
        for k in FEATURES:
            vals = [f[k] for f, _ in mine if f.get(k) is not None]
            if len(vals) >= 3:
                med = float(np.median(vals))
                devs[k] += [v - med for v in vals]
            later = [f[k] for f, _ in mine if not f["first"] and f.get(k) is not None]
            firsts = [f[k] for f, _ in mine if f["first"] and f.get(k) is not None]
            if firsts and len(later) >= 2:
                pickup[k].append(firsts[0] - float(np.median(later)))
    C = np.stack(centroids) if centroids else np.zeros((0, 256))
    if len(C) >= 2:
        cs = C @ C.T
        cross = list(cs[np.triu_indices(len(C), 1)])
    spread = {k: round(1.4826 * float(np.median(np.abs(v))), 3) for k, v in devs.items() if len(v) >= 10}
    out = {
        "spread": spread,
        "default_spread": DEFAULT_SPREAD,
        # pickup "Hello?" minus the same caller's later median (applied to correct a one-turn baseline)
        "pickup_offset": {k: round(float(np.median(v)), 3) for k, v in pickup.items() if len(v) >= 5},
        "pickup_calls": {k: len(v) for k, v in pickup.items()},
        "calls": n_calls, "turns": n_turns,
        "speaker": {
            "same_caller_sim_p10": round(float(np.percentile(same, 10)), 3) if same else None,
            "same_caller_sim_median": round(float(np.median(same)), 3) if same else None,
            "other_caller_sim_p90": round(float(np.percentile(cross, 90)), 3) if cross else None,
            "other_caller_sim_max": round(float(np.max(cross)), 3) if cross else None,
        },
        "feature_ms_per_audio_s": round(float(np.median(t_feat)), 1) if t_feat else None,
        "fitted_at": time.strftime("%Y-%m-%d %H:%M"),
    }
    print(json.dumps(out, indent=1))
    if write:
        STATS_PATH.write_text(json.dumps(out, indent=1) + "\n")
        print("wrote", STATS_PATH)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(Path(args[0]) if args else Path("data/voice/calls"), "--write" in sys.argv)
