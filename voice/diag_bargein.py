"""How well does a SHORT clip (the start of an interruption) match the caller's voice print?
For every caller utterance (>= 1.2 s, matching the caller at >= 0.6 on its full length) and every
noise-like utterance (< 0.3), score its first 0.5 / 0.8 / 1.0 s against the caller centroid.

  python -m voice.diag_bargein CALLS_DIR
"""
import sys
import wave
from pathlib import Path

import numpy as np

from .speaker import SpeakerLock, embed


def load(p):
    with wave.open(str(p)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768


def main(src: Path):
    rows = {"caller": {0.5: [], 0.8: [], 1.0: []}, "other": {0.5: [], 0.8: [], 1.0: []}}
    for d in sorted(p for p in src.iterdir() if p.is_dir()):
        xs = [load(w) for w in sorted(d.glob("customer_*.wav"))]
        full = [(x, embed(x)) for x in xs if len(x) >= 8000]
        lock = SpeakerLock()
        lock.embs = [e for x, e in full if e is not None and len(x) >= 16000]
        c = lock._caller()
        if c is None:
            continue
        for x, e in full:
            if e is None:
                continue
            s = float(e @ c)
            kind = "caller" if s >= 0.6 and len(x) >= 19200 else "other" if s < 0.3 else None
            if kind is None:
                continue
            for sec in (0.5, 0.8, 1.0):
                # skip the 300 ms of pre-roll the VAD keeps before speech
                clip = x[4800:4800 + int(16000 * sec)]
                ec = embed(clip) if len(clip) >= 8000 else None
                if ec is not None:
                    rows[kind][sec].append(float(ec @ c))
    for kind, by in rows.items():
        for sec, v in by.items():
            if v:
                v = np.array(v)
                print(f"{kind:6s} first {sec}s: n={len(v)} p10={np.percentile(v,10):.2f} p25={np.percentile(v,25):.2f} "
                      f"median={np.median(v):.2f} p90={np.percentile(v,90):.2f}  share>=0.45 {np.mean(v>=0.45):.2f} "
                      f">=0.35 {np.mean(v>=0.35):.2f} >=0.30 {np.mean(v>=0.30):.2f}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
