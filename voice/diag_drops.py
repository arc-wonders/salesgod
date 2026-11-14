"""Why was the caller's speech dropped? Re-cut recorded calls into utterances like the live call does,
transcribe each one and print every segment Whisper produced, with the filter that would drop it.

  python -m voice.diag_drops CALLS_DIR call_id [call_id ...]
"""

from __future__ import annotations

import sys
import wave
from pathlib import Path

import numpy as np

from .audio import pcm16_to_ulaw, ulaw_to_float, upsample_8k_to_16k
from .speaker import SpeakerLock, embed
from .stt import WhisperSTT
from .vad import TurnDetector


def channels(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with wave.open(str(path)) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").reshape(-1, w.getnchannels())
    # the customer line as Plivo sent it (mu-law), and the agent channel as played
    cust = ulaw_to_float(pcm16_to_ulaw(x[:, 0].astype("<i2").tobytes()))
    agent = x[:, 1].astype(np.float32) / 32768 if x.shape[1] > 1 else np.zeros(len(x), np.float32)
    return cust, agent


def main(src: Path, ids: list[str]) -> None:
    stt = WhisperSTT()
    for cid in ids:
        cust, agent = channels(src / cid / "call.wav")
        det = TurnDetector()
        # the agent's voice print: its own channel (TTS as played), resampled to 16 kHz
        a16 = upsample_8k_to_16k(agent)
        loud = np.concatenate([a16[k:k + 8000] for k in range(0, len(a16) - 8000, 8000)
                               if np.sqrt(np.mean(a16[k:k + 8000] ** 2)) > 0.03][:20] or [np.zeros(16000, np.float32)])
        agent_print = embed(loud)
        utts = []
        print(f"\n=== {cid}")
        for i in range(0, len(cust), 160):
            for kind, data in det.feed(upsample_8k_to_16k(cust[i:i + 160])):
                if kind != "utterance":
                    continue
                end = (i + 160) / 8000
                start = end - len(data) / 16000
                a = agent[int(start * 8000):int(end * 8000)]
                agent_rms = float(np.sqrt(np.mean(a * a))) if len(a) else 0.0
                rms = float(np.sqrt(np.mean(data * data)))
                e = embed(data)
                utts.append((start, end, rms, agent_rms, e))
                sa = None if e is None or agent_print is None else round(float(e @ agent_print), 2)
                print(f"  -- {start:.1f}s sim_to_agent_voice {sa}")
                for lang in ("hi", "en"):
                    segs = stt._run(data, lang, None)
                    for s in segs:
                        why = []
                        if s.compression_ratio >= 2.4: why.append("repetitive")
                        if s.avg_logprob <= -1.2: why.append("low-conf")
                        if s.no_speech_prob > 0.6 and s.avg_logprob < -0.7: why.append("nospeech+lowconf")
                        if s.no_speech_prob > 0.4 and rms < 0.025: why.append("quiet+nospeech")
                        if s.no_speech_prob > 0.5 and s.avg_logprob < -0.3 and rms < 0.035: why.append("quiet+nospeech2")
                        if rms < 0.008: why.append("too-quiet")
                        print(f"  {start:6.1f}-{end:5.1f}s rms {rms:.3f} agent_rms {agent_rms:.3f} [{lang}] "
                              f"nsp {s.no_speech_prob:.2f} lp {s.avg_logprob:.2f} cr {s.compression_ratio:.1f} "
                              f"{'DROP ' + ','.join(why) if why else 'keep'}: {s.text.strip()!r}")
                    if not segs:
                        print(f"  {start:6.1f}-{end:5.1f}s rms {rms:.3f} agent_rms {agent_rms:.3f} [{lang}] (no segments)")
        caller_sims(utts)


def caller_sims(utts):
    lock = SpeakerLock()
    lock.embs = [e for s, en, r, ar, e in utts if e is not None and ar < 0.01 and r > 0.04 and en - s >= 1.0]
    c = lock._caller()
    if c is None:
        return
    for s, en, r, ar, e in utts:
        print(f"  caller-sim {s:6.1f}s {None if e is None else round(float(e @ c), 2)}  rms {r:.3f} agent {ar:.3f}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2:])
