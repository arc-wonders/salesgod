"""Speaker embeddings for locking onto the caller's voice (WeSpeaker ResNet34, ONNX, CPU).

Live call 8: someone in the caller's background kept being transcribed as the caller and
cutting the agent off. The caller literally said "lock onto the primary voice and suppress the
background one". Pitch/loudness alone could not separate them (same pitch range, ~5 dB apart),
so each utterance gets a 256-d voice embedding and is compared with the caller's own voice.

Model: Wespeaker/wespeaker-voxceleb-resnet34-LM (Apache-2.0), input 80-dim Kaldi fbank of 16 kHz
audio with mean normalisation, output embedding. Cosine similarity: same speaker typically
0.5-0.8 over a phone line, different speakers ~0-0.3.
"""

from __future__ import annotations

import numpy as np
import onnxruntime

_SESSION = None


def _session():
    global _SESSION
    if _SESSION is None:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download("Wespeaker/wespeaker-voxceleb-resnet34-LM", "voxceleb_resnet34_LM.onnx")
        opts = onnxruntime.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.log_severity_level = 4
        _SESSION = onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"], sess_options=opts)
    return _SESSION


def _fbank(audio: np.ndarray) -> np.ndarray:
    import kaldi_native_fbank as knf
    opts = knf.FbankOptions()
    opts.frame_opts.samp_freq = 16000
    opts.frame_opts.dither = 0.0
    opts.mel_opts.num_bins = 80
    fb = knf.OnlineFbank(opts)
    fb.accept_waveform(16000, (audio * 32768).astype(np.float32).tolist())
    fb.input_finished()
    feats = np.array([fb.get_frame(i) for i in range(fb.num_frames_ready)], dtype=np.float32)
    return feats - feats.mean(axis=0, keepdims=True)


def embed(audio: np.ndarray) -> np.ndarray | None:
    """L2-normalised voice embedding of a 16 kHz utterance (None if shorter than 0.5 s)."""
    if len(audio) < 8000:
        return None
    feats = _fbank(audio)
    s = _session()
    out = s.run(None, {s.get_inputs()[0].name: feats[None, :, :]})[0][0]
    return out / (np.linalg.norm(out) + 1e-9)


class SpeakerLock:
    """Learns the caller's voice, then flags turns spoken by someone else.

    The caller's voice = the largest group of turns that sound alike (pairwise similarity >=
    SAME). Live call 4b10be40a9: turns 1, 4, 6, 7 matched each other at 0.54-0.81, while the
    garbled "हेलो" / "पादरी की..." / "कि आधे हैं..." turns were 0.14-0.34 to them: a second voice.
    Until two turns agree, nothing is dropped. When unsure, the turn is kept.
    """

    SAME = 0.5

    def __init__(self, drop_below: float = 0.25):
        # 0.4 dropped the real caller on degraded lines (live 2c4ec7a4ff: his own turns at 0.29-0.40 thrown away as
        # "background", six "can you hear me?" prompts). Background voices seen so far scored 0.08-0.24.
        self.drop_below = drop_below
        self.dropped: list[np.ndarray] = []  # recent rejected voices: two that agree = the caller's voice changed
        self.embs: list[np.ndarray] = []

    def _caller(self) -> np.ndarray | None:
        if len(self.embs) < 2:
            return None
        E = np.stack(self.embs)
        sim = E @ E.T
        neighbours = (sim >= self.SAME).sum(axis=1) - 1
        best = int(np.argmax(neighbours))
        if neighbours[best] < 1:
            return None
        group = E[sim[best] >= self.SAME]
        c = group.mean(axis=0)
        return c / (np.linalg.norm(c) + 1e-9)

    def score(self, audio: np.ndarray) -> tuple[np.ndarray | None, bool, float | None]:
        """(embedding, is_caller, similarity to the caller's voice or None if not known yet).
        Does not learn: call `learn` once the turn is accepted (not voicemail / echo / noise)."""
        e = embed(audio)
        if e is None:
            return None, True, None
        caller = self._caller()
        sim = None if caller is None else round(float(e @ caller), 3)
        return e, (sim is None or sim >= self.drop_below), sim

    def voice_changed(self, e: np.ndarray | None, n_samples: int) -> bool:
        """A rejected turn that sounds like an earlier rejected turn: the caller's voice print changed (new line,
        switched language, moved rooms), not a second person. Re-learn from the agreeing turns and keep it."""
        if e is None or n_samples < 16000:
            return False
        if any(float(e @ x) >= self.SAME for x in self.dropped):
            agree = [x for x in self.dropped if float(e @ x) >= self.SAME] + [e]
            self.embs = (self.embs + agree)[-24:]
            self.dropped = []
            return True
        self.dropped = (self.dropped + [e])[-4:]
        return False

    def learn(self, e: np.ndarray | None, n_samples: int) -> None:
        if e is not None and n_samples >= 16000:  # >= 1 s: reliable enough to learn from
            self.embs = (self.embs + [e])[-24:]

    def provisional_sim(self, e: np.ndarray | None) -> float | None:
        """Before the caller is confirmed: best similarity to the clean turns learned so far (None if none)."""
        if e is None or not self.embs:
            return None
        return round(max(float(e @ x) for x in self.embs), 3)

    @property
    def confirmed(self) -> bool:
        return self._caller() is not None

    def check(self, audio: np.ndarray) -> tuple[bool, float | None]:
        """Returns (is_caller, similarity to the caller's voice or None if not known yet)."""
        e, ok, sim = self.score(audio)
        self.learn(e, len(audio))
        return ok, sim
