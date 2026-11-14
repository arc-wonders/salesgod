"""Streaming turn detection with Silero VAD v5 (ONNX, CPU).

`TurnDetector.feed()` takes 16 kHz float audio and emits events:
  ("speech_start", None)          customer started talking (used for barge-in)
  ("utterance", np.ndarray)       a complete customer turn (speech + short padding)
"""

from __future__ import annotations

import os

import numpy as np
import onnxruntime

FRAME = 512  # 32 ms at 16 kHz
CONTEXT = 64
FRAME_MS = 32


class SileroVAD:
    _session = None

    @classmethod
    def _get_session(cls):
        if cls._session is None:
            opts = onnxruntime.SessionOptions()
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = 1
            opts.log_severity_level = 4
            # official Silero VAD v5.1.2 ONNX (MIT), vendored in voice/assets
            path = os.path.join(os.path.dirname(__file__), "assets", "silero_vad.onnx")
            cls._session = onnxruntime.InferenceSession(path,
                                                        providers=["CPUExecutionProvider"], sess_options=opts)
        return cls._session

    def __init__(self):
        self.session = self._get_session()
        # faster-whisper ships different Silero exports: v5 (input, state, sr) or v6 (input, h, c)
        self.inputs = {i.name for i in self.session.get_inputs()}
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.h = np.zeros((1, 1, 128), dtype=np.float32)
        self.c = np.zeros((1, 1, 128), dtype=np.float32)
        self.context = np.zeros(CONTEXT, dtype=np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        x = np.concatenate([self.context, frame]).reshape(1, -1).astype(np.float32)
        self.context = frame[-CONTEXT:]
        if "state" in self.inputs:
            feed = {"input": x, "state": self.state}
            if "sr" in self.inputs:
                feed["sr"] = np.array(16000, dtype=np.int64)
            out, self.state = self.session.run(None, feed)
        else:
            out, self.h, self.c = self.session.run(None, {"input": x, "h": self.h, "c": self.c})
        return float(np.asarray(out).reshape(-1)[0])


class TurnDetector:
    def __init__(self, threshold: float = 0.5, end_silence_ms: int = 700, min_speech_ms: int = 200,
                 max_utterance_s: float = 20.0, pre_roll_ms: int = 300):
        self.vad = SileroVAD()
        self.threshold = threshold
        self.end_frames = end_silence_ms // FRAME_MS
        self.min_frames = min_speech_ms // FRAME_MS
        self.max_frames = int(max_utterance_s * 1000 / FRAME_MS)
        self.pre_roll = pre_roll_ms // FRAME_MS
        self.buf = np.zeros(0, dtype=np.float32)
        self.history: list[np.ndarray] = []  # pre-roll frames
        self.frames: list[np.ndarray] = []
        self.in_speech = False
        self.speech_frames = 0
        self.silence_frames = 0
        self.started_announced = False
        self.last_prob = 0.0
        # continuous-speech run (for barge-in): live call 7 showed short "haan"/noise bursts adding up
        # inside one VAD segment and cutting the agent off every turn
        self.run_frames = 0
        self.run_energy = 0.0
        self.frame_no = 0
        self.run_start = -1  # frame number where the current confident-speech run began
        self.on_frame = None  # optional callback(frame, vad_prob): the call's calibrator learns line noise

    def feed(self, audio: np.ndarray) -> list[tuple[str, np.ndarray | None]]:
        events: list[tuple[str, np.ndarray | None]] = []
        self.buf = np.concatenate([self.buf, audio])
        while len(self.buf) >= FRAME:
            frame, self.buf = self.buf[:FRAME], self.buf[FRAME:]
            p = self.vad(frame)
            self.last_prob = p
            if self.on_frame is not None:
                self.on_frame(frame, p)
            speech = p >= self.threshold
            rms = float(np.sqrt(np.mean(frame * frame)))
            self.frame_no += 1
            if p >= 0.7:
                if self.run_frames == 0:
                    self.run_start = self.frame_no
                self.run_energy = (self.run_energy * self.run_frames + rms) / (self.run_frames + 1)
                self.run_frames += 1
            else:
                self.run_frames, self.run_energy = 0, 0.0
            if not self.in_speech:
                self.history = (self.history + [frame])[-self.pre_roll:]
                if speech:
                    self.in_speech = True
                    self.frames = list(self.history)
                    self.speech_frames, self.silence_frames, self.started_announced = 1, 0, False
                continue
            self.frames.append(frame)
            if speech:
                self.speech_frames += 1
                self.silence_frames = 0
                if not self.started_announced and self.speech_frames >= self.min_frames:
                    self.started_announced = True
                    events.append(("speech_start", None))
            else:
                self.silence_frames += 1
            if self.silence_frames >= self.end_frames or len(self.frames) >= self.max_frames:
                if self.speech_frames >= self.min_frames:
                    events.append(("utterance", np.concatenate(self.frames)))
                self.in_speech = False
                self.frames, self.history = [], []
        return events
