import time

import numpy as np

from acie.schemas import SessionState, TurnInput
from acie.state.estimator import HumanStateEstimator
from voice.calibrator import VoiceCalibrator, label, measure, spectral_effort, syllable_rate
from voice.prosody import f0_track

from test_prosody import through_phone_line

SPREAD = {"pitch": 1.5, "pitch_var": 0.6, "loudness": 3.0, "effort": 2.5, "rate": 0.9}


def speech(f0=150.0, seconds=2.0, amp=0.2, vibrato=0.6, tilt=1.0, syl=4.0, seed=0):
    """Harmonic voice with syllable-like loudness pulses. tilt < 1 = more high-frequency energy
    (a strained / raised voice); vibrato = pitch movement in semitones; syl = syllables per second."""
    rate = 16000
    t = np.arange(int(seconds * rate)) / rate
    rng = np.random.default_rng(seed)
    f = f0 * 2 ** ((vibrato * np.sin(2 * np.pi * 2.3 * t) + 0.2 * rng.standard_normal()) / 12)
    phase = 2 * np.pi * np.cumsum(f) / rate
    x = sum(np.sin(k * phase) / k ** tilt for k in range(1, 16))
    env = 0.08 + 0.92 * np.abs(np.sin(np.pi * syl * t)) ** 2  # one pulse per syllable
    x = x * env
    return through_phone_line((amp * x / np.abs(x).max()).astype(np.float32))


def calibrated(n=3):
    c = VoiceCalibrator(dict(SPREAD))
    for i in range(n):  # the first few calm turns after pickup
        c.accept(measure(speech(seed=i, amp=0.2 * (1 + 0.05 * i)))[1])
    return c


def test_noise_floor_learns_only_from_quiet_line_not_voice_or_echo():
    c = VoiceCalibrator(dict(SPREAD))
    rng = np.random.default_rng(1)
    assert c.noise_db is None
    for _ in range(60):
        c.observe((0.003 * rng.standard_normal(512)).astype(np.float32), vad_prob=0.02)
    for _ in range(60):  # speech and our own echo never raise the floor
        c.observe((0.2 * rng.standard_normal(512)).astype(np.float32), vad_prob=0.95)
        c.observe((0.2 * rng.standard_normal(512)).astype(np.float32), vad_prob=0.05, agent_speaking=True)
    assert abs(c.noise_db - 20 * np.log10(0.003)) < 1.5


def test_syllable_rate_and_effort_are_measured():
    for syl in (2.5, 4.0, 6.0):
        f0, rms = f0_track(speech(syl=syl, seconds=3))
        sp = rms > max(0.01, 0.1 * np.percentile(rms, 95))
        assert abs(syllable_rate(rms, sp) - syl) < 0.12 * syl, (syl, syllable_rate(rms, sp))
    # louder = louder, whatever the gate does
    assert measure(speech(amp=0.4, seed=3))[1]["loudness"] - measure(speech(amp=0.2, seed=3))[1]["loudness"] > 5
    a, b = speech(tilt=1.0), speech(tilt=0.5)
    sa = f0_track(a)[1] > 0.01
    sb = f0_track(b)[1] > 0.01
    assert spectral_effort(b, sb) > spectral_effort(a, sa) + 3


def test_first_turn_calibrates_then_same_voice_reads_calm():
    c = VoiceCalibrator(dict(SPREAD), {})
    _, f = measure(speech(seed=0))
    r = c.read(f)
    assert not r.calibrated and label(r)["label"] == "calibrating"
    c.accept(f)  # the pickup alone is not a baseline
    assert not c.read(measure(speech(seed=1))[1]).calibrated
    c.accept(measure(speech(seed=1))[1])
    r = c.read(measure(speech(seed=2))[1])
    assert r.calibrated and abs(r.arousal) < 0.25, r
    assert label(r)["label"] == "calm"


def test_raised_voice_reads_as_aroused_and_words_pick_the_direction():
    c = calibrated()
    raised = speech(f0=150 * 2 ** (3.5 / 12), amp=0.45, vibrato=2.0, tilt=0.6, syl=5.5, seed=7)
    r = c.read(measure(raised)[1])
    assert r.arousal > 0.5 and r.confidence > 0.6, r
    assert r.z["pitch"] > 1.5 and r.z["loudness"] > 1.5 and r.z["effort"] > 1 and r.z["pitch_var"] > 1 \
        and r.z["rate"] > 0.3, r.z
    assert any("pitch up" in e for e in r.evidence)
    assert label(r, frustration=0.6)["label"] == "angry"
    assert label(r, frustration=0.0, positive=True)["label"] == "excited"
    assert label(r, frustration=0.0)["label"] in ("animated", "anxious")  # a question is not "excited"


def test_flat_quiet_slow_voice_reads_low():
    c = calibrated()
    low = speech(f0=150 * 2 ** (-2 / 12), amp=0.09, vibrato=0.1, tilt=1.3, syl=2.5, seed=8)
    r = c.read(measure(low)[1])
    assert r.arousal < -0.4, r
    assert r.z["pitch"] < -1 and r.z["loudness"] < -1.5 and r.z["rate"] < -1, r.z
    assert label(r)["label"] == "low"


def test_pitch_is_relative_to_the_caller_not_absolute():
    """A calm high voice must not read as angry just because it is high."""
    woman = VoiceCalibrator(dict(SPREAD))
    for i in range(3):
        woman.accept(measure(speech(f0=240, seed=i))[1])
    r = woman.read(measure(speech(f0=240, seed=5))[1])
    assert abs(r.arousal) < 0.25 and label(r)["label"] == "calm"
    assert all(abs(v) < 1 for v in r.z.values()), r.z


def test_another_voice_is_not_read_as_emotion():
    c = calibrated()
    r = c.read(measure(speech(f0=150 * 2 ** (10 / 12), seed=9))[1])  # +10 st: someone else
    assert r.arousal == 0.0 and "not the same voice" in r.evidence[0]


def test_baseline_survives_a_short_outburst():
    c = calibrated(4)
    c.accept(measure(speech(f0=150 * 2 ** (4 / 12), amp=0.45, vibrato=2, tilt=0.6, seed=11))[1])
    r = c.read(measure(speech(seed=12))[1])
    assert abs(r.arousal) < 0.3, r


def test_estimator_uses_the_calibrated_reading(monkeypatch):
    monkeypatch.setenv("ACIE_PROSODY", "1")
    c = calibrated()
    raised = c.read(measure(speech(f0=150 * 2 ** (3.5 / 12), amp=0.45, vibrato=2.0, tilt=0.6, seed=7))[1])
    est, sess = HumanStateEstimator(), SessionState(session_id="s", customer_id="c")
    words = est.estimate(TurnInput(session_id="s", customer_id="c", text="ye kya hai yaar"), sess)
    both = est.estimate(TurnInput(session_id="s", customer_id="c", text="ye kya hai yaar", voice=raised), sess)
    assert both.frustration > words.frustration + 0.15


def test_fast_enough_for_live_calls():
    c = calibrated()
    x = speech(seconds=6)
    t0 = time.perf_counter()
    for _ in range(3):
        c.read(measure(x)[1])
    assert (time.perf_counter() - t0) / 3 < 0.25  # per 6 s turn, on CPU, while Whisper runs on the GPU


def test_pickup_hello_does_not_skew_the_baseline():
    """Real call: baseline = a bright pickup "Hello?" made a later angry line read as LOW."""
    c = VoiceCalibrator(dict(SPREAD))
    c.accept(measure(speech(f0=150 * 2 ** (4 / 12), amp=0.4, tilt=0.6, seed=20))[1])  # "Hello?"
    for i in range(3):
        c.accept(measure(speech(seed=21 + i))[1])
    r = c.read(measure(speech(seed=30))[1])
    assert abs(r.arousal) < 0.3 and label(r)["label"] == "calm", r


def test_negative_words_with_a_flat_voice_read_frustrated_not_low():
    c = calibrated()
    flat = c.read(measure(speech(f0=150 * 2 ** (-2 / 12), amp=0.09, vibrato=0.1, tilt=1.3, syl=2.5, seed=8))[1])
    assert label(flat, frustration=0.5)["label"] == "frustrated"
    assert label(flat, frustration=0.3)["label"] != "low"


def test_borderline_voice_print_is_kept_when_pitch_and_loudness_match():
    c = calibrated()
    same = measure(speech(seed=40))[1]
    other = measure(speech(f0=150 * 2 ** (5 / 12), seed=41))[1]  # 5 st away: another person
    quieter = measure(speech(amp=0.06, seed=42))[1]  # ~10 dB down: someone in the background
    assert c.same_voice(same, 0.38)
    assert not c.same_voice(other, 0.38)
    assert not c.same_voice(quieter, 0.38)
    assert not c.same_voice(same, 0.1)  # clearly another voice print: never overridden


def test_fitted_pickup_offset_allows_a_reading_from_turn_two():
    """With the fitted "Hello?" offset, a one-turn baseline is corrected and usable (lower confidence)."""
    off = {"pitch": 2.0, "loudness": 2.0, "effort": 1.5}
    c = VoiceCalibrator(dict(SPREAD), off)
    hello = speech(f0=150 * 2 ** (2 / 12), amp=0.25, tilt=0.85, seed=50)  # brighter, higher pickup
    c.accept(measure(hello)[1])
    assert c.calibrated
    calm = c.read(measure(speech(seed=51))[1])
    assert abs(calm.arousal) < 0.35 and calm.confidence <= 0.5, calm
    raised = c.read(measure(speech(f0=150 * 2 ** (5 / 12), amp=0.5, vibrato=2, tilt=0.5, seed=52))[1])
    assert raised.arousal > 0.5, raised
