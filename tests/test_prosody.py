import numpy as np

from acie.schemas import Prosody, SessionState, TurnInput, TurnTiming
from acie.state.estimator import HumanStateEstimator, prosody_arousal
from voice.audio import pcm16_to_ulaw, ulaw_to_float, upsample_8k_to_16k
from voice.prosody import extract, f0_track


def voice(f0, seconds=1.5, rate=16000, vibrato=0.0, amp=0.3):
    """Harmonic 'voice': fundamental + 10 harmonics, optional pitch movement (semitones)."""
    t = np.arange(int(seconds * rate)) / rate
    f = f0 * 2 ** (vibrato * np.sin(2 * np.pi * 3 * t) / 12)
    phase = 2 * np.pi * np.cumsum(f) / rate
    x = sum(np.sin(k * phase) / k for k in range(1, 11))
    return (amp * x / np.abs(x).max()).astype(np.float32)


def through_phone_line(x16):
    """Band-limit to 300-3400 Hz, downsample to 8 kHz mu-law and back, like Plivo."""
    spec = np.fft.rfft(x16)
    freqs = np.fft.rfftfreq(len(x16), 1 / 16000)
    spec[(freqs < 300) | (freqs > 3400)] = 0
    y = np.fft.irfft(spec, len(x16))[::2]
    pcm = (np.clip(y, -1, 1) * 32767).astype("<i2").tobytes()
    return upsample_8k_to_16k(ulaw_to_float(pcm16_to_ulaw(pcm)))


def test_yin_tracks_pitch_even_without_the_fundamental():
    for f0 in (110, 180, 260):
        f, _ = f0_track(through_phone_line(voice(f0)))
        assert abs(np.median(f[f > 0]) - f0) / f0 < 0.04, f0  # fundamental itself is filtered out


def test_pitch_variation_is_measured():
    flat = extract(through_phone_line(voice(150)))
    lively = extract(through_phone_line(voice(150, vibrato=3)))
    assert flat.f0_std_st < 0.5 < lively.f0_std_st


def P(st, var, db):
    return Prosody(f0_median_hz=100 * 2 ** (st / 12), f0_median_st=st, f0_std_st=var, f0_range_st=var * 2.5,
                   loudness_db=db, voiced_ratio=0.6)


def test_arousal_is_relative_to_the_same_caller():
    sess = SessionState(session_id="s", customer_id="c", prosody_history=[P(8, 1.5, -26), P(8.3, 1.4, -25)])
    raised = TurnInput(session_id="s", customer_id="c", text="x", prosody=P(11.5, 2.6, -20))
    flat = TurnInput(session_id="s", customer_id="c", text="x", prosody=P(7, 0.6, -30))
    same = TurnInput(session_id="s", customer_id="c", text="x", prosody=P(8.1, 1.5, -25.5))
    assert prosody_arousal(raised, sess)[0] > 0.4
    assert prosody_arousal(flat, sess)[0] < -0.3
    assert abs(prosody_arousal(same, sess)[0]) < 0.1
    # no baseline yet -> no claim
    assert prosody_arousal(raised, SessionState(session_id="s", customer_id="c"))[0] == 0.0


def test_raised_voice_amplifies_negative_words_but_not_neutral_ones(monkeypatch):
    monkeypatch.setenv("ACIE_PROSODY", "1")
    est = HumanStateEstimator()
    base = [P(8, 1.5, -26), P(8.2, 1.4, -25), P(7.9, 1.5, -25.5)]
    angry_voice = P(11.5, 2.6, -20)
    sess = SessionState(session_id="s", customer_id="c", prosody_history=base)
    words_only = est.estimate(TurnInput(session_id="s", customer_id="c", text="ye kya hai yaar"), sess)
    words_and_voice = est.estimate(TurnInput(session_id="s", customer_id="c", text="ye kya hai yaar",
                                             prosody=angry_voice), sess)
    assert words_and_voice.frustration > words_only.frustration + 0.15
    assert any("pitch" in s for s in words_and_voice.signals)
    excited = est.estimate(TurnInput(session_id="s", customer_id="c", text="haan batao", prosody=angry_voice), sess)
    assert excited.frustration < 0.2 and excited.engagement > 0.6


def test_a_different_voice_does_not_poison_the_baseline():
    """Live call 8cfac1b1f5: a 369 Hz voicemail prompt was turn 1."""
    sess = SessionState(session_id="s", customer_id="c", prosody_history=[P(22.6, 1.4, -21), P(8.0, 1.5, -22), P(8.4, 1.6, -21)])
    calm = TurnInput(session_id="s", customer_id="c", text="x", prosody=P(8.2, 1.5, -21.5))
    assert abs(prosody_arousal(calm, sess)[0]) < 0.15
    two = SessionState(session_id="s", customer_id="c", prosody_history=[P(22.6, 1.4, -21)])
    a, ev = prosody_arousal(calm, two)
    assert a == 0.0 and "not the same voice" in ev[0]
