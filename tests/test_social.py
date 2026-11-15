"""Social signals (acie/state/social.py) and how the style policy uses them (persona stress test, 15 Nov)."""
from acie.psyche import style
from acie.state import social


def _st(text, dialog, turn=4, row="neutral", elder=False, state=None):
    social.update(dialog, text)
    return style.decide_style(row=row, state=state or {"frustration": 0.0, "engagement": 0.6}, text=text, intent="answer",
                              turn_index=turn, dialog=dialog, elder=elder)


def test_rule_read_dims():
    assert social.rule_read("Oh wow, another sales call, my lucky day")["sarcasm"] >= 0.7
    assert social.rule_read("Bakwas mat karo, dimaag mat khao")["hostility"] >= 0.6
    assert social.rule_read("Haha robot ho kya bhai, chai piyoge?")["humour"] >= 0.6
    assert social.rule_read("What exactly is included, and how do I join?")["interest"] >= 0.7
    assert social.rule_read("I have 30 seconds, be quick")["impatience"] >= 0.7
    assert social.rule_read("ये फ्रॉड कॉल तो नहीं है?")["scepticism"] >= 0.7
    plain = social.rule_read("okay tell me")
    assert plain["sarcasm"] == plain["hostility"] == plain["humour"] == 0


def test_nlu_fused_and_smoothed():
    d = {}
    r = social.update(d, "fine", {"sarcasm": 0.9})
    assert r["sarcasm"] > 0.6 and social.turn_value(d, "sarcasm") == 0.9
    r2 = social.update(d, "ok what is the price?")
    assert 0 < r2["sarcasm"] < r["sarcasm"]  # decays slowly, not to zero at once


def test_rapport_moves():
    d = {}
    up = social.update(d, "haha nice one, tell me more")["rapport"]
    assert up > 0.5
    down = social.update(d, "stupid bot, bakwas")["rapport"]
    assert down < up


def test_sarcasm_not_literal():
    st = _st("Oh wow, an AI sales call, my lucky day. What's the pitch?", {}, turn=1)
    assert st.style == "dry-honest" and st.lead_key == "SARCASM_ACK" and not st.humor_ok and not st.ask_ok


def test_rude_gets_calm_no_humour():
    st = _st("Bakwas mat karo, seedha batao kitne ka hai?", {}, turn=1)
    assert st.style == "calm" and not st.humor_ok and st.row == 2


def test_joke_laugh_back_but_limits_hold():
    st = _st("Haha bhai robot ho kya, chai piyoge?", {}, turn=1)
    assert st.style == "playful" and st.lead_key == "LAUGH" and not st.humor_ok  # laugh back; own jokes still wait for turn 3
    assert _st("Haha bhai robot ho kya, chai piyoge?", {}, turn=4, elder=True).style != "playful"
    assert _st("Haha bhai robot ho kya, chai piyoge?", {}, turn=4, row="frustrated").style != "playful"
    assert _st("haha", {}, turn=4, state={"frustration": 0.5, "engagement": 0.6}).style != "playful"


def test_impatience_brief():
    assert _st("I have 30 seconds, get to the point", {}, turn=1).style == "brief"


def test_bank_lines_are_aap_only():
    for key in ("SARCASM_ACK", "LAUGH", "CALM"):
        for lang in ("hi", "hinglish"):
            for line in style.BANKS[key][lang]:
                assert not any(w in line.split() for w in ("तू", "तेरा", "tu", "tera"))


def test_peak_keeps_a_sarcastic_call_final():
    d = {}
    social.update(d, "Oh wow, another sales call, my lucky day")
    social.update(d, "ok what's included?")
    social.update(d, "One whole session? Generous. No thanks.")
    assert social.peak(d, "sarcasm") >= 0.8
