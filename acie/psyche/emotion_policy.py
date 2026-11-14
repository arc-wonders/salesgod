"""Emotion policy: a deterministic TABLE from the 14-dim HumanState (+ the voice reading) to a response strategy.

Replaces the old single special case ("distress -> separate code path"). Crisis is now the most severe ROW of
one table; its regex (acie/safety.py) is one input signal, never a separate branch. Rows are checked in a fixed
priority order, the first match wins, thresholds are fixed numbers (no randomness, no learning at runtime):

  prio row         when (state dims are 0..1; voice arousal -1..1, confidence 0..1)
  ---- ----------- ------------------------------------------------------------------------------------------
   1   crisis      crisis words (self-harm / suicide) OR the call is already flagged (sticky for the whole call)
   2   angry       frustration >= 0.70, or voice arousal >= 0.60 (confidence >= 0.5) with frustration >= 0.40
   3   frustrated  frustration >= 0.40
   4   anxious     anxiety >= 0.60
   5   sad         satisfaction <= 0.25 and engagement <= 0.40 and excitement <= 0.20, or nlu "personal" with low calm
   6   confused    confusion >= 0.60
   7   overwhelmed cognitive_load >= 0.70
   8   suspicious  trust <= 0.30
   9   busy        urgency >= 0.60 and engagement <= 0.40
  10   excited     excitement >= 0.60
  11   calm        everything else

Each row gives: warmth (low/med/high), verbosity (low/med), continue_business (may we keep selling?),
offer_human_or_callback, end_call, acknowledge (say a feeling-word first), sticky.

Crisis always has continue_business False, ack True, sticky True, and the reviewed helpline template stays
the wording (acie/safety.py). The soft rows are applied only with ACIE_PSYCHE=1; the crisis row always.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..safety import is_distress

ROWS = ("crisis", "angry", "frustrated", "anxious", "sad", "confused", "overwhelmed", "suspicious", "busy", "excited", "calm")


@dataclass(frozen=True)
class Strategy:
    row: str
    warmth: str = "med"          # low | med | high
    verbosity: str = "med"       # low | med
    continue_business: bool = True
    offer_human_or_callback: bool = False
    end_call: bool = False
    acknowledge: bool = False
    sticky: bool = False


_TABLE = {
    "crisis": Strategy("crisis", "high", "low", False, True, False, True, True),
    "angry": Strategy("angry", "high", "low", False, True, False, True),
    "frustrated": Strategy("frustrated", "high", "low", False, False, False, True),
    "anxious": Strategy("anxious", "high", "med", False, False, False, True),
    "sad": Strategy("sad", "high", "low", False, False, False, True),
    "confused": Strategy("confused", "med", "low", True, False, False, False),
    "overwhelmed": Strategy("overwhelmed", "med", "low", True, False, False, False),
    "suspicious": Strategy("suspicious", "med", "med", False, False, False, True),
    "busy": Strategy("busy", "low", "low", False, True, False, False),
    "excited": Strategy("excited", "med", "med", True, False, False, False),
    "calm": Strategy("calm"),
}


def _v(voice, name, default=0.0):
    return float(getattr(voice, name, default) or default) if voice is not None else default


def decide(state, text: str = "", voice=None, flagged: bool = False, nlu_intent: str = "") -> Strategy:
    """state: HumanState or dict of dims. First matching row wins (see the module table)."""
    g = (lambda k, d=0.0: float(state.get(k, d))) if isinstance(state, dict) else (lambda k, d=0.0: float(getattr(state, k, d)))
    if flagged or is_distress(text):
        return _TABLE["crisis"]
    arousal = _v(voice, "arousal")
    voice_ok = _v(voice, "confidence") >= 0.5 and bool(getattr(voice, "calibrated", True))
    fr = g("frustration")
    if fr >= 0.70 or (voice_ok and arousal >= 0.60 and fr >= 0.40):
        return _TABLE["angry"]
    if fr >= 0.40:
        return _TABLE["frustrated"]
    if g("anxiety") >= 0.60:
        return _TABLE["anxious"]
    if (g("satisfaction", 0.5) <= 0.25 and g("engagement", 0.5) <= 0.40 and g("excitement") <= 0.20) \
            or (nlu_intent == "personal" and g("calmness", 0.5) <= 0.40):
        return _TABLE["sad"]
    if g("confusion") >= 0.60:
        return _TABLE["confused"]
    if g("cognitive_load") >= 0.70:
        return _TABLE["overwhelmed"]
    if g("trust", 0.5) <= 0.30:
        return _TABLE["suspicious"]
    if g("urgency") >= 0.60 and g("engagement", 0.5) <= 0.40:
        return _TABLE["busy"]
    if g("excitement") >= 0.60:
        return _TABLE["excited"]
    return _TABLE["calm"]


def apply_to_expression(expr, strat: Strategy):
    """Soft rows only (ACIE_PSYCHE=1): nudge the style parameters; explicit user preferences stay untouched
    because the policy already set them and we only move toward calmer/shorter."""
    upd = {}
    if strat.verbosity == "low" and getattr(expr, "verbosity", "low") != "low":
        upd["verbosity"] = "low"
    if strat.warmth == "high" and getattr(expr, "warmth", "high") != "high":
        upd["warmth"] = "high"
    if strat.acknowledge and hasattr(expr, "acknowledge_emotion") and not expr.acknowledge_emotion:
        upd["acknowledge_emotion"] = True
    return expr.model_copy(update=upd) if upd else expr
