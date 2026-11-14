"""Layer 3 - Adaptation policy  A_t = pi(H_t, P_t, C_t).

v1 is an explicit, auditable rule policy with hysteresis. Later this becomes a
contextual bandit over the same discrete ExpressionParams, using the rule
policy as the safe default / prior.
"""

from __future__ import annotations

import re

from ..schemas import CustomerProfile, ExpressionParams, HumanState, Language, SessionState
from ..text import detect_language, padded
from . import knobs as _knobs

# Explicit customer requests -> preference overrides. Patterns run on normalized
# ASR text padded with spaces (" tok tok "), so " x " is a whole-word match and
# Devanagari has no nukta/chandrabindu ("थोड़ा" -> "थोडा", "अंग्रेज़ी" -> "अंग्रेजी").
_HI = r"(?:hindi|hindee|hindhi|हिंदी|हिन्दी)"
_EN = r"(?:english|inglish|englis|angrezi|angreji|angrezee|अंग्रेजी|इंग्लिश|इंगलिश)"
_IN = r"(?:me|mein|mai|main|mei|men|में|मे|मैं)"
_NEG = r"(?:nahi|nahin|nhi|नहीं|नही|mat|मत)"
_KNOW = r"(?:aati|aata|aaati|ati|samajh\S*|आती|आता|समझ\S*)"
_DONT = r"(?:dont|do not|cant|cannot|can not)"
_SPEAK = r"(?:speak|talk|reply|answer|continue|baat|bol\S*|बात|बोल\S*)"


def _rx(*alts: str) -> re.Pattern[str]:
    return re.compile("|".join(alts))


_EXPLICIT = [
    (_rx(rf" {_HI} {_IN} ", rf" {_SPEAK} (?:in )?{_HI} ", rf" in {_HI} ", rf" {_HI} please ",
         rf" please {_HI} ", rf" {_HI} (?:bol|बोल)\S* ", rf" {_HI} (?:chalegi|chalega|चलेगी|चलेगा) ",
         rf" {_EN} {_NEG} {_KNOW} ", rf" {_DONT} (?:understand|speak|know) {_EN} "),
     "language", "hi"),
    (_rx(rf" {_EN} {_IN} ", rf" {_SPEAK} (?:in )?{_EN} ", rf" in {_EN} ", rf" {_EN} please ",
         rf" please {_EN} ", rf" {_EN} (?:bol|बोल)\S* ", rf" {_EN} (?:chalegi|chalega|चलेगी|चलेगा) ",
         rf" {_HI} {_NEG} {_KNOW} ", rf" {_DONT} (?:understand|speak|know) {_HI} ",
         # live call 9: "I am not able to understand your language... can you shift to English"
         rf" (?:shift|switch|change|move|go) (?:to |in |into )?{_EN} ", rf" (?:not able to|cannot|cant|unable to) "
         rf"understand (?:your|this|the) language ", rf" {_EN} (?:mein|me|में) (?:baat|बात|bolo|बोलो) "),
     "language", "en"),
    (_rx(r" hinglish "), "language", "hinglish"),
    (_rx(r" short (?:me|mein|mai|में|मे) ", r" in short ", r" (?:be|keep it|tell me|explain) brief ", r" briefly ", r" keep it short ",
         r" short (?:rakho|rakhiye|rakhna|karo|kijiye|रखो|रखिए|रखें|करो) ", r" seedh[ae] bol\S* ",
         r" seedhi baat ", r" सीधा ", r" सीधे बोल\S* ", r" सीधी बात ", r" jaldi jaldi (?:bata|bol)\S* ",
         r" जल्दी जल्दी (?:बता|बोल)\S* ", r" to the point ", r" point (?:pe|par) aa\S* ",
         r" पॉइंट पर आ\S* ", r" kam shabdon ", r" कम शब्दों ", r" sankshep ", r" संक्षेप ",
         r" छोटे में ", r" lamba mat ", r" लंबा मत ", r" in a nutshell ", r" just the (?:number|amount) "),
     "verbosity", "low"),
    (_rx(r" detail (?:me|mein|mai|में|मे|se|से) ", r" in detail ", r" detailed ", r" details? (?:me|mein) ",
         r" vistar\S* ", r" विस्तार\S* ", r" डिटेल (?:में|से|मे) ", r" full detail", r" poori (?:detail|jankari) ",
         r" पूरी (?:डिटेल|जानकारी) ", r" explain (?:it )?(?:properly|fully|more) ", r" tell me more ",
         r" more details? "),
     "verbosity", "high"),
    (_rx(r" simple (?:me|mein|में|language|words|bhasha|shabdon|terms|karke|tarike|way) ",
         r" in simple ", r" simply ", r" keep it simple ", r" make it simple ",
         r" (?:aasan|asaan|aasaan|asan) (?:bhasha|shabd\S*|words|language|tarike|karke|mein|me) ",
         r" (?:आसान|सरल) (?:भाषा|शब्द\S*|तरीके|करके|में) ", r" saral ", r" easy (?:words|language|way) ",
         r" layman", r" (?:thoda|thodi|thora) (?:dheere|dhire|slow)", r" थोडा धीरे", r" थोडी धीरे",
         r" (?:dheere|dhire) (?:bol|bata|samjha)\S* ", r" धीरे (?:बोल|बता|समझा)\S* ",
         r" slow(?:ly| down) ", r" speak slower ", r" slower "),
     "complexity", "low"),
]
# A match here cancels that value ("hindi nahi aati", "not in english").
_EXPLICIT_BLOCK = [
    (_rx(rf" {_HI} (?:{_IN} )?{_NEG} ", rf" not (?:in )?{_HI} ", rf" no {_HI} ",
         rf" {_DONT} (?:understand|speak|know) {_HI} ",
         # persona review: "Why are you speaking Hindi? Speak English" switched the call TO Hindi
         rf" why (?:are you |r u |you )?(?:speaking|talking|talk|speak) (?:in )?{_HI} "), "language", "hi"),
    (_rx(rf" {_EN} (?:{_IN} )?{_NEG} ", rf" not (?:in )?{_EN} ", rf" no {_EN} ",
         rf" {_DONT} (?:understand|speak|know) {_EN} "), "language", "en"),
]

_EXPLICIT.extend(_knobs.EXPLICIT)
_EXPLICIT_BLOCK.extend(_knobs.BLOCK)

LEVELS = ("low", "med", "high")
STYLES = ("direct", "example", "step_by_step")
ENTER, EXIT = 0.5, 0.3  # hysteresis thresholds for state flags
LANGUAGE_SWITCH_STREAK = 2  # implicit language switch needs N consecutive turns
ACK_COOLDOWN_TURNS = 3  # don't say "I understand your frustration" every turn
PREF_MIN_CONFIDENCE = 0.6
HI_ENTER, HI_EXIT = 0.7, 0.5  # stricter band for dims whose neutral value is 0.5
_KNOBS = {"formality", "verbosity", "complexity", "warmth", "proactivity", "info_density", "explanation_style"}


def detect_explicit_requests(text: str) -> dict[str, str]:
    low = padded(text)
    blocked = {(key, val) for rx, key, val in _EXPLICIT_BLOCK if rx.search(low)}
    out: dict[str, str] = {}
    pos: dict[str, int] = {}
    for rx, key, val in _EXPLICIT:
        if (key, val) in blocked:
            continue
        m = None
        for m in rx.finditer(low):  # last occurrence: "english... nahi, hindi mein" -> the later one
            pass
        if m and m.start() >= pos.get(key, -1):
            out[key], pos[key] = val, m.start()
    return out


# Engagement starts at 0.5 (neutral), so "disengaged" needs its own, stricter band;
# with the shared 0.5 threshold every ordinary turn counted as disengaged.
DISENGAGED_ENTER, DISENGAGED_EXIT = 0.7, 0.5


def _flag(session: SessionState, name: str, value: float,
          enter: float = ENTER, exit_: float = EXIT) -> bool:
    """Schmitt trigger: on at/above `enter`, off at/below `exit_`, otherwise keep previous."""
    prev = session.active_flags.get(name, False)
    now = value >= enter if not prev else value > exit_
    session.active_flags[name] = now
    return now


def _valid(value: str, allowed: tuple[str, ...], default: str) -> str:
    return value if value in allowed else default


class AdaptationPolicy:
    def decide(self, text: str, state: HumanState, profile: CustomerProfile,
               session: SessionState, explicit: dict[str, str], intent: str | None = None) -> ExpressionParams:
        return self.decide_traced(text, state, profile, session, explicit, intent)[0]

    def decide_traced(self, text: str, state: HumanState, profile: CustomerProfile,
                      session: SessionState, explicit: dict[str, str],
                      intent: str | None = None) -> tuple[ExpressionParams, set[str]]:
        """Rule decision plus the knobs that are *protected*: set by an explicit request or a
        safety/state rule, so a learned policy must never override them."""
        prefs = profile.preferences
        protected: set[str] = set()

        def pref(key: str, default: str) -> str:
            if key in explicit:
                return explicit[key]
            p = prefs.get(key)
            if p and (p.explicit or p.confidence >= PREF_MIN_CONFIDENCE):
                return p.value
            return default

        variant = pref("regional_variant", "")
        a = ExpressionParams(
            language=self._language(text, profile, session, explicit),
            formality=pref("formality", "med"),  # type: ignore[arg-type]
            verbosity=pref("verbosity", "med"),  # type: ignore[arg-type]
            complexity=pref("complexity", "med"),  # type: ignore[arg-type]
            regional_variant=None if variant in ("", "standard", "none") else variant,
            explanation_style=_valid(pref("explanation_style", "direct"), STYLES, "direct"),  # type: ignore[arg-type]
            info_density=_valid(pref("info_density", "med"), LEVELS, "med"),  # type: ignore[arg-type]
        )

        # A clarification request is direct evidence of confusion even when the
        # wording isn't in the estimator's lexicon ("which limit", "money went or not").
        confused = _flag(session, "confused", max(state.confusion, ENTER if intent == "clarify" else 0.0))
        frustrated = _flag(session, "frustrated", state.frustration)
        urgent = _flag(session, "urgent", state.urgency)
        disengaged = _flag(session, "disengaged", 1 - state.engagement, DISENGAGED_ENTER, DISENGAGED_EXIT)
        # newer state dimensions (state lane): read defensively, neutral defaults keep every flag off
        anxious = _flag(session, "anxious", getattr(state, "anxiety", 0.0))
        distrustful = _flag(session, "distrustful", 1 - getattr(state, "trust", 0.5), HI_ENTER, HI_EXIT)
        overloaded = _flag(session, "overloaded", getattr(state, "cognitive_load", 0.0))
        certain = _flag(session, "certain", getattr(state, "certainty", 0.5), HI_ENTER, HI_EXIT)
        engaged = _flag(session, "engaged", state.engagement, HI_ENTER, HI_EXIT)

        if confused:
            a.complexity = "low"
            a.verbosity = "med" if a.verbosity == "high" else a.verbosity  # simpler, not longer
            a.proactivity = "low"
            protected |= {"complexity", "verbosity", "proactivity"}
        if frustrated:
            a.warmth = "high"
            a.verbosity = "low"
            a.proactivity = "low"
            a.formality = "high" if a.formality != "low" else "med"
            protected |= {"warmth", "verbosity", "proactivity", "formality"}
            if session.turn_index - session.last_ack_turn >= ACK_COOLDOWN_TURNS:
                a.acknowledge_emotion = True
                session.last_ack_turn = session.turn_index
        if urgent:
            a.verbosity = "low"
            a.proactivity = "low"
            protected |= {"verbosity", "proactivity"}
        if disengaged and not (confused or frustrated):
            a.verbosity = "low"
            protected.add("verbosity")
        if anxious or distrustful or explicit.get("reassurance") == "high":
            a.reassurance = "high"
            a.explanation_style = "step_by_step"
            protected |= {"explanation_style"}
        if overloaded:
            a.info_density = "low"
            a.complexity = "low"
            protected |= {"info_density", "complexity"}
        elif certain and engaged and "info_density" not in explicit:
            a.info_density = "high"

        # explicit requests always win over state-driven changes
        for key in ("verbosity", "complexity", "info_density", "explanation_style", "reassurance"):
            if key in explicit and explicit[key] in (LEVELS + STYLES):
                setattr(a, key, explicit[key])
        if "regional_variant" in explicit:
            a.regional_variant = None if explicit["regional_variant"] == "standard" else explicit["regional_variant"]
        protected |= {k for k in explicit if k in _KNOBS}
        if session.dialog.get("safety_flag"):
            protected |= _KNOBS
        return a, protected

    def _language(self, text: str, profile: CustomerProfile, session: SessionState,
                  explicit: dict[str, str]) -> Language:
        if "language" in explicit:
            session.language_streak = {}
            return explicit["language"]  # type: ignore[return-value]
        p = profile.preferences.get("language")
        if p and p.explicit:
            return p.value  # type: ignore[return-value]

        # the agent's opening (spoken in the operator's chosen language) counts: a one-word first reply
        # must not flip a Hindi call to English from an old profile (call 77b155aed4: "बेबाई" -> English pitch)
        current = session.expression.language if (session.turn_index > 0 or session.history) else None
        # loanword-only turns ("ओके थैंक यू", "ok") carry no language signal: keep the current one
        detected = detect_language(text, neutral=current)
        if current is None:  # first turn: trust profile if confident, else mirror the customer
            # ...unless they clearly speak another language right now (live call 9: an English
            # "Yes, I have a minute, what do you want to talk about?" got a Hindi reply from history)
            if len(text.split()) >= 4:
                return detected
            if p and p.confidence >= PREF_MIN_CONFIDENCE:
                return p.value  # type: ignore[return-value]
            return detected
        if detected == current:
            session.language_streak = {}
            return current
        streak = session.language_streak.get(detected, 0) + 1
        session.language_streak = {detected: streak}
        # one short turn ("ok", "haan") shouldn't flip the language; a full sentence in another
        # language is a clear signal, so it switches immediately
        needed = 1 if len(text.split()) >= 4 else LANGUAGE_SWITCH_STREAK
        return detected if streak >= needed else current


def make_default_policy():
    """Rule policy unless ACIE_POLICY=bandit (learned policy with the rule policy as its prior)."""
    import os
    if os.environ.get("ACIE_POLICY", "").lower() == "bandit":
        from .bandit import BanditPolicy
        return BanditPolicy()
    return AdaptationPolicy()
