"""Shared data contracts between all ACIE layers.

Every layer talks to the next only through these models, so any layer can be
swapped (rules -> ML model, mock API -> real API, template -> LLM) without
touching the others.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field

Level = Literal["low", "med", "high"]
Language = Literal["en", "hi", "hinglish"]
Role = Literal["customer", "agent"]


def _call_context():
    from .context import CallContext
    return CallContext()


def _now() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- input


class TurnTiming(BaseModel):
    """Optional metadata from the STT / telephony layer. All fields may be absent."""

    barge_in: bool = False  # customer interrupted the agent's previous utterance
    response_latency_ms: int | None = None  # gap between agent finishing and customer starting
    speech_rate_wps: float | None = None  # words per second


class Prosody(BaseModel):
    """Acoustic features of one customer utterance (voice/prosody.py). Semitones are re 100 Hz."""

    f0_median_hz: float
    f0_median_st: float  # pitch level
    f0_std_st: float  # pitch variation within the turn
    f0_range_st: float  # p90 - p10 pitch
    loudness_db: float  # mean dBFS of speech frames
    voiced_ratio: float
    words_per_s: float | None = None


class VoiceReading(BaseModel):
    """How this turn sounds compared with the SAME caller's calibrated baseline (voice/calibrator.py).

    z-scores are in units of normal turn-to-turn spread for one calm caller (fitted on real calls),
    so +2 on pitch means "clearly higher than this person usually speaks", whatever their voice.
    """

    calibrated: bool = False  # False while the baseline is still being learnt (first turn)
    baseline_turns: int = 0
    arousal: float = 0.0  # [-1, 1]: flat/quiet/slow ... raised/loud/fast
    confidence: float = 0.0  # [0, 1]
    z: dict[str, float] = Field(default_factory=dict)  # pitch, pitch_var, loudness, effort, rate
    snr_db: float | None = None  # caller speech over this line's noise floor
    evidence: list[str] = Field(default_factory=list)


class TurnInput(BaseModel):
    session_id: str
    customer_id: str
    text: str
    timing: TurnTiming = Field(default_factory=TurnTiming)
    prosody: Prosody | None = None
    voice: VoiceReading | None = None  # set by the voice calibrator; preferred over raw prosody history
    stt_avg_logprob: float | None = None  # speech-to-text confidence from the call record, when it has them (low = garbled)
    stt_no_speech: float | None = None


# --------------------------------------------------------------------------- layer 2: human state


class HumanState(BaseModel):
    """Observable conversational estimates in [0, 1]. Not emotions-as-ground-truth."""

    frustration: float = 0.0
    confusion: float = 0.0
    urgency: float = 0.0
    engagement: float = 0.5
    # extended dims (state lane); all in [0, 1]
    satisfaction: float = 0.5  # liking vs disliking what is offered / said
    anxiety: float = 0.0
    excitement: float = 0.0
    calmness: float = 0.5
    trust: float = 0.5
    certainty: float = 0.5
    cognitive_load: float = 0.0
    attention: float = 0.5
    participation: float = 0.5
    receptiveness: float = 0.5
    signals: list[str] = Field(default_factory=list)  # human-readable evidence for this turn


# --------------------------------------------------------------------------- layer 3: personalization


class PreferenceField(BaseModel):
    value: str
    confidence: float = 0.0  # 0..1
    explicit: bool = False  # customer said it directly ("Hindi mein baat karo") -> always wins
    updated_at: datetime = Field(default_factory=_now)


class CustomerProfile(BaseModel):
    """Long-term preference vector P_t. Persisted across calls."""

    customer_id: str
    preferences: dict[str, PreferenceField] = Field(default_factory=dict)  # language/formality/verbosity/complexity
    interaction_count: int = 0
    outcomes: list[str] = Field(default_factory=list)  # e.g. "resolved", "offer_declined"
    likes: list[str] = Field(default_factory=list)
    dislikes: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)


class ExpressionParams(BaseModel):
    """Short-term adaptation vector A_t: HOW to say the next response."""

    language: Language = "en"
    formality: Level = "med"
    verbosity: Level = "med"
    complexity: Level = "med"
    warmth: Level = "med"
    proactivity: Level = "med"
    acknowledge_emotion: bool = False
    regional_variant: str | None = None  # "delhi", "mumbai", "lucknow", "bihar", "south_indian_english", ...
    explanation_style: Literal["direct", "example", "step_by_step"] = "direct"
    info_density: Level = "med"
    reassurance: Level = "low"
    tone: str = ""  # professional | mid | casual (the campaign's tone_level); "" = mid


# --------------------------------------------------------------------------- layer 4: decision


class Fact(BaseModel):
    """A business fact. The LLM only ever sees the slot name, never writes the value."""

    value: str
    description: str
    must_mention: bool = True
    # verbatim=False: descriptive text (benefits, FAQ answers) the LLM may paraphrase or translate.
    # It is shown to the LLM as a talking point instead of a {slot}. Numbers/prices stay verbatim.
    verbatim: bool = True
    # hidden: used only to fill the fallback template; never shown to (or usable by) the LLM. The
    # natural style replaces "offer_price"/"original_price" with one locked phrase so the LLM can't swap them.
    hidden: bool = False


class TurnPlan(BaseModel):
    """Response specification: WHAT to communicate. Produced deterministically."""

    intent: str
    objective: str
    facts: dict[str, Fact] = Field(default_factory=dict)
    decision: str | None = None
    allowed_actions: list[str] = Field(default_factory=list)
    must_not_say: list[str] = Field(default_factory=list)
    ask: str | None = None  # question the agent must end with, if any
    fallback: dict[str, str] = Field(default_factory=dict)  # language -> safe template using {slots}
    end_call: bool = False
    # False for high-stakes turns (prices, closing the sale): speak the reviewed template only.
    # The first live call had the LLM swap the offer and original prices in a correct-looking sentence.
    llm_allowed: bool = True
    # Reference knowledge about the entity being discussed (what the product is, who makes it).
    # Shown to the LLM to understand questions; not a script to recite.
    context: str | None = None
    # The customer asked about something unrelated: the LLM may answer briefly from general
    # knowledge (no numbers, no claims about the product beyond the facts).
    general_knowledge: bool = False
    # Text fetched for this turn by the world-knowledge lookup (acie/world.py): weather, clock, encyclopedia.
    reference: str | None = None
    # operator-written text spoken BEFORE the LLM reply, word for word (exact price terms: the 14B model
    # garbled "{offer}" when it phrased prices itself). Per language, may use {slots}.
    lead: dict[str, str] | None = None
    persona: str | None = None  # tone of voice from the campaign config (Product.tone)
    # validator rule packs for this product/plan (acie/rules): e.g. ["sales_compliance", "banking"]. Empty = none.
    rule_packs: list[str] = Field(default_factory=list)
    tone_level: str = ""  # professional | mid | casual: the register of this reply (set by the domain; "" = mid)


# --------------------------------------------------------------------------- session


class Utterance(BaseModel):
    role: Role
    text: str


class SessionState(BaseModel):
    session_id: str
    customer_id: str
    turn_index: int = 0
    history: list[Utterance] = Field(default_factory=list)
    state: HumanState = Field(default_factory=HumanState)  # smoothed
    expression: ExpressionParams = Field(default_factory=ExpressionParams)
    active_flags: dict[str, bool] = Field(default_factory=dict)  # hysteresis memory per state dimension
    last_ack_turn: int = -100
    language_streak: dict[str, int] = Field(default_factory=dict)
    dialog: dict[str, str] = Field(default_factory=dict)  # FSM state + slots (domain-owned)
    prosody_history: list[Prosody] = Field(default_factory=list)  # this caller's earlier turns = baseline
    context: "CallContext" = Field(default_factory=lambda: _call_context())  # whole-call ledger (acie/context.py)


# --------------------------------------------------------------------------- layers 5-7


class CompiledPrompt(BaseModel):
    system: str
    messages: list[dict[str, str]]


class ValidationReport(BaseModel):
    ok: bool
    used_fallback: bool = False
    issues: list[str] = Field(default_factory=list)
    issue_codes: list[str] = Field(default_factory=list)  # machine-readable twin of `issues`
    signals: dict[str, int] = Field(default_factory=dict)  # count per issue code (for the feedback lane)


class TurnResult(BaseModel):
    text: str
    plan: TurnPlan
    state: HumanState
    expression: ExpressionParams
    validation: ValidationReport
    timings_ms: dict[str, float] = Field(default_factory=dict)
    result: dict | None = None  # the call-type pack's CallResult so far (acie/calltypes); None for the plain sales call


from .context import CallContext  # noqa: E402  (resolves the forward reference)

SessionState.model_rebuild()
