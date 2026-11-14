"""Call types: the contract shared with the CRM / API lane (they import `CallType` and `CallResult` from here).

A call type is DATA: a template of one kind of call (objective, opening, ordered questions, allowed actions, success
outcomes, closing, tone, limits). The domain packs in acie/decision/domains/ run it deterministically.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

LANGS = ("en", "hi", "hinglish")
AnswerType = Literal["yes_no", "choice", "number", "text", "date_time", "money"]
# which domain pack runs this type (sales_pitch = today's ProductSalesDomain, unchanged)
Pack = Literal["sales_pitch", "lead_qualification", "follow_up", "steps"]

KNOWN_ACTIONS = {"send_link", "book_callback", "book_demo", "book_appointment", "transfer_human", "opt_out",
                 "send_sms", "take_payment_promise", "close"}

Text3 = dict[str, str]  # {"en": ..., "hi": ..., "hinglish": ...}


def _text3(v):
    """A plain string means the same line in every language."""
    if isinstance(v, str):
        return {l: v for l in LANGS}
    return v


class Choice(BaseModel):
    id: str
    label: Text3 = Field(default_factory=dict)
    keywords: list[str] = Field(default_factory=list)  # extra words that select this choice (any language)
    outcome: str = ""  # the call outcome this choice earns (e.g. "declined", "paid")
    ends_call: bool = False  # choosing it ends the call politely (no further steps)

    @model_validator(mode="before")
    @classmethod
    def _plain(cls, v):
        if isinstance(v, str):
            return {"id": v}
        return v

    @field_validator("label", mode="before")
    @classmethod
    def _lab(cls, v):
        return _text3(v)


class Step(BaseModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,39}$")
    text: Text3
    answer_type: AnswerType = "text"
    choices: list[Choice] = Field(default_factory=list)
    required: bool = True
    skip_if: str = ""  # "authority == no", "budget < 5000", "decision != need_time", "score < 40", "answered:need" (all lower case values)
    # optional extras (additive): words that let a later step be answered early ("budget" in "mera budget 10 hazaar hai")
    keywords: list[str] = Field(default_factory=list)
    # deterministic score: value -> points (yes_no / choice values), plus points for any real answer
    score: dict[str, int] = Field(default_factory=dict)
    score_answered: int = 0
    # when this step gets a real (positive) answer: the next action the CRM schedules / the call outcome it earns
    action: str = ""
    outcome: str = ""

    @field_validator("text", mode="before")
    @classmethod
    def _t(cls, v):
        return _text3(v)


class CallType(BaseModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,39}$")
    name: str
    objective: str = ""  # what success is, in the operator's words
    pack: Pack = "steps"
    languages: list[str] = Field(default_factory=lambda: list(LANGS))
    opening: Text3 = Field(default_factory=dict)  # per language; must say who is calling and why (AI disclosure is added if missing)
    steps: list[Step] = Field(default_factory=list)
    allowed_actions: list[str] = Field(default_factory=list)
    success_outcomes: list[str] = Field(default_factory=list)
    closing: Text3 = Field(default_factory=dict)  # normal end of the call
    closing_declined: Text3 = Field(default_factory=dict)  # they said no / not now: a polite close, never pushy
    tone: str = ""
    max_call_s: int = Field(300, ge=30, le=3600)
    selling_allowed: bool = False
    negotiation_allowed: bool = False
    compliance_lines: list[Text3] = Field(default_factory=list)  # extra lines spoken right after the opening (e.g. "this call is recorded")
    builtin: bool = False
    project_id: str = ""
    description: str = ""

    @field_validator("opening", "closing", "closing_declined", mode="before")
    @classmethod
    def _t(cls, v):
        return _text3(v)

    @field_validator("compliance_lines", mode="before")
    @classmethod
    def _cl(cls, v):
        return [_text3(x) for x in (v or [])]

    def step(self, step_id: str) -> Step | None:
        return next((s for s in self.steps if s.id == step_id), None)

    def line(self, field: str, lang: str) -> str:
        """The operator's text in `lang`, else Hinglish, else English, else any."""
        d = getattr(self, field) or {}
        return d.get(lang) or d.get("hinglish") or d.get("en") or next(iter(d.values()), "")


class NextAction(BaseModel):
    type: str = ""  # follow_up_call | book_demo | book_callback | book_appointment | send_link | human_call | nurture | none
    due_at: str | None = None  # ISO 8601 with offset (Asia/Kolkata by default), None = not scheduled
    call_type: str = ""  # which call type the next call should use


class CallResult(BaseModel):
    """What one call produced, for the CRM lane to apply to the customer (stage, score, next action, answers)."""
    outcome: str = ""  # e.g. demo_booked | qualified | not_qualified | callback_requested | declined | opt_out | link_sent | completed ...
    stage_suggestion: str = ""  # new | contacted | interested | qualified | demo_booked | negotiating | won | lost | do_not_call
    next_action: NextAction = Field(default_factory=NextAction)
    answers: dict[str, str | int | float | bool | None] = Field(default_factory=dict)  # step_id -> value
    score_delta: int = 0  # deterministic points this call earned (the CRM adds it to the customer's 0-100 score)
    # additive extras for the CRM timeline
    call_type: str = ""
    facts: dict[str, str] = Field(default_factory=dict)  # agreed_price, callback_text, link_sent, objections, open_questions ...
    complete: bool = False  # the call ended (False while the call is still going)


# outcome -> suggested CRM stage (deterministic, one place)
STAGE_FOR_OUTCOME = {
    "demo_booked": "demo_booked", "appointment_booked": "demo_booked", "qualified": "qualified", "interested": "interested",
    "callback_requested": "contacted", "callback_booked": "contacted", "human_requested": "interested", "not_qualified": "contacted",
    "declined": "lost", "not_interested": "lost", "opt_out": "do_not_call", "link_sent": "negotiating", "ready_to_buy": "negotiating",
    "promised_to_pay": "negotiating", "paid": "won", "completed": "contacted", "feedback_given": "contacted",
    "no_answer": "contacted", "ended_early": "contacted", "undecided": "contacted", "safety": "contacted",
}


def validate_call_type(ct: CallType) -> list[str]:
    """Human-readable problems that would break a call (empty list = fine)."""
    errs: list[str] = []
    if not ct.name.strip():
        errs.append("name is empty")
    for l in ct.languages:
        if l not in LANGS:
            errs.append(f"unknown language {l!r} (use en, hi, hinglish)")
    langs = [l for l in ct.languages if l in LANGS] or ["en"]
    if not ct.opening and ct.pack != "sales_pitch":
        errs.append("opening is missing")
    for l in langs:
        if ct.opening and not ct.opening.get(l):
            errs.append(f"opening has no {l} text")
        if ct.closing and not ct.closing.get(l):
            errs.append(f"closing has no {l} text")
    if ct.pack in ("steps", "lead_qualification", "follow_up") and not ct.steps:
        errs.append("a step-driven call type needs at least one step")
    seen: set[str] = set()
    for s in ct.steps:
        if s.id in seen:
            errs.append(f"duplicate step id {s.id!r}")
        seen.add(s.id)
        for l in langs:
            if not s.text.get(l):
                errs.append(f"step {s.id!r} has no {l} text")
        if s.answer_type == "choice" and len(s.choices) < 2:
            errs.append(f"step {s.id!r} is a choice but has fewer than 2 choices")
        if s.answer_type != "choice" and s.choices:
            errs.append(f"step {s.id!r} has choices but its answer type is {s.answer_type}")
        if s.action and s.action not in KNOWN_ACTIONS:
            errs.append(f"step {s.id!r} uses unknown action {s.action!r}")
        if s.action and ct.allowed_actions and s.action not in ct.allowed_actions:
            errs.append(f"step {s.id!r} uses action {s.action!r} which is not in allowed_actions")
        if s.skip_if and not _SKIP_RE.match(s.skip_if.strip()):
            errs.append(f"step {s.id!r} has an unreadable skip_if {s.skip_if!r}")
    for a in ct.allowed_actions:
        if a not in KNOWN_ACTIONS:
            errs.append(f"unknown allowed action {a!r}")
    if not ct.success_outcomes:
        errs.append("success_outcomes is empty")
    outs = set(ct.success_outcomes) | {"declined", "opt_out", "callback_requested", "human_requested", "completed", "ended_early", "undecided", "not_qualified"}
    for s in ct.steps:
        if s.outcome and s.outcome not in outs and s.outcome not in STAGE_FOR_OUTCOME:
            errs.append(f"step {s.id!r} earns outcome {s.outcome!r} which is not listed in success_outcomes")
    return errs


_SKIP_RE = re.compile(r"^(answered:[a-z][a-z0-9_]*|unanswered:[a-z][a-z0-9_]*|[a-z][a-z0-9_]*\s*(==|!=|>=|<=|>|<)\s*\S.*)$")
