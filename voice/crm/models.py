"""Pydantic models for the CRM layer (the contract is PLATFORM_SPEC.md). All times are epoch seconds (float)."""

from __future__ import annotations

import re
import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

STAGES = ["new", "contacted", "interested", "qualified", "demo_booked", "negotiating", "won", "lost", "do_not_call"]
TERMINAL = {"won", "lost", "do_not_call"}
ID_RE = re.compile(r"^[a-z0-9][a-z0-9_\-]{0,39}$")
LANGS = ("hi", "en", "hinglish")


def new_id(prefix: str = "") -> str:
    return prefix + uuid.uuid4().hex[:10]


def slug(s: str, fallback: str = "x") -> str:
    out = re.sub(r"[^a-z0-9]+", "-", (s or "").lower()).strip("-")[:40].strip("-")
    return out or fallback


class FieldDef(BaseModel):
    key: str
    label: str = ""
    type: Literal["text", "number", "date", "choice", "bool"] = "text"
    applies_to: Literal["customer", "lead", "both"] = "both"
    choices: list[str] = Field(default_factory=list)
    required: bool = False

    @field_validator("key")
    @classmethod
    def _key(cls, v: str) -> str:
        if not re.match(r"^[a-z][a-z0-9_]{0,39}$", v or ""):
            raise ValueError("field key: lower-case letters, digits and _, starting with a letter")
        return v


class Project(BaseModel):
    id: str
    name: str
    color: str = "#888888"
    languages: list[str] = Field(default_factory=lambda: ["hi", "en"])
    timezone: str = "Asia/Kolkata"
    call_hours: list[int] = Field(default_factory=lambda: [10, 20])  # [start_hour, end_hour) local to timezone
    compliance_notes: str = ""
    fields: list[FieldDef] = Field(default_factory=list)
    webhook_token: str = ""  # lead webhook secret; generated on create, shown only to admins
    created_at: float = Field(default_factory=time.time)

    @field_validator("id")
    @classmethod
    def _id(cls, v: str) -> str:
        if not ID_RE.match(v or ""):
            raise ValueError("id: lower-case letters, digits, - and _")
        return v

    @field_validator("call_hours")
    @classmethod
    def _hours(cls, v: list[int]) -> list[int]:
        if len(v) != 2 or not (0 <= v[0] < v[1] <= 24):
            raise ValueError("call_hours: [start, end] with 0 <= start < end <= 24")
        return v

    @field_validator("timezone")
    @classmethod
    def _tz(cls, v: str) -> str:
        from zoneinfo import ZoneInfo
        try:
            ZoneInfo(v)
        except Exception:
            raise ValueError(f"unknown timezone {v!r}")
        return v


class NextAction(BaseModel):
    type: str = "call"  # call | retry | follow_up | callback | nurture | send_link
    due_at: float | None = None
    call_type: str | None = None
    campaign_id: str | None = None
    note: str = ""
    due_estimated: bool = False  # True when the time was not given by the caller (a default was used)


class Consent(BaseModel):
    dnc: bool = False
    reason: str = ""
    at: float | None = None


class History(BaseModel):
    call_ids: list[str] = Field(default_factory=list)
    outcomes: list[dict] = Field(default_factory=list)  # {call_id, outcome, at}
    agreed_price: float | None = None
    objections: list[str] = Field(default_factory=list)
    likes: list[str] = Field(default_factory=list)
    facts: dict[str, Any] = Field(default_factory=dict)


class Customer(BaseModel):
    id: str = Field(default_factory=lambda: new_id("cu"))
    project_id: str
    name: str = ""
    phone: str = ""  # E.164
    email: str = ""
    language: str = "hi"
    city: str = ""
    tags: list[str] = Field(default_factory=list)
    source: Literal["manual", "csv", "webhook", "call"] = "manual"
    stage: str = "new"
    score: int = 0
    consent: Consent = Field(default_factory=Consent)
    custom: dict[str, Any] = Field(default_factory=dict)
    notes: str = ""
    next_action: NextAction | None = None
    history: History = Field(default_factory=History)
    events: list[dict] = Field(default_factory=list)  # timeline: {t, type, ...}
    score_delta_total: int = 0
    last_contact_at: float | None = None
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)

    @field_validator("stage")
    @classmethod
    def _stage(cls, v: str) -> str:
        if v not in STAGES:
            raise ValueError(f"stage must be one of {STAGES}")
        return v


# ---------------------------------------------------------------- campaigns

class RetryPolicy(BaseModel):
    max_attempts: int = Field(3, ge=1, le=10)
    retry_after_minutes: int = Field(240, ge=1)
    retry_on: list[str] = Field(default_factory=lambda: ["no_answer", "voicemail", "failed"])


class Schedule(BaseModel):
    window: list[int] | None = None  # [start_hour, end_hour) overriding the project's call_hours
    days: list[int] = Field(default_factory=lambda: [0, 1, 2, 3, 4, 5])  # Monday=0
    max_parallel: int = Field(1, ge=1, le=20)
    pacing_seconds: int = Field(60, ge=0)  # minimum gap between two dials of one campaign
    retry: RetryPolicy = Field(default_factory=RetryPolicy)


class Audience(BaseModel):
    stages: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)  # any of
    languages: list[str] = Field(default_factory=list)
    min_score: int | None = None
    customer_ids: list[str] = Field(default_factory=list)  # explicit list; when set, the other filters still apply


class Campaign(BaseModel):
    id: str = Field(default_factory=lambda: new_id("cp"))
    project_id: str
    product_id: str = ""
    call_type_id: str = "sales_pitch"
    name: str
    audience: Audience = Field(default_factory=Audience)
    language: str = "hi"
    schedule: Schedule = Field(default_factory=Schedule)
    status: Literal["draft", "running", "paused", "done"] = "draft"
    dial_mode: Literal["manual", "auto"] = "manual"  # auto needs the explicit operator switch (see api.py)
    auto_confirmed: bool = False
    stats: dict[str, int] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)


# ---------------------------------------------------------------- call types (mirror of acie.calltypes.schema)
# TODO(merge): when acie/calltypes/schema.py exists, import CallType from there and delete this mirror.

class CallStep(BaseModel):
    id: str
    text: dict[str, str] = Field(default_factory=dict)  # per language
    answer_type: Literal["text", "yes_no", "number", "choice", "date", "none"] = "text"
    required: bool = False


class CallType(BaseModel):
    id: str
    name: str
    project_id: str | None = None  # None = built-in
    objective: str = ""
    opening: dict[str, str] = Field(default_factory=dict)
    steps: list[CallStep] = Field(default_factory=list)
    allowed_actions: list[str] = Field(default_factory=list)
    success_outcomes: list[str] = Field(default_factory=list)
    closing: dict[str, str] = Field(default_factory=dict)
    tone: str = ""
    max_call_seconds: int = 360
    selling_allowed: bool = True
    negotiation_allowed: bool = True
    compliance_lines: list[str] = Field(default_factory=list)
    builtin: bool = False


BUILTIN_CALLTYPES = [
    CallType(id="sales_pitch", name="Sales pitch", builtin=True, objective="Present the product and close the sale at a fair price",
             allowed_actions=["send_link", "book_callback", "transfer_human", "opt_out"], success_outcomes=["accepted"]),
    CallType(id="lead_qualification", name="Lead qualification", builtin=True, selling_allowed=False, negotiation_allowed=False,
             objective="Find need, budget, authority and timeline; book a demo or callback, no hard sell",
             allowed_actions=["book_demo", "book_callback", "opt_out"], success_outcomes=["demo_booked", "callback"]),
    CallType(id="follow_up", name="Follow-up", builtin=True, objective="Continue from the last call: answer open questions, close or reschedule",
             allowed_actions=["send_link", "book_callback", "transfer_human", "opt_out"], success_outcomes=["accepted", "callback"]),
    CallType(id="payment_reminder", name="Payment reminder", builtin=True, selling_allowed=False, negotiation_allowed=False,
             objective="Remind about a pending payment politely", allowed_actions=["send_link", "book_callback", "opt_out"]),
    CallType(id="feedback_survey", name="Feedback survey", builtin=True, selling_allowed=False, negotiation_allowed=False,
             objective="Collect short feedback", allowed_actions=["opt_out"]),
    CallType(id="appointment_booking", name="Appointment booking", builtin=True, selling_allowed=False,
             objective="Book an appointment or demo", allowed_actions=["book_demo", "book_callback", "opt_out"]),
    CallType(id="custom", name="Custom", builtin=True, objective="Operator-defined steps", allowed_actions=["opt_out"]),
]


# ---------------------------------------------------------------- AI assist: kind -> model of the returned draft

class ProductDraft(BaseModel):
    name: str
    one_liner: str = ""
    price: float | None = None
    description: str = ""


class StepDraft(BaseModel):
    id: str = ""
    text: dict[str, str] = Field(default_factory=dict)
    answer_type: Literal["text", "yes_no", "number", "choice", "date", "none"] = "text"
    required: bool = False


class CampaignDraft(BaseModel):
    name: str
    call_type_id: str = "sales_pitch"
    language: str = "hi"
    audience: Audience = Field(default_factory=Audience)
    schedule: Schedule = Field(default_factory=Schedule)


class CustomerFieldsDraft(BaseModel):
    fields: list[FieldDef]


class ReplyDraft(BaseModel):
    text: dict[str, str]


ASSIST_MODELS: dict[str, type[BaseModel]] = {
    "product": ProductDraft, "calltype": CallType, "campaign": CampaignDraft,
    "customer_fields": CustomerFieldsDraft, "step": StepDraft, "reply": ReplyDraft,
}
