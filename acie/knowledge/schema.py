"""Product-agnostic knowledge base schema.

One KB per (tenant, product). It holds a topic taxonomy, Q&A entries in en / hi / hinglish, offerings (add-ons the
operator can sell) and an interest-tag taxonomy that lets the decision tree infer what a caller cares about.

Rules that are enforced here (not left to convention):
  * every entry carries a provenance: "operator" | "public" | "synthetic" (mandatory, no default);
  * a KB with allow_synthetic=False can never hold or serve a synthetic entry (real companies' products);
  * prices never appear as literals in answer text: answers use slots ({offer_price}, {price:OFFERING_ID}) that are
    filled from the product config and the offering table at render time (see render.py).
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

LANGS = ("en", "hi", "hinglish")
Provenance = Literal["operator", "public", "synthetic"]
_ID = re.compile(r"^[a-z0-9][a-z0-9_\-]*$")
_PRICE_LITERAL = re.compile(r"(₹|\bRs\.?|\bINR|\$|रुपये|रुपए|रुपया)\s?\d|\d[\d,]*\s?(₹|रुपये|रुपए|rupees|rupaye)", re.I)


class Text3(BaseModel):
    """The same text in the three call languages. English is required; missing hi/hinglish fall back to it."""
    model_config = ConfigDict(extra="forbid")
    en: str
    hi: str = ""
    hinglish: str = ""

    def get(self, lang: str) -> str:
        return (getattr(self, lang, "") or self.en) if lang in LANGS else self.en

    def all(self) -> list[str]:
        return [t for t in (self.en, self.hi, self.hinglish) if t]


class Variants(BaseModel):
    """Ways a caller might ask the same thing, per language."""
    model_config = ConfigDict(extra="forbid")
    en: list[str] = Field(default_factory=list)
    hi: list[str] = Field(default_factory=list)
    hinglish: list[str] = Field(default_factory=list)

    def all(self) -> list[str]:
        return [*self.en, *self.hi, *self.hinglish]


class Topic(BaseModel):
    """A node of the taxonomy. path is "topic" or "topic/subtopic"; overview is what we say when nothing matches."""
    model_config = ConfigDict(extra="forbid")
    path: str
    title: Text3
    overview: Text3
    interests: list[str] = Field(default_factory=list)  # tags this topic speaks to

    @property
    def parent(self) -> str:
        return self.path.rsplit("/", 1)[0] if "/" in self.path else ""

    @field_validator("path")
    @classmethod
    def _path(cls, v: str) -> str:
        if not v or any(not _ID.match(p) for p in v.split("/")):
            raise ValueError(f"bad topic path {v!r}")
        return v


class Entry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    topic_path: str
    question_variants: Variants
    answer: Text3
    facts: dict[str, str | int | float] = Field(default_factory=dict)  # slot values and numbers the answer relies on
    offerings: list[str] = Field(default_factory=list)                  # offering ids this entry relates to
    interests: list[str] = Field(default_factory=list)                  # interest tags it serves
    provenance: Provenance                                              # mandatory: no default on purpose
    confidence: float = Field(1.0, ge=0, le=1)
    source: str = ""                                                    # where a public / operator fact comes from
    teach: bool = False                                                 # a short tip worth teaching unprompted

    @field_validator("id")
    @classmethod
    def _id(cls, v: str) -> str:
        if not _ID.match(v):
            raise ValueError(f"bad entry id {v!r}")
        return v

    @model_validator(mode="after")
    def _answers(self):
        if not self.answer.en.strip():
            raise ValueError(f"{self.id}: English answer required")
        if not self.question_variants.all():
            raise ValueError(f"{self.id}: at least one question variant")
        return self


class Offering(BaseModel):
    """Something the operator can sell on top of / inside the base product (add-on, program, pack)."""
    model_config = ConfigDict(extra="forbid")
    id: str
    name: Text3
    who_suits: list[str]                       # interest tags it fits
    price_inr: int | None = None               # None = no separate price (included, or shown elsewhere)
    price_label: Text3 | None = None           # spoken price, e.g. "₹3,000 per month"; the ONLY place prices live
    included: bool = False                     # part of the base membership
    floor_inr: int | None = None               # profit floor for this offering, when the operator sets one
    pitch: Variants = Field(default_factory=Variants)   # pitch lines per language (no prices; prices come as a slot)
    upsell_from: list[str] = Field(default_factory=list)  # topic paths that lead naturally to it
    provenance: Provenance
    min_fit: float = 1.0                       # interest-match score needed before it may be pitched

    @model_validator(mode="after")
    def _price(self):
        if self.price_inr is not None and self.price_label is None:
            raise ValueError(f"{self.id}: price_inr needs price_label")
        if self.price_inr is not None and self.floor_inr is not None and self.price_inr < self.floor_inr:
            raise ValueError(f"{self.id}: price {self.price_inr} below floor {self.floor_inr}")
        if not self.who_suits:
            raise ValueError(f"{self.id}: who_suits must name at least one interest tag")
        return self


class InterestTag(BaseModel):
    """An interest the caller can show. `keywords` are what they say (any script); related tags spread a little score."""
    model_config = ConfigDict(extra="forbid")
    id: str
    label: Text3
    keywords: list[str]
    related: list[str] = Field(default_factory=list)
    discovery: Variants = Field(default_factory=Variants)  # a question that finds out more about this interest


class KnowledgeBase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    product_id: str
    tenant: str = "default"
    allow_synthetic: bool = False
    description: str = ""
    topics: list[Topic]
    entries: list[Entry]
    offerings: list[Offering] = Field(default_factory=list)
    interests: list[InterestTag] = Field(default_factory=list)
    root_overview: Text3 | None = None

    @model_validator(mode="after")
    def _consistent(self):
        tp = {t.path for t in self.topics}
        if len(tp) != len(self.topics):
            raise ValueError("duplicate topic path")
        for t in self.topics:
            if t.parent and t.parent not in tp:
                raise ValueError(f"topic {t.path}: parent {t.parent} missing")
        ids = [e.id for e in self.entries]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate entry id")
        oid = {o.id for o in self.offerings}
        tags = {i.id for i in self.interests}
        for e in self.entries:
            if e.topic_path not in tp:
                raise ValueError(f"{e.id}: unknown topic {e.topic_path}")
            for o in e.offerings:
                if o not in oid:
                    raise ValueError(f"{e.id}: unknown offering {o}")
            for i in e.interests:
                if tags and i not in tags:
                    raise ValueError(f"{e.id}: unknown interest {i}")
            if not self.allow_synthetic and e.provenance == "synthetic":
                raise ValueError(f"{e.id}: synthetic entry in a KB with allow_synthetic=false")
            for lang in LANGS:
                if _PRICE_LITERAL.search(e.answer.get(lang)) and lang_has(e, lang):
                    raise ValueError(f"{e.id}: price written as a literal in the {lang} answer; use a slot")
        for o in self.offerings:
            if not self.allow_synthetic and o.provenance == "synthetic":
                raise ValueError(f"{o.id}: synthetic offering in a KB with allow_synthetic=false")
            for i in o.who_suits:
                if tags and i not in tags:
                    raise ValueError(f"{o.id}: unknown interest {i}")
            for p in o.upsell_from:
                if p not in tp:
                    raise ValueError(f"{o.id}: unknown upsell topic {p}")
        return self

    # ---- lookups (cheap, built once)
    def entry_map(self) -> dict[str, Entry]:
        return {e.id: e for e in self.entries}

    def topic_map(self) -> dict[str, Topic]:
        return {t.path: t for t in self.topics}

    def offering_map(self) -> dict[str, Offering]:
        return {o.id: o for o in self.offerings}

    def usable_entries(self) -> list[Entry]:
        """Entries that may be served: synthetic ones only when the KB allows them."""
        return [e for e in self.entries if self.allow_synthetic or e.provenance != "synthetic"]


def lang_has(e: Entry, lang: str) -> bool:
    return bool(getattr(e.answer, lang, ""))
