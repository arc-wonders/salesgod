"""The generation spec: operator inputs plus the explicit CRITERIA a generated version must meet.

Both are saved with every version, so a version always says what it was asked for and what it was checked against."""

from __future__ import annotations

import re
import unicodedata
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

LANGS = ("en", "hi", "hinglish")

# Names that are real companies' products. Matching is on whole words of the product name / seller. This list is a
# safety net, not a database: the operator can also set `real_brand` explicitly in the spec.
KNOWN_BRANDS = frozenset("""
claude chatgpt openai anthropic gemini google copilot microsoft apple iphone amazon prime flipkart netflix spotify hotstar disney
youtube zomato swiggy uber ola paytm phonepe gpay razorpay jio airtel vodafone vi tata reliance infosys wipro hdfc icici sbi axis kotak
nike adidas puma samsung xiaomi oneplus oppo vivo realme sony lg dell hp lenovo asus nestle cadbury amul britannia parle maggi
starbucks mcdonalds dominos kfc subway pizza-hut haldiram patanjali dabur himalaya lakme loreal nivea dove colgate pepsi coca-cola coke
cult.fit cultfit gold's goldsgym anytime planet-fitness byjus unacademy coursera udemy linkedin canva notion slack zoom figma adobe
photoshop dropbox github gitlab atlassian jira salesforce hubspot zoho freshworks shopify wix godaddy tesla bmw audi mercedes toyota
honda hyundai maruti mahindra bajaj hero royal-enfield lic policybazaar groww zerodha upstox cred phonepay meesho myntra nykaa ajio
bigbasket blinkit zepto dunzo makemytrip goibibo irctc oyo airbnb booking.com
""".split())

_SLUG_BAD = re.compile(r"[^a-z0-9]+")


def slugify(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    s = _SLUG_BAD.sub("-", s).strip("-")
    return (s or "product")[:40].strip("-") or "product"


def looks_like_real_brand(*names: str) -> str | None:
    """The known brand a name matches, or None. Whole-word, case-insensitive."""
    for n in names:
        low = unicodedata.normalize("NFKC", n or "").lower()
        words = set(re.findall(r"[a-z0-9][a-z0-9.\-']*", low))
        words |= {w.strip(".-'") for w in words}
        words |= set(re.findall(r"[a-z0-9]+", low))
        for b in sorted(KNOWN_BRANDS):
            if b in words or (" " in b and b in low):
                return b
    return None


class Price(BaseModel):
    """What the customer is quoted. Numbers are in `currency` (default rupees). offer must be below list."""
    model_config = ConfigDict(extra="forbid")
    list_price: float | None = Field(None, ge=0)
    offer_price: float | None = Field(None, ge=0)
    currency: str = "₹"
    unit: str = ""          # what one price covers: "3 months", "per month"
    validity: str = ""      # "if you join this week" - only if the operator says so (never invented urgency)

    @model_validator(mode="after")
    def _order(self):
        if self.list_price is not None and self.offer_price is not None and self.offer_price >= self.list_price:
            raise ValueError("offer price must be below the list price")
        return self


class Economics(BaseModel):
    """Operator economics. The generator NEVER invents a floor: with none given, no negotiation is suggested."""
    model_config = ConfigDict(extra="forbid")
    floor: float | None = Field(None, ge=0)           # lowest price we may ever accept, or
    cost: float | None = Field(None, ge=0)            # cost + margin % (floor = cost * (1 + margin/100))
    min_margin_pct: float | None = Field(None, ge=0, le=1000)
    ladder: list[dict] = Field(default_factory=list)  # operator-written concession steps (used as given)
    perks: list[str] = Field(default_factory=list)    # value perks the operator allows ("one extra month free")

    @property
    def profit_floor(self) -> float | None:
        if self.floor is not None:
            return float(self.floor)
        if self.cost is not None and self.min_margin_pct is not None:
            return round(self.cost * (1 + self.min_margin_pct / 100.0), 2)
        return None


class Criteria(BaseModel):
    """What a version must satisfy. Every number is shown in the UI and saved with the version."""
    model_config = ConfigDict(extra="forbid")
    min_topics: int = Field(15, ge=1)                 # top-level topics
    max_topics: int = Field(30, ge=1)
    min_entries: int = Field(150, ge=1)               # Q&A entries
    min_offerings: int = Field(5, ge=0)
    max_offerings: int = Field(10, ge=0)
    min_situations: int = Field(14, ge=1)             # playbook situations
    max_situations: int = Field(20, ge=1)
    min_ladder_depth: int = Field(4, ge=1)            # discovery ladder rungs
    require_all_languages: bool = True                # every entry in en + hi + hinglish
    require_provenance: bool = True
    check_numeric_consistency: bool = True            # every price >= floor, offer < list
    check_banned_content: bool = True
    max_repair_rounds: int = Field(3, ge=0, le=5)

    @model_validator(mode="after")
    def _ranges(self):
        if self.min_topics > self.max_topics or self.min_offerings > self.max_offerings or self.min_situations > self.max_situations:
            raise ValueError("a minimum is above its maximum")
        return self


class Spec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(..., min_length=2, max_length=80)
    category: str = ""
    seller: str = ""
    website: str = ""
    notes: str = Field("", max_length=6000)           # description / website notes
    city: str = "India"
    languages: list[Literal["en", "hi", "hinglish"]] = Field(default_factory=lambda: list(LANGS))
    price: Price = Field(default_factory=Price)
    economics: Economics = Field(default_factory=Economics)
    tone: str = ""
    audience: str = ""
    compliance_notes: str = ""
    banned_phrases: list[str] = Field(default_factory=list)     # extra phrases the output may never contain
    authorised: bool = False                          # "I own / am authorised to sell this"
    real_brand: bool | None = None                    # None = decide from the name; True/False = operator override
    tenant: str = "default"
    product_id: str = ""
    criteria: Criteria = Field(default_factory=Criteria)

    @field_validator("name", "seller", "category", mode="before")
    @classmethod
    def _strip(cls, v):
        return v.strip() if isinstance(v, str) else v

    @field_validator("languages")
    @classmethod
    def _langs(cls, v):
        out = list(dict.fromkeys(["en", *v]))
        return out

    @field_validator("product_id")
    @classmethod
    def _pid(cls, v):
        if v and not re.match(r"^[a-z0-9][a-z0-9_\-]{0,39}$", v):
            raise ValueError("product_id: lower-case letters, digits, - and _")
        return v

    @field_validator("tenant")
    @classmethod
    def _tenant(cls, v):
        if not re.match(r"^[a-z0-9][a-z0-9_\-]{0,39}$", v or ""):
            raise ValueError("tenant: lower-case letters, digits, - and _")
        return v

    @model_validator(mode="after")
    def _floor_vs_price(self):
        f = self.economics.profit_floor
        o = self.price.offer_price if self.price.offer_price is not None else self.price.list_price
        if f is not None and o is not None and o < f:
            raise ValueError(f"the quoted price {o} is below the floor {f}")
        return self

    @property
    def pid(self) -> str:
        return self.product_id or slugify(self.name)


class Effective(BaseModel):
    """What the spec resolves to once the real-brand rule is applied."""
    product_id: str
    real_brand: bool
    brand_match: str | None
    allow_synthetic: bool
    provenance: Literal["public", "synthetic"]
    floor: float | None
    criteria: Criteria
    notes: list[str] = Field(default_factory=list)


def resolve(spec: Spec) -> Effective:
    """Apply the real-brand rule. A product that looks like a real company's, without the operator ticking
    "I own / am authorised to sell this", is public-knowledge only: provenance "public", allow_synthetic false."""
    match = looks_like_real_brand(spec.name, spec.seller)
    if spec.real_brand is True:
        real = True
    elif spec.real_brand is False:
        real = False
    else:
        real = bool(match) and not spec.authorised
    notes: list[str] = []
    crit = spec.criteria.model_copy()
    if real:
        notes.append("real-brand rule: only public-knowledge facts, provenance 'public', no synthetic detail; every claim is marked for review")
        # public facts are limited: asking for 150 Q&A would force invention, so the quantity targets relax (and say so)
        crit = crit.model_copy(update={"min_topics": min(crit.min_topics, 5), "min_entries": min(crit.min_entries, 30),
                                       "min_offerings": 0, "min_situations": min(crit.min_situations, 10)})
        notes.append("quantity criteria relaxed for a real brand: topics >= 5, Q&A >= 30, offerings >= 0, situations >= 10")
    return Effective(product_id=spec.pid, real_brand=real, brand_match=match, allow_synthetic=not real,
                     provenance="public" if real else "synthetic", floor=spec.economics.profit_floor, criteria=crit, notes=notes)
