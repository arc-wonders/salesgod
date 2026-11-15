"""The order being booked on this call: one serialisable record of the slots the caller has given (tier/zone/plan, quantity,
location/show, date), so a filled slot is never asked again and the close reads back the caller's own order with the chosen tier's
price and the total.

Live call c1d9da4eca (concert): the caller said VIP, five tickets and Delhi; the agent asked again for the city, the zone and the count,
listed the shows twice, read "Danny" (STT for Delhi) as an artist, and never read the order back or closed it.

Everything here is derived from the product's own data, so it works for any product:
  * tiers     <- the terms line: "₹1,999 for General Standing, ₹7,999 for Fan Pit, ..." ("₹N for NAME" / "NAME ₹N")
  * locations <- the discovery_ladder question whose id is city/location/branch/show/venue: "Delhi on 12 December, Mumbai on ..."
                 (venue + weekday from the knowledge sheet when it gives "Delhi, 12 December 2026, Jawaharlal Nehru Stadium")
  * quantity  <- asked only when the economics unit is countable per person (ticket, pass, seat, entry, person)
  * next steps <- the knowledge sentence that starts "After payment ..."
The state lives in session.dialog["_order"] (persisted with the session to NATS) and is mirrored on the checklist (`order`) for the
console. Standard library + pydantic only.
"""

from __future__ import annotations

import datetime as _dt
import difflib
import json
import re

from pydantic import BaseModel, Field

_I = re.IGNORECASE
_COUNTABLE = re.compile(r"\b(ticket|pass|seat|entry|entries|person|people|head|admission|spot|slot)s?\b", _I)
_LOC_IDS = ("city", "location", "branch", "show", "venue", "centre", "center", "area", "date")
_TIER_IDS = ("zone", "tier", "plan", "package", "variant", "category", "class")
_MONTHS = "january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
_NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
              "eleven": 11, "twelve": 12, "a couple of": 2, "couple of": 2, "pair of": 2,
              "ek": 1, "do": 2, "teen": 3, "char": 4, "chaar": 4, "paanch": 5, "panch": 5, "chhe": 6, "che": 6, "saat": 7,
              "aath": 8, "nau": 9, "das": 10,
              "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पाँच": 5, "पांच": 5, "छह": 6, "छः": 6, "सात": 7, "आठ": 8, "नौ": 9, "दस": 10}
# STT mishears units too: "5 pickets" (call c1d9da4eca)
_UNIT_NOUN = (r"(?:tickets?|tikets?|tickts?|pickets?|tackets?|kickets?|tix|passes|pass|seats?|entries|entry|people|persons?|members?|"
              r"memberships?|log|jan|janon|logon|of us|टिकट|टिकटें|लोग|लोगों|जन)")
_NUM_TOKEN = r"(\d{1,2}|" + "|".join(sorted((re.escape(k) for k in _NUM_WORDS), key=len, reverse=True)) + r")"
_QTY_UNIT = re.compile(r"(?<![\w₹,.:])" + _NUM_TOKEN + r"\s+(?:(?:[a-z]+|वीआईपी|गोल्ड)\s+){0,2}?" + _UNIT_NOUN + r"(?![\w])", _I)
_QTY_FOR = re.compile(r"\b(?:for|of|we are|we're|hum|हम)\s+" + _NUM_TOKEN + r"\s*(?:people|persons|of us|log|लोग|jan|members|adults)?\b(?!\s*(?:" + _MONTHS + r"|months?|weeks?|days?|years?|hours?|minutes?|mins?|mahine|hafte|din|saal|ghante|pm|am|baje|bje|%|k\b|thousand|hazar|hundred|rupees|rs\b))", _I)
_QTY_BARE = re.compile(r"^\W*(?:(?:ok(?:ay)?|yes|yeah|haan|ha|ji|so|um+|uh+|make it|just|only|sirf|total)\W+)*" + _NUM_TOKEN + r"(?:\s+(?:please|only|hi|bas|total))?\W*$", _I)
_PARTY = (("just me", 1), ("only me", 1), ("myself", 1), ("alone", 1), ("akela", 1), ("sirf main", 1), ("me and my (?:wife|husband|girlfriend|boyfriend|friend|partner|brother|sister|mother|father|mom|dad)", 2),
          ("with my (?:wife|husband|girlfriend|boyfriend|partner)", 2), ("couple", 2))
_PREMIUM = re.compile(r"\b(most expensive|costliest|priciest|the (?:best|top|premium|highest|luxury)\b(?! price| deal| offer| discount)|best (?:one|zone|seat|seats|option|experience|tier|plan|package)|"
                      r"top (?:one|zone|tier|option|category|plan|package)|premium (?:one|zone|option|tier|plan|package|seats?|tickets?)|luxur\w*|"
                      r"highest (?:one|zone|tier|category)|sabse (?:mehenga|mehnga|mahanga|accha|achha|best|badhiya|upar wala|premium))\b|सबसे (?:महंगा|महँगा|अच्छा|बढ़िया)", _I)
_CHEAPEST = re.compile(r"\b(cheapest|the lowest (?:one|price|zone|tier)|least expensive|most affordable|basic (?:one|zone|tier|ticket)|sabse (?:sasta|kam wala))\b|सबसे सस्ता", _I)
# common STT confusions for Indian city names (a word that is not the city but the line made it): applied only when the caller is
# talking about the booking (a booking word, "show", the order is open, or we just asked for the city)
_NEAR_MISS = {
    "delhi": ("danny", "daily", "deli", "delly", "dilly", "dehli", "dehly", "delhy", "deli's", "telly", "dilhi", "the lee", "the li", "dely", "denny", "delhie"),
    "mumbai": ("mumbay", "bombay", "mumba", "mum bye", "mom bye", "moon bye", "bambai", "mumbhai"),
    "chandigarh": ("chandigar", "chandi garh", "chandigardh", "chandigad", "chandighar", "chandi"),
    "bengaluru": ("bangalore", "banglore", "bengaluru", "bangaluru"),
    "kolkata": ("calcutta", "kolkatta", "kolkota"),
    "pune": ("poona", "puna"),
    "hyderabad": ("hydrabad", "hyderbad"),
    "chennai": ("madras", "chenai"),
}
_CITY_ALIASES = {
    "delhi": ("dilli", "दिल्ली", "new delhi", "ncr", "gurgaon", "gurugram", "noida", "नई दिल्ली"),
    "mumbai": ("मुंबई", "मुम्बई", "navi mumbai", "बॉम्बे", "bombay", "thane"),
    "chandigarh": ("चंडीगढ़", "mohali", "मोहाली", "panchkula", "पंचकूला"),
    "bengaluru": ("बेंगलुरु", "bangalore"),
    "kolkata": ("कोलकाता", "calcutta"),
    "pune": ("पुणे",),
    "hyderabad": ("हैदराबाद",),
    "chennai": ("चेन्नई",),
    "jaipur": ("जयपुर",),
    "ahmedabad": ("अहमदाबाद",),
    "lucknow": ("लखनऊ",),
}
_BOOK_CTX = re.compile(r"\b(book|booking|reserve|show|concert|city|tickets?|pickets?|order|note (it|that)|for me|chahiye|karna|kar do)\b|बुक|शो|टिकट", _I)
_NEG_CITY = re.compile(r"\b(not|no|nahi|nahin|na)\s+(?:the\s+)?$", _I)
_HI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
_GENERIC_TIER_WORDS = {"the", "for", "and", "per", "ticket", "tickets", "zone", "lounge", "standing", "plan", "pass", "seat", "month", "months",
                       "week", "year", "years", "day", "days"}


class Tier(BaseModel):
    name: str
    price: float
    aliases: list[str] = Field(default_factory=list)
    note: str = ""     # "if you book this week instead of ₹4,999"
    about: str = ""    # the knowledge sheet's description of it


class Location(BaseModel):
    name: str
    date: str = ""     # "12 December"
    weekday: str = ""  # "Saturday"
    venue: str = ""


class Catalog(BaseModel):
    tiers: list[Tier] = Field(default_factory=list)
    locations: list[Location] = Field(default_factory=list)
    unit: str = ""
    countable: bool = False
    currency: str = "₹"
    tier_noun: str = "option"
    max_qty: int = 0
    next_steps: str = ""
    location_question: dict[str, str] = Field(default_factory=dict)

    @property
    def slots(self) -> list[str]:
        """The slots this product needs before a booking can be read back, in the order they are asked."""
        out = []
        if len(self.tiers) > 1:
            out.append("tier")
        if self.countable:
            out.append("quantity")
        if self.locations:
            out.append("location")
        return out

    def tier(self, name: str) -> Tier | None:
        return next((t for t in self.tiers if t.name == name), None)

    def location(self, name: str) -> Location | None:
        return next((l for l in self.locations if l.name == name), None)

    @property
    def unit_plural(self) -> str:
        u = (self.unit or "").strip()
        if not u or not self.countable:
            return ""
        return u if u.endswith("s") else (u[:-1] + "ies" if u.endswith("y") else u + ("es" if u.endswith(("s", "sh", "ch")) else "s"))


class OrderState(BaseModel):
    tier: str = ""
    tier_price: float = 0.0
    quantity: int = 0
    location: str = ""
    date: str = ""
    active: bool = False            # the caller is booking (a booking word, a want with a slot, or an answer to a slot question)
    asked: str = ""                 # the slot asked last turn
    asked_n: dict[str, int] = Field(default_factory=dict)
    status: str = ""                # "" | summarised | link | desk
    summary: str = ""
    total: float = 0.0
    filled_turn: dict[str, int] = Field(default_factory=dict)

    def missing(self, cat: Catalog) -> list[str]:
        return [s for s in cat.slots if not self.has(s)]

    def has(self, slot: str) -> bool:
        return bool({"tier": self.tier, "quantity": self.quantity, "location": self.location}.get(slot))

    def complete(self, cat: Catalog) -> bool:
        return bool(cat.slots) and not self.missing(cat)

    def public(self) -> dict:
        d = {k: v for k, v in self.model_dump().items() if k in ("tier", "tier_price", "quantity", "location", "date", "status", "total", "summary") and v}
        return d


def _money(s: str) -> float:
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return 0.0


def fmt(v: float, cur: str = "₹") -> str:
    return f"{cur}{int(round(v)):,}"


def _aliases(name: str) -> list[str]:
    low = name.lower().strip()
    out = [low]
    words = [w for w in re.findall(r"[a-z0-9]+", low) if w not in _GENERIC_TIER_WORDS and len(w) >= 2]
    if len(re.findall(r"[a-z0-9]+", low)) > 1:
        out += [w for w in words if len(w) >= 3 or w in ("ga",)]
    if low.startswith("vip"):
        out += ["vip", "v.i.p", "v i p", "वीआईपी", "वी आई पी"]
    if "general" in low:
        out += ["general", "जनरल", "ga"]
    if "gold" in low:
        out += ["गोल्ड"]
    if "pit" in low:
        out += ["pit", "फैन पिट", "फ़ैन पिट"]
    seen, res = set(), []
    for a in out:
        if a and a not in seen:
            seen.add(a)
            res.append(a)
    return res


_TIER_A = re.compile(r"(?:₹|rs\.?\s*|inr\s*|\$)\s*(\d{1,3}(?:,\d{2,3})+|\d{2,7})\s+(?:for|per)\s+(?:a\s+|one\s+|the\s+)?([A-Z][\w]*(?:\s+[A-Z][\w]*)*|\d+\s+\w+)"
                     r"(?P<note>(?:[^,;]|,(?=\d))*)", 0)
_TIER_B = re.compile(r"([A-Z][\w]*(?:\s+[A-Z][\w]*)*)\s+(?:is\s+|at\s+)?(?:₹|rs\.?\s*|\$)\s*(\d{1,3}(?:,\d{2,3})+|\d{2,7})")


def _parse_tiers(terms: str, knowledge: str) -> list[Tier]:
    tiers: list[Tier] = []
    for m in _TIER_A.finditer(terms or ""):
        name = m.group(2).strip()
        if not name or any(t.name.lower() == name.lower() for t in tiers):
            continue
        note = re.sub(r"^\W+|\W+$", "", (m.group("note") or "").split(" and ")[0]).strip()
        note = note if re.search(r"\b(if|instead|till|until|this week|today|valid)\b", note, _I) else ""
        tiers.append(Tier(name=name, price=_money(m.group(1)), note=note))
    if not tiers:
        for m in _TIER_B.finditer(terms or ""):
            name = m.group(1).strip()
            if name and not any(t.name.lower() == name.lower() for t in tiers):
                tiers.append(Tier(name=name, price=_money(m.group(2))))
    for t in tiers:
        t.aliases = _aliases(t.name)
        km = re.search(re.escape(t.name) + r"[^(;]{0,80}\(([^)]{10,400})\)", knowledge or "")
        if km:
            about = re.split(r";\s*about \d+% left|;", km.group(1))[0].strip()
            t.about = about
    return tiers


_LOC_PAIR = re.compile(r"([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)\s+(?:on\s+)?(\d{1,2}\s+(?i:" + _MONTHS + r")\b|(?i:" + _MONTHS + r")\s+\d{1,2}\b)")


def _parse_locations(ladder: list[dict], knowledge: str) -> tuple[list[Location], dict[str, str]]:
    for step in ladder or []:
        if str(step.get("id", "")).lower() not in _LOC_IDS:
            continue
        q = step.get("question") or {}
        en = q.get("en", "") if isinstance(q, dict) else str(q)
        locs: list[Location] = []
        for m in _LOC_PAIR.finditer(en):
            name = m.group(1).strip()
            name = re.sub(r"^(?:Which|Would|Or|And|Come|To)\s+", "", name, flags=_I).strip()
            if not name or name.lower() in ("or", "and") or not name[0].isupper():
                continue
            date = m.group(2).strip()
            loc = Location(name=name, date=date)
            km = re.search(re.escape(name) + r",\s*(?:\w+day,?\s*)?(\d{1,2}\s+(?:" + _MONTHS + r")\s+(\d{4}))\s*,\s*([^,;]+)", knowledge or "", _I)
            if km:
                loc.venue = km.group(3).strip()
                try:
                    loc.weekday = _dt.datetime.strptime(km.group(1), "%d %B %Y").strftime("%A")
                except ValueError:
                    pass
            locs.append(loc)
        if not locs:  # a plain list: "Which branch: Andheri, Bandra or Powai?"
            tail = en.split(":", 1)[1] if ":" in en else ""
            for nm in re.split(r",\s*|\s+or\s+|\s+and\s+", tail.strip(" ?.")):
                nm = nm.strip()
                if nm and nm[0].isupper() and len(nm) <= 30:
                    locs.append(Location(name=nm))
        if locs:
            return locs, (q if isinstance(q, dict) else {"en": en})
    return [], {}


def catalog(product) -> Catalog:
    """The bookable options of a product, read from its data (cached on the product object)."""
    cached = getattr(product, "__dict__", {}).get("_order_catalog")
    if cached is not None:
        return cached
    terms = (getattr(product, "terms", None) or {}).get("en", "") or ""
    knowledge = getattr(product, "knowledge", "") or ""
    eco = getattr(product, "economics", None)
    unit = (getattr(eco, "unit", "") or "") if eco is not None else ""
    cur = (getattr(eco, "currency", "") or "₹") if eco is not None else "₹"
    tiers = _parse_tiers(terms, knowledge)
    ladder = getattr(product, "discovery_ladder", None) or []
    locs, lq = _parse_locations(ladder, knowledge)
    noun = next((str(s.get("id")) for s in ladder if str(s.get("id", "")).lower() in _TIER_IDS), "option")
    mx = re.search(r"\bmax(?:imum)?\.?\s+(\d{1,2})\s+\w+\s+per\s+order", knowledge, _I)
    nxt = re.search(r"(After (?:payment|paying|you pay)[^.;]*)[.;]", knowledge)
    cat = Catalog(tiers=tiers, locations=locs, unit=unit, countable=bool(_COUNTABLE.search(unit or "")), currency=cur,
                  tier_noun=noun, max_qty=int(mx.group(1)) if mx else 0, next_steps=(re.sub(r"\bthe customer\b", "you", re.sub(r"\bthe customer's\b", "your", nxt.group(1).strip())) + ".") if nxt else "",
                  location_question=lq)
    try:
        object.__setattr__(product, "_order_catalog", cat) if hasattr(product, "__dict__") else None
        product.__dict__["_order_catalog"] = cat
    except Exception:
        pass
    return cat


# ---------------------------------------------------------------------- reading slots from what the caller said

def _num(tok: str) -> int:
    tok = tok.lower().strip().translate(_HI_DIGITS)
    if tok.isdigit():
        return int(tok)
    return _NUM_WORDS.get(tok, 0)


def parse_quantity(text: str, asked: bool = False) -> int:
    """How many (tickets / people / memberships) the caller wants; 0 = not said. Dates, times and prices are never a quantity."""
    t = (text or "").translate(_HI_DIGITS)
    # the count with an item noun ("3 Gold tickets") beats a head count ("do logon") said in the same turn; the last one wins
    hits = [(bool(re.search(r"tick|pick|tix|pass|seat|entr|member|टिकट", m.group(0), _I)), m.start(), _num(m.group(1))) for m in _QTY_UNIT.finditer(t)]
    hits = [h for h in hits if 0 < h[2] <= 50]
    if hits:
        return sorted(hits)[-1][2]
    for m in _QTY_FOR.finditer(t):
        n = _num(m.group(1))
        if 0 < n <= 20:
            return n
    if asked:
        m = _QTY_BARE.match(t)
        if m and 0 < _num(m.group(1)) <= 50:
            return _num(m.group(1))
    low = t.lower()
    for pat, n in _PARTY:
        if re.search(r"\b" + pat + r"\b", low):
            return n
    return 0


def parse_tier(text: str, cat: Catalog) -> str:
    if not cat.tiers:
        return ""
    low = (text or "").lower()
    if _PREMIUM.search(text or ""):
        return max(cat.tiers, key=lambda t: t.price).name
    if _CHEAPEST.search(text or ""):
        return min(cat.tiers, key=lambda t: t.price).name
    best = ("", 0)
    for t in cat.tiers:
        for a in t.aliases:
            if re.search(r"(?<![\w])" + re.escape(a) + r"(?![\w])", low) and len(a) > best[1]:
                best = (t.name, len(a))
    return best[0]


def parse_location(text: str, cat: Catalog, booking_ctx: bool = False) -> str:
    """The location the caller chose. A near-miss of a city name ("Danny", "daily" for Delhi) counts only while the caller is
    talking about the booking; "not Delhi" never fills it."""
    if not cat.locations:
        return ""
    low = " " + re.sub(r"[^\w\s']", " ", (text or "").lower()) + " "
    hits: list[tuple[int, str]] = []
    for loc in cat.locations:
        key = loc.name.lower()
        names = [key] + list(_CITY_ALIASES.get(key, ()))
        for nm in names:
            for m in re.finditer(r"(?<![\w])" + re.escape(nm) + r"(?![\w])", low):
                if not _NEG_CITY.search(low[:m.start()]):
                    hits.append((m.start(), loc.name))
    if hits:
        return sorted(hits)[-1][1] if len({h[1] for h in hits}) > 1 and re.search(r"\b(actually|instead|change|rather|make it)\b", low) else \
            max(set(h[1] for h in hits), key=lambda n: sum(1 for h in hits if h[1] == n))
    if not booking_ctx:
        return ""
    for loc in cat.locations:
        key = loc.name.lower()
        for nm in _NEAR_MISS.get(key, ()):
            if re.search(r"(?<![\w])" + re.escape(nm) + r"(?![\w])", low):
                return loc.name
    words = re.findall(r"[a-z]{4,}", low)
    for loc in cat.locations:
        key = loc.name.lower()
        if len(key) >= 5 and any(difflib.SequenceMatcher(None, w, key).ratio() >= 0.8 for w in words):
            return loc.name
    return ""


def heard_city_near_miss(text: str, cat: Catalog) -> bool:
    low = (text or "").lower()
    return any(re.search(r"(?<![\w])" + re.escape(nm) + r"(?![\w])", low) for loc in cat.locations for nm in _NEAR_MISS.get(loc.name.lower(), ()))


def premium_ask(text: str) -> bool:
    return bool(_PREMIUM.search(text or ""))


# ---------------------------------------------------------------------- persistence

def load(dialog: dict) -> OrderState:
    raw = (dialog or {}).get("_order")
    if not raw:
        return OrderState()
    try:
        return OrderState.model_validate(json.loads(raw) if isinstance(raw, str) else raw)
    except Exception:
        return OrderState()


def save(order: OrderState, dialog: dict, ctx=None) -> None:
    dialog["_order"] = order.model_dump_json()
    ck = getattr(ctx, "checklist", None)
    if ck is not None and hasattr(ck, "order"):
        ck.order = order.public()


def fill(order: OrderState, text: str, cat: Catalog, turn: int = 0, extra: str = "") -> list[str]:
    """Fold one caller turn into the order; -> the slots that changed. `extra`: the NLU's gist of the turn (used only for slots the
    words themselves did not give)."""
    changed: list[str] = []
    booking_ctx = order.active or order.asked == "location" or bool(_BOOK_CTX.search(text or ""))
    for src in (text, extra):
        if not src:
            continue
        tr = parse_tier(src, cat)
        if tr and tr != order.tier and ("tier" not in changed):
            if src is text or not order.tier:
                order.tier = tr
                t = cat.tier(tr)
                order.tier_price = t.price if t else 0.0
                changed.append("tier")
        q = parse_quantity(src, asked=(order.asked == "quantity" and src is text))
        if q and q != order.quantity and "quantity" not in changed and (src is text or not order.quantity):
            order.quantity = q
            changed.append("quantity")
        loc = parse_location(src, cat, booking_ctx=booking_ctx and src is text)
        if loc and loc != order.location and "location" not in changed and (src is text or not order.location):
            order.location = loc
            l = cat.location(loc)
            order.date = l.date if l else ""
            changed.append("location")
    if len(cat.tiers) == 1 and not order.tier:
        order.tier, order.tier_price = cat.tiers[0].name, cat.tiers[0].price
    for s in changed:
        order.filled_turn[s] = turn
    if changed and order.status == "summarised":
        order.status = ""  # a change after the read-back: read it back again
    return changed


# ---------------------------------------------------------------------- wording

def describe(order: OrderState, cat: Catalog, product_name: str = "", with_price: bool = True) -> str:
    """'5 VIP Lounge tickets for Delhi, Saturday 12 December, ₹14,999 each, ₹74,995 total'"""
    q = order.quantity or 1
    parts = []
    unit = cat.unit_plural if (q > 1 or not order.quantity) else (cat.unit if cat.countable else "")
    head = f"{q} " if (cat.countable and order.quantity) or q > 1 else ""
    tier = order.tier if len(cat.tiers) > 1 else ""
    noun = unit or (product_name if not tier else "")
    head += " ".join(x for x in (tier, noun) if x).strip() or product_name
    parts.append(head.strip())
    if order.location:
        loc = cat.location(order.location)
        when = " ".join(x for x in ((loc.weekday if loc else ""), order.date) if x)
        parts[0] += f" for {order.location}" + (f", {when}" if when else "")
    if with_price and order.tier_price:
        each = fmt(order.tier_price, cat.currency)
        if q > 1:
            order.total = order.tier_price * q
            parts.append(f"{each} each, {fmt(order.total, cat.currency)} total")
        else:
            order.total = order.tier_price
            parts.append(each)
    return ", ".join(parts)


def slot_question(slot: str, cat: Catalog, n: int = 0, lang: str = "en") -> dict[str, str]:
    """One question for one missing slot (the second time, a shorter one)."""
    cur = cat.currency
    if slot == "tier":
        opts = sorted(cat.tiers, key=lambda t: t.price)
        if n == 0:
            lst = [f"{t.name} at {fmt(t.price, cur)}" for t in opts]
            en = f"Which {cat.tier_noun} would you like: " + ", ".join(lst[:-1]) + f", or {lst[-1]}?"
        else:
            en = f"Which {cat.tier_noun} shall I note: " + ", ".join(t.name for t in opts[:-1]) + f" or {opts[-1].name}?"
        hi_l = ", ".join(f"{t.name} {fmt(t.price, cur)}" for t in opts)
        return {"en": en, "hi": f"कौन सा {cat.tier_noun} लेंगे: {hi_l}?", "hinglish": f"Kaun sa {cat.tier_noun} lenge: {hi_l}?"}
    if slot == "quantity":
        u = cat.unit_plural or "tickets"
        return {"en": f"How many {u} would you like?" if n == 0 else f"And how many {u} shall I note?",
                "hi": "कितने टिकट चाहिए?" if "ticket" in u else "कितने चाहिए?",
                "hinglish": "Kitne tickets chahiye?" if "ticket" in u else "Kitne chahiye?"}
    if slot == "location":
        q = cat.location_question or {}
        if n == 0 and q.get("en"):
            return {"en": q.get("en", ""), "hi": q.get("hi", q.get("en", "")), "hinglish": q.get("hinglish", q.get("en", ""))}
        names = [l.name for l in cat.locations]
        lst = ", ".join(names[:-1]) + f" or {names[-1]}" if len(names) > 1 else names[0]
        return {"en": f"Which city works for you: {lst}?", "hi": f"कौन सा शहर ठीक रहेगा: {lst}?", "hinglish": f"Kaun sa city theek rahega: {lst}?"}
    return {"en": "", "hi": "", "hinglish": ""}


def stt_vocabulary(product) -> list[str]:
    """Names Whisper should expect on this call: the options and places of the order (voice/call.py adds them to its prompt)."""
    try:
        cat = catalog(product)
    except Exception:
        return []
    out = [t.name for t in cat.tiers if not re.match(r"^\d", t.name)] + [l.name for l in cat.locations]
    return list(dict.fromkeys(out))
