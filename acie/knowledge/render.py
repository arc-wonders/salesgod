"""Fill answer slots from the product config and the offering table. Prices only ever come from here."""

from __future__ import annotations

import re
from typing import Any

from .schema import Entry, KnowledgeBase, Offering

SLOT = re.compile(r"\{([a-z_]+)(?::([a-z0-9_\-]+))?\}")
PRODUCT_SLOTS = ("offer_price", "original_price", "discount", "offer_validity", "product_name", "seller_name", "call_to_action", "terms",
                 "agent_name")


def pget(product: Any, key: str, default: Any = "") -> Any:
    if product is None:
        return default
    if isinstance(product, dict):
        return product.get(key, default)
    return getattr(product, key, default)


def product_slot(product: Any, name: str, lang: str) -> str:
    v = pget(product, name, "")
    if isinstance(v, dict):
        return str(v.get(lang) or v.get("en") or "")
    return str(v or "")


def offering_price(o: Offering, lang: str) -> str:
    return o.price_label.get(lang) if o.price_label else ""


def render_text(text: str, lang: str, kb: KnowledgeBase, product: Any = None) -> str:
    offs = kb.offering_map()

    def sub(m: re.Match) -> str:
        name, arg = m.group(1), m.group(2)
        if name == "price" and arg in offs:
            return offering_price(offs[arg], lang)
        if name == "name" and arg in offs:
            return offs[arg].name.get(lang)
        if arg is None and name in PRODUCT_SLOTS:
            return product_slot(product, name, lang)
        return m.group(0)

    return SLOT.sub(sub, text)


def render_answer(e: Entry, lang: str, kb: KnowledgeBase, product: Any = None) -> str:
    return render_text(e.answer.get(lang), lang, kb, product).strip()


def slot_issues(kb: KnowledgeBase) -> list[str]:
    """Slots in any answer or pitch that nothing can fill."""
    offs = kb.offering_map()
    bad: list[str] = []

    def check(where: str, text: str) -> None:
        for m in SLOT.finditer(text):
            name, arg = m.group(1), m.group(2)
            ok = (name in ("price", "name") and arg in offs and (name != "price" or offs[arg].price_label is not None)) or \
                 (arg is None and name in PRODUCT_SLOTS)
            if not ok:
                bad.append(f"{where}: unfillable slot {m.group(0)}")

    for e in kb.entries:
        for t in e.answer.all():
            check(e.id, t)
    for o in kb.offerings:
        for t in o.pitch.all():
            check(o.id, t)
    return bad
