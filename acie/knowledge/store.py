"""Load / save / validate KB files: data/voice/knowledge/<tenant>/<product_id>.json (override the root with ACIE_KB_DIR)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .render import pget, slot_issues
from .schema import KnowledgeBase

ROOT = Path(__file__).resolve().parents[2]
_CACHE: dict[tuple[str, float], KnowledgeBase] = {}


def kb_dir() -> Path:
    return Path(os.getenv("ACIE_KB_DIR") or ROOT / "data" / "voice" / "knowledge")


def kb_path(tenant: str, product_id: str, base: Path | None = None) -> Path:
    return (base or kb_dir()) / (tenant or "default") / f"{product_id}.json"


def validate(kb: KnowledgeBase, product: Any = None) -> list[str]:
    """Problems that the schema alone cannot see: slots, prices vs the profit floor, provenance rules."""
    issues = slot_issues(kb)
    floor = pget(product, "profit_floor", None) if product is not None else None
    for o in kb.offerings:
        f = o.floor_inr
        if isinstance(floor, dict):
            f = floor.get(o.id, f)
        elif isinstance(floor, (int, float)) and not o.included:
            f = max(f or 0, int(floor))
        if o.price_inr is not None and f is not None and o.price_inr < f:
            issues.append(f"{o.id}: price {o.price_inr} below profit floor {f}")
    if not kb.allow_synthetic and any(e.provenance == "synthetic" for e in kb.entries):
        issues.append("synthetic entries in a KB with allow_synthetic=false")
    return issues


def save_kb(kb: KnowledgeBase, path: Path | None = None) -> Path:
    p = path or kb_path(kb.tenant, kb.product_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(kb.model_dump(mode="json"), ensure_ascii=False, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return p


def load_path(path: Path) -> KnowledgeBase:
    key = (str(path), path.stat().st_mtime)
    kb = _CACHE.get(key)
    if kb is None:
        kb = KnowledgeBase.model_validate_json(path.read_text(encoding="utf-8"))
        bad = slot_issues(kb)
        if bad:
            raise ValueError("; ".join(bad[:5]))
        _CACHE.clear()
        _CACHE[key] = kb
    return kb


def load_kb(product: Any, base: Path | None = None) -> KnowledgeBase | None:
    """The KB for a product (object or dict with id / tenant), or None when there is none."""
    pid = pget(product, "id", "") if not isinstance(product, str) else product
    tenant = pget(product, "tenant", "default") if not isinstance(product, str) else "default"
    if not pid:
        return None
    for t in dict.fromkeys([tenant or "default", "default"]):
        p = kb_path(t, pid, base)
        if p.exists():
            try:
                return load_path(p)
            except Exception:  # a broken KB file must never break a call
                return None
    return None


def kb_enabled() -> bool:
    return os.getenv("ACIE_KB", "0") == "1"
