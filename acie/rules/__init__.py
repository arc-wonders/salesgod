"""Domain rule packs for the validator (Layer 7). A product/plan picks packs via TurnPlan.rule_packs."""

from __future__ import annotations

from .banking import BankingPack
from .base import RulePack
from .hindi_register import HindiRegisterPack
from .sales_compliance import SalesCompliancePack

REGISTRY: dict[str, type] = {
    "sales_compliance": SalesCompliancePack,
    "banking": BankingPack,
    "hindi_register": HindiRegisterPack,
}


def load_packs(names) -> list[RulePack]:
    """Names (or ready-made pack objects) -> packs. Unknown names are ignored."""
    out: list[RulePack] = []
    for n in names or []:
        if isinstance(n, str):
            cls = REGISTRY.get(n)
            if cls:
                out.append(cls())
        elif hasattr(n, "check"):
            out.append(n)
    return out


__all__ = ["RulePack", "REGISTRY", "load_packs", "SalesCompliancePack", "BankingPack", "HindiRegisterPack"]
