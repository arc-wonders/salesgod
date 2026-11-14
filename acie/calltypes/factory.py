"""Pick the domain pack for a call: `build_domain(call_type_id, product, ...)`. Unknown / broken call types and `sales_pitch`
give today's ProductSalesDomain, so a bad configuration never breaks a call."""

from __future__ import annotations

import logging
from pathlib import Path

from .registry import CallTypeRegistry

log = logging.getLogger("acie.calltypes")


def build_domain(call_type_id: str | None, product, *, data_dir: Path | str | None = None, project_id: str = "", customer_name: str = "",
                 customer_age: int | None = None, world=None, playbook=None, customer_id: str = "", phone: str = "", call_id: str = "",
                 registry: CallTypeRegistry | None = None, now_fn=None, tz: str = "Asia/Kolkata", **kw):
    from ..decision.domains.product_sales import ProductSalesDomain
    sales_kw = dict(customer_name=customer_name, world=world, customer_age=customer_age, playbook=playbook, **kw)
    ct = None
    if call_type_id and call_type_id != "sales_pitch":
        reg = registry or CallTypeRegistry(data_dir)
        ct = reg.get(call_type_id, project_id)
        if ct is None:
            log.warning("unknown call type %r: using the sales pitch", call_type_id)
    if ct is None or ct.pack == "sales_pitch":
        return ProductSalesDomain(product, now_fn=now_fn, **sales_kw)
    common = dict(call_type=ct, product=product, now_fn=now_fn, tz=tz, **sales_kw)
    if ct.pack == "lead_qualification":
        from ..decision.domains.lead_qualification import LeadQualificationDomain
        return LeadQualificationDomain(**common)
    if ct.pack == "follow_up":
        from ..decision.domains.follow_up import FollowUpDomain
        return FollowUpDomain(**common, data_dir=data_dir, customer_id=customer_id, phone=phone, call_id=call_id)
    from ..decision.domains.custom_steps import StepCallDomain
    return StepCallDomain(**common)
