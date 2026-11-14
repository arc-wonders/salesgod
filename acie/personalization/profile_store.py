"""Long-term customer preference storage (P_t).

`JsonProfileStore` is the dev backend (one file per customer). A Postgres +
pgvector backend implements the same two async methods later.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Protocol

from ..schemas import CustomerProfile, PreferenceField, _now


class ProfileStore(Protocol):
    async def load(self, customer_id: str) -> CustomerProfile: ...
    async def save(self, profile: CustomerProfile) -> None: ...


class MemoryProfileStore:
    def __init__(self) -> None:
        self._data: dict[str, CustomerProfile] = {}

    async def load(self, customer_id: str) -> CustomerProfile:
        return self._data.get(customer_id, CustomerProfile(customer_id=customer_id)).model_copy(deep=True)

    async def save(self, profile: CustomerProfile) -> None:
        self._data[profile.customer_id] = profile.model_copy(deep=True)


class JsonProfileStore:
    def __init__(self, root: str | Path = "data/profiles") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, customer_id: str) -> Path:
        safe = "".join(c for c in customer_id if c.isalnum() or c in "-_")
        return self.root / f"{safe}.json"

    async def load(self, customer_id: str) -> CustomerProfile:
        p = self._path(customer_id)
        if not p.exists():
            return CustomerProfile(customer_id=customer_id)
        raw = await asyncio.to_thread(p.read_text)
        return CustomerProfile.model_validate_json(raw)

    async def save(self, profile: CustomerProfile) -> None:
        await asyncio.to_thread(self._path(profile.customer_id).write_text, profile.model_dump_json(indent=2))


def update_preference(profile: CustomerProfile, key: str, value: str, *, explicit: bool = False,
                      lr: float = 0.2) -> None:
    """Confidence-weighted preference update.

    - explicit statements set confidence 1.0 and lock the field until another explicit statement
    - implicit evidence for the same value raises confidence; for a different value lowers it,
      and the value flips only once confidence drops below 0.3
    """
    cur = profile.preferences.get(key)
    if explicit:
        profile.preferences[key] = PreferenceField(value=value, confidence=1.0, explicit=True)
        return
    if cur is None:
        profile.preferences[key] = PreferenceField(value=value, confidence=lr)
        return
    if cur.explicit:
        return
    if cur.value == value:
        cur.confidence = round(min(1.0, cur.confidence + lr * (1 - cur.confidence)), 3)
    else:
        cur.confidence = round(max(0.0, cur.confidence - lr), 3)
        if cur.confidence < 0.3:
            cur.value, cur.confidence = value, lr
    cur.updated_at = _now()
