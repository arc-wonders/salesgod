"""Postgres profile store (asyncpg). Same two methods as ProfileStore; the table is created on first use."""

from __future__ import annotations

import asyncio

from ..schemas import CustomerProfile

SCHEMA = """
CREATE TABLE IF NOT EXISTS customer_profiles (
    customer_id text PRIMARY KEY,
    profile     jsonb NOT NULL,
    updated_at  timestamptz NOT NULL DEFAULT now()
)"""


class PostgresProfileStore:
    def __init__(self, dsn: str, pool=None, min_size: int = 1, max_size: int = 5) -> None:
        self.dsn, self._pool, self._min, self._max = dsn, pool, min_size, max_size
        self._lock = asyncio.Lock()

    async def _get_pool(self):
        if self._pool is None:
            async with self._lock:
                if self._pool is None:
                    import asyncpg  # lazy: optional dependency
                    pool = await asyncpg.create_pool(self.dsn, min_size=self._min, max_size=self._max)
                    async with pool.acquire() as c:
                        await c.execute(SCHEMA)  # auto-migrate
                    self._pool = pool
        return self._pool

    async def load(self, customer_id: str) -> CustomerProfile:
        pool = await self._get_pool()
        async with pool.acquire() as c:
            raw = await c.fetchval("SELECT profile::text FROM customer_profiles WHERE customer_id=$1", customer_id)
        return CustomerProfile.model_validate_json(raw) if raw else CustomerProfile(customer_id=customer_id)

    async def save(self, profile: CustomerProfile) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as c:
            await c.execute(
                "INSERT INTO customer_profiles(customer_id, profile, updated_at) VALUES($1, $2::jsonb, now()) "
                "ON CONFLICT (customer_id) DO UPDATE SET profile=EXCLUDED.profile, updated_at=now()",
                profile.customer_id, profile.model_dump_json())

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
