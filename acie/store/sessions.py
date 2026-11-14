"""Session storage: where a call's SessionState lives.

`InMemorySessionStore` is today's behaviour (a dict). `RedisSessionStore` keeps a JSON copy with a TTL so a
process restart in the middle of a call can resume from the last finished turn. redis is imported lazily.
"""

from __future__ import annotations

from typing import Protocol

from ..schemas import SessionState

DEFAULT_TTL_S = 2 * 3600


class SessionStore(Protocol):
    async def get(self, session_id: str) -> SessionState | None: ...
    async def put(self, session: SessionState) -> None: ...
    async def delete(self, session_id: str) -> None: ...


class InMemorySessionStore:
    def __init__(self) -> None:
        self._data: dict[str, SessionState] = {}

    async def get(self, session_id: str) -> SessionState | None:
        s = self._data.get(session_id)
        return s.model_copy(deep=True) if s else None

    async def put(self, session: SessionState) -> None:
        self._data[session.session_id] = session.model_copy(deep=True)

    async def delete(self, session_id: str) -> None:
        self._data.pop(session_id, None)


class RedisSessionStore:
    def __init__(self, url: str = "redis://localhost:6379/0", ttl_s: int = DEFAULT_TTL_S, prefix: str = "acie:session:",
                 client=None) -> None:
        if client is None:
            import redis.asyncio as aioredis  # lazy: optional dependency
            client = aioredis.from_url(url, decode_responses=True)
        self.r, self.ttl_s, self.prefix = client, ttl_s, prefix

    async def get(self, session_id: str) -> SessionState | None:
        raw = await self.r.get(self.prefix + session_id)
        return SessionState.model_validate_json(raw) if raw else None

    async def put(self, session: SessionState) -> None:
        await self.r.set(self.prefix + session.session_id, session.model_dump_json(), ex=self.ttl_s)

    async def delete(self, session_id: str) -> None:
        await self.r.delete(self.prefix + session_id)

    async def close(self) -> None:
        await self.r.aclose()
