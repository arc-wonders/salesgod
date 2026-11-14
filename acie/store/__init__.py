"""Store factory: pick backends from the environment. Defaults are today's behaviour (nothing external)."""

from __future__ import annotations

import os
from pathlib import Path

from .sessions import InMemorySessionStore, RedisSessionStore, SessionStore

__all__ = ["build_stores_from_env", "SessionStore", "InMemorySessionStore", "RedisSessionStore"]


def build_stores_from_env(env=None, profile_dir: str | Path = "data/profiles", embedder=None, embed_dim: int | None = None):
    """-> (session_store, profile_store, memory_store | None, sinks).

    ACIE_SESSION_STORE=redis (ACIE_REDIS_URL) | ACIE_PROFILE_STORE=postgres (ACIE_PG_DSN) |
    ACIE_MEMORY_STORE=pgvector (ACIE_PG_DSN, ACIE_MEMORY_DIM) | ACIE_NATS_URL set -> NatsSink in `sinks`.
    `sinks` holds root sinks; per call use `sink.for_call(call_id)` when it has that method.
    """
    env = os.environ if env is None else env
    from ..personalization.profile_store import JsonProfileStore

    if env.get("ACIE_SESSION_STORE", "memory") == "redis":
        sessions = RedisSessionStore(env.get("ACIE_REDIS_URL", "redis://localhost:6379/0"))
    else:
        sessions = InMemorySessionStore()

    if env.get("ACIE_PROFILE_STORE", "json") == "postgres":
        from ..personalization.pg_store import PostgresProfileStore
        profiles = PostgresProfileStore(env["ACIE_PG_DSN"])
    else:
        profiles = JsonProfileStore(profile_dir)

    memory = None
    if env.get("ACIE_MEMORY_STORE", "") == "pgvector":
        from .pgvector_memory import PgVectorMemoryStore
        if embedder is None:
            try:
                from ..memory.base import HashingEmbedder  # feedback lane's dependency-free fallback
                embedder = HashingEmbedder()
            except Exception as e:
                raise RuntimeError("ACIE_MEMORY_STORE=pgvector needs an embedder") from e
        dim = embed_dim or int(env.get("ACIE_MEMORY_DIM", "256"))
        memory = PgVectorMemoryStore(env["ACIE_PG_DSN"], embedder, dim)

    sinks: list = []
    if env.get("ACIE_NATS_URL"):
        from ..events import NatsSink
        sinks.append(NatsSink(env["ACIE_NATS_URL"]))
    return sessions, profiles, memory, sinks
