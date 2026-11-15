"""Durable call context on NATS JetStream: the agent never loses where the conversation is.

Per call (session id `call-<call_id>`):
  stream  SG_CALLS, subject `sg.call.<call_id>.event`  one event per turn (customer text, agent text, plan intent,
          decision, checklist delta, deal view, language, timestamps) plus the full session snapshot, so the call
          can also be rebuilt by replaying the stream; ledger events go to `sg.call.<call_id>.ctx`.
  KV      bucket `sg_session`, key = session id: the latest full SessionState (context ledger incl. `deal` and the
          checklist, dialog keys, history), written after every turn.

Rules: a turn is never blocked. Writes are queued and sent by one background task (per-key coalescing: only the
latest snapshot matters); reads have a tight timeout (default 50 ms) and fall back to the in-process copy. If NATS is
down the store keeps working from memory and logs it (once a minute). nats-py is imported lazily.
"""

from __future__ import annotations

import asyncio
import collections
import json
import logging
import re
import time

from ..schemas import SessionState

log = logging.getLogger("acie.store.nats")

STREAM = "SG_CALLS"
SUBJECT_PREFIX = "sg.call"
BUCKET = "sg_session"
_SAFE = re.compile(r"[^A-Za-z0-9_-]")


def safe_token(s: str) -> str:
    return _SAFE.sub("_", s or "unknown")[:120] or "unknown"


def call_token(session_id: str) -> str:
    return safe_token(session_id[5:] if session_id.startswith("call-") else session_id)


class NatsSessionStore:
    """SessionStore (get/put/delete) backed by JetStream KV + a per-call event stream, with an in-process mirror."""

    def __init__(self, url: str = "nats://127.0.0.1:4222", *, bucket: str = BUCKET, stream: str = STREAM, prefix: str = SUBJECT_PREFIX,
                 read_timeout_s: float = 0.05, write_timeout_s: float = 0.5, max_age_s: float = 3 * 86400,
                 local_max: int = 256, max_queue: int = 2000, client=None) -> None:
        self.url, self.bucket_name, self.stream_name, self.prefix = url, bucket, stream, prefix
        self.read_timeout_s, self.write_timeout_s, self.max_age_s = read_timeout_s, write_timeout_s, max_age_s
        self.local: collections.OrderedDict[str, str] = collections.OrderedDict()  # session id -> latest snapshot json
        self.local_max, self.max_queue = local_max, max_queue
        self._nc, self._js, self._kv = client, None, None
        self._injected = client is not None
        self._kv_pending: dict[str, bytes] = {}  # key -> latest snapshot not yet in KV (coalesced)
        self._events: collections.deque = collections.deque()  # (subject, bytes)
        self._task: asyncio.Task | None = None
        self._wake: asyncio.Event | None = None
        self._ready = False
        self._last_warn = 0.0
        self.stats = {"kv_put": 0, "events": 0, "dropped": 0, "errors": 0, "restored": 0, "restore_miss": 0,
                      "restore_timeout": 0, "last_error": ""}

    # ------------------------------------------------------------------ connection

    @property
    def connected(self) -> bool:
        nc = self._nc
        return bool(self._ready and nc is not None and getattr(nc, "is_connected", True))

    async def start(self) -> None:
        """Connect and make sure the stream and the KV bucket exist. Never raises (NATS down = memory only)."""
        try:
            await asyncio.wait_for(self._ensure(), 3.0)
        except Exception as e:
            self._warn(f"NATS not available at start ({type(e).__name__}: {e}); call context stays in memory until it is")
        self._ensure_task()

    async def _ensure(self):
        if self._ready and self._nc is not None and getattr(self._nc, "is_connected", True):
            return
        if self._nc is None:
            import nats  # lazy

            async def _err(e):
                self.stats["last_error"] = f"{type(e).__name__}: {e}"[:200]

            self._nc = await nats.connect(self.url, connect_timeout=1, max_reconnect_attempts=-1, reconnect_time_wait=1,
                                          allow_reconnect=True, error_cb=_err, name="salesgod-ctx")
        self._js = self._nc.jetstream(timeout=self.write_timeout_s * 4)
        await self._ensure_stream()
        await self._ensure_kv()
        self._ready = True
        log.info("call context on NATS JetStream: stream %s (%s.*.event), KV %s", self.stream_name, self.prefix, self.bucket_name)

    async def _ensure_stream(self) -> None:
        from nats.js.api import DiscardPolicy, StorageType, StreamConfig
        cfg = StreamConfig(name=self.stream_name, subjects=[f"{self.prefix}.>"], storage=StorageType.FILE,
                           max_age=self.max_age_s, max_bytes=512 * 1024 ** 2, max_msg_size=2 * 1024 ** 2,
                           discard=DiscardPolicy.OLD, num_replicas=1)
        try:
            await self._js.stream_info(self.stream_name)
        except Exception:
            await self._js.add_stream(cfg)

    async def _ensure_kv(self) -> None:
        from nats.js.api import KeyValueConfig, StorageType
        try:
            self._kv = await self._js.key_value(self.bucket_name)
        except Exception:
            self._kv = await self._js.create_key_value(KeyValueConfig(
                bucket=self.bucket_name, history=1, ttl=self.max_age_s, storage=StorageType.FILE,
                max_value_size=2 * 1024 ** 2, max_bytes=256 * 1024 ** 2))

    def _warn(self, msg: str) -> None:
        self.stats["errors"] += 1
        self.stats["last_error"] = msg[:200]
        now = time.monotonic()
        if now - self._last_warn > 60:
            self._last_warn = now
            log.warning("%s", msg)

    # ------------------------------------------------------------------ background writer

    def _ensure_task(self) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if self._task is None or self._task.done():
            self._wake = self._wake or asyncio.Event()
            self._task = loop.create_task(self._run())
        self._wake.set()

    async def _run(self) -> None:
        while True:
            if not (self._kv_pending or self._events):
                await self._wake.wait()
                self._wake.clear()
            try:
                await asyncio.wait_for(self._ensure(), 3.0)
                while self._kv_pending:
                    key, data = next(iter(self._kv_pending.items()))
                    await asyncio.wait_for(self._kv.put(key, data), self.write_timeout_s * 4)
                    if self._kv_pending.get(key) is data:
                        del self._kv_pending[key]
                    self.stats["kv_put"] += 1
                while self._events:
                    subject, data = self._events[0]
                    await asyncio.wait_for(self._js.publish(subject, data), self.write_timeout_s * 4)
                    self._events.popleft()
                    self.stats["events"] += 1
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self._warn(f"NATS write failed ({type(e).__name__}: {e}); call context kept in memory, retrying")
                if not self._injected:
                    self._ready = False
                await asyncio.sleep(2.0)

    # ------------------------------------------------------------------ SessionStore protocol

    def _remember(self, session_id: str, raw: str) -> None:
        self.local[session_id] = raw
        self.local.move_to_end(session_id)
        while len(self.local) > self.local_max:
            self.local.popitem(last=False)

    def get_cached(self, session_id: str) -> SessionState | None:
        """The in-process mirror (same process, pipeline recreated or session evicted): no I/O."""
        raw = self.local.get(session_id)
        if not raw:
            return None
        try:
            return SessionState.model_validate_json(raw)
        except Exception:
            return None

    async def get(self, session_id: str) -> SessionState | None:
        """Mirror first, then KV (tight timeout), then the last event on the stream. None when nothing is found in time."""
        s = self.get_cached(session_id)
        if s is not None:
            return s
        if not self.connected:
            self.stats["restore_miss"] += 1
            return None
        try:
            raw = await asyncio.wait_for(self._fetch(session_id), self.read_timeout_s)
        except asyncio.TimeoutError:
            self.stats["restore_timeout"] += 1
            log.warning("session %s: NATS restore timed out (%.0f ms); starting from memory", session_id, self.read_timeout_s * 1000)
            return None
        except Exception as e:
            self._warn(f"NATS restore failed for {session_id}: {type(e).__name__}: {e}")
            return None
        if not raw:
            self.stats["restore_miss"] += 1
            return None
        try:
            s = SessionState.model_validate_json(raw)
        except Exception as e:
            self._warn(f"bad snapshot for {session_id}: {e}")
            return None
        self.stats["restored"] += 1
        self._remember(session_id, raw if isinstance(raw, str) else raw.decode())
        log.info("session %s restored from NATS (turn %d, %d history lines)", session_id, s.turn_index, len(s.history))
        return s

    async def _fetch(self, session_id: str):
        key = safe_token(session_id)
        try:
            entry = await self._kv.get(key)
            if entry is not None and entry.value:
                return entry.value
        except Exception as e:
            if "NotFound" not in type(e).__name__ and "not found" not in str(e).lower():
                raise
        # KV miss: replay the stream (the last turn event carries the snapshot)
        try:
            msg = await self._js.get_last_msg(self.stream_name, f"{self.prefix}.{call_token(session_id)}.event")
        except Exception:
            return None
        try:
            ev = json.loads(msg.data)
        except (ValueError, TypeError):
            return None
        snap = ev.get("session")
        return json.dumps(snap) if snap else None

    async def put(self, session: SessionState) -> None:
        self.record_turn(session, None)

    def record_turn(self, session: SessionState, event: dict | None) -> None:
        """Called at the end of every turn. Non-blocking: serialise, mirror, queue; the background task sends it."""
        try:
            raw = session.model_dump_json()
        except Exception as e:
            self._warn(f"snapshot failed for {session.session_id}: {e}")
            return
        self._remember(session.session_id, raw)
        key = safe_token(session.session_id)
        self._kv_pending.pop(key, None)
        self._kv_pending[key] = raw.encode()
        while len(self._kv_pending) > self.local_max:  # broker down for long: keep only the newest calls' snapshots
            self._kv_pending.pop(next(iter(self._kv_pending)))
            self.stats["dropped"] += 1
        if event is not None:
            ev = dict(event)
            ev["session"] = json.loads(raw)
            self.publish(f"{self.prefix}.{call_token(session.session_id)}.event", ev)
        self._ensure_task()

    def publish(self, subject: str, payload: dict) -> None:
        if len(self._events) >= self.max_queue:
            self.stats["dropped"] += 1
            return
        try:
            self._events.append((subject, json.dumps(payload, ensure_ascii=False, default=str).encode()))
        except Exception as e:
            self._warn(f"event not serialisable: {e}")
            return
        self._ensure_task()

    async def delete(self, session_id: str) -> None:
        """End of call: the in-process copy goes; the durable snapshot stays (bucket TTL) for the console and audits."""
        self.local.pop(session_id, None)

    async def load_snapshot(self, session_id: str, timeout_s: float = 0.3) -> dict | None:
        """For the console endpoint: the latest snapshot as a dict (mirror, else KV/stream)."""
        raw = self.local.get(session_id)
        if raw is None and self.connected:
            try:
                raw = await asyncio.wait_for(self._fetch(session_id), timeout_s)
            except Exception:
                raw = None
        if not raw:
            return None
        try:
            return json.loads(raw)
        except (ValueError, TypeError):
            return None

    # ------------------------------------------------------------------ sink for ledger events (pipeline.context_sinks)

    def for_call(self, call_id: str):
        subject = f"{self.prefix}.{safe_token(call_id)}.ctx"

        def sink(ev) -> None:
            try:
                self.publish(subject, ev.model_dump() if hasattr(ev, "model_dump") else dict(ev))
            except Exception:
                pass
        return sink

    def status(self) -> dict:
        return {"url": self.url.split("@")[-1], "connected": self.connected, "pending_kv": len(self._kv_pending),
                "pending_events": len(self._events), **self.stats}

    async def flush(self, timeout: float = 2.0) -> None:
        end = time.monotonic() + timeout
        self._ensure_task()
        while (self._kv_pending or self._events) and time.monotonic() < end:
            await asyncio.sleep(0.01)

    async def close(self) -> None:
        await self.flush()
        if self._task:
            self._task.cancel()
        if self._nc is not None and not self._injected:
            try:
                await self._nc.close()
            except Exception:
                pass
