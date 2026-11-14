"""NATS event sink for call context events.

`NatsSink` is a pipeline context_sink: `pipeline.context_sinks.append(sink.for_call(call_id))`. Each event goes to
subject `acie.calls.<call_id>.context`. It never blocks or raises into the call: events go into a bounded queue
(dropped when full) and a background task publishes them. nats-py is imported lazily.
"""

from __future__ import annotations

import asyncio
import collections
import logging

log = logging.getLogger("acie.events")


class NatsSink:
    def __init__(self, url: str, call_id: str = "", max_queue: int = 1000, client=None, _shared=None) -> None:
        self.url, self.call_id, self.max_queue = url, call_id, max_queue
        self._shared = _shared or self  # call-bound children share the root's queue, task and connection
        if _shared is None:
            self._q: collections.deque = collections.deque()
            self._nc = client
            self._injected = client is not None
            self._task: asyncio.Task | None = None
            self._wake: asyncio.Event | None = None
            self.dropped = 0
            self.published = 0

    def for_call(self, call_id: str) -> "NatsSink":
        return NatsSink(self.url, call_id, self.max_queue, _shared=self._shared)

    def subject(self) -> str:
        return f"acie.calls.{self.call_id or 'unknown'}.context"

    def __call__(self, ev) -> None:
        try:
            root = self._shared
            if len(root._q) >= root.max_queue:
                root.dropped += 1
                return
            root._q.append((self.subject(), ev.model_dump_json().encode()))
            root._ensure_task()
            root._wake.set()
        except Exception as e:  # never into the call
            log.debug("nats sink dropped event: %s", e)

    def _ensure_task(self) -> None:
        if self._task is None or self._task.done():
            self._wake = self._wake or asyncio.Event()
            self._task = asyncio.get_running_loop().create_task(self._run())

    async def _connect(self):
        if self._nc is None:
            import nats  # lazy
            self._nc = await nats.connect(self.url, connect_timeout=2, max_reconnect_attempts=-1)
        return self._nc

    async def _run(self) -> None:
        while True:
            await self._wake.wait()
            self._wake.clear()
            while self._q:
                subject, data = self._q[0]
                try:
                    nc = await self._connect()
                    await nc.publish(subject, data)
                    self._q.popleft()
                    self.published += 1
                except Exception as e:
                    log.debug("nats publish failed: %s", e)
                    if not self._injected:
                        self._nc = None  # reconnect next round
                    await asyncio.sleep(1.0)  # broker down: keep the bounded queue, retry; overflow drops new events

    async def flush(self, timeout: float = 2.0) -> None:
        root = self._shared
        end = asyncio.get_running_loop().time() + timeout
        while root._q and asyncio.get_running_loop().time() < end:
            await asyncio.sleep(0.01)
        if root._nc is not None:
            try:
                await asyncio.wait_for(root._nc.flush(), timeout)
            except Exception:
                pass

    async def close(self) -> None:
        root = self._shared
        await self.flush()
        if root._task:
            root._task.cancel()
        if root._nc is not None:
            try:
                await root._nc.close()
            except Exception:
                pass
