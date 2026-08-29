"""In-memory run store for the local single-user visualizer."""

from __future__ import annotations

import asyncio
import json
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from rag.trace import RunStatus, RunTraceCollector, utc_now

TERMINAL_EVENTS = frozenset({"run_completed", "run_failed"})


def sse_frame(event: str, payload: dict[str, Any] | None = None) -> str:
    """Format one SSE message with a named event and JSON data."""
    data = json.dumps(payload if payload is not None else {}, ensure_ascii=False)
    return f"event: {event}\ndata: {data}\n\n"


def sse_comment(text: str) -> str:
    return f": {text}\n\n"


@dataclass
class RunRecord:
    run_id: str
    status: str = RunStatus.QUEUED.value
    created_at: str = field(default_factory=utc_now)
    request: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    trace: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    subscribers: list[asyncio.Queue] = field(default_factory=list)
    finished: asyncio.Event | None = None
    done_flag: threading.Event = field(default_factory=threading.Event)

    def snapshot(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "status": self.status,
            "created_at": self.created_at,
            "request": self.request,
            "trace": self.trace,
            "error": self.error,
            "event_count": len(self.events),
        }


class RunStore:
    """Keep the most recent runs in memory. Nothing is written to disk."""

    def __init__(self, *, max_runs: int = 20, keepalive_seconds: float = 15.0) -> None:
        self.max_runs = max_runs
        self.keepalive_seconds = keepalive_seconds
        self.runs: dict[str, RunRecord] = {}
        self.order: deque[str] = deque()
        self.work: asyncio.Queue[str] = asyncio.Queue()
        self._lock = threading.Lock()
        self.loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop

    async def create(self, run_id: str, request: dict[str, Any]) -> RunRecord:
        collector = RunTraceCollector(run_id=run_id)
        record = RunRecord(run_id=run_id, request=request)
        record.finished = asyncio.Event()
        record.trace = collector.snapshot().to_dict()
        record.trace["status"] = RunStatus.QUEUED.value
        with self._lock:
            self.runs[run_id] = record
            self.order.append(run_id)
            evicted: list[RunRecord] = []
            while len(self.order) > self.max_runs:
                old_id = self.order.popleft()
                old = self.runs.pop(old_id, None)
                if old is not None:
                    evicted.append(old)
        for old in evicted:
            self._mark_finished(old)
            self._notify(old, None)
        await self.work.put(run_id)
        return record

    def get(self, run_id: str) -> RunRecord | None:
        with self._lock:
            return self.runs.get(run_id)

    def list_ids(self) -> list[str]:
        with self._lock:
            return list(self.order)

    def publish_sync(self, run_id: str, event: str, payload: dict[str, Any], trace: dict[str, Any]) -> None:
        """Thread-safe: the RAG worker thread may call this directly."""
        with self._lock:
            record = self.runs.get(run_id)
            if record is None:
                return
            item = {"event": event, "payload": payload, "trace": trace}
            record.events.append(item)
            record.trace = trace
            record.status = trace.get("status", record.status)
            if event in TERMINAL_EVENTS:
                record.error = trace.get("error")
                record.done_flag.set()
            subscribers = list(record.subscribers)
        self._notify(record, item, list(subscribers))
        if event in TERMINAL_EVENTS:
            self._mark_finished(record)
            self._notify(record, None, list(subscribers))

    async def subscribe(self, run_id: str) -> AsyncIterator[str]:
        record = self.get(run_id)
        if record is None:
            return
        with self._lock:
            replay = list(record.events)
            done = bool(record.finished and record.finished.is_set())
        for item in replay:
            yield sse_frame(item["event"], item)
        if done:
            return
        queue: asyncio.Queue = asyncio.Queue()
        with self._lock:
            record.subscribers.append(queue)
            # Events may have arrived between replay and subscribe.
            extra = record.events[len(replay) :]
            done = bool(record.finished and record.finished.is_set())
        for item in extra:
            yield sse_frame(item["event"], item)
        if done:
            with self._lock:
                if queue in record.subscribers:
                    record.subscribers.remove(queue)
            return
        try:
            while True:
                try:
                    item = await asyncio.wait_for(
                        queue.get(), timeout=self.keepalive_seconds
                    )
                except TimeoutError:
                    current = self.get(run_id)
                    if current is None:
                        return
                    try:
                        item = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        yield sse_comment("keepalive")
                        continue
                if item is None:
                    return
                yield sse_frame(item["event"], item)
                if item.get("event") in TERMINAL_EVENTS:
                    return
        finally:
            with self._lock:
                if queue in record.subscribers:
                    record.subscribers.remove(queue)

    def _mark_finished(self, record: RunRecord) -> None:
        if record.finished is None:
            return
        loop = self.loop
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if loop is not None and running is not loop:
            loop.call_soon_threadsafe(record.finished.set)
        else:
            record.finished.set()

    def _notify(
        self,
        record: RunRecord,
        item: dict[str, Any] | None,
        subscribers: list[asyncio.Queue] | None = None,
    ) -> None:
        loop = self.loop
        queues = subscribers if subscribers is not None else list(record.subscribers)
        for queue in queues:
            if loop is None:
                continue
            loop.call_soon_threadsafe(queue.put_nowait, item)
