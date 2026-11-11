"""Per-engine concurrency limits.

A bounded thread pool stops the process from spawning unbounded work, but it
does not stop *one* engine from being oversubscribed: a four-thread pool with
four kokoro slots will happily run four renders of the same model, which is
slower than two and multiplies peak memory.  :class:`EngineLimiter` is the
per-engine gate:

* ``kokoro`` is CPU-bound and allows two overlapping renders by default;
* GPU and expressive engines allow one, because a second render competes for
  the same VRAM.

Limits are configured through ``SPEECH_ENGINE_MAX_CONCURRENCY`` and are read
per engine, so adding a second engine needs no code change here.

The limiter also tracks queue depth, which is what ``/health`` and ``/ready``
report.  ``queue_depth`` is the number of calls *waiting* for a slot; it is
separate from ``in_flight``.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from app.config import settings


class EngineLimiter:
    """An async semaphore plus the counters an operator needs to see."""

    def __init__(self, engine_id: str, limit: int | None = None) -> None:
        self.engine_id = engine_id
        self._limit = max(1, int(limit if limit is not None else _limit_for(engine_id)))
        self._semaphore = asyncio.Semaphore(self._limit)
        self._in_flight = 0
        self._waiting = 0
        self._peak = 0

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def in_flight(self) -> int:
        return self._in_flight

    @property
    def waiting(self) -> int:
        """Calls currently queued for a slot — the value reported as queue depth."""
        return self._waiting

    @property
    def peak(self) -> int:
        """Highest observed concurrency, for diagnostics and tests."""
        return self._peak

    @property
    def busy(self) -> bool:
        """Whether the engine is doing work or has work queued."""
        return self._in_flight > 0 or self._waiting > 0

    async def acquire(self) -> None:
        self._waiting += 1
        try:
            await self._semaphore.acquire()
        finally:
            self._waiting -= 1
        self._in_flight += 1
        self._peak = max(self._peak, self._in_flight)

    def release(self) -> None:
        if self._in_flight > 0:
            self._in_flight -= 1
        self._semaphore.release()

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[None]:
        """Hold one concurrency slot for the duration of the block."""
        await self.acquire()
        try:
            yield
        finally:
            self.release()

    def snapshot(self) -> dict[str, Any]:
        """A JSON-friendly view for ``/health`` and ``/ready``."""
        return {
            "max_concurrency": self._limit,
            "in_flight": self._in_flight,
            "queue_depth": self._waiting,
            "peak_concurrency": self._peak,
        }


def _limit_for(engine_id: str) -> int:
    return settings.engine_concurrency_limit(engine_id)


__all__ = ["EngineLimiter"]
