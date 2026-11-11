"""Bounded thread pool for inference.

Synthesis is synchronous CPU work.  Running it inside an ``async def`` handler
blocks the event loop, so one long render stalls *every* other request —
including ``/health``.  Every engine call therefore goes through
:class:`InferenceExecutor`, which owns an explicit, bounded
:class:`~concurrent.futures.ThreadPoolExecutor` and awaits the result with
``loop.run_in_executor``.

Why an explicit executor rather than the default anyio thread pool: the default
pool is shared with synchronous route handlers and is far larger than a small
edge node can usefully run.  A dedicated pool makes the worst-case number of
simultaneous renders a configuration value (``SPEECH_THREAD_POOL_SIZE``) instead
of an accident.

The pool is created lazily so importing the service stays cheap, and it is
created once per process.  It is *not* the per-engine limit: that is
:class:`app.engines.concurrency.EngineLimiter`, which protects a single model
from being oversubscribed even when the pool has spare threads.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

from app.config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T")


class InferenceExecutor:
    """A bounded thread pool that never runs work on the event loop thread."""

    def __init__(self, max_workers: int | None = None, *, name: str = "tts-inference") -> None:
        self._max_workers = max(1, int(max_workers or settings.speech_thread_pool_size))
        self._name = name
        self._executor: ThreadPoolExecutor | None = None
        self._lock = threading.Lock()

    @property
    def max_workers(self) -> int:
        """The maximum number of simultaneously running inference calls."""
        return self._max_workers

    @property
    def executor(self) -> ThreadPoolExecutor:
        """The underlying pool, created on first use."""
        with self._lock:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=self._max_workers,
                    thread_name_prefix=self._name,
                )
            return self._executor

    async def run(self, func: Callable[..., T], /, *args: Any, **kwargs: Any) -> T:
        """Run ``func`` in the pool and await its result.

        Exceptions raised by ``func`` propagate unchanged, so engine-level
        errors (``EngineUnavailableError``, ``UnknownVoiceError``) keep the same
        meaning they have in the synchronous path.
        """
        loop = asyncio.get_running_loop()
        call = functools.partial(func, *args, **kwargs)
        return await loop.run_in_executor(self.executor, call)

    def shutdown(self, wait: bool = False) -> None:
        """Stop accepting work and release the threads."""
        with self._lock:
            executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=wait)

    def __enter__(self) -> InferenceExecutor:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.shutdown()


__all__ = ["InferenceExecutor"]
