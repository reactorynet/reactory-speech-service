"""Engine manager: registration, lifecycle, readiness and concurrency limits.

Phase 1 registered engines and initialised them all at startup.  That is fine
for one model and wrong for two: every loaded engine keeps its weights resident,
so a second model can exhaust a small edge container.  This module now owns the
lifecycle described in ``docs/architecture/speech-platform.md``:

* **lazy load on first use** — an engine that is not loaded is loaded by
  :meth:`ensure_loaded` when a request actually needs it;
* **``max_loaded_engines``** — loading a new engine evicts the least recently
  used *idle* engine first, and fails rather than oversubscribing memory when
  every loaded engine is busy;
* **idle unload** — :meth:`reap_idle` unloads engines that have been unused for
  ``SPEECH_ENGINE_IDLE_TIMEOUT_SECONDS``; a later request reloads them;
* **per-engine concurrency** — :meth:`limiter` exposes an
  :class:`app.engines.concurrency.EngineLimiter` so one model is never
  oversubscribed;
* **status** — :meth:`status` reports readiness, lifecycle state, in-flight
  work and queue depth for ``/health`` and ``/ready``.

Startup stays backwards compatible: :meth:`initialize` loads the first
``max_loaded_engines`` engines eagerly (with one registered engine that is all of
them), and any engine beyond that budget is deferred to first use.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Iterable
from typing import Any

from app.config import settings
from app.core.contracts import EngineCapabilities, TTSEngine
from app.engines.base import EngineCapacityError, EngineUnavailableError, UnknownEngineError
from app.engines.concurrency import EngineLimiter
from app.services import voice_registry

logger = logging.getLogger(__name__)


class EngineManager:
    """Owns the registered engine adapters for this deployment."""

    def __init__(
        self,
        engines: Iterable[TTSEngine] | None = None,
        *,
        max_loaded_engines: int | None = None,
        idle_timeout_seconds: float | None = None,
        clock: Any = None,
    ) -> None:
        self._engines: dict[str, TTSEngine] = {}
        self._max_loaded = (
            int(max_loaded_engines)
            if max_loaded_engines is not None
            else int(settings.speech_max_loaded_engines)
        )
        self._idle_timeout = (
            float(idle_timeout_seconds)
            if idle_timeout_seconds is not None
            else float(settings.speech_engine_idle_timeout_seconds)
        )
        self._clock = clock or time.monotonic
        #: engine id -> monotonic timestamp of the last request that used it.
        self._last_used: dict[str, float] = {}
        #: engine ids currently being loaded.
        self._loading: set[str] = set()
        #: primitives are per event loop: TestClient creates one per request.
        self._limiters: dict[tuple[int, str], tuple[asyncio.AbstractEventLoop, EngineLimiter]] = {}
        self._locks: dict[
            tuple[int, str], tuple[asyncio.AbstractEventLoop, asyncio.Lock]
        ] = {}
        self._reaper: asyncio.Task | None = None
        #: engine id -> the exception from the most recent failed load.
        #:
        #: The cause is preserved so the caller-facing fallback warning can carry
        #: the actionable detail (model path, download command) instead of a
        #: generic "could not be loaded" that only the log ever explains.
        self._load_errors: dict[str, BaseException] = {}
        for engine in engines or ():
            self.register(engine)

    # ------------------------------------------------------------------
    # Registration and lookup
    # ------------------------------------------------------------------

    def register(self, engine: TTSEngine) -> TTSEngine:
        """Register an adapter; re-registering the same id replaces it."""
        engine_id = getattr(engine, "id", None)
        if not engine_id:
            raise ValueError("an engine adapter must expose a non-empty 'id'")
        self._engines[str(engine_id)] = engine
        return engine

    def get(self, engine_id: str | None) -> TTSEngine | None:
        """Return a registered engine, or ``None``."""
        if engine_id is None:
            return None
        return self._engines.get(engine_id)

    def require(self, engine_id: str) -> TTSEngine:
        """Return a registered engine or raise :class:`UnknownEngineError`."""
        engine = self.get(engine_id)
        if engine is None:
            raise UnknownEngineError(
                f"Unknown engine '{engine_id}'. Registered engines: "
                f"{', '.join(self.ids) or 'none'}. GET /api/tts/capabilities lists them."
            )
        return engine

    @property
    def ids(self) -> list[str]:
        """Registered engine ids, in registration order."""
        return list(self._engines)

    @property
    def engines(self) -> list[TTSEngine]:
        """Registered adapters, in registration order."""
        return list(self._engines.values())

    @property
    def max_loaded_engines(self) -> int:
        """The configured budget.  ``0`` (or less) means unlimited."""
        return self._max_loaded

    @property
    def idle_timeout_seconds(self) -> float:
        return self._idle_timeout

    # ------------------------------------------------------------------
    # Readiness
    # ------------------------------------------------------------------

    def is_ready(self, engine_id: str | None) -> bool:
        """Whether one engine is loaded and can render."""
        engine = self.get(engine_id)
        return bool(getattr(engine, "is_ready", False)) if engine is not None else False

    def any_ready(self) -> bool:
        """Whether at least one engine can render."""
        return any(self.is_ready(engine_id) for engine_id in self.ids)

    def readiness(self) -> dict[str, bool]:
        """Per-engine readiness, for ``/health`` and ``/ready``."""
        return {engine_id: self.is_ready(engine_id) for engine_id in self.ids}

    def loaded_ids(self) -> list[str]:
        """Engine ids that currently hold a model."""
        return [engine_id for engine_id in self.ids if self.is_ready(engine_id)]

    def touch(self, engine_id: str) -> None:
        """Record that an engine was just used, for idle-timeout accounting."""
        if engine_id in self._engines:
            self._last_used[engine_id] = self._clock()

    # ------------------------------------------------------------------
    # Concurrency
    # ------------------------------------------------------------------

    def limiter(self, engine_id: str) -> EngineLimiter:
        """The per-engine concurrency limiter, bound to the running loop.

        Semaphores are loop-bound, so a limiter is created per event loop.  A
        process normally has one loop; the indirection exists so the test client
        (which creates a portal per request) and any future worker isolation
        cannot leak a bound semaphore across loops.
        """
        loop = asyncio.get_running_loop()
        key = (id(loop), engine_id)
        existing = self._limiters.get(key)
        if existing is not None and existing[0] is loop:
            return existing[1]
        limiter = EngineLimiter(engine_id, settings.engine_concurrency_limit(engine_id))
        self._limiters[key] = (loop, limiter)
        self._prune_loop_state()
        return limiter

    def limiter_snapshot(self, engine_id: str) -> dict[str, Any]:
        """Aggregate limiter counters for one engine, loop-independent."""
        self._prune_loop_state()
        in_flight = 0
        queue_depth = 0
        peak = 0
        for (_, registered_id), (_, limiter) in self._limiters.items():
            if registered_id != engine_id:
                continue
            in_flight += limiter.in_flight
            queue_depth += limiter.waiting
            peak = max(peak, limiter.peak)
        return {
            "max_concurrency": settings.engine_concurrency_limit(engine_id),
            "in_flight": in_flight,
            "queue_depth": queue_depth,
            "peak_concurrency": peak,
        }

    def _prune_loop_state(self) -> None:
        """Forget primitives bound to event loops that no longer exist."""
        for key, (loop, _) in list(self._limiters.items()):
            if loop.is_closed():
                del self._limiters[key]
        for key, (loop, _) in list(self._locks.items()):
            if loop.is_closed():
                del self._locks[key]

    def _lock_for(self, engine_id: str) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        key = (id(loop), engine_id)
        entry = self._locks.get(key)
        if entry is not None and entry[0] is loop:
            return entry[1]
        lock = asyncio.Lock()
        self._locks[key] = (loop, lock)
        return lock

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Load up to ``max_loaded_engines`` engines; defer the rest to first use.

        An adapter may declare ``lazy_load = True`` (the optional expressive
        engine does) to be skipped entirely at startup.  That keeps startup time
        unchanged when the flag is on and means a missing 3.1 GB model is not
        discovered by blocking the boot path.
        """
        budget = self._max_loaded if self._max_loaded > 0 else len(self._engines)
        loaded = 0
        for engine_id, engine in self._engines.items():
            if getattr(engine, "lazy_load", False):
                logger.info(
                    "Engine '%s' is lazy-loaded; it will load on first use.", engine_id
                )
                continue
            if loaded >= budget:
                logger.info(
                    "Engine '%s' is registered but not loaded; max_loaded_engines=%d. "
                    "It will load on first use.",
                    engine_id,
                    budget,
                )
                continue
            await self._load(engine_id)
            if self.is_ready(engine_id):
                loaded += 1

    async def _load(self, engine_id: str) -> None:
        """Initialise one engine, isolating (and reporting) failures.

        The failure is *recorded*, not discarded: :meth:`ensure_loaded` re-raises
        it as the cause of its :class:`EngineUnavailableError`, so a caller who
        removes an optional engine's model receives the actionable detail in the
        response rather than only in the server log.

        An :class:`EngineUnavailableError` is an expected, operator-actionable
        condition (the optional model is absent, or torch is not installed).  It
        is logged once at WARNING with no traceback: a stack trace per request
        for a documented deployment state is noise that hides real faults.
        """
        engine = self.require(engine_id)
        initialise = getattr(engine, "initialize", None)
        if not callable(initialise):
            return
        self._loading.add(engine_id)
        previously_failed = engine_id in self._load_errors
        try:
            await initialise()
        except EngineUnavailableError as exc:
            # An expected, operator-actionable state: log it once, not per
            # request, and never with a traceback.
            self._load_errors[engine_id] = exc
            if previously_failed:
                logger.debug("TTS engine '%s' is still unavailable: %s", engine_id, exc)
            else:
                logger.warning("TTS engine '%s' is unavailable: %s", engine_id, exc)
        except Exception as exc:
            self._load_errors[engine_id] = exc
            logger.exception("Failed to initialise TTS engine '%s'", engine_id)
        else:
            self._load_errors.pop(engine_id, None)
        finally:
            self._loading.discard(engine_id)

    def last_load_error(self, engine_id: str) -> BaseException | None:
        """The cause of the most recent failed load for one engine, if any."""
        return self._load_errors.get(engine_id)

    async def ensure_loaded(self, engine_id: str) -> TTSEngine:
        """Return a ready engine, loading it on first use.

        Raises :class:`EngineUnavailableError` when the engine cannot be loaded
        and :class:`EngineCapacityError` when ``max_loaded_engines`` is reached
        and no loaded engine is idle enough to evict.
        """
        engine = self.require(engine_id)
        if self.is_ready(engine_id):
            self.touch(engine_id)
            return engine

        async with self._lock_for(engine_id):
            if self.is_ready(engine_id):
                self.touch(engine_id)
                return engine
            self.reap_idle()
            self._make_room(engine_id)
            await self._load(engine_id)
            if not self.is_ready(engine_id):
                cause = self._load_errors.get(engine_id)
                detail = f"{cause}" if cause is not None else "no model could be loaded"
                raise EngineUnavailableError(
                    f"TTS engine '{engine_id}' could not be loaded on this deployment: {detail}"
                ) from cause
            self.touch(engine_id)
            return engine

    def _make_room(self, engine_id: str) -> None:
        """Evict idle engines until a slot is free.

        Eviction is LRU over *idle* engines only: an engine with work in flight
        is never unloaded underneath a request.  If every loaded engine is busy
        the request fails with :class:`EngineCapacityError` rather than
        exceeding the memory budget.
        """
        if self._max_loaded <= 0:
            return
        tried: set[str] = set()
        while len(self.loaded_ids()) >= self._max_loaded:
            candidates = [
                candidate
                for candidate in self.loaded_ids()
                if candidate != engine_id
                and candidate not in tried
                and not self.limiter_snapshot(candidate)["in_flight"]
            ]
            if not candidates:
                raise EngineCapacityError(
                    f"max_loaded_engines={self._max_loaded} reached and no loaded engine could "
                    f"be released to load '{engine_id}'"
                )
            victim = min(candidates, key=lambda candidate: self._last_used.get(candidate, 0.0))
            tried.add(victim)
            before = len(self.loaded_ids())
            logger.info(
                "Unloading engine '%s' to make room for '%s' (max_loaded_engines=%d)",
                victim,
                engine_id,
                self._max_loaded,
            )
            released = self.unload(victim)
            # ``unload`` returns False when the adapter exposes no lifecycle hook
            # or when its hook raises.  Either way the engine is still resident,
            # so retrying the same eviction would spin forever *synchronously*,
            # blocking the event loop — the exact failure this ticket removes.
            # Fail with ``EngineCapacityError`` instead, which the router already
            # treats as "this engine could not be loaded".
            if not released or len(self.loaded_ids()) >= before:
                raise EngineCapacityError(
                    f"max_loaded_engines={self._max_loaded} reached and engine '{victim}' "
                    f"could not be released; cannot load '{engine_id}'"
                )

    def unload(self, engine_id: str, *, force: bool = False) -> bool:
        """Release one engine's model.  Returns whether it was unloaded."""
        engine = self.get(engine_id)
        if engine is None:
            return False
        if not force and self.limiter_snapshot(engine_id)["in_flight"]:
            return False
        unload_hook = getattr(engine, "unload", None)
        if callable(unload_hook):
            try:
                unload_hook()
            except Exception:
                logger.exception("Failed to unload TTS engine '%s'", engine_id)
                return False
        else:
            setter = getattr(engine, "set_ready", None)
            if callable(setter):
                setter(False)
            elif getattr(engine, "is_ready", False):
                # An adapter without a lifecycle hook cannot be unloaded safely.
                logger.debug("Engine '%s' has no unload hook; leaving it loaded", engine_id)
                return False
        self._last_used.pop(engine_id, None)
        self._load_errors.pop(engine_id, None)
        logger.info("Unloaded TTS engine '%s'", engine_id)
        return True

    def reap_idle(self, now: float | None = None) -> list[str]:
        """Unload engines idle for longer than the configured timeout.

        One engine is always kept resident.  Readiness and the next request must
        not pay a cold model load just because traffic paused, so the last loaded
        engine is never reaped; the memory this frees is the second and further
        models, which is exactly the pressure ``max_loaded_engines`` exists for.
        """
        if self._idle_timeout <= 0:
            return []
        moment = self._clock() if now is None else now
        unloaded: list[str] = []
        for engine_id in self.loaded_ids():
            if len(self.loaded_ids()) <= 1:
                break
            last_used = self._last_used.get(engine_id)
            if last_used is None:
                continue
            if moment - last_used < self._idle_timeout:
                continue
            if self.limiter_snapshot(engine_id)["in_flight"]:
                continue
            if self.unload(engine_id):
                unloaded.append(engine_id)
        if unloaded:
            logger.info("Unloaded idle TTS engines: %s", ", ".join(unloaded))
        return unloaded

    async def start_reaper(self, interval_seconds: float | None = None) -> None:
        """Run :meth:`reap_idle` on a background task until the process stops."""
        if self._reaper is not None and not self._reaper.done():
            return
        interval = max(
            0.01,
            float(
                interval_seconds
                if interval_seconds is not None
                else settings.speech_engine_reaper_interval_seconds
            ),
        )

        async def _loop() -> None:
            while True:
                await asyncio.sleep(interval)
                try:
                    self.reap_idle()
                except Exception:  # pragma: no cover - defensive
                    logger.exception("Idle engine reaper failed")

        self._reaper = asyncio.create_task(_loop(), name="tts-engine-reaper")

    async def stop_reaper(self) -> None:
        """Cancel the background reaper, if one is running."""
        reaper, self._reaper = self._reaper, None
        if reaper is None:
            return
        reaper.cancel()
        try:
            await reaper
        except asyncio.CancelledError:
            # Expected: we just cancelled the reaper ourselves. Cancellation is
            # control flow here, not a failure, so it is deliberately swallowed.
            pass
        except Exception:  # pragma: no cover - defensive
            logger.exception("Idle engine reaper stopped unexpectedly")

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def status(self) -> dict[str, dict[str, Any]]:
        """Per-engine readiness, lifecycle and queue depth.

        This is a read-only snapshot and never loads or unloads anything, so it
        is safe to call from ``/health`` while syntheses are running.
        """
        now = self._clock()
        payload: dict[str, dict[str, Any]] = {}
        for engine_id, engine in self._engines.items():
            ready = self.is_ready(engine_id)
            if ready:
                lifecycle = "ready"
            elif engine_id in self._loading:
                lifecycle = "loading"
            else:
                lifecycle = "unloaded"
            last_used = self._last_used.get(engine_id)
            limiter = self.limiter_snapshot(engine_id)
            capabilities = _safe_capabilities(engine)
            payload[engine_id] = {
                "id": engine_id,
                "ready": ready,
                "loaded": ready,
                "lifecycle": lifecycle,
                "queue_depth": limiter["queue_depth"],
                "in_flight": limiter["in_flight"],
                "max_concurrency": limiter["max_concurrency"],
                "peak_concurrency": limiter["peak_concurrency"],
                "idle_seconds": None if last_used is None else round(max(0.0, now - last_used), 3),
                "gpu_required": capabilities.gpu_required if capabilities else False,
                "max_chars": capabilities.max_chars if capabilities else None,
                "voices": _safe_voice_count(engine),
            }
        return payload

    def total_queue_depth(self) -> int:
        """Queued (not yet running) inference calls across all engines."""
        return sum(row["queue_depth"] for row in self.status().values())

    def capabilities(self) -> list[EngineCapabilities]:
        """The capability matrix for every registered engine."""
        return [engine.capabilities() for engine in self.engines]

    def descriptors(self) -> list[dict[str, Any]]:
        """The capability payload for every registered engine.

        Adapters may provide a richer ``descriptor()``; the fallback is built from
        the engine-agnostic contract so a new adapter cannot break the endpoint.
        """
        rows: list[dict[str, Any]] = []
        for engine in self.engines:
            descriptor = getattr(engine, "descriptor", None)
            if callable(descriptor):
                rows.append(descriptor())
            else:
                rows.append(self._generic_descriptor(engine))
        return rows

    @staticmethod
    def _generic_descriptor(engine: TTSEngine) -> dict[str, Any]:
        capabilities = engine.capabilities()
        return {
            "id": capabilities.engine,
            "engine": capabilities.engine,
            "name": getattr(engine, "name", capabilities.engine),
            "available": bool(getattr(engine, "is_ready", False)),
            "license": getattr(engine, "license", None),
            "languages": capabilities.languages,
            "unsupported_languages": capabilities.unsupported_languages,
            "languages_with_caveats": capabilities.languages_with_caveats,
            "languages_pending": capabilities.languages_pending,
            "language_support": capabilities.language_support,
            "voice_count": len(engine.voices()),
            "default_voice": voice_registry.default_voice_id(),
            "native_rate": capabilities.native_rate,
            "native_pitch": capabilities.native_pitch,
            "native_volume": capabilities.native_volume,
            "styles": capabilities.styles,
            "supports_cloning": capabilities.supports_cloning,
            "supports_streaming": capabilities.supports_streaming,
            "supports_ssml": capabilities.supports_ssml,
            "sample_rates": capabilities.sample_rates,
            "max_chars": capabilities.max_chars,
            "gpu_required": capabilities.gpu_required,
            "attribution": capabilities.attribution,
            "variant": capabilities.variant,
            "compute_cost": capabilities.compute_cost,
            "capability_notes": capabilities.capability_notes,
            "supports_paralinguistic_tags": capabilities.supports_paralinguistic_tags,
        }


def _safe_capabilities(engine: TTSEngine) -> EngineCapabilities | None:
    try:
        return engine.capabilities()
    except Exception:  # pragma: no cover - a broken adapter must not break /health
        return None


def _safe_voice_count(engine: TTSEngine) -> int | None:
    try:
        return len(engine.voices())
    except Exception:  # pragma: no cover - a broken adapter must not break /health
        return None


__all__ = ["EngineManager"]
