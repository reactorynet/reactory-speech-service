"""Tests for engine lifecycle: lazy load, eviction, idle unload and health.

The acceptance criteria this file covers:

* ``max_loaded_engines`` is enforced and is visible in the status payload;
* an engine that is idle is unloaded and can be reloaded on demand;
* engines beyond the budget are loaded lazily on first use;
* ``/health`` and ``/ready`` report per-engine readiness and queue depth while
  keeping the legacy ``tts`` / ``stt`` booleans.
"""

from __future__ import annotations

import asyncio
import threading
import time
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi import FastAPI

from app.core.contracts import AudioChunk, AudioResult, EngineCapabilities
from app.engines.base import EngineCapacityError
from app.engines.manager import EngineManager
from app.routers import health, tts
from app.services.inference import InferenceExecutor
from app.services.synthesis_cache import SynthesisCache
from app.services.tts_service import TTSService

SAMPLE_RATE = 24000


class LifecycleEngine:
    """An engine that records load/unload calls and can block in inference."""

    name = "Lifecycle engine"
    license = "MIT"

    def __init__(
        self,
        engine_id: str,
        *,
        ready: bool = False,
        gate: threading.Event | None = None,
        delay: float = 0.0,
    ) -> None:
        self.id = engine_id
        self.is_ready = ready
        self.gate = gate
        self.delay = delay
        self.loads = 0
        self.unloads = 0
        self.in_flight = 0
        self.peak = 0
        self._lock = threading.Lock()

    async def initialize(self) -> None:
        self.loads += 1
        self.is_ready = True

    def unload(self) -> None:
        self.unloads += 1
        self.is_ready = False

    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            engine=self.id,
            languages=["en-US"],
            native_rate=True,
            sample_rates=[SAMPLE_RATE],
            max_chars=10000,
            gpu_required=False,
        )

    def voices(self):
        return []

    def synthesize(self, req):
        with self._lock:
            self.in_flight += 1
            self.peak = max(self.peak, self.in_flight)
        try:
            if self.gate is not None:
                self.gate.wait(timeout=10)
            if self.delay:
                time.sleep(self.delay)
            audio = f"RIFF-{self.id}-{req.text}".encode()
            return AudioResult(
                audio=audio,
                duration=0.1,
                sample_rate=SAMPLE_RATE,
                engine=self.id,
            )
        finally:
            with self._lock:
                self.in_flight -= 1

    def stream(self, req):
        yield AudioChunk(audio=b"RIFF", duration=0.1, sample_rate=SAMPLE_RATE)


def _service(tmp_path, *engines: LifecycleEngine, **manager_kwargs) -> TTSService:
    return TTSService(
        manager=EngineManager(list(engines), **manager_kwargs),
        cache=SynthesisCache(enabled=False, directory=tmp_path / "cache"),
        executor=InferenceExecutor(max_workers=4),
    )


# ─────────────────────────────────────────────────────────────
# Lazy loading and max_loaded_engines
# ─────────────────────────────────────────────────────────────

async def test_startup_loads_only_the_configured_number_of_engines():
    first = LifecycleEngine("first")
    second = LifecycleEngine("second")
    manager = EngineManager([first, second], max_loaded_engines=1)

    await manager.initialize()

    assert first.is_ready is True
    assert second.is_ready is False
    assert second.loads == 0
    assert manager.status()["second"]["lifecycle"] == "unloaded"


async def test_an_engine_beyond_the_budget_loads_on_first_use_and_evicts_an_idle_one():
    first = LifecycleEngine("first")
    second = LifecycleEngine("second")
    manager = EngineManager([first, second], max_loaded_engines=1)
    await manager.initialize()

    await manager.ensure_loaded("second")

    assert second.is_ready is True
    assert second.loads == 1
    assert first.is_ready is False
    assert first.unloads == 1
    assert manager.loaded_ids() == ["second"]


async def test_max_loaded_engines_fails_rather_than_evicting_a_busy_engine():
    gate = threading.Event()
    busy = LifecycleEngine("busy", ready=True, gate=gate)
    newcomer = LifecycleEngine("newcomer")
    manager = EngineManager([busy, newcomer], max_loaded_engines=1)
    service = TTSService(
        manager=manager,
        cache=SynthesisCache(enabled=False),
        executor=InferenceExecutor(max_workers=2),
    )

    task = asyncio.create_task(service.synthesize_async("Hold the engine.", cache=False))
    await _wait_for(lambda: busy.in_flight == 1)

    with pytest.raises(EngineCapacityError):
        await manager.ensure_loaded("newcomer")

    gate.set()
    _ = await task
    assert newcomer.loads == 0


async def test_an_idle_engine_is_unloaded_and_reloaded_on_demand():
    now = [0.0]
    resident = LifecycleEngine("resident", ready=True)
    engine = LifecycleEngine("engine", ready=True)
    manager = EngineManager(
        [resident, engine], idle_timeout_seconds=60.0, clock=lambda: now[0]
    )
    manager.touch("resident")  # t=0
    now[0] = 100.0
    manager.touch("engine")  # t=100
    now[0] = 150.0
    manager.touch("resident")  # t=150, still warm

    # Nothing has been idle for 60s yet: the engine is 50s old.
    assert manager.reap_idle() == []
    assert engine.is_ready is True

    now[0] = 200.0
    assert manager.reap_idle() == ["engine"]
    assert engine.is_ready is False
    assert engine.unloads == 1
    assert resident.is_ready is True, "the last loaded engine stays resident"

    await manager.ensure_loaded("engine")
    assert engine.is_ready is True
    assert engine.loads == 1


async def test_the_background_reaper_unloads_idle_engines():
    now = [0.0]
    first = LifecycleEngine("first", ready=True)
    second = LifecycleEngine("second", ready=True)
    manager = EngineManager([first, second], idle_timeout_seconds=1.0, clock=lambda: now[0])
    manager.touch("first")
    manager.touch("second")

    await manager.start_reaper(interval_seconds=0.02)
    try:
        now[0] = 5.0
        await _wait_for(lambda: sum(not e.is_ready for e in (first, second)) == 1, timeout=2.0)
    finally:
        await manager.stop_reaper()

    assert [first.unloads, second.unloads].count(1) == 1
    assert len(manager.loaded_ids()) == 1


async def test_an_engine_without_a_lifecycle_hook_is_left_alone():
    class Hookless:
        id = "hookless"
        is_ready = True

        def capabilities(self):
            return EngineCapabilities(engine=self.id)

        def voices(self):
            return []

        def synthesize(self, req):  # pragma: no cover - never reached
            raise AssertionError

        def stream(self, req):  # pragma: no cover - never reached
            yield None

    manager = EngineManager([Hookless()], idle_timeout_seconds=1.0, clock=lambda: 0.0)
    manager.touch("hookless")

    assert manager.reap_idle(now=10.0) == []
    await manager.ensure_loaded("hookless")


# ─────────────────────────────────────────────────────────────
# Status payload
# ─────────────────────────────────────────────────────────────

async def test_status_reports_readiness_lifecycle_and_queue_depth():
    engine = LifecycleEngine("engine", ready=True)
    manager = EngineManager([engine])
    manager.touch("engine")

    row = manager.status()["engine"]

    assert row["id"] == "engine"
    assert row["ready"] is True
    assert row["loaded"] is True
    assert row["lifecycle"] == "ready"
    assert row["queue_depth"] == 0
    assert row["in_flight"] == 0
    assert row["max_concurrency"] >= 1
    assert row["peak_concurrency"] == 0
    assert row["idle_seconds"] is not None
    assert row["gpu_required"] is False
    assert row["max_chars"] == 10000
    assert row["voices"] == 0


# ─────────────────────────────────────────────────────────────
# Health and readiness endpoints
# ─────────────────────────────────────────────────────────────

def _app(service: TTSService) -> FastAPI:
    app = FastAPI()
    app.state.tts_service = service
    app.state.stt_service = MagicMock(is_ready=True)
    app.include_router(health.router)
    app.include_router(tts.router, prefix="/api/tts")
    return app


async def _get_json(app: FastAPI, path: str) -> dict:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(path)
    assert response.status_code == 200
    return response.json()


async def test_health_reports_per_engine_status_and_keeps_the_legacy_booleans(tmp_path):
    service = _service(tmp_path, LifecycleEngine("engine", ready=True))
    payload = await _get_json(_app(service), "/health")

    assert payload["status"] == "ok"
    assert payload["tts"] is True
    assert payload["stt"] is True
    assert payload["queue_depth"] == 0

    engine = payload["engines"]["engine"]
    assert engine["ready"] is True
    assert engine["loaded"] is True
    assert engine["lifecycle"] == "ready"
    assert engine["queue_depth"] == 0
    assert engine["max_concurrency"] >= 1


async def test_ready_reports_per_engine_status_and_keeps_the_ready_key(tmp_path):
    engine = LifecycleEngine("engine", ready=True)
    service = _service(tmp_path, engine)
    payload = await _get_json(_app(service), "/ready")

    assert payload["ready"] is True
    assert payload["engines"]["engine"]["ready"] is True
    assert payload["queue_depth"] == 0


async def test_health_reports_queue_depth_while_syntheses_are_queued(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "app.config.settings.speech_engine_max_concurrency",
        {"engine": 1, "default": 1},
    )
    gate = threading.Event()
    engine = LifecycleEngine("engine", ready=True, gate=gate)
    service = _service(tmp_path, engine)
    app = _app(service)

    tasks = [
        asyncio.create_task(service.synthesize_async(f"Sentence {index}.", cache=False))
        for index in range(3)
    ]
    try:
        await _wait_for(lambda: engine.in_flight == 1)
        await _wait_for(lambda: service.engine_manager.status()["engine"]["queue_depth"] == 2)

        payload = await _get_json(app, "/health")

        assert payload["tts"] is True
        assert payload["engines"]["engine"]["in_flight"] == 1
        assert payload["engines"]["engine"]["queue_depth"] == 2
        assert payload["queue_depth"] == 2
    finally:
        gate.set()
        await asyncio.gather(*tasks)

    assert engine.peak == 1


async def _wait_for(predicate, *, timeout: float = 5.0, interval: float = 0.005):
    """Poll ``predicate`` from the event loop until it is true."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(interval)
    raise AssertionError("condition was not met within the timeout")

# ─────────────────────────────────────────────────────────────
# B3: _make_room must fail, never spin
# ─────────────────────────────────────────────────────────────
#
# ``unload()`` returns False when an adapter exposes no lifecycle hook or when its
# hook raises.  The eviction loop used to ignore that return value and recompute
# the same candidate set, spinning *synchronously* and blocking the event loop —
# the very failure mode this ticket exists to remove.  Each test below is wrapped
# in ``asyncio.wait_for``: a hang fails the test instead of hanging the suite.

class _HooklessEngine:
    """Protocol-only adapter: no ``unload``, no ``set_ready``, always "ready".

    This is a legitimate third-party adapter shape — which is exactly the
    scenario ``max_loaded_engines`` exists for.
    """

    name = "Hookless engine"
    license = "MIT"
    is_ready = True

    def __init__(self, engine_id: str) -> None:
        self.id = engine_id

    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(engine=self.id, sample_rates=[SAMPLE_RATE])

    def voices(self):
        return []

    def synthesize(self, req):  # pragma: no cover - never reached
        raise AssertionError("this engine should not render")

    def stream(self, req):  # pragma: no cover - never reached
        yield None


class _RaisingUnloadEngine(_HooklessEngine):
    """An adapter whose ``unload`` raises: ``unload()`` logs and returns False."""

    def __init__(self, engine_id: str) -> None:
        super().__init__(engine_id)

    def unload(self) -> None:
        raise OSError("cannot release the model on this platform")


class _LyingUnloadEngine(_HooklessEngine):
    """An adapter whose ``unload`` returns cleanly but leaves itself resident."""

    def unload(self) -> None:
        return None


async def test_make_room_raises_instead_of_spinning_when_no_hook_exists():
    resident = _HooklessEngine("resident")
    newcomer = LifecycleEngine("newcomer", ready=False)
    manager = EngineManager([resident, newcomer], max_loaded_engines=1)

    with pytest.raises(EngineCapacityError, match="could not be released"):
        await asyncio.wait_for(manager.ensure_loaded("newcomer"), timeout=3.0)

    assert newcomer.loads == 0
    assert resident.is_ready is True


async def test_make_room_raises_when_the_victims_unload_raises():
    resident = _RaisingUnloadEngine("resident")
    newcomer = LifecycleEngine("newcomer", ready=False)
    manager = EngineManager([resident, newcomer], max_loaded_engines=1)

    with pytest.raises(EngineCapacityError, match="could not be released"):
        await asyncio.wait_for(manager.ensure_loaded("newcomer"), timeout=3.0)

    assert newcomer.loads == 0


async def test_make_room_raises_when_unload_does_not_actually_release():
    """A hook that returns without releasing must not be retried forever."""
    resident = _LyingUnloadEngine("resident")
    newcomer = LifecycleEngine("newcomer", ready=False)
    manager = EngineManager([resident, newcomer], max_loaded_engines=1)

    with pytest.raises(EngineCapacityError, match="could not be released"):
        await asyncio.wait_for(manager.ensure_loaded("newcomer"), timeout=3.0)


async def test_a_failed_eviction_returns_promptly_instead_of_spinning():
    """A spin would run until the wait_for deadline (or forever, never yielding)."""
    resident = _HooklessEngine("resident")
    newcomer = LifecycleEngine("newcomer", ready=False)
    manager = EngineManager([resident, newcomer], max_loaded_engines=1)
    ticks: list[int] = []

    async def ticker() -> None:
        while True:
            ticks.append(1)
            await asyncio.sleep(0.01)

    beat = asyncio.create_task(ticker())
    await asyncio.sleep(0.05)  # let the heartbeat establish a baseline
    baseline = len(ticks)
    started = time.monotonic()
    try:
        with pytest.raises(EngineCapacityError):
            await asyncio.wait_for(manager.ensure_loaded("newcomer"), timeout=3.0)
    finally:
        beat.cancel()
        with pytest.raises(asyncio.CancelledError):
            _ = await beat
    elapsed = time.monotonic() - started

    assert baseline >= 2, "the heartbeat did not run before the eviction attempt"
    assert elapsed < 1.0, f"the failing eviction took {elapsed:.2f}s; it must fail fast"
    assert manager.loaded_ids() == ["resident"]


async def test_a_working_eviction_still_succeeds_after_the_guard():
    """The guard must not break the happy path it was added to protect."""
    idle = LifecycleEngine("idle", ready=True)
    newcomer = LifecycleEngine("newcomer", ready=False)
    manager = EngineManager([idle, newcomer], max_loaded_engines=1)

    await asyncio.wait_for(manager.ensure_loaded("newcomer"), timeout=3.0)

    assert newcomer.is_ready is True
    assert idle.is_ready is False
    assert manager.loaded_ids() == ["newcomer"]

# ─────────────────────────────────────────────────────────────
# F3: the idle reaper must not unload an engine a request is using
# ─────────────────────────────────────────────────────────────
#
# ``synthesize_async`` used to call ``ensure_loaded()`` *then* acquire the engine
# slot.  ``reap_idle()`` skips engines with ``in_flight > 0``, so in that window a
# request could be promised an engine that the reaper then unloaded, turning a
# valid request into a 503.  The slot is now taken first.

async def test_the_idle_reaper_cannot_unload_an_engine_mid_request(tmp_path):
    gate = threading.Event()
    now = [1000.0]
    engine = LifecycleEngine("kokoro", ready=True, gate=gate)
    other = LifecycleEngine("other", ready=True)
    manager = EngineManager(
        [engine, other],
        max_loaded_engines=2,
        idle_timeout_seconds=1.0,
        clock=lambda: now[0],
    )
    service = TTSService(
        manager=manager,
        cache=SynthesisCache(enabled=False, directory=tmp_path / "cache"),
        executor=InferenceExecutor(max_workers=2),
    )
    manager.touch("kokoro")
    manager.touch("other")

    task = asyncio.create_task(service.synthesize_async("Hold on.", cache=False))
    try:
        await _wait_for(lambda: engine.in_flight == 1)

        # Every engine is now far past the idle timeout *and* one is rendering.
        now[0] = 10_000.0
        unloaded = manager.reap_idle()

        assert "kokoro" not in unloaded, "the reaper unloaded an engine with work in flight"
        assert engine.is_ready is True
    finally:
        gate.set()
        result = await task

    assert result.audio


async def test_a_request_that_starts_after_the_grace_period_reloads_cleanly(tmp_path):
    """The engine may still be reaped when genuinely idle; the next request reloads it."""
    now = [0.0]
    engine = LifecycleEngine("kokoro", ready=True)
    other = LifecycleEngine("other", ready=True)
    manager = EngineManager(
        [engine, other], max_loaded_engines=2, idle_timeout_seconds=1.0, clock=lambda: now[0]
    )
    service = TTSService(
        manager=manager,
        cache=SynthesisCache(enabled=False, directory=tmp_path / "cache"),
        executor=InferenceExecutor(max_workers=2),
    )
    manager.touch("kokoro")
    manager.touch("other")

    now[0] = 100.0
    assert "kokoro" in manager.reap_idle()
    assert engine.is_ready is False

    result = await service.synthesize_async("Back again.", engine="kokoro", cache=False)

    assert result.audio
    assert engine.loads >= 1
    assert engine.is_ready is True


async def test_the_engine_slot_is_held_while_the_engine_is_loaded(tmp_path):
    """Structural proof of the ordering, independent of timing.

    ``ensure_loaded`` must be called while the request already holds the engine's
    concurrency slot.  Only then is ``in_flight > 0`` during the load, which is
    what makes ``reap_idle()`` refuse to unload the engine in that window.
    """
    engine = LifecycleEngine("kokoro", ready=True)
    manager = EngineManager([engine], max_loaded_engines=1)
    service = TTSService(
        manager=manager,
        cache=SynthesisCache(enabled=False, directory=tmp_path / "cache"),
        executor=InferenceExecutor(max_workers=2),
    )

    observed_in_flight: list[int] = []
    original = manager.ensure_loaded

    async def spy(engine_id: str):
        observed_in_flight.append(manager.limiter_snapshot(engine_id)["in_flight"])
        return await original(engine_id)

    manager.ensure_loaded = spy  # type: ignore[method-assign]

    await service.synthesize_async("Ordered.", engine="kokoro", cache=False)

    assert observed_in_flight == [1], (
        "ensure_loaded ran with in_flight="
        f"{observed_in_flight} while the request held no slot; the idle reaper "
        "can unload the engine between the load check and the render"
    )
