"""Concurrency smoke tests for the synthesis request path.

These tests exist because a code review cannot prove that inference left the
event loop.  They observe real overlap and real latency:

* ten simultaneous syntheses must not serialise behind one another;
* ``/health`` must answer while multi-second renders are in flight, within a
  stated budget (:data:`HEALTH_LATENCY_BUDGET_SECONDS`);
* a heartbeat coroutine must keep ticking while a render is in progress, which
  it cannot do if the loop is blocked;
* inference must run on worker threads, never on the event-loop thread;
* the per-engine limit must cap concurrency (2 for ``kokoro`` by default, 1 for
  an engine that is not listed).

The engine doubles block on a ``threading.Event`` or ``time.sleep`` — genuine
blocking calls, exactly like ONNX inference — so a regression back to
synchronous handling fails these tests rather than merely looking suspicious.
"""

from __future__ import annotations

import asyncio
import threading
import time
from unittest.mock import MagicMock

import httpx
import numpy as np
import pytest
from fastapi import FastAPI

from app.config import settings
from app.core.contracts import AudioChunk, AudioResult, EngineCapabilities
from app.engines.manager import EngineManager
from app.routers import health, tts
from app.services.inference import InferenceExecutor
from app.services.synthesis_cache import SynthesisCache
from app.services.tts_service import TTSService

SAMPLE_RATE = 24000

#: Stated budget for a health probe while synthesis is in flight.  The probe does
#: no inference and no locking, so it should be milliseconds; the budget is
#: deliberately loose enough for a busy CI machine but far below the render time.
HEALTH_LATENCY_BUDGET_SECONDS = 0.5

REQUESTS = 10
RENDER_SECONDS = 0.25


class BlockingEngine:
    """Blocking engine double: records overlap, thread identity and peak."""

    name = "Blocking engine"
    license = "MIT"

    def __init__(
        self,
        engine_id: str = "kokoro",
        *,
        gate: threading.Event | None = None,
        delay: float = 0.0,
    ) -> None:
        self.id = engine_id
        self.is_ready = True
        self.gate = gate
        self.delay = delay
        self.in_flight = 0
        self.peak = 0
        self.threads: set[str] = set()
        self._lock = threading.Lock()

    async def initialize(self) -> None:
        self.is_ready = True

    def unload(self) -> None:
        self.is_ready = False

    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            engine=self.id,
            languages=["en-US"],
            native_rate=True,
            sample_rates=[SAMPLE_RATE],
            max_chars=10000,
        )

    def voices(self):
        return []

    def synthesize(self, req):
        with self._lock:
            self.in_flight += 1
            self.peak = max(self.peak, self.in_flight)
            self.threads.add(threading.current_thread().name)
        try:
            if self.gate is not None:
                assert self.gate.wait(timeout=10), "the test never released the engine"
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


def _service(tmp_path, engine: BlockingEngine, *, workers: int = 4) -> TTSService:
    return TTSService(
        manager=EngineManager([engine], max_loaded_engines=1),
        cache=SynthesisCache(enabled=False, directory=tmp_path / "cache"),
        executor=InferenceExecutor(max_workers=workers),
    )


def _app(service: TTSService) -> FastAPI:
    app = FastAPI()
    app.state.tts_service = service
    app.state.stt_service = MagicMock(is_ready=True)
    app.include_router(health.router)
    app.include_router(tts.router, prefix="/api/tts")
    return app


async def _get_json(app: FastAPI, path: str) -> tuple[dict, float]:
    transport = httpx.ASGITransport(app=app)
    started = time.perf_counter()
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(path)
    elapsed = time.perf_counter() - started
    assert response.status_code == 200
    return response.json(), elapsed


async def _wait_for(predicate, *, timeout: float = 10.0, interval: float = 0.005):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(interval)
    raise AssertionError("condition was not met within the timeout")


@pytest.fixture(autouse=True)
def _kokoro_limit(monkeypatch):
    """Pin the documented per-engine limits for the duration of each test."""
    monkeypatch.setattr(
        settings, "speech_engine_max_concurrency", {"kokoro": 2, "default": 1}
    )


# ─────────────────────────────────────────────────────────────
# The core acceptance criteria
# ─────────────────────────────────────────────────────────────

async def test_health_responds_while_ten_syntheses_are_in_flight(tmp_path):
    """A health probe must succeed while renders are blocked in worker threads."""
    gate = threading.Event()
    engine = BlockingEngine(gate=gate)
    service = _service(tmp_path, engine)
    app = _app(service)

    tasks = [
        asyncio.create_task(service.synthesize_async(f"Sentence {index}.", cache=False))
        for index in range(REQUESTS)
    ]
    try:
        # Two renders occupy the kokoro slots; the rest queue behind them.
        await _wait_for(lambda: engine.peak == 2)
        await _wait_for(
            lambda: service.engine_manager.status()["kokoro"]["queue_depth"] == REQUESTS - 2
        )

        payload, latency = await _get_json(app, "/health")

        assert latency < HEALTH_LATENCY_BUDGET_SECONDS, latency
        assert payload["status"] == "ok"
        assert payload["tts"] is True
        assert payload["stt"] is True
        assert payload["engines"]["kokoro"]["in_flight"] == 2
        assert payload["engines"]["kokoro"]["queue_depth"] == REQUESTS - 2
        assert payload["queue_depth"] == REQUESTS - 2
    finally:
        gate.set()
        results = await asyncio.gather(*tasks)

    assert len(results) == REQUESTS
    assert all(result.cache == "miss" for result in results)
    assert engine.peak == 2, "the per-engine limit must cap concurrency at 2"
    assert engine.in_flight == 0


async def test_ten_concurrent_syntheses_overlap_instead_of_serialising(tmp_path):
    engine = BlockingEngine(delay=RENDER_SECONDS)
    service = _service(tmp_path, engine)

    started = time.perf_counter()
    results = await asyncio.gather(
        *(
            service.synthesize_async(f"Sentence {index}.", cache=False)
            for index in range(REQUESTS)
        )
    )
    elapsed = time.perf_counter() - started
    serial_seconds = REQUESTS * RENDER_SECONDS

    assert len(results) == REQUESTS
    assert engine.peak == 2, "exactly two renders should run at a time"
    assert elapsed < serial_seconds * 0.8, (
        f"{REQUESTS} renders took {elapsed:.2f}s; serial would be {serial_seconds:.2f}s"
    )


async def test_the_event_loop_keeps_ticking_during_a_synthesis(tmp_path):
    """Heartbeat proof: a blocked loop cannot run a 10 ms timer."""
    engine = BlockingEngine(delay=0.4)
    service = _service(tmp_path, engine)

    ticks: list[float] = []
    stop = asyncio.Event()

    async def heartbeat() -> None:
        while not stop.is_set():
            ticks.append(time.monotonic())
            await asyncio.sleep(0.01)

    ticker = asyncio.create_task(heartbeat())
    try:
        await service.synthesize_async("One long sentence.", cache=False)
    finally:
        stop.set()
        _ = await ticker

    gaps = [later - earlier for earlier, later in zip(ticks, ticks[1:], strict=False)]
    assert len(ticks) >= 10, f"only {len(ticks)} heartbeats during a 0.4s render"
    assert max(gaps) < 0.2, f"the loop stalled for {max(gaps):.3f}s"


async def test_inference_runs_on_worker_threads_not_the_event_loop(tmp_path):
    engine = BlockingEngine()
    service = _service(tmp_path, engine)

    await service.synthesize_async("Where did I run?", cache=False)

    assert engine.threads, "inference never ran"
    assert threading.main_thread().name not in engine.threads
    assert all(name.startswith("tts-inference") for name in engine.threads)


# ─────────────────────────────────────────────────────────────
# Per-engine limits are real and configurable
# ─────────────────────────────────────────────────────────────

async def test_a_higher_configured_limit_allows_more_overlap(tmp_path, monkeypatch):
    monkeypatch.setattr(
        settings, "speech_engine_max_concurrency", {"kokoro": 3, "default": 1}
    )
    gate = threading.Event()
    engine = BlockingEngine(gate=gate)
    service = _service(tmp_path, engine)

    tasks = [
        asyncio.create_task(service.synthesize_async(f"Sentence {index}.", cache=False))
        for index in range(6)
    ]
    try:
        await _wait_for(lambda: engine.peak == 3)
        payload, _ = await _get_json(_app(service), "/health")
        assert payload["engines"]["kokoro"]["max_concurrency"] == 3
        assert payload["engines"]["kokoro"]["in_flight"] == 3
    finally:
        gate.set()
        await asyncio.gather(*tasks)

    assert engine.peak == 3


async def test_an_unlisted_engine_defaults_to_one_concurrent_render(tmp_path):
    gate = threading.Event()
    engine = BlockingEngine("gpu-engine", gate=gate)
    service = _service(tmp_path, engine)

    tasks = [
        asyncio.create_task(service.synthesize_async(f"Sentence {index}.", cache=False))
        for index in range(4)
    ]
    try:
        await _wait_for(lambda: engine.peak == 1)
        await asyncio.sleep(0.05)
        assert engine.peak == 1
    finally:
        gate.set()
        await asyncio.gather(*tasks)

    assert engine.peak == 1

# ─────────────────────────────────────────────────────────────
# R1: disabling phonetic preprocessing must not serialise inference
# ─────────────────────────────────────────────────────────────
#
# ``kokoro_onnx.Kokoro.create(..., is_phonemes=False)`` phonemises *and* runs the
# ONNX session inside one Python call.  When that call was wrapped in G2P_LOCK,
# a caller sending `"phonetic_preprocess": false` (or a deployment with
# PHONETIC_PROCESSING_ENABLED=false) collapsed engine concurrency to 1: four
# 0.2 s renders took 0.81 s instead of 0.21 s.  These tests drive the real
# KokoroEngine through the real service path to prove the limit still holds.

class _TokenisingKokoro:
    """A ``kokoro_onnx.Kokoro`` stand-in with the two steps it fuses.

    ``tokenizer.phonemize`` is the espeak-ng call; ``create`` is the ONNX session.
    Both record how many threads were inside at once.
    """

    voices = {"af_heart": None}

    def __init__(self, delay: float = 0.2) -> None:
        self.delay = delay
        self.tokenizer = self
        self.renders = 0
        self.unsplit = 0
        self._guard = threading.Lock()

    def phonemize(self, text: str, lang: str = "en-us") -> str:
        return f"phonemes-for:{text}"

    def create(self, text, voice, speed=1.0, lang="en-us", is_phonemes=False):
        with self._guard:
            if is_phonemes:
                self.renders += 1
            else:
                self.unsplit += 1
        time.sleep(self.delay)
        return np.zeros(24, dtype=np.float32), 24000


async def _run_preprocessing_disabled_concurrently(tmp_path, flag):
    """Gather 4 real-engine renders with preprocessing off; return the peak."""
    from app.engines.kokoro import KokoroEngine

    engine = KokoroEngine()
    engine.model = _TokenisingKokoro(delay=0.2)
    engine.set_ready(True)

    service = TTSService(
        manager=EngineManager([engine], max_loaded_engines=1),
        cache=SynthesisCache(enabled=False, directory=tmp_path / "cache"),
        executor=InferenceExecutor(max_workers=4),
    )

    started = time.perf_counter()
    await asyncio.gather(
        *(
            service.synthesize_async(
                f"Sentence number {index}.",
                voice="af_heart",
                phonetic_preprocess=flag,
                cache=False,
            )
            for index in range(4)
        )
    )
    elapsed = time.perf_counter() - started
    return engine.model, elapsed


@pytest.mark.parametrize("flag", [False], ids=["phonetic_preprocess_false"])
async def test_preprocessing_disabled_still_overlaps_inference(tmp_path, flag):
    """The configured kokoro limit must be reached with preprocessing disabled."""
    model, elapsed = await _run_preprocessing_disabled_concurrently(tmp_path, flag)

    assert model.unsplit == 0, "create() was called with is_phonemes=False"
    assert model.renders == 4
    # 4 renders of 0.2 s at concurrency 2 takes ~0.4 s; serial would be ~0.8 s.
    assert elapsed < 0.7, f"renders serialised: 4 x 0.2 s took {elapsed:.2f}s"


async def test_preprocessing_disabled_engine_peak_matches_the_configured_limit(
    tmp_path, monkeypatch
):
    """Peak concurrency inside the ONNX call equals the kokoro limit (2)."""
    from app.engines.kokoro import KokoroEngine

    monkeypatch.setattr(
        settings, "speech_engine_max_concurrency", {"kokoro": 2, "default": 1}
    )

    class _PeakKokoro(_TokenisingKokoro):
        def __init__(self) -> None:
            super().__init__(delay=0.2)
            self.active = 0
            self.peak = 0

        def create(self, text, voice, speed=1.0, lang="en-us", is_phonemes=False):
            with self._guard:
                self.active += 1
                self.peak = max(self.peak, self.active)
                if is_phonemes:
                    self.renders += 1
                else:
                    self.unsplit += 1
            try:
                time.sleep(self.delay)
                return np.zeros(24, dtype=np.float32), 24000
            finally:
                with self._guard:
                    self.active -= 1

    model = _PeakKokoro()
    engine = KokoroEngine()
    engine.model = model
    engine.set_ready(True)

    service = TTSService(
        manager=EngineManager([engine], max_loaded_engines=1),
        cache=SynthesisCache(enabled=False, directory=tmp_path / "cache"),
        executor=InferenceExecutor(max_workers=4),
    )

    await asyncio.gather(
        *(
            service.synthesize_async(
                f"Sentence number {index}.",
                voice="af_heart",
                phonetic_preprocess=False,
                cache=False,
            )
            for index in range(4)
        )
    )

    assert model.unsplit == 0
    assert model.peak == 2, (
        f"engine peak concurrency was {model.peak}, expected the configured limit of 2"
    )


async def test_preprocessing_disabled_does_not_starve_the_pool(tmp_path):
    """Lock waiters used to occupy worker threads; they must not any more.

    With the lock held across ``create``, four renders at limit 2 occupied four
    pool threads (two running, two blocked on the lock), leaving none for a
    G2P-only call.  Now the surplus renders wait on the *semaphore*, not a
    thread, so pool threads stay available.

    The probe deliberately does no espeak work: the first espeak call on a fresh
    thread costs ~1.5 s of initialisation, which would measure warm-up rather
    than the property under test.
    """
    from app.engines.kokoro import KokoroEngine

    gate = threading.Event()
    entered = []
    entered_guard = threading.Lock()

    class _GatedKokoro(_TokenisingKokoro):
        def create(self, text, voice, speed=1.0, lang="en-us", is_phonemes=False):
            with entered_guard:
                entered.append(text)
            assert gate.wait(timeout=10), "the test never released the engine"
            return np.zeros(24, dtype=np.float32), 24000

    engine = KokoroEngine()
    engine.model = _GatedKokoro(delay=0.0)
    engine.set_ready(True)

    service = TTSService(
        manager=EngineManager([engine], max_loaded_engines=1),
        cache=SynthesisCache(enabled=False, directory=tmp_path / "cache"),
        executor=InferenceExecutor(max_workers=4),
    )

    renders = [
        asyncio.create_task(
            service.synthesize_async(
                f"Sentence number {index}.",
                voice="af_heart",
                phonetic_preprocess=False,
                cache=False,
            )
        )
        for index in range(4)
    ]
    try:
        # Both kokoro slots are now blocked inside create().  Poll rather than
        # blocking on a threading primitive: this runs on the event loop.
        await _wait_for(lambda: len(entered) == 2)

        started = time.perf_counter()
        result = await asyncio.wait_for(
            service.executor.run(lambda: "pool-thread-available"), timeout=2.0
        )
        waited = time.perf_counter() - started
    finally:
        gate.set()
        await asyncio.gather(*renders)

    assert result == "pool-thread-available"
    assert waited < 0.5, (
        f"a pool thread took {waited:.2f}s while 2 renders were blocked; the "
        "surplus renders are holding threads instead of waiting on the semaphore"
    )
# ─────────────────────────────────────────────────────────────
# N3: the acceptance criterion says "several seconds"
# ─────────────────────────────────────────────────────────────
#
# The tests above prove the mechanism with sub-second renders, which is what makes
# them fast and deterministic.  This one closes the literal wording gap: a render
# is held for several seconds of wall clock and /health must answer inside the
# budget throughout.

SEVERAL_SECONDS = 3.0


async def test_health_answers_during_a_multi_second_synthesis(tmp_path):
    gate = threading.Event()
    engine = BlockingEngine(gate=gate)
    service = _service(tmp_path, engine)
    app = _app(service)

    task = asyncio.create_task(service.synthesize_async("A long sentence.", cache=False))
    try:
        await _wait_for(lambda: engine.in_flight == 1)

        # Probe repeatedly across the hold: every probe must stay inside budget.
        latencies: list[float] = []
        started = time.perf_counter()
        while time.perf_counter() - started < SEVERAL_SECONDS:
            payload, latency = await _get_json(app, "/health")
            latencies.append(latency)
            assert payload["tts"] is True
            assert payload["engines"]["kokoro"]["in_flight"] == 1
            await asyncio.sleep(0.25)

        held = time.perf_counter() - started
    finally:
        gate.set()
        _ = await task

    assert held >= SEVERAL_SECONDS - 0.1, f"the render was only held for {held:.2f}s"
    assert len(latencies) >= 8, f"only {len(latencies)} probes in {held:.2f}s"
    assert max(latencies) < HEALTH_LATENCY_BUDGET_SECONDS, (
        f"slowest health probe was {max(latencies):.3f}s during a "
        f"{held:.1f}s synthesis"
    )