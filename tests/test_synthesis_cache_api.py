"""API-level tests for synthesis caching and the ``X-Cache`` contract.

These drive the real :class:`TTSService` through the real router with a
deterministic engine double, so the cache, the key builder, the direction
lowering and the response headers are all exercised together.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.contracts import AudioChunk, AudioResult, EngineCapabilities
from app.engines.manager import EngineManager
from app.routers import health, tts
from app.services.inference import InferenceExecutor
from app.services.synthesis_cache import SynthesisCache
from app.services.tts_service import TTSService

SAMPLE_RATE = 24000


class CountingEngine:
    """A deterministic engine that counts renders and can be slowed down."""

    id = "counter"
    name = "Counting engine"
    license = "MIT"

    def __init__(self, *, delay: float = 0.0, start_ready: bool = True) -> None:
        self.is_ready = start_ready
        self.delay = delay
        self.calls: list[dict] = []
        self.loads = 0
        self.unloads = 0

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
            supports_streaming=True,
            sample_rates=[SAMPLE_RATE],
            max_chars=10000,
        )

    def voices(self):
        return []

    def synthesize(self, req):
        if self.delay:
            time.sleep(self.delay)
        self.calls.append({"text": req.text, "voice": req.voice, "rate": req.direction.prosody.rate})
        audio = f"RIFF-{req.text}-{req.voice or 'default'}".encode()
        return AudioResult(
            audio=audio,
            duration=len(audio) / 1000.0,
            sample_rate=SAMPLE_RATE,
            engine=self.id,
            voice=req.voice,
        )

    def stream(self, req):
        yield AudioChunk(audio=b"RIFF", duration=0.1, sample_rate=SAMPLE_RATE)


def _service(tmp_path, engine: CountingEngine, *, cache_enabled=True) -> TTSService:
    return TTSService(
        manager=EngineManager([engine], max_loaded_engines=1),
        cache=SynthesisCache(
            enabled=cache_enabled,
            directory=tmp_path / "cache",
            ttl_seconds=3600.0,
        ),
        executor=InferenceExecutor(max_workers=2),
    )


def _client(service: TTSService) -> TestClient:
    app = FastAPI()
    app.state.tts_service = service
    app.state.stt_service = MagicMock(is_ready=True)
    app.include_router(health.router)
    app.include_router(tts.router, prefix="/api/tts")
    return TestClient(app)


def _body(**overrides) -> dict:
    payload = {"text": "Hello from the Reactory speech service.", "voice": "af_heart"}
    payload.update(overrides)
    return payload


# ─────────────────────────────────────────────────────────────
# Hit / miss
# ─────────────────────────────────────────────────────────────

def test_repeated_identical_request_hits_the_cache_and_returns_identical_bytes(tmp_path):
    engine = CountingEngine()
    client = _client(_service(tmp_path, engine))

    first = client.post("/api/tts/synthesize", json=_body())
    second = client.post("/api/tts/synthesize", json=_body())

    assert first.status_code == 200 and second.status_code == 200
    assert first.headers["x-cache"] == "miss"
    assert first.headers["x-cache-tier"] == "none"
    assert second.headers["x-cache"] == "hit"
    assert second.headers["x-cache-tier"] == "lru"
    assert second.content == first.content
    assert len(engine.calls) == 1


def test_a_cache_hit_is_measurably_faster(tmp_path):
    engine = CountingEngine(delay=0.15)
    client = _client(_service(tmp_path, engine))

    started = time.perf_counter()
    first = client.post("/api/tts/synthesize", json=_body())
    first_seconds = time.perf_counter() - started

    started = time.perf_counter()
    second = client.post("/api/tts/synthesize", json=_body())
    second_seconds = time.perf_counter() - started

    assert first.headers["x-cache"] == "miss"
    assert second.headers["x-cache"] == "hit"
    assert first.content == second.content
    assert second_seconds < first_seconds / 2, (first_seconds, second_seconds)


def test_the_json_endpoint_reports_cache_state_in_the_body_and_headers(tmp_path):
    client = _client(_service(tmp_path, CountingEngine()))

    client.post("/api/tts/synthesize/json", json=_body())
    second = client.post("/api/tts/synthesize/json", json=_body())

    assert second.headers["x-cache"] == "hit"
    payload = second.json()
    assert payload["cache"] == "hit"
    assert payload["cache_tier"] == "lru"


# ─────────────────────────────────────────────────────────────
# Cache identity at the API boundary
# ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "changed",
    [
        {"voice": "am_adam"},
        {"speed": 1.3},
        {"direction": {"prosody": {"rate": 0.8}}},
        {"is_phonemes": True, "text": "həˈloʊ"},
    ],
    ids=["voice", "speed", "direction", "phonemes"],
)
def test_a_changed_request_component_produces_a_miss(tmp_path, changed):
    engine = CountingEngine()
    client = _client(_service(tmp_path, engine))

    client.post("/api/tts/synthesize", json=_body())
    second = client.post("/api/tts/synthesize", json=_body(**changed))

    assert second.headers["x-cache"] == "miss"
    assert len(engine.calls) == 2


def test_a_changed_engine_produces_a_miss(tmp_path):
    first_engine = CountingEngine()
    other_engine = CountingEngine()
    other_engine.id = "other-engine"

    service = _service(tmp_path, first_engine)
    service.engine_manager.register(other_engine)
    client = _client(service)

    first = client.post("/api/tts/synthesize", json=_body())
    assert first.headers["x-cache"] == "miss"
    assert first.headers["x-engine"] == first_engine.id

    second = client.post("/api/tts/synthesize", json=_body(engine=other_engine.id))
    assert second.headers["x-cache"] == "miss"
    assert second.headers["x-engine"] == other_engine.id

    # The same request on the second engine now hits its own entry.
    third = client.post("/api/tts/synthesize", json=_body(engine=other_engine.id))
    assert third.headers["x-cache"] == "hit"
    assert len(first_engine.calls) == 1
    assert len(other_engine.calls) == 1


# ─────────────────────────────────────────────────────────────
# Disable switches
# ─────────────────────────────────────────────────────────────

def test_cache_can_be_disabled_per_request(tmp_path):
    engine = CountingEngine()
    client = _client(_service(tmp_path, engine))
    body = _body(cache=False)

    first = client.post("/api/tts/synthesize", json=body)
    second = client.post("/api/tts/synthesize", json=body)

    assert first.headers["x-cache"] == "miss"
    assert second.headers["x-cache"] == "miss"
    assert second.headers["x-cache-tier"] == "none"
    assert len(engine.calls) == 2


def test_cache_can_be_disabled_globally(tmp_path):
    engine = CountingEngine()
    client = _client(_service(tmp_path, engine, cache_enabled=False))

    client.post("/api/tts/synthesize", json=_body())
    second = client.post("/api/tts/synthesize", json=_body())

    assert second.headers["x-cache"] == "miss"
    assert len(engine.calls) == 2


# ─────────────────────────────────────────────────────────────
# A hit does not need the model
# ─────────────────────────────────────────────────────────────

def test_a_cache_hit_is_served_without_reloading_an_unloaded_engine(tmp_path):
    engine = CountingEngine(start_ready=False)
    service = _service(tmp_path, engine)
    client = _client(service)

    first = client.post("/api/tts/synthesize", json=_body())
    assert first.headers["x-cache"] == "miss"
    assert engine.loads == 1

    # Unload the model: the next request must be a hit *before* the engine is
    # loaded, so a repeated preview does not pay the model-load cost again.
    assert service.engine_manager.unload(engine.id)
    assert engine.is_ready is False

    second = client.post("/api/tts/synthesize", json=_body())

    assert second.headers["x-cache"] == "hit"
    assert second.content == first.content
    assert engine.loads == 1
    assert engine.unloads == 1

# ─────────────────────────────────────────────────────────────
# /phonemize runs G2P off the event loop
# ─────────────────────────────────────────────────────────────

def test_phonemize_endpoint_uses_the_inference_pool(tmp_path, monkeypatch):
    """The endpoint must run blocking G2P on a pool thread, not the event loop.

    Asserting only the response body would pass even if the executor branch were
    removed, so this records the thread the conversion actually ran on.
    """
    thread_names: list[str] = []

    class _ThreadRecordingConverter:
        is_available = True

        def convert_to_phonetic(self, text, language="en-us", custom_lexicon=None):
            thread_names.append(threading.current_thread().name)
            return "həlˈoʊ wˈɜːld"

        def normalize_text(self, text, custom_lexicon=None):
            return text

    monkeypatch.setattr(
        "app.services.tts_service.phonetic_converter", _ThreadRecordingConverter()
    )

    engine = CountingEngine()
    service = _service(tmp_path, engine)
    client = _client(service)

    response = client.post("/api/tts/phonemize", json={"text": "Hello world."})

    assert response.status_code == 200
    payload = response.json()
    assert payload["text"] == "Hello world."
    assert "phonemes" in payload and "normalized" in payload
    assert thread_names, "the converter was never called"
    assert all(name.startswith("tts-inference") for name in thread_names), (
        f"G2P ran on {thread_names}, not the inference pool"
    )


def test_phonemize_endpoint_still_works_without_an_executor():
    """A service double has no pool; the endpoint must degrade, not 500."""
    service = MagicMock()
    service.phonemize.return_value = {
        "text": "Hello",
        "normalized": "Hello",
        "phonemes": "həlˈoʊ",
        "language": "en-us",
    }
    del service.executor  # MagicMock would otherwise fabricate one

    app = FastAPI()
    app.state.tts_service = service
    app.state.stt_service = MagicMock(is_ready=True)
    app.include_router(tts.router, prefix="/api/tts")
    client = TestClient(app)

    response = client.post("/api/tts/phonemize", json={"text": "Hello"})

    assert response.status_code == 200
    assert response.json()["phonemes"] == "həlˈoʊ"