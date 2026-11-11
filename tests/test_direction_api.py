"""API-level tests for the direction contract and backwards compatibility.

These run against a real :class:`TTSService` wired to a deterministic engine
double, so the audio really is produced through the adapter, the registry and the
lowering path rather than through a mock.
"""

from __future__ import annotations

import base64
import json
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.core.contracts import AudioChunk, AudioResult, EngineCapabilities
from app.main import app as fastapi_app
from app.services.tts_service import TTSService
from app.services.voice_registry import KOKORO_SAMPLE_RATE, SHIPPED_VOICE_IDS


class FakeKokoro:
    """Deterministic engine double; sample count is inversely proportional to speed."""

    def __init__(self, voice_ids=SHIPPED_VOICE_IDS):
        self.voices = {voice_id: None for voice_id in voice_ids}
        self.calls: list[dict] = []

    def create(self, text, voice, speed=1.0, lang="en-us", is_phonemes=False):
        if voice not in self.voices:
            raise ValueError(f"Voice {voice} not found in available voices")
        if not 0.5 <= speed <= 2.0:
            raise ValueError(f"Speed should be between 0.5 and 2.0, got {speed}")
        self.calls.append({"text": text, "voice": voice, "speed": speed, "lang": lang})
        samples = np.zeros(max(1, round(2400 / speed)), dtype=np.float32)
        return samples, KOKORO_SAMPLE_RATE

    def stream(self, *args, **kwargs):  # pragma: no cover - not used here
        yield from ()


class StubStreamEngine:
    """A second, distinguishable engine used to prove routing end-to-end."""

    id = "stub-stream"
    name = "Stub streaming engine"
    license = "MIT"
    is_ready = True

    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            engine=self.id,
            languages=["en-US"],
            supports_streaming=True,
            sample_rates=[KOKORO_SAMPLE_RATE],
        )

    def voices(self):
        return []

    def synthesize(self, req):
        return AudioResult(
            audio=b"STUB-WAV", duration=0.25, sample_rate=KOKORO_SAMPLE_RATE, engine=self.id
        )

    def stream(self, req):
        yield AudioChunk(
            audio=b"STUB-WAV", duration=0.25, sample_rate=KOKORO_SAMPLE_RATE
        )


def _service() -> TTSService:
    service = TTSService()
    service._kokoro = FakeKokoro()
    service._ready = True
    return service


@pytest.fixture
def real_service() -> TTSService:
    return _service()


@pytest.fixture
def api(real_service) -> TestClient:
    fastapi_app.state.tts_service = real_service
    fastapi_app.state.stt_service = MagicMock(is_ready=False)
    client = TestClient(fastapi_app)
    yield client
    fastapi_app.state.tts_service = TTSService()


def _audio(response) -> bytes:
    return base64.b64decode(response.json()["audio_base64"])


MINIMAL_TEXT = "Hello from the Reactory speech service."


# ─────────────────────────────────────────────────────────────
# Legacy path — unchanged behaviour
# ─────────────────────────────────────────────────────────────

def test_legacy_request_is_byte_identical_to_the_previous_api(api, real_service):
    """No direction, no engine: the router must delegate to the old call verbatim."""
    response = api.post(
        "/api/tts/synthesize/json",
        json={"text": MINIMAL_TEXT, "voice": SHIPPED_VOICE_IDS[0], "phonetic_preprocess": False},
    )

    assert response.status_code == 200
    expected, expected_duration = real_service.synthesize(
        MINIMAL_TEXT, voice=SHIPPED_VOICE_IDS[0], phonetic_preprocess=False
    )

    assert _audio(response) == expected
    assert response.json()["duration"] == pytest.approx(round(expected_duration, 3))
    assert real_service._kokoro.calls[-1]["speed"] == 1.0
    assert real_service._kokoro.calls[-1]["voice"] == SHIPPED_VOICE_IDS[0]


def test_request_without_a_voice_uses_the_configured_default_voice(api, real_service, monkeypatch):
    """B1 regression: KOKORO_DEFAULT_VOICE still selects the default voice."""
    monkeypatch.setattr(settings, "kokoro_default_voice", "am_adam")
    real_service._kokoro.calls.clear()

    response = api.post("/api/tts/synthesize/json", json={"text": MINIMAL_TEXT})

    assert response.status_code == 200
    assert real_service._kokoro.calls[-1]["voice"] == "am_adam"


def test_legacy_request_reports_no_applied_or_ignored_direction(api):
    response = api.post("/api/tts/synthesize/json", json={"text": MINIMAL_TEXT})

    body = response.json()
    assert body["direction_applied"] == {}
    assert body["direction_ignored"] == []
    assert body["direction_warnings"] == []
    assert response.headers["x-direction-applied"] == "{}"
    assert response.headers["x-direction-ignored"] == ""


def test_legacy_speed_still_works_and_is_reported_as_rate(api, real_service):
    response = api.post(
        "/api/tts/synthesize/json",
        json={"text": MINIMAL_TEXT, "voice": SHIPPED_VOICE_IDS[0], "speed": 1.5},
    )

    assert response.status_code == 200
    assert response.json()["direction_applied"] == {"rate": 1.5}
    assert real_service._kokoro.calls[-1]["speed"] == 1.5


def test_legacy_optional_fields_are_still_accepted(api):
    response = api.post(
        "/api/tts/synthesize/json",
        json={
            "text": "həlˈoʊ",
            "voice": SHIPPED_VOICE_IDS[0],
            "is_phonemes": True,
            "phonetic_preprocess": False,
            "custom_lexicon": {"Reactory": "Ree-actory"},
        },
    )

    assert response.status_code == 200
    assert response.json()["audio_base64"]


# ─────────────────────────────────────────────────────────────
# Direction: rate applied, pitch/style ignored
# ─────────────────────────────────────────────────────────────

def test_direction_rate_changes_the_audio_and_is_applied(api, real_service):
    legacy = api.post(
        "/api/tts/synthesize/json",
        json={"text": MINIMAL_TEXT, "voice": SHIPPED_VOICE_IDS[0]},
    )
    directed = api.post(
        "/api/tts/synthesize/json",
        json={
            "text": MINIMAL_TEXT,
            "voice": SHIPPED_VOICE_IDS[0],
            "direction": {"prosody": {"rate": 0.9}},
        },
    )

    assert directed.status_code == 200
    assert directed.json()["direction_applied"] == {"rate": 0.9}
    assert directed.json()["direction_ignored"] == []
    assert _audio(directed) != _audio(legacy)
    assert real_service._kokoro.calls[-1]["speed"] == 0.9
    assert json.loads(directed.headers["x-direction-applied"]) == {"rate": 0.9}


def test_pitch_and_style_are_ignored_but_return_200_with_warnings(api):
    response = api.post(
        "/api/tts/synthesize/json",
        json={
            "text": MINIMAL_TEXT,
            "voice": SHIPPED_VOICE_IDS[0],
            "direction": {"style": "empathetic", "prosody": {"pitch": -2}},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert set(body["direction_ignored"]) == {"style", "pitch"}
    assert body["direction_applied"] == {}
    assert len(body["direction_warnings"]) >= 2
    warnings = " ".join(body["direction_warnings"]).lower()
    assert "empathetic" in warnings
    assert "pitch" in warnings

    assert set(response.headers["x-direction-ignored"].split(",")) == {"style", "pitch"}


def test_wav_response_carries_the_direction_headers(api):
    response = api.post(
        "/api/tts/synthesize",
        json={
            "text": MINIMAL_TEXT,
            "voice": SHIPPED_VOICE_IDS[0],
            "direction": {"style": "empathetic", "prosody": {"rate": 0.9, "pitch": -2}},
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.content.startswith(b"RIFF")
    assert json.loads(response.headers["x-direction-applied"]) == {"rate": 0.9}
    assert set(response.headers["x-direction-ignored"].split(",")) == {"style", "pitch"}
    assert response.headers["x-engine"] == "kokoro"
    assert "pitch" in response.headers["x-direction-warnings"].lower()


def test_explicit_intensity_is_ignored_when_the_style_cannot_be_applied(api):
    response = api.post(
        "/api/tts/synthesize/json",
        json={
            "text": MINIMAL_TEXT,
            "voice": SHIPPED_VOICE_IDS[0],
            "direction": {"style": "empathetic", "intensity": 0.8},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert set(body["direction_ignored"]) == {"style", "intensity"}


def test_unsupported_pause_shaping_is_ignored_not_faked(api):
    response = api.post(
        "/api/tts/synthesize/json",
        json={
            "text": MINIMAL_TEXT,
            "voice": SHIPPED_VOICE_IDS[0],
            "direction": {"prosody": {"pauses": "dramatic"}},
        },
    )

    assert response.status_code == 200
    assert "pauses" in response.json()["direction_ignored"]


# ─────────────────────────────────────────────────────────────
# speed vs direction precedence
# ─────────────────────────────────────────────────────────────

def test_direction_rate_wins_over_speed_and_the_response_says_so(api, real_service):
    response = api.post(
        "/api/tts/synthesize/json",
        json={
            "text": MINIMAL_TEXT,
            "voice": SHIPPED_VOICE_IDS[0],
            "speed": 1.5,
            "direction": {"prosody": {"rate": 0.9}},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["direction_applied"] == {"rate": 0.9}
    assert any("alias" in warning and "took precedence" in warning for warning in body["direction_warnings"])
    assert real_service._kokoro.calls[-1]["speed"] == 0.9


def test_speed_fills_in_when_direction_omits_rate(api, real_service):
    response = api.post(
        "/api/tts/synthesize/json",
        json={
            "text": MINIMAL_TEXT,
            "voice": SHIPPED_VOICE_IDS[0],
            "speed": 1.2,
            "direction": {"style": "neutral"},
        },
    )

    assert response.status_code == 200
    assert response.json()["direction_applied"] == {"rate": 1.2}
    assert real_service._kokoro.calls[-1]["speed"] == 1.2


# ─────────────────────────────────────────────────────────────
# Errors
# ─────────────────────────────────────────────────────────────

def test_unknown_voice_returns_422_pointing_at_the_voice_list(api):
    response = api.post(
        "/api/tts/synthesize", json={"text": MINIMAL_TEXT, "voice": "zz_unknown"}
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "zz_unknown" in detail
    assert "GET /api/tts/voices" in detail


def test_http_engine_override_routes_the_audio(api, real_service):
    """The HTTP override reaches the rendering engine, mirroring the WebSocket path."""
    real_service.engine_manager.register(StubStreamEngine())

    response = api.post(
        "/api/tts/synthesize",
        json={"text": MINIMAL_TEXT, "engine": "stub-stream"},
    )

    assert response.status_code == 200
    assert response.content == b"STUB-WAV"
    assert response.headers["x-engine"] == "stub-stream"
    assert real_service._kokoro.calls == []


def test_unknown_engine_override_returns_422_pointing_at_capabilities(api):
    response = api.post(
        "/api/tts/synthesize",
        json={"text": MINIMAL_TEXT, "engine": "chatterbox"},
    )

    assert response.status_code == 422
    assert "capabilities" in response.json()["detail"]


@pytest.mark.parametrize("rate", [0.4, 2.5])
def test_out_of_range_direction_rate_returns_422(api, rate):
    response = api.post(
        "/api/tts/synthesize/json",
        json={"text": MINIMAL_TEXT, "direction": {"prosody": {"rate": rate}}},
    )

    assert response.status_code == 422


# ─────────────────────────────────────────────────────────────
# Capabilities
# ─────────────────────────────────────────────────────────────

def test_capabilities_lists_every_registered_engine_with_the_full_matrix(api):
    response = api.get("/api/tts/capabilities")

    assert response.status_code == 200
    engines = response.json()["engines"]
    assert [engine["id"] for engine in engines] == ["kokoro"]

    kokoro = engines[0]
    for field in (
        "engine",
        "languages",
        "native_rate",
        "native_pitch",
        "native_volume",
        "styles",
        "supports_cloning",
        "supports_streaming",
        "supports_ssml",
        "sample_rates",
        "max_chars",
        "gpu_required",
    ):
        assert field in kokoro, field

    assert kokoro["engine"] == "kokoro"
    assert kokoro["native_rate"] is True
    assert kokoro["native_pitch"] is False
    assert kokoro["styles"] == ["neutral"]
    assert kokoro["gpu_required"] is False
    assert kokoro["voice_count"] == 54


# ─────────────────────────────────────────────────────────────
# WebSocket: per-message direction, unchanged audio frames
# ─────────────────────────────────────────────────────────────

def test_websocket_accepts_direction_and_reports_what_was_applied(api, real_service):
    with api.websocket_connect("/api/tts/stream") as ws:
        ws.send_text(
            json.dumps(
                {
                    "text": "Hello world.",
                    "voice": SHIPPED_VOICE_IDS[0],
                    "direction": {"style": "empathetic", "prosody": {"rate": 0.9}},
                }
            )
        )
        audio = ws.receive_bytes()
        done = ws.receive_json()

    assert audio.startswith(b"RIFF")
    assert done["done"] is True
    assert done["direction_applied"] == {"rate": 0.9}
    assert "style" in done["direction_ignored"]
    assert real_service._kokoro.calls[-1]["speed"] == 0.9


def test_websocket_rejects_an_unknown_engine_without_a_direction(api):
    """B2: an explicit engine is never silently dropped."""
    with api.websocket_connect("/api/tts/stream") as ws:
        ws.send_text(json.dumps({"text": "Hi.", "engine": "chatterbox"}))
        error = ws.receive_json()
        ws.send_text(json.dumps({"text": "Hi."}))
        audio = ws.receive_bytes()
        done = ws.receive_json()

    assert "chatterbox" in error["error"]
    assert "capabilities" in error["error"]
    assert audio.startswith(b"RIFF")
    assert done["done"] is True


def test_websocket_engine_override_routes_the_audio(api, real_service):
    """B2: the override must reach streaming synthesis, not just the plan."""
    real_service.engine_manager.register(StubStreamEngine())

    with api.websocket_connect("/api/tts/stream") as ws:
        ws.send_text(json.dumps({"text": "Hi.", "engine": "stub-stream"}))
        audio = ws.receive_bytes()
        done = ws.receive_json()

    assert audio == b"STUB-WAV"
    assert done["engine"] == "stub-stream"
    assert real_service._kokoro.calls == []


def test_websocket_engine_override_also_works_with_a_direction(api, real_service):
    real_service.engine_manager.register(StubStreamEngine())

    with api.websocket_connect("/api/tts/stream") as ws:
        ws.send_text(
            json.dumps(
                {
                    "text": "Hi.",
                    "engine": "stub-stream",
                    "direction": {"style": "empathetic"},
                }
            )
        )
        audio = ws.receive_bytes()
        done = ws.receive_json()

    assert audio == b"STUB-WAV"
    assert done["engine"] == "stub-stream"
    assert "style" in done["direction_ignored"]


def test_websocket_accepts_a_registered_engine_override(api, real_service):
    with api.websocket_connect("/api/tts/stream") as ws:
        ws.send_text(json.dumps({"text": "Hi.", "engine": "kokoro"}))
        audio = ws.receive_bytes()
        done = ws.receive_json()

    assert audio.startswith(b"RIFF")
    assert done["engine"] == "kokoro"
    assert real_service._kokoro.calls


def test_websocket_rejects_an_invalid_direction_without_closing(api):
    with api.websocket_connect("/api/tts/stream") as ws:
        ws.send_text(
            json.dumps({"text": "Hello.", "direction": {"prosody": {"rate": 9.0}}})
        )
        error = ws.receive_json()
        ws.send_text(json.dumps({"text": "Hello."}))
        audio = ws.receive_bytes()
        done = ws.receive_json()

    assert "error" in error
    assert audio.startswith(b"RIFF")
    assert done["done"] is True
