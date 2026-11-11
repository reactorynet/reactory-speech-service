"""Tests for STT endpoints (Phase 7 / SPEECH-008)."""

import io
import json
import tempfile
import threading
import time
from pathlib import Path

import pytest

from tests.stt_fakes import (
    FakeModel,
    FakeSegment,
    FakeWord,
    ScriptedSTTService,
    build_service,
)

PCM_SECOND = b"\x00\x00" * 16000  # one second of 16 kHz mono s16le


@pytest.fixture
def temp_dir(tmp_path, monkeypatch):
    """Point ``tempfile`` at an empty directory and return it."""
    directory = tmp_path / "tempdir"
    directory.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(directory))
    return directory


# ----------------------------------------------------------------------
# REST
# ----------------------------------------------------------------------
def test_transcribe_returns_text(client, mock_stt_service, sample_wav_bytes):
    """POST /api/stt/transcribe returns transcription result."""
    response = client.post(
        "/api/stt/transcribe",
        files={"file": ("test.wav", io.BytesIO(sample_wav_bytes), "audio/wav")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["text"] == "Hello world"
    assert data["language"] == "en"
    assert len(data["segments"]) == 1
    assert data["segments"][0]["text"] == "Hello world"
    assert data["duration"] == 1.5
    # Word timestamps stay off unless requested, so the default payload stays small.
    assert "words" not in data["segments"][0]


def test_transcribe_with_language(client, mock_stt_service, sample_wav_bytes):
    """Transcription accepts a language parameter."""
    response = client.post(
        "/api/stt/transcribe",
        files={"file": ("test.wav", io.BytesIO(sample_wav_bytes), "audio/wav")},
        data={"language": "en"},
    )
    assert response.status_code == 200
    mock_stt_service.transcribe.assert_called_once()
    call_args = mock_stt_service.transcribe.call_args
    assert call_args[1].get("language") == "en" or call_args[0][1] == "en"


def test_transcribe_empty_file(client, mock_stt_service):
    """Reject empty audio file."""
    response = client.post(
        "/api/stt/transcribe",
        files={"file": ("empty.wav", io.BytesIO(b""), "audio/wav")},
    )
    assert response.status_code == 422


def test_transcribe_file_too_large(client, mock_stt_service):
    """Reject audio files exceeding 25MB."""
    large_data = b"\x00" * (26 * 1024 * 1024)
    response = client.post(
        "/api/stt/transcribe",
        files={"file": ("large.wav", io.BytesIO(large_data), "audio/wav")},
    )
    assert response.status_code == 413


def test_transcribe_service_not_ready(client, mock_stt_service, sample_wav_bytes):
    """Return 503 when STT service is not initialized."""
    mock_stt_service.is_ready = False
    response = client.post(
        "/api/stt/transcribe",
        files={"file": ("test.wav", io.BytesIO(sample_wav_bytes), "audio/wav")},
    )
    assert response.status_code == 503


def test_transcribe_returns_words_when_requested(client, app, sample_wav_bytes, temp_dir):
    """``word_timestamps=true`` returns words with start/end/probability."""
    words = [FakeWord("Hello", 0.0, 0.4, 0.93), FakeWord("world", 0.4, 1.0, 0.88)]
    app.state.stt_service = build_service(
        FakeModel(segments=[FakeSegment(0.0, 1.0, "Hello world", words)])
    )

    response = client.post(
        "/api/stt/transcribe",
        files={"file": ("test.wav", io.BytesIO(sample_wav_bytes), "audio/wav")},
        data={"word_timestamps": "true"},
    )

    assert response.status_code == 200
    segment_words = response.json()["segments"][0]["words"]
    assert [word["word"] for word in segment_words] == ["Hello", "world"]
    assert segment_words[0]["start"] == 0.0
    assert segment_words[0]["probability"] == 0.93
    assert list(temp_dir.iterdir()) == []


def test_transcribe_unknown_model_returns_400_with_available_models(
    client, app, sample_wav_bytes, temp_dir
):
    """An unmounted model id is a clear 4xx listing what is available."""
    app.state.stt_service = build_service(FakeModel(), model_id="base")

    response = client.post(
        "/api/stt/transcribe",
        files={"file": ("test.wav", io.BytesIO(sample_wav_bytes), "audio/wav")},
        data={"model": "distil-large-v3"},
    )

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail["error"] == "model_not_available"
    assert detail["model"] == "distil-large-v3"
    assert detail["available_models"] == ["base"]
    assert list(temp_dir.iterdir()) == []


def test_transcribe_selects_requested_model(client, app, sample_wav_bytes, temp_dir):
    """A mounted model can be selected per request."""
    service = build_service(FakeModel(), model_id="base")
    tiny = FakeModel(segments=[FakeSegment(0.0, 1.0, "tiny")])
    service._paths["tiny"] = Path("/models/tiny")
    service._models["tiny"] = tiny
    app.state.stt_service = service

    response = client.post(
        "/api/stt/transcribe",
        files={"file": ("test.wav", io.BytesIO(sample_wav_bytes), "audio/wav")},
        data={"model": "tiny"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["model"] == "tiny"
    assert payload["text"] == "tiny"
    assert list(temp_dir.iterdir()) == []


def test_transcribe_returns_language_confidence(client, app, sample_wav_bytes):
    """Detected language and confidence are part of the response."""
    from tests.stt_fakes import FakeInfo

    info = FakeInfo(
        language="en",
        language_probability=0.87,
        all_language_probs=[("en", 0.87), ("nl", 0.08)],
    )
    app.state.stt_service = build_service(FakeModel(info=info))

    response = client.post(
        "/api/stt/transcribe",
        files={"file": ("test.wav", io.BytesIO(sample_wav_bytes), "audio/wav")},
    )

    payload = response.json()
    assert payload["language"] == "en"
    assert payload["language_confidence"] == 0.87
    assert payload["needs_language_confirmation"] is False
    assert payload["language_probabilities"][0] == {"language": "en", "probability": 0.87}


def test_capabilities_lists_models_and_features(client, app):
    """GET /api/stt/capabilities reports mounted models and enabled features."""
    service = build_service(FakeModel(), model_id="base")
    service._paths["tiny"] = Path("/models/tiny")
    app.state.stt_service = service

    response = client.get("/api/stt/capabilities")

    assert response.status_code == 200
    payload = response.json()
    assert payload["available_models"] == ["base", "tiny"]
    assert payload["loaded_models"] == ["base"]
    assert payload["default_model"] == "base"
    assert payload["streaming"] is True
    assert payload["word_timestamps"] is True
    assert payload["language_confidence"] is True
    assert payload["vad"]["enabled"] is True


# ----------------------------------------------------------------------
# WebSocket
# ----------------------------------------------------------------------
def test_stt_websocket_stream(client, mock_stt_service, sample_wav_bytes):
    """WebSocket STT processes audio chunks and returns segments."""
    with client.websocket_connect("/api/stt/stream") as ws:
        # Optional config
        ws.send_text(json.dumps({"language": "en"}))
        config_msg = ws.receive_json()
        assert config_msg["status"] == "configured"

        # Send audio bytes
        ws.send_bytes(sample_wav_bytes)

        # Signal done
        ws.send_text(json.dumps({"done": True}))

        # Receive segment(s)
        segment = ws.receive_json()
        assert "text" in segment
        assert segment["text"] == "Hello world"

        # Receive done
        done_msg = ws.receive_json()
        assert done_msg["done"] is True


def test_stt_websocket_no_audio(client, mock_stt_service):
    """WebSocket STT returns error when no audio sent before done."""
    with client.websocket_connect("/api/stt/stream") as ws:
        ws.send_text(json.dumps({"done": True}))
        msg = ws.receive_json()
        assert "error" in msg


def test_stt_websocket_emits_partial_before_done(client, app):
    """A partial segment arrives before the client has finished sending audio."""
    app.state.stt_service = ScriptedSTTService()

    with client.websocket_connect("/api/stt/stream") as ws:
        ws.send_text(
            json.dumps(
                {"encoding": "pcm_s16le", "sample_rate": 16000, "min_window_seconds": 2.0}
            )
        )
        configured = ws.receive_json()
        assert configured["status"] == "configured"
        assert configured["encoding"] == "pcm_s16le"

        ws.send_bytes(PCM_SECOND)
        ws.send_bytes(PCM_SECOND)

        # Two seconds have been buffered and the client has NOT sent {"done": true}
        # yet, so this segment is a genuine partial.
        partial = ws.receive_json()
        assert set(partial) >= {"text", "start", "end", "partial_text", "language"}
        assert partial["is_partial"] is True
        assert partial["text"] == "word0"
        assert partial["partial_text"] == "word0"
        assert partial["language"] == "en"

        ws.send_text(json.dumps({"done": True}))

        final = ws.receive_json()
        assert final["is_partial"] is False
        assert final["text"] == "word1"

        done = ws.receive_json()
        assert done["done"] is True
        assert done["text"] == "word0 word1"
        assert done["language"] == "en"


def test_stt_websocket_does_not_duplicate_partial_segments(client, app):
    """Each segment is emitted once, as a partial and never again on done."""
    app.state.stt_service = ScriptedSTTService()

    with client.websocket_connect("/api/stt/stream") as ws:
        ws.send_text(json.dumps({"encoding": "pcm_s16le", "min_window_seconds": 1.0}))
        ws.receive_json()

        # 3 s of audio: the first window has no stable segment (the whole segment
        # sits inside the stability margin), then one new segment per second.
        for _ in range(3):
            ws.send_bytes(PCM_SECOND)

        first = ws.receive_json()
        second = ws.receive_json()
        assert [first["text"], second["text"]] == ["word0", "word1"]
        assert [first["end"], second["end"]] == [0.9, 1.8]

        ws.send_text(json.dumps({"done": True}))
        final = ws.receive_json()
        done = ws.receive_json()

        assert final["text"] == "word2"
        assert final["is_partial"] is False
        assert done["done"] is True
        assert done["text"] == "word0 word1 word2"


def test_stt_websocket_unknown_model_reports_available_models(client, app):
    """A bad model id on the stream is reported without killing the service."""
    app.state.stt_service = build_service(FakeModel(), model_id="base")

    with client.websocket_connect("/api/stt/stream") as ws:
        ws.send_text(json.dumps({"model": "medium"}))
        configured = ws.receive_json()
        assert configured["model"] == "medium"

        ws.send_bytes(PCM_SECOND)
        ws.send_text(json.dumps({"done": True}))

        error = ws.receive_json()
        assert error["error"] == "model_not_available"
        assert error["available_models"] == ["base"]


def test_stt_websocket_invalid_config(client, mock_stt_service):
    """An unknown encoding is rejected with a clear error."""
    with client.websocket_connect("/api/stt/stream") as ws:
        ws.send_text(json.dumps({"encoding": "flac-in-a-can"}))
        msg = ws.receive_json()
        assert msg["error"] == "Invalid configuration"
        assert "encoding" in msg["detail"].lower()


def test_websocket_disconnect_leaves_no_temp_files(client, app, temp_dir):
    """A client that disconnects mid-transcription leaves no temp file behind."""
    gate = threading.Event()
    model = FakeModel(
        segments=[FakeSegment(0.0, 0.9, "hello")],
        gate=gate,
    )
    app.state.stt_service = build_service(model)

    with client.websocket_connect("/api/stt/stream") as ws:
        ws.send_text(json.dumps({"encoding": "pcm_s16le", "min_window_seconds": 1.0}))
        assert ws.receive_json()["status"] == "configured"

        ws.send_bytes(PCM_SECOND)
        assert model.started.wait(10), "the server never started transcribing"

        # The in-flight request owns a temp file at this point.
        assert list(temp_dir.iterdir()) != []

        # Client goes away while the decode is still running.
        ws.close()
        gate.set()

    deadline = time.time() + 10
    while time.time() < deadline and list(temp_dir.iterdir()):
        time.sleep(0.02)

    assert list(temp_dir.iterdir()) == []


# ----------------------------------------------------------------------
# RC-1: a failing VAD must not be reported as silence
# ----------------------------------------------------------------------
def _detector_with_broken_silero(monkeypatch):
    """The real detector, with only the Silero call broken.

    Everything the bug hid behind — ``detect_regions`` deciding between "unknown"
    and "silence", ``trim_samples`` building the TrimResult, and the service
    branch that either short-circuits or transcribes — stays production code. Only
    the container decode is replaced, because PyAV cannot decode in this venv.
    """
    import faster_whisper.vad as fw_vad
    import numpy as np

    from app.services.vad import VoiceActivityDetector

    def boom(*args, **kwargs):
        raise RuntimeError("onnxruntime exploded")

    monkeypatch.setattr(fw_vad, "get_speech_timestamps", boom)
    detector = VoiceActivityDetector()
    samples = np.zeros(16000, dtype=np.float32)
    detector.trim_file = lambda _path: detector.trim_samples(samples)
    return detector


def test_failing_detector_still_transcribes_via_rest(
    client, app, sample_wav_bytes, temp_dir, monkeypatch
):
    """Detector failure falls back to untrimmed transcription (HTTP path)."""
    model = FakeModel(segments=[FakeSegment(0.0, 1.0, "Hello world")])
    app.state.stt_service = build_service(model, vad=_detector_with_broken_silero(monkeypatch))

    response = client.post(
        "/api/stt/transcribe",
        files={"file": ("test.wav", io.BytesIO(sample_wav_bytes), "audio/wav")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["text"] == "Hello world", "the model must still be called"
    assert len(payload["segments"]) == 1
    assert payload["vad_applied"] is False
    assert payload["vad_failed"] is True
    assert len(model.calls) == 1
    assert list(temp_dir.iterdir()) == []


# ----------------------------------------------------------------------
# RC-3: WHISPER_WORD_TIMESTAMPS is the default; the form field wins
# ----------------------------------------------------------------------
def test_rest_honours_word_timestamp_setting_when_field_omitted(
    client, app, sample_wav_bytes, monkeypatch
):
    """Omitting ``word_timestamps`` uses the deployment setting."""
    from app.config import settings

    monkeypatch.setattr(settings, "whisper_word_timestamps", True)
    model = FakeModel(segments=[FakeSegment(0.0, 1.0, "Hello", [FakeWord("Hello", 0.0, 1.0)])])
    app.state.stt_service = build_service(model)

    response = client.post(
        "/api/stt/transcribe",
        files={"file": ("test.wav", io.BytesIO(sample_wav_bytes), "audio/wav")},
    )

    assert response.status_code == 200
    assert model.calls[0]["kwargs"]["word_timestamps"] is True
    assert response.json()["segments"][0]["words"][0]["word"] == "Hello"


def test_rest_form_field_overrides_word_timestamp_setting(
    client, app, sample_wav_bytes, monkeypatch
):
    """An explicit ``word_timestamps=false`` beats the deployment setting."""
    from app.config import settings

    monkeypatch.setattr(settings, "whisper_word_timestamps", True)
    model = FakeModel(segments=[FakeSegment(0.0, 1.0, "Hello")])
    app.state.stt_service = build_service(model)

    response = client.post(
        "/api/stt/transcribe",
        files={"file": ("test.wav", io.BytesIO(sample_wav_bytes), "audio/wav")},
        data={"word_timestamps": "false"},
    )

    assert response.status_code == 200
    assert model.calls[0]["kwargs"]["word_timestamps"] is False
    assert "words" not in response.json()["segments"][0]


def test_rest_word_timestamps_default_off(client, app, sample_wav_bytes):
    """With the setting off and the field omitted, no words are returned."""
    model = FakeModel(segments=[FakeSegment(0.0, 1.0, "Hello")])
    app.state.stt_service = build_service(model)

    response = client.post(
        "/api/stt/transcribe",
        files={"file": ("test.wav", io.BytesIO(sample_wav_bytes), "audio/wav")},
    )

    assert response.status_code == 200
    assert model.calls[0]["kwargs"]["word_timestamps"] is False


# ----------------------------------------------------------------------
# Streaming: re-segmentation drift must not duplicate words (F3)
# ----------------------------------------------------------------------
def test_streaming_resegmentation_does_not_duplicate_words(client, app):
    """A boundary that moves outwards supersedes the earlier partial.

    The first partial says "hello"; the next pass returns the same region as
    "hello world". The transcript must read "hello world", not "hello hello world".
    """
    from tests.stt_fakes import DriftingSTTService

    app.state.stt_service = DriftingSTTService()

    with client.websocket_connect("/api/stt/stream") as ws:
        ws.send_text(json.dumps({"encoding": "pcm_s16le", "min_window_seconds": 1.0}))
        assert ws.receive_json()["status"] == "configured"

        # 1 s: too short to emit (stability margin withholds the tail).
        ws.send_bytes(PCM_SECOND)
        # 2 s: first stable partial.
        ws.send_bytes(PCM_SECOND)
        first = ws.receive_json()
        assert first["text"] == "hello"
        assert first["partial_text"] == "hello"

        # 3 s: re-decode moves the boundary; the segment must supersede, not append.
        ws.send_bytes(PCM_SECOND)
        second = ws.receive_json()
        assert second["text"] == "hello world"
        assert second["partial_text"] == "hello world"
        assert second["end"] == 1.2

        # 4 s: genuinely new speech appends.
        ws.send_bytes(PCM_SECOND)
        third = ws.receive_json()
        assert third["text"] == "again"
        assert third["partial_text"] == "hello world again"

        ws.send_text(json.dumps({"done": True}))
        done = ws.receive_json()

        assert done["done"] is True
        assert done["text"] == "hello world again"
        assert "hello hello" not in done["text"]


# ----------------------------------------------------------------------
# Capabilities / discovery (F6c, F6d)
# ----------------------------------------------------------------------
def test_capabilities_sees_a_model_mounted_after_boot(client, app, tmp_path, monkeypatch):
    """A model directory created after startup appears in capabilities."""
    from app.config import settings

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    monkeypatch.setattr(settings, "models_dir", str(models_dir))
    monkeypatch.setattr(settings, "whisper_model_path", None)

    service = build_service(FakeModel(), model_id="base")
    app.state.stt_service = service
    assert "tiny" not in service.available_models

    late = models_dir / "whisper-tiny"
    late.mkdir()
    (late / "model.bin").write_bytes(b"x")

    payload = client.get("/api/stt/capabilities").json()

    assert "tiny" in payload["available_models"]
    assert "base" in payload["available_models"]


def test_stream_capacity_reports_the_duration_limit(client, app, monkeypatch):
    """Hit the duration cap: the error names seconds, not the byte cap (F6f)."""
    from app.config import settings

    monkeypatch.setattr(settings, "whisper_stream_max_buffer_seconds", 0.5)
    app.state.stt_service = ScriptedSTTService()

    with client.websocket_connect("/api/stt/stream") as ws:
        ws.send_text(json.dumps({"encoding": "pcm_s16le", "min_window_seconds": 10}))
        assert ws.receive_json()["status"] == "configured"

        ws.send_bytes(PCM_SECOND)

        error = ws.receive_json()
        assert error["limit"] == "seconds"
        assert error["limit_seconds"] == 0.5
        assert "limit_bytes" not in error
        assert error["duration_seconds"] == pytest.approx(1.0)
