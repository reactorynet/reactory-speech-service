"""Tests for STT endpoints."""

import json
import io


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
