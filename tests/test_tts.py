"""Tests for TTS endpoints."""

import json


def test_synthesize_returns_wav(client, mock_tts_service):
    """POST /api/tts/synthesize returns WAV audio with correct headers."""
    response = client.post(
        "/api/tts/synthesize",
        json={"text": "Hello world"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.headers["x-audio-duration"] == "0.5"
    assert response.headers["x-audio-format"] == "wav"
    assert len(response.content) > 0
    mock_tts_service.synthesize.assert_called_once_with(
        "Hello world", voice=None, speed=None
    )


def test_synthesize_with_voice_and_speed(client, mock_tts_service):
    """Synthesis accepts voice and speed parameters."""
    response = client.post(
        "/api/tts/synthesize",
        json={"text": "Test", "voice": "am_adam", "speed": 1.5},
    )
    assert response.status_code == 200
    mock_tts_service.synthesize.assert_called_once_with(
        "Test", voice="am_adam", speed=1.5
    )


def test_synthesize_json_returns_base64(client, mock_tts_service):
    """POST /api/tts/synthesize/json returns base64 audio in JSON."""
    response = client.post(
        "/api/tts/synthesize/json",
        json={"text": "Hello"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "audio_base64" in data
    assert data["duration"] == 0.5
    assert data["format"] == "wav"
    assert data["sample_rate"] == 24000

    # Verify base64 is decodable
    import base64

    audio = base64.b64decode(data["audio_base64"])
    assert len(audio) > 0


def test_synthesize_empty_text(client):
    """Reject empty text input."""
    response = client.post(
        "/api/tts/synthesize",
        json={"text": ""},
    )
    assert response.status_code == 422


def test_synthesize_text_too_long(client):
    """Reject text exceeding 10000 characters."""
    response = client.post(
        "/api/tts/synthesize",
        json={"text": "x" * 10001},
    )
    assert response.status_code == 422


def test_synthesize_invalid_speed(client):
    """Reject speed outside valid range."""
    response = client.post(
        "/api/tts/synthesize",
        json={"text": "Test", "speed": 10.0},
    )
    assert response.status_code == 422


def test_synthesize_service_not_ready(client, mock_tts_service):
    """Return 503 when TTS service is not initialized."""
    mock_tts_service.is_ready = False
    response = client.post(
        "/api/tts/synthesize",
        json={"text": "Hello"},
    )
    assert response.status_code == 503


def test_synthesize_value_error(client, mock_tts_service):
    """Return 422 when synthesis raises ValueError (e.g., unknown voice)."""
    mock_tts_service.synthesize.side_effect = ValueError("Unknown voice 'bad'")
    response = client.post(
        "/api/tts/synthesize",
        json={"text": "Hello"},
    )
    assert response.status_code == 422


def test_tts_websocket_stream(client, mock_tts_service):
    """WebSocket TTS streams binary audio chunks then done message."""
    with client.websocket_connect("/api/tts/stream") as ws:
        ws.send_text(json.dumps({"text": "Hello world"}))

        # Should receive binary WAV chunk
        data = ws.receive_bytes()
        assert len(data) > 0

        # Should receive JSON done
        msg = ws.receive_json()
        assert msg["done"] is True
        assert "total_duration" in msg


def test_tts_websocket_empty_text(client, mock_tts_service):
    """WebSocket TTS returns error for empty text."""
    with client.websocket_connect("/api/tts/stream") as ws:
        ws.send_text(json.dumps({"text": ""}))
        msg = ws.receive_json()
        assert "error" in msg


def test_tts_websocket_invalid_json(client, mock_tts_service):
    """WebSocket TTS returns error for invalid JSON."""
    with client.websocket_connect("/api/tts/stream") as ws:
        ws.send_text("not json")
        msg = ws.receive_json()
        assert "error" in msg
