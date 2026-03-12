"""Tests for health and readiness endpoints."""


def test_health_all_ready(client):
    """Health endpoint returns ok when both services are ready."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["tts"] is True
    assert data["stt"] is True
    assert "version" in data


def test_health_tts_not_ready(client, mock_tts_service):
    """Health reports TTS unavailable when model not loaded."""
    mock_tts_service.is_ready = False
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["tts"] is False
    assert data["stt"] is True


def test_health_stt_not_ready(client, mock_stt_service):
    """Health reports STT unavailable when model not loaded."""
    mock_stt_service.is_ready = False
    response = client.get("/health")
    data = response.json()
    assert data["tts"] is True
    assert data["stt"] is False


def test_ready_all_ok(client):
    """Readiness returns true when all models loaded."""
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["ready"] is True


def test_ready_not_ok(client, mock_tts_service):
    """Readiness returns false when a model is not loaded."""
    mock_tts_service.is_ready = False
    response = client.get("/ready")
    assert response.json()["ready"] is False
