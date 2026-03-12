"""Shared test fixtures for the Reactory Speech Service."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np

from fastapi.testclient import TestClient


@pytest.fixture
def mock_tts_service():
    """A mock TTSService that returns predictable audio."""
    service = MagicMock()
    service.is_ready = True
    # Generate 0.5s of silence as WAV bytes
    sample_rate = 24000
    samples = np.zeros(sample_rate // 2, dtype=np.float32)

    import io
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV", subtype="PCM_16")
    buf.seek(0)
    fake_wav = buf.read()

    service.synthesize.return_value = (fake_wav, 0.5)
    service.synthesize_streaming.return_value = iter([(fake_wav, 0.5)])
    service.get_voices.return_value = [
        {"id": "af_heart", "name": "Heart (Female)", "language": "en-us"},
        {"id": "am_adam", "name": "Adam (Male)", "language": "en-us"},
    ]
    return service


@pytest.fixture
def mock_stt_service():
    """A mock STTService that returns predictable transcription."""
    service = MagicMock()
    service.is_ready = True
    service.transcribe.return_value = {
        "text": "Hello world",
        "language": "en",
        "segments": [{"start": 0.0, "end": 1.5, "text": "Hello world"}],
        "duration": 1.5,
    }
    service.transcribe_streaming.return_value = iter(
        [
            {
                "start": 0.0,
                "end": 1.5,
                "text": "Hello world",
                "partial_text": "Hello world",
                "language": "en",
            }
        ]
    )
    return service


@pytest.fixture
def app(mock_tts_service, mock_stt_service):
    """Create a FastAPI test app with mocked services."""
    from app.main import app as fastapi_app

    fastapi_app.state.tts_service = mock_tts_service
    fastapi_app.state.stt_service = mock_stt_service
    return fastapi_app


@pytest.fixture
def client(app):
    """HTTP test client with mocked services."""
    return TestClient(app)


@pytest.fixture
def sample_wav_bytes():
    """Generate a minimal valid WAV file for testing."""
    import io
    import soundfile as sf

    samples = np.zeros(24000, dtype=np.float32)  # 1 second of silence
    buf = io.BytesIO()
    sf.write(buf, samples, 24000, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()
