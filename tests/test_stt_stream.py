"""Tests for the streaming audio buffer used by ``WS /api/stt/stream``."""

import io
import wave

import pytest

from app.services.stt_stream import StreamingAudioBuffer, normalise_encoding


def test_normalise_encoding_accepts_pcm_and_container():
    """Encoding aliases collapse to two canonical values."""
    assert normalise_encoding("pcm_s16le") == "pcm_s16le"
    assert normalise_encoding("PCM16") == "pcm_s16le"
    assert normalise_encoding("wav") == "container"
    assert normalise_encoding(None) == "container"


def test_normalise_encoding_rejects_unknown():
    """An unsupported encoding is a ValueError, not a silent fallback."""
    with pytest.raises(ValueError, match="Unsupported encoding"):
        normalise_encoding("flac-in-a-can")


def test_pcm_duration_is_exact():
    """Raw PCM duration is derived from the byte count."""
    buffer = StreamingAudioBuffer(encoding="pcm_s16le", sample_rate=16000)
    buffer.append(b"\x00\x00" * 16000)

    assert buffer.duration_seconds == 1.0


def test_pcm_is_wrapped_in_a_valid_wav_container():
    """PCM reaches the decoder inside a real WAV container."""
    buffer = StreamingAudioBuffer(encoding="pcm_s16le", sample_rate=16000)
    buffer.append(b"\x00\x00" * 8000)

    payload = buffer.audio_bytes()

    with wave.open(io.BytesIO(payload), "rb") as handle:
        assert handle.getframerate() == 16000
        assert handle.getnchannels() == 1
        assert handle.getnframes() == 8000


def test_container_duration_reads_the_wav_header():
    """A WAV container reports its buffered duration from the RIFF header."""
    samples = b"\x00\x00" * 16000
    out = io.BytesIO()
    with wave.open(out, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(samples)

    buffer = StreamingAudioBuffer(encoding="container")
    buffer.append(out.getvalue())

    assert buffer.duration_seconds == 1.0
    assert buffer.audio_bytes() == out.getvalue()


def test_container_duration_is_zero_until_enough_bytes_arrive():
    """A prefix of an unknown container cannot be timed yet."""
    buffer = StreamingAudioBuffer(encoding="container")
    buffer.append(b"\x1a\x45\xdf\xa3rest-of-a-webm")

    assert buffer.duration_seconds == 0.0


def test_clear_empties_the_buffer():
    """Ending a session drops the buffered audio."""
    buffer = StreamingAudioBuffer(encoding="pcm_s16le")
    buffer.append(b"\x00\x00" * 100)

    buffer.clear()

    assert len(buffer) == 0
    assert buffer.duration_seconds == 0.0
