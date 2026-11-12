"""The round-trip transcriber: local, deterministic, and honest about absence.

Two properties matter and are both asserted here: the harness asks faster-whisper
for greedy decoding (so a WER diff is not sampling noise), and a deployment with no
Whisper model raises instead of returning an empty transcript that would look like a
perfect score.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf

from app.config import settings
from app.eval.stt import (
    LocalTranscriber,
    NullTranscriber,
    TranscriberUnavailable,
    build_transcriber,
)
from tests.stt_fakes import FakeModel, build_service


def _wav() -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, np.zeros(16_000, dtype=np.float32), 16_000, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def test_local_transcriber_asks_for_deterministic_decoding() -> None:
    model = FakeModel()
    transcriber = LocalTranscriber(build_service(model), "base")

    assert transcriber.available is True
    assert transcriber.transcribe(_wav(), "en") == "Hello world"

    kwargs = model.calls[0]["kwargs"]
    # A scalar temperature disables Whisper's sampled fallback ladder: on garbage
    # audio that fallback samples, and two identical renders transcribe differently.
    assert kwargs["temperature"] == 0.0
    # VAD must not trim the audio the WER is measuring.
    assert kwargs["vad_filter"] is False
    assert kwargs["word_timestamps"] is False
    assert kwargs["language"] == "en"


def test_null_transcriber_reports_unmeasured() -> None:
    transcriber = NullTranscriber()
    assert transcriber.available is False
    assert transcriber.transcribe(_wav(), "en") is None
    assert transcriber.describe()["enabled"] is False


def test_build_transcriber_disabled_returns_null() -> None:
    transcriber = build_transcriber(enabled=False)
    assert isinstance(transcriber, NullTranscriber)
    assert transcriber.describe()["enabled"] is False


def test_build_transcriber_without_a_mounted_model_raises(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(settings, "models_dir", str(tmp_path))
    monkeypatch.setattr(settings, "whisper_model_path", None)
    with pytest.raises(TranscriberUnavailable, match="no local Whisper model"):
        build_transcriber(enabled=True, model="base")
