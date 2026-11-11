"""Reusable doubles for the STT tests.

A real ``faster-whisper`` model is hundreds of megabytes and is not present in CI,
so the tests drive the service with a fake model object that exposes the same
``transcribe(source, **kwargs) -> (segments_iter, info)`` contract.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

from app.services.stt_service import STTService
from app.services.vad import TrimResult


@dataclass
class FakeWord:
    """Stand-in for ``faster_whisper.transcribe.Word``."""

    word: str
    start: float
    end: float
    probability: float | None = 0.9


@dataclass
class FakeSegment:
    """Stand-in for ``faster_whisper.transcribe.Segment``."""

    start: float
    end: float
    text: str
    words: list[FakeWord] | None = None


@dataclass
class FakeInfo:
    """Stand-in for ``faster_whisper.transcribe.TranscriptionInfo``."""

    language: str = "en"
    language_probability: float = 0.94
    duration: float = 1.5
    all_language_probs: list[tuple[str, float]] | None = None


class FakeModel:
    """A faster-whisper model double.

    ``fail_at`` selects the failure point: ``"call"`` raises from ``transcribe``,
    ``"iterate"`` raises after the first segment (the realistic decode error).
    ``gate`` blocks ``transcribe`` until released, so a test can observe an
    in-flight request.
    """

    def __init__(
        self,
        segments: list[FakeSegment] | None = None,
        info: FakeInfo | None = None,
        *,
        fail_at: str | None = None,
        gate: threading.Event | None = None,
        observer=None,
    ) -> None:
        self.segments = (
            segments if segments is not None else [FakeSegment(0.0, 1.5, "Hello world")]
        )
        self.info = info or FakeInfo()
        self.fail_at = fail_at
        self.gate = gate
        self.observer = observer
        self.started = threading.Event()
        self.calls: list[dict] = []

    def transcribe(self, source, **kwargs):
        self.calls.append({"source": source, "kwargs": kwargs})
        if self.observer is not None:
            self.observer()
        if self.gate is not None:
            self.started.set()
            self.gate.wait(timeout=10)
        if self.fail_at == "call":
            raise RuntimeError("decoder exploded")
        if self.fail_at == "iterate":
            return self._failing_segments(), self.info
        return iter(self.segments), self.info

    def _failing_segments(self):
        yield self.segments[0]
        raise RuntimeError("decode failed mid-stream")


class NoVAD:
    """VAD stub that trims nothing."""

    def __init__(self) -> None:
        self.calls = 0

    def trim_file(self, path) -> TrimResult:
        self.calls += 1
        return TrimResult(applied=False)

    def describe(self) -> dict:
        return {
            "enabled": True,
            "available": True,
            "threshold": 0.5,
            "min_silence_ms": 500,
            "min_speech_ms": 250,
            "pad_ms": 200,
        }


class StubVAD(NoVAD):
    """VAD stub returning a fixed trim result."""

    def __init__(self, result: TrimResult) -> None:
        super().__init__()
        self.result = result

    def trim_file(self, path) -> TrimResult:
        self.calls += 1
        return self.result


def build_service(model, *, model_id: str = "base", vad=None) -> STTService:
    """Build an STTService with one already-loaded fake model."""
    service = STTService(vad=vad if vad is not None else NoVAD())
    service._paths = {model_id: Path(f"/models/{model_id}")}
    service._models = {model_id: model}
    service._default_model = model_id
    service._ready = True
    return service


class ScriptedSTTService:
    """STT double for the WebSocket tests.

    Produces one segment per 0.9 s of audio received, so the streaming session's
    windowing and de-duplication logic can be asserted deterministically without a
    model. Duration is derived from the WAV header faster-whisper would decode.
    """

    is_ready = True

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def transcribe(self, audio_bytes, language=None, *, model=None, word_timestamps=False, vad=None):
        self.calls.append(
            {
                "bytes": len(audio_bytes),
                "language": language,
                "model": model,
                "word_timestamps": word_timestamps,
            }
        )
        duration = max(0.0, (len(audio_bytes) - 44) / 32000.0)
        segments = []
        index = 0
        while (index + 1) * 0.9 <= duration + 1e-6:
            segment = {
                "start": round(index * 0.9, 3),
                "end": round((index + 1) * 0.9, 3),
                "text": f"word{index}",
            }
            if word_timestamps:
                segment["words"] = [
                    {
                        "word": f"word{index}",
                        "start": segment["start"],
                        "end": segment["end"],
                        "probability": 0.9,
                    }
                ]
            segments.append(segment)
            index += 1
        return {
            "text": " ".join(segment["text"] for segment in segments),
            "language": "en",
            "language_confidence": 0.9,
            "segments": segments,
            "duration": round(duration, 3),
            "model": "base",
            "vad_applied": False,
        }

    def capabilities(self) -> dict:
        return {
            "ready": True,
            "default_model": "base",
            "available_models": ["base"],
            "loaded_models": ["base"],
            "device": "cpu",
            "compute_type": "int8",
            "streaming": True,
            "word_timestamps": True,
            "language_confidence": True,
            "vad": NoVAD().describe(),
        }


class DriftingSTTService(ScriptedSTTService):
    """Reproduces real Whisper re-segmentation across streaming passes.

    With more audio buffered the decoder re-segments: the first pass sees a short
    segment, the next pass returns the same words inside a longer segment, and the
    final pass appends new speech. A streaming implementation that de-duplicates on
    segment *end time* alone re-emits the shared words ("hello hello world").
    """

    def transcribe(self, audio_bytes, language=None, *, model=None, word_timestamps=False, vad=None):
        self.calls.append({"bytes": len(audio_bytes), "language": language})
        duration = max(0.0, (len(audio_bytes) - 44) / 32000.0)
        if duration < 2.9:
            segments = [{"start": 0.0, "end": 0.9, "text": "hello"}]
        elif duration < 3.9:
            # Same region, boundary moved outward: this supersedes "hello".
            segments = [{"start": 0.0, "end": 1.2, "text": "hello world"}]
        else:
            segments = [
                {"start": 0.0, "end": 1.2, "text": "hello world"},
                {"start": 1.2, "end": 1.5, "text": "again"},
            ]
        return {
            "text": " ".join(segment["text"] for segment in segments),
            "language": "en",
            "language_confidence": 0.9,
            "segments": segments,
            "duration": round(duration, 3),
            "model": "base",
            "vad_applied": False,
        }
