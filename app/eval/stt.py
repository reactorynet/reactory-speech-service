"""Round-trip transcription for the evaluation harness.

The primary metric is *round-trip intelligibility*: synthesise a sentence,
transcribe the audio with the **local** STT service, and compare the transcript
with the source. The transcriber here is therefore the existing
:class:`app.services.stt_service.STTService` — faster-whisper, ``local_files_only``,
no hosted API. Nothing in this module may reach the network.

Two deliberate choices:

* ``vad=False`` — the render is already a clean, single utterance. Running
  voice-activity detection would add a variable that can only remove speech, and a
  metric that silently trims the audio it is measuring is not a measurement.
* an unavailable transcriber is an *error*, not an empty transcript. If a caller
  asks for round-trip scoring and no Whisper model is mounted, the harness exits
  non-zero with the reason instead of writing a report full of ``null`` WER that
  looks like a pass. ``--no-stt`` is the explicit way to opt out, and the report
  then says ``stt: disabled``.
"""

from __future__ import annotations

import asyncio
from typing import Protocol, runtime_checkable

from app.services.stt_service import STTService


# The script prompts live in the STT service so a direct /api/stt/transcribe call
# and this harness produce the same characters (review finding F4). Re-exported
# here because the harness's describe() documents them.
from app.services.stt_service import SCRIPT_PROMPTS  # noqa: E402


class TranscriberUnavailable(RuntimeError):
    """Round-trip scoring was requested but no local STT model is mounted."""


@runtime_checkable
class Transcriber(Protocol):
    """Turns rendered audio into text with a local model."""

    available: bool
    model: str | None

    def describe(self) -> dict:
        """Serialisable description for the report header."""
        raise NotImplementedError

    def transcribe(self, wav_bytes: bytes, language: str | None = None) -> str | None:
        """Transcript of ``wav_bytes``, or ``None`` when no transcript is available."""
        raise NotImplementedError


class NullTranscriber:
    """The ``--no-stt`` transcriber: deliberately produces no transcript.

    ``available`` is ``False`` so the runner records ``wer: null`` with
    ``stt: disabled`` rather than a fabricated 0.0.
    """

    available = False
    model = None

    def describe(self) -> dict:
        return {
            "enabled": False,
            "available": False,
            "model": None,
            "description": "disabled by --no-stt; round-trip WER/CER not measured",
        }

    def transcribe(self, wav_bytes: bytes, language: str | None = None) -> str | None:
        return None


class LocalTranscriber:
    """Round-trip scoring through the local faster-whisper service."""

    def __init__(
        self,
        service: STTService,
        model: str | None = None,
        script_prompts: dict[str, str] | None = None,
    ) -> None:
        self._service = service
        self.model = model or service.default_model
        self._script_prompts = dict(SCRIPT_PROMPTS if script_prompts is None else script_prompts)
        # Any mounted model can transcribe; ``create`` validates the requested id.
        self.available = bool(service.available_models)

    @classmethod
    def create(cls, model: str | None = None) -> LocalTranscriber:
        """Load the local STT service, raising when nothing is mounted."""
        service = STTService()
        asyncio.run(service.initialize())
        available = service.available_models
        if not available:
            raise TranscriberUnavailable(
                "no local Whisper model found under MODELS_DIR. Mount a faster-whisper model "
                "directory containing model.bin (scripts/download_models.py) or pass --no-stt "
                "to run the harness without round-trip intelligibility."
            )
        if model is not None and model not in available:
            raise TranscriberUnavailable(
                f"Whisper model {model!r} is not mounted. Available: {', '.join(available)}"
            )
        return cls(service, model)

    def describe(self) -> dict:
        return {
            "enabled": True,
            "available": self.available,
            "model": self.model,
            "description": "local faster-whisper (CTranslate2), local_files_only, vad disabled",
            "script_prompts": sorted(self._script_prompts),
            "script_prompt_note": (
                "fixed language-generic prompt per script, applied to select Simplified "
                "Chinese / Devanagari output; it adds no corpus content"
            ),
        }

    def transcribe(self, wav_bytes: bytes, language: str | None = None) -> str | None:
        result = self._service.transcribe(
            wav_bytes,
            language=language,
            model=self.model,
            # Determinism over convenience: see the module docstring. VAD would trim
            # the audio it is measuring; a scalar temperature disables Whisper's
            # sampled fallback ladder, which otherwise transcribes garbage audio
            # differently on every run and turns a WER diff into noise.
            vad=False,
            word_timestamps=False,
            temperature=0.0,
            initial_prompt=self._script_prompts.get(
                (language or "").strip().lower().split("-")[0]
            ),
        )
        text = result.get("text")
        return text if isinstance(text, str) else None


def build_transcriber(*, enabled: bool = True, model: str | None = None) -> Transcriber:
    """Build the transcriber a run should use.

    Raises :class:`TranscriberUnavailable` when round-trip scoring is requested but
    the deployment has no local model. That is a configuration error, not a metric
    of zero.
    """
    if not enabled:
        return NullTranscriber()
    return LocalTranscriber.create(model)


__all__ = [
    "LocalTranscriber",
    "NullTranscriber",
    "SCRIPT_PROMPTS",
    "Transcriber",
    "TranscriberUnavailable",
    "build_transcriber",
]
