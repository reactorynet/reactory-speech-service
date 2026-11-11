"""Synthesis orchestrator.

:class:`TTSService` is now a thin orchestrator over the engine adapter layer:

* it resolves a request to an engine through :class:`EngineRouter`,
* it reports what direction the engine applied or ignored,
* and it keeps the long-standing ``synthesize`` / ``synthesize_streaming``
  signatures working byte-for-byte for callers that send nothing new.

Kokoro-specific knowledge (model loading, voice resolution, phonemisation,
language mapping, speed clamping) lives in :mod:`app.engines.kokoro`.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator

import numpy as np

from app.core.contracts import (
    AudioResult,
    Direction,
    DirectionPlan,
    EngineRequest,
    Prosody,
)
from app.core.direction import resolve_direction
from app.engines.base import EngineUnavailableError, UnknownEngineError, lower_direction
from app.engines.kokoro import KokoroEngine
from app.engines.manager import EngineManager
from app.engines.router import EngineRouter
from app.services.phonetic_converter import phonetic_converter
from app.services.voice_registry import (
    KOKORO_ENGINE,
    KOKORO_SAMPLE_RATE,
    KOKORO_SPEED_MAX,
    KOKORO_SPEED_MIN,
    VoiceDescriptor,
)

logger = logging.getLogger(__name__)

#: Native output sample rate of the loaded Kokoro model, in Hz.
SAMPLE_RATE = KOKORO_SAMPLE_RATE

#: Engine identifier reported by the voice catalogue and the capabilities endpoint.
ENGINE_ID = KOKORO_ENGINE

#: Speed multiplier range accepted by kokoro-onnx (it raises outside this range).
SPEED_MIN = KOKORO_SPEED_MIN
SPEED_MAX = KOKORO_SPEED_MAX


def _legacy_direction(speed: float | None) -> Direction:
    """Build a direction from the legacy ``speed`` alias.

    ``speed`` reaches this function from the WebSocket endpoint, which bypasses
    the schema, so the value is passed through *without* re-validation and the
    adapter clamps it defensively — exactly as before this change.
    """
    if speed is None:
        return Direction()
    return Direction(prosody=Prosody.model_construct(rate=float(speed)))


class TTSService:
    """Engine-agnostic synthesis orchestrator with a Kokoro adapter registered."""

    def __init__(self) -> None:
        self._kokoro_engine = KokoroEngine()
        self._manager = EngineManager([self._kokoro_engine])
        self._router = EngineRouter(self._manager, order=[KOKORO_ENGINE])

    # ------------------------------------------------------------------
    # Backwards-compatible internals
    # ------------------------------------------------------------------
    # The Phase 0 test-suite (and, historically, the service itself) reaches for
    # ``_kokoro`` and ``_ready``.  They now proxy the Kokoro adapter instead of
    # duplicating its state.

    @property
    def _kokoro(self):
        return self._kokoro_engine.model

    @_kokoro.setter
    def _kokoro(self, value) -> None:
        self._kokoro_engine.model = value

    @property
    def _ready(self) -> bool:
        return self._kokoro_engine.is_ready

    @_ready.setter
    def _ready(self, value: bool) -> None:
        self._kokoro_engine.set_ready(value)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Load every registered engine. Call once at startup."""
        await self._manager.initialize()

    @property
    def is_ready(self) -> bool:
        return self._manager.any_ready()

    @property
    def engine_manager(self) -> EngineManager:
        """The engine manager, for readiness and capability reporting."""
        return self._manager

    @property
    def engine_router(self) -> EngineRouter:
        """The router, exposed so callers can inspect the fallback order."""
        return self._router

    # ------------------------------------------------------------------
    # Voices
    # ------------------------------------------------------------------

    @property
    def engine_voice_ids(self) -> list[str] | None:
        """Voice ids exposed by the loaded engine, or ``None`` when none is loaded."""
        return self._kokoro_engine.engine_voice_ids

    @property
    def voice_ids(self) -> list[str]:
        """All voice ids this deployment exposes."""
        return self._kokoro_engine.available_voice_ids()

    def get_voices(self) -> list[dict]:
        """Return the enriched voice catalogue for this deployment."""
        catalogue: list[VoiceDescriptor] = []
        for engine in self._manager.engines:
            catalogue.extend(engine.voices())
        return list(catalogue)

    def get_voice(self, voice_id: str) -> VoiceDescriptor | None:
        """Return the descriptor for one voice, or ``None`` when it is unknown."""
        for engine in self._manager.engines:
            for descriptor in engine.voices():
                if descriptor["id"] == voice_id:
                    return descriptor
        return None

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------

    def capabilities(self) -> dict:
        """Describe the default engine (Phase 0 shape, kept for compatibility)."""
        return self._kokoro_engine.descriptor()

    def capability_matrix(self) -> list[dict]:
        """One capability payload per registered engine."""
        return self._manager.descriptors()

    # ------------------------------------------------------------------
    # Direction
    # ------------------------------------------------------------------

    def resolve_engine(
        self,
        *,
        voice: str | None = None,
        engine: str | None = None,
        fallback: Iterable[str] | None = None,
    ):
        """Resolve the primary engine for a request."""
        return self._router.resolve(engine=engine, voice=voice, fallback=fallback)

    def plan_direction(
        self,
        *,
        voice: str | None = None,
        engine: str | None = None,
        direction: Direction | None = None,
        fallback: Iterable[str] | None = None,
    ) -> DirectionPlan:
        """Report what a direction would apply, without running inference.

        The adapter's own lowering is used when it provides one, so the plan and
        the synthesis result cannot drift (for example on rate clamping).
        """
        target = self._router.resolve(engine=engine, voice=voice, fallback=fallback)
        requested = direction or Direction()
        lower = getattr(target, "lower", None)
        if callable(lower):
            return lower(requested)
        return lower_direction(requested, target.capabilities())

    @staticmethod
    def clamp_speed(speed: float | None) -> float:
        """Clamp the speed multiplier into the range kokoro-onnx accepts."""
        return KokoroEngine.clamp_speed(speed)

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    def synthesize(
        self,
        text: str,
        voice: str | None = None,
        speed: float | None = None,
        is_phonemes: bool = False,
        phonetic_preprocess: bool | None = None,
        custom_lexicon: dict[str, str] | None = None,
    ) -> tuple[bytes, float]:
        """Synthesize text to WAV audio bytes.

        Returns ``(wav_bytes, duration_seconds)``.  This legacy signature is
        unchanged; the richer result is available from
        :meth:`synthesize_detailed`.
        """
        result = self.synthesize_detailed(
            text,
            voice=voice,
            direction=_legacy_direction(speed),
            is_phonemes=is_phonemes,
            phonetic_preprocess=phonetic_preprocess,
            custom_lexicon=custom_lexicon,
        )
        return result.audio, result.duration

    def synthesize_detailed(
        self,
        text: str,
        *,
        voice: str | None = None,
        speed: float | None = None,
        direction: Direction | None = None,
        engine: str | None = None,
        is_phonemes: bool = False,
        phonetic_preprocess: bool | None = None,
        custom_lexicon: dict[str, str] | None = None,
        fallback: Iterable[str] | None = None,
    ) -> AudioResult:
        """Synthesize and return the audio plus the applied/ignored contract.

        ``direction`` wins over ``speed`` when both carry a rate; the router calls
        :func:`app.core.direction.resolve_direction` first, so that precedence is
        reported in the response.  Engines that cannot render (model not loaded)
        are skipped in favour of the next candidate in the ordered fallback list.
        """
        merged, _ = resolve_direction(speed, direction)
        request = EngineRequest(
            text=text,
            voice=voice,
            direction=merged,
            is_phonemes=is_phonemes,
            phonetic_preprocess=phonetic_preprocess,
            custom_lexicon=custom_lexicon,
        )

        unavailable: EngineUnavailableError | None = None
        candidates = self._router.candidates(engine=engine, voice=voice, fallback=fallback)
        for target in candidates:
            try:
                return target.synthesize(request)
            except EngineUnavailableError as exc:
                logger.warning("TTS engine '%s' is unavailable: %s", target.id, exc)
                unavailable = exc

        if unavailable is not None:
            raise unavailable
        raise UnknownEngineError("No TTS engine could serve the request")

    def synthesize_streaming(
        self,
        text: str,
        voice: str | None = None,
        speed: float | None = None,
        is_phonemes: bool = False,
        phonetic_preprocess: bool | None = None,
        custom_lexicon: dict[str, str] | None = None,
        engine: str | None = None,
        direction: Direction | None = None,
    ) -> Iterator[tuple[bytes, float]]:
        """Generator that yields ``(wav_chunk_bytes, duration)`` per sentence.

        ``engine`` is an explicit override and is resolved through the same
        router as one-shot synthesis, so an unknown engine raises
        :class:`UnknownEngineError` and a known one really is the engine that
        renders the audio.  ``direction`` defaults to the legacy ``speed`` alias.
        """
        request = EngineRequest(
            text=text,
            voice=voice,
            direction=direction if direction is not None else _legacy_direction(speed),
            is_phonemes=is_phonemes,
            phonetic_preprocess=phonetic_preprocess,
            custom_lexicon=custom_lexicon,
        )
        target = self._router.resolve(engine=engine, voice=voice)
        for chunk in target.stream(request):
            yield chunk.audio, chunk.duration

    # ------------------------------------------------------------------
    # Phonemisation (transport-facing helper, untouched by the adapter split)
    # ------------------------------------------------------------------

    def phonemize(
        self,
        text: str,
        language: str = "en-us",
        custom_lexicon: dict[str, str] | None = None,
    ) -> dict:
        """Convert text to phonetic writing and report normalized version."""
        normalized = phonetic_converter.normalize_text(text, custom_lexicon=custom_lexicon)
        phonemes = phonetic_converter.convert_to_phonetic(
            text, language=language, custom_lexicon=custom_lexicon
        )
        return {
            "text": text,
            "normalized": normalized,
            "phonemes": phonemes,
            "language": language,
        }

    @staticmethod
    def _samples_to_wav(samples: np.ndarray, sample_rate: int) -> bytes:
        """Convert float32 numpy samples to WAV bytes."""
        return KokoroEngine.samples_to_wav(samples, sample_rate)

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Simple sentence splitter for streaming TTS."""
        return KokoroEngine.split_sentences(text)


__all__: list[str] = [
    "ENGINE_ID",
    "SAMPLE_RATE",
    "SPEED_MAX",
    "SPEED_MIN",
    "TTSService",
    "lower_direction",
]
