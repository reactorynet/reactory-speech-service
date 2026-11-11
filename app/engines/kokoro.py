"""Kokoro ONNX engine adapter.

Every Kokoro-specific detail lives here: model loading, voice resolution,
phonemisation, language mapping and speed clamping.  The adapter knows nothing
about HTTP, FastAPI or the public request/response models — it speaks
:class:`app.core.contracts.EngineRequest` in and
:class:`app.core.contracts.AudioResult` out.

Kokoro-82M is Apache-2.0, runs on CPU through ONNX Runtime and needs no GPU, so
it remains the reference engine for the edge profile.
"""

from __future__ import annotations

import io
import logging
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from app.config import settings
from app.core.contracts import (
    AudioChunk,
    AudioResult,
    DirectionPlan,
    EngineCapabilities,
    EngineRequest,
    VoiceDescriptor,
)
from app.engines.base import EngineUnavailableError, UnknownVoiceError, lower_direction
from app.services import voice_registry
from app.services.phonetic_converter import phonetic_converter
from app.services.voice_registry import (
    DEFAULT_LOCALE,
    KOKORO_ENGINE,
    KOKORO_SAMPLE_RATE,
    KOKORO_SPEED_MAX,
    KOKORO_SPEED_MIN,
    KOKORO_STYLES,
    build_catalogue,
    catalogue_languages,
)

logger = logging.getLogger(__name__)


class KokoroEngine:
    """Adapter for the Kokoro-82M ONNX model served by ``kokoro-onnx``."""

    id = KOKORO_ENGINE
    name = "Kokoro-82M (ONNX)"
    license = "Apache-2.0"
    attribution = "Kokoro-82M v1.0"
    max_chars = 10000

    def __init__(self) -> None:
        self._model = None
        self._ready = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def model(self):
        """The underlying ``kokoro_onnx.Kokoro`` instance (``None`` when unloaded)."""
        return self._model

    @model.setter
    def model(self, value) -> None:
        self._model = value

    @property
    def is_ready(self) -> bool:
        return self._ready

    def set_ready(self, ready: bool) -> None:
        self._ready = bool(ready)

    async def initialize(self) -> None:
        """Load the Kokoro ONNX model. Call once at startup."""
        try:
            from kokoro_onnx import Kokoro

            model_path = None
            if settings.kokoro_model_resolved.exists():
                model_path = str(settings.kokoro_model_resolved)
            else:
                candidates = list(Path(settings.models_dir).glob("kokoro*.onnx"))
                if candidates:
                    model_path = str(candidates[0])

            if not model_path:
                logger.warning(
                    "Kokoro ONNX model not found in %s. TTS service disabled.",
                    settings.models_dir,
                )
                self._ready = False
                return

            voices_path = None
            if settings.kokoro_voices_resolved.exists():
                voices_path = str(settings.kokoro_voices_resolved)
            else:
                candidates = list(Path(settings.models_dir).glob("voices*.bin"))
                if candidates:
                    voices_path = str(candidates[0])

            if not voices_path:
                logger.warning(
                    "Kokoro voices pack not found in %s. TTS service disabled.",
                    settings.models_dir,
                )
                self._ready = False
                return

            logger.info("Loading Kokoro TTS model from %s (voices: %s)", model_path, voices_path)
            self._model = Kokoro(model_path, voices_path)
            self._ready = True
            logger.info(
                "Kokoro TTS model loaded successfully with %d voices.",
                len(self.available_voice_ids()),
            )
        except FileNotFoundError:
            logger.error("Kokoro model files not found. Run scripts/download_models.py first.")
            self._ready = False
        except Exception:
            logger.exception("Failed to load Kokoro TTS model")
            self._ready = False

    # ------------------------------------------------------------------
    # Voices
    # ------------------------------------------------------------------

    @property
    def engine_voice_ids(self) -> list[str] | None:
        """Voice ids exposed by the loaded engine, or ``None`` when none is loaded.

        The voice pack is the single source of truth for *which* voices exist:
        this reads the keys of ``kokoro_onnx.Kokoro.voices`` instead of a
        hard-coded allow-list.
        """
        kokoro = self._model
        if kokoro is None:
            return None

        voices = getattr(kokoro, "voices", None)
        if voices is None:
            return None

        try:
            keys = list(voices.keys())
        except AttributeError:
            return None

        return sorted(str(key) for key in keys)

    def _is_enabled(self, voice_id: str) -> bool:
        record = voice_registry.get_record(voice_id)
        return record is None or record.enabled

    def available_voice_ids(self) -> list[str]:
        """All voice ids this engine exposes on this deployment.

        Engine keys win whenever an engine is loaded; before the model is loaded
        we fall back to the registry so that voice discovery works without
        inference.  Registry-disabled voices are never advertised.
        """
        engine_ids = self.engine_voice_ids
        if engine_ids is not None:
            return [voice_id for voice_id in engine_ids if self._is_enabled(voice_id)]
        return list(voice_registry.VOICE_REGISTRY.ids(engine=self.id, enabled_only=True))

    def voices(self) -> list[VoiceDescriptor]:
        """The enriched catalogue for this engine, sorted by language then id."""
        return build_catalogue(self.available_voice_ids(), engine=self.id)

    def get_voice(self, voice_id: str) -> VoiceDescriptor | None:
        """Describe one voice, or ``None`` when this engine does not expose it."""
        if voice_id not in self.available_voice_ids():
            return None
        return build_catalogue([voice_id], engine=self.id)[0]

    def resolve_voice(self, voice_id: str | None) -> tuple[str, str]:
        """Resolve a caller-facing voice id to ``(engine voice key, locale)``.

        Raises :class:`UnknownVoiceError` (a ``ValueError``) for ids this engine
        cannot render, keeping the long-standing 422 contract.
        """
        requested = voice_id or voice_registry.default_voice_id(
            available=self.engine_voice_ids
        )
        record = voice_registry.get_record(requested)

        if record is not None:
            if not record.enabled:
                raise UnknownVoiceError(
                    f"Voice '{requested}' is disabled on this deployment. "
                    "GET /api/tts/voices lists the enabled voices."
                )
            if record.engine != self.id:
                raise UnknownVoiceError(
                    f"Voice '{requested}' belongs to engine '{record.engine}', not '{self.id}'. "
                    "GET /api/tts/voices lists the voices available here."
                )
            engine_key = record.engine_voice_id
            locale = record.locale
        else:
            engine_key = requested
            locale = DEFAULT_LOCALE

        available = self.engine_voice_ids
        if available is not None and engine_key not in available:
            raise UnknownVoiceError(
                f"Unknown voice '{requested}'. This deployment exposes "
                f"{len(available)} voices; GET /api/tts/voices lists them all."
            )

        return engine_key, locale

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------

    def capabilities(self) -> EngineCapabilities:
        """Describe Kokoro's native capability matrix.

        Native rate (``speed``) only: no pitch, volume, emotion, SSML or cloning.
        """
        voices = self.voices()
        return EngineCapabilities(
            engine=self.id,
            languages=catalogue_languages(voices),
            native_rate=True,
            native_pitch=False,
            native_volume=False,
            styles=list(KOKORO_STYLES),
            supports_cloning=False,
            supports_streaming=True,
            supports_ssml=False,
            sample_rates=[KOKORO_SAMPLE_RATE],
            max_chars=self.max_chars,
            gpu_required=False,
        )

    def descriptor(self) -> dict[str, Any]:
        """The Phase 0 capability payload plus the Phase 1 contract fields.

        ``id`` is retained as the legacy key; ``engine`` carries the contract
        name.  Both are the same value.
        """
        capabilities = self.capabilities()
        return {
            "id": capabilities.engine,
            "engine": capabilities.engine,
            "name": self.name,
            "available": self.is_ready,
            "license": self.license,
            "languages": capabilities.languages,
            "voice_count": len(self.voices()),
            "default_voice": voice_registry.default_voice_id(),
            "native_rate": capabilities.native_rate,
            "native_pitch": capabilities.native_pitch,
            "native_volume": capabilities.native_volume,
            "styles": capabilities.styles,
            "supports_cloning": capabilities.supports_cloning,
            "supports_streaming": capabilities.supports_streaming,
            "supports_ssml": capabilities.supports_ssml,
            "sample_rates": capabilities.sample_rates,
            "max_chars": capabilities.max_chars,
            "gpu_required": capabilities.gpu_required,
            "speed_range": [KOKORO_SPEED_MIN, KOKORO_SPEED_MAX],
        }

    # ------------------------------------------------------------------
    # Direction lowering
    # ------------------------------------------------------------------

    @staticmethod
    def clamp_speed(speed: float | None) -> float:
        """Clamp a speed multiplier into the range kokoro-onnx accepts.

        The request schema rejects out-of-range values with a 422 before
        inference; this is the defensive backstop for callers that bypass schema
        validation (the WebSocket endpoint) and for an out-of-range
        ``KOKORO_DEFAULT_SPEED``.
        """
        value = settings.kokoro_default_speed if speed is None else float(speed)
        return max(KOKORO_SPEED_MIN, min(KOKORO_SPEED_MAX, value))

    def lower(self, direction) -> DirectionPlan:
        """Lower a provider-neutral direction to Kokoro's native controls."""
        plan = lower_direction(direction, self.capabilities())
        requested_rate = plan.applied.get("rate")
        if requested_rate is not None:
            rate = self.clamp_speed(float(requested_rate))
            if rate != float(requested_rate):
                plan.applied["rate"] = rate
                plan.warnings.append(
                    f"rate {requested_rate} was clamped to {rate}, the range kokoro-onnx accepts"
                )
        return plan

    def _rate_for(self, plan: DirectionPlan) -> float:
        rate = plan.applied.get("rate")
        return self.clamp_speed(None if rate is None else float(rate))

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def synthesize(self, req: EngineRequest) -> AudioResult:
        """Render a complete buffer and report what was applied."""
        if not self._ready or self._model is None:
            raise EngineUnavailableError("TTS service not initialized")

        plan = self.lower(req.direction)
        voice_key, locale = self.resolve_voice(req.voice)
        rate = self._rate_for(plan)

        samples, sample_rate = self._render(
            req.text,
            voice=voice_key,
            rate=rate,
            locale=locale,
            is_phonemes=req.is_phonemes,
            phonetic_preprocess=req.phonetic_preprocess,
            custom_lexicon=req.custom_lexicon,
        )

        return AudioResult(
            audio=self.samples_to_wav(samples, sample_rate),
            duration=len(samples) / sample_rate,
            sample_rate=sample_rate,
            engine=self.id,
            voice=voice_key,
            applied=plan.applied,
            ignored=plan.ignored,
            warnings=plan.warnings,
        )

    def stream(self, req: EngineRequest) -> Iterator[AudioChunk]:
        """Yield one rendered chunk per sentence."""
        if not self._ready or self._model is None:
            raise EngineUnavailableError("TTS service not initialized")

        plan = self.lower(req.direction)
        voice_key, locale = self.resolve_voice(req.voice)
        rate = self._rate_for(plan)

        for sentence in self.split_sentences(req.text):
            if not sentence.strip():
                continue
            samples, sample_rate = self._render(
                sentence,
                voice=voice_key,
                rate=rate,
                locale=locale,
                is_phonemes=req.is_phonemes,
                phonetic_preprocess=req.phonetic_preprocess,
                custom_lexicon=req.custom_lexicon,
            )
            yield AudioChunk(
                audio=self.samples_to_wav(samples, sample_rate),
                duration=len(samples) / sample_rate,
                sample_rate=sample_rate,
            )

    def _render(
        self,
        text: str,
        *,
        voice: str,
        rate: float,
        locale: str,
        is_phonemes: bool,
        phonetic_preprocess: bool | None,
        custom_lexicon: dict[str, str] | None,
    ) -> tuple[np.ndarray, int]:
        """Run inference for one span of text, with the historical phoneme path."""
        model = self._model
        should_phonemize = (
            phonetic_preprocess
            if phonetic_preprocess is not None
            else settings.phonetic_processing_enabled
        )

        if is_phonemes:
            # Caller supplied phonemes directly.
            return model.create(
                text, voice=voice, speed=rate, lang=locale, is_phonemes=True
            )
        if should_phonemize and phonetic_converter.is_available:
            phonetic_text = phonetic_converter.convert_to_phonetic(
                text, language=locale, custom_lexicon=custom_lexicon
            )
            return model.create(
                phonetic_text, voice=voice, speed=rate, lang=locale, is_phonemes=True
            )
        return model.create(text, voice=voice, speed=rate, lang=locale, is_phonemes=False)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def samples_to_wav(samples: np.ndarray, sample_rate: int) -> bytes:
        """Convert float32 numpy samples to WAV bytes."""
        buf = io.BytesIO()
        sf.write(buf, samples, sample_rate, format="WAV", subtype="PCM_16")
        buf.seek(0)
        return buf.read()

    @staticmethod
    def split_sentences(text: str) -> list[str]:
        """Simple sentence splitter for streaming synthesis."""
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [part for part in parts if part.strip()]
