"""TTS service wrapping Kokoro ONNX for text-to-speech synthesis."""

import io
import logging
from pathlib import Path

import numpy as np
import soundfile as sf

from app.config import settings
from app.services.phonetic_converter import phonetic_converter
from app.services.voice_registry import (
    DEFAULT_LOCALE,
    KOKORO_ENGINE,
    KOKORO_SAMPLE_RATE,
    KOKORO_SPEED_MAX,
    KOKORO_SPEED_MIN,
    SHIPPED_VOICE_IDS,
    VoiceDescriptor,
    build_catalogue,
    describe_voice,
    kokoro_capabilities,
)

logger = logging.getLogger(__name__)

#: Native output sample rate of the loaded Kokoro model, in Hz.
SAMPLE_RATE = KOKORO_SAMPLE_RATE

#: Engine identifier reported by the voice catalogue and the capabilities endpoint.
ENGINE_ID = KOKORO_ENGINE

#: Speed multiplier range accepted by kokoro-onnx (it raises outside this range).
SPEED_MIN = KOKORO_SPEED_MIN
SPEED_MAX = KOKORO_SPEED_MAX


class TTSService:
    """Text-to-speech service using Kokoro ONNX with phonetic pre-processing and custom lexicons."""

    def __init__(self) -> None:
        self._kokoro = None
        self._ready = False

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
            self._kokoro = Kokoro(model_path, voices_path)
            self._ready = True
            logger.info(
                "Kokoro TTS model loaded successfully with %d voices.",
                len(self.voice_ids),
            )
        except FileNotFoundError:
            logger.error("Kokoro model files not found. Run scripts/download_models.py first.")
            self._ready = False
        except Exception:
            logger.exception("Failed to load Kokoro TTS model")
            self._ready = False

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def engine_voice_ids(self) -> list[str] | None:
        """Voice ids exposed by the loaded engine, or ``None`` when none is loaded.

        The voice pack is the single source of truth for the catalogue: this reads
        the keys of ``kokoro_onnx.Kokoro.voices`` directly instead of a hard-coded
        allow-list.
        """
        kokoro = self._kokoro
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

    @property
    def voice_ids(self) -> list[str]:
        """All voice ids this deployment exposes.

        Engine keys win whenever an engine is loaded; before the model is loaded we
        fall back to the manifest of the shipped pack so that voice discovery works
        without inference.  The fallback is never used to reject a voice.
        """
        engine_ids = self.engine_voice_ids
        if engine_ids is not None:
            return engine_ids
        return list(SHIPPED_VOICE_IDS)

    def get_voices(self) -> list[dict]:
        """Return the enriched voice catalogue for this deployment."""
        return list(build_catalogue(self.voice_ids, engine=ENGINE_ID))

    def get_voice(self, voice_id: str) -> VoiceDescriptor | None:
        """Return the descriptor for one voice, or ``None`` when it is unknown."""
        if voice_id not in self.voice_ids:
            return None
        return build_catalogue([voice_id], engine=ENGINE_ID)[0]

    def capabilities(self) -> dict:
        """Describe the TTS engines available in this deployment."""
        return kokoro_capabilities(
            self.get_voices(),
            available=self._ready,
            default_voice=settings.kokoro_default_voice,
        )

    @staticmethod
    def clamp_speed(speed: float | None) -> float:
        """Clamp the speed multiplier into the range kokoro-onnx accepts.

        The request schema rejects out-of-range values with a 422 before inference;
        this is the defensive backstop for callers that bypass schema validation
        (the WebSocket endpoint) and for an out-of-range ``KOKORO_DEFAULT_SPEED``.
        """
        value = settings.kokoro_default_speed if speed is None else float(speed)
        return max(SPEED_MIN, min(SPEED_MAX, value))

    def _resolve_voice(self, voice: str | None) -> tuple[str, str]:
        """Resolve a requested voice to its engine id and engine language code."""
        voice_id = voice or settings.kokoro_default_voice
        available = self.engine_voice_ids
        if available is not None and voice_id not in available:
            raise ValueError(
                f"Unknown voice '{voice_id}'. This deployment exposes "
                f"{len(available)} voices; GET /api/tts/voices lists them all."
            )

        descriptor = describe_voice(voice_id, engine=ENGINE_ID)
        return voice_id, descriptor["locale"] if descriptor else DEFAULT_LOCALE

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

        Returns (wav_bytes, duration_seconds).
        """
        if not self._ready or self._kokoro is None:
            raise RuntimeError("TTS service not initialized")

        voice_id, voice_lang = self._resolve_voice(voice)
        speed = self.clamp_speed(speed)

        should_phonemize = (
            phonetic_preprocess
            if phonetic_preprocess is not None
            else settings.phonetic_processing_enabled
        )

        if is_phonemes:
            # Caller supplied phonemes directly
            samples, sample_rate = self._kokoro.create(
                text, voice=voice_id, speed=speed, lang=voice_lang, is_phonemes=True
            )
        elif should_phonemize and phonetic_converter.is_available:
            phonetic_text = phonetic_converter.convert_to_phonetic(
                text, language=voice_lang, custom_lexicon=custom_lexicon
            )
            samples, sample_rate = self._kokoro.create(
                phonetic_text, voice=voice_id, speed=speed, lang=voice_lang, is_phonemes=True
            )
        else:
            samples, sample_rate = self._kokoro.create(
                text, voice=voice_id, speed=speed, lang=voice_lang, is_phonemes=False
            )

        duration = len(samples) / sample_rate
        wav_bytes = self._samples_to_wav(samples, sample_rate)
        return wav_bytes, duration

    def synthesize_streaming(
        self,
        text: str,
        voice: str | None = None,
        speed: float | None = None,
        is_phonemes: bool = False,
        phonetic_preprocess: bool | None = None,
        custom_lexicon: dict[str, str] | None = None,
    ):
        """Generator that yields (wav_chunk_bytes, duration) per sentence."""
        if not self._ready or self._kokoro is None:
            raise RuntimeError("TTS service not initialized")

        voice_id, voice_lang = self._resolve_voice(voice)
        speed = self.clamp_speed(speed)

        should_phonemize = (
            phonetic_preprocess
            if phonetic_preprocess is not None
            else settings.phonetic_processing_enabled
        )

        sentences = self._split_sentences(text)
        for sentence in sentences:
            if not sentence.strip():
                continue

            if is_phonemes:
                samples, sample_rate = self._kokoro.create(
                    sentence, voice=voice_id, speed=speed, lang=voice_lang, is_phonemes=True
                )
            elif should_phonemize and phonetic_converter.is_available:
                p_text = phonetic_converter.convert_to_phonetic(
                    sentence, language=voice_lang, custom_lexicon=custom_lexicon
                )
                samples, sample_rate = self._kokoro.create(
                    p_text, voice=voice_id, speed=speed, lang=voice_lang, is_phonemes=True
                )
            else:
                samples, sample_rate = self._kokoro.create(
                    sentence, voice=voice_id, speed=speed, lang=voice_lang, is_phonemes=False
                )

            duration = len(samples) / sample_rate
            wav_bytes = self._samples_to_wav(samples, sample_rate)
            yield wav_bytes, duration

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
        buf = io.BytesIO()
        sf.write(buf, samples, sample_rate, format="WAV", subtype="PCM_16")
        buf.seek(0)
        return buf.read()

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Simple sentence splitter for streaming TTS."""
        import re

        parts = re.split(r"(?<=[.!?])\s+", text)
        return [p for p in parts if p.strip()]
