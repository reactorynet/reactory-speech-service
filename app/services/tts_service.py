"""TTS service wrapping Kokoro ONNX for text-to-speech synthesis."""

import io
import logging
from pathlib import Path

import numpy as np
import soundfile as sf

from app.config import settings
from app.services.phonetic_converter import phonetic_converter

logger = logging.getLogger(__name__)

# Available Kokoro voices
KOKORO_VOICES = {
    "af_heart": {"name": "Heart (Female)", "language": "en-us"},
    "af_bella": {"name": "Bella (Female)", "language": "en-us"},
    "af_nicole": {"name": "Nicole (Female)", "language": "en-us"},
    "af_sarah": {"name": "Sarah (Female)", "language": "en-us"},
    "af_sky": {"name": "Sky (Female)", "language": "en-us"},
    "am_adam": {"name": "Adam (Male)", "language": "en-us"},
    "am_michael": {"name": "Michael (Male)", "language": "en-us"},
    "bf_emma": {"name": "Emma (Female, British)", "language": "en-gb"},
    "bf_isabella": {"name": "Isabella (Female, British)", "language": "en-gb"},
    "bm_george": {"name": "George (Male, British)", "language": "en-gb"},
    "bm_lewis": {"name": "Lewis (Male, British)", "language": "en-gb"},
}

SAMPLE_RATE = 24000


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
            logger.info("Kokoro TTS model loaded successfully.")
        except FileNotFoundError:
            logger.error("Kokoro model files not found. Run scripts/download_models.py first.")
            self._ready = False
        except Exception:
            logger.exception("Failed to load Kokoro TTS model")
            self._ready = False

    @property
    def is_ready(self) -> bool:
        return self._ready

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

        voice = voice or settings.kokoro_default_voice
        speed = speed or settings.kokoro_default_speed

        if voice not in KOKORO_VOICES:
            raise ValueError(
                f"Unknown voice '{voice}'. Available: {list(KOKORO_VOICES.keys())}"
            )

        voice_lang = KOKORO_VOICES[voice].get("language", "en-us")

        should_phonemize = (
            phonetic_preprocess
            if phonetic_preprocess is not None
            else settings.phonetic_processing_enabled
        )

        if is_phonemes:
            # Caller supplied phonemes directly
            samples, sample_rate = self._kokoro.create(
                text, voice=voice, speed=speed, lang=voice_lang, is_phonemes=True
            )
        elif should_phonemize and phonetic_converter.is_available:
            phonetic_text = phonetic_converter.convert_to_phonetic(
                text, language=voice_lang, custom_lexicon=custom_lexicon
            )
            samples, sample_rate = self._kokoro.create(
                phonetic_text, voice=voice, speed=speed, lang=voice_lang, is_phonemes=True
            )
        else:
            samples, sample_rate = self._kokoro.create(
                text, voice=voice, speed=speed, lang=voice_lang, is_phonemes=False
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

        voice = voice or settings.kokoro_default_voice
        speed = speed or settings.kokoro_default_speed

        if voice not in KOKORO_VOICES:
            raise ValueError(
                f"Unknown voice '{voice}'. Available: {list(KOKORO_VOICES.keys())}"
            )

        voice_lang = KOKORO_VOICES[voice].get("language", "en-us")

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
                    sentence, voice=voice, speed=speed, lang=voice_lang, is_phonemes=True
                )
            elif should_phonemize and phonetic_converter.is_available:
                p_text = phonetic_converter.convert_to_phonetic(
                    sentence, language=voice_lang, custom_lexicon=custom_lexicon
                )
                samples, sample_rate = self._kokoro.create(
                    p_text, voice=voice, speed=speed, lang=voice_lang, is_phonemes=True
                )
            else:
                samples, sample_rate = self._kokoro.create(
                    sentence, voice=voice, speed=speed, lang=voice_lang, is_phonemes=False
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

    def get_voices(self) -> list[dict]:
        """Return list of available voices."""
        return [
            {"id": vid, "name": info["name"], "language": info["language"]}
            for vid, info in KOKORO_VOICES.items()
        ]

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
