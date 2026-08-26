"""TTS service wrapping Kokoro ONNX for text-to-speech synthesis."""

import io
import logging
from pathlib import Path

import numpy as np
import soundfile as sf

from app.config import settings

logger = logging.getLogger(__name__)

# Available Kokoro voices (American English subset)
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
    """Text-to-speech service using Kokoro ONNX."""

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
                    "Kokoro ONNX model not found in %s. TTS service disabled. "
                    "Mount model files into %s to enable.",
                    settings.models_dir,
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
                    "Kokoro voices pack not found in %s. TTS service disabled. "
                    "Mount voices pack into %s to enable.",
                    settings.models_dir,
                    settings.models_dir,
                )
                self._ready = False
                return

            logger.info("Loading Kokoro TTS model from %s (voices: %s)", model_path, voices_path)
            self._kokoro = Kokoro(model_path, voices_path)
            self._ready = True
            logger.info("Kokoro TTS model loaded successfully.")
        except FileNotFoundError:
            logger.error(
                "Kokoro model files not found. Run 'make download-models' first."
            )
            self._ready = False
        except Exception:
            logger.exception("Failed to load Kokoro TTS model")
            self._ready = False

    @property
    def is_ready(self) -> bool:
        return self._ready

    def synthesize(
        self, text: str, voice: str | None = None, speed: float | None = None
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

        samples, sample_rate = self._kokoro.create(text, voice=voice, speed=speed)

        duration = len(samples) / sample_rate
        wav_bytes = self._samples_to_wav(samples, sample_rate)
        return wav_bytes, duration

    def synthesize_streaming(
        self, text: str, voice: str | None = None, speed: float | None = None
    ):
        """Generator that yields (wav_chunk_bytes, duration) per sentence.

        Each chunk is a complete WAV file for one sentence fragment.
        """
        if not self._ready or self._kokoro is None:
            raise RuntimeError("TTS service not initialized")

        voice = voice or settings.kokoro_default_voice
        speed = speed or settings.kokoro_default_speed

        if voice not in KOKORO_VOICES:
            raise ValueError(
                f"Unknown voice '{voice}'. Available: {list(KOKORO_VOICES.keys())}"
            )

        # kokoro_onnx.create returns all audio at once; split by sentence for streaming
        # For true streaming, we split text into sentences and synthesize each
        sentences = self._split_sentences(text)
        for sentence in sentences:
            if not sentence.strip():
                continue
            samples, sample_rate = self._kokoro.create(
                sentence, voice=voice, speed=speed
            )
            duration = len(samples) / sample_rate
            wav_bytes = self._samples_to_wav(samples, sample_rate)
            yield wav_bytes, duration

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
