"""STT service wrapping faster-whisper for speech-to-text transcription."""

import io
import logging
import tempfile
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)


class STTService:
    """Speech-to-text service using faster-whisper (CTranslate2)."""

    def __init__(self) -> None:
        self._model = None
        self._ready = False

    async def initialize(self) -> None:
        """Load the faster-whisper model. Call once at startup."""
        try:
            from faster_whisper import WhisperModel

            model_size = settings.whisper_model_size
            device = settings.whisper_device
            compute_type = settings.whisper_compute_type

            # Check if a custom or pre-downloaded local model directory exists
            model_to_load = None
            if settings.whisper_model_path and Path(settings.whisper_model_path).exists():
                model_to_load = settings.whisper_model_path
            else:
                candidates = [
                    Path(settings.models_dir) / f"whisper-{model_size}",
                    Path(settings.models_dir) / model_size,
                    Path(settings.models_dir),
                ]
                for cand in candidates:
                    if cand.exists() and (cand / "model.bin").exists():
                        model_to_load = str(cand)
                        break

            if not model_to_load:
                logger.warning(
                    "No local Whisper model files found in %s. STT service disabled. "
                    "Mount model files into %s/whisper-%s to enable.",
                    settings.models_dir,
                    settings.models_dir,
                    model_size,
                )
                self._ready = False
                return

            logger.info(
                "Loading faster-whisper model from local path: %s, device=%s, compute=%s",
                model_to_load,
                device,
                compute_type,
            )
            self._model = WhisperModel(
                model_to_load,
                device=device,
                compute_type=compute_type,
                local_files_only=True,
            )
            self._ready = True
            logger.info("faster-whisper model loaded successfully.")
        except Exception:
            logger.exception("Failed to load faster-whisper model")
            self._ready = False

    @property
    def is_ready(self) -> bool:
        return self._ready

    def transcribe(
        self, audio_bytes: bytes, language: str | None = None
    ) -> dict:
        """Transcribe audio bytes to text.

        Args:
            audio_bytes: Audio file bytes (WAV, WebM, OGG, MP3, etc.)
            language: Optional language code. Auto-detects if None.

        Returns:
            Dict with keys: text, language, segments, duration
        """
        if not self._ready or self._model is None:
            raise RuntimeError("STT service not initialized")

        # Write to temp file — faster-whisper needs a file path or file-like object
        audio_file = self._prepare_audio(audio_bytes)

        kwargs: dict = {"beam_size": 5, "vad_filter": True}
        if language:
            kwargs["language"] = language

        segments_iter, info = self._model.transcribe(audio_file, **kwargs)

        segments = []
        full_text_parts = []
        for segment in segments_iter:
            segments.append(
                {
                    "start": round(segment.start, 3),
                    "end": round(segment.end, 3),
                    "text": segment.text.strip(),
                }
            )
            full_text_parts.append(segment.text.strip())

        return {
            "text": " ".join(full_text_parts),
            "language": info.language,
            "segments": segments,
            "duration": round(info.duration, 3),
        }

    def transcribe_streaming(self, audio_bytes: bytes, language: str | None = None):
        """Generator that yields partial transcription segments as they are produced.

        Each yield is a dict: {start, end, text, partial_text}
        """
        if not self._ready or self._model is None:
            raise RuntimeError("STT service not initialized")

        audio_file = self._prepare_audio(audio_bytes)

        kwargs: dict = {"beam_size": 5, "vad_filter": True}
        if language:
            kwargs["language"] = language

        segments_iter, info = self._model.transcribe(audio_file, **kwargs)

        accumulated_text = []
        for segment in segments_iter:
            text = segment.text.strip()
            accumulated_text.append(text)
            yield {
                "start": round(segment.start, 3),
                "end": round(segment.end, 3),
                "text": text,
                "partial_text": " ".join(accumulated_text),
                "language": info.language,
            }

    @staticmethod
    def _prepare_audio(audio_bytes: bytes) -> str:
        """Write audio bytes to a temp file and return the path.

        faster-whisper can handle WAV, MP3, OGG, FLAC, etc. via ffmpeg.
        """
        suffix = ".wav"
        # Detect WebM by magic bytes
        if audio_bytes[:4] == b"\x1a\x45\xdf\xa3":
            suffix = ".webm"
        elif audio_bytes[:4] == b"OggS":
            suffix = ".ogg"
        elif audio_bytes[:3] == b"ID3" or audio_bytes[:2] == b"\xff\xfb":
            suffix = ".mp3"

        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp.write(audio_bytes)
        tmp.flush()
        tmp.close()
        return tmp.name
