from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Server
    speech_service_host: str = "0.0.0.0"
    speech_service_port: int = 8765

    # TTS (Kokoro)
    kokoro_model_path: str = "models/kokoro-v1.0.onnx"
    kokoro_voices_path: str = "models/voices-v1.0.bin"
    kokoro_default_voice: str = "af_heart"
    kokoro_default_speed: float = 1.0

    # STT (faster-whisper)
    whisper_model_size: str = "base"
    whisper_model_path: str | None = None
    whisper_device: str = "auto"
    whisper_compute_type: str = "auto"

    # STT — Phase 7 (SPEECH-008). Model tiering, VAD and streaming tuning.
    # Models are discovered from directories under ``models_dir`` that contain a
    # ``model.bin``; the ``whisper_model_size`` above selects the one loaded at boot.
    whisper_word_timestamps: bool = False
    whisper_vad_enabled: bool = True
    whisper_vad_threshold: float = 0.5
    whisper_vad_min_silence_ms: int = 500
    whisper_vad_min_speech_ms: int = 250
    whisper_vad_pad_ms: int = 200
    whisper_stream_min_window_seconds: float = 2.0
    whisper_stream_partial_interval_seconds: float = 1.0
    whisper_stream_stability_margin_seconds: float = 0.35
    whisper_stream_max_buffer_seconds: float = 300.0
    whisper_language_confidence_threshold: float = 0.6

    # Models
    models_dir: str = "models"

    # Phonetic processing
    phonetic_processing_enabled: bool = True
    lexicon_file_path: str = "data/lexicon.json"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    @property
    def kokoro_model_resolved(self) -> Path:
        return Path(self.kokoro_model_path)

    @property
    def kokoro_voices_resolved(self) -> Path:
        return Path(self.kokoro_voices_path)


settings = Settings()
