from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # Server
    speech_service_host: str = "0.0.0.0"
    speech_service_port: int = 8765

    # TTS (Kokoro)
    kokoro_model_path: str = "models/kokoro-v1_9.onnx"
    kokoro_voices_path: str = "models/voices-v1_0.bin"
    kokoro_default_voice: str = "af_heart"
    kokoro_default_speed: float = 1.0

    # STT (faster-whisper)
    whisper_model_size: str = "base"
    whisper_model_path: str | None = None
    whisper_device: str = "auto"
    whisper_compute_type: str = "auto"

    # Models
    models_dir: str = "models"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    @property
    def kokoro_model_resolved(self) -> Path:
        return Path(self.kokoro_model_path)

    @property
    def kokoro_voices_resolved(self) -> Path:
        return Path(self.kokoro_voices_path)


settings = Settings()
