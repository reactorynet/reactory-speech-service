from pathlib import Path

from pydantic import Field
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

    # TTS — Phase 3 (SPEECH-004): concurrency, lifecycle and caching.
    #
    # Inference never runs on the event loop: every engine call is dispatched to
    # this bounded thread pool.  The pool is deliberately small so one node
    # cannot be oversubscribed by concurrent requests.
    speech_thread_pool_size: int = 4
    # Per-engine concurrency limits.  ``default`` applies to any engine that is
    # not listed by id.  Kokoro is CPU-bound and benefits from two overlapping
    # renders; GPU/expressive engines are limited to one to protect VRAM.
    speech_engine_max_concurrency: dict[str, int] = Field(
        default_factory=lambda: {"kokoro": 2, "default": 1}
    )
    # Engine lifecycle.  Engines beyond ``max_loaded_engines`` are loaded lazily
    # on first use; an engine idle for longer than the timeout is unloaded and
    # reloaded on demand.
    speech_max_loaded_engines: int = 2
    speech_engine_idle_timeout_seconds: float = 300.0
    speech_engine_reaper_interval_seconds: float = 30.0

    # TTS synthesis cache (Phase 3).  Two tiers: an in-process LRU for hot
    # phrases plus a TTL-bounded disk cache for longer text and previews.  Both
    # tiers are local to the node; there is no Redis or hosted cache.
    speech_cache_enabled: bool = True
    speech_cache_lru_entries: int = 256
    speech_cache_lru_max_bytes: int = 16 * 1024 * 1024
    speech_cache_lru_max_entry_bytes: int = 512 * 1024
    speech_cache_dir: str = ".cache/tts"
    speech_cache_ttl_seconds: float = 24 * 60 * 60
    speech_cache_disk_max_bytes: int = 512 * 1024 * 1024

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

    @property
    def speech_cache_dir_resolved(self) -> Path:
        return Path(self.speech_cache_dir)

    def engine_concurrency_limit(self, engine_id: str) -> int:
        """Concurrent inference slots for one engine.

        An engine named in ``SPEECH_ENGINE_MAX_CONCURRENCY`` uses its own limit;
        everything else uses the ``default`` entry.  Values below one are
        clamped, because a zero limit would deadlock the engine.
        """
        limits = self.speech_engine_max_concurrency or {}
        value = limits.get(engine_id, limits.get("default", 1))
        try:
            return max(1, int(value))
        except (TypeError, ValueError):
            return 1


settings = Settings()
