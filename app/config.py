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
    # TTS — Phase 5 (SPEECH-006): consent-first voice cloning (ADR-0003).
    #
    # Cloning is off unless the deployment supplies all of the following.  These
    # files are per-instance configuration and MUST NOT be committed:
    #
    # * ``clone_approvers.json`` — ``{"clone_approvers": ["<email>", ...]}``.
    #   Absent or empty means cloning is UNAVAILABLE, never unrestricted.
    # * ``clone_signing_key_path`` — an Ed25519 private key (PEM).  It signs
    #   consent records and provenance manifests.  It is never generated at boot:
    #   a regenerated key would invalidate every existing signature.
    # * ``clone_reference_key_path`` — 32 bytes for AES-256-GCM, deliberately a
    #   different file from ordinary application data.
    #
    # The store holds consent records, encrypted embeddings, the durable
    # provenance manifest registry and the audit log.
    clone_store_dir: str = "data/clones"
    clone_approvers_path: str = "data/clone_approvers.json"
    clone_signing_key_path: str = "data/keys/clone_signing_key.pem"
    clone_reference_key_path: str = "data/keys/clone_reference.key"
    #: Stable identity of this deployment, carried inside every signature.  Empty
    #: falls back to the hostname, which is stable for a container per instance.
    clone_instance_id: str = ""

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
    #: Apply the fixed per-language script prompt (Devanagari for ``hi``, Simplified
    #: for ``zh``) when a request does not supply one. Off means Whisper picks the
    #: script from acoustics alone, which returns Perso-Arabic for Hindi on the
    #: pinned ``base`` model (SPEECH-010 review finding F4).
    whisper_script_prompts_enabled: bool = True

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
    #: After a load failure, do not retry that engine for this long.  A broken
    #: optional engine (absent weights, broken torch/perth install) would
    #: otherwise repeat its whole preflight + construction on every request that
    #: routes to it, under the engine lock, before falling back.  Set to 0 to
    #: disable the backoff.
    speech_engine_load_retry_seconds: float = 30.0

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

    # TTS — Phase 2 (SPEECH-003): designed voices.
    #
    # A designed voice is a *recipe* — a weighted blend of base Kokoro style
    # tensors, or an alias for one base voice — persisted as JSON.  The tensor is
    # always derived on demand and cached in memory; it is never stored.  The
    # store lives outside the package so a container can mount it read-write, and
    # loading tolerates a missing or malformed file (pack voices keep working).
    designed_voices_path: str = "data/designed_voices.json"
    # Where preview WAVs are cached.  Defaults to a ``previews`` directory beside
    # the store file when left unset.
    designed_voices_preview_dir: str | None = None
    # Upper bound on resolved blend tensors held in memory.  One tensor is
    # 510*1*256 float32 = ~522 KB, so 64 entries is ~33 MB worst case.
    designed_voice_tensor_cache_size: int = 64

    # TTS — Phase 4 (SPEECH-005): optional expressive engine (Chatterbox, MIT).
    #
    # Off by default.  When false the adapter module is never imported and the
    # engine is never registered, so the default speech-core deployment behaves
    # exactly as before and needs neither torch nor the 3.1 GB weights.
    chatterbox_enabled: bool = False
    #: ``standard`` (ChatterboxTTS) or ``turbo`` (ChatterboxTurboTTS).  There is
    #: no "Nano" class in chatterbox-tts 0.1.7; see the measurements page.
    chatterbox_variant: str = "standard"
    chatterbox_device: str = "cpu"
    #: Weights live in the models volume and are downloaded on demand, never at
    #: container build time.  The directory carries a manifest.json with the
    #: licence (MIT) and a SHA-256 per file.
    chatterbox_model_dir: str = "models/chatterbox"

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

    @property
    def chatterbox_model_dir_resolved(self) -> Path:
        return Path(self.chatterbox_model_dir)

    @property
    def designed_voices_resolved(self) -> Path:
        return Path(self.designed_voices_path)

    @property
    def designed_voices_preview_resolved(self) -> Path:
        """Preview cache directory: configured, or ``<store dir>/previews``."""
        if self.designed_voices_preview_dir:
            return Path(self.designed_voices_preview_dir)
        return self.designed_voices_resolved.parent / "previews"

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
