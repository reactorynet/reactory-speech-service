from typing import Any

from pydantic import BaseModel, Field

from app.core.contracts import Direction


class TTSRequest(BaseModel):
    """Body of ``POST /api/tts/synthesize``.

    The Phase 0 fields are unchanged.  ``direction`` and ``engine`` are additive:
    ``speed`` remains an accepted alias for ``direction.prosody.rate``, and when
    both are supplied ``direction`` wins and the response says so.
    """

    text: str = Field(..., min_length=1, max_length=10000, description="Text to synthesize")
    voice: str | None = Field(None, description="Voice ID, as listed by GET /api/tts/voices")
    speed: float | None = Field(
        None,
        ge=0.5,
        le=2.0,
        description=(
            "Speech speed multiplier (0.5-2.0). Legacy alias for direction.prosody.rate; "
            "when both are supplied, direction wins."
        ),
    )
    engine: str | None = Field(
        None,
        description="Optional engine override (e.g. 'kokoro'); omit to let the voice decide",
    )
    direction: Direction | None = Field(
        None,
        description=(
            "Provider-neutral direction: style, intensity and prosody. Omit for the legacy "
            "path, which behaves exactly as before this field existed."
        ),
    )
    is_phonemes: bool = Field(default=False, description="Whether text is already phonetic (IPA) writing")
    phonetic_preprocess: bool | None = Field(default=None, description="Explicitly enable/disable phonetic preprocessing")
    custom_lexicon: dict[str, str] | None = Field(default=None, description="Per-request pronunciation overrides (word -> respelling or /ipa/)")
    cache: bool = Field(
        default=True,
        description=(
            "Whether this request may be served from the synthesis cache. Set false to force "
            "a fresh render; the service-wide switch is SPEECH_CACHE_ENABLED."
        ),
    )


class PhonemizeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="Text to phonemize")
    language: str | None = Field(default="en-us", description="Language code ('en-us' or 'en-gb')")
    custom_lexicon: dict[str, str] | None = Field(default=None, description="Optional pronunciation overrides for phonemization")


class PhonemizeResponse(BaseModel):
    text: str = Field(..., description="Original input text")
    normalized: str = Field(..., description="Normalized text after expansions")
    phonemes: str = Field(..., description="Phonetic (IPA) representation")
    language: str = Field(default="en-us", description="Language code used")


class LexiconEntryRequest(BaseModel):
    word: str = Field(..., min_length=1, max_length=200, description="Word or phrase to match")
    replacement: str | None = Field(default=None, description="Phonetic respelling (e.g. 'koo-ber-net-eez')")
    ipa: str | None = Field(default=None, description="Direct International Phonetic Alphabet (IPA) representation")


class LexiconEntryResponse(BaseModel):
    word: str
    replacement: str | None = None
    ipa: str | None = None
    source: str = "custom"


class LexiconListResponse(BaseModel):
    entries: dict[str, dict]
    total: int


class TTSResponse(BaseModel):
    audio_base64: str = Field(..., description="Base64-encoded WAV audio")
    duration: float = Field(..., description="Audio duration in seconds")
    format: str = Field(default="wav", description="Audio format")
    sample_rate: int = Field(default=24000, description="Sample rate in Hz")
    engine: str | None = Field(default=None, description="Engine that rendered the audio")
    direction_applied: dict[str, Any] = Field(
        default_factory=dict,
        description="Direction controls the engine actually applied",
    )
    direction_ignored: list[str] = Field(
        default_factory=list,
        description="Direction controls the engine could not apply",
    )
    direction_warnings: list[str] = Field(
        default_factory=list,
        description="Human-readable explanation for each ignored control",
    )
    cache: str = Field(
        default="miss",
        description="'hit' when the audio came from the synthesis cache, otherwise 'miss'",
    )
    cache_tier: str = Field(
        default="none",
        description="Which cache tier served the audio: 'lru', 'disk' or 'none'",
    )


class STTRequest(BaseModel):
    language: str | None = Field(None, description="Language code (e.g., 'en'). Auto-detect if None")


class TranscriptionSegment(BaseModel):
    start: float = Field(..., description="Segment start time in seconds")
    end: float = Field(..., description="Segment end time in seconds")
    text: str = Field(..., description="Segment text")


class TranscriptionResult(BaseModel):
    text: str = Field(..., description="Full transcription text")
    language: str = Field(..., description="Detected or specified language")
    segments: list[TranscriptionSegment] = Field(default_factory=list)
    duration: float = Field(default=0.0, description="Audio duration in seconds")


class VoiceInfo(BaseModel):
    """A single TTS voice.

    ``id``/``name``/``language`` are always present; the remaining fields are
    populated when the engine exposes the metadata that allows deriving them.
    ``quality`` is ``null`` for the voices the catalogue of record does not grade.
    """

    id: str
    name: str
    language: str
    engine: str | None = Field(default=None, description="Engine id (e.g. 'kokoro')")
    gender: str | None = Field(default=None, description="'female' or 'male'")
    locale: str | None = Field(
        default=None, description="Engine language code accepted during synthesis (e.g. 'en-us')"
    )
    quality: str | None = Field(
        default=None,
        description="Published quality grade (e.g. 'A', 'C-'); null when the catalogue does not grade the voice",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Filterable tags (gender, language, region, grade-*, recommended, unrated)",
    )


class VoicesResponse(BaseModel):
    voices: list[VoiceInfo]
    total: int
    engine: str | None = Field(default=None, description="Engine the catalogue was read from")
    available: bool = Field(
        default=True,
        description=(
            "Whether the engine that provides these voices is loaded. When false the "
            "catalogue is the shipped-pack manifest and synthesis returns 503."
        ),
    )


class EngineCapabilities(BaseModel):
    """What one TTS engine can and cannot do.

    ``id`` is the stable key from Phase 0; ``engine`` carries the same value under
    the engine-agnostic contract name.  Both are emitted so clients written
    against either shape keep working.
    """

    id: str
    engine: str | None = None
    name: str
    available: bool = False
    license: str | None = None
    languages: list[str] = Field(default_factory=list)
    voice_count: int = 0
    default_voice: str | None = None
    native_rate: bool = False
    native_pitch: bool = False
    native_volume: bool = False
    styles: list[str] = Field(default_factory=list)
    supports_cloning: bool = False
    supports_streaming: bool = False
    supports_ssml: bool = False
    sample_rates: list[int] = Field(default_factory=list)
    max_chars: int | None = None
    gpu_required: bool = False
    speed_range: list[float] = Field(default_factory=list)


class CapabilitiesResponse(BaseModel):
    engines: list[EngineCapabilities]


class EngineHealth(BaseModel):
    """Readiness and queue depth for a single TTS engine."""

    id: str
    ready: bool = False
    loaded: bool = False
    lifecycle: str = Field(
        default="unknown",
        description="'ready', 'loading' or 'unloaded'",
    )
    queue_depth: int = Field(
        default=0,
        description="Syntheses waiting for a concurrency slot on this engine",
    )
    in_flight: int = Field(default=0, description="Syntheses currently rendering")
    max_concurrency: int = Field(default=1, description="Configured per-engine concurrency limit")
    peak_concurrency: int = Field(default=0, description="Highest observed concurrency")
    idle_seconds: float | None = Field(
        default=None, description="Seconds since the engine last served a request"
    )
    gpu_required: bool = False
    max_chars: int | None = None
    voices: int | None = None


class HealthResponse(BaseModel):
    """Health payload.

    ``tts`` and ``stt`` are unchanged for backwards compatibility.  ``engines``,
    ``queue_depth`` and ``cache`` are additive: old clients keep reading the
    booleans they already depend on.
    """

    status: str
    tts: bool
    stt: bool
    version: str = "1.0.0"
    engines: dict[str, EngineHealth] = Field(
        default_factory=dict,
        description="Per-engine readiness, lifecycle and queue depth",
    )
    queue_depth: int = Field(
        default=0, description="Total syntheses queued across all TTS engines"
    )
    cache: dict[str, Any] | None = Field(
        default=None, description="Synthesis cache configuration and counters"
    )


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
