from pydantic import BaseModel, Field


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="Text to synthesize")
    voice: str | None = Field(None, description="Voice ID (e.g., 'af_heart')")
    speed: float | None = Field(
        None,
        ge=0.5,
        le=2.0,
        description="Speech speed multiplier (0.5-2.0, the range kokoro-onnx accepts)",
    )
    is_phonemes: bool = Field(default=False, description="Whether text is already phonetic (IPA) writing")
    phonetic_preprocess: bool | None = Field(default=None, description="Explicitly enable/disable phonetic preprocessing")
    custom_lexicon: dict[str, str] | None = Field(default=None, description="Per-request pronunciation overrides (word -> respelling or /ipa/)")


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
        default=None, description="Published quality grade for the voice (e.g. 'A', 'C+')"
    )
    tags: list[str] = Field(
        default_factory=list, description="Filterable tags (gender, language, grade, ...)"
    )


class VoicesResponse(BaseModel):
    voices: list[VoiceInfo]
    total: int


class EngineCapabilities(BaseModel):
    """What one TTS engine can and cannot do."""

    id: str
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
    sample_rates: list[int] = Field(default_factory=list)
    gpu_required: bool = False
    speed_range: list[float] = Field(default_factory=list)


class CapabilitiesResponse(BaseModel):
    engines: list[EngineCapabilities]


class HealthResponse(BaseModel):
    status: str
    tts: bool
    stt: bool
    version: str = "1.0.0"


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
