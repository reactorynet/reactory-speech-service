from pydantic import BaseModel, Field


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="Text to synthesize")
    voice: str | None = Field(None, description="Voice ID (e.g., 'af_heart')")
    speed: float | None = Field(None, ge=0.25, le=4.0, description="Speech speed multiplier")
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
    id: str
    name: str
    language: str


class HealthResponse(BaseModel):
    status: str
    tts: bool
    stt: bool
    version: str = "1.0.0"


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
