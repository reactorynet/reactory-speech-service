from pydantic import BaseModel, Field


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="Text to synthesize")
    voice: str | None = Field(None, description="Voice ID (e.g., 'af_heart')")
    speed: float | None = Field(None, ge=0.25, le=4.0, description="Speech speed multiplier")


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
