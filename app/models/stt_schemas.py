"""Request/response models for the speech-to-text surface.

These live outside ``app.models.schemas`` on purpose: the STT payload grew a
nested ``words`` array, detected-language probabilities and model metadata, and
keeping the STT models in their own module keeps the text-to-speech models (and
their import graph) untouched.
"""

from pydantic import BaseModel, Field


class STTWord(BaseModel):
    """One word with its position on the source timeline."""

    word: str = Field(..., description="The word as transcribed")
    start: float = Field(..., description="Word start in seconds")
    end: float = Field(..., description="Word end in seconds")
    probability: float | None = Field(
        default=None, description="Model confidence for this word (0-1)"
    )


class STTSegment(BaseModel):
    start: float = Field(..., description="Segment start time in seconds")
    end: float = Field(..., description="Segment end time in seconds")
    text: str = Field(..., description="Segment text")
    words: list[STTWord] | None = Field(
        default=None,
        description="Word-level timestamps; only present when word_timestamps=true was requested",
    )


class STTLanguageProbability(BaseModel):
    language: str = Field(..., description="ISO language code")
    probability: float = Field(..., description="Detected probability for that language (0-1)")


class STTTranscriptionResult(BaseModel):
    text: str = Field(..., description="Full transcription text")
    language: str | None = Field(
        default=None, description="Detected or requested language; null when the clip held no speech"
    )
    language_confidence: float | None = Field(
        default=None,
        description="Probability assigned to the detected language; low values mean the client should ask",
    )
    language_probabilities: list[STTLanguageProbability] | None = Field(
        default=None, description="The top detected languages, highest probability first"
    )
    needs_language_confirmation: bool = Field(
        default=False,
        description="True when language_confidence is below the configured confirmation threshold",
    )
    segments: list[STTSegment] = Field(default_factory=list)
    duration: float = Field(
        default=0.0,
        description=(
            "Duration of the submitted audio in seconds. Not shortened by VAD trimming."
        ),
    )
    model: str | None = Field(default=None, description="Model id used for this transcription")
    vad_applied: bool = Field(
        default=False, description="Whether silence was trimmed by voice-activity detection"
    )
    vad_failed: bool = Field(
        default=False,
        description=(
            "True when voice-activity detection could not run (missing module/asset, "
            "decoder error). The audio was then transcribed untrimmed; this is NOT "
            "reported as an empty transcript."
        ),
    )
    speech_duration: float | None = Field(
        default=None, description="Seconds of speech remaining after VAD trimming"
    )


class STTVADInfo(BaseModel):
    enabled: bool
    available: bool
    threshold: float
    min_silence_ms: int
    min_speech_ms: int
    pad_ms: int


class STTCapabilitiesResponse(BaseModel):
    """What this deployment can do, and which models it can actually load."""

    ready: bool
    default_model: str | None = None
    available_models: list[str] = Field(default_factory=list)
    loaded_models: list[str] = Field(default_factory=list)
    device: str = "auto"
    compute_type: str = "auto"
    streaming: bool = True
    word_timestamps: bool = True
    language_confidence: bool = True
    vad: STTVADInfo
