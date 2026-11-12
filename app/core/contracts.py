"""Engine-agnostic contracts for the speech platform.

This module is the boundary described by ADR-0002 and
``docs/architecture/speech-platform.md``.  It contains *only* data shapes and
the engine protocol: no HTTP, no FastAPI, no model loading and no voice data.

The public API lowers a :class:`Direction` into an engine's native controls;
each adapter reports back what it :class:`applied <DirectionPlan.applied>` and
what it could not, so a caller is never told a control was honoured when it was
silent.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Literal, Protocol, TypedDict, runtime_checkable

from pydantic import BaseModel, Field

#: Named styles the platform understands.  Engines advertise the subset they can
#: actually render through :attr:`EngineCapabilities.styles`.
STYLE_NAMES: tuple[str, ...] = (
    "neutral",
    "warm",
    "calm",
    "cheerful",
    "serious",
    "empathetic",
    "excited",
    "whisper",
    "narration",
)


class Prosody(BaseModel):
    """Mechanical delivery controls, expressed in provider-neutral units.

    Defaults describe "say it normally".  Because the defaults are also valid
    requests, callers that omit a field are distinguished from callers that ask
    for the default explicitly by pydantic's ``model_fields_set``; only controls
    the caller actually expressed are reported in ``applied`` / ``ignored``.
    """

    rate: float | None = Field(
        default=1.0,
        ge=0.5,
        le=2.0,
        description="Speech rate multiplier (0.5-2.0). 1.0 is the engine default.",
    )
    pitch: float | None = Field(
        default=0.0,
        ge=-12.0,
        le=12.0,
        description="Pitch shift in semitones; 0 is unchanged.",
    )
    volume: float | None = Field(
        default=0.0,
        ge=-12.0,
        le=12.0,
        description="Gain in dB; 0 is unchanged.",
    )
    pauses: Literal["none", "natural", "dramatic"] = Field(
        default="natural",
        description="Pause shaping: punctuation-driven, suppressed, or lengthened.",
    )


class Direction(BaseModel):
    """Provider-neutral expression request: style plus prosody.

    Callers express *intent*; they never name a model parameter.
    """

    style: str | None = Field(
        default="neutral",
        description="Named delivery style (e.g. 'warm', 'empathetic', 'narration').",
    )
    intensity: float | None = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="How strongly the style should be expressed (0-1).",
    )
    prosody: Prosody = Field(
        default_factory=Prosody,
        description="Rate, pitch, volume and pause shaping.",
    )


class EngineCapabilities(BaseModel):
    """What one engine can and cannot do.

    This is the capability matrix exposed at ``GET /api/tts/capabilities`` so a
    caller can decide whether to route elsewhere instead of silently receiving
    neutral speech.
    """

    engine: str
    languages: list[str] = Field(default_factory=list)
    #: Languages the voice pack advertises but the text front end cannot
    #: phonemise. They are not in ``languages``; requests for them return 422.
    unsupported_languages: list[str] = Field(default_factory=list)
    #: Languages that phonemise but whose round-trip quality is not clean:
    #: ``{language, quality, caveat, measured}``. Advertising them as plain
    #: language support would repeat the SPEECH-010 mistake one level down.
    languages_with_caveats: list[dict] = Field(default_factory=list)
    #: Measured G2P support *and* quality rows:
    #: ``{language, locale, supported, quality, caveat, reason, measured}``.
    language_support: list[dict] = Field(default_factory=list)
    #: Advertised languages whose text front end has not been probed yet in this
    #: process. They are neither advertised nor refused until measured.
    languages_pending: list[str] = Field(default_factory=list)
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


class EngineRequest(BaseModel):
    """A fully resolved synthesis request handed to an engine adapter.

    ``voice`` is the caller-facing id; the adapter resolves it through the voice
    registry to its own native voice key.  ``direction`` is already merged with
    the legacy ``speed`` alias by the orchestrator.
    """

    text: str
    voice: str | None = None
    language: str | None = None
    direction: Direction = Field(default_factory=Direction)
    is_phonemes: bool = False
    phonetic_preprocess: bool | None = None
    custom_lexicon: dict[str, str] | None = None


class AudioResult(BaseModel):
    """Rendered audio plus an honest account of the direction that was applied."""

    audio: bytes
    duration: float
    sample_rate: int
    engine: str
    voice: str | None = None
    applied: dict[str, Any] = Field(default_factory=dict)
    ignored: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}


class AudioChunk(BaseModel):
    """One streamed slice of rendered audio."""

    audio: bytes
    duration: float
    sample_rate: int

    model_config = {"arbitrary_types_allowed": True}


class DirectionPlan(BaseModel):
    """The outcome of lowering a :class:`Direction` for one engine.

    Computed before inference when a caller only needs to know what *will* be
    honoured, and embedded in :class:`AudioResult` for the response contract.
    """

    engine: str
    applied: dict[str, Any] = Field(default_factory=dict)
    ignored: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class VoiceDescriptor(TypedDict):
    """Presentation metadata for a single voice, as served by the voice API.

    ``quality`` is ``None`` when the catalogue of record does not grade the
    voice; ``locale`` is the code handed to the engine for phonemisation.
    """

    id: str
    name: str
    language: str
    locale: str
    gender: str
    engine: str
    quality: str | None
    tags: list[str]


@runtime_checkable
class TTSEngine(Protocol):
    """The one contract every engine adapter implements.

    Adapters own model lifecycle, native parameter mapping and inference.  They
    must not know about HTTP or the pydantic request/response models.
    """

    id: str

    def capabilities(self) -> EngineCapabilities:
        """Describe what this engine can render natively."""
        ...

    def voices(self) -> list[VoiceDescriptor]:
        """List the voices this engine exposes, enriched from the registry."""
        ...

    def synthesize(self, req: EngineRequest) -> AudioResult:
        """Render a request to a complete audio buffer."""
        ...

    def stream(self, req: EngineRequest) -> Iterator[AudioChunk]:
        """Render a request incrementally."""
        ...
