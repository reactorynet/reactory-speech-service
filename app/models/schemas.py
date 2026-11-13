from typing import Any

from pydantic import AliasChoices, BaseModel, Field

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
    language: str | None = Field(
        default="en-us",
        description=(
            "Language tag or engine locale ('zh', 'zh-CN', 'cmn', 'en-gb'). Resolved "
            "through the voice registry, so 'zh' reaches the phonemizer as 'cmn'."
        ),
    )
    voice: str | None = Field(
        default=None,
        description=(
            "Optional voice id. When supplied, the voice's own engine locale is used "
            "instead of ``language``, so phonemization matches synthesis exactly."
        ),
    )
    custom_lexicon: dict[str, str] | None = Field(default=None, description="Optional pronunciation overrides for phonemization")


class PhonemizeResponse(BaseModel):
    text: str = Field(..., description="Original input text")
    normalized: str = Field(..., description="Normalized text after expansions")
    phonemes: str = Field(..., description="Phonetic (IPA) representation")
    language: str = Field(
        default="en-us",
        description="Engine locale actually used to phonemize (e.g. 'cmn' for Chinese)",
    )


class LexiconEntryRequest(BaseModel):
    word: str = Field(..., min_length=1, max_length=200, description="Word or phrase to match")
    replacement: str | None = Field(default=None, description="Phonetic respelling (e.g. 'koo-ber-net-eez')")
    ipa: str | None = Field(default=None, description="Direct International Phonetic Alphabet (IPA) representation")
    languages: list[str] | None = Field(
        default=None,
        description=(
            "Language tags this entry applies to (e.g. ['fr-FR']). Omit for an "
            "English entry: an unscoped respelling is not applied to other languages."
        ),
    )


class LexiconEntryResponse(BaseModel):
    word: str
    replacement: str | None = None
    ipa: str | None = None
    languages: list[str] | None = None
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
    provenance: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Signed provenance manifest, present only for cloned voices. Verification is a "
            "registry lookup on audioSha256; absence of a record is never 'not ours'."
        ),
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
    license: str | None = Field(
        default=None,
        description="Licence of the engine weights that render this voice (e.g. 'MIT', 'Apache-2.0')",
    )
    attribution: str | None = Field(
        default=None,
        description="Attribution required by the engine licence (e.g. 'Chatterbox (Resemble AI)')",
    )


class CloneVoiceInfo(VoiceInfo):
    """A cloned voice plus the governance facts a caller needs.

    Deliberately a **separate** model rather than extra fields on :class:`VoiceInfo`:
    the pack-voice payload is an existing contract, and adding nullable keys to it
    would change every row in ``GET /voices``.  Clone fields appear only where a
    clone is described.
    """

    clone: bool = Field(default=True, description="Whether this voice is a consent-gated clone")
    active: bool = Field(default=True, description="Whether the clone may be synthesised now")
    # ``_describe`` emits camelCase keys for the wire; without the alias Pydantic
    # silently dropped the reason and the field was always null.
    inactive_reason: str | None = Field(
        default=None, alias="inactiveReason", description="Why an inactive clone is inert"
    )
    consent_id: str | None = Field(default=None, description="Signed consent record id")
    created_at: str | None = Field(default=None, description="When the clone was created")
    created_by: str | None = Field(default=None, description="Who created the clone")
    approved_by: str | None = Field(default=None, description="Approver recorded on the consent")
    scope: str | None = Field(
        default=None, description="Recorded scope; NOT enforced by this service"
    )
    expires_at: str | None = Field(default=None, description="Consent expiry (ISO-8601)")
    reference_retained: bool = Field(
        default=False, description="Whether the reference clip was retained as an exception"
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


class VoiceComponent(BaseModel):
    """One ingredient of a blended voice."""

    voice: str = Field(
        ...,
        min_length=1,
        description="Shipped pack voice id (a designed voice is rejected with 422)",
    )
    weight: float = Field(
        ...,
        description=(
            "Relative weight. Weights must be finite and non-negative, sum to more than "
            "zero, and are normalised to 1 internally. One component must carry at "
            "least 0.6 of the total."
        ),
    )


class VoiceDesignRequest(BaseModel):
    """Body of ``POST /api/tts/voices/design``.

    A designed voice is a **recipe**: a weighted blend of 2-4 base style
    tensors.  Blending changes timbre, **not** emotion; an emotional delivery
    needs an expressive engine, not a blend.
    """

    name: str = Field(..., min_length=1, max_length=80, description="Display name; the id is derived from it")
    components: list[VoiceComponent] = Field(
        ...,
        description=(
            "2-4 shipped pack voices from one language family, with weights. A designed "
            "voice (blend or base alias) is not a valid component."
        ),
    )
    language: str | None = Field(
        default=None,
        description="Optional language assertion; must match the components' family",
    )
    direction: Direction | None = Field(
        default=None,
        validation_alias=AliasChoices("direction", "default_direction"),
        description=(
            "Optional default direction applied when a synthesis request names this voice "
            "and expresses no direction of its own. Kokoro applies rate natively; pitch "
            "is reported as ignored, exactly as for base voices."
        ),
    )
    tags: list[str] | None = Field(default=None, description="Optional discovery tags")


class BaseVoiceAliasRequest(BaseModel):
    """Body of ``POST /api/tts/voices`` — register an alias for a base voice.

    The alias stores ``style_spec: {type: base, voice: <pack id>}``; pack data is
    never copied or duplicated.
    """

    name: str = Field(..., min_length=1, max_length=80, description="Display name; the id is derived from it")
    base_voice: str = Field(
        ...,
        min_length=1,
        validation_alias=AliasChoices("base_voice", "voice", "base"),
        description="Shipped pack voice id to alias; a designed voice is rejected with 422",
    )
    language: str | None = Field(default=None, description="Optional language assertion")
    direction: Direction | None = Field(
        default=None,
        validation_alias=AliasChoices("direction", "default_direction"),
        description="Optional default direction",
    )
    tags: list[str] | None = Field(default=None, description="Optional discovery tags")


class VoiceDesignComponent(BaseModel):
    """A component of a designed voice, with its raw and normalised weight."""

    voice: str
    weight: float = Field(..., description="Weight as supplied and persisted")
    normalised_weight: float = Field(..., description="Weight divided by the total, so the set sums to 1")


class VoiceDesignDescriptor(BaseModel):
    """A designed voice plus the recipe metadata a client needs to explain it.

    ``components`` and ``normalised_weights`` are descriptive; the durable
    artefact is ``style_spec`` (the recipe). The resolved style tensor is never
    returned or persisted.
    """

    id: str
    name: str
    engine: str
    language: str
    locale: str
    gender: str
    quality: str | None = None
    tags: list[str] = Field(default_factory=list)
    style_type: str = Field(..., description="'blend' or 'base'")
    components: list[VoiceDesignComponent] = Field(default_factory=list)
    normalised_weights: dict[str, float] = Field(default_factory=dict)
    resolved_language: str
    created_at: str | None = None
    preview_url: str
    default_direction: dict[str, Any] | None = None
    preset: bool = False
    style_spec: dict[str, Any] = Field(
        ...,
        description="The persisted recipe; the tensor is derived from it, never stored",
    )


class VoiceDeleteResponse(BaseModel):
    status: str
    id: str


class CloneTakedownRequest(BaseModel):
    """Body of ``POST /api/tts/voices/{id}/takedown``.

    The service cannot model privilege (ADR-0003 #5): the caller is an
    authorising gateway and this body is the assertion it passes through.  The
    endpoint is internal-only and must not be exposed to untrusted callers.
    """

    requestedBy: str = Field(..., min_length=1, description="The assertion the service records")
    reason: str = Field(..., min_length=1, description="Why the takedown was requested")


class ConsentLookupResponse(BaseModel):
    """A consent record plus what its signature does and does not prove."""

    consentId: str | None = None
    record: dict[str, Any]
    signatureValid: bool
    verifiedAgainstInstanceId: str | None = None
    notice: str
    #: Record fields withheld from this response (the subject's identity).
    redactedFields: list[str] = Field(default_factory=list)


class ProvenanceLookupResponse(BaseModel):
    """Outcome of a manifest registry lookup.

    ``status`` is ``verified``, ``signature invalid``, ``verification
    unavailable`` or ``no record found``.  It is never ``not ours``.
    """

    status: str
    audioSha256: str
    manifest: dict[str, Any] | None = None


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
    languages: list[str] = Field(
        default_factory=list,
        description="Languages the pipeline can actually phonemise on this deployment",
    )
    unsupported_languages: list[str] = Field(
        default_factory=list,
        description=(
            "Languages the voice pack can render acoustically but the text front end "
            "cannot phonemise; requests for them return 422 rather than garbage"
        ),
    )
    languages_with_caveats: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Languages that phonemise but whose measured round-trip quality is not "
            "clean (quality='caveats'), with the caveat and the measured CER/WER"
        ),
    )
    language_support: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Measured G2P support and quality per language "
            "(language, locale, supported, quality, caveat, reason, measured)"
        ),
    )
    languages_pending: list[str] = Field(
        default_factory=list,
        description=(
            "Languages whose text front end has not been probed yet in this process. "
            "They are neither advertised nor refused until measured."
        ),
    )
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
    attribution: str | None = Field(
        default=None, description="Attribution required by the engine licence"
    )
    variant: str | None = Field(
        default=None, description="Engine variant/model size where one exists (e.g. 'standard', 'turbo')"
    )
    compute_cost: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Measured compute-cost advisory. At a realtime factor well above zero the "
            "routing signal is cost-per-second of audio, not native_rate: a caller can "
            "tell whether the engine suits batch narration or interactive turns."
        ),
    )
    capability_notes: list[str] | None = Field(
        default=None,
        description="Caveats a caller must know before routing to this engine",
    )
    supports_paralinguistic_tags: list[str] | None = Field(
        default=None,
        description="Paralinguistic tags the engine understands (e.g. '[laugh]')",
    )
    supports_exaggeration: bool | None = Field(
        default=None,
        description=(
            "Whether the deployed variant can actually honour exaggeration. False for "
            "Chatterbox Turbo, which accepts the argument and ignores it."
        ),
    )


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
    g2p: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Measured text-front-end (G2P) support: supported languages and the reason "
            "each unsupported language cannot be phonemised"
        ),
    )
    cloning: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Consent-first cloning availability and why it is unavailable. Absent or "
            "empty approver configuration means unavailable, never unrestricted."
        ),
    )


class ErrorResponse(BaseModel):
    error: str
    detail: str | None = None
