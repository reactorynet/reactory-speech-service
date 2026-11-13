"""TTS router — REST and WebSocket endpoints for text-to-speech and lexicon management.

The HTTP layer owns transport, validation and backwards compatibility only.  It
resolves the caller's direction into the provider-neutral contract, asks the
orchestrator what the engine will actually honour, and returns that honestly in
``direction_applied`` / ``direction_ignored`` (headers on the WAV response, body
fields on the JSON response).
"""

from __future__ import annotations

import base64
import functools
import json
import logging
from typing import Any

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.core.contracts import Direction, DirectionPlan
from app.core.direction import resolve_direction
from app.engines.base import EngineUnavailableError
from app.models.schemas import (
    BaseVoiceAliasRequest,
    CapabilitiesResponse,
    CloneTakedownRequest,
    CloneVoiceInfo,
    ConsentLookupResponse,
    EngineCapabilities,
    ErrorResponse,
    LexiconEntryRequest,
    LexiconEntryResponse,
    LexiconListResponse,
    PhonemizeRequest,
    PhonemizeResponse,
    ProvenanceLookupResponse,
    TTSRequest,
    TTSResponse,
    VoiceDeleteResponse,
    VoiceDesignDescriptor,
    VoiceDesignRequest,
    VoiceInfo,
    VoicesResponse,
)
from app.services.designed_voices import VoiceConflictError, VoiceRecipeError
from app.services.phonetic_converter import (
    PhoneticConversionError,
    UnsupportedLanguageError,
    phonetic_converter,
)
from app.services.inference import InferenceExecutor
from app.services.tts_service import ENGINE_ID, SAMPLE_RATE, SynthesisOutcome
from app.services.voice_design import VoiceDesignService, default_direction
from app.services.voice_cloning import (
    ApproverNotAllowedError,
    CloneInactiveError,
    CloneNotFoundError,
    CloneStoreError,
    CloneValidationError,
    CloningUnavailableError,
    get_clone_service,
)
from app.services.voice_registry import filter_catalogue

logger = logging.getLogger(__name__)

router = APIRouter()


def _audio_headers(duration: float) -> dict[str, str]:
    """Audio metadata headers, emitted under both canonical and alias names.

    The service historically sent ``X-Audio-Sample-Rate`` while clients read
    ``x-sample-rate``, silently falling back to a default.  Both spellings are
    emitted — on the binary *and* the JSON synthesis endpoint — so existing
    clients work unchanged regardless of which one they read.
    """
    sample_rate = str(SAMPLE_RATE)
    duration_str = str(round(duration, 3))
    return {
        "X-Sample-Rate": sample_rate,
        "X-Audio-Sample-Rate": sample_rate,
        "X-Duration": duration_str,
        "X-Audio-Duration": duration_str,
        "X-Format": "wav",
        "X-Audio-Format": "wav",
    }


def _ascii_header_value(value: str) -> str:
    """Header values must be latin-1; keep warnings ASCII so they always send."""
    return str(value).encode("ascii", "replace").decode("ascii")


def _can_serve(tts_service: Any) -> bool:
    """Whether a request may be admitted to the TTS service.

    ``can_serve`` means "an engine is registered, so it can be loaded"; it is
    deliberately weaker than ``is_ready`` ("a model is resident"), because a
    lazily loaded or idle-unloaded engine still serves the request.  A service
    double without the property falls back to ``is_ready`` so a 503 still means
    "TTS unavailable" in unit tests and in older deployments.
    """
    value = getattr(tts_service, "can_serve", None)
    if isinstance(value, bool):
        return value
    return bool(getattr(tts_service, "is_ready", False))


def _cache_headers(outcome: SynthesisOutcome) -> dict[str, str]:
    """Report cache behaviour on the response.

    ``X-Cache`` is ``hit`` or ``miss``; ``X-Cache-Tier`` explains where a hit
    came from (``lru``, ``disk``) or ``none`` when inference ran.  A request that
    disables the cache reports ``miss`` / ``none`` rather than inventing a third
    state, so clients can treat the header as a boolean.
    """
    return {"X-Cache": outcome.cache, "X-Cache-Tier": outcome.cache_tier}


def _provenance_headers(outcome: SynthesisOutcome) -> dict[str, str]:
    """Point at the signed manifest for a cloned render.

    The manifest itself is emitted in the JSON body; the WAV response carries
    only its id, which is the registry lookup key (``audioSha256``).  Nothing is
    written into the audio: provenance is out of band by decision 4a.
    """
    if not outcome.manifest:
        return {}
    digest = str(outcome.manifest.get("audioSha256") or "")
    return {
        "X-Provenance-Id": digest,
        "X-Provenance-Signed": "true",
        "X-Provenance-Lookup": f"/api/tts/provenance/{digest}",
    }


def _direction_headers(plan: DirectionPlan) -> dict[str, str]:
    """Emit the applied/ignored contract on an audio response."""
    headers = {
        "X-Engine": _ascii_header_value(plan.engine),
        "X-Direction-Applied": _ascii_header_value(
            json.dumps(plan.applied, ensure_ascii=True, separators=(",", ":"))
        ),
        "X-Direction-Ignored": _ascii_header_value(",".join(plan.ignored)),
    }
    if plan.warnings:
        headers["X-Direction-Warnings"] = _ascii_header_value(" | ".join(plan.warnings))
    return headers


def _plan_for(
    tts_service: Any, body: TTSRequest
) -> tuple[Direction, DirectionPlan, list[str]]:
    """Merge the legacy ``speed`` alias and ask the orchestrator to lower it.

    Returns the effective direction, the lowering plan and any alias notes (the
    ``speed`` vs ``direction.prosody.rate`` precedence message).

    A designed voice may carry a *default* direction.  It is used only when the
    request expresses nothing itself, so an explicit ``speed`` or ``direction``
    always wins and the plan honestly reflects what synthesis will apply.
    """
    requested = body.direction
    if requested is None and body.speed is None:
        requested = default_direction(body.voice)
    direction, alias_warnings = resolve_direction(body.speed, requested)
    plan = tts_service.plan_direction(
        voice=body.voice,
        engine=body.engine,
        direction=direction,
    )
    if alias_warnings:
        plan = plan.model_copy(update={"warnings": [*alias_warnings, *plan.warnings]})
    return direction, plan, alias_warnings


def _synthesis_kwargs(body: TTSRequest) -> dict[str, Any]:
    """Legacy optional arguments, forwarded only when the caller supplied them."""
    kwargs: dict[str, Any] = {}
    if body.is_phonemes:
        kwargs["is_phonemes"] = True
    if body.phonetic_preprocess is not None:
        kwargs["phonetic_preprocess"] = body.phonetic_preprocess
    if body.custom_lexicon is not None:
        kwargs["custom_lexicon"] = body.custom_lexicon
    return kwargs


async def _render(tts_service: Any, body: TTSRequest) -> SynthesisOutcome:
    """Render audio plus the direction contract, off the event loop.

    A request that sends neither ``direction`` nor ``engine`` takes the unchanged
    legacy argument shape — same call, same keyword arguments, same bytes — but
    the call is now awaited so inference runs in the engine's thread pool rather
    than in the request handler.
    """
    _direction, plan, alias_warnings = _plan_for(tts_service, body)
    kwargs = _synthesis_kwargs(body)

    # Only forward the additive arguments the caller actually sent, so a legacy
    # request stays visibly legacy (and behaves exactly as it did before).
    forwarded: dict[str, Any] = {}
    if body.direction is not None:
        forwarded["direction"] = body.direction
    if body.engine is not None:
        forwarded["engine"] = body.engine
    if not body.cache:
        forwarded["cache"] = False

    outcome = await tts_service.synthesize_async(
        body.text,
        voice=body.voice,
        speed=body.speed,
        **forwarded,
        **kwargs,
    )

    applied_plan = DirectionPlan(
        engine=outcome.plan.engine or plan.engine,
        applied=outcome.plan.applied,
        ignored=outcome.plan.ignored,
        warnings=[*alias_warnings, *outcome.plan.warnings],
    )
    return SynthesisOutcome(
        audio=outcome.audio,
        duration=outcome.duration,
        plan=applied_plan,
        cache=outcome.cache,
        cache_tier=outcome.cache_tier,
        cache_key=outcome.cache_key,
        engine=outcome.engine,
        manifest=outcome.manifest,
    )


# ─────────────────────────────────────────────────────────────
# TTS Synthesis Endpoints
# ─────────────────────────────────────────────────────────────

@router.post(
    "/synthesize",
    responses={
        200: {"content": {"audio/wav": {}}},
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def synthesize(request: Request, body: TTSRequest) -> Response:
    """Synthesize text or phonemes to speech, returning WAV audio."""
    tts_service = request.app.state.tts_service
    if not _can_serve(tts_service):
        raise HTTPException(status_code=503, detail="TTS service not available")

    try:
        outcome = await _render(tts_service, body)
        return Response(
            content=outcome.audio,
            media_type="audio/wav",
            headers={
                **_audio_headers(outcome.duration),
                **_direction_headers(outcome.plan),
                **_cache_headers(outcome),
                **_provenance_headers(outcome),
            },
        )
    except CloneInactiveError as e:
        # An inert clone is gone, not unknown: 410 tells the caller the record
        # exists but may not be used (takedown or expired consent).
        raise HTTPException(status_code=410, detail=str(e))
    except CloningUnavailableError as e:
        logger.warning("Cloned voice unavailable: %s", e)
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except EngineUnavailableError as e:
        logger.warning("TTS engine unavailable: %s", e)
        raise HTTPException(status_code=503, detail=str(e))
    except Exception:
        logger.exception("TTS synthesis failed")
        raise HTTPException(status_code=500, detail="Synthesis failed")


@router.post("/synthesize/json", response_model=TTSResponse)
async def synthesize_json(
    request: Request, body: TTSRequest, response: Response
) -> TTSResponse:
    """Synthesize text or phonemes to speech, returning base64-encoded audio in JSON.

    The same audio metadata is returned in the body *and* as ``X-Sample-Rate`` /
    ``X-Audio-Sample-Rate`` (plus the duration and format aliases), so a consumer
    can read either endpoint the same way.  The applied/ignored direction contract
    is returned in the body and in the ``X-Direction-*`` headers.
    """
    tts_service = request.app.state.tts_service
    if not _can_serve(tts_service):
        raise HTTPException(status_code=503, detail="TTS service not available")

    try:
        outcome = await _render(tts_service, body)
        response.headers.update(
            {
                **_audio_headers(outcome.duration),
                **_direction_headers(outcome.plan),
                **_cache_headers(outcome),
                **_provenance_headers(outcome),
            }
        )
        return TTSResponse(
            audio_base64=base64.b64encode(outcome.audio).decode("ascii"),
            duration=round(outcome.duration, 3),
            format="wav",
            sample_rate=SAMPLE_RATE,
            engine=outcome.plan.engine,
            direction_applied=outcome.plan.applied,
            direction_ignored=outcome.plan.ignored,
            direction_warnings=outcome.plan.warnings,
            cache=outcome.cache,
            cache_tier=outcome.cache_tier,
            provenance=outcome.manifest,
        )
    except CloneInactiveError as e:
        raise HTTPException(status_code=410, detail=str(e))
    except CloningUnavailableError as e:
        logger.warning("Cloned voice unavailable: %s", e)
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except EngineUnavailableError as e:
        logger.warning("TTS engine unavailable: %s", e)
        raise HTTPException(status_code=503, detail=str(e))
    except Exception:
        logger.exception("TTS synthesis failed")
        raise HTTPException(status_code=500, detail="Synthesis failed")


# ─────────────────────────────────────────────────────────────
# Voice Catalogue Endpoints
# ─────────────────────────────────────────────────────────────

@router.get("/voices", response_model=VoicesResponse)
async def list_voices(
    request: Request,
    engine: str | None = Query(None, description="Filter by engine id (e.g. 'kokoro')"),
    language: str | None = Query(
        None, description="Filter by language tag or engine code (e.g. 'en-US', 'en-us', 'ja')"
    ),
    gender: str | None = Query(None, description="Filter by gender ('female' or 'male')"),
    tag: str | None = Query(
        None, description="Filter by tag (e.g. 'recommended', 'grade-c-plus', 'unrated')"
    ),
) -> VoicesResponse:
    """List the voices exposed by the loaded engine, with optional filters.

    ``available`` reports whether the engine is actually loaded: when it is not,
    the catalogue is the registry's manifest and synthesis will return 503.

    Every row carries ``license`` and ``attribution``, including the Kokoro rows
    with the flag off: each registry record declares its own provenance, so the
    keys are additive rather than engine-specific. That makes the
    permissive-licence requirement (ADR-0001) checkable from one response without
    a second lookup. Clients written before those keys existed are unaffected —
    they are additional fields on the row.
    """
    tts_service = request.app.state.tts_service
    voices = filter_catalogue(
        tts_service.get_voices(),
        engine=engine,
        language=language,
        gender=gender,
        tag=tag,
    )
    return VoicesResponse(
        voices=voices,
        total=len(voices),
        engine=ENGINE_ID,
        available=bool(tts_service.is_ready),
    )


@router.get("/voices/{voice_id}", response_model=VoiceInfo | CloneVoiceInfo)
async def get_voice(
    voice_id: str, request: Request
) -> VoiceInfo | CloneVoiceInfo:
    """Describe a single voice, or 404 when it is not present in the voice pack.

    A clone carries its engine, licence, consent id, creator, approver, recorded
    scope and expiry.  A disabled or expired clone is **410**, not 404: the
    record exists, but synthesising it is refused.
    """
    tts_service = request.app.state.tts_service
    # Clones are resolved through the clone store first: `get_voice` on the
    # service also knows them, but only the store can say whether the clone is
    # active, and a known-but-inert clone must be 410 rather than 200.
    clone = get_clone_service().describe(voice_id)
    if clone is not None:
        if not clone.get("active", False):
            raise HTTPException(
                status_code=410,
                detail=(
                    f"Voice '{voice_id}' is not active: "
                    f"{clone.get('inactiveReason') or 'it cannot be synthesised'}"
                ),
            )
        return CloneVoiceInfo(**clone)

    voice = tts_service.get_voice(voice_id)
    if voice is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Unknown voice '{voice_id}'. GET /api/tts/voices lists the "
                f"{len(tts_service.voice_ids)} voices available on this deployment."
            ),
        )
    return VoiceInfo(**voice)


#: Keys this change added to an engine's capability row.  They are dropped from a
#: row when their value is ``None`` so an engine that does not declare them (every
#: Kokoro row) keeps exactly the payload shape it had before the expressive engine
#: existed.  Deliberately an explicit list rather than
#: ``response_model_exclude_none=True``: that option also removes pre-existing
#: ``None`` values — ``default_voice`` and the ``caveat: null`` entries inside
#: ``language_support`` / ``languages_with_caveats`` — which would change the
#: existing response for every client.
_ADDITIVE_ENGINE_KEYS: tuple[str, ...] = (
    "attribution",
    "variant",
    "compute_cost",
    "capability_notes",
    "supports_paralinguistic_tags",
    "supports_exaggeration",
)


@router.get("/capabilities", response_model=CapabilitiesResponse)
async def capabilities(request: Request) -> CapabilitiesResponse | JSONResponse:
    """Report the capability matrix of every TTS engine in this deployment.

    With the expressive engine registered, its row carries a measured
    ``compute_cost`` advisory: at RTF 0.13 the routing signal is cost per second
    of audio, not ``native_rate``.

    The payload is pruned by :data:`_ADDITIVE_ENGINE_KEYS` rather than by
    excluding every ``None``, so adding those keys did not alter the existing
    Kokoro row (see ``test_capabilities_flag_off_payload_is_unchanged``).
    """
    tts_service = request.app.state.tts_service
    response = CapabilitiesResponse(
        engines=[EngineCapabilities(**row) for row in tts_service.capability_matrix()]
    )
    payload = response.model_dump()
    pruned = False
    for row in payload["engines"]:
        for key in _ADDITIVE_ENGINE_KEYS:
            if key in row and row[key] is None:
                del row[key]
                pruned = True
    return JSONResponse(content=payload) if pruned else response


# ─────────────────────────────────────────────────────────────
# Designed voices (SPEECH-003)
#
# Blending changes TIMBRE, not EMOTION.  Kokoro has no emotion control, so a
# designed voice is a different vocal identity, never a delivery style.
# ─────────────────────────────────────────────────────────────

@router.post(
    "/voices/design",
    response_model=VoiceDesignDescriptor,
    status_code=201,
    responses={
        409: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
    },
)
async def design_voice(request: Request, body: VoiceDesignRequest) -> VoiceDesignDescriptor:
    """Design a new voice by blending 2-4 base Kokoro style tensors.

    The recipe is persisted; the resolved tensor is derived and cached, so no
    model is copied, re-downloaded or retrained.  Blending changes **timbre**,
    not **emotion** — it cannot make a voice sound empathetic, calm or cheerful.
    Emotion needs an expressive engine (SPEECH-005).

    Rules: 2-4 components, finite non-negative weights summing above zero,
    one language family, one dominant component at 0.6 or more of the weight,
    and every component a **shipped pack voice**.  A designed voice (blend or
    base alias) is not a valid ingredient: composing user-managed entries would
    let deleting one silently break another, so it is rejected with 422.
    """
    service = VoiceDesignService(request.app.state.tts_service)
    try:
        descriptor = await service.create_blend(
            name=body.name,
            components=[component.model_dump() for component in body.components],
            language=body.language,
            direction=body.direction,
            tags=body.tags,
        )
    except VoiceConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except VoiceRecipeError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return VoiceDesignDescriptor(**descriptor)


@router.post(
    "/voices",
    response_model=VoiceDesignDescriptor,
    status_code=201,
    responses={
        409: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
    },
)
async def register_base_voice(
    request: Request, body: BaseVoiceAliasRequest
) -> VoiceDesignDescriptor:
    """Register a product alias for a shipped base voice.

    The registry entry stores ``style_spec: {type: base, voice: <pack id>}``; the
    pack's audio data is never copied or duplicated.  This is a naming tool, not
    a second voice: the audio is byte-for-byte the base voice's.

    The target must be a **shipped pack voice**.  Aliasing a designed voice is
    rejected with 422: the alias would advertise a second name for a recipe that
    can be deleted underneath it.
    """
    service = VoiceDesignService(request.app.state.tts_service)
    try:
        descriptor = await service.create_base_alias(
            name=body.name,
            base_voice=body.base_voice,
            language=body.language,
            direction=body.direction,
            tags=body.tags,
        )
    except VoiceConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except VoiceRecipeError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return VoiceDesignDescriptor(**descriptor)


@router.get(
    "/voices/{voice_id}/design",
    response_model=VoiceDesignDescriptor,
    responses={404: {"model": ErrorResponse}},
)
async def designed_voice_recipe(voice_id: str, request: Request) -> VoiceDesignDescriptor:
    """Return a designed voice's recipe and design metadata.

    The recipe is the durable artefact; the resolved style tensor is never
    returned.  Blending changes timbre, not emotion.
    """
    service = VoiceDesignService(request.app.state.tts_service)
    descriptor = service.descriptor(voice_id)
    if descriptor is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Unknown designed voice '{voice_id}'. "
                "GET /api/tts/voices lists the available voices."
            ),
        )
    return VoiceDesignDescriptor(**descriptor)


@router.get(
    "/voices/{voice_id}/preview",
    responses={
        200: {"content": {"audio/wav": {}}},
        404: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def designed_voice_preview(voice_id: str, request: Request) -> Response:
    """Serve a designed voice's preview WAV, synthesising only on a cache miss.

    The preview is cached by the recipe hash: a cache hit does not run
    inference, and editing the recipe invalidates the cached WAV.
    """
    service = VoiceDesignService(request.app.state.tts_service)
    try:
        result = await service.ensure_preview(voice_id)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Unknown designed voice '{voice_id}'. "
                "GET /api/tts/voices lists the available voices."
            ),
        )
    except EngineUnavailableError as exc:
        logger.warning("Preview unavailable: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    duration = float(result.get("duration", 0.0))
    return Response(
        content=result["audio"],
        media_type="audio/wav",
        headers={
            "X-Preview-Cache": result["cache"],
            "X-Sample-Rate": str(result.get("sample_rate", SAMPLE_RATE)),
            "X-Duration": str(round(duration, 3)),
            "X-Format": "wav",
        },
    )


@router.delete(
    "/voices/{voice_id}",
    response_model=VoiceDeleteResponse,
    responses={
        404: {"model": ErrorResponse},
        409: {"model": ErrorResponse},
    },
)
async def delete_voice(
    voice_id: str,
    request: Request,
    requested_by: str | None = Query(None, alias="requestedBy"),
    reason: str | None = Query(None),
) -> VoiceDeleteResponse:
    """Delete a designed voice, or a clone.

    Deleting a **clone** removes its audio, embedding, registry entry and any
    preview, and records who asked, when and why.  It succeeds when a file is
    already missing — a corrupt or absent clip must not block removal of the
    reference to it.  The audit entry contains no audio and no hash of it.

    Pack voices and shipped presets remain immutable (409).
    """
    clone_service = get_clone_service()
    if clone_service.describe(voice_id) is not None:
        try:
            result = clone_service.delete(
                voice_id, requested_by=requested_by or "", reason=reason or ""
            )
        except CloneNotFoundError:
            raise HTTPException(status_code=404, detail=f"Unknown clone '{voice_id}'.")
        except CloneValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except CloneStoreError as exc:
            # No request-derived value in the log line (py/log-injection); the
            # id is in the response and in the clone audit log.
            logger.exception("Could not delete a clone.")
            raise HTTPException(status_code=500, detail=str(exc))
        return VoiceDeleteResponse(status=result["status"], id=result["id"])

    service = VoiceDesignService(request.app.state.tts_service)
    try:
        service.delete(voice_id)
    except VoiceConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Unknown voice '{voice_id}'. "
                "GET /api/tts/voices lists the available voices."
            ),
        )
    return VoiceDeleteResponse(status="deleted", id=voice_id)


# ─────────────────────────────────────────────────────────────
# Consent-first voice cloning (SPEECH-006 / ADR-0003)
#
# Cloning is a governance feature, not a model feature. The clone itself is one
# library call; the consent record, retention rule, provenance chain and
# takedown path are the product.
# ─────────────────────────────────────────────────────────────

#: Reference uploads are a short clip, not an archive. Bounded to keep a single
#: request from filling the node's disk.
MAX_REFERENCE_BYTES = 25 * 1024 * 1024


def _parse_consent(raw: str | None) -> dict[str, Any]:
    if raw is None or not raw.strip():
        # Empty is not a malformed object: let the service name every missing
        # field individually rather than inventing one generic error here.
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=422, detail=f"consent must be a JSON object: {exc}"
        )
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="consent must be a JSON object")
    return payload


@router.post(
    "/voices/clone",
    response_model=CloneVoiceInfo,
    status_code=201,
    responses={
        403: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def create_voice_clone(
    request: Request,
    reference_audio: UploadFile = File(..., description="Short reference clip (WAV)"),
    consent: str | None = Form(
        None,
        description=(
            "JSON object with subjectName, purpose, capturedAt, expiresAt, approver, "
            "evidence and scope. Missing fields are reported individually."
        ),
    ),
    created_by: str = Form("unspecified", description="Who is creating the clone"),
    retain_reference: bool = Form(
        False, description="Retain the clip as a recorded exception (<= 90 days)"
    ),
    retention_reason: str | None = Form(None),
    retention_days: int | None = Form(None),
) -> CloneVoiceInfo:
    """Create a consent-first voice clone from a reference clip.

    The reference clip is deleted as soon as the embedding is extracted unless
    retention is explicitly requested.  Retention is capped at 90 days and needs
    a reason on the consent record.

    Refusals: 422 for a missing/invalid field or an out-of-enumeration scope,
    403 for an approver outside the deployed allowlist, 503 when cloning is not
    configured on this instance (absent allowlist or keys).
    """
    audio = await reference_audio.read()
    if len(audio) > MAX_REFERENCE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"reference_audio is larger than {MAX_REFERENCE_BYTES} bytes",
        )
    service = get_clone_service()
    try:
        descriptor = service.create(
            audio,
            _parse_consent(consent),
            created_by=created_by,
            retain_reference=retain_reference,
            retention_reason=retention_reason,
            retention_days=retention_days,
        )
    except ApproverNotAllowedError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except CloneValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except CloningUnavailableError as exc:
        logger.warning("Clone creation refused: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc))
    except CloneStoreError as exc:
        logger.exception("Clone creation failed")
        raise HTTPException(status_code=500, detail=str(exc))
    return CloneVoiceInfo(**descriptor)


@router.post(
    "/voices/{voice_id}/takedown",
    response_model=CloneVoiceInfo,
    responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
async def takedown_voice(
    voice_id: str, body: CloneTakedownRequest, request: Request
) -> CloneVoiceInfo:
    """Disable a clone and remove its artefacts, effective immediately.

    **Internal-only.** This endpoint must sit behind the authorising gateway; it
    is not for untrusted callers. The service cannot model privilege
    (ADR-0003 #5), so it records the assertion it is given (who, when, why) and
    does not verify the authority behind it. Once applied the clone is inert
    without a redeploy: 410 on synthesis, absent from ``/voices``, and an
    in-flight render is discarded.
    """
    try:
        descriptor = get_clone_service().takedown(
            voice_id, requested_by=body.requestedBy, reason=body.reason
        )
    except CloneNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except CloneValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return CloneVoiceInfo(**descriptor)


@router.get(
    "/voices/{voice_id}/consent",
    response_model=ConsentLookupResponse,
    responses={404: {"model": ErrorResponse}},
)
async def clone_consent(voice_id: str) -> ConsentLookupResponse:
    """Return a clone's consent record and whether its instance signature verifies.

    The signature is chain of custody: it proves which instance recorded the
    consent and that it is unaltered.  It does **not** prove the subject
    consented; the response says so in ``notice``.
    """
    result = get_clone_service().consent_record(voice_id)
    if result is None:
        raise HTTPException(
            status_code=404, detail=f"No consent record for clone '{voice_id}'."
        )
    record = result["record"]
    return ConsentLookupResponse(
        consentId=record.get("consentId"),
        record=record,
        signatureValid=result["signatureValid"],
        verifiedAgainstInstanceId=result["verifiedAgainstInstanceId"],
        notice=result["notice"],
    )


@router.post(
    "/provenance/verify",
    response_model=ProvenanceLookupResponse,
    responses={404: {"model": ErrorResponse}, 422: {"model": ErrorResponse}},
)
async def verify_provenance(
    file: UploadFile = File(..., description="The audio file to look up")
) -> JSONResponse:
    """Look up a file's signed manifest by hashing its **decoded PCM**.

    A miss returns 404 with the status ``no record found``.  It never says the
    file is foreign: absence of a record is not evidence of absence.
    """
    audio = await file.read()
    if len(audio) > MAX_REFERENCE_BYTES:
        # The lookup decodes the whole file into memory; cap it like an upload.
        raise HTTPException(
            status_code=413,
            detail=f"file is larger than {MAX_REFERENCE_BYTES} bytes",
        )
    try:
        result = get_clone_service().lookup_provenance(audio)
    except (ValueError, RuntimeError, OSError) as exc:
        # Only decode failures are the caller's fault. An internal error must not
        # be reported as "your file was undecodable".
        raise HTTPException(
            status_code=422, detail=f"The file could not be decoded as audio: {exc}"
        )
    status_code = 404 if result["manifest"] is None else 200
    return JSONResponse(status_code=status_code, content=result)


@router.get(
    "/provenance/{audio_sha256}",
    response_model=ProvenanceLookupResponse,
    responses={404: {"model": ErrorResponse}},
)
async def lookup_provenance(audio_sha256: str) -> JSONResponse:
    """Look up a manifest by ``audioSha256`` (the registry key)."""
    result = get_clone_service().lookup_by_digest(audio_sha256.strip().lower())
    status_code = 404 if result["manifest"] is None else 200
    return JSONResponse(status_code=status_code, content=result)


@router.post(
    "/phonemize",
    response_model=PhonemizeResponse,
    responses={
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def phonemize_text(request: Request, body: PhonemizeRequest) -> PhonemizeResponse:
    """Convert text into phonetic (IPA) writing using normalization and phonetic conversion.

    Grapheme-to-phoneme conversion is CPU work over a process-global espeak-ng
    backend, so it is dispatched to the inference pool instead of running on the
    event loop.

    The locale is the **voice's** engine locale when ``voice`` is supplied, and
    otherwise the caller's language tag resolved through the voice registry, so
    ``zh`` reaches the backend as ``cmn``. A language the backend cannot read is a
    ``422`` — the alternative is a phoneme string that spells the text out.
    """
    tts_service = request.app.state.tts_service
    call = functools.partial(
        tts_service.phonemize,
        body.text,
        language=body.language or "en-us",
        custom_lexicon=body.custom_lexicon,
        voice=body.voice,
    )
    try:
        executor = getattr(tts_service, "executor", None)
        if isinstance(executor, InferenceExecutor):
            result = await executor.run(call)
        else:
            # A service double (or an older deployment) has no inference pool.
            result = call()
        return PhonemizeResponse(**result)
    except (UnsupportedLanguageError, PhoneticConversionError, ValueError) as e:
        # Unsupported language, or text the phonemizer cannot convert: both are
        # the caller's input, not a server fault.
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        logger.exception("Phonemization failed")
        raise HTTPException(status_code=500, detail="Phonemization failed")
@router.websocket("/stream")
async def tts_stream(websocket: WebSocket) -> None:
    """WebSocket endpoint for streaming TTS.

    Binary audio frames and the trailing ``{"done": true, ...}`` message are
    unchanged.  An optional per-message ``direction`` is lowered before
    synthesis, and the applied/ignored contract is added to the ``done`` frame.
    """
    await websocket.accept()
    tts_service = websocket.app.state.tts_service

    if not _can_serve(tts_service):
        await websocket.send_json({"error": "TTS service not available"})
        await websocket.close(code=1011)
        return

    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON"})
                continue

            text = msg.get("text", "").strip()
            if not text:
                await websocket.send_json({"error": "Empty text"})
                continue

            voice = msg.get("voice")
            speed = msg.get("speed")
            engine_override = msg.get("engine")
            direction_payload = msg.get("direction")
            is_phonemes = bool(msg.get("is_phonemes", False))
            phonetic_preprocess = msg.get("phonetic_preprocess")
            custom_lexicon = msg.get("custom_lexicon")
            total_duration = 0.0
            effective_direction: Direction | None = None

            # The direction and the engine override are validated the same way
            # whether or not the caller supplied a direction.  An explicit engine
            # is never silently dropped: it either routes the audio or errors.
            if direction_payload is not None:
                try:
                    requested = Direction.model_validate(direction_payload)
                except ValidationError as exc:
                    await websocket.send_json(
                        {"error": f"Invalid direction: {exc.errors(include_url=False)}"}
                    )
                    continue
                effective_direction, notes = resolve_direction(speed, requested)
            else:
                notes = []

            try:
                plan = tts_service.plan_direction(
                    voice=voice,
                    engine=engine_override,
                    direction=effective_direction,
                    streaming=True,
                )
            except ValueError as exc:
                await websocket.send_json({"error": str(exc)})
                continue

            if notes:
                plan = plan.model_copy(update={"warnings": [*notes, *plan.warnings]})
            if effective_direction is not None:
                speed = plan.applied.get("rate", speed)

            try:
                kwargs = {}
                if is_phonemes:
                    kwargs["is_phonemes"] = True
                if phonetic_preprocess is not None:
                    kwargs["phonetic_preprocess"] = phonetic_preprocess
                if custom_lexicon is not None:
                    kwargs["custom_lexicon"] = custom_lexicon

                # The engine that renders a stream can differ from the one the
                # pre-computed plan named: a Chatterbox stream whose weights are
                # absent falls back to Kokoro, which then also has to use its own
                # default voice.  ``report`` is filled in by the orchestrator so the
                # completion frame can say what actually happened instead of
                # describing the plan.
                report: dict = {}
                async for wav_chunk, duration in tts_service.synthesize_streaming_async(
                    text,
                    voice=voice,
                    speed=speed,
                    engine=engine_override,
                    direction=effective_direction,
                    report=report,
                    **kwargs,
                ):
                    await websocket.send_bytes(wav_chunk)
                    total_duration += duration

                actual_engine = report.get("engine") or plan.engine
                ignored = list(plan.ignored)
                if report.get("voice_substituted") and "voice" not in ignored:
                    ignored.append("voice")
                warnings = [*plan.warnings, *report.get("voice_notes", [])]
                if report.get("fallback_reason"):
                    warnings.append(
                        f"engine '{plan.engine}' was unavailable "
                        f"({report['fallback_reason']}); streamed with '{actual_engine}'"
                    )
                if report.get("engine") and plan.engine and actual_engine != plan.engine:
                    if "engine" not in ignored:
                        ignored.append("engine")

                await websocket.send_json(
                    {
                        "done": True,
                        "total_duration": round(total_duration, 3),
                        "engine": actual_engine,
                        "engine_planned": plan.engine,
                        "voice": report.get("voice", voice),
                        "direction_applied": plan.applied,
                        "direction_ignored": ignored,
                        "direction_warnings": warnings,
                    }
                )
            except ValueError as e:
                await websocket.send_json({"error": str(e)})
            except Exception:
                logger.exception("TTS streaming failed")
                await websocket.send_json({"error": "Synthesis failed"})

    except WebSocketDisconnect:
        logger.info("TTS WebSocket client disconnected")


# ─────────────────────────────────────────────────────────────
# Lexicon CRUD Endpoints
# ─────────────────────────────────────────────────────────────

@router.get("/lexicon", response_model=LexiconListResponse)
async def list_lexicon() -> LexiconListResponse:
    """List all active pronunciation dictionary entries (built-in + custom)."""
    entries = phonetic_converter.get_all_entries()
    return LexiconListResponse(entries=entries, total=len(entries))


@router.get("/lexicon/custom", response_model=LexiconListResponse)
async def list_custom_lexicon() -> LexiconListResponse:
    """List only custom, user-defined pronunciation dictionary entries."""
    entries = phonetic_converter.get_custom_entries()
    return LexiconListResponse(entries=entries, total=len(entries))


@router.get("/lexicon/{word}", response_model=LexiconEntryResponse)
async def get_lexicon_entry(word: str) -> LexiconEntryResponse:
    """Get pronunciation definition for a specific word."""
    entry = phonetic_converter.get_entry(word)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Pronunciation entry not found for '{word}'")
    return LexiconEntryResponse(
        word=word,
        replacement=entry.get("replacement"),
        ipa=entry.get("ipa"),
        languages=entry.get("languages"),
        source=entry.get("source", "custom"),
    )


@router.post("/lexicon", response_model=LexiconEntryResponse)
async def create_or_update_lexicon_entry(body: LexiconEntryRequest) -> LexiconEntryResponse:
    """Add or update a custom pronunciation dictionary entry.

    Provide either 'replacement' (phonetic respelling) or 'ipa' (IPA notation), or both.
    """
    if not body.replacement and not body.ipa:
        raise HTTPException(
            status_code=422,
            detail="At least one of 'replacement' (respelling) or 'ipa' must be provided",
        )

    res = phonetic_converter.set_entry(
        word=body.word,
        replacement=body.replacement,
        ipa=body.ipa,
        languages=body.languages,
    )
    return LexiconEntryResponse(**res)


@router.delete("/lexicon/{word}")
async def delete_lexicon_entry(word: str) -> dict:
    """Delete a custom pronunciation entry."""
    success = phonetic_converter.delete_entry(word)
    if not success:
        raise HTTPException(
            status_code=404, detail=f"Custom pronunciation entry not found for '{word}'"
        )
    return {"status": "deleted", "word": word}
