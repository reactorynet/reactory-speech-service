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
    HTTPException,
    Query,
    Request,
    Response,
    WebSocket,
    WebSocketDisconnect,
)
from pydantic import ValidationError

from app.core.contracts import Direction, DirectionPlan
from app.core.direction import resolve_direction
from app.engines.base import EngineUnavailableError
from app.models.schemas import (
    BaseVoiceAliasRequest,
    CapabilitiesResponse,
    EngineCapabilities,
    ErrorResponse,
    LexiconEntryRequest,
    LexiconEntryResponse,
    LexiconListResponse,
    PhonemizeRequest,
    PhonemizeResponse,
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
            },
        )
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
        )
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


@router.get("/voices/{voice_id}", response_model=VoiceInfo)
async def get_voice(voice_id: str, request: Request) -> VoiceInfo:
    """Describe a single voice, or 404 when it is not present in the voice pack."""
    tts_service = request.app.state.tts_service
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


@router.get("/capabilities", response_model=CapabilitiesResponse)
async def capabilities(request: Request) -> CapabilitiesResponse:
    """Report the capability matrix of every TTS engine in this deployment."""
    tts_service = request.app.state.tts_service
    return CapabilitiesResponse(
        engines=[EngineCapabilities(**row) for row in tts_service.capability_matrix()]
    )


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
async def delete_voice(voice_id: str, request: Request) -> VoiceDeleteResponse:
    """Delete a designed voice. Pack voices and shipped presets are immutable (409)."""
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

                async for wav_chunk, duration in tts_service.synthesize_streaming_async(
                    text,
                    voice=voice,
                    speed=speed,
                    engine=engine_override,
                    direction=effective_direction,
                    **kwargs,
                ):
                    await websocket.send_bytes(wav_chunk)
                    total_duration += duration

                await websocket.send_json(
                    {
                        "done": True,
                        "total_duration": round(total_duration, 3),
                        "engine": plan.engine,
                        "direction_applied": plan.applied,
                        "direction_ignored": plan.ignored,
                        "direction_warnings": plan.warnings,
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
