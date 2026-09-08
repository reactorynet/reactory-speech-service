"""TTS router — REST and WebSocket endpoints for text-to-speech and lexicon management."""

import base64
import json
import logging

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

from app.models.schemas import (
    TTSRequest,
    TTSResponse,
    ErrorResponse,
    PhonemizeRequest,
    PhonemizeResponse,
    LexiconEntryRequest,
    LexiconEntryResponse,
    LexiconListResponse,
)
from app.services.phonetic_converter import phonetic_converter

logger = logging.getLogger(__name__)

router = APIRouter()


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
    if not tts_service.is_ready:
        raise HTTPException(status_code=503, detail="TTS service not available")

    try:
        kwargs = {}
        if body.is_phonemes:
            kwargs["is_phonemes"] = True
        if body.phonetic_preprocess is not None:
            kwargs["phonetic_preprocess"] = body.phonetic_preprocess
        if body.custom_lexicon is not None:
            kwargs["custom_lexicon"] = body.custom_lexicon

        wav_bytes, duration = tts_service.synthesize(
            body.text,
            voice=body.voice,
            speed=body.speed,
            **kwargs,
        )
        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={
                "X-Audio-Duration": str(round(duration, 3)),
                "X-Audio-Format": "wav",
                "X-Audio-Sample-Rate": "24000",
            },
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        logger.exception("TTS synthesis failed")
        raise HTTPException(status_code=500, detail="Synthesis failed")


@router.post("/synthesize/json", response_model=TTSResponse)
async def synthesize_json(request: Request, body: TTSRequest) -> TTSResponse:
    """Synthesize text or phonemes to speech, returning base64-encoded audio in JSON."""
    tts_service = request.app.state.tts_service
    if not tts_service.is_ready:
        raise HTTPException(status_code=503, detail="TTS service not available")

    try:
        kwargs = {}
        if body.is_phonemes:
            kwargs["is_phonemes"] = True
        if body.phonetic_preprocess is not None:
            kwargs["phonetic_preprocess"] = body.phonetic_preprocess
        if body.custom_lexicon is not None:
            kwargs["custom_lexicon"] = body.custom_lexicon

        wav_bytes, duration = tts_service.synthesize(
            body.text,
            voice=body.voice,
            speed=body.speed,
            **kwargs,
        )
        return TTSResponse(
            audio_base64=base64.b64encode(wav_bytes).decode("ascii"),
            duration=round(duration, 3),
            format="wav",
            sample_rate=24000,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception:
        logger.exception("TTS synthesis failed")
        raise HTTPException(status_code=500, detail="Synthesis failed")


@router.post(
    "/phonemize",
    response_model=PhonemizeResponse,
    responses={
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def phonemize_text(request: Request, body: PhonemizeRequest) -> PhonemizeResponse:
    """Convert text into phonetic (IPA) writing using normalization and phonetic conversion."""
    tts_service = request.app.state.tts_service
    try:
        result = tts_service.phonemize(
            body.text,
            language=body.language or "en-us",
            custom_lexicon=body.custom_lexicon,
        )
        return PhonemizeResponse(**result)
    except Exception as e:
        logger.exception("Phonemization failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/stream")
async def tts_stream(websocket: WebSocket) -> None:
    """WebSocket endpoint for streaming TTS."""
    await websocket.accept()
    tts_service = websocket.app.state.tts_service

    if not tts_service.is_ready:
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
            is_phonemes = bool(msg.get("is_phonemes", False))
            phonetic_preprocess = msg.get("phonetic_preprocess")
            custom_lexicon = msg.get("custom_lexicon")
            total_duration = 0.0

            try:
                kwargs = {}
                if is_phonemes:
                    kwargs["is_phonemes"] = True
                if phonetic_preprocess is not None:
                    kwargs["phonetic_preprocess"] = phonetic_preprocess
                if custom_lexicon is not None:
                    kwargs["custom_lexicon"] = custom_lexicon

                for wav_chunk, duration in tts_service.synthesize_streaming(
                    text,
                    voice=voice,
                    speed=speed,
                    **kwargs,
                ):
                    await websocket.send_bytes(wav_chunk)
                    total_duration += duration

                await websocket.send_json(
                    {"done": True, "total_duration": round(total_duration, 3)}
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
