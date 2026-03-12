"""TTS router — REST and WebSocket endpoints for text-to-speech."""

import base64
import json
import logging

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

from app.models.schemas import TTSRequest, TTSResponse, ErrorResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/synthesize",
    responses={
        200: {"content": {"audio/wav": {}}},
        422: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
)
async def synthesize(request: Request, body: TTSRequest) -> Response:
    """Synthesize text to speech, returning WAV audio."""
    tts_service = request.app.state.tts_service
    if not tts_service.is_ready:
        raise HTTPException(status_code=503, detail="TTS service not available")

    try:
        wav_bytes, duration = tts_service.synthesize(
            body.text, voice=body.voice, speed=body.speed
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
    """Synthesize text to speech, returning base64-encoded audio in JSON."""
    tts_service = request.app.state.tts_service
    if not tts_service.is_ready:
        raise HTTPException(status_code=503, detail="TTS service not available")

    try:
        wav_bytes, duration = tts_service.synthesize(
            body.text, voice=body.voice, speed=body.speed
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


@router.websocket("/stream")
async def tts_stream(websocket: WebSocket) -> None:
    """WebSocket endpoint for streaming TTS.

    Client sends JSON: {"text": "...", "voice": "af_heart", "speed": 1.0}
    Server responds with binary WAV chunks (one per sentence), then a JSON
    completion message: {"done": true, "total_duration": ...}
    """
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
            total_duration = 0.0

            try:
                for wav_chunk, duration in tts_service.synthesize_streaming(
                    text, voice=voice, speed=speed
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
