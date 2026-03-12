"""STT router — REST and WebSocket endpoints for speech-to-text."""

import json
import logging

from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)

from app.models.schemas import TranscriptionResult, TranscriptionSegment, ErrorResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/transcribe",
    response_model=TranscriptionResult,
    responses={503: {"model": ErrorResponse}},
)
async def transcribe(
    request: Request,
    file: UploadFile = File(..., description="Audio file (WAV, WebM, OGG, MP3)"),
    language: str | None = Form(None, description="Language code (auto-detect if empty)"),
) -> TranscriptionResult:
    """Transcribe an uploaded audio file to text."""
    stt_service = request.app.state.stt_service
    if not stt_service.is_ready:
        raise HTTPException(status_code=503, detail="STT service not available")

    audio_bytes = await file.read()
    if len(audio_bytes) == 0:
        raise HTTPException(status_code=422, detail="Empty audio file")

    # Cap file size at 25MB
    if len(audio_bytes) > 25 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Audio file exceeds 25MB limit")

    try:
        result = stt_service.transcribe(audio_bytes, language=language)
        return TranscriptionResult(
            text=result["text"],
            language=result["language"],
            segments=[TranscriptionSegment(**s) for s in result["segments"]],
            duration=result["duration"],
        )
    except Exception:
        logger.exception("STT transcription failed")
        raise HTTPException(status_code=500, detail="Transcription failed")


@router.websocket("/stream")
async def stt_stream(websocket: WebSocket) -> None:
    """WebSocket endpoint for streaming STT.

    Client sends:
      - JSON config: {"language": "en"} (optional, sent once at start)
      - Binary audio chunks (accumulated until a JSON {"done": true} message)

    Server responds with JSON per segment:
      {"text": "...", "start": 0.0, "end": 1.5, "partial_text": "...", "language": "en"}

    When client sends {"done": true}, server processes accumulated audio and
    streams back segment-by-segment results.
    """
    await websocket.accept()
    stt_service = websocket.app.state.stt_service

    if not stt_service.is_ready:
        await websocket.send_json({"error": "STT service not available"})
        await websocket.close(code=1011)
        return

    language = None
    audio_buffer = bytearray()

    try:
        while True:
            # Use receive() to handle both text and binary messages
            message = await websocket.receive()
            msg_type = message.get("type", "")

            if msg_type == "websocket.disconnect":
                break

            if "text" in message:
                try:
                    data = json.loads(message["text"])
                except json.JSONDecodeError:
                    await websocket.send_json({"error": "Invalid JSON"})
                    continue

                # Config message
                if "language" in data and "done" not in data:
                    language = data["language"]
                    await websocket.send_json(
                        {"status": "configured", "language": language}
                    )
                    continue

                # Done signal — process accumulated audio
                if data.get("done"):
                    if not audio_buffer:
                        await websocket.send_json({"error": "No audio received"})
                        continue

                    try:
                        for segment in stt_service.transcribe_streaming(
                            bytes(audio_buffer), language=language
                        ):
                            await websocket.send_json(segment)

                        await websocket.send_json({"done": True})
                    except Exception:
                        logger.exception("STT streaming transcription failed")
                        await websocket.send_json(
                            {"error": "Transcription failed"}
                        )
                    finally:
                        audio_buffer.clear()
                    continue

            elif "bytes" in message:
                # Binary audio chunk
                audio_buffer.extend(message["bytes"])

    except WebSocketDisconnect:
        logger.info("STT WebSocket client disconnected")
