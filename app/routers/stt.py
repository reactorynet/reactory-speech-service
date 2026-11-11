"""STT router — REST and WebSocket endpoints for speech-to-text.

The REST endpoint and the capabilities endpoint are thin; the interesting part is
:class:`_StreamingSession`, which turns a chunked WebSocket upload into incremental
transcript segments while keeping the message contract existing clients already
understand.
"""

from __future__ import annotations

import json
import logging
from typing import Annotated, Any

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
from fastapi.concurrency import run_in_threadpool

from app.config import settings
from app.models.stt_schemas import STTCapabilitiesResponse, STTTranscriptionResult
from app.services.stt_service import ModelNotAvailableError
from app.services.stt_stream import StreamingAudioBuffer, normalise_encoding

logger = logging.getLogger(__name__)

router = APIRouter()

#: Hard cap for a single upload or a single streaming session.
MAX_AUDIO_BYTES = 25 * 1024 * 1024

#: Config keys accepted on the WebSocket before the first audio chunk.
_CONFIG_KEYS = frozenset(
    {
        "language",
        "encoding",
        "sample_rate",
        "channels",
        "model",
        "word_timestamps",
        "min_window_seconds",
        "partial_interval_seconds",
    }
)


# ----------------------------------------------------------------------
# REST
# ----------------------------------------------------------------------
@router.post(
    "/transcribe",
    response_model=STTTranscriptionResult,
    response_model_exclude_none=True,
    responses={
        400: {"description": "Requested model is not mounted on this deployment"},
        503: {"description": "STT service not available"},
    },
)
async def transcribe(
    request: Request,
    file: Annotated[UploadFile, File(description="Audio file (WAV, WebM, OGG, MP3)")],
    language: Annotated[str | None, Form(description="Language code (auto-detect if empty)")] = None,
    model: Annotated[
        str | None, Form(description="Model id to use (see GET /api/stt/capabilities)")
    ] = None,
    word_timestamps: Annotated[
        bool | None,
        Form(
            description=(
                "Return word-level timestamps per segment. Omit to use the "
                "deployment default (WHISPER_WORD_TIMESTAMPS); true/false wins."
            )
        ),
    ] = None,
    vad: Annotated[
        bool | None, Form(description="Override voice-activity detection for this request")
    ] = None,
) -> STTTranscriptionResult:
    """Transcribe an uploaded audio file to text.

    ``duration`` is the duration of the submitted file; ``speech_duration`` is the
    speech portion after VAD trimming. ``vad_failed`` is true when detection could
    not run, in which case the audio was transcribed untrimmed (never silently
    reported as an empty transcript).

    ``word_timestamps`` omitted means "use the deployment default".
    """
    stt_service = request.app.state.stt_service
    if not stt_service.is_ready:
        raise HTTPException(status_code=503, detail="STT service not available")

    audio_bytes = await file.read()
    if len(audio_bytes) == 0:
        raise HTTPException(status_code=422, detail="Empty audio file")
    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="Audio file exceeds 25MB limit")

    try:
        result = await run_in_threadpool(
            stt_service.transcribe,
            audio_bytes,
            language,
            model=model,
            word_timestamps=word_timestamps,
            vad=vad,
        )
    except ModelNotAvailableError as exc:
        logger.info("Rejected transcription for unavailable model '%s'", exc.model)
        raise HTTPException(
            status_code=400,
            detail={
                "error": "model_not_available",
                "message": str(exc),
                "model": exc.model,
                "available_models": exc.available,
            },
        ) from exc
    except Exception:
        logger.exception("STT transcription failed")
        raise HTTPException(status_code=500, detail="Transcription failed") from None

    return STTTranscriptionResult.model_validate(result)


@router.get("/capabilities", response_model=STTCapabilitiesResponse)
async def capabilities(request: Request) -> STTCapabilitiesResponse:
    """Report which models are mounted and which STT features are enabled."""
    stt_service = request.app.state.stt_service
    return STTCapabilitiesResponse.model_validate(stt_service.capabilities())


# ----------------------------------------------------------------------
# WebSocket streaming
# ----------------------------------------------------------------------
class _StreamingSession:
    """Incremental transcription for one WebSocket connection.

    The session keeps the received audio in memory, and once a minimum window has
    arrived it re-decodes the window and emits the segments that are *stable* — those
    that end at least ``whisper_stream_stability_margin_seconds`` before the end of
    the buffer. The unstable tail is withheld until the next window or until the
    client sends ``{"done": true}``.

    Re-decoding a growing buffer can move segment boundaries, so a segment that comes
    back with the same start but longer text (``"hello"`` -> ``"hello world"``)
    supersedes the earlier one instead of being appended again; see ``_commit``.
    """

    def __init__(self, service: Any) -> None:
        self._service = service
        self.language: str | None = None
        self.model: str | None = None
        self.word_timestamps: bool = bool(settings.whisper_word_timestamps)
        self._buffer = StreamingAudioBuffer()
        self._min_window = float(settings.whisper_stream_min_window_seconds)
        self._partial_interval = float(settings.whisper_stream_partial_interval_seconds)
        self._stability_margin = float(settings.whisper_stream_stability_margin_seconds)
        self._max_buffer_seconds = float(settings.whisper_stream_max_buffer_seconds)
        self._last_partial_duration = 0.0
        self._emitted_end = 0.0
        # Segments already sent, as {"start", "end", "text"}. Kept in full so a
        # re-decode that shifts a boundary can supersede the previous segment
        # instead of duplicating its words.
        self._emitted: list[dict] = []
        self._reported_language: str | None = None
        self._reported_confidence: float | None = None

    # -- configuration -------------------------------------------------
    def configure(self, data: dict) -> None:
        """Apply a config message. Unknown keys are ignored."""
        if "language" in data:
            self.language = data.get("language") or None
        if "model" in data:
            self.model = data.get("model") or None
        if "word_timestamps" in data:
            self.word_timestamps = bool(data["word_timestamps"])
        self._min_window = self._positive_float(
            data.get("min_window_seconds"), self._min_window
        )
        self._partial_interval = self._positive_float(
            data.get("partial_interval_seconds"), self._partial_interval
        )
        if {"encoding", "sample_rate", "channels"} & data.keys():
            self._reconfigure_buffer(data)

    def _reconfigure_buffer(self, data: dict) -> None:
        if self._buffer:
            # Audio already buffered: switching container mid-stream would split it.
            logger.warning("Ignoring audio-format change after audio was received")
            return
        try:
            encoding = normalise_encoding(data.get("encoding") or self._buffer.encoding)
            sample_rate = int(data.get("sample_rate") or self._buffer.sample_rate)
            channels = int(data.get("channels") or self._buffer.channels)
            self._buffer = StreamingAudioBuffer(
                encoding=encoding, sample_rate=sample_rate, channels=channels
            )
        except (TypeError, ValueError) as exc:
            logger.info("Rejected stream configuration: %s", exc)
            raise

    @staticmethod
    def _positive_float(value: Any, fallback: float) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return fallback
        return number if number > 0 else fallback

    def describe(self) -> dict:
        return {
            "status": "configured",
            "language": self.language,
            "model": self.model,
            "encoding": self._buffer.encoding,
            "sample_rate": self._buffer.sample_rate,
            "word_timestamps": self.word_timestamps,
            "min_window_seconds": self._min_window,
        }

    # -- audio ---------------------------------------------------------
    def append(self, chunk: bytes) -> None:
        self._buffer.append(chunk)

    def capacity_error(self) -> dict | None:
        """Return the specific limit that was exceeded, or ``None``.

        Reporting the actual limit (bytes vs seconds) matters: a client that hits
        the duration cap is not helped by being told about a byte cap it never
        approached.
        """
        buffered = len(self._buffer)
        if buffered > MAX_AUDIO_BYTES:
            return {
                "error": "Audio exceeds the maximum streaming size",
                "limit": "bytes",
                "limit_bytes": MAX_AUDIO_BYTES,
                "received_bytes": buffered,
            }
        duration = self._buffer.duration_seconds
        if self._max_buffer_seconds > 0 and duration > self._max_buffer_seconds:
            return {
                "error": "Audio exceeds the maximum streaming duration",
                "limit": "seconds",
                "limit_seconds": self._max_buffer_seconds,
                "duration_seconds": duration,
            }
        return None

    @property
    def over_capacity(self) -> bool:
        return self.capacity_error() is not None

    @property
    def has_audio(self) -> bool:
        return bool(self._buffer)

    def should_emit_partial(self) -> bool:
        duration = self._buffer.duration_seconds
        if duration <= 0 or duration < self._min_window:
            return False
        return (duration - self._last_partial_duration) >= self._partial_interval

    def reset(self) -> None:
        self._buffer.clear()
        self._last_partial_duration = 0.0
        self._emitted_end = 0.0
        self._emitted.clear()
        self._reported_language = None
        self._reported_confidence = None

    # -- transcription -------------------------------------------------
    async def emit_partial(self, emit) -> None:
        """Transcribe the current window and emit whichever segments are stable."""
        if self.should_emit_partial():
            await self._run_pass(emit, final=False)

    async def finish(self, emit) -> None:
        """Transcribe the remainder, then signal completion."""
        if not self._buffer:
            await emit({"error": "No audio received"})
            return
        completed = await self._run_pass(emit, final=True)
        if completed:
            await emit(
                {
                    "done": True,
                    "text": self.transcript(),
                    "language": self._reported_language,
                    "language_confidence": self._reported_confidence,
                }
            )
        self.reset()

    async def _run_pass(self, emit, *, final: bool) -> bool:
        self._last_partial_duration = self._buffer.duration_seconds
        try:
            result = await run_in_threadpool(
                self._service.transcribe,
                self._buffer.audio_bytes(),
                self.language,
                model=self.model,
                word_timestamps=self.word_timestamps,
            )
        except ModelNotAvailableError as exc:
            logger.info("Streaming request for unavailable model '%s'", exc.model)
            await emit(
                {
                    "error": "model_not_available",
                    "message": str(exc),
                    "model": exc.model,
                    "available_models": exc.available,
                }
            )
            return False
        except Exception:
            logger.exception("STT streaming transcription failed")
            await emit({"error": "Transcription failed"})
            return False

        await self._emit_new_segments(result, emit, final=final)
        return True

    # -- transcript bookkeeping -----------------------------------------
    def transcript(self) -> str:
        """The transcript built from every segment sent so far."""
        return " ".join(entry["text"] for entry in self._emitted if entry["text"])

    def _commit(self, start: float, end: float, text: str) -> str:
        """Reconcile a decoded segment against what has already been sent.

        A re-decode over a growing buffer can move a segment boundary, so the same
        words can come back inside a longer segment ("hello" -> "hello world"). The
        comparison is therefore on *text over the same region*, not on the end time
        alone:

        * ``"skip"``    — this exact text was already sent for this region;
        * ``"replace"`` — supersedes the last segment (which is dropped from the
          transcript so its words are not counted twice);
        * ``"append"``  — genuinely new speech.
        """
        if not text or not self._emitted:
            return "append"
        last = self._emitted[-1]
        same_region = start <= last["start"] + 1e-6
        if same_region:
            if text == last["text"]:
                return "skip"
            if text.startswith(last["text"]):
                return "replace"
            if last["text"].startswith(text):
                # A shorter re-decode of a region we already reported in full.
                return "skip"
        return "append"

    async def _emit_new_segments(self, result: dict, emit, *, final: bool) -> int:
        segments = result.get("segments") or []
        total = _as_float(result.get("duration"))
        if total is None or total <= 0:
            total = self._buffer.duration_seconds
        limit = total if final else total - self._stability_margin
        cutoff = self._emitted_end
        emitted = 0

        for segment in segments:
            end = _as_float(segment.get("end")) or 0.0
            start = _as_float(segment.get("start")) or 0.0
            if end <= cutoff + 1e-6:
                # Audio watermark already covers this span.
                continue
            if not final and end > limit:
                # Unstable tail: re-decoded next window with better context.
                continue
            text = str(segment.get("text") or "").strip()
            disposition = self._commit(start, end, text)
            if disposition == "skip":
                continue
            if disposition == "replace":
                self._emitted.pop()
            self._emitted.append({"start": round(start, 3), "end": round(end, 3), "text": text})
            payload = {
                "start": round(start, 3),
                "end": round(end, 3),
                "text": text,
                "partial_text": self.transcript(),
                "language": result.get("language") or self._reported_language,
                "is_partial": not final,
            }
            if segment.get("words") is not None:
                payload["words"] = segment["words"]
            await emit(payload)
            cutoff = max(cutoff, end)
            emitted += 1

        self._emitted_end = cutoff
        language = result.get("language")
        if isinstance(language, str):
            self._reported_language = language
        confidence = _as_float(result.get("language_confidence"))
        if confidence is not None:
            self._reported_confidence = confidence
        return emitted


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


async def _handle_text_message(
    session: _StreamingSession, websocket: WebSocket, raw: str
) -> None:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        await websocket.send_json({"error": "Invalid JSON"})
        return
    if not isinstance(data, dict):
        await websocket.send_json({"error": "Invalid JSON"})
        return

    if data.get("done"):
        await session.finish(websocket.send_json)
        return

    if _CONFIG_KEYS & data.keys():
        try:
            session.configure(data)
        except (TypeError, ValueError) as exc:
            await websocket.send_json({"error": "Invalid configuration", "detail": str(exc)})
            return
        await websocket.send_json(session.describe())
        return

    await websocket.send_json({"error": "Unsupported message"})


@router.websocket("/stream")
async def stt_stream(websocket: WebSocket) -> None:
    """WebSocket endpoint for streaming STT.

    Client sends:
      - JSON config, once at the start (all keys optional)::

            {"language": "en", "encoding": "pcm_s16le", "sample_rate": 16000,
             "model": "base", "word_timestamps": false, "min_window_seconds": 2.0}

        ``encoding`` is ``pcm_s16le`` for raw little-endian 16-bit PCM (recommended:
        any byte boundary is safe) or ``container`` (default) for WAV/WebM/OGG/MP3.
      - Binary audio chunks, as many as required.
      - ``{"done": true}`` to flush the tail and finish.

    Server responds with one JSON object per segment, using the original contract
    plus two additive fields::

        {"text": "...", "start": 0.0, "end": 1.5, "partial_text": "...",
         "language": "en", "is_partial": true, "words": [...]}

    Partials start flowing as soon as ``min_window_seconds`` of audio is buffered,
    i.e. before the client has finished sending. The final message is::

        {"done": true, "text": "...", "language": "en", "language_confidence": 0.98}

    Errors are reported as ``{"error": "..."}`` and do not close the socket unless
    stated.
    """
    await websocket.accept()
    stt_service = websocket.app.state.stt_service

    if not stt_service.is_ready:
        await websocket.send_json({"error": "STT service not available"})
        await websocket.close(code=1011)
        return

    session = _StreamingSession(stt_service)

    try:
        while True:
            message = await websocket.receive()
            msg_type = message.get("type", "")

            if msg_type == "websocket.disconnect":
                break

            if message.get("text") is not None:
                await _handle_text_message(session, websocket, message["text"])
                continue

            chunk = message.get("bytes")
            if chunk is None:
                continue

            session.append(chunk)
            capacity = session.capacity_error()
            if capacity is not None:
                await websocket.send_json(capacity)
                await websocket.close(code=1009)
                break
            await session.emit_partial(websocket.send_json)

    except WebSocketDisconnect:
        logger.info("STT WebSocket client disconnected")
    except RuntimeError:
        # Sending on a socket the client already closed raises RuntimeError.
        logger.info("STT WebSocket closed while streaming")
    except Exception:
        logger.exception("STT WebSocket session failed")
    finally:
        # Drops buffered audio; any in-flight temp file was already released by the
        # service's own context manager.
        session.reset()
