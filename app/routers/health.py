"""Health and readiness endpoints."""

from fastapi import APIRouter, Request

from app.models.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Health check — returns service status and model availability."""
    tts_service = request.app.state.tts_service
    stt_service = request.app.state.stt_service
    return HealthResponse(
        status="ok",
        tts=tts_service.is_ready,
        stt=stt_service.is_ready,
    )


@router.get("/ready")
async def ready(request: Request) -> dict:
    """Readiness probe — returns 200 only when all models are loaded."""
    tts_service = request.app.state.tts_service
    stt_service = request.app.state.stt_service
    if tts_service.is_ready and stt_service.is_ready:
        return {"ready": True}
    return {"ready": False}
