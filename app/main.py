"""Reactory Speech Service — FastAPI application entry point."""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import settings
from app.routers import health, stt, tts
from app.services.g2p_capability import g2p_capability
from app.services.stt_service import STTService
from app.services.tts_service import TTSService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Shared service instances
tts_service = TTSService()
stt_service = STTService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: load models, probe G2P support, start the reaper. Shutdown: cleanup."""
    logger.info("Starting Reactory Speech Service on :%d", settings.speech_service_port)
    await tts_service.initialize()
    await stt_service.initialize()
    await tts_service.engine_manager.start_reaper()

    # SPEECH-010: measure which languages the text front end can actually
    # phonemise and log the matrix. It is a background thread because espeak-ng
    # loads one dictionary per language (about 1.5 s each); boot must not wait on
    # it, but the first capabilities/health caller can. Cached for the process.
    probe = asyncio.create_task(asyncio.to_thread(g2p_capability.log_report))
    app.state.g2p_probe = probe

    logger.info(
        "Services ready — TTS: %s, STT: %s",
        tts_service.is_ready,
        stt_service.is_ready,
    )
    yield
    logger.info("Shutting down Reactory Speech Service.")
    if not probe.done():
        probe.cancel()
    await tts_service.shutdown()


app = FastAPI(
    title="Reactory Speech Service",
    description="Local TTS (Kokoro) and STT (faster-whisper) microservice for the Reactory platform.",
    version="1.0.0",
    lifespan=lifespan,
)

# Make services available to routers via app.state
app.state.tts_service = tts_service
app.state.stt_service = stt_service

# Register routers
app.include_router(health.router, tags=["Health"])
app.include_router(tts.router, prefix="/api/tts", tags=["TTS"])
app.include_router(stt.router, prefix="/api/stt", tags=["STT"])
