"""Health and readiness endpoints.

Both endpoints keep the fields they have always returned (``status``, ``tts``,
``stt``, ``version``, ``ready``) and add per-engine readiness plus queue depth.
Old clients that only read the booleans keep working; an operator can now see
*which* engine is loaded and how much work is waiting behind it.

The engine rows are read from a snapshot method that never loads or unloads
anything, so probing health during a long synthesis is cheap and side-effect
free.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Request

from app.models.schemas import EngineHealth, HealthResponse

logger = logging.getLogger(__name__)

router = APIRouter()


def _engine_rows(tts_service: Any) -> dict[str, EngineHealth]:
    """Per-engine health, tolerating a service double without a manager."""
    manager = getattr(tts_service, "engine_manager", None)
    status = getattr(manager, "status", None) if manager is not None else None
    if not callable(status):
        return {}
    try:
        raw = status()
    except Exception:  # pragma: no cover - health must never fail the probe
        logger.warning("Could not read TTS engine status", exc_info=True)
        return {}
    if not isinstance(raw, dict):
        return {}

    rows: dict[str, EngineHealth] = {}
    for engine_id, value in raw.items():
        if not isinstance(value, dict):
            continue
        payload = {name: field for name, field in value.items() if name in EngineHealth.model_fields}
        payload["id"] = str(payload.get("id") or engine_id)
        try:
            rows[str(engine_id)] = EngineHealth(**payload)
        except Exception:  # pragma: no cover - a malformed row must not break health
            logger.warning("Ignoring malformed health row for engine '%s'", engine_id)
    return rows


def _cache_stats(tts_service: Any) -> dict[str, Any] | None:
    """Cache counters, or ``None`` when the service does not expose a cache."""
    cache = getattr(tts_service, "cache", None)
    stats = getattr(cache, "stats_dict", None)
    if not callable(stats):
        return None
    try:
        payload = stats()
    except Exception:  # pragma: no cover - health must never fail the probe
        logger.warning("Could not read synthesis cache stats", exc_info=True)
        return None
    return payload if isinstance(payload, dict) else None


def _queue_depth(engines: dict[str, EngineHealth]) -> int:
    return sum(row.queue_depth for row in engines.values())


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    """Health check — service status, engine availability and queue depth."""
    tts_service = request.app.state.tts_service
    stt_service = request.app.state.stt_service
    engines = _engine_rows(tts_service)
    return HealthResponse(
        status="ok",
        tts=bool(tts_service.is_ready),
        stt=bool(stt_service.is_ready),
        engines=engines,
        queue_depth=_queue_depth(engines),
        cache=_cache_stats(tts_service),
    )


@router.get("/ready")
async def ready(request: Request) -> dict:
    """Readiness probe — 200 only when every required model is loaded.

    The payload keeps ``ready`` and adds the per-engine detail so a caller can
    tell *why* the service is not ready.
    """
    tts_service = request.app.state.tts_service
    stt_service = request.app.state.stt_service
    engines = _engine_rows(tts_service)
    return {
        "ready": bool(tts_service.is_ready and stt_service.is_ready),
        "engines": {engine_id: row.model_dump() for engine_id, row in engines.items()},
        "queue_depth": _queue_depth(engines),
    }
