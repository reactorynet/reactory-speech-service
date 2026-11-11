"""Engine manager: registration, readiness and capability reporting.

Phase 1 keeps lifecycle deliberately simple: engines are registered at
construction and initialised once at startup.  Lazy loading, ``max_loaded_engines``
and idle unloading are SPEECH-004 (performance and lifecycle), not this ticket.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any

from app.core.contracts import EngineCapabilities, TTSEngine
from app.engines.base import UnknownEngineError
from app.services import voice_registry

logger = logging.getLogger(__name__)


class EngineManager:
    """Owns the registered engine adapters for this deployment."""

    def __init__(self, engines: Iterable[TTSEngine] | None = None) -> None:
        self._engines: dict[str, TTSEngine] = {}
        for engine in engines or ():
            self.register(engine)

    def register(self, engine: TTSEngine) -> TTSEngine:
        """Register an adapter; re-registering the same id replaces it."""
        engine_id = getattr(engine, "id", None)
        if not engine_id:
            raise ValueError("an engine adapter must expose a non-empty 'id'")
        self._engines[str(engine_id)] = engine
        return engine

    def get(self, engine_id: str | None) -> TTSEngine | None:
        """Return a registered engine, or ``None``."""
        if engine_id is None:
            return None
        return self._engines.get(engine_id)

    def require(self, engine_id: str) -> TTSEngine:
        """Return a registered engine or raise :class:`UnknownEngineError`."""
        engine = self.get(engine_id)
        if engine is None:
            raise UnknownEngineError(
                f"Unknown engine '{engine_id}'. Registered engines: "
                f"{', '.join(self.ids) or 'none'}. GET /api/tts/capabilities lists them."
            )
        return engine

    @property
    def ids(self) -> list[str]:
        """Registered engine ids, in registration order."""
        return list(self._engines)

    @property
    def engines(self) -> list[TTSEngine]:
        """Registered adapters, in registration order."""
        return list(self._engines.values())

    def is_ready(self, engine_id: str | None) -> bool:
        """Whether one engine is loaded and can render."""
        engine = self.get(engine_id)
        return bool(getattr(engine, "is_ready", False)) if engine is not None else False

    def any_ready(self) -> bool:
        """Whether at least one engine can render."""
        return any(self.is_ready(engine_id) for engine_id in self.ids)

    def readiness(self) -> dict[str, bool]:
        """Per-engine readiness, for ``/health`` and ``/ready``."""
        return {engine_id: self.is_ready(engine_id) for engine_id in self.ids}

    def capabilities(self) -> list[EngineCapabilities]:
        """The capability matrix for every registered engine."""
        return [engine.capabilities() for engine in self.engines]

    def descriptors(self) -> list[dict[str, Any]]:
        """The capability payload for every registered engine.

        Adapters may provide a richer ``descriptor()``; the fallback is built from
        the engine-agnostic contract so a new adapter cannot break the endpoint.
        """
        rows: list[dict[str, Any]] = []
        for engine in self.engines:
            descriptor = getattr(engine, "descriptor", None)
            if callable(descriptor):
                rows.append(descriptor())
            else:
                rows.append(self._generic_descriptor(engine))
        return rows

    @staticmethod
    def _generic_descriptor(engine: TTSEngine) -> dict[str, Any]:
        capabilities = engine.capabilities()
        return {
            "id": capabilities.engine,
            "engine": capabilities.engine,
            "name": getattr(engine, "name", capabilities.engine),
            "available": bool(getattr(engine, "is_ready", False)),
            "license": getattr(engine, "license", None),
            "languages": capabilities.languages,
            "voice_count": len(engine.voices()),
            "default_voice": voice_registry.default_voice_id(),
            "native_rate": capabilities.native_rate,
            "native_pitch": capabilities.native_pitch,
            "native_volume": capabilities.native_volume,
            "styles": capabilities.styles,
            "supports_cloning": capabilities.supports_cloning,
            "supports_streaming": capabilities.supports_streaming,
            "supports_ssml": capabilities.supports_ssml,
            "sample_rates": capabilities.sample_rates,
            "max_chars": capabilities.max_chars,
            "gpu_required": capabilities.gpu_required,
        }

    async def initialize(self) -> None:
        """Initialise every registered engine, isolating failures per engine."""
        for engine_id, engine in self._engines.items():
            initialise = getattr(engine, "initialize", None)
            if not callable(initialise):
                continue
            try:
                await initialise()
            except Exception:
                logger.exception("Failed to initialise TTS engine '%s'", engine_id)
