"""Engine router with an ordered fallback list.

Resolution order:

1. an explicit ``engine`` override (an unknown override is an error, never a
   silent fallback — the caller named an engine);
2. the engine that owns the requested voice, according to the voice registry;
3. an engine that natively advertises the requested ``direction.style`` — the
   expressive engine, when it is registered and enabled.  A pinned voice always
   wins over style: routing a Kokoro voice to Chatterbox would silently change
   its timbre, which is not what the caller asked for;
4. the router's ordered fallback list.

With one engine the routing decision is trivial, but the contract and its tests
exist so adding the second engine does not change the API layer.
"""

from __future__ import annotations

from collections.abc import Iterable

from app.core.contracts import TTSEngine
from app.engines.base import UnknownEngineError
from app.engines.manager import EngineManager
from app.services import voice_registry


class EngineRouter:
    """Chooses the engine (or ordered candidates) for a request."""

    def __init__(self, manager: EngineManager, order: Iterable[str] | None = None) -> None:
        self._manager = manager
        self._order = list(order) if order is not None else manager.ids

    @property
    def order(self) -> list[str]:
        """The ordered fallback list, as engine ids."""
        return list(self._order)

    def _style_engine_ids(self, style: str) -> list[str]:
        """Registered engines that advertise ``style`` natively, in order."""
        matching: list[str] = []
        for engine in self._manager.engines:
            try:
                styles = engine.capabilities().styles
            except Exception:  # pragma: no cover - a broken adapter must not break routing
                continue
            if style in styles:
                matching.append(engine.id)
        return matching

    def _candidate_ids(
        self,
        *,
        engine: str | None,
        voice: str | None,
        fallback: Iterable[str] | None,
        style: str | None = None,
    ) -> list[str]:
        ordered: list[str] = []

        def add(engine_id: str | None) -> None:
            if engine_id and engine_id not in ordered:
                ordered.append(engine_id)

        if engine:
            add(engine)
        elif voice:
            record = voice_registry.get_record(voice)
            if record is not None:
                add(record.engine)
        elif style and style != "neutral":
            # No engine and no voice pinned: let the expressive engine satisfy a
            # style it actually advertises.  Anything it cannot do stays on the
            # ordered fallback list, and a load failure still falls through.
            for engine_id in self._style_engine_ids(style):
                add(engine_id)

        for engine_id in fallback or ():
            add(engine_id)
        for engine_id in self._order:
            add(engine_id)
        return ordered

    def candidates(
        self,
        *,
        engine: str | None = None,
        voice: str | None = None,
        fallback: Iterable[str] | None = None,
        style: str | None = None,
    ) -> list[TTSEngine]:
        """Ordered engines to try, primary first.

        Raises :class:`UnknownEngineError` when the explicit override is not
        registered, and when nothing in the list resolves.
        """
        if engine is not None and self._manager.get(engine) is None:
            self._manager.require(engine)

        resolved = [
            self._manager.get(engine_id)
            for engine_id in self._candidate_ids(
                engine=engine, voice=voice, fallback=fallback, style=style
            )
        ]
        engines = [candidate for candidate in resolved if candidate is not None]
        if not engines:
            raise UnknownEngineError(
                "No TTS engine is registered. GET /api/tts/capabilities reports the "
                "engines available in this deployment."
            )
        return engines

    def resolve(
        self,
        *,
        engine: str | None = None,
        voice: str | None = None,
        fallback: Iterable[str] | None = None,
        style: str | None = None,
    ) -> TTSEngine:
        """The primary engine for a request."""
        return self.candidates(engine=engine, voice=voice, fallback=fallback, style=style)[0]
