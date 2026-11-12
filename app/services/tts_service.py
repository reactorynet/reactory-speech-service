"""Synthesis orchestrator.

:class:`TTSService` is a thin orchestrator over the engine adapter layer:

* it resolves a request to an engine through :class:`EngineRouter`,
* it reports what direction the engine applied or ignored,
* it dispatches inference to a bounded thread pool under a per-engine
  concurrency limit, so a long render never blocks the event loop,
* it serves repeated identical requests from the two-tier synthesis cache,
* and it keeps the long-standing ``synthesize`` / ``synthesize_streaming``
  signatures working byte-for-byte for callers that send nothing new.

Kokoro-specific knowledge (model loading, voice resolution, phonemisation,
language mapping, speed clamping) lives in :mod:`app.engines.kokoro`.

Two entry points exist on purpose:

* :meth:`synthesize_async` / :meth:`synthesize_streaming_async` are the HTTP and
  WebSocket request path.  They await, so the event loop stays free.
* :meth:`synthesize` / :meth:`synthesize_streaming` are synchronous and kept for
  backwards compatibility (library callers and the Phase 1 tests).  They render
  inline and do not touch the cache.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Iterable, Iterator
from dataclasses import dataclass

import numpy as np

from app.core.contracts import (
    AudioResult,
    Direction,
    DirectionPlan,
    EngineRequest,
    Prosody,
)
from app.core.direction import resolve_direction
from app.engines.base import EngineUnavailableError, UnknownEngineError, lower_direction
from app.engines.kokoro import KokoroEngine
from app.engines.manager import EngineManager
from app.engines.router import EngineRouter
from app.services.inference import InferenceExecutor
from app.services.phonetic_converter import phonetic_converter
from app.services.synthesis_cache import CacheEntry, SynthesisCache, build_cache_key
from app.services.voice_design import default_direction
from app.services.voice_registry import (
    KOKORO_ENGINE,
    KOKORO_SAMPLE_RATE,
    KOKORO_SPEED_MAX,
    KOKORO_SPEED_MIN,
    VoiceDescriptor,
)

logger = logging.getLogger(__name__)

#: Native output sample rate of the loaded Kokoro model, in Hz.
SAMPLE_RATE = KOKORO_SAMPLE_RATE

#: Engine identifier reported by the voice catalogue and the capabilities endpoint.
ENGINE_ID = KOKORO_ENGINE

#: Speed multiplier range accepted by kokoro-onnx (it raises outside this range).
SPEED_MIN = KOKORO_SPEED_MIN
SPEED_MAX = KOKORO_SPEED_MAX

#: Output format of the audio this service returns.  Part of the cache key.
OUTPUT_FORMAT = "wav"


def _legacy_direction(speed: float | None) -> Direction:
    """Build a direction from the legacy ``speed`` alias.

    ``speed`` reaches this function from the WebSocket endpoint, which bypasses
    the schema, so the value is passed through *without* re-validation and the
    adapter clamps it defensively — exactly as before this change.
    """
    if speed is None:
        return Direction()
    return Direction(prosody=Prosody.model_construct(rate=float(speed)))


@dataclass(frozen=True)
class SynthesisOutcome:
    """A rendered request plus where it came from.

    ``cache`` is ``"hit"`` or ``"miss"`` and is what the ``X-Cache`` response
    header reports.  ``cache_tier`` is ``"lru"``, ``"disk"`` or ``"none"`` and
    explains *where* a hit came from.
    """

    audio: bytes
    duration: float
    plan: DirectionPlan
    cache: str = "miss"
    cache_tier: str = "none"
    cache_key: str | None = None
    engine: str = ""

    @property
    def cache_hit(self) -> bool:
        return self.cache == "hit"


class TTSService:
    """Engine-agnostic synthesis orchestrator with a Kokoro adapter registered."""

    def __init__(
        self,
        *,
        cache: SynthesisCache | None = None,
        executor: InferenceExecutor | None = None,
        manager: EngineManager | None = None,
    ) -> None:
        self._kokoro_engine = KokoroEngine()
        self._manager = manager or EngineManager([self._kokoro_engine])
        self._router = EngineRouter(
            self._manager, order=self._manager.ids or [KOKORO_ENGINE]
        )
        self._cache = cache or SynthesisCache()
        self._executor = executor or InferenceExecutor()

    # ------------------------------------------------------------------
    # Backwards-compatible internals
    # ------------------------------------------------------------------
    # The Phase 0 test-suite (and, historically, the service itself) reaches for
    # ``_kokoro`` and ``_ready``.  They now proxy the Kokoro adapter instead of
    # duplicating its state.

    @property
    def _kokoro(self):
        return self._kokoro_engine.model

    @_kokoro.setter
    def _kokoro(self, value) -> None:
        self._kokoro_engine.model = value

    @property
    def _ready(self) -> bool:
        return self._kokoro_engine.is_ready

    @_ready.setter
    def _ready(self, value: bool) -> None:
        self._kokoro_engine.set_ready(value)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Load the primary engine(s). Call once at startup.

        Engines beyond ``max_loaded_engines`` are deferred to first use; see
        :class:`app.engines.manager.EngineManager`.
        """
        await self._manager.initialize()

    async def shutdown(self) -> None:
        """Release the inference pool and stop the idle reaper."""
        await self._manager.stop_reaper()
        self._executor.shutdown(wait=False)

    @property
    def is_ready(self) -> bool:
        """Whether an engine is loaded and can render *now*.

        This is the truthful ``tts`` field of ``/health``.  Use :attr:`can_serve`
        to decide whether a request may be admitted: an engine that is registered
        but not yet loaded can still serve it through a lazy load.
        """
        return self._manager.any_ready()

    @property
    def can_serve(self) -> bool:
        """Whether this deployment has an engine that can (be loaded to) render."""
        return bool(self._manager.ids)

    @property
    def engine_manager(self) -> EngineManager:
        """The engine manager, for readiness and capability reporting."""
        return self._manager

    @property
    def engine_router(self) -> EngineRouter:
        """The router, exposed so callers can inspect the fallback order."""
        return self._router

    @property
    def cache(self) -> SynthesisCache:
        """The synthesis cache, exposed for health reporting and operator resets."""
        return self._cache

    @property
    def executor(self) -> InferenceExecutor:
        """The bounded inference pool."""
        return self._executor

    # ------------------------------------------------------------------
    # Voices
    # ------------------------------------------------------------------

    @property
    def engine_voice_ids(self) -> list[str] | None:
        """Voice ids exposed by the loaded engine, or ``None`` when none is loaded."""
        return self._kokoro_engine.engine_voice_ids

    @property
    def voice_ids(self) -> list[str]:
        """All voice ids this deployment exposes."""
        return self._kokoro_engine.available_voice_ids()

    def get_voices(self) -> list[dict]:
        """Return the enriched voice catalogue for this deployment."""
        catalogue: list[VoiceDescriptor] = []
        for engine in self._manager.engines:
            catalogue.extend(engine.voices())
        return list(catalogue)

    def get_voice(self, voice_id: str) -> VoiceDescriptor | None:
        """Return the descriptor for one voice, or ``None`` when it is unknown."""
        for engine in self._manager.engines:
            for descriptor in engine.voices():
                if descriptor["id"] == voice_id:
                    return descriptor
        return None

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------

    def capabilities(self) -> dict:
        """Describe the default engine (Phase 0 shape, kept for compatibility)."""
        return self._kokoro_engine.descriptor()

    def capability_matrix(self) -> list[dict]:
        """One capability payload per registered engine."""
        return self._manager.descriptors()

    # ------------------------------------------------------------------
    # Direction
    # ------------------------------------------------------------------

    def resolve_engine(
        self,
        *,
        voice: str | None = None,
        engine: str | None = None,
        fallback: Iterable[str] | None = None,
    ):
        """Resolve the primary engine for a request."""
        return self._router.resolve(engine=engine, voice=voice, fallback=fallback)

    def plan_direction(
        self,
        *,
        voice: str | None = None,
        engine: str | None = None,
        direction: Direction | None = None,
        fallback: Iterable[str] | None = None,
    ) -> DirectionPlan:
        """Report what a direction would apply, without running inference.

        The adapter's own lowering is used when it provides one, so the plan and
        the synthesis result cannot drift (for example on rate clamping).
        """
        target = self._router.resolve(engine=engine, voice=voice, fallback=fallback)
        return self._plan_for_engine(target, direction or Direction())

    @staticmethod
    def _plan_for_engine(target, direction: Direction) -> DirectionPlan:
        """Lower a direction for one resolved engine adapter."""
        lower = getattr(target, "lower", None)
        if callable(lower):
            return lower(direction)
        return lower_direction(direction, target.capabilities())

    @staticmethod
    def clamp_speed(speed: float | None) -> float:
        """Clamp the speed multiplier into the range kokoro-onnx accepts."""
        return KokoroEngine.clamp_speed(speed)

    # ------------------------------------------------------------------
    # Synthesis — asynchronous request path
    # ------------------------------------------------------------------

    async def synthesize_async(
        self,
        text: str,
        voice: str | None = None,
        speed: float | None = None,
        *,
        direction: Direction | None = None,
        engine: str | None = None,
        is_phonemes: bool = False,
        phonetic_preprocess: bool | None = None,
        custom_lexicon: dict[str, str] | None = None,
        fallback: Iterable[str] | None = None,
        cache: bool | None = None,
    ) -> SynthesisOutcome:
        """Render a request off the event loop and return the audio.

        Order of operations for each candidate engine:

        1. compute the direction plan (cheap, no inference),
        2. look up the synthesis cache — a hit returns without loading a model,
        3. lazily load the engine if it is not resident,
        4. take a per-engine concurrency slot and run inference in the pool,
        5. store the render in the cache.

        ``cache=False`` bypasses both tiers for this request; the global switch
        is ``SPEECH_CACHE_ENABLED``.
        """
        # A designed voice may ship a default direction (rate/pitch).  It applies
        # only when the caller expressed nothing, so an explicit speed or
        # direction always wins.
        if direction is None and speed is None:
            direction = default_direction(voice)
        merged, _ = resolve_direction(speed, direction)
        request = EngineRequest(
            text=text,
            voice=voice,
            direction=merged,
            is_phonemes=is_phonemes,
            phonetic_preprocess=phonetic_preprocess,
            custom_lexicon=custom_lexicon,
        )
        use_cache = self._cache.enabled and cache is not False

        unavailable: EngineUnavailableError | None = None
        candidates = self._router.candidates(engine=engine, voice=voice, fallback=fallback)
        for target in candidates:
            plan = self._plan_for_engine(target, merged)
            key = self._cache_key_for(target, request)
            if use_cache and key is not None:
                entry = await self._cache.get(key)
                if entry is not None:
                    return SynthesisOutcome(
                        audio=entry.audio,
                        duration=entry.duration,
                        plan=plan,
                        cache="hit",
                        cache_tier=entry.tier,
                        cache_key=key,
                        engine=target.id,
                    )

            # Take the engine's concurrency slot *before* the readiness check.
            # ``reap_idle()`` refuses to unload an engine with work in flight, so
            # holding the slot from this point makes the engine un-unloadable for
            # the duration of the request.  Loading first and slotting afterwards
            # left a window in which the idle reaper could unload the engine the
            # request had just been promised, turning a valid request into a 503.
            limiter = self._manager.limiter(target.id)
            try:
                async with limiter.slot():
                    await self._manager.ensure_loaded(target.id)
                    result = await self._executor.run(target.synthesize, request)
            except EngineUnavailableError as exc:
                logger.warning("TTS engine '%s' is unavailable: %s", target.id, exc)
                unavailable = exc
                continue
            finally:
                self._manager.touch(target.id)

            applied_plan = DirectionPlan(
                engine=result.engine,
                applied=result.applied,
                ignored=result.ignored,
                warnings=result.warnings,
            )
            if use_cache and key is not None:
                await self._cache.put(
                    key,
                    CacheEntry(
                        audio=result.audio,
                        duration=result.duration,
                        sample_rate=result.sample_rate,
                        engine=result.engine,
                    ),
                )
            return SynthesisOutcome(
                audio=result.audio,
                duration=result.duration,
                plan=applied_plan,
                cache="miss",
                cache_tier="none",
                cache_key=key,
                engine=result.engine,
            )

        if unavailable is not None:
            raise unavailable
        raise UnknownEngineError("No TTS engine could serve the request")

    async def synthesize_streaming_async(
        self,
        text: str,
        voice: str | None = None,
        speed: float | None = None,
        *,
        direction: Direction | None = None,
        engine: str | None = None,
        is_phonemes: bool = False,
        phonetic_preprocess: bool | None = None,
        custom_lexicon: dict[str, str] | None = None,
        fallback: Iterable[str] | None = None,
    ) -> AsyncIterator[tuple[bytes, float]]:
        """Stream ``(wav_chunk_bytes, duration)`` per sentence without blocking.

        Each span is rendered in the inference pool under the engine's
        concurrency slot, so the event loop keeps serving other requests between
        sentences.  Streaming is deliberately not cached: a cache entry is a
        complete buffer, and chunk assembly is the caller's protocol.
        """
        if direction is None and speed is None:
            direction = default_direction(voice)
        merged = direction if direction is not None else _legacy_direction(speed)
        request = EngineRequest(
            text=text,
            voice=voice,
            direction=merged,
            is_phonemes=is_phonemes,
            phonetic_preprocess=phonetic_preprocess,
            custom_lexicon=custom_lexicon,
        )
        target = self._router.resolve(engine=engine, voice=voice, fallback=fallback)
        await self._manager.ensure_loaded(target.id)
        limiter = self._manager.limiter(target.id)

        split = getattr(target, "split_sentences", None)
        span_renderer = getattr(target, "synthesize_span", None)
        if callable(split) and callable(span_renderer):
            spans = [span for span in split(text) if span.strip()]
            for span in spans:
                try:
                    async with limiter.slot():
                        chunk = await self._executor.run(span_renderer, request, span)
                finally:
                    self._manager.touch(target.id)
                yield chunk.audio, chunk.duration
            return

        # An adapter without span support renders the whole buffer at once; it is
        # still dispatched to the pool so the loop is never blocked.
        async with limiter.slot():
            try:
                result = await self._executor.run(target.synthesize, request)
            finally:
                self._manager.touch(target.id)
        yield result.audio, result.duration

    # ------------------------------------------------------------------
    # Cache identity
    # ------------------------------------------------------------------

    def _cache_key_for(self, target, request: EngineRequest) -> str | None:
        """The cache key for one engine, or ``None`` when it cannot be built.

        The key includes the *engine-resolved* voice and locale, so two callers
        that name the same voice differently share an entry, and a deployment
        whose default voice changed does not reuse the old audio.
        """
        voice_key = request.voice
        resolve = getattr(target, "resolve_voice", None)
        if callable(resolve):
            try:
                voice_key, _locale = resolve(request.voice)
            except ValueError:
                # An unknown voice is a 422, not a cache lookup; let inference
                # raise the caller-facing error.
                return None

        sample_rate = SAMPLE_RATE
        capabilities = getattr(target, "capabilities", None)
        if callable(capabilities):
            try:
                rates = capabilities().sample_rates
            except Exception:  # pragma: no cover - defensive
                rates = []
            if rates:
                sample_rate = int(rates[0])

        return build_cache_key(
            text=request.text,
            voice=voice_key,
            direction=request.direction,
            engine=str(target.id),
            output_format=OUTPUT_FORMAT,
            sample_rate=sample_rate,
            is_phonemes=request.is_phonemes,
            phonetic_preprocess=request.phonetic_preprocess,
            custom_lexicon=request.custom_lexicon,
        )

    # ------------------------------------------------------------------
    # Synthesis — synchronous compatibility path
    # ------------------------------------------------------------------

    def synthesize(
        self,
        text: str,
        voice: str | None = None,
        speed: float | None = None,
        is_phonemes: bool = False,
        phonetic_preprocess: bool | None = None,
        custom_lexicon: dict[str, str] | None = None,
    ) -> tuple[bytes, float]:
        """Synthesize text to WAV audio bytes.

        Returns ``(wav_bytes, duration_seconds)``.  This legacy signature is
        unchanged; the richer result is available from
        :meth:`synthesize_detailed`, and the non-blocking request path from
        :meth:`synthesize_async`.
        """
        result = self.synthesize_detailed(
            text,
            voice=voice,
            direction=_legacy_direction(speed),
            is_phonemes=is_phonemes,
            phonetic_preprocess=phonetic_preprocess,
            custom_lexicon=custom_lexicon,
        )
        return result.audio, result.duration

    def synthesize_detailed(
        self,
        text: str,
        *,
        voice: str | None = None,
        speed: float | None = None,
        direction: Direction | None = None,
        engine: str | None = None,
        is_phonemes: bool = False,
        phonetic_preprocess: bool | None = None,
        custom_lexicon: dict[str, str] | None = None,
        fallback: Iterable[str] | None = None,
    ) -> AudioResult:
        """Synthesize and return the audio plus the applied/ignored contract.

        ``direction`` wins over ``speed`` when both carry a rate; the router calls
        :func:`app.core.direction.resolve_direction` first, so that precedence is
        reported in the response.  Engines that cannot render (model not loaded)
        are skipped in favour of the next candidate in the ordered fallback list.

        This is the inline, synchronous API.  HTTP and WebSocket handlers use
        :meth:`synthesize_async` so inference never runs on the event loop.
        """
        if direction is None and speed is None:
            direction = default_direction(voice)
        merged, _ = resolve_direction(speed, direction)
        request = EngineRequest(
            text=text,
            voice=voice,
            direction=merged,
            is_phonemes=is_phonemes,
            phonetic_preprocess=phonetic_preprocess,
            custom_lexicon=custom_lexicon,
        )

        unavailable: EngineUnavailableError | None = None
        candidates = self._router.candidates(engine=engine, voice=voice, fallback=fallback)
        for target in candidates:
            try:
                return target.synthesize(request)
            except EngineUnavailableError as exc:
                logger.warning("TTS engine '%s' is unavailable: %s", target.id, exc)
                unavailable = exc

        if unavailable is not None:
            raise unavailable
        raise UnknownEngineError("No TTS engine could serve the request")

    def synthesize_streaming(
        self,
        text: str,
        voice: str | None = None,
        speed: float | None = None,
        is_phonemes: bool = False,
        phonetic_preprocess: bool | None = None,
        custom_lexicon: dict[str, str] | None = None,
        engine: str | None = None,
        direction: Direction | None = None,
    ) -> Iterator[tuple[bytes, float]]:
        """Generator that yields ``(wav_chunk_bytes, duration)`` per sentence.

        ``engine`` is an explicit override and is resolved through the same
        router as one-shot synthesis, so an unknown engine raises
        :class:`UnknownEngineError` and a known one really is the engine that
        renders the audio.  ``direction`` defaults to the legacy ``speed`` alias.

        This is the inline, synchronous API; the WebSocket endpoint uses
        :meth:`synthesize_streaming_async`.
        """
        if direction is None and speed is None:
            direction = default_direction(voice)
        request = EngineRequest(
            text=text,
            voice=voice,
            direction=direction if direction is not None else _legacy_direction(speed),
            is_phonemes=is_phonemes,
            phonetic_preprocess=phonetic_preprocess,
            custom_lexicon=custom_lexicon,
        )
        target = self._router.resolve(engine=engine, voice=voice)
        for chunk in target.stream(request):
            yield chunk.audio, chunk.duration

    # ------------------------------------------------------------------
    # Phonemisation (transport-facing helper, untouched by the adapter split)
    # ------------------------------------------------------------------

    def phonemize(
        self,
        text: str,
        language: str = "en-us",
        custom_lexicon: dict[str, str] | None = None,
    ) -> dict:
        """Convert text to phonetic writing and report normalized version."""
        normalized = phonetic_converter.normalize_text(text, custom_lexicon=custom_lexicon)
        phonemes = phonetic_converter.convert_to_phonetic(
            text, language=language, custom_lexicon=custom_lexicon
        )
        return {
            "text": text,
            "normalized": normalized,
            "phonemes": phonemes,
            "language": language,
        }

    @staticmethod
    def _samples_to_wav(samples: np.ndarray, sample_rate: int) -> bytes:
        """Convert float32 numpy samples to WAV bytes."""
        return KokoroEngine.samples_to_wav(samples, sample_rate)

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Simple sentence splitter for streaming TTS."""
        return KokoroEngine.split_sentences(text)


__all__: list[str] = [
    "ENGINE_ID",
    "OUTPUT_FORMAT",
    "SAMPLE_RATE",
    "SPEED_MAX",
    "SPEED_MIN",
    "SynthesisOutcome",
    "TTSService",
    "lower_direction",
]
