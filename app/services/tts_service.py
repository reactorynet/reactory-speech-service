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

from app.config import settings
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
from app.services import voice_registry
from app.services.g2p_capability import g2p_capability
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
    engine_locale,
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
        expressive: bool | None = None,
    ) -> None:
        self._kokoro_engine = KokoroEngine()
        self._chatterbox_engine = None
        if manager is not None:
            self._manager = manager
        else:
            engines = [self._kokoro_engine]
            enabled = settings.chatterbox_enabled if expressive is None else expressive
            if enabled:
                # Import the adapter only when the flag is on: with it off the
                # module is never imported, so startup works on a deployment
                # where torch is not installed at all.
                from app.engines.chatterbox import ChatterboxEngine

                self._chatterbox_engine = ChatterboxEngine()
                engines.append(self._chatterbox_engine)
            self._manager = EngineManager(engines)
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
        streaming: bool = False,
    ) -> DirectionPlan:
        """Report what a direction would apply, without running inference.

        The adapter's own lowering is used when it provides one, so the plan and
        the synthesis result cannot drift (for example on rate clamping).

        ``streaming=True`` disables style-based routing: the expressive engine
        renders complete buffers and declares ``supports_streaming=False``, so a
        stream is served by the ordered fallback list instead of pretending the
        style will be honoured.
        """
        direction = direction or Direction()
        style = None if streaming else self._requested_style(direction)
        target = self._router.resolve(
            engine=engine, voice=voice, fallback=fallback, style=style
        )
        return self._plan_for_engine(target, direction)

    @staticmethod
    def _requested_style(direction: Direction | None) -> str | None:
        """The explicitly requested non-neutral style, or ``None``.

        Only a style the caller actually expressed participates in routing.  A
        default :class:`Direction` must not silently route away from the default
        engine.
        """
        if direction is None:
            return None
        if "style" not in direction.model_fields_set:
            return None
        if direction.style in (None, "neutral"):
            return None
        return direction.style

    @staticmethod
    def _plan_for_engine(target, direction: Direction) -> DirectionPlan:
        """Lower a direction for one resolved engine adapter."""
        lower = getattr(target, "lower", None)
        if callable(lower):
            return lower(direction)
        return lower_direction(direction, target.capabilities())

    @staticmethod
    def _default_voice_for(target) -> str | None:
        """A voice the target engine can actually render, or ``None``.

        Kokoro keeps the deployment default (``KOKORO_DEFAULT_VOICE``), so a
        fallback does not silently change the voice a caller is used to hearing.

        Every other engine is asked for *its own* first available voice.  That
        list is the adapter's, not the registry's: the Chatterbox adapter filters
        by the deployed ``CHATTERBOX_VARIANT``.  Reading the registry directly
        returned ``chatterbox_standard`` on a turbo deployment, which the adapter
        then refused as a variant mismatch — a 422 on the very fallback path that
        is meant to degrade gracefully.
        """
        if str(target.id) == KOKORO_ENGINE:
            return voice_registry.default_voice_id()
        try:
            ids = target.available_voice_ids()
        except Exception:  # pragma: no cover - defensive
            ids = []
        if ids:
            return ids[0]
        return voice_registry.default_voice_for_engine(str(target.id))

    def _prepare_request(
        self, target, request: EngineRequest
    ) -> tuple[EngineRequest, list[str], bool]:
        """Adapt a request to a candidate engine, reporting any voice substitution.

        When a request is routed to an engine other than the one that owns its
        voice — the expressive-engine fallback is the only such path today — the
        voice cannot be rendered as asked.  Substituting the engine's default and
        saying so is better than a 422 or a silent timbre change.
        """
        voice = request.voice
        if not voice:
            return request, [], False
        record = voice_registry.get_record(voice)
        if record is None:
            return request, [], False
        # Same engine is not the same as renderable: a Chatterbox deployment runs
        # one variant, and the other variant's voice is registered under the same
        # engine id.  Ask the adapter rather than comparing ids, so pinning the
        # wrong variant degrades to the deployed one instead of 422-ing where the
        # flag-off path would have substituted.
        can_render = getattr(target, "can_render_voice", None)
        if record.engine == str(target.id):
            if callable(can_render) and not can_render(voice):
                pass  # fall through to substitution
            else:
                return request, [], False
        substitute = self._default_voice_for(target)
        if substitute is None or substitute == voice:
            return request, [], False
        declared_variant = str(record.style_spec.get("variant") or "")
        deployed_variant = str(getattr(target, "variant", "") or "")
        if record.engine == str(target.id) and declared_variant and deployed_variant:
            why = (
                f"voice '{voice}' belongs to the '{declared_variant}' variant, but this "
                f"deployment runs '{deployed_variant}'"
            )
        elif record.engine == str(target.id):
            why = f"engine '{target.id}' cannot render voice '{voice}'"
        else:
            why = f"voice '{voice}' belongs to engine '{record.engine}'"
        note = (
            f"{why}; engine '{target.id}' rendered its default voice "
            f"'{substitute}' instead"
        )
        return request.model_copy(update={"voice": substitute}), [note], True

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
        fallback_notes: list[str] = []
        candidates = self._router.candidates(
            engine=engine, voice=voice, fallback=fallback, style=self._requested_style(merged)
        )
        for target in candidates:
            target_request, voice_notes, voice_substituted = self._prepare_request(
                target, request
            )
            plan = self._plan_for_engine(target, merged)
            if voice_substituted:
                plan = plan.model_copy(
                    update={"ignored": [*plan.ignored, "voice"], "warnings": [*voice_notes, *plan.warnings]}
                )
            key = self._cache_key_for(target, target_request)
            if use_cache and key is not None:
                entry = await self._cache.get(key)
                if entry is not None:
                    return SynthesisOutcome(
                        audio=entry.audio,
                        duration=entry.duration,
                        plan=plan.model_copy(
                            update={"warnings": [*fallback_notes, *plan.warnings]}
                        ),
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
                    result = await self._executor.run(target.synthesize, target_request)
            except EngineUnavailableError as exc:
                logger.warning("TTS engine '%s' is unavailable: %s", target.id, exc)
                unavailable = exc
                fallback_notes.append(
                    f"engine '{target.id}' was unavailable ({exc}); fell back to the next engine"
                )
                continue
            finally:
                self._manager.touch(target.id)

            applied_plan = DirectionPlan(
                engine=result.engine,
                applied=result.applied,
                ignored=(
                    [*result.ignored, "voice"] if voice_substituted else list(result.ignored)
                ),
                warnings=[*fallback_notes, *voice_notes, *result.warnings],
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
        report: dict | None = None,
    ) -> AsyncIterator[tuple[bytes, float]]:
        """Stream ``(wav_chunk_bytes, duration)`` per sentence without blocking.

        Each span is rendered in the inference pool under the engine's
        concurrency slot, so the event loop keeps serving other requests between
        sentences.  Streaming is deliberately not cached: a cache entry is a
        complete buffer, and chunk assembly is the caller's protocol.

        ``report``, when supplied, is filled in with the engine and voice that
        actually rendered (``engine``, ``voice``, ``voice_substituted``,
        ``voice_notes``, ``warnings``).  A stream can fall back to a different
        engine than the caller planned for, and switching voice silently is
        exactly what the HTTP path refuses to do; the transport needs the same
        facts to report in its completion frame.
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
        # Streaming never style-routes to the expressive engine: it renders whole
        # buffers and declares ``supports_streaming=False``.  Voice ownership and
        # the explicit override still decide, and an unavailable engine falls
        # through to the next candidate rather than failing the socket.
        target = None
        unavailable: EngineUnavailableError | None = None
        for candidate in self._router.candidates(
            engine=engine, voice=voice, fallback=fallback
        ):
            try:
                await self._manager.ensure_loaded(candidate.id)
            except EngineUnavailableError as exc:
                logger.warning("TTS engine '%s' is unavailable: %s", candidate.id, exc)
                unavailable = exc
                continue
            target = candidate
            break
        if target is None:
            if unavailable is not None:
                raise unavailable
            raise UnknownEngineError("No TTS engine could serve the stream")

        request, voice_notes, voice_substituted = self._prepare_request(target, request)
        if report is not None:
            report["engine"] = target.id
            report["voice"] = request.voice
            report["voice_substituted"] = voice_substituted
            report["voice_notes"] = list(voice_notes)
            if unavailable is not None:
                report["fallback_reason"] = str(unavailable)
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
        fallback_notes: list[str] = []
        candidates = self._router.candidates(
            engine=engine, voice=voice, fallback=fallback, style=self._requested_style(merged)
        )
        for target in candidates:
            target_request, voice_notes, voice_substituted = self._prepare_request(
                target, request
            )
            try:
                result = target.synthesize(target_request)
            except EngineUnavailableError as exc:
                logger.warning("TTS engine '%s' is unavailable: %s", target.id, exc)
                unavailable = exc
                fallback_notes.append(
                    f"engine '{target.id}' was unavailable ({exc}); fell back to the next engine"
                )
                continue
            if fallback_notes or voice_notes:
                result = result.model_copy(
                    update={
                        "ignored": (
                            [*result.ignored, "voice"] if voice_substituted else result.ignored
                        ),
                        "warnings": [*fallback_notes, *voice_notes, *result.warnings],
                    }
                )
            return result

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
        target = None
        unavailable: EngineUnavailableError | None = None
        for candidate in self._router.candidates(engine=engine, voice=voice):
            if not getattr(candidate, "is_ready", False):
                unavailable = unavailable or EngineUnavailableError(
                    f"TTS engine '{candidate.id}' is not loaded"
                )
                continue
            target = candidate
            break
        if target is None:
            if unavailable is not None:
                raise unavailable
            raise UnknownEngineError("No TTS engine could serve the stream")
        request, _voice_notes, _substituted = self._prepare_request(target, request)
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
        voice: str | None = None,
    ) -> dict:
        """Convert text to phonetic writing and report the normalised version.

        The engine locale comes from the **voice** when one is supplied, otherwise
        from the language tag resolved through the registry: ``zh`` becomes ``cmn``,
        the code the Chinese voices actually carry. A language the phonemizer cannot
        handle raises :class:`UnsupportedLanguageError`, which the router returns as
        a 422 rather than a phoneme string full of espeak's spelling fallback.
        """
        locale = engine_locale(language)
        if voice:
            _voice_key, locale = self._kokoro_engine.resolve_voice(voice)
        g2p_capability.ensure_supported(locale)
        normalized = phonetic_converter.normalize_text(
            text, custom_lexicon=custom_lexicon, language=locale
        )
        phonemes = phonetic_converter.convert_to_phonetic(
            text, language=locale, custom_lexicon=custom_lexicon
        )
        return {
            "text": text,
            "normalized": normalized,
            "phonemes": phonemes,
            "language": locale,
        }

    def g2p_capabilities(self) -> dict:
        """Measured text-front-end support, for ``/health``.

        Non-blocking on purpose: the probe loads one espeak-ng dictionary per
        language (about 1.5 s each), and a health check must not wait for that.
        Languages not yet probed are reported as ``pending`` rather than assumed
        supported.
        """
        return g2p_capability.snapshot()

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
