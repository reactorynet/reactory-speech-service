"""Thread-safety regression tests for grapheme-to-phoneme (G2P) conversion.

Background — why this file exists
---------------------------------
Phase 3 raised the default kokoro concurrency to 2.  That exposed a real defect:
espeak-ng keeps its configuration and synthesis state in process globals, and
``phonemizer`` caches exactly one ``EspeakBackend`` per language in a module-level
dict, so every worker thread shares one espeak instance.  Measured on this
repository before the fix:

======================  ==========================
workers                 wrong results out of 200
======================  ==========================
1                       0
2                       35-50
4                       50+
======================  ==========================

The failure modes were both silent: a swallowed decode error that returned the
*prose* where phonemes were expected, and cross-request contamination (one
request receiving another request's phonemes).  A corrupted render is then
stored in the new synthesis cache under the correct text key for the whole TTL.

The fix serialises every G2P call in the process behind
:data:`app.services.phonetic_converter.G2P_LOCK`, and Kokoro's own internal
phonemisation (``model.create(..., is_phonemes=False)``) behind the same lock.

These tests run without model files: the tokenizer and espeak are installed with
``kokoro-onnx``, so CI exercises the real G2P path.  If espeak genuinely is not
installed the real-phoneme test skips with a reason, but the structural tests
below (which use a tokenizer double) still run and still fail if the lock is
removed.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from app.core.contracts import EngineCapabilities, EngineRequest
from app.engines.kokoro import KokoroEngine
from app.services import phonetic_converter as pc_module
from app.services.phonetic_converter import G2P_LOCK, PhoneticConverter

SENTENCES = {
    "Hello world.": "həlˈoʊ wˈɜːld",
    "Numbers one two three four five.": "nˈʌmbɚz wˈʌn tˈuː θɹˈiː fˈɔːɹ fˈaɪv",
}

THREADS = 2
CALLS_PER_THREAD = 40


def _expected(text: str) -> str:
    """The reference phonemisation, measured single-threaded before the threads start."""
    return pc_module.phonetic_converter.convert_to_phonetic(text)


requires_g2p = pytest.mark.skipif(
    not pc_module.phonetic_converter.is_available,
    reason="espeak-ng / Kokoro tokenizer is not installed, so real G2P cannot be exercised",
)


# ─────────────────────────────────────────────────────────────
# The bug itself, against real espeak
# ─────────────────────────────────────────────────────────────

@requires_g2p
def test_concurrent_g2p_returns_each_requests_own_phonemes():
    """2 threads x N calls of two distinct sentences: every answer must be its own.

    Before the lock this failed with tens of wrong results at 2 workers.
    """
    baseline = {text: _expected(text) for text in SENTENCES}
    texts = list(SENTENCES)

    def worker(offset: int) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        for index in range(CALLS_PER_THREAD):
            text = texts[(offset + index) % len(texts)]
            out.append((text, pc_module.phonetic_converter.convert_to_phonetic(text)))
        return out

    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        batches = list(pool.map(worker, range(THREADS)))

    wrong = [
        (text, produced, baseline[text])
        for batch in batches
        for text, produced in batch
        if produced != baseline[text]
    ]

    assert wrong == [], (
        f"{len(wrong)} of {THREADS * CALLS_PER_THREAD} concurrent G2P calls returned "
        f"another request's phonemes or prose: {wrong[:3]}"
    )


@requires_g2p
def test_concurrent_g2p_never_returns_the_input_prose():
    """The old failure mode returned the normalised text, which the engine read as IPA."""
    text = "Numbers one two three four five."
    baseline = _expected(text)
    assert baseline != text, "the baseline should be phonemes, not prose"

    def worker(_: int) -> str:
        return pc_module.phonetic_converter.convert_to_phonetic(text)

    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        results = list(pool.map(worker, range(THREADS * CALLS_PER_THREAD)))

    assert set(results) == {baseline}


# ─────────────────────────────────────────────────────────────
# Structural: the lock really serialises the tokenizer call
# ─────────────────────────────────────────────────────────────

class _ConcurrencyDetectingTokenizer:
    """A tokenizer double that records the highest concurrent entry count."""

    def __init__(self, delay: float = 0.002) -> None:
        self.delay = delay
        self.active = 0
        self.peak = 0
        self.calls = 0
        self._guard = threading.Lock()

    def phonemize(self, text: str, lang: str = "en-us") -> str:
        with self._guard:
            self.active += 1
            self.calls += 1
            self.peak = max(self.peak, self.active)
        try:
            # A blocking sleep: if two threads can be inside at once, the peak
            # observed here is greater than one.
            threading.Event().wait(self.delay)
            return f"phonemes-for:{text}"
        finally:
            with self._guard:
                self.active -= 1


def _converter_with(tokenizer) -> PhoneticConverter:
    converter = PhoneticConverter.__new__(PhoneticConverter)
    converter._tokenizer = tokenizer
    converter._ready = True
    converter.builtin_lexicon = {}
    converter.builtin_language_ipa = {}
    converter.abbreviations = {}
    converter.units = {}
    converter._custom_lexicon = {}
    converter._last_mtime = None
    converter._lexicon_lock = threading.RLock()
    return converter


def test_converter_serialises_the_shared_tokenizer():
    """The tokenizer call must never overlap itself, whatever the thread count."""
    tokenizer = _ConcurrencyDetectingTokenizer()
    converter = _converter_with(tokenizer)

    def worker(index: int) -> str:
        return converter.convert_to_phonetic(f"Sentence number {index}.")

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(worker, range(24)))

    assert tokenizer.calls == 24
    assert tokenizer.peak == 1, (
        f"{tokenizer.peak} threads entered the shared espeak backend at once; "
        "G2P must be serialised behind G2P_LOCK"
    )
    assert len(set(results)) == 24


def test_the_converter_lock_is_the_documented_module_lock():
    assert pc_module.G2P_LOCK is G2P_LOCK
    # Reentrant: an engine may hold it while the converter acquires it again.
    with G2P_LOCK:
        with G2P_LOCK:
            pass


def test_a_g2p_failure_is_reported_not_swallowed_as_prose():
    """Silently returning the text made the engine read prose as phonemes."""

    class _BrokenTokenizer:
        def phonemize(self, text: str, lang: str = "en-us") -> str:
            raise ValueError("number of lines in input and output must be equal")

    converter = _converter_with(_BrokenTokenizer())

    with pytest.raises(pc_module.PhoneticConversionError):
        converter.convert_to_phonetic("Hello world.")


# ─────────────────────────────────────────────────────────────
# Kokoro's own phonemisation path shares the same lock
# ─────────────────────────────────────────────────────────────

class _RecordingTokenizer:
    """Records the peak number of threads inside ``phonemize`` at once.

    This is the step that must be serialised: it is the espeak-ng call.
    """

    def __init__(self, delay: float = 0.002) -> None:
        self.delay = delay
        self.active = 0
        self.peak = 0
        self.calls: list[tuple[str, str]] = []
        self._guard = threading.Lock()

    def phonemize(self, text: str, lang: str = "en-us") -> str:
        with self._guard:
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.calls.append((text, lang))
        try:
            threading.Event().wait(self.delay)
            return f"phonemes-for:{text}"
        finally:
            with self._guard:
                self.active -= 1


class _ConcurrencyDetectingKokoro:
    """Stand-in for ``kokoro_onnx.Kokoro`` that watches both of its steps.

    ``create`` is where the ONNX session runs.  ``tokenizer`` is the espeak-ng
    call.  These are the two halves ``create(is_phonemes=False)`` fuses, and the
    fix is precisely that only the second half may be serialised.
    """

    def __init__(self, *, delay: float = 0.002, with_tokenizer: bool = True) -> None:
        self.voices = {"af_heart": None}
        self.tokenizer = _RecordingTokenizer(delay) if with_tokenizer else None
        self.delay = delay
        self.active = 0
        self.render_peak = 0
        self.unsplit_calls = 0
        self.creates: list[dict] = []
        self._guard = threading.Lock()

    def create(self, text, voice, speed=1.0, lang="en-us", is_phonemes=False):
        with self._guard:
            self.active += 1
            self.render_peak = max(self.render_peak, self.active)
            self.creates.append({"text": text, "is_phonemes": is_phonemes})
            if not is_phonemes:
                self.unsplit_calls += 1
        try:
            threading.Event().wait(self.delay)
            return np.zeros(24, dtype=np.float32), 24000
        finally:
            with self._guard:
                self.active -= 1


def _engine_with(model) -> KokoroEngine:
    engine = KokoroEngine()
    engine.model = model
    engine.set_ready(True)
    return engine


def test_internal_phonemisation_is_still_serialised():
    """The engine's own espeak call must never overlap itself."""
    model = _ConcurrencyDetectingKokoro()
    engine = _engine_with(model)

    # Force the engine-phonemises branch: no converter pre-processing.
    def render(index: int) -> None:
        engine._render(
            f"Sentence number {index}.",
            voice="af_heart",
            rate=1.0,
            locale="en-us",
            is_phonemes=False,
            phonetic_preprocess=False,
            custom_lexicon=None,
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(render, range(16)))

    assert model.tokenizer.peak == 1, (
        f"{model.tokenizer.peak} threads phonemised inside kokoro concurrently; "
        "the engine path must hold G2P_LOCK around the tokenizer call"
    )


def test_internal_phonemisation_still_renders_the_same_phonemes():
    """The split must be byte-for-byte what ``create(is_phonemes=False)`` does.

    ``create`` phonemises, then splits and renders.  Phonemising first and calling
    ``create(..., is_phonemes=True)`` therefore produces the same audio.
    """
    model = _ConcurrencyDetectingKokoro(delay=0.0)
    engine = _engine_with(model)

    engine._render(
        "Numbers one two three.",
        voice="af_heart",
        rate=1.0,
        locale="en-us",
        is_phonemes=False,
        phonetic_preprocess=False,
        custom_lexicon=None,
    )

    assert model.tokenizer.calls == [("Numbers one two three.", "en-us")]
    assert len(model.creates) == 1
    assert model.creates[0]["is_phonemes"] is True
    assert model.creates[0]["text"] == "phonemes-for:Numbers one two three."
    assert model.unsplit_calls == 0, "create was called with is_phonemes=False"


def test_kokoro_does_not_hold_the_g2p_lock_while_rendering_when_preprocessing_is_disabled():
    """The R1 regression: disabling preprocessing must not serialise renders.

    ``create(is_phonemes=False)`` phonemises *and* runs the ONNX session inside one
    call.  Holding G2P_LOCK around it collapses engine concurrency to 1, so two
    renders must be able to sit inside ``create`` at the same time.
    """
    inside = threading.Semaphore(0)
    release = threading.Event()

    class _GatedKokoro:
        voices = {"af_heart": None}

        def __init__(self) -> None:
            self.tokenizer = _RecordingTokenizer(delay=0.0)
            self.phonemes_seen: list[str] = []
            self._guard = threading.Lock()

        def create(self, text, voice, speed=1.0, lang="en-us", is_phonemes=False):
            with self._guard:
                self.phonemes_seen.append(text)
            inside.release()
            release.wait(timeout=5)
            return np.zeros(24, dtype=np.float32), 24000

    model = _GatedKokoro()
    engine = _engine_with(model)
    errors: list[BaseException] = []

    def render(index: int) -> None:
        try:
            engine._render(
                f"Sentence number {index}.",
                voice="af_heart",
                rate=1.0,
                locale="en-us",
                is_phonemes=False,
                phonetic_preprocess=False,
                custom_lexicon=None,
            )
        except Exception as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=render, args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    try:
        assert inside.acquire(timeout=5), "first render never reached the model"
        assert inside.acquire(timeout=5), (
            "the G2P lock was held across ONNX inference when phonetic "
            "preprocessing is disabled, so renders serialised"
        )
    finally:
        release.set()
        for thread in threads:
            thread.join(timeout=5)

    assert errors == []
    assert model.phonemes_seen == ["phonemes-for:Sentence number 0.", "phonemes-for:Sentence number 1."]


def test_a_model_without_a_tokenizer_keeps_the_serialised_fallback():
    """A double with no tokenizer cannot be split; it stays correct, if slower."""
    model = _ConcurrencyDetectingKokoro(with_tokenizer=False)
    engine = _engine_with(model)

    def render(index: int) -> None:
        engine._render(
            f"Sentence number {index}.",
            voice="af_heart",
            rate=1.0,
            locale="en-us",
            is_phonemes=False,
            phonetic_preprocess=False,
            custom_lexicon=None,
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(render, range(8)))

    assert model.unsplit_calls == 8
    assert model.render_peak == 1, "the unsplittable fallback must hold the lock"


def test_kokoro_falls_back_to_engine_phonemisation_when_the_converter_fails(monkeypatch):
    """A converter failure must not hand prose to the engine as IPA."""
    model = _ConcurrencyDetectingKokoro()
    engine = _engine_with(model)

    class _BrokenConverter:
        is_available = True

        def convert_to_phonetic(self, text, language="en-us", custom_lexicon=None):
            raise pc_module.PhoneticConversionError("espeak fell over")

    monkeypatch.setattr("app.engines.kokoro.phonetic_converter", _BrokenConverter())

    engine._render(
        "Hello world.",
        voice="af_heart",
        rate=1.0,
        locale="en-us",
        is_phonemes=False,
        phonetic_preprocess=True,
        custom_lexicon=None,
    )

    # The engine phonemised the original text itself, rather than being handed
    # the prose as if it were IPA.
    assert model.tokenizer.calls == [("Hello world.", "en-us")]
    assert model.creates[-1]["is_phonemes"] is True
    assert model.creates[-1]["text"] == "phonemes-for:Hello world."


def test_kokoro_does_not_hold_the_g2p_lock_while_rendering_phonemes():
    """The ONNX inference must stay concurrent: only G2P is serialised."""
    started = threading.Event()
    release = threading.Event()
    inside = threading.Semaphore(0)

    class _BlockingKokoro:
        voices = {"af_heart": None}

        def create(self, text, voice, speed=1.0, lang="en-us", is_phonemes=True):
            inside.release()
            started.set()
            release.wait(timeout=5)
            return np.zeros(24, dtype=np.float32), 24000

    engine = _engine_with(_BlockingKokoro())
    request = EngineRequest(text="phoneme-input", voice="af_heart", is_phonemes=True)

    errors: list[BaseException] = []

    def render() -> None:
        try:
            engine.synthesize(request)
        except Exception as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=render) for _ in range(2)]
    for thread in threads:
        thread.start()
    try:
        # Both renders must reach ``create`` at the same time: the lock is not
        # held across the inference call.
        assert inside.acquire(timeout=5), "first render never reached the model"
        assert inside.acquire(timeout=5), "the G2P lock was held across inference"
    finally:
        release.set()
        for thread in threads:
            thread.join(timeout=5)

    assert errors == []
    assert started.is_set()


def test_engine_capabilities_are_unchanged_by_the_lock():
    """Guard against an accidental behavioural change while adding the lock."""
    caps = KokoroEngine().capabilities()

    assert caps.engine == "kokoro"
    assert isinstance(caps, EngineCapabilities)

# ─────────────────────────────────────────────────────────────
# End-to-end: the service's own concurrency path with real G2P
# ─────────────────────────────────────────────────────────────
#
# The unit tests above prove the lock exists.  This proves the *deployed* path —
# TTSService -> per-engine limiter (kokoro=2) -> inference pool -> engine
# synthesize -> real converter — cannot mix phonemes between requests.

@requires_g2p
async def test_concurrent_syntheses_through_the_service_never_mix_phonemes(tmp_path, monkeypatch):
    import asyncio
    import time

    from app.core.contracts import AudioChunk, AudioResult
    from app.engines.manager import EngineManager
    from app.services.inference import InferenceExecutor
    from app.services.synthesis_cache import SynthesisCache
    from app.services.tts_service import TTSService

    monkeypatch.setattr(
        "app.config.settings.speech_engine_max_concurrency", {"kokoro": 2, "default": 1}
    )

    class PhonemisingEngine:
        """Does what Kokoro does: real G2P then (here simulated) inference."""

        id = "kokoro"
        name = "Phonemising engine"
        license = "Apache-2.0"
        is_ready = True

        def capabilities(self):
            return EngineCapabilities(engine=self.id, sample_rates=[24000])

        def voices(self):
            return []

        def __init__(self):
            self.active = 0
            self.peak = 0
            self._guard = threading.Lock()

        def synthesize(self, req):
            with self._guard:
                self.active += 1
                self.peak = max(self.peak, self.active)
            try:
                phonemes = pc_module.phonetic_converter.convert_to_phonetic(req.text)
                time.sleep(0.01)  # stand-in for the ONNX render
                return AudioResult(
                    audio=phonemes.encode(),
                    duration=0.1,
                    sample_rate=24000,
                    engine=self.id,
                )
            finally:
                with self._guard:
                    self.active -= 1

        def stream(self, req):  # pragma: no cover - not used
            yield AudioChunk(audio=b"", duration=0.0, sample_rate=24000)

    engine = PhonemisingEngine()
    service = TTSService(
        manager=EngineManager([engine], max_loaded_engines=1),
        cache=SynthesisCache(enabled=False, directory=tmp_path / "cache"),
        executor=InferenceExecutor(max_workers=4),
    )

    texts = list(SENTENCES)
    baseline = {text: _expected(text) for text in texts}

    outcomes = await asyncio.gather(
        *(
            service.synthesize_async(texts[index % len(texts)], speed=1.0, phonetic_preprocess=False)
            for index in range(20)
        )
    )

    wrong = [
        (texts[index % len(texts)], outcome.audio.decode(), baseline[texts[index % len(texts)]])
        for index, outcome in enumerate(outcomes)
        if outcome.audio.decode() != baseline[texts[index % len(texts)]]
    ]

    assert wrong == [], f"{len(wrong)} of 20 renders used another request's phonemes: {wrong[:3]}"
    # ...and the test really did run renders concurrently, so it is not vacuous.
    assert engine.peak == 2, f"expected two overlapping renders, saw {engine.peak}"