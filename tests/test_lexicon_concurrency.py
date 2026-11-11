"""Thread-safety tests for the persistent pronunciation lexicon.

Phase 3 moved every render into a bounded thread pool.  Renders call
``phonetic_converter.convert_to_phonetic`` → ``normalize_text``, which iterates the
custom lexicon, while ``POST``/``DELETE /api/tts/lexicon`` mutate that same
dictionary from the event loop.  Two threads over one dict raised

    RuntimeError: dictionary changed size during iteration

which escaped ``convert_to_phonetic`` (before its try block) and
``KokoroEngine._render`` (which catches only ``PhoneticConversionError``) and
became an HTTP 500 on an otherwise valid synthesis.

The lexicon is now copy-on-write with snapshot reads and atomic persistence.
These tests cover both halves: the readers never trip, and a mutation is never
observable half-applied or half-written.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import settings
from app.core.contracts import AudioResult, EngineCapabilities
from app.engines.manager import EngineManager
from app.routers import tts
from app.services.inference import InferenceExecutor
from app.services.phonetic_converter import PhoneticConverter
from app.services.synthesis_cache import SynthesisCache
from app.services.tts_service import TTSService

SAMPLE_RATE = 24000
LEXICON_ENTRIES = 300
WRITER_ITERATIONS = 300


@pytest.fixture
def converter(tmp_path, monkeypatch) -> PhoneticConverter:
    """A converter over a temporary lexicon file, seeded with real entries."""
    path = tmp_path / "lexicon.json"
    monkeypatch.setattr(settings, "lexicon_file_path", str(path))
    instance = PhoneticConverter()
    for index in range(LEXICON_ENTRIES):
        instance.set_entry(f"term{index}", replacement=f"repl{index}")
    return instance


# ─────────────────────────────────────────────────────────────
# The bug itself: readers vs writer, tight loop
# ─────────────────────────────────────────────────────────────

def test_renders_never_raise_while_the_lexicon_is_mutated(converter):
    """Reads through the real normalise/G2P path while entries churn.

    Before the fix this raised ``RuntimeError: dictionary changed size during
    iteration`` within a few hundred iterations.
    """
    errors: list[str] = []
    stop = threading.Event()

    def reader() -> None:
        while not stop.is_set():
            try:
                converter.normalize_text(
                    "Reactory Kubernetes API 5km and term1 term2.",
                )
                converter.convert_to_phonetic("Reactory uses term3 daily.")
            except BaseException as exc:  # noqa: BLE001 - the point of the test
                errors.append(f"{type(exc).__name__}: {exc}")
                return

    def writer() -> None:
        # Grow, then shrink.  CPython's dict iterator only complains when the
        # size *differs* at the check, so alternating add/delete (which restores
        # the size) would hide the race that a real lexicon edit exposes.
        added = []
        for index in range(LEXICON_ENTRIES, LEXICON_ENTRIES + WRITER_ITERATIONS):
            converter.set_entry(f"term{index}", replacement=f"repl{index}")
            added.append(index)
        for index in added:
            converter.delete_entry(f"term{index}")

    readers = [threading.Thread(target=reader, daemon=True) for _ in range(3)]
    for thread in readers:
        thread.start()
    writer_thread = threading.Thread(target=writer)
    writer_thread.start()
    writer_thread.join(timeout=120)
    stop.set()
    for thread in readers:
        thread.join(timeout=10)

    assert errors == []


def test_a_snapshot_is_isolated_from_later_mutations(converter):
    before = converter.lexicon_snapshot()
    assert before

    converter.set_entry("brand-new", replacement="brand new")
    converter.delete_entry("term0")

    # The snapshot taken earlier is untouched: it can be iterated safely even
    # though two mutations have happened since.
    assert "brand-new" not in before
    assert "term0" in before
    assert list(before.items())
    assert "brand-new" in converter.lexicon_snapshot()
    assert "term0" not in converter.lexicon_snapshot()


def test_a_snapshot_holds_consistent_values(converter):
    for snapshot in (converter.lexicon_snapshot() for _ in range(5)):
        for word, info in snapshot.items():
            assert isinstance(word, str)
            assert isinstance(info, dict)
            assert info.get("replacement") == word.replace("term", "repl")


def test_get_custom_entries_returns_a_copy(converter):
    entries = converter.get_custom_entries()
    entries["injected"] = {"replacement": "nope"}

    assert "injected" not in converter.lexicon_snapshot()


# ─────────────────────────────────────────────────────────────
# Copy-on-write publication
# ─────────────────────────────────────────────────────────────

def test_set_entry_rebinds_instead_of_mutating_in_place(converter):
    before = converter.lexicon_snapshot()

    converter.set_entry("another-one", replacement="another one")

    assert "another-one" not in before, "the live mapping was mutated in place"
    assert "another-one" in converter.lexicon_snapshot()


def test_delete_entry_rebinds_instead_of_deleting_in_place(converter):
    before = converter.lexicon_snapshot()
    assert converter.delete_entry("term1") is True

    assert "term1" in before, "the live mapping was mutated in place"
    assert "term1" not in converter.lexicon_snapshot()


def test_deleting_an_absent_entry_is_a_no_op(converter):
    assert converter.delete_entry("never-existed") is False


# ─────────────────────────────────────────────────────────────
# Atomic persistence
# ─────────────────────────────────────────────────────────────

def test_the_lexicon_file_is_always_parseable_while_being_written(converter, tmp_path):
    """A reader must never observe a half-written file.

    The previous ``open(..., "w") + json.dump`` left a window in which a
    concurrent ``load_lexicon()`` read an empty or partial file, failed to parse
    it, and silently kept the old entries.
    """
    path = Path(settings.lexicon_file_path)
    bad_reads: list[str] = []
    stop = threading.Event()

    def reader() -> None:
        while not stop.is_set():
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                bad_reads.append("missing")
            except json.JSONDecodeError as exc:
                bad_reads.append(f"unparseable: {exc}")
            except OSError as exc:  # pragma: no cover - platform dependent
                bad_reads.append(f"oserror: {exc}")

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    try:
        for index in range(200):
            converter.set_entry(f"atomic{index}", replacement=f"atomic {index}")
    finally:
        stop.set()
        thread.join(timeout=10)

    assert bad_reads == []


def test_a_write_leaves_no_temporary_files_behind(converter, tmp_path):
    for index in range(20):
        converter.set_entry(f"tmp{index}", replacement=f"tmp {index}")

    leftovers = [p.name for p in tmp_path.iterdir() if p.name.startswith(".lexicon-")]
    assert leftovers == []


def test_an_unwritable_lexicon_path_keeps_the_entry_in_memory(converter, monkeypatch):
    """Persistence stays best-effort, as before: the process still honours it."""
    monkeypatch.setattr(settings, "lexicon_file_path", "/proc/nonexistent/lexicon.json")

    result = converter.set_entry("in-memory-only", replacement="in memory")

    assert result["word"] == "in-memory-only"
    assert "in-memory-only" in converter.lexicon_snapshot()


def test_an_unparseable_file_keeps_the_published_entries(converter):
    Path(settings.lexicon_file_path).write_text("{ this is not json", encoding="utf-8")
    # Force a reload attempt: the mtime changed.
    os.utime(settings.lexicon_file_path, None)

    converter.load_lexicon()

    assert "term0" in converter.lexicon_snapshot(), (
        "a transient read failure dropped the operator's pronunciations"
    )


def test_a_deleted_file_clears_the_lexicon(tmp_path, monkeypatch):
    path = tmp_path / "lexicon.json"
    monkeypatch.setattr(settings, "lexicon_file_path", str(path))
    instance = PhoneticConverter()
    instance.set_entry("gone", replacement="gone")
    assert "gone" in instance.lexicon_snapshot()

    path.unlink()
    instance.load_lexicon()

    assert instance.lexicon_snapshot() == {}


# ─────────────────────────────────────────────────────────────
# End to end: renders through the pool while the API mutates the lexicon
# ─────────────────────────────────────────────────────────────

class _PhonemisingEngine:
    """A Kokoro stand-in that runs the real converter, like the real engine does."""

    id = "kokoro"
    name = "Phonemising engine"
    license = "Apache-2.0"
    is_ready = True

    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(engine=self.id, sample_rates=[SAMPLE_RATE])

    def voices(self):
        return []

    def synthesize(self, req):
        from app.services.phonetic_converter import phonetic_converter

        phonemes = phonetic_converter.convert_to_phonetic(req.text)
        return AudioResult(
            audio=phonemes.encode() or b"RIFF",
            duration=0.01,
            sample_rate=SAMPLE_RATE,
            engine=self.id,
        )

    def stream(self, req):  # pragma: no cover - not used here
        yield


def test_synthesis_returns_no_5xx_while_the_lexicon_is_mutated(tmp_path):
    """The reviewer's scenario: pooled renders against a live lexicon endpoint.

    Real router, real service, real inference pool, real converter.  Before the
    fix this produced an occasional HTTP 500 from
    ``RuntimeError: dictionary changed size during iteration``.
    """
    service = TTSService(
        manager=EngineManager([_PhonemisingEngine()], max_loaded_engines=1),
        cache=SynthesisCache(enabled=False, directory=tmp_path / "cache"),
        executor=InferenceExecutor(max_workers=4),
    )
    app = FastAPI()
    app.state.tts_service = service
    app.state.stt_service = MagicMock(is_ready=True)
    app.include_router(tts.router, prefix="/api/tts")
    client = TestClient(app)

    # Seed enough entries that the render-side iteration has a real window,
    # which is what makes the race reachable.
    for index in range(LEXICON_ENTRIES):
        client.post("/api/tts/lexicon", json={"word": f"term{index}", "replacement": f"repl{index}"})

    statuses: list[int] = []
    errors: list[str] = []
    stop = threading.Event()

    def renderer() -> None:
        while not stop.is_set():
            response = client.post(
                "/api/tts/synthesize",
                json={"text": "Reactory uses term1 and term2 daily.", "voice": "af_heart"},
            )
            statuses.append(response.status_code)
            if response.status_code >= 500:
                errors.append(response.text)
                return

    def mutator() -> None:
        # Grow first, then shrink: an add/delete pair restores the dict size and
        # the iterator would not notice, hiding the real race.
        added = []
        for index in range(LEXICON_ENTRIES, LEXICON_ENTRIES + WRITER_ITERATIONS):
            client.post(
                "/api/tts/lexicon",
                json={"word": f"term{index}", "replacement": f"repl{index}"},
            )
            added.append(index)
        for index in added:
            client.delete(f"/api/tts/lexicon/term{index}")

    threads = [threading.Thread(target=renderer, daemon=True) for _ in range(2)]
    for thread in threads:
        thread.start()
    writer = threading.Thread(target=mutator)
    writer.start()
    writer.join(timeout=180)
    stop.set()
    for thread in threads:
        thread.join(timeout=30)

    assert errors == [], f"5xx responses: {errors[:1]}"
    assert statuses, "no synthesis ran"
    assert set(statuses) == {200}, sorted(set(statuses))
    assert len(statuses) >= 20, f"only {len(statuses)} renders ran"


def test_the_lexicon_endpoint_reflects_a_mutation_immediately(tmp_path):
    path = tmp_path / "lexicon.json"
    settings_override = path
    service = TTSService(
        manager=EngineManager([_PhonemisingEngine()], max_loaded_engines=1),
        cache=SynthesisCache(enabled=False, directory=tmp_path / "cache"),
        executor=InferenceExecutor(max_workers=2),
    )
    app = FastAPI()
    app.state.tts_service = service
    app.state.stt_service = MagicMock(is_ready=True)
    app.include_router(tts.router, prefix="/api/tts")
    client = TestClient(app)

    created = client.post(
        "/api/tts/lexicon", json={"word": "Zebra", "replacement": "zee-bra"}
    )
    assert created.status_code == 200

    listing = client.get("/api/tts/lexicon/custom").json()
    assert "Zebra" in listing["entries"]

    deleted = client.delete("/api/tts/lexicon/Zebra")
    assert deleted.status_code == 200
    assert "Zebra" not in client.get("/api/tts/lexicon/custom").json()["entries"]
    assert settings_override.name == "lexicon.json"


def test_a_mutation_during_iteration_does_not_break_a_render(converter, monkeypatch):
    """Deterministic interleaving of the exact race.

    A reader is paused *inside* the custom-lexicon loop of ``normalize_text``
    while the writer adds and removes an entry.  Because readers iterate a
    snapshot, the paused loop is unaffected; iterating the live mapping (the
    pre-fix behaviour) raises ``RuntimeError: dictionary changed size during
    iteration`` at the next ``__next__``.

    The statistical test above depends on timing; this one does not.
    """
    import app.services.phonetic_converter as pc_module

    # Strip the built-in passes so the only re.sub calls are the inline-bracket
    # pass (1) and then one per custom entry: call #2 is inside the lexicon loop.
    converter.units = {}
    converter.abbreviations = {}
    converter.builtin_lexicon = {}

    inside_loop = threading.Event()
    mutation_done = threading.Event()

    class _SyncRe:
        def __init__(self, real):
            self._real = real
            self.calls = 0

        def sub(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 2:
                inside_loop.set()
                mutation_done.wait(timeout=10)
            return self._real.sub(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._real, name)

    monkeypatch.setattr(pc_module, "re", _SyncRe(pc_module.re))

    errors: list[str] = []

    def reader() -> None:
        try:
            converter.normalize_text("Reactory uses term5 and term6.")
        except BaseException as exc:  # noqa: BLE001 - the point of the test
            errors.append(f"{type(exc).__name__}: {exc}")

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    try:
        assert inside_loop.wait(timeout=10), "the reader never reached the lexicon loop"
        # Mutate while the reader is mid-iteration.  The mapping *grows*, which
        # is what CPython's dict iterator notices: an add-then-remove that
        # leaves the size unchanged goes undetected, so a single add is the
        # sharpest form of this race.
        converter.set_entry("mid-iteration", replacement="mid iteration")
    finally:
        mutation_done.set()
        thread.join(timeout=10)

    assert errors == []
