"""Tests for the two-tier synthesis cache.

The acceptance criteria this file covers:

* a repeated identical request is served from cache and returns identical bytes;
* the key changes with voice, direction, engine, output format and sample rate;
* an entry expires per TTL;
* the cache can be disabled globally and per request;
* the disk tier survives a restart (a new cache instance over the same directory);
* a corrupt or truncated entry is ignored rather than served.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from app.config import settings
from app.core.contracts import Direction, Prosody
from app.services.synthesis_cache import (
    CacheEntry,
    KEY_COMPONENTS,
    SynthesisCache,
    build_cache_key,
    renderer_fingerprint,
)


def _key(**overrides) -> str:
    params = {
        "text": "Hello from the Reactory speech service.",
        "voice": "af_heart",
        "direction": Direction(),
        "engine": "kokoro",
        "output_format": "wav",
        "sample_rate": 24000,
    }
    params.update(overrides)
    return build_cache_key(**params)


def _entry(audio: bytes = b"RIFF-identical-audio", duration: float = 1.25) -> CacheEntry:
    return CacheEntry(audio=audio, duration=duration, sample_rate=24000, engine="kokoro")


def _cache(tmp_path, **kwargs) -> SynthesisCache:
    options = {
        "enabled": True,
        "directory": tmp_path / "cache",
        "ttl_seconds": 3600.0,
    }
    options.update(kwargs)
    return SynthesisCache(**options)


# ─────────────────────────────────────────────────────────────
# Key identity
# ─────────────────────────────────────────────────────────────

def test_key_is_stable_and_is_a_sha256_digest():
    first = _key()
    second = _key()

    assert first == second
    assert len(first) == 64
    assert all(character in "0123456789abcdef" for character in first)


def test_key_documents_its_six_components():
    assert KEY_COMPONENTS == (
        "text",
        "voice",
        "direction",
        "engine",
        "output_format",
        "sample_rate",
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("text", "A different sentence."),
        ("voice", "am_adam"),
        ("direction", Direction(prosody=Prosody(rate=1.2))),
        ("engine", "chatterbox"),
        ("output_format", "mp3"),
        ("sample_rate", 48000),
    ],
)
def test_every_keyed_field_changes_the_key(field, value):
    assert _key(**{field: value}) != _key()


def test_direction_values_are_hashed_canonically():
    """The same direction expressed twice hashes to one key."""
    first = Direction(style="calm", prosody=Prosody(rate=0.9))
    second = Direction.model_validate({"prosody": {"rate": 0.9}, "style": "calm"})

    assert _key(direction=first) == _key(direction=second)


def test_legacy_switches_that_change_audio_are_part_of_the_key():
    plain = _key()
    phonemes = build_cache_key(
        text="həˈloʊ",
        voice="af_heart",
        direction=Direction(),
        engine="kokoro",
        output_format="wav",
        sample_rate=24000,
        is_phonemes=True,
    )
    lexicon = build_cache_key(
        text="Hello from the Reactory speech service.",
        voice="af_heart",
        direction=Direction(),
        engine="kokoro",
        output_format="wav",
        sample_rate=24000,
        custom_lexicon={"Reactory": "ree-ak-tor-ee"},
    )

    assert phonemes != plain
    assert lexicon != plain


# ─────────────────────────────────────────────────────────────
# Tiers
# ─────────────────────────────────────────────────────────────

async def test_identical_render_is_served_from_the_lru_byte_for_byte(tmp_path):
    cache = _cache(tmp_path)
    key = _key()
    entry = _entry()

    await cache.put(key, entry)
    hit = await cache.get(key)

    assert hit is not None
    assert hit.tier == "lru"
    assert hit.audio == entry.audio
    assert hit.duration == entry.duration
    assert cache.stats.hits_lru == 1
    assert cache.stats.hits == 1


async def test_a_new_instance_reads_the_disk_tier(tmp_path):
    """Simulates a restart: the process LRU is empty, the files are not."""
    key = _key()
    first = _cache(tmp_path)
    await first.put(key, _entry())

    second = _cache(tmp_path)
    hit = await second.get(key)

    assert hit is not None
    assert hit.tier == "disk"
    assert hit.audio == _entry().audio
    assert second.stats.hits_disk == 1


async def test_an_entry_expires_per_ttl(tmp_path):
    now = [1000.0]
    key = _key()
    cache = _cache(tmp_path, ttl_seconds=60.0, clock=lambda: now[0])

    await cache.put(key, _entry())
    assert await cache.get(key) is not None

    # A fresh instance, so the LRU cannot mask the expired disk entry.
    reader = _cache(tmp_path, ttl_seconds=60.0, clock=lambda: now[0])
    now[0] += 61.0

    assert await reader.get(key) is None
    assert not list(cache.directory.glob("*/*.audio"))


async def test_a_missing_entry_is_a_miss(tmp_path):
    cache = _cache(tmp_path)

    assert await cache.get(_key()) is None
    assert cache.stats.misses == 1


async def test_oversized_entries_skip_the_lru_but_stay_on_disk(tmp_path):
    cache = _cache(tmp_path, lru_max_entry_bytes=8)
    key = _key()
    big = CacheEntry(audio=b"x" * 64, duration=1.0, sample_rate=24000, engine="kokoro")

    await cache.put(key, big)
    hit = await cache.get(key)

    assert hit is not None
    assert hit.tier == "disk"
    assert hit.audio == big.audio


async def test_a_truncated_disk_entry_is_ignored(tmp_path):
    cache = _cache(tmp_path)
    key = _key()
    await cache.put(key, _entry())
    audio_path, meta_path = cache._paths(key)
    audio_path.write_bytes(b"")

    reader = _cache(tmp_path)
    assert await reader.get(key) is None
    assert reader.stats.errors == 1
    assert not audio_path.exists()
    assert not meta_path.exists()


async def test_clear_drops_both_tiers(tmp_path):
    cache = _cache(tmp_path)
    key = _key()
    await cache.put(key, _entry())

    cache.clear()

    assert cache.stats.lru_entries == 0
    assert await cache.get(key) is None


# ─────────────────────────────────────────────────────────────
# Disable switches
# ─────────────────────────────────────────────────────────────

async def test_global_disable_prevents_reads_and_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "speech_cache_enabled", False)
    cache = _cache(tmp_path, enabled=None)
    key = _key()

    await cache.put(key, _entry())

    assert cache.enabled is False
    assert await cache.get(key) is None
    assert not cache.directory.exists() or not list(cache.directory.glob("*/*.audio"))
    assert cache.stats.writes == 0


async def test_explicit_disable_overrides_the_setting(tmp_path):
    cache = _cache(tmp_path, enabled=False)

    await cache.put(_key(), _entry())

    assert await cache.get(_key()) is None


# ─────────────────────────────────────────────────────────────
# Concurrency and hygiene
# ─────────────────────────────────────────────────────────────

async def test_concurrent_puts_and_gets_do_not_corrupt_entries(tmp_path):
    cache = _cache(tmp_path)
    keys = [_key(text=f"Sentence number {index}.") for index in range(24)]

    await asyncio.gather(*(cache.put(key, _entry(audio=key.encode())) for key in keys))
    hits = await asyncio.gather(*(cache.get(key) for key in keys))

    assert [hit.audio for hit in hits if hit is not None] == [key.encode() for key in keys]


async def test_stats_expose_the_running_configuration(tmp_path):
    cache = _cache(tmp_path, ttl_seconds=42.0)
    stats = cache.stats_dict()

    assert stats["enabled"] is True
    assert stats["ttl_seconds"] == 42.0
    assert stats["hits"] == 0
    assert stats["misses"] == 0
    assert stats["directory"] == str(tmp_path / "cache")
    assert json.dumps(stats)  # JSON-serialisable for /health

# ─────────────────────────────────────────────────────────────
# TTL applies to the in-process tier too
# ─────────────────────────────────────────────────────────────
#
# A hot phrase is never evicted by pressure, so a TTL check that only ran on the
# disk path would never apply to exactly the entries the LRU exists to serve: a
# shortened SPEECH_CACHE_TTL_SECONDS would leave them live forever.  These tests
# use a single instance — no fresh reader — so the LRU is the tier under test.

async def test_a_hot_lru_entry_expires_per_ttl_on_the_same_instance(tmp_path):
    now = [1000.0]
    key = _key()
    cache = _cache(tmp_path, ttl_seconds=60.0, clock=lambda: now[0])

    await cache.put(key, _entry())
    hit = await cache.get(key)
    assert hit is not None
    assert hit.tier == "lru"

    # Still inside the TTL: the LRU keeps serving it.
    now[0] += 59.0
    assert await cache.get(key) is not None

    # Past the TTL, same object, no restart.
    now[0] += 2.0
    assert await cache.get(key) is None, "the in-process tier must honour the TTL"


async def test_an_expired_lru_entry_is_dropped_and_counted(tmp_path):
    now = [1000.0]
    key = _key()
    cache = _cache(tmp_path, ttl_seconds=60.0, clock=lambda: now[0])
    await cache.put(key, _entry())

    now[0] += 61.0
    assert await cache.get(key) is None

    assert cache.stats.lru_entries == 0
    assert cache.stats.lru_bytes == 0
    assert cache.stats.evictions >= 1
    assert cache.stats.hits == 0


async def test_the_synchronous_lookup_honours_the_ttl_too(tmp_path):
    now = [1000.0]
    key = _key()
    cache = _cache(tmp_path, ttl_seconds=60.0, clock=lambda: now[0])
    cache.put_sync(key, _entry())

    assert cache.get_sync(key) is not None

    now[0] += 61.0
    assert cache.get_sync(key) is None


async def test_a_zero_ttl_means_no_expiry_on_either_tier(tmp_path):
    now = [1000.0]
    key = _key()
    cache = _cache(tmp_path, ttl_seconds=0.0, clock=lambda: now[0])
    await cache.put(key, _entry())

    now[0] += 10_000_000.0

    hit = await cache.get(key)
    assert hit is not None
    assert hit.tier == "lru"


async def test_shortening_the_ttl_takes_effect_for_an_already_hot_entry(tmp_path):
    """An operator shortening the TTL must not wait for the LRU to evict."""
    now = [1000.0]
    key = _key()
    cache = _cache(tmp_path, ttl_seconds=None, clock=lambda: now[0])
    cache._ttl_seconds = 3600.0  # long TTL at first
    await cache.put(key, _entry())
    assert (await cache.get(key)) is not None

    cache._ttl_seconds = 60.0  # operator shortens it
    now[0] += 120.0

    assert await cache.get(key) is None


# ─────────────────────────────────────────────────────────────
# Entry timestamps
# ─────────────────────────────────────────────────────────────

async def test_an_entry_carries_the_time_it_was_stored(tmp_path):
    now = [1234.5]
    cache = _cache(tmp_path, clock=lambda: now[0])
    key = _key()

    await cache.put(key, _entry(audio=b"stamped"))

    entry = cache.lru.get(key)
    assert entry is not None
    assert entry.created == 1234.5


async def test_a_store_restamps_an_entry_constructed_by_the_caller(tmp_path):
    """Callers build ``CacheEntry`` without a timestamp; store fills it in."""
    now = [500.0]
    cache = _cache(tmp_path, ttl_seconds=10.0, clock=lambda: now[0])

    await cache.put(_key(), _entry())
    entry = cache.lru.get(_key())

    assert entry is not None and entry.created == 500.0
    assert await cache.get(_key()) is not None
# ─────────────────────────────────────────────────────────────
# N1: the reported tier must be the tier that served the hit
# ─────────────────────────────────────────────────────────────

async def test_a_disk_entry_promoted_into_the_lru_reports_the_lru_from_then_on(tmp_path):
    """After a restart, later hits come from memory and must say so."""
    key = _key()
    writer = _cache(tmp_path)
    await writer.put(key, _entry())

    reader = _cache(tmp_path)
    first = await reader.get(key)
    second = await reader.get(key)

    assert first is not None and first.tier == "disk"
    assert second is not None and second.tier == "lru", (
        "the promoted entry kept tier='disk', so X-Cache-Tier lied about serving it from memory"
    )

    stats = reader.stats_dict()
    assert stats["hits"] == 2
    assert stats["hits_disk"] == 1
    assert stats["hits_lru"] == 1


async def test_promotion_preserves_the_audio_and_duration(tmp_path):
    key = _key()
    entry = _entry(audio=b"RIFF-promoted", duration=2.5)
    await _cache(tmp_path).put(key, entry)

    reader = _cache(tmp_path)
    from_disk = await reader.get(key)
    promoted = await reader.get(key)

    assert from_disk is not None and promoted is not None
    assert promoted.audio == entry.audio
    assert promoted.duration == entry.duration
    # The promotion keeps the original creation time, so the TTL still counts
    # from the render and not from the restart.
    assert promoted.created == from_disk.created > 0.0


async def test_the_synchronous_lookup_also_promotes_with_the_right_tier(tmp_path):
    key = _key()
    _cache(tmp_path).put_sync(key, _entry())

    reader = _cache(tmp_path)
    assert reader.get_sync(key).tier == "disk"
    assert reader.get_sync(key).tier == "lru"
    assert reader.stats.hits_disk == 1
    assert reader.stats.hits_lru == 1


# ─────────────────────────────────────────────────────────────
# N2: the renderer fingerprint keeps deployment changes out of stale entries
# ─────────────────────────────────────────────────────────────

def test_renderer_fingerprint_is_stable_and_short():
    first = renderer_fingerprint()
    second = renderer_fingerprint()

    assert first == second
    assert len(first) == 16
    assert all(character in "0123456789abcdef" for character in first)


def test_the_fingerprint_changes_when_the_default_speed_changes(monkeypatch):
    before = renderer_fingerprint()

    monkeypatch.setattr(settings, "kokoro_default_speed", 0.75)

    assert renderer_fingerprint() != before, (
        "KOKORO_DEFAULT_SPEED is applied when the caller sends no rate, so a "
        "change must invalidate the cache"
    )


def test_the_fingerprint_changes_when_phonetic_processing_toggles(monkeypatch):
    monkeypatch.setattr(settings, "phonetic_processing_enabled", True)
    enabled = renderer_fingerprint()

    monkeypatch.setattr(settings, "phonetic_processing_enabled", False)
    disabled = renderer_fingerprint()

    assert enabled != disabled


def test_the_fingerprint_tracks_the_lexicon_file(tmp_path, monkeypatch):
    lexicon = tmp_path / "lexicon.json"
    monkeypatch.setattr(settings, "lexicon_file_path", str(lexicon))

    missing = renderer_fingerprint()
    lexicon.write_text('{"entries": {}}', encoding="utf-8")
    empty = renderer_fingerprint()
    lexicon.write_text('{"entries": {"Reactory": {"replacement": "Ree-actory"}}}', encoding="utf-8")
    edited = renderer_fingerprint()

    assert missing != empty, "creating the lexicon must change the fingerprint"
    assert empty != edited, "editing the lexicon must change the fingerprint"


def test_a_lexicon_mutation_produces_a_miss(tmp_path, monkeypatch):
    """The operator scenario: a pronunciation fix must not serve old audio."""
    lexicon = tmp_path / "lexicon.json"
    lexicon.write_text('{"entries": {}}', encoding="utf-8")
    monkeypatch.setattr(settings, "lexicon_file_path", str(lexicon))

    before = _key()
    lexicon.write_text('{"entries": {"Reactory": {"replacement": "Ree-actory"}}}', encoding="utf-8")
    after = _key()

    assert before != after


def test_a_default_speed_change_produces_a_miss(monkeypatch):
    before = _key()

    monkeypatch.setattr(settings, "kokoro_default_speed", 0.8)

    assert _key() != before


def test_an_explicit_renderer_overrides_the_environment():
    """Tests that only care about per-request identity can pin the fingerprint."""
    assert _key(renderer="pinned") == _key(renderer="pinned")
    assert _key(renderer="pinned") != _key(renderer="other")
    assert _key(renderer="pinned") != _key()

# ─────────────────────────────────────────────────────────────
# F2: the fingerprint also covers deploy and model identity
# ─────────────────────────────────────────────────────────────

def test_the_fingerprint_tracks_the_application_version(monkeypatch):
    """A deploy that changes the built-in lexicon or the render code must miss.

    ``SPEECH_CACHE_DIR`` can be a persisted volume, so without this a release
    would keep serving audio rendered by the previous code for the whole TTL.
    """
    import app.services.synthesis_cache as cache_module

    before = renderer_fingerprint()
    monkeypatch.setattr(cache_module, "_APP_VERSION", "99.0.0")

    assert renderer_fingerprint() != before


def test_the_fingerprint_tracks_the_model_and_voice_pack(tmp_path, monkeypatch):
    """Swapping the ONNX model or the voices bundle must invalidate entries."""
    model = tmp_path / "kokoro-v1.0.onnx"
    voices = tmp_path / "voices-v1.0.bin"
    monkeypatch.setattr(settings, "kokoro_model_path", str(model))
    monkeypatch.setattr(settings, "kokoro_voices_path", str(voices))

    absent = renderer_fingerprint()

    model.write_bytes(b"model-v1")
    with_model = renderer_fingerprint()
    assert with_model != absent, "adding the model did not change the fingerprint"

    voices.write_bytes(b"voices-v1")
    with_voices = renderer_fingerprint()
    assert with_voices != with_model, "adding the voice pack did not change the fingerprint"

    model.write_bytes(b"model-v2-with-a-different-size")
    assert renderer_fingerprint() != with_voices, "replacing the model did not change the fingerprint"


def test_a_missing_model_is_a_stable_fingerprint(tmp_path, monkeypatch):
    """The model files are absent before download; the key must still be stable."""
    monkeypatch.setattr(settings, "kokoro_model_path", str(tmp_path / "none.onnx"))
    monkeypatch.setattr(settings, "kokoro_voices_path", str(tmp_path / "none.bin"))

    assert renderer_fingerprint() == renderer_fingerprint()


def test_the_fingerprint_costs_only_file_metadata():
    """Documents the cost: a handful of stat calls, no file reads or hashing.

    It runs on every synthesis, so reading or hashing the lexicon and the model
    would be unacceptable.  Inspecting the code object (not the docstring) keeps
    this honest if the implementation later reaches for file contents.
    """
    import app.services.synthesis_cache as cache_module

    names = cache_module.renderer_fingerprint.__code__.co_names
    for forbidden in ("read_bytes", "read_text", "loads", "open"):
        assert forbidden not in names, f"renderer_fingerprint calls {forbidden}"
    # The only hashing is the final digest over the assembled string.
    assert cache_module.renderer_fingerprint.__code__.co_names.count("sha256") <= 1
