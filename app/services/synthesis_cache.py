"""Two-tier synthesis cache: an in-process LRU plus a TTL-bounded disk tier.

The edge profile has no Redis and no hosted cache, so the cache lives in the
process and on local disk:

* **LRU** — small, hot phrases (UI greetings, prompts, short previews).  It is
  bounded by entry count, total bytes and a per-entry ceiling so one long
  paragraph cannot evict the whole hot set.
* **disk** — everything the LRU accepts plus the larger renders.  Files are
  written atomically (``os.replace``) and expire on a TTL, so a restart reuses
  warm audio without pinning storage forever.

Cache identity is a SHA-256 over the inputs that change the bytes::

    text | voice id | direction | engine id | output format | sample rate

plus the legacy switches that also change the audio (``is_phonemes``,
``phonetic_preprocess``, ``custom_lexicon``).  A key is never *narrower* than
the inputs it represents: two requests with different keys may still produce
the same bytes, but two requests with the same key always must.

A hit returns the stored audio bytes verbatim — byte-identical to the response
that populated the entry.  Duration and sample rate travel with the audio so a
hit does not re-run inference or re-measure anything.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import tempfile
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from app import __version__ as _APP_VERSION
from app.config import settings
from app.core.contracts import Direction

logger = logging.getLogger(__name__)

#: The six components the cache key is defined over.
KEY_COMPONENTS: tuple[str, ...] = (
    "text",
    "voice",
    "direction",
    "engine",
    "output_format",
    "sample_rate",
)

AUDIO_SUFFIX = ".audio"
META_SUFFIX = ".json"


def canonical_direction(direction: Direction | None) -> str:
    """A stable, order-independent serialisation of a direction.

    Defaults are included: two directions that render identically must hash
    identically, and a field that is set-but-default is still part of identity.
    """
    if direction is None:
        return ""
    payload = direction.model_dump(mode="json")
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _file_identity(path: str | Path | None) -> tuple[int, int]:
    """``(mtime_ns, size)`` for a file, or ``(0, 0)`` when it is absent.

    A stat is three orders of magnitude cheaper than hashing, and either value
    changes whenever the file is replaced, which is what "the model changed"
    means in practice.
    """
    if path is None:
        return 0, 0
    try:
        stat = Path(path).stat()
    except OSError:
        return 0, 0
    return stat.st_mtime_ns, stat.st_size


def renderer_fingerprint() -> str:
    """A cheap digest of the *deployment-level* inputs that change the audio.

    The per-request key covers everything the caller controls.  The inputs below
    are decided by the deployment instead, and all of them change the rendered
    bytes:

    * the persistent lexicon (``data/lexicon.json``), which an operator can
      mutate at runtime through ``POST /api/tts/lexicon``;
    * ``KOKORO_DEFAULT_SPEED``, applied when the caller expresses no rate, while
      the key records ``Direction()``'s default rate;
    * ``PHONETIC_PROCESSING_ENABLED``, which chooses the render path when
      ``phonetic_preprocess`` is ``None``;
    * the application version, which covers the built-in lexicon, the
      normalisation rules and the render path itself;
    * the model and voice pack identities, so swapping the ONNX model (or the
      voices bundle) cannot be served by audio rendered by the previous one.

    Without them a pronunciation fix, a default-speed change or a deploy would
    keep serving the old audio for the whole TTL — a day by default, and across
    restarts, because the disk tier deliberately survives them.  A persisted
    ``SPEECH_CACHE_DIR`` across a deploy is therefore only as fresh as this
    fingerprint.

    Everything here is file metadata plus settings, so the cost is a handful of
    ``stat`` calls per synthesis.
    """
    lexicon_mtime, lexicon_size = _file_identity(settings.lexicon_file_path)
    model_mtime, model_size = _file_identity(
        getattr(settings, "kokoro_model_resolved", None)
    )
    voices_mtime, voices_size = _file_identity(
        getattr(settings, "kokoro_voices_resolved", None)
    )
    material = "|".join(
        (
            str(lexicon_mtime),
            str(lexicon_size),
            repr(float(settings.kokoro_default_speed)),
            "1" if settings.phonetic_processing_enabled else "0",
            _APP_VERSION,
            str(model_mtime),
            str(model_size),
            str(voices_mtime),
            str(voices_size),
        )
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def build_cache_key(
    *,
    text: str,
    voice: str | None,
    direction: Direction | None,
    engine: str,
    output_format: str,
    sample_rate: int,
    is_phonemes: bool = False,
    phonetic_preprocess: bool | None = None,
    custom_lexicon: dict[str, str] | None = None,
    renderer: str | None = None,
) -> str:
    """Return the SHA-256 cache key for one synthesis request.

    ``renderer`` is the :func:`renderer_fingerprint` of the deployment; it is
    computed on demand when omitted, so a caller cannot accidentally build a key
    that ignores the lexicon or the configured defaults.  Pass ``""`` to opt out
    (tests that only care about per-request identity).
    """
    lexicon = ""
    if custom_lexicon:
        lexicon = json.dumps(
            dict(sorted(custom_lexicon.items())),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
    material = "|".join(
        (
            text,
            voice or "",
            canonical_direction(direction),
            engine,
            output_format,
            str(int(sample_rate)),
            "1" if is_phonemes else "0",
            "" if phonetic_preprocess is None else ("1" if phonetic_preprocess else "0"),
            lexicon,
            renderer if renderer is not None else renderer_fingerprint(),
        )
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CacheEntry:
    """One cached render: the audio plus what a response needs to describe it.

    ``created`` is the epoch time the entry was stored.  Both tiers read it, so
    the TTL applies to an in-process hit exactly as it does to a disk hit
    (``asyncio``); a hot phrase must not outlive the configured TTL merely
    because it never falls out of the LRU.
    """

    audio: bytes
    duration: float
    sample_rate: int
    engine: str
    tier: str = "lru"
    key: str = ""
    created: float = 0.0

    def __len__(self) -> int:  # lets ``len(entry)`` report the audio size
        return len(self.audio)


@dataclass
class CacheStats:
    hits: int = 0
    hits_lru: int = 0
    hits_disk: int = 0
    misses: int = 0
    writes: int = 0
    evictions: int = 0
    errors: int = 0
    lru_entries: int = 0
    lru_bytes: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "hits_lru": self.hits_lru,
            "hits_disk": self.hits_disk,
            "misses": self.misses,
            "writes": self.writes,
            "evictions": self.evictions,
            "errors": self.errors,
            "lru_entries": self.lru_entries,
            "lru_bytes": self.lru_bytes,
        }


@dataclass
class _LRU:
    """A byte- and entry-bounded LRU map."""

    max_entries: int
    max_bytes: int
    max_entry_bytes: int
    entries: OrderedDict[str, CacheEntry] = field(default_factory=OrderedDict)
    total_bytes: int = 0

    def get(self, key: str) -> CacheEntry | None:
        entry = self.entries.get(key)
        if entry is None:
            return None
        self.entries.move_to_end(key)
        return entry

    def put(self, key: str, entry: CacheEntry) -> bool:
        """Store an entry; returns ``False`` when it is too large for this tier."""
        if len(entry.audio) > self.max_entry_bytes:
            return False
        previous = self.entries.pop(key, None)
        if previous is not None:
            self.total_bytes -= len(previous.audio)
        self.entries[key] = entry
        self.total_bytes += len(entry.audio)
        while len(self.entries) > self.max_entries or self.total_bytes > self.max_bytes:
            _, evicted = self.entries.popitem(last=False)
            self.total_bytes -= len(evicted.audio)
        return True

    def pop(self, key: str) -> CacheEntry | None:
        """Remove and return an entry, keeping the byte total accurate."""
        entry = self.entries.pop(key, None)
        if entry is not None:
            self.total_bytes -= len(entry.audio)
        return entry

    def clear(self) -> None:
        self.entries.clear()
        self.total_bytes = 0
    


class SynthesisCache:
    """In-process LRU fronting a TTL-bounded on-disk cache.

    ``enabled``, ``ttl_seconds`` and the directory are read from settings on
    every access unless a constructor argument overrides them, so an operator
    can change the environment and a test can isolate itself without rebuilding
    the service.
    """

    def __init__(
        self,
        *,
        enabled: bool | None = None,
        directory: str | Path | None = None,
        ttl_seconds: float | None = None,
        lru_entries: int | None = None,
        lru_max_bytes: int | None = None,
        lru_max_entry_bytes: int | None = None,
        disk_max_bytes: int | None = None,
        clock: Any = None,
    ) -> None:
        self._enabled = enabled
        self._directory = Path(directory) if directory is not None else None
        self._ttl_seconds = ttl_seconds
        self._lru_entries = lru_entries
        self._lru_max_bytes = lru_max_bytes
        self._lru_max_entry_bytes = lru_max_entry_bytes
        self._disk_max_bytes = disk_max_bytes
        self._clock = clock or time.time
        #: Guards the LRU and the counters.  Disk IO runs in a worker thread
        #: (``asyncio.to_thread``), so the tiers are genuinely shared state.
        self._lock = threading.RLock()
        self._lru: _LRU | None = None
        self._lru_signature: tuple[int, int, int] | None = None
        self.stats = CacheStats()
        self._disk_writes_since_prune = 0

    # ------------------------------------------------------------------
    # Configuration (read dynamically unless explicitly overridden)
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        if self._enabled is not None:
            return bool(self._enabled)
        return bool(settings.speech_cache_enabled)

    @property
    def ttl_seconds(self) -> float:
        if self._ttl_seconds is not None:
            return float(self._ttl_seconds)
        return float(settings.speech_cache_ttl_seconds)

    @property
    def directory(self) -> Path:
        if self._directory is not None:
            return self._directory
        return settings.speech_cache_dir_resolved

    @property
    def _tier_limits(self) -> tuple[int, int, int]:
        return (
            int(self._lru_entries if self._lru_entries is not None else settings.speech_cache_lru_entries),
            int(
                self._lru_max_bytes
                if self._lru_max_bytes is not None
                else settings.speech_cache_lru_max_bytes
            ),
            int(
                self._lru_max_entry_bytes
                if self._lru_max_entry_bytes is not None
                else settings.speech_cache_lru_max_entry_bytes
            ),
        )

    @property
    def lru(self) -> _LRU:
        with self._lock:
            signature = self._tier_limits
            if self._lru is None or self._lru_signature != signature:
                previous = self._lru
                self._lru = _LRU(
                    max_entries=signature[0],
                    max_bytes=signature[1],
                    max_entry_bytes=signature[2],
                )
                if previous is not None:
                    for key, entry in previous.entries.items():
                        self._lru.put(key, entry)
                self._lru_signature = signature
            return self._lru

    @property
    def disk_enabled(self) -> bool:
        return bool(str(self.directory))

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    async def get(self, key: str) -> CacheEntry | None:
        """Look up a key, preferring the in-process tier.

        The TTL is enforced on *both* tiers.  A hot phrase is never evicted by
        pressure, so without this check a shortened ``SPEECH_CACHE_TTL_SECONDS``
        would never take effect for exactly the entries the LRU exists to serve.
        An expired LRU entry is dropped and counted as an eviction, then the disk
        tier is consulted (which applies the same TTL to the same timestamp).
        """
        if not self.enabled:
            return None
        with self._lock:
            entry = self._live_lru_entry(key)
            if entry is not None:
                self._record_hit(entry)
                return entry
        entry = await asyncio.to_thread(self._disk_get, key)
        if entry is not None:
            with self._lock:
                self._promote(key, entry)
                self._record_hit(entry)
            return entry
        with self._lock:
            self.stats.misses += 1
        return None

    def get_sync(self, key: str) -> CacheEntry | None:
        """Synchronous lookup for the legacy (non-request-path) callers."""
        if not self.enabled:
            return None
        with self._lock:
            entry = self._live_lru_entry(key)
            if entry is not None:
                self._record_hit(entry)
                return entry
        entry = self._disk_get(key)
        if entry is not None:
            with self._lock:
                self._promote(key, entry)
                self._record_hit(entry)
            return entry
        with self._lock:
            self.stats.misses += 1
        return None

    def _expired(self, entry: CacheEntry) -> bool:
        """Whether an entry has outlived the configured TTL.

        A non-positive TTL means "no expiry", matching the disk tier's behaviour.
        """
        ttl = self.ttl_seconds
        if ttl <= 0:
            return False
        return (self._clock() - entry.created) > ttl

    def _live_lru_entry(self, key: str) -> CacheEntry | None:
        """Return a non-expired LRU entry, dropping an expired one.

        Callers hold :attr:`_lock`.
        """
        entry = self.lru.get(key)
        if entry is None:
            return None
        if not self._expired(entry):
            return entry
        self.lru.pop(key)
        self.stats.evictions += 1
        self.stats.lru_entries = len(self.lru.entries)
        self.stats.lru_bytes = self.lru.total_bytes
        return None

    def _promote(self, key: str, entry: CacheEntry) -> None:
        """Insert a disk entry into the LRU as an LRU-resident copy.

        The copy must be re-tagged ``tier="lru"``.  Without that, every later
        request served *from memory* still reports ``X-Cache-Tier: disk`` and
        increments ``hits_disk``, so the operator can never see the LRU working
        after a restart.  Callers hold :attr:`_lock`.
        """
        self.lru.put(key, replace(entry, tier="lru"))

    def _record_hit(self, entry: CacheEntry) -> None:
        """Count a hit against the tier that served *this* request.

        ``entry`` is the entry that was just read, so a first request after a
        restart counts as a disk hit and every subsequent one as an LRU hit.
        Callers hold :attr:`_lock`.
        """
        self.stats.hits += 1
        if entry.tier == "disk":
            self.stats.hits_disk += 1
        else:
            self.stats.hits_lru += 1

    # ------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------

    async def put(self, key: str, entry: CacheEntry) -> None:
        """Store a render in both tiers (the LRU may reject large entries)."""
        if not self.enabled:
            return
        await asyncio.to_thread(self._store, key, entry)

    def put_sync(self, key: str, entry: CacheEntry) -> None:
        if not self.enabled:
            return
        self._store(key, entry)

    def _store(self, key: str, entry: CacheEntry) -> None:
        stored = CacheEntry(
            audio=entry.audio,
            duration=entry.duration,
            sample_rate=entry.sample_rate,
            engine=entry.engine,
            tier="lru",
            key=key,
            created=self._clock(),
        )
        with self._lock:
            if not self.lru.put(key, stored):
                self.stats.evictions += 1
            self.stats.writes += 1
            self.stats.lru_entries = len(self.lru.entries)
            self.stats.lru_bytes = self.lru.total_bytes
        if self.disk_enabled:
            self._disk_put(key, stored)

    # ------------------------------------------------------------------
    # Disk tier
    # ------------------------------------------------------------------

    def _paths(self, key: str) -> tuple[Path, Path]:
        shard = self.directory / key[:2]
        return shard / f"{key}{AUDIO_SUFFIX}", shard / f"{key}{META_SUFFIX}"

    def _disk_get(self, key: str) -> CacheEntry | None:
        if not self.disk_enabled:
            return None
        audio_path, meta_path = self._paths(key)
        try:
            if not audio_path.is_file() or not meta_path.is_file():
                return None
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            created = float(meta.get("created", 0.0))
            if self.ttl_seconds > 0 and (self._clock() - created) > self.ttl_seconds:
                self._discard(audio_path, meta_path)
                return None
            audio = audio_path.read_bytes()
            if len(audio) != int(meta.get("size", len(audio))):
                # A truncated file is a corrupt entry, not a cache hit.
                self._discard(audio_path, meta_path)
                with self._lock:
                    self.stats.errors += 1
                return None
            return CacheEntry(
                audio=audio,
                duration=float(meta.get("duration", 0.0)),
                sample_rate=int(meta.get("sample_rate", 0)),
                engine=str(meta.get("engine", "")),
                tier="disk",
                key=key,
                created=created,
            )
        except (OSError, ValueError, TypeError):
            logger.warning("Ignoring unreadable cache entry %s", key, exc_info=True)
            with self._lock:
                self.stats.errors += 1
            return None

    def _disk_put(self, key: str, entry: CacheEntry) -> None:
        audio_path, meta_path = self._paths(key)
        meta = {
            "created": self._clock(),
            "duration": entry.duration,
            "sample_rate": entry.sample_rate,
            "engine": entry.engine,
            "size": len(entry.audio),
            "ttl_seconds": self.ttl_seconds,
        }
        try:
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            self._atomic_write(audio_path, entry.audio)
            self._atomic_write(
                meta_path,
                json.dumps(meta, sort_keys=True, separators=(",", ":")).encode("utf-8"),
            )
        except OSError:
            logger.warning("Could not write cache entry %s", key, exc_info=True)
            with self._lock:
                self.stats.errors += 1
            return
        with self._lock:
            self._disk_writes_since_prune += 1
            prune = self._disk_writes_since_prune >= 64
            if prune:
                self._disk_writes_since_prune = 0
        if prune:
            self._prune_disk()

    @staticmethod
    def _atomic_write(path: Path, data: bytes) -> None:
        fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=path.suffix)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_name, path)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

    @staticmethod
    def _discard(*paths: Path) -> None:
        for path in paths:
            try:
                path.unlink()
            except OSError:
                pass

    def _disk_size(self) -> tuple[int, list[tuple[float, Path]]]:
        entries: list[tuple[float, Path]] = []
        total = 0
        try:
            for path in self.directory.glob("*/*" + AUDIO_SUFFIX):
                try:
                    stat = path.stat()
                except OSError:
                    continue
                total += stat.st_size
                entries.append((stat.st_mtime, path))
        except OSError:
            return total, entries
        return total, entries

    def _prune_disk(self) -> None:
        """Keep the disk tier inside its byte budget by deleting the oldest entries."""
        limit = int(
            self._disk_max_bytes
            if self._disk_max_bytes is not None
            else settings.speech_cache_disk_max_bytes
        )
        if limit <= 0:
            return
        total, entries = self._disk_size()
        if total <= limit:
            return
        entries.sort()
        for _, audio_path in entries:
            if total <= limit:
                break
            try:
                size = audio_path.stat().st_size
            except OSError:
                continue
            self._discard(audio_path, audio_path.with_suffix(META_SUFFIX))
            total -= size
            with self._lock:
                self.stats.evictions += 1

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Drop both tiers.  Used by tests and by an operator-facing reset."""
        with self._lock:
            self.lru.clear()
            self.stats.lru_entries = 0
            self.stats.lru_bytes = 0
        if not self.disk_enabled or not self.directory.exists():
            return
        for path in list(self.directory.glob("*/*")):
            if path.suffix in (AUDIO_SUFFIX, META_SUFFIX):
                self._discard(path)

    def stats_dict(self) -> dict[str, Any]:
        """Cache counters plus the configuration they are running under."""
        with self._lock:
            payload = self.stats.as_dict()
            payload.update(
                {
                    "enabled": self.enabled,
                    "ttl_seconds": self.ttl_seconds,
                    "directory": str(self.directory),
                    "lru_entries": len(self.lru.entries),
                    "lru_bytes": self.lru.total_bytes,
                }
            )
        return payload


__all__ = [
    "AUDIO_SUFFIX",
    "CacheEntry",
    "CacheStats",
    "KEY_COMPONENTS",
    "META_SUFFIX",
    "SynthesisCache",
    "build_cache_key",
    "canonical_direction",
    "renderer_fingerprint",
]
