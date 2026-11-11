"""Designed-voice registry: recipes, persistence and derivation helpers.

A *designed voice* is a durable **recipe**, never a tensor.  Two recipe kinds
are supported:

``blend``
    A weighted sum of 2-4 base Kokoro style tensors.  Because every Kokoro
    voice is a style tensor of shape ``(510, 1, 256)``, the weighted sum is a
    valid style input for ``Kokoro.create()`` — a new voice identity with no
    training and no second model.

``base``
    An alias for one base voice, so a product can give the shipped voice a
    product name without duplicating pack data.

The resolved tensor is derived on demand by
:mod:`app.engines.kokoro` and cached there keyed by :func:`recipe_hash`; this
module only ever handles the recipe, the metadata and the preview cache.

Persistence is one JSON file (``DESIGNED_VOICES_PATH``).  Loading is
deliberately tolerant: a missing or malformed file logs and yields **zero**
designed voices, so a bad file can never stop the pack voices from loading or
the service from starting.

Blending changes **timbre**, not **emotion**.  Nothing in this module claims
otherwise; an expressive delivery needs the Phase 4/5 work, not a blend.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)

#: Repository root, used to resolve the default relative data paths.
REPO_ROOT = Path(__file__).resolve().parents[2]

#: Shipped designed-voice registry, resolved relative to the package so the
#: service finds it regardless of the process working directory.
DEFAULT_DESIGNED_VOICES_PATH = REPO_ROOT / "data" / "designed_voices.json"

#: Minimum and maximum number of components in a blend.
MIN_BLEND_COMPONENTS = 2
MAX_BLEND_COMPONENTS = 4

#: The heaviest component must carry at least this share of the total weight;
#: flatter mixes sound like neither voice.
MIN_DOMINANT_WEIGHT = 0.6

#: Tolerance for floating-point comparisons of normalised weights.
WEIGHT_EPSILON = 1e-9

#: Tag every designed voice carries; the ``preset`` *field* marks shipped
#: registry data (and is what makes a voice immutable).
DESIGNED_TAG = "designed"

#: Style-spec kinds this module can persist.
DESIGNED_STYLE_TYPES: frozenset[str] = frozenset({"base", "blend"})

#: Schema version of the designed-voice file.
SCHEMA_VERSION = 1

#: Sentence synthesised for a preview.  Fixed so a preview cache hit is
#: byte-identical regardless of which voice triggered it.
PREVIEW_TEXT = "This is a preview of a designed voice from the Reactory speech service."


class VoiceRecipeError(ValueError):
    """A recipe violates a blending rule.  The HTTP layer maps this to 422."""


class VoiceConflictError(VoiceRecipeError):
    """The id or name is already taken, or the voice is immutable.  Maps to 409."""


class DesignedVoiceStoreError(RuntimeError):
    """The registry could not be written.  Maps to 500."""


# ─────────────────────────────────────────────────────────────
# Recipe helpers (pure)
# ─────────────────────────────────────────────────────────────

def _component_weight(component: Any) -> float:
    weight = component.get("weight") if isinstance(component, dict) else None
    if isinstance(weight, bool) or not isinstance(weight, (int, float)):
        raise VoiceRecipeError("Every component needs a numeric 'weight'.")
    value = float(weight)
    if not math.isfinite(value):
        raise VoiceRecipeError("Component weights must be finite numbers.")
    return value


def raw_components(style_spec: dict[str, Any]) -> list[tuple[str, float]]:
    """Return ``(voice, weight)`` pairs exactly as persisted (order preserved)."""
    components = style_spec.get("components")
    if not isinstance(components, list):
        return []
    parsed: list[tuple[str, float]] = []
    for component in components:
        voice = component.get("voice") if isinstance(component, dict) else None
        if not isinstance(voice, str) or not voice:
            raise VoiceRecipeError("Every component needs a non-empty 'voice' id.")
        parsed.append((voice, _component_weight(component)))
    return parsed


def normalised_components(style_spec: dict[str, Any]) -> list[dict[str, Any]]:
    """The recipe with weights normalised to 1 and ordered by voice id.

    Ordering by voice id makes the result (and therefore :func:`recipe_hash`)
    independent of the order components were supplied in, so two callers who
    describe the same blend share one cache entry.  Rounding keeps the hash
    stable against floating-point noise from the normalisation itself.
    """
    parsed = raw_components(style_spec)
    total = sum(weight for _, weight in parsed)
    if total <= 0:
        return []
    return [
        {"voice": voice, "weight": round(weight / total, 9)}
        for voice, weight in sorted(parsed, key=lambda item: item[0])
    ]


def recipe_hash(style_spec: dict[str, Any]) -> str:
    """A stable SHA-256 of the *normalised* recipe.

    This is the tensor-cache key and the preview-cache key: the same recipe
    always maps to the same hash, and any change to a weight or a component
    produces a different one.
    """
    style_type = str(style_spec.get("type", "base"))
    if style_type == "base":
        canonical: dict[str, Any] = {
            "type": "base",
            "voice": str(style_spec.get("voice", "")),
        }
    else:
        canonical = {"type": style_type, "components": normalised_components(style_spec)}
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_components(components: list[dict[str, Any]]) -> list[tuple[str, float]]:
    """Validate a caller-supplied component list, returning ``(voice, weight)``.

    Only the rules that do not need the voice registry live here (count,
    duplicates, weight sanity, dominant weight); component *existence* and the
    single-language-family rule are enforced by the design service, which can
    see the catalogue.
    """
    if len(components) < MIN_BLEND_COMPONENTS:
        raise VoiceRecipeError(
            f"A blend needs at least {MIN_BLEND_COMPONENTS} components; "
            f"got {len(components)}. Blend two to four base voices — a "
            "single component is just that base voice."
        )
    if len(components) > MAX_BLEND_COMPONENTS:
        raise VoiceRecipeError(
            f"A blend accepts at most {MAX_BLEND_COMPONENTS} components; "
            f"got {len(components)}. More components produce mud rather than a "
            "new identity."
        )

    parsed = raw_components({"type": "blend", "components": components})

    voices = [voice for voice, _ in parsed]
    duplicates = sorted({voice for voice in voices if voices.count(voice) > 1})
    if duplicates:
        raise VoiceRecipeError(
            f"Duplicate component voices: {', '.join(duplicates)}. "
            "Give each base voice one weight."
        )

    negative = [voice for voice, weight in parsed if weight < 0]
    if negative:
        raise VoiceRecipeError(
            f"Component weights must be non-negative; {', '.join(negative)} "
            "carries a negative weight."
        )

    total = sum(weight for _, weight in parsed)
    if total <= 0:
        raise VoiceRecipeError(
            "Component weights must sum to a value greater than zero; "
            f"the supplied weights sum to {total:g}."
        )

    dominant_voice, dominant_weight = max(parsed, key=lambda item: item[1])
    if dominant_weight / total + WEIGHT_EPSILON < MIN_DOMINANT_WEIGHT:
        raise VoiceRecipeError(
            "A blend needs one dominant component carrying at least "
            f"{MIN_DOMINANT_WEIGHT:.0%} of the total weight; the largest is "
            f"'{dominant_voice}' at {dominant_weight / total:.0%}. "
            "Raise its weight or drop a component."
        )

    return parsed


# ─────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────

def _absolute(path: Path) -> Path:
    """Anchor a relative path at the repository root (never at the CWD)."""
    return path if path.is_absolute() else REPO_ROOT / path


def resolve_store_path(path: str | Path | None = None) -> Path:
    """Resolve the registry path from ``Settings`` (or an explicit override).

    Both this module and :mod:`app.config` need the path; this is the only
    place that decides it, and it defers to ``Settings.designed_voices_resolved``
    so the two cannot drift.
    """
    if path is not None:
        return _absolute(Path(path))
    return _absolute(settings.designed_voices_resolved)


def resolve_preview_dir(store_path: Path | None = None) -> Path:
    """Resolve the preview directory, deferring to ``Settings``.

    A configured ``DESIGNED_VOICES_PREVIEW_DIR`` always wins.  Otherwise the
    previews live beside the registry file, which lets a store built on an
    explicit path keep its previews with it.
    """
    if settings.designed_voices_preview_dir:
        return _absolute(settings.designed_voices_preview_resolved)
    base = store_path if store_path is not None else _absolute(
        settings.designed_voices_resolved
    )
    return base.parent / "previews"


def _entry_to_record(entry: dict[str, Any]) -> Any:
    """Convert one persisted entry into a :class:`VoiceRecord`.

    Reuses the pack registry's record parser so a designed voice has exactly
    the same shape as a pack voice everywhere downstream (descriptors, filters,
    engine resolution).
    """
    from app.services.voice_registry import _parse_record

    record = _parse_record(entry, 0)
    style_type = record.style_type
    if style_type not in DESIGNED_STYLE_TYPES:
        raise ValueError(f"designed voice '{record.id}' has unsupported style type {style_type!r}")
    if style_type == "blend":
        components = entry["style_spec"].get("components")
        if not isinstance(components, list) or not components:
            raise ValueError(f"designed voice '{record.id}' has no blend components")
        raw_components(entry["style_spec"])
    return record


class DesignedVoiceStore:
    """A thread-safe, file-backed collection of designed-voice entries."""

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = resolve_store_path(path)
        self._entries: dict[str, dict[str, Any]] = {}
        self._records: dict[str, Any] = {}
        self._lock = threading.RLock()
        #: Why the last load could not use the whole file, or ``None``.  Cleared
        #: once the original has been preserved, so the first write after a bad
        #: load never overwrites evidence it could not read.
        self._unusable_source: str | None = None
        #: Entries dropped by the last load.  A tolerant load can hold fewer
        #: entries than the file, so a rewrite must preserve the original first.
        self._discarded_entries = 0
        self.load()

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> None:
        """(Re)load the file, tolerating a missing or malformed registry.

        A missing file is normal (nothing designed yet).  Anything unreadable is
        logged and degrades to the pack voices only — startup never fails.  The
        reason is remembered so :meth:`_write` can move the original aside
        instead of silently replacing it.
        """
        with self._lock:
            self._entries = {}
            self._records = {}
            self._unusable_source = None
            self._discarded_entries = 0
            try:
                text = self._path.read_text(encoding="utf-8")
            except FileNotFoundError:
                logger.info(
                    "Designed-voice registry %s does not exist yet; continuing with "
                    "the shipped pack voices only.",
                    self._path,
                )
                return
            except OSError as exc:
                self._unusable_source = f"unreadable: {exc}"
                logger.warning(
                    "Designed-voice registry %s could not be read (%s); continuing "
                    "with the shipped pack voices only.",
                    self._path,
                    exc,
                )
                return

            try:
                payload = json.loads(text)
            except json.JSONDecodeError as exc:
                self._unusable_source = f"invalid JSON: {exc}"
                logger.warning(
                    "Designed-voice registry %s is not valid JSON (%s); continuing "
                    "with the shipped pack voices only.",
                    self._path,
                    exc,
                )
                return

            entries = payload.get("voices") if isinstance(payload, dict) else None
            if not isinstance(entries, list):
                self._unusable_source = "no 'voices' list"
                logger.warning(
                    "Designed-voice registry %s has no 'voices' list; continuing with "
                    "the shipped pack voices only.",
                    self._path,
                )
                return

            for index, entry in enumerate(entries):
                if not isinstance(entry, dict):
                    self._discarded_entries += 1
                    logger.warning("Skipping designed-voice entry #%d: not an object.", index)
                    continue
                try:
                    record = _entry_to_record(entry)
                except Exception as exc:  # noqa: BLE001 - tolerant by design
                    self._discarded_entries += 1
                    logger.warning(
                        "Skipping invalid designed-voice entry #%d (%s): %s",
                        index,
                        entry.get("id", "<no id>"),
                        exc,
                    )
                    continue
                if record.id in self._records:
                    self._discarded_entries += 1
                    logger.warning(
                        "Skipping duplicate designed-voice id %r in %s.",
                        record.id,
                        self._path,
                    )
                    continue
                self._entries[record.id] = dict(entry)
                self._records[record.id] = record

            logger.info(
                "Loaded %d designed voice(s) from %s.", len(self._records), self._path
            )

    # -- reads ---------------------------------------------------------

    def get(self, voice_id: str) -> Any:
        """Return the :class:`VoiceRecord` for a designed voice, or ``None``."""
        with self._lock:
            return self._records.get(voice_id)

    def get_entry(self, voice_id: str) -> dict[str, Any] | None:
        """Return the raw persisted entry (a copy), or ``None``."""
        with self._lock:
            entry = self._entries.get(voice_id)
            return dict(entry) if entry is not None else None

    def records(self) -> list[Any]:
        with self._lock:
            return list(self._records.values())

    def ids(self, *, enabled_only: bool = False) -> tuple[str, ...]:
        with self._lock:
            return tuple(
                voice_id
                for voice_id, record in self._records.items()
                if record.enabled or not enabled_only
            )

    def preset_ids(self) -> tuple[str, ...]:
        """Ids of the shipped, immutable preset voices in this registry."""
        with self._lock:
            return tuple(
                sorted(
                    voice_id
                    for voice_id, entry in self._entries.items()
                    if entry.get("preset")
                )
            )

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)

    # -- writes --------------------------------------------------------

    def add(self, entry: dict[str, Any]) -> Any:
        """Validate and persist a new entry, returning its record."""
        record = _entry_to_record(entry)
        with self._lock:
            if record.id in self._records:
                raise VoiceConflictError(
                    f"A voice with id '{record.id}' already exists. "
                    "GET /api/tts/voices lists the current ids."
                )
            entries = dict(self._entries)
            entries[record.id] = dict(entry)
            self._write(entries)
            # Only mutate in-memory state after the file write succeeded, so a
            # failed write cannot leave a voice that vanishes on restart.
            self._entries = entries
            self._records[record.id] = record
            return record

    def update(self, voice_id: str, updates: dict[str, Any]) -> Any:
        """Merge fields into an existing entry and persist it."""
        with self._lock:
            current = self._entries.get(voice_id)
            if current is None:
                raise KeyError(voice_id)
            entry = {**current, **updates}
            record = _entry_to_record(entry)
            entries = dict(self._entries)
            entries[voice_id] = entry
            self._write(entries)
            self._entries = entries
            self._records[voice_id] = record
            return record

    def delete(self, voice_id: str) -> dict[str, Any]:
        """Remove an entry from the registry and return it."""
        with self._lock:
            entry = self._entries.get(voice_id)
            if entry is None:
                raise KeyError(voice_id)
            entries = dict(self._entries)
            entries.pop(voice_id, None)
            self._write(entries)
            self._entries = entries
            self._records.pop(voice_id, None)
            return dict(entry)

    def _preserve_source_before_overwrite(self) -> None:
        """Move a partly or wholly unreadable registry aside before rewriting it.

        Tolerant loading means the in-memory state can hold fewer entries than
        the file (a corrupt file yields none; a partly corrupt one loses the bad
        row).  Rewriting in place would turn that recoverable problem into
        permanent loss — including the four shipped presets — so the original is
        preserved once, under a timestamped ``.corrupt-*`` name, and its path is
        logged.  If it cannot be preserved the write is refused rather than
        destroying data.
        """
        if self._unusable_source is None and not self._discarded_entries:
            return
        if not self._path.exists():  # pragma: no cover - deleted between load and write
            self._unusable_source = None
            self._discarded_entries = 0
            return

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup = self._path.with_name(f"{self._path.name}.corrupt-{stamp}")
        try:
            os.replace(self._path, backup)
        except OSError as exc:
            logger.error(
                "Refusing to overwrite the unusable designed-voice registry %s: its "
                "contents could not be preserved (%s). Repair or move the file, then "
                "restart.",
                self._path,
                exc,
            )
            raise DesignedVoiceStoreError(
                "The designed-voice registry could not be read and its contents could "
                f"not be preserved, so the write was refused: {exc}"
            ) from exc

        discarded = self._discarded_entries
        logger.warning(
            "Preserved the unusable designed-voice registry as %s (%s, %d discarded "
            "%s). A fresh registry is written next; the backup still holds the shipped "
            "presets and any recoverable entries.",
            backup,
            self._unusable_source or "unreadable entries",
            discarded,
            "entry" if discarded == 1 else "entries",
        )
        self._unusable_source = None
        self._discarded_entries = 0

    def _write(self, entries: dict[str, dict[str, Any]]) -> None:
        # Single-process store: this rewrites the whole file, so concurrent
        # writers are last-writer-wins and a multi-worker deployment must use one
        # worker (or an external store) to keep designed voices consistent.
        self._preserve_source_before_overwrite()
        payload = {
            "schema_version": SCHEMA_VERSION,
            "description": (
                "Designed voices. A record stores the RECIPE, never the resolved "
                "style tensor: blends are derived and cached at runtime."
            ),
            "license": "Apache-2.0",
            "attribution": "Kokoro-82M v1.0",
            "voices": list(entries.values()),
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_name(f".{self._path.name}.tmp")
        try:
            temporary.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            os.replace(temporary, self._path)
        except OSError as exc:
            logger.exception("Could not write the designed-voice registry %s", self._path)
            try:
                temporary.unlink(missing_ok=True)
            except OSError:  # pragma: no cover - best effort cleanup
                pass
            raise DesignedVoiceStoreError(
                f"Could not persist the designed-voice registry: {exc}"
            ) from exc


# ─────────────────────────────────────────────────────────────
# Process-wide store
# ─────────────────────────────────────────────────────────────

_store: DesignedVoiceStore | None = None
_store_lock = threading.Lock()


def get_store() -> DesignedVoiceStore:
    """The process-wide store, reloaded when the configured path changes."""
    global _store
    path = resolve_store_path()
    with _store_lock:
        if _store is None or _store.path != path:
            _store = DesignedVoiceStore(path)
        return _store


def reset_store(path: str | Path | None = None) -> DesignedVoiceStore:
    """Replace the process-wide store (tests, and operators clearing state)."""
    global _store
    with _store_lock:
        _store = DesignedVoiceStore(path)
        return _store


def preset_voice_ids(path: str | Path | None = None) -> tuple[str, ...]:
    """Ids of the shipped preset voices in ``path`` (default: the shipped file)."""
    store = DesignedVoiceStore(path or DEFAULT_DESIGNED_VOICES_PATH)
    return store.preset_ids()
