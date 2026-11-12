"""Data-driven voice registry.

``app/data/voices.json`` is the **only** source of voice data for the speech
service.  Engine adapters, routers and services must never hard-code a voice id;
they ask this module instead.  A test greps ``app/services``, ``app/routers`` and
``app/engines`` for registry ids to keep that true.

Two things are deliberately *not* voice data and stay in code:

* :data:`LANGUAGE_FAMILIES` — the Kokoro id *scheme* (``<language><gender>_<name>``)
  used as a fallback for ids that are not (yet) in the registry, and
* engine constants such as the sample rate and the speed range, which describe
  the engine rather than a voice.

A record carries identity, language, licence, provenance, capability flags and a
``style_spec`` describing how to render it.  ``enabled`` is honoured everywhere:
a disabled record is neither advertised nor synthesised.

Designed voices (blends / base aliases, see :mod:`app.services.designed_voices`)
live in a second, user-writable registry.  :func:`get_record` consults both so a
designed voice is indistinguishable from a pack voice to every caller.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.config import settings
from app.core.contracts import VoiceDescriptor

logger = logging.getLogger(__name__)

#: Engine id used for the Kokoro ONNX engine in voice descriptors and capabilities.
KOKORO_ENGINE = "kokoro"

#: Native output sample rate of the Kokoro-82M ONNX model, in Hz.
KOKORO_SAMPLE_RATE = 24000

#: Speed multiplier range accepted by kokoro-onnx (it raises outside this range).
KOKORO_SPEED_MIN = 0.5
KOKORO_SPEED_MAX = 2.0

#: Audio styles expressed natively by the Kokoro voice pack.  Kokoro has no
#: emotion/prosody control, so every voice exposes only the neutral style.
KOKORO_STYLES = ["neutral"]

#: Fallback language code for ids that do not follow the Kokoro id scheme.
DEFAULT_LOCALE = "en-us"

#: Tag applied to a voice whose quality grade the catalogue does not publish.
UNRATED_TAG = "unrated"

#: Tag applied to a voice graded ``C+`` or above.
RECOMMENDED_TAG = "recommended"

#: Grade slugs used for the ``grade-*`` tags, so ``A`` and ``A-`` stay distinct.
GRADE_TAGS: dict[str, str] = {
    "A": "grade-a",
    "A-": "grade-a-minus",
    "B-": "grade-b-minus",
    "C+": "grade-c-plus",
    "C": "grade-c",
    "C-": "grade-c-minus",
    "D": "grade-d",
    "D-": "grade-d-minus",
    "F+": "grade-f-plus",
}

#: Grades the catalogue calls "candidates for customer-facing defaults".
CUSTOMER_FACING_GRADES: frozenset[str] = frozenset({"A", "A-", "B-", "C+"})

#: Licences accepted for a voice record.  The product direction is permissive
#: models only; adding a licence here is a deliberate decision, not a default.
PERMISSIVE_LICENSES: frozenset[str] = frozenset(
    {
        "Apache-2.0",
        "MIT",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "ISC",
        "0BSD",
    }
)

#: Fields every registry record must carry (see the ticket's scope item 4).
REQUIRED_VOICE_FIELDS: tuple[str, ...] = (
    "id",
    "name",
    "engine",
    "language",
    "gender",
    "tags",
    "quality",
    "license",
    "attribution",
    "style_spec",
    "supports",
    "enabled",
)

#: Style spec kinds defined by ADR-0002.
STYLE_SPEC_TYPES: frozenset[str] = frozenset({"base", "blend", "reference"})

#: Location of the registry file, resolved relative to the package so the service
#: works regardless of the process working directory.
DEFAULT_REGISTRY_PATH = Path(__file__).resolve().parents[1] / "data" / "voices.json"


class VoiceRegistryError(ValueError):
    """The registry file is missing, malformed or violates the schema."""


@dataclass(frozen=True)
class LanguageFamily:
    """Language metadata for one Kokoro id prefix."""

    language: str
    """Canonical BCP-47 tag exposed to clients (e.g. ``en-US``)."""

    locale: str
    """Lower-case language code accepted by ``kokoro_onnx`` (e.g. ``en-us``)."""

    name: str
    """English display name (e.g. ``english``)."""

    variant: str | None = None
    """Optional regional variant tag (e.g. ``american`` / ``british``)."""


#: Kokoro id prefix → language family.
LANGUAGE_FAMILIES: dict[str, LanguageFamily] = {
    "a": LanguageFamily("en-US", "en-us", "english", "american"),
    "b": LanguageFamily("en-GB", "en-gb", "english", "british"),
    "j": LanguageFamily("ja", "ja", "japanese"),
    "z": LanguageFamily("zh", "cmn", "chinese", "mandarin"),
    "e": LanguageFamily("es", "es", "spanish"),
    "f": LanguageFamily("fr-FR", "fr-fr", "french"),
    "h": LanguageFamily("hi", "hi", "hindi"),
    "i": LanguageFamily("it", "it", "italian"),
    "p": LanguageFamily("pt-BR", "pt-br", "portuguese"),
}

#: Kokoro id second character → gender.
GENDERS: dict[str, str] = {"f": "female", "m": "male"}


@dataclass(frozen=True)
class VoiceRecord:
    """One voice record, exactly as stored in the registry file."""

    id: str
    name: str
    engine: str
    language: str
    gender: str
    tags: tuple[str, ...]
    quality: str | None
    license: str
    attribution: str
    style_spec: dict[str, Any]
    supports: dict[str, bool]
    enabled: bool
    locale: str = DEFAULT_LOCALE
    raw: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def descriptor(self, engine: str | None = None) -> VoiceDescriptor:
        """Return the public descriptor shape for this record.

        ``license`` and ``attribution`` travel with every descriptor so a caller
        can verify the permissive-licence requirement (ADR-0001) from
        ``GET /api/tts/voices`` without a second lookup.
        """
        return VoiceDescriptor(
            id=self.id,
            name=self.name,
            language=self.language,
            locale=self.locale,
            gender=self.gender,
            engine=engine or self.engine,
            quality=self.quality,
            tags=list(self.tags),
            license=self.license,
            attribution=self.attribution,
        )

    @property
    def style_type(self) -> str:
        """The style-spec kind: ``base``, ``blend`` or ``reference``."""
        return str(self.style_spec.get("type", "base"))

    @property
    def engine_voice_id(self) -> str:
        """The key the engine itself uses for this voice.

        For a ``base`` record this is the ``style_spec.voice`` value; for any
        other style kind the record's own id, because resolving it is the
        adapter's (future) job.
        """
        if self.style_type == "base":
            return str(self.style_spec.get("voice") or self.id)
        return self.id


@dataclass(frozen=True)
class VoiceRegistry:
    """An immutable view over the voice registry file."""

    voices: dict[str, VoiceRecord]
    default_voice: str
    license: str = "Apache-2.0"
    attribution: str = ""
    schema_version: int = 1

    def get(self, voice_id: str) -> VoiceRecord | None:
        """Return one record, or ``None`` when the id is not registered."""
        return self.voices.get(voice_id)

    def ids(self, *, engine: str | None = None, enabled_only: bool = False) -> tuple[str, ...]:
        """Every registered id, optionally filtered by engine / enabled flag."""
        return tuple(
            record.id
            for record in self.voices.values()
            if (engine is None or record.engine == engine)
            and (record.enabled or not enabled_only)
        )

    def for_engine(self, engine: str, *, enabled_only: bool = True) -> list[VoiceRecord]:
        """Every record belonging to one engine, in registry order."""
        return [
            record
            for record in self.voices.values()
            if record.engine == engine and (record.enabled or not enabled_only)
        ]

    def descriptors(
        self, voice_ids: Iterable[str] | None = None
    ) -> list[VoiceDescriptor]:
        """Descriptors for the given ids (default: every enabled record)."""
        ids = list(voice_ids) if voice_ids is not None else list(self.ids(enabled_only=True))
        result: list[VoiceDescriptor] = []
        for voice_id in ids:
            record = self.get(voice_id)
            if record is not None:
                result.append(record.descriptor())
        return result


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VoiceRegistryError(message)


def _parse_record(entry: dict[str, Any], index: int) -> VoiceRecord:
    missing = [name for name in REQUIRED_VOICE_FIELDS if name not in entry]
    _require(not missing, f"voice record #{index} is missing required fields: {missing}")

    voice_id = entry["id"]
    _require(isinstance(voice_id, str) and bool(voice_id), f"voice record #{index} has no id")

    license_name = str(entry["license"])
    _require(
        license_name in PERMISSIVE_LICENSES,
        f"voice '{voice_id}' declares licence '{license_name}', which is not a "
        f"permissive licence ({sorted(PERMISSIVE_LICENSES)})",
    )

    style_spec = entry["style_spec"]
    _require(isinstance(style_spec, dict), f"voice '{voice_id}' has a non-object style_spec")
    style_type = str(style_spec.get("type", "base"))
    _require(
        style_type in STYLE_SPEC_TYPES,
        f"voice '{voice_id}' has unknown style_spec type '{style_type}'",
    )
    if style_type == "base":
        _require(
            bool(style_spec.get("voice")),
            f"voice '{voice_id}' has a base style_spec without an engine voice id",
        )

    tags = entry["tags"]
    _require(isinstance(tags, list), f"voice '{voice_id}' tags must be a list")

    language = str(entry["language"])
    locale = str(entry.get("locale") or _locale_for_language(language))

    return VoiceRecord(
        id=voice_id,
        name=str(entry["name"]),
        engine=str(entry["engine"]),
        language=language,
        gender=str(entry["gender"]),
        tags=tuple(str(tag) for tag in tags),
        quality=entry["quality"],
        license=license_name,
        attribution=str(entry["attribution"]),
        style_spec=dict(style_spec),
        supports=dict(entry["supports"]),
        enabled=bool(entry["enabled"]),
        locale=locale,
        raw=dict(entry),
    )


def _locale_for_language(language: str) -> str:
    """Map a canonical language tag to the code espeak-ng/kokoro-onnx expects."""
    for family in LANGUAGE_FAMILIES.values():
        if family.language == language:
            return family.locale
    return DEFAULT_LOCALE


def _normalise_language_code(value: str) -> str:
    """Lower-case a language tag and make ``_`` and ``-`` equivalent."""
    return value.strip().lower().replace("_", "-")


def _family_for(value: str | None) -> LanguageFamily | None:
    """Resolve a language tag *or* an engine locale to its family, or ``None``.

    Exact matches win over prefix matches, and the registry order decides ties, so
    the function is deterministic without a second mapping table. The prefix pass
    is what makes ``zh-CN`` resolve to the family that carries ``locale="cmn"`` and
    ``en`` resolve to ``en-US`` (the first English family in the registry).
    """
    if not value:
        return None
    wanted = _normalise_language_code(value)
    if not wanted:
        return None

    families = list(LANGUAGE_FAMILIES.values())
    for family in families:
        if family.locale == wanted:
            return family
    for family in families:
        if family.language.lower() == wanted:
            return family

    base = wanted.split("-")[0]
    for family in families:
        if family.language.lower().split("-")[0] == base:
            return family
    for family in families:
        if family.locale.split("-")[0] == base:
            return family
    return None


def engine_locale(language: str | None, *, default: str = DEFAULT_LOCALE) -> str:
    """Return the engine locale for a caller-supplied language tag.

    This is the single place a language tag (``zh``, ``zh-CN``, ``pt``, ``fr-FR``)
    becomes the code the engine and its phonemizer expect (``cmn``, ``pt-br``,
    ``fr-fr``). The mapping is *read from* :data:`LANGUAGE_FAMILIES`; there is no
    parallel table to keep in sync. An unknown tag is lower-cased and passed
    through so the phonemizer, not this function, reports that it cannot handle it.
    """
    if not language or not language.strip():
        return default
    family = _family_for(language)
    if family is not None:
        return family.locale
    return _normalise_language_code(language)


def canonical_language(locale: str | None) -> str | None:
    """Return the canonical language tag for an engine locale, or ``None``.

    ``cmn`` -> ``zh``; ``en-gb`` -> ``en-GB``. Used when reporting support by
    language instead of by engine code.
    """
    family = _family_for(locale)
    return family.language if family is not None else None


def all_canonical_languages() -> list[str]:
    """Every canonical language tag the registry knows, deduplicated and sorted."""
    return sorted({family.language for family in LANGUAGE_FAMILIES.values()})


def load_voice_registry(path: str | Path | None = None) -> VoiceRegistry:
    """Load and validate the registry file.

    Raises :class:`VoiceRegistryError` for a missing file, invalid JSON, missing
    required fields, duplicate ids or a non-permissive licence.  Failing loudly at
    startup is intentional: a half-loaded registry would silently hide voices.
    """
    registry_path = Path(path) if path is not None else DEFAULT_REGISTRY_PATH
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise VoiceRegistryError(f"voice registry not found: {registry_path}") from exc
    except json.JSONDecodeError as exc:
        raise VoiceRegistryError(f"voice registry is not valid JSON: {exc}") from exc

    _require(isinstance(payload, dict), "voice registry must be a JSON object")
    entries = payload.get("voices")
    _require(isinstance(entries, list) and bool(entries), "voice registry has no 'voices' list")

    voices: dict[str, VoiceRecord] = {}
    for index, entry in enumerate(entries):
        _require(isinstance(entry, dict), f"voice record #{index} is not an object")
        record = _parse_record(entry, index)
        _require(record.id not in voices, f"duplicate voice id '{record.id}' in the registry")
        voices[record.id] = record

    default_voice = str(payload.get("default_voice") or next(iter(voices)))
    _require(
        default_voice in voices,
        f"default_voice '{default_voice}' is not present in the registry",
    )

    return VoiceRegistry(
        voices=voices,
        default_voice=default_voice,
        license=str(payload.get("license", "Apache-2.0")),
        attribution=str(payload.get("attribution", "")),
        schema_version=int(payload.get("schema_version", 1)),
    )


#: The process-wide registry.  Reading it is cheap; writing it is a later phase.
VOICE_REGISTRY = load_voice_registry()

#: Every voice id registered for **any** engine, in registry order.
ALL_REGISTERED_VOICE_IDS: tuple[str, ...] = VOICE_REGISTRY.ids()

#: Voice ids of the shipped Kokoro pack.  Phase 0-4 code and tests use this as
#: "the pack"; optional engines (Chatterbox) live in the same registry file but
#: are not pack voices, so they are excluded here deliberately.
SHIPPED_VOICE_IDS: tuple[str, ...] = VOICE_REGISTRY.ids(engine=KOKORO_ENGINE)

#: Published quality grades, derived from the Kokoro pack (never hand-maintained).
QUALITY_GRADES: dict[str, str] = {
    record.id: record.quality
    for record in VOICE_REGISTRY.for_engine(KOKORO_ENGINE, enabled_only=False)
    if record.quality is not None
}

#: Kokoro voices the catalogue of record does not grade; ``quality`` is ``None``.
UNGRADED_VOICE_IDS: tuple[str, ...] = tuple(
    record.id
    for record in VOICE_REGISTRY.for_engine(KOKORO_ENGINE, enabled_only=False)
    if record.quality is None
)


#: Hook through which the designed-voice store contributes records.
#:
#: ``voice_registry`` and ``designed_voices`` used to import each other (each
#: function-locally, to dodge the load-time cycle), which is a cycle all the
#: same: any refactor that hoisted either import to module level would break
#: startup, and static analysis cannot verify the ordering.  The dependency now
#: runs one way only -- ``designed_voices`` registers a lookup callable here at
#: import time, and this module never mentions it by name.
#:
#: ``None`` simply means "no designed voices loaded", so a process that never
#: imports the store still resolves pack voices correctly.
_record_lookup: Callable[[str], "VoiceRecord | None"] | None = None


def set_record_lookup(lookup: Callable[[str], "VoiceRecord | None"] | None) -> None:
    """Register (or clear) the designed-voice lookup used by :func:`get_record`."""
    global _record_lookup
    _record_lookup = lookup


def get_record(voice_id: str) -> VoiceRecord | None:
    """Return the record for a voice id, or ``None``.

    The shipped pack registry is consulted first.  Designed voices (blends and
    base aliases) live in a separate, user-writable store; they resolve to the
    same :class:`VoiceRecord` shape so descriptors, filters and engine
    resolution never need to know which registry a voice came from.  The lookup
    is defensive: a broken designed-voice store must not hide pack voices.
    """
    record = VOICE_REGISTRY.get(voice_id)
    if record is not None:
        return record
    lookup = _record_lookup
    if lookup is None:
        return None
    try:
        return lookup(voice_id)
    except Exception:  # pragma: no cover - defensive: never break pack voices
        logger.warning(
            "Designed-voice registry lookup failed for %r; treating it as unknown.",
            voice_id,
            exc_info=True,
        )
        return None


def default_voice_for_engine(engine: str) -> str | None:
    """First enabled registered voice for one engine, or ``None``.

    Used when a request is routed to an engine other than the one that owns the
    requested voice (a fallback).  The engine's own first record is the closest
    stable substitute; the caller reports the substitution rather than hiding
    it.
    """
    for record in VOICE_REGISTRY.for_engine(engine, enabled_only=True):
        return record.id
    return None


def default_voice_id(available: Iterable[str] | None = None) -> str:
    """The effective default voice id for this deployment.

    ``KOKORO_DEFAULT_VOICE`` predates the registry and is still honoured: a
    deployment that configures a default must keep getting that voice, otherwise
    upgrading silently changes the audio it serves.

    The configured value is validated against the registry so a typo cannot
    silently switch the voice: an id that is unknown (and not exposed by the
    engine) or explicitly disabled in the registry falls back to the registry's
    own ``default_voice`` with a warning.  ``available`` is the voice ids the
    engine currently exposes, which lets a custom voice pack keep a configured
    default the registry does not describe yet.
    """
    configured = (getattr(settings, "kokoro_default_voice", "") or "").strip()
    if configured:
        # ``get_record`` also sees designed voices, so a preset can be configured
        # as the deployment default without editing the shipped pack registry.
        record = get_record(configured)
        if record is not None:
            if record.enabled:
                return record.id
            logger.warning(
                "KOKORO_DEFAULT_VOICE=%r is disabled in the voice registry; using %r instead.",
                configured,
                VOICE_REGISTRY.default_voice,
            )
            return VOICE_REGISTRY.default_voice
        if available is not None and configured in set(available):
            # A custom voice pack may ship ids the registry does not describe yet.
            logger.warning(
                "KOKORO_DEFAULT_VOICE=%r is not in the voice registry; using it because the "
                "engine exposes it.",
                configured,
            )
            return configured
        logger.warning(
            "KOKORO_DEFAULT_VOICE=%r is not an enabled voice in the registry; using %r instead. "
            "GET /api/tts/voices lists the valid ids.",
            configured,
            VOICE_REGISTRY.default_voice,
        )
    return VOICE_REGISTRY.default_voice


def _display_name(voice_id: str) -> str:
    """Derive a display name from an unregistered id suffix."""
    _, _, suffix = voice_id.partition("_")
    return suffix.replace("_", " ").title()


def describe_voice(voice_id: str, engine: str | None = None) -> VoiceDescriptor | None:
    """Return the descriptor for a registered voice, or ``None`` when unknown."""
    record = get_record(voice_id)
    if record is None:
        return None
    return record.descriptor(engine=engine)


def build_catalogue(
    voice_ids: Iterable[str], engine: str = KOKORO_ENGINE
) -> list[VoiceDescriptor]:
    """Enrich every id an engine exposes, sorted by language then id.

    Ids that are not in the registry are still returned (as ``und`` language,
    ``unknown`` gender) rather than silently hidden: the engine's voice pack is
    the source of truth for *which* voices exist, and the registry is the source
    of truth for *what they are*.
    """
    catalogue: list[VoiceDescriptor] = []
    for voice_id in voice_ids:
        descriptor = describe_voice(voice_id, engine=engine)
        if descriptor is None:
            logger.warning(
                "Voice id %r is not in the voice registry; exposing it with default metadata.",
                voice_id,
            )
            descriptor = VoiceDescriptor(
                id=voice_id,
                name=_display_name(voice_id) or voice_id,
                language="und",
                locale=DEFAULT_LOCALE,
                gender="unknown",
                engine=engine,
                quality=None,
                tags=["unknown"],
                license="unknown",
                attribution="",
            )
        catalogue.append(descriptor)
    return sorted(catalogue, key=lambda voice: (voice["language"], voice["id"]))


def filter_catalogue(
    voices: Iterable[VoiceDescriptor],
    *,
    engine: str | None = None,
    language: str | None = None,
    gender: str | None = None,
    tag: str | None = None,
) -> list[VoiceDescriptor]:
    """Filter a catalogue by engine, language, gender and tag.

    All comparisons are case-insensitive.  ``language`` matches either the
    canonical tag (``en-US``) or the engine locale code (``en-us``).
    """
    result = list(voices)
    if engine:
        wanted = engine.lower()
        result = [voice for voice in result if voice["engine"].lower() == wanted]
    if language:
        wanted_language = language.lower()
        result = [
            voice
            for voice in result
            if wanted_language in (voice["language"].lower(), voice["locale"].lower())
        ]
    if gender:
        wanted_gender = gender.lower()
        result = [voice for voice in result if voice["gender"].lower() == wanted_gender]
    if tag:
        wanted_tag = tag.lower()
        result = [
            voice for voice in result if any(t.lower() == wanted_tag for t in voice["tags"])
        ]
    return result


def catalogue_languages(voices: Iterable[VoiceDescriptor]) -> list[str]:
    """Return the distinct canonical language tags in a catalogue, sorted."""
    return sorted({voice["language"] for voice in voices})


def kokoro_capabilities(
    voices: Iterable[VoiceDescriptor],
    *,
    available: bool,
    default_voice: str,
) -> dict:
    """Legacy capability payload for the Kokoro engine.

    Kept for callers written against Phase 0; new code reads
    :meth:`app.engines.kokoro.KokoroEngine.capabilities` and
    :meth:`app.engines.kokoro.KokoroEngine.descriptor`.
    """
    voice_list = list(voices)
    return {
        "id": KOKORO_ENGINE,
        "engine": KOKORO_ENGINE,
        "name": "Kokoro-82M (ONNX)",
        "available": available,
        "license": "Apache-2.0",
        "languages": catalogue_languages(voice_list),
        "voice_count": len(voice_list),
        "default_voice": default_voice,
        "native_rate": True,
        "native_pitch": False,
        "native_volume": False,
        "styles": list(KOKORO_STYLES),
        "supports_cloning": False,
        "supports_streaming": True,
        "supports_ssml": False,
        "sample_rates": [KOKORO_SAMPLE_RATE],
        "max_chars": 10000,
        "gpu_required": False,
        "speed_range": [KOKORO_SPEED_MIN, KOKORO_SPEED_MAX],
    }
