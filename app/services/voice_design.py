"""Designed-voice service: validation, persistence, previews and deletion.

The HTTP layer owns transport only.  Everything that decides whether a recipe is
*usable* lives here, so the rules are testable without a request:

* 2-4 components,
* finite, non-negative weights with a positive sum (normalised to 1 internally),
* every component a base voice the engine currently exposes,
* all components in **one** language family — the phonemiser is chosen from the
  voice's language, so a cross-family blend degrades pronunciation,
* one dominant component carrying at least 60 % of the weight,
* unknown ids rejected with the valid ids named.

A designed voice stores the **recipe**, never the tensor.  The preview WAV is a
cache derived from the recipe: it is regenerated whenever the recipe hash
changes.

Blending changes **timbre**, not **emotion**.  This module never claims an
emotional effect; a voice picker that implied one would be a product lie.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.contracts import Direction
from app.services import designed_voices, voice_registry
from app.services.designed_voices import (
    DESIGNED_TAG,
    PREVIEW_TEXT,
    VoiceConflictError,
    VoiceRecipeError,
    normalised_components,
    raw_components,
    recipe_hash,
    resolve_preview_dir,
    validate_components,
)
from app.services.voice_registry import KOKORO_ENGINE, KOKORO_SAMPLE_RATE

logger = logging.getLogger(__name__)

#: Fields a designed voice exposes.  Mirrors the pack descriptor plus the
#: design metadata required by SPEECH-003 (components, weights, language, time).
DESIGN_SUPPORTS: dict[str, bool] = {
    "rate": True,
    "pitch": False,
    "style": False,
    "clone": False,
}

#: Attribution inherited from the only engine that can render these recipes.
DESIGN_LICENSE = "Apache-2.0"
DESIGN_ATTRIBUTION = "Kokoro-82M v1.0"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _slug(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    return slug[:48]


def _safe_preview_path(preview_dir: Path, name: str) -> Path | None:
    """Resolve ``name`` to a file directly inside ``preview_dir``, or ``None``.

    ``preview.file`` is read back from the persisted registry JSON, which lives
    on disk and can be edited outside this process.  Joining it onto the
    preview directory and calling ``read_bytes()`` or ``unlink()`` would let a
    corrupted or hand-edited registry name a file outside that directory.

    A preview is always a bare filename, so anything that is not one -- a path
    separator, an absolute path, a ``..`` segment -- is treated as absent
    rather than resolved.  The containment check afterwards is belt and braces
    for platforms where a bare name can still escape (e.g. a symlinked entry).
    """
    if not name or "\x00" in name:
        return None
    # A bare filename is its own basename. '../x' -> 'x', 'a/b' -> 'b',
    # '/etc/passwd' -> 'passwd' and '..' -> '', so any of those fail here.
    if name != Path(name).name:
        return None
    try:
        candidate = (preview_dir / name).resolve()
        candidate.relative_to(preview_dir.resolve())
    except (OSError, ValueError):
        return None
    return candidate


def _dedupe_tags(tags: list[str] | None) -> list[str]:
    seen: list[str] = []
    for tag in tags or []:
        cleaned = str(tag).strip().lower()
        if cleaned and cleaned not in seen:
            seen.append(cleaned)
    if DESIGNED_TAG not in seen:
        seen.append(DESIGNED_TAG)
    return seen


def _direction_payload(direction: Direction | None) -> dict[str, Any] | None:
    """Persist only the controls the caller actually expressed."""
    if direction is None:
        return None
    payload = direction.model_dump(exclude_unset=True, exclude_none=True)
    return payload or None


def default_direction(voice_id: str | None) -> Direction | None:
    """The default direction stored on a designed voice, if it has one.

    Used when a synthesis request names the voice but expresses no direction of
    its own.  Any failure to read the store is logged and ignored: a missing
    default must never break synthesis of a base voice.
    """
    if not voice_id:
        return None
    try:
        entry = designed_voices.get_store().get_entry(voice_id)
    except Exception:  # pragma: no cover - defensive
        logger.warning("Could not read the default direction for %r", voice_id, exc_info=True)
        return None
    if not entry:
        return None
    payload = entry.get("default_direction")
    if not payload:
        return None
    try:
        return Direction.model_validate(payload)
    except Exception:  # pragma: no cover - tolerant of hand-edited files
        logger.warning(
            "Designed voice %r has an invalid default_direction; ignoring it.", voice_id
        )
        return None


class VoiceDesignService:
    """Create, preview and delete designed voices for one TTS service."""

    def __init__(self, tts_service: Any) -> None:
        self._tts = tts_service

    # ------------------------------------------------------------------
    # Catalogue access
    # ------------------------------------------------------------------

    def base_voice_ids(self) -> list[str]:
        """Shipped pack voices available as ingredients (components / aliases).

        The loaded pack wins; before the model loads the shipped registry is the
        manifest, so designing a voice does not require a resident model.

        Only the **pack** registry is consulted.  A designed voice — a blend or a
        base alias — is never a valid ingredient: composing user-managed entries
        would make one recipe depend on another, and deleting the inner voice
        would silently break every recipe that names it.  Resolving ingredients
        through the pack alone keeps the derivation one level deep and means
        every accepted ingredient is guaranteed to render.
        """
        ids = getattr(self._tts, "voice_ids", None)
        if not isinstance(ids, list):
            ids = getattr(self._tts, "engine_voice_ids", None)
        if not isinstance(ids, list):
            ids = list(
                voice_registry.VOICE_REGISTRY.ids(engine=KOKORO_ENGINE, enabled_only=True)
            )
        pack = voice_registry.VOICE_REGISTRY
        return [
            voice_id
            for voice_id in ids
            if (record := pack.get(voice_id)) is not None
            and record.style_type == "base"
            and record.enabled
        ]

    def _base_record(self, voice_id: str, *, role: str = "Component") -> Any:
        """Resolve an ingredient to a **pack** record, or raise a 422 reason.

        ``role`` only shapes the message (``Component`` for a blend ingredient,
        ``Alias target`` for ``POST /voices``), so the caller always gets the
        rule and the valid ids rather than a bare lookup failure.
        """
        record = voice_registry.VOICE_REGISTRY.get(voice_id)
        if record is None:
            designed = designed_voices.get_store().get(voice_id)
            if designed is not None:
                raise VoiceRecipeError(
                    f"{role} '{voice_id}' is a designed voice "
                    f"({designed.style_type}), not a shipped base voice. "
                    "Ingredients must be base voices from the voice pack so the "
                    "derivation stays one level deep and cannot break when "
                    "another designed voice is deleted. "
                    f"Valid base voice ids: {', '.join(self.base_voice_ids())}."
                )
            raise VoiceRecipeError(
                f"Unknown {role.lower()} voice '{voice_id}'. "
                f"Valid base voice ids: {', '.join(self.base_voice_ids())}."
            )
        if record.style_type != "base":
            raise VoiceRecipeError(
                f"{role} '{voice_id}' is a shipped voice with style type "
                f"'{record.style_type}', not a base voice. "
                f"Valid base voice ids: {', '.join(self.base_voice_ids())}."
            )
        if record.engine != KOKORO_ENGINE:
            raise VoiceRecipeError(
                f"{role} '{voice_id}' belongs to engine '{record.engine}', not "
                f"'{KOKORO_ENGINE}'."
            )
        if not record.enabled:
            raise VoiceRecipeError(
                f"{role} '{voice_id}' is disabled on this deployment; "
                "GET /api/tts/voices lists the enabled base voices."
            )
        return record

    def validate_blend(
        self,
        components: list[dict[str, Any]],
        *,
        language: str | None = None,
    ) -> dict[str, Any]:
        """Validate a blend recipe and return its resolved design metadata.

        Raises :class:`VoiceRecipeError` (mapped to 422) with an actionable
        message for every rule in the ticket.
        """
        parsed = validate_components(components)

        # One authoritative lookup per ingredient.  It resolves against the pack
        # registry only, so a designed voice (blend *or* base alias) is rejected
        # with the actionable 422 instead of being persisted as a voice that can
        # never render.  See the ``_zero_weight`` note in the docs for the
        # deliberate acceptance of a 0.0-weight second component.
        records = [(voice, weight, self._base_record(voice)) for voice, weight in parsed]

        families = {record.language for _, _, record in records}
        if len(families) > 1:
            described = ", ".join(
                f"{voice} ({record.language})" for voice, _, record in records
            )
            raise VoiceRecipeError(
                f"Components span different language families: {described}. "
                "A blend must stay within one language family because the "
                "phonemiser is chosen from the voice's language, so a "
                "cross-family blend degrades pronunciation."
            )

        resolved_language = records[0][2].language
        if language is not None:
            wanted = language.strip().lower()
            accepted = {resolved_language.lower(), records[0][2].locale.lower()}
            if wanted not in accepted:
                raise VoiceRecipeError(
                    f"Language '{language}' does not match the components' "
                    f"language family '{resolved_language}'. Blend voices from "
                    f"'{language}' or omit the field."
                )

        dominant_voice, dominant_weight, dominant_record = max(
            records, key=lambda item: item[1]
        )
        total = sum(weight for _, weight, _ in records)
        return {
            "components": [
                {"voice": voice, "weight": weight} for voice, weight, _ in records
            ],
            "language": resolved_language,
            "locale": records[0][2].locale,
            "gender": dominant_record.gender,
            "dominant_voice": dominant_voice,
            "dominant_weight": dominant_weight / total,
        }

    def validate_base_alias(
        self, base_voice: str, *, language: str | None = None
    ) -> dict[str, Any]:
        """Validate an alias recipe (``style_spec.type == "base"``).

        The target must be a shipped pack voice, for the same reason blend
        components must be: an alias of a designed voice would advertise a
        second name for a recipe that can be deleted underneath it.
        """
        record = self._base_record(base_voice, role="Alias target")
        if language is not None:
            wanted = language.strip().lower()
            if wanted not in {record.language.lower(), record.locale.lower()}:
                raise VoiceRecipeError(
                    f"Language '{language}' does not match base voice "
                    f"'{base_voice}' ({record.language})."
                )
        return {
            "base_voice": base_voice,
            "language": record.language,
            "locale": record.locale,
            "gender": record.gender,
        }

    # ------------------------------------------------------------------
    # Creation
    # ------------------------------------------------------------------

    def _claim_id(self, name: str) -> str:
        voice_id = _slug(name)
        if not voice_id:
            raise VoiceRecipeError(
                "Voice name must contain at least one letter or digit so an id can be derived."
            )
        if (
            voice_registry.VOICE_REGISTRY.get(voice_id) is not None
            or designed_voices.get_store().get(voice_id) is not None
        ):
            raise VoiceConflictError(
                f"A voice with id '{voice_id}' already exists. "
                "Choose a different name; GET /api/tts/voices lists the current ids."
            )
        return voice_id

    def _entry(
        self,
        *,
        voice_id: str,
        name: str,
        style_spec: dict[str, Any],
        language: str,
        locale: str,
        gender: str,
        tags: list[str] | None,
        direction: Direction | None,
    ) -> dict[str, Any]:
        return {
            "id": voice_id,
            "name": name.strip(),
            "engine": KOKORO_ENGINE,
            "language": language,
            "locale": locale,
            "gender": gender,
            "tags": _dedupe_tags(tags),
            "quality": None,
            "license": DESIGN_LICENSE,
            "attribution": DESIGN_ATTRIBUTION,
            "style_spec": style_spec,
            "supports": dict(DESIGN_SUPPORTS),
            "enabled": True,
            "preset": False,
            "created_at": _utc_now(),
            "default_direction": _direction_payload(direction),
            "preview": None,
        }

    async def create_blend(
        self,
        *,
        name: str,
        components: list[dict[str, Any]],
        language: str | None = None,
        direction: Direction | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Validate, persist and preview a new blended voice."""
        resolved = self.validate_blend(components, language=language)
        voice_id = self._claim_id(name)
        entry = self._entry(
            voice_id=voice_id,
            name=name,
            style_spec={"type": "blend", "components": resolved["components"]},
            language=resolved["language"],
            locale=resolved["locale"],
            gender=resolved["gender"],
            tags=tags,
            direction=direction,
        )
        designed_voices.get_store().add(entry)
        logger.info(
            "Designed blend voice %r created from %r",
            voice_id,
            ", ".join(
                f"{component['voice']}@{component['weight']:g}"
                for component in resolved["components"]
            ),
        )
        await self._preview_best_effort(voice_id)
        return self.descriptor(voice_id) or self._descriptor_from_entry(entry)

    async def create_base_alias(
        self,
        *,
        name: str,
        base_voice: str,
        language: str | None = None,
        direction: Direction | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Validate, persist and preview a new base-voice alias."""
        resolved = self.validate_base_alias(base_voice, language=language)
        voice_id = self._claim_id(name)
        entry = self._entry(
            voice_id=voice_id,
            name=name,
            style_spec={"type": "base", "voice": resolved["base_voice"]},
            language=resolved["language"],
            locale=resolved["locale"],
            gender=resolved["gender"],
            tags=tags,
            direction=direction,
        )
        designed_voices.get_store().add(entry)
        logger.info("Designed base alias %r created for %r", voice_id, base_voice)
        await self._preview_best_effort(voice_id)
        return self.descriptor(voice_id) or self._descriptor_from_entry(entry)

    # ------------------------------------------------------------------
    # Descriptors
    # ------------------------------------------------------------------

    def descriptor(self, voice_id: str) -> dict[str, Any] | None:
        entry = designed_voices.get_store().get_entry(voice_id)
        if entry is None:
            return None
        return self._descriptor_from_entry(entry)

    @staticmethod
    def _descriptor_from_entry(entry: dict[str, Any]) -> dict[str, Any]:
        style_spec = entry["style_spec"]
        style_type = str(style_spec.get("type", "base"))
        components: list[dict[str, Any]] = []
        normalised_weights: dict[str, float] = {}
        if style_type == "blend":
            raw = raw_components(style_spec)
            total = sum(weight for _, weight in raw) or 1.0
            normalised = {
                component["voice"]: component["weight"]
                for component in normalised_components(style_spec)
            }
            normalised_weights = dict(sorted(normalised.items()))
            components = [
                {
                    "voice": voice,
                    "weight": weight,
                    "normalised_weight": round(weight / total, 9),
                }
                for voice, weight in raw
            ]

        return {
            "id": entry["id"],
            "name": entry["name"],
            "engine": entry["engine"],
            "language": entry["language"],
            "locale": entry["locale"],
            "gender": entry["gender"],
            "quality": entry.get("quality"),
            "tags": list(entry.get("tags", [])),
            "style_type": style_type,
            "components": components,
            "normalised_weights": normalised_weights,
            "resolved_language": entry["language"],
            "created_at": entry.get("created_at"),
            "preview_url": f"/api/tts/voices/{entry['id']}/preview",
            "default_direction": entry.get("default_direction"),
            "preset": bool(entry.get("preset")),
            "style_spec": style_spec,
        }

    # ------------------------------------------------------------------
    # Previews
    # ------------------------------------------------------------------

    async def _preview_best_effort(self, voice_id: str) -> None:
        """Warm the preview cache; never fail voice creation over it."""
        try:
            await self.ensure_preview(voice_id)
        except Exception as exc:  # noqa: BLE001 - creation must still succeed
            logger.warning(
                "Preview for designed voice %r will be generated on first request: %r",
                voice_id,
                exc,
            )

    async def ensure_preview(self, voice_id: str, *, force: bool = False) -> dict[str, Any]:
        """Return the cached preview, synthesising only when the recipe changed.

        Returns ``{"audio": bytes, "cache": "hit"|"miss", "sample_rate": int,
        "duration": float, "recipe_hash": str}``.
        """
        store = designed_voices.get_store()
        entry = store.get_entry(voice_id)
        if entry is None:
            raise KeyError(voice_id)

        current_hash = recipe_hash(entry["style_spec"])
        preview = entry.get("preview") or {}
        preview_dir = resolve_preview_dir(store.path)

        if not force and preview.get("recipe_hash") == current_hash:
            cached_path = _safe_preview_path(preview_dir, str(preview.get("file") or ""))
            if cached_path is not None:
                try:
                    audio = cached_path.read_bytes()
                except OSError:
                    audio = b""
                if audio:
                    return {
                        "audio": audio,
                        "cache": "hit",
                        "sample_rate": int(preview.get("sample_rate", KOKORO_SAMPLE_RATE)),
                        "duration": float(preview.get("duration", 0.0)),
                        "recipe_hash": current_hash,
                    }

        outcome = await self._tts.synthesize_async(
            PREVIEW_TEXT,
            voice=voice_id,
            cache=False,
        )
        duration = float(getattr(outcome, "duration", 0.0))
        audio = bytes(outcome.audio)

        file_name = f"{voice_id}-{current_hash[:16]}.wav"
        preview_dir.mkdir(parents=True, exist_ok=True)
        target = preview_dir / file_name
        temporary = target.with_name(f".{target.name}.tmp")
        temporary.write_bytes(audio)
        os.replace(temporary, target)

        # Drop a stale render for an older recipe; otherwise every edit would
        # leave an orphan WAV behind until the voice is deleted.
        stale = preview.get("file")
        if stale and stale != file_name:
            stale_path = _safe_preview_path(preview_dir, str(stale))
            if stale_path is not None:
                try:
                    stale_path.unlink(missing_ok=True)
                except OSError:  # pragma: no cover - best effort cleanup
                    logger.warning("Could not remove stale preview %r", stale)

        store.update(
            voice_id,
            {
                "preview": {
                    "recipe_hash": current_hash,
                    "file": file_name,
                    "text": PREVIEW_TEXT,
                    "sample_rate": KOKORO_SAMPLE_RATE,
                    "duration": round(duration, 3),
                    "created_at": _utc_now(),
                }
            },
        )
        return {
            "audio": audio,
            "cache": "miss",
            "sample_rate": KOKORO_SAMPLE_RATE,
            "duration": duration,
            "recipe_hash": current_hash,
        }

    # ------------------------------------------------------------------
    # Deletion
    # ------------------------------------------------------------------

    def delete(self, voice_id: str) -> dict[str, Any]:
        """Delete a custom designed voice.

        Pack voices and shipped presets are immutable: returning 200 for either
        would let a caller silently remove shipped product data, so they raise
        :class:`VoiceConflictError` (409) instead.
        """
        if voice_registry.VOICE_REGISTRY.get(voice_id) is not None:
            raise VoiceConflictError(
                f"Voice '{voice_id}' is a shipped pack voice and is immutable. "
                "Only designed voices can be deleted."
            )

        store = designed_voices.get_store()
        entry = store.get_entry(voice_id)
        if entry is None:
            raise KeyError(voice_id)
        if entry.get("preset"):
            raise VoiceConflictError(
                f"Voice '{voice_id}' is a shipped preset and is immutable. "
                "Design a new voice if you need a different recipe."
            )

        removed = store.delete(voice_id)
        self._remove_previews(store.path, voice_id)
        logger.info("Designed voice %r deleted", voice_id)
        return removed

    @staticmethod
    def _remove_previews(store_path: Path, voice_id: str) -> None:
        preview_dir = resolve_preview_dir(store_path)
        # A glob pattern is a path expression too: an id containing a separator
        # or '..' would widen it beyond this directory.  voice_id is slugged on
        # creation, but this also runs for ids read back from the registry file.
        safe_id = _slug(voice_id)
        if not safe_id:
            return
        root = preview_dir.resolve()
        for candidate in preview_dir.glob(f"{safe_id}-*.wav"):
            try:
                if candidate.resolve().parent != root:
                    continue
                candidate.unlink()
            except OSError:  # pragma: no cover - best effort cleanup
                logger.warning("Could not remove preview for %r", voice_id)
