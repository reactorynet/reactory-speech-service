"""Voice catalogue for the Kokoro voice pack.

The shipped voice pack (``models/voices-v1.0.bin``) is the single source of truth
for *which* voices exist.  :meth:`app.services.tts_service.TTSService.get_voices`
enumerates the keys exposed by the loaded ``kokoro_onnx.Kokoro`` instance and
passes them to :func:`build_catalogue`; this module never acts as an allow-list.
It only *enriches* ids with presentation metadata.

Metadata is derived from the voice id wherever possible.  Kokoro ids follow the
scheme ``<language><gender>_<name>``:

* first character  → language family (``a`` → ``en-US``, ``b`` → ``en-GB``,
  ``j`` → ``ja``, ``z`` → ``zh``, ``e`` → ``es``, ``f`` → ``fr-FR``,
  ``h`` → ``hi``, ``i`` → ``it``, ``p`` → ``pt-BR``)
* second character → gender (``f`` → female, ``m`` → male)
* the remainder    → display name (``af_heart`` → ``Heart``)

``locale`` is the code handed to the engine for phonemization.  It is the
lower-case form of the language tag, except for Mandarin Chinese: espeak-ng (the
G2P backend used by kokoro-onnx) only accepts the ``cmn`` alias for it and fails
on ``zh``.

Quality grades are the published Kokoro-82M voice grades (documented in
``docs/voices/kokoro-voice-catalogue.md``) and are kept in a static table keyed
by voice id.  A voice that is not graded there inherits :data:`DEFAULT_QUALITY`;
it is still exposed, never dropped.
"""

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TypedDict

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

#: Grade assigned to a voice that has no published quality grade.
DEFAULT_QUALITY = "C"


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

#: Published Kokoro-82M quality grades, keyed by voice id.
QUALITY_GRADES: dict[str, str] = {
    # American English
    "af_alloy": "C",
    "af_aoede": "C+",
    "af_bella": "A-",
    "af_heart": "A",
    "af_jessica": "D",
    "af_kore": "C+",
    "af_nicole": "B-",
    "af_nova": "C",
    "af_river": "D",
    "af_sarah": "C+",
    "af_sky": "C-",
    "am_adam": "F+",
    "am_echo": "D",
    "am_eric": "D",
    "am_fenrir": "C+",
    "am_liam": "D",
    "am_michael": "C+",
    "am_onyx": "D",
    "am_puck": "C+",
    "am_santa": "D-",
    # British English
    "bf_alice": "D",
    "bf_emma": "B-",
    "bf_isabella": "C",
    "bf_lily": "D",
    "bm_daniel": "D",
    "bm_fable": "C",
    "bm_george": "C",
    "bm_lewis": "D",
    # Spanish
    "ef_dora": "C",
    "em_alex": "C",
    "em_santa": "C+",
    # French
    "ff_siwis": "C",
    # Hindi
    "hf_alpha": "C",
    "hf_beta": "C",
    "hm_omega": "C",
    "hm_psi": "C",
    # Italian
    "if_sara": "C",
    "im_nicola": "C",
    # Japanese
    "jf_alpha": "C+",
    "jf_gongitsune": "C",
    "jf_nezumi": "C",
    "jf_tebukuro": "C",
    "jm_kumo": "C",
    # Brazilian Portuguese
    "pf_dora": "C",
    "pm_alex": "C",
    "pm_santa": "C+",
    # Mandarin Chinese
    "zf_xiaobei": "C",
    "zf_xiaoni": "C",
    "zf_xiaoxiao": "C",
    "zf_xiaoyi": "C",
    "zm_yunjian": "C",
    "zm_yunxi": "C",
    "zm_yunxia": "C",
    "zm_yunyang": "C",
}

#: Voice ids shipped in the v1.0 voice pack.
#:
#: Used for voice discovery before the engine is loaded.  As soon as an engine is
#: available its own ``voices`` keys take precedence — this tuple is never used to
#: reject a voice.
SHIPPED_VOICE_IDS: tuple[str, ...] = tuple(sorted(QUALITY_GRADES))


class VoiceDescriptor(TypedDict):
    """Enriched description of a single voice."""

    id: str
    name: str
    language: str
    locale: str
    gender: str
    engine: str
    quality: str
    tags: list[str]


def _display_name(voice_id: str) -> str:
    """Derive the display name from the id suffix (``af_heart`` → ``Heart``)."""
    _, _, suffix = voice_id.partition("_")
    return suffix.replace("_", " ").title()


def _build_tags(family: LanguageFamily, gender: str, quality: str) -> list[str]:
    """Derive the filterable tag list for a voice."""
    tags = [gender, family.name]
    if family.variant:
        tags.append(family.variant)
    tags.append(f"grade-{quality[0].lower()}")
    if quality[0].upper() == "A":
        tags.append("recommended")
    return tags


def describe_voice(voice_id: str, engine: str = KOKORO_ENGINE) -> VoiceDescriptor | None:
    """Enrich a Kokoro voice id with presentation metadata.

    Returns ``None`` when the id does not follow the Kokoro scheme, so callers can
    distinguish "unknown id" from "known id with default metadata".
    """
    if len(voice_id) < 3 or "_" not in voice_id[2:]:
        return None

    family = LANGUAGE_FAMILIES.get(voice_id[0])
    gender = GENDERS.get(voice_id[1])
    if family is None or gender is None:
        return None

    name = _display_name(voice_id)
    if not name:
        return None

    quality = QUALITY_GRADES.get(voice_id, DEFAULT_QUALITY)
    return VoiceDescriptor(
        id=voice_id,
        name=name,
        language=family.language,
        locale=family.locale,
        gender=gender,
        engine=engine,
        quality=quality,
        tags=_build_tags(family, gender, quality),
    )


def build_catalogue(
    voice_ids: Iterable[str], engine: str = KOKORO_ENGINE
) -> list[VoiceDescriptor]:
    """Enrich every voice id an engine exposes, sorted by language then id.

    Ids that do not follow the Kokoro scheme are still returned (as ``und``
    language) rather than silently hidden, mirroring the fact that the pack is
    the source of truth for which voices exist.
    """
    catalogue: list[VoiceDescriptor] = []
    for voice_id in voice_ids:
        descriptor = describe_voice(voice_id, engine=engine)
        if descriptor is None:
            logger.warning(
                "Voice id %r does not follow the Kokoro id scheme; exposing it with "
                "default metadata.",
                voice_id,
            )
            descriptor = VoiceDescriptor(
                id=voice_id,
                name=_display_name(voice_id) or voice_id,
                language="und",
                locale=DEFAULT_LOCALE,
                gender="unknown",
                engine=engine,
                quality=DEFAULT_QUALITY,
                tags=["unknown"],
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
    """Describe what the Kokoro engine can and cannot do.

    Kokoro is a CPU-first ONNX model with native rate (speed) control but no
    native pitch, volume, style or cloning control.
    """
    voice_list = list(voices)
    return {
        "id": KOKORO_ENGINE,
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
        "sample_rates": [KOKORO_SAMPLE_RATE],
        "gpu_required": False,
        "speed_range": [KOKORO_SPEED_MIN, KOKORO_SPEED_MAX],
    }
