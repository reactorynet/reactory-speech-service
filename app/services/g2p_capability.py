"""Runtime capability probe for the grapheme-to-phoneme (G2P) backend.

The voice registry says which languages the *acoustic model* can speak. It cannot
say which languages the *text front end* can phonemise: that depends on the
espeak-ng build in this image and on the script of the text. On this repository
the two disagree — the Kokoro pack ships five Japanese voices, but espeak-ng
cannot read Japanese kanji and spells it out as "chinese letter" instead.

Advertising a language the pipeline cannot phonemise is how SPEECH-010 happened:
the catalogue said Japanese worked, the phonemizer produced 16 seconds of
"english japanese letter" for a 21-character sentence, and every layer reported
success because the WAV was valid.

This module closes that gap by **measuring** the backend instead of trusting the
catalogue:

* one representative sentence per language, phonemised through the real converter;
* a language is supported only when that produces phonemes with no spelling
  fallback (espeak emits a language-switch marker pair per character when it has
  to spell a script out);
* the result is cached for the process and reported by ``GET /api/tts/capabilities``
  and ``/health``, and it gates synthesis and ``/api/tts/phonemize``.

The probe is deliberately cheap (milliseconds, once per process) and has no model
dependency: it exercises exactly the component that was broken.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

from app.services.phonetic_converter import (
    PhoneticConversionError,
    PhoneticConverter,
    UnsupportedLanguageError,
    ensure_text_is_readable,
    marks_spelling_fallback,
    marks_spelling_output,
    strip_language_markers,
)

from app.services.voice_registry import (
    LANGUAGE_FAMILIES,
    canonical_language,
    engine_locale,
)

logger = logging.getLogger(__name__)

#: Representative sentence per canonical language tag.
#:
#: Keyed by the tag :class:`app.services.voice_registry.LanguageFamily` exposes,
#: so a voice's language and its probe are the same identifier. The sentences are
#: ordinary platform phrasing, not tongue twisters: the probe asks "can this
#: backend read a normal sentence of the language", which is the question the
#: catalogue was answering wrongly. The Japanese and Chinese samples deliberately
#: use the scripts real text uses (kanji and Han characters), because that is
#: where the backend fails.
PROBE_SENTENCES: dict[str, str] = {
    "en-US": "The platform turns a ticket into a reviewable change.",
    "en-GB": "The platform turns a ticket into a reviewable change.",
    "es": "La plataforma convierte un ticket en un cambio revisable.",
    "pt-BR": "A plataforma converte um ticket em uma mudança revisável.",
    "fr-FR": "La plateforme transforme un ticket en une modification vérifiable.",
    "it": "La piattaforma trasforma un ticket in una modifica verificabile.",
    "hi": "नमस्ते, मैं आपकी मदद कर सकता हूँ।",
    "zh": "你好，今天我能帮你做什么？",
    "ja": "リアクトリーの音声サービスはとても便利です。",
}


@dataclass(frozen=True)
class LanguageSupport:
    """Measured G2P support for one language."""

    language: str
    """Canonical language tag (e.g. ``zh``)."""

    locale: str
    """Engine locale the probe used (e.g. ``cmn``)."""

    supported: bool
    """Whether the backend produced usable phonemes for the probe sentence."""

    reason: str | None = None
    """Why the language is unsupported, or ``None`` when it is supported."""

    marker_pairs: int = 0
    """Language-switch marker pairs espeak emitted for the probe sentence."""

    def as_dict(self) -> dict:
        return {
            "language": self.language,
            "locale": self.locale,
            "supported": self.supported,
            "reason": self.reason,
            "marker_pairs": self.marker_pairs,
        }


class G2PCapability:
    """Probes and caches which languages the phonemizer can actually handle."""

    def __init__(self, converter: PhoneticConverter | None = None) -> None:
        self._converter = converter
        self._results: dict[str, LanguageSupport] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Probing
    # ------------------------------------------------------------------

    def _backend(self) -> PhoneticConverter:
        if self._converter is not None:
            return self._converter
        # Imported lazily so this module never participates in an import cycle
        # with the converter singleton it is measuring.
        from app.services.phonetic_converter import phonetic_converter

        return phonetic_converter

    def probe(self, language: str) -> LanguageSupport:
        """Measure one canonical language, caching the verdict for the process.

        The lock is held across the measurement (about 1.5 s: most of it is
        espeak-ng loading that language's dictionary) so two concurrent first
        callers cannot both pay for, and race on, the same backend.
        """
        wanted = canonical_language(language) or language
        with self._lock:
            cached = self._results.get(wanted)
            if cached is not None:
                return cached
            result = self._probe_uncached(wanted)
            self._results[wanted] = result
            return result

    def _probe_uncached(self, language: str) -> LanguageSupport:
        locale = engine_locale(language)
        sample = PROBE_SENTENCES.get(language)
        if sample is None:
            # A language the registry knows but this probe has no sentence for is
            # *unverified*, not silently blessed. Say so.
            return LanguageSupport(
                language=language,
                locale=locale,
                supported=False,
                reason="no G2P probe sentence is defined for this language",
            )

        converter = self._backend()
        if not converter.is_available:
            return LanguageSupport(
                language=language,
                locale=locale,
                supported=False,
                reason="the espeak-ng/phonemizer backend is not available",
            )

        # The probe must answer the same question the render path asks, so it
        # applies the deterministic script guard too: a language whose own
        # orthography cannot be read by its G2P is not "supported".
        try:
            ensure_text_is_readable(sample, locale)
        except UnsupportedLanguageError as exc:
            return LanguageSupport(
                language=language,
                locale=locale,
                supported=False,
                reason=str(exc.reason),
            )

        try:
            raw = converter.phonemize_raw(sample, language=locale)
        except UnsupportedLanguageError as exc:
            return LanguageSupport(
                language=language,
                locale=locale,
                supported=False,
                reason=str(exc.reason),
            )
        except PhoneticConversionError as exc:
            return LanguageSupport(
                language=language,
                locale=locale,
                supported=False,
                reason=f"the backend raised: {exc}",
            )
        clean, markers = strip_language_markers(raw)
        pairs = len(markers) // 2
        if not clean:
            return LanguageSupport(
                language=language,
                locale=locale,
                supported=False,
                reason="the backend produced no phonemes for the probe sentence",
                marker_pairs=pairs,
            )
        if marks_spelling_fallback(sample, pairs):
            return LanguageSupport(
                language=language,
                locale=locale,
                supported=False,
                reason=(
                    "espeak-ng cannot read this script: it spells the characters "
                    f"out instead ({pairs} language switches for the probe sentence, "
                    f"e.g. {markers[:4]})"
                ),
                marker_pairs=pairs,
            )
        if marks_spelling_output(sample, clean):
            return LanguageSupport(
                language=language,
                locale=locale,
                supported=False,
                reason=(
                    "espeak-ng spelt the probe sentence out one syllable per "
                    f"character (e.g. {' '.join(clean.split()[:3])})"
                ),
                marker_pairs=pairs,
            )
        return LanguageSupport(
            language=language,
            locale=locale,
            supported=True,
            marker_pairs=pairs,
        )

    def refresh(self) -> dict[str, LanguageSupport]:
        """Drop the cache and re-probe every language the registry knows."""
        with self._lock:
            self._results.clear()
        return self.report()

    def report(self) -> dict[str, LanguageSupport]:
        """Probe every canonical language known to the voice registry."""
        languages = sorted({family.language for family in LANGUAGE_FAMILIES.values()})
        return {language: self.probe(language) for language in languages}

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def support_for(self, value: str | None) -> LanguageSupport | None:
        """Resolve a language tag or engine locale to its measured support."""
        if not value:
            return None
        language = canonical_language(value)
        if language is None:
            return None
        return self.probe(language)

    def is_supported(self, value: str | None) -> bool:
        """Whether a language/locale is supported.

        An unknown code is treated as supported here and left to the backend to
        reject: refusing a language this registry does not describe would break a
        custom voice pack, while the converter still guards the output.
        """
        support = self.support_for(value)
        return True if support is None else support.supported

    def ensure_supported(self, value: str | None) -> None:
        """Raise :class:`UnsupportedLanguageError` when the backend cannot read it."""
        support = self.support_for(value)
        if support is not None and not support.supported:
            raise UnsupportedLanguageError(
                support.locale,
                support.reason or "the phonemizer cannot handle this language",
            )

    def snapshot(self) -> dict:
        """Non-blocking view of what has been probed so far.

        Safe for a health endpoint: it never pays the ~1.5 s per-language backend
        load, and it names the languages still pending rather than omitting them
        (an omitted language would look supported).
        """
        known = sorted({family.language for family in LANGUAGE_FAMILIES.values()})
        with self._lock:
            rows = dict(self._results)
        pending = [language for language in known if language not in rows]
        return {
            "supported": [language for language, row in rows.items() if row.supported],
            "unsupported": [row.as_dict() for row in rows.values() if not row.supported],
            "pending": pending,
            "languages": {language: row.as_dict() for language, row in rows.items()},
        }

    def supported_languages(self) -> list[str]:
        """Canonical tags the pipeline can phonemise on this deployment."""
        return [language for language, row in self.report().items() if row.supported]

    def unsupported(self) -> list[LanguageSupport]:
        """Rows for languages the pipeline cannot phonemise."""
        return [row for row in self.report().values() if not row.supported]

    # ------------------------------------------------------------------
    # Startup signal
    # ------------------------------------------------------------------

    def log_report(self) -> dict[str, LanguageSupport]:
        """Probe and log the support matrix so a bad image is visible at boot."""
        report = self.report()
        supported = [language for language, row in report.items() if row.supported]
        unsupported = [row for row in report.values() if not row.supported]
        logger.info(
            "G2P languages supported (%d): %s", len(supported), ", ".join(supported) or "none"
        )
        for row in unsupported:
            logger.warning(
                "G2P language UNAVAILABLE: %s (locale %s) — %s",
                row.language,
                row.locale,
                row.reason,
            )
        return report


#: Process-wide probe cache. Cheap to read; the first read runs the probe.
g2p_capability = G2PCapability()


__all__ = [
    "G2PCapability",
    "LanguageSupport",
    "PROBE_SENTENCES",
    "g2p_capability",
]
