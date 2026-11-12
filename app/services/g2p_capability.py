"""Runtime capability and quality signal for the speech text front end.

Two different questions, two different answers
----------------------------------------------

The voice registry says which languages the *acoustic model* can speak. It cannot
say which languages the *text front end* can phonemise, and it says nothing about
how well a language round-trips. On this repository both gaps are real:

* the Kokoro pack ships five Japanese voices, but espeak-ng cannot read Japanese
  kanji and spells it out as "japanese letter" instead;
* Mandarin phonemises correctly and still scores CER 0.92 on a normal-length
  narration sentence with the local Whisper model.

Advertising either of those as a clean "yes" is how SPEECH-010 happened, and it is
why this module reports **two** measured facts per language rather than a boolean:

``supported``
    Whether the backend produces usable phonemes for a representative sentence of
    the language (no language-switch spelling fallback, no script it cannot read).

``quality``
    ``production`` when every entry the language has in the committed golden corpus
    round-tripped within its target, ``caveats`` when at least one did not (with
    the failing entries named), ``unsupported`` when the first check failed.

The numbers behind ``quality`` are *not* maintained here. They are measured by
``scripts/record_language_quality.py`` against ``app/eval/data/golden_corpus.json``
and committed to ``app/eval/data/language-quality.json``; this module only reads
that record. A language with no record is reported ``unverified`` rather than
assumed good, so a missing measurement cannot become an advertisement.

Cost, measured
--------------

The support probe loads one espeak-ng dictionary per language: **about 13.5 s for
all nine on this image, once per process** (measured: first
``g2p_capability.report()`` 13.45 s, second 0.000 s). That is far too much to pay
on a request path, so:

* ``report()`` is the blocking measurement — startup and tests use it;
* ``report(block=False)`` and :meth:`G2PCapability.snapshot` never probe; they
  return what has been measured so far and name the rest as ``pending``;
* ``GET /api/tts/capabilities`` and ``/health`` read the non-blocking view, so no
  request ever pays a dictionary load (review finding F3).
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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

#: Committed measurement of round-trip quality per language. Written by
#: ``scripts/record_language_quality.py``; read-only here.
QUALITY_RECORD_PATH = (
    Path(__file__).resolve().parent.parent / "eval" / "data" / "language-quality.json"
)

#: Quality bands. There is deliberately no "close enough" band: the ticket's point
#: is that a measured number either met its corpus target or it did not.
QUALITY_PRODUCTION = "production"

#: Below this many measured sentences, a passing language is reported with an
#: explicit "small sample" caveat. Portuguese, Italian and Hindi have a single
#: corpus sentence each: they passed, which is real evidence, but presenting one
#: sentence as production coverage would repeat the mistake this module exists to
#: prevent in a subtler form.
THIN_EVIDENCE_SENTENCES = 3
QUALITY_CAVEATS = "caveats"
QUALITY_UNSUPPORTED = "unsupported"
QUALITY_UNVERIFIED = "unverified"

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

class _QualityRecordCache:
    """The committed quality record, cached by the file's mtime.

    A cache is a value plus the condition under which that value is still
    valid; keeping the two together means invalidation cannot be done
    half-way. That is also why this is an object rather than two module
    globals behind a lock: the read and the write are one concern, and
    :func:`load_quality_record` below never has to reason about them
    separately.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._languages: dict[str, Any] = {}
        self._mtime: float | None = None

    def get(self, mtime: float) -> dict[str, Any] | None:
        """The cached record if it was read at ``mtime``, otherwise ``None``.

        ``None`` is the miss signal rather than an empty mapping, because a
        valid record may legitimately be an empty mapping.
        """
        with self._lock:
            if self._mtime is not None and mtime == self._mtime:
                return self._languages
            return None

    def put(self, mtime: float, languages: dict[str, Any]) -> None:
        with self._lock:
            self._languages = languages
            self._mtime = mtime

    def clear(self) -> None:
        with self._lock:
            self._languages = {}
            self._mtime = None


_QUALITY_RECORD_CACHE = _QualityRecordCache()


def reset_quality_cache() -> None:
    """Forget the cached quality record (tests, or after regenerating it)."""
    _QUALITY_RECORD_CACHE.clear()


def load_quality_record(path: Path | None = None) -> dict[str, Any]:
    """Read the committed per-language quality record.

    Returns ``{}`` when the file is missing or malformed: an absent measurement
    must make a language ``unverified``, never silently ``production``. A read
    failure is logged, not raised, because this runs inside a capabilities call.
    """
    record_path = path or QUALITY_RECORD_PATH
    try:
        mtime = record_path.stat().st_mtime
    except OSError:
        return {}

    cached = _QUALITY_RECORD_CACHE.get(mtime)
    if cached is not None:
        return cached

    try:
        payload = json.loads(record_path.read_text(encoding="utf-8"))
        languages = payload.get("languages")
        if not isinstance(languages, dict):
            raise ValueError("'languages' must be an object")
    except Exception as exc:  # noqa: BLE001 - a bad record must not break capabilities
        logger.warning("Could not read the language quality record %s: %s", record_path, exc)
        return {}

    _QUALITY_RECORD_CACHE.put(mtime, languages)
    return languages


def _quality_for(language: str) -> dict[str, Any]:
    """Quality row for a canonical language tag (``en-US`` -> ``en``)."""
    record = load_quality_record()
    if not record:
        return {}
    base = (language or "").strip().lower().split("-")[0]
    row = record.get(base)
    return row if isinstance(row, dict) else {}


@dataclass(frozen=True)
class LanguageSupport:
    """Measured G2P support *and* round-trip quality for one language."""

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

    quality: str = QUALITY_UNVERIFIED
    """``production`` / ``caveats`` / ``unsupported`` / ``unverified``."""

    caveat: str | None = None
    """One-line human summary of the measured limitation, when there is one."""

    measured: dict[str, Any] = field(default_factory=dict)
    """The measured figures behind :attr:`quality` (worst WER/CER, failing entries)."""

    def as_dict(self) -> dict:
        return {
            "language": self.language,
            "locale": self.locale,
            "supported": self.supported,
            "quality": self.quality,
            "caveat": self.caveat,
            "reason": self.reason,
            "marker_pairs": self.marker_pairs,
            "measured": self.measured,
        }


def _with_quality(row: LanguageSupport) -> LanguageSupport:
    """Attach the committed quality measurement to a support verdict."""
    if not row.supported:
        return LanguageSupport(
            language=row.language,
            locale=row.locale,
            supported=False,
            reason=row.reason,
            marker_pairs=row.marker_pairs,
            quality=QUALITY_UNSUPPORTED,
        )

    record = _quality_for(row.language)
    if not record:
        return LanguageSupport(
            language=row.language,
            locale=row.locale,
            supported=True,
            reason=row.reason,
            marker_pairs=row.marker_pairs,
            quality=QUALITY_UNVERIFIED,
            caveat=(
                "round-trip quality has not been measured for this language; it is "
                "phonemisable but not verified as usable"
            ),
        )

    quality = str(record.get("quality") or QUALITY_UNVERIFIED)
    failing = list(record.get("entries_over_target") or [])
    measured = {
        "worst_wer": record.get("worst_wer"),
        "worst_cer": record.get("worst_cer"),
        "worst_entry": record.get("worst_entry"),
        "entries_measured": record.get("entries_measured"),
        "entries_over_target": failing,
    }
    measured_count = record.get("entries_measured") or 0
    caveat = None
    if quality == QUALITY_CAVEATS:
        caveat = (
            f"usable with caveats: {len(failing)} of "
            f"{measured_count} measured sentences exceeded target "
            f"({', '.join(failing)}); worst CER {record.get('worst_cer')}"
        )
    elif measured_count < THIN_EVIDENCE_SENTENCES:
        # "production" here means every *measured* sentence passed. With one or
        # two sentences that is a smoke test, not coverage, and saying so is the
        # difference between evidence and a non-empty WAV.
        caveat = (
            f"verified on {measured_count} corpus sentence"
            f"{'' if measured_count == 1 else 's'} only; a small sample, not a "
            "statement of broad quality"
        )
    return LanguageSupport(
        language=row.language,
        locale=row.locale,
        supported=True,
        reason=row.reason,
        marker_pairs=row.marker_pairs,
        quality=quality,
        caveat=caveat,
        measured=measured,
    )


class G2PCapability:
    """Probes which languages the phonemizer can handle, and reports quality."""

    def __init__(self, converter: PhoneticConverter | None = None) -> None:
        self._converter = converter
        self._results: dict[str, LanguageSupport] = {}
        #: Last *complete* report, so a non-blocking read has something to return.
        self._last_report: dict[str, LanguageSupport] = {}
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
            result = _with_quality(self._probe_uncached(wanted))
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
            self._last_report = {}
        return self.report()

    def report(self, *, block: bool = True) -> dict[str, LanguageSupport]:
        """Support and quality for every canonical language in the registry.

        ``block=True`` measures anything not yet probed — that costs one espeak
        dictionary load per pending language (about 13.5 s for all nine on a cold
        process), so it is for startup, tests and the evaluation harness only.

        ``block=False`` is the request-path shape: it returns the last complete
        report, or whatever has been probed so far plus the rest as pending. It
        never loads a dictionary, which is what keeps ``capabilities()`` off the
        synthesis hot path (review finding F3).
        """
        languages = sorted({family.language for family in LANGUAGE_FAMILIES.values()})
        if not block:
            with self._lock:
                if self._last_report:
                    return dict(self._last_report)
                return dict(self._results)

        result = {language: self.probe(language) for language in languages}
        with self._lock:
            self._last_report = dict(result)
        return result

    def pending_languages(self) -> list[str]:
        """Canonical languages that have not been measured yet in this process."""
        known = sorted({family.language for family in LANGUAGE_FAMILIES.values()})
        with self._lock:
            rows = set(self._results)
        return [language for language in known if language not in rows]

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

        Safe for a health/capabilities endpoint: it never pays the ~1.5 s
        per-language backend load, and it names the languages still pending rather
        than omitting them (an omitted language would look supported).
        """
        with self._lock:
            rows = dict(self._results)
        pending = self.pending_languages()
        return {
            "supported": [language for language, row in rows.items() if row.supported],
            "with_caveats": [
                row.as_dict()
                for row in rows.values()
                if row.supported and row.quality == QUALITY_CAVEATS
            ],
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
        for row in report.values():
            if row.supported and row.quality == QUALITY_CAVEATS:
                logger.warning(
                    "G2P language SUPPORTED WITH CAVEATS: %s — %s", row.language, row.caveat
                )
            elif row.supported and row.quality == QUALITY_UNVERIFIED:
                logger.warning(
                    "G2P language SUPPORTED BUT UNVERIFIED: %s — no round-trip measurement",
                    row.language,
                )
        for row in unsupported:
            logger.warning(
                "G2P language UNAVAILABLE: %s (locale %s) — %s",
                row.language,
                row.locale,
                row.reason,
            )
        return report


#: Process-wide probe cache. Cheap to read; the first blocking read runs the probe.
g2p_capability = G2PCapability()


__all__ = [
    "G2PCapability",
    "LanguageSupport",
    "PROBE_SENTENCES",
    "QUALITY_CAVEATS",
    "QUALITY_PRODUCTION",
    "QUALITY_RECORD_PATH",
    "QUALITY_UNSUPPORTED",
    "QUALITY_UNVERIFIED",
    "THIN_EVIDENCE_SENTENCES",
    "g2p_capability",
    "load_quality_record",
    "reset_quality_cache",
]
