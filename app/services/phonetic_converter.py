"""Phonetic converter for text-to-speech synthesis.

Provides text normalization, pronunciation dictionary mapping, persistent custom lexicon,
per-request lexicon overrides, inline phonetic markup ([word|pronunciation] and <phoneme ph="ipa">),
and grapheme-to-phoneme (G2P) conversion using espeak-ng / phonemizer via Kokoro's Tokenizer.

Language policy
---------------

* **Locale resolution.** ``language`` may be a caller tag (``zh``) or an engine
  locale (``cmn``); :func:`app.services.voice_registry.engine_locale` resolves it
  from the voice registry, so the ``cmn`` the voice records already carry is the
  code espeak receives.
* **Markers are metadata.** espeak brackets a span it read in another language
  with ``(en)`` / ``(ja)``. Those tokens are stripped before the phoneme string
  leaves this module; a vocoder would otherwise speak them literally.
* **Lexicon scope.** The built-in units, abbreviations and technology respellings
  are English and apply to English voices only. Other languages use their own G2P
  or an explicit lexicon entry (``ipa`` is exact).
* **Refusal over garbage.** When espeak spells the input out because it cannot
  read the script (the Japanese kanji case), the conversion raises
  :class:`UnsupportedLanguageError`, which the HTTP layer returns as ``422``.

Thread safety
-------------
espeak-ng is a single C library that keeps its configuration and synthesis state in
process globals; its documentation states it is *not* designed to be used in
multithreaded contexts.  ``phonemizer`` reinforces that by caching exactly one
``EspeakBackend`` per language in a module-level dict (``_PHONEMIZER_CACHE``), so every
thread in this process shares one backend over one library instance.

Calling G2P from two threads at once therefore corrupts the input: measured on this
repository at two workers, 35-50 of 200 calls returned the wrong result (phonemes
belonging to another request, and a swallowed decode error that returned *prose*
where phonemes were expected).  Kokoro's ``Tokenizer.tokenize`` is pure and safe, but
``Tokenizer.phonemize`` is not.

Every grapheme-to-phoneme call in this process must therefore hold :data:`G2P_LOCK`.
The critical section is a few milliseconds next to a render of hundreds of
milliseconds, so serialising it keeps model-level concurrency while removing the
shared-state race.

The persistent lexicon is the second piece of shared state
-----------------------------------------------------------
Renders run in a bounded thread pool while ``POST``/``DELETE /api/tts/lexicon``
mutate the custom lexicon from the event loop.  A dict iterated in one thread and
mutated in another raises ``RuntimeError('dictionary changed size during
iteration')``, which surfaced as a 500 on an otherwise valid synthesis.  The
lexicon is therefore **copy-on-write**: readers take a snapshot under
:attr:`PhoneticConverter._lexicon_lock` and iterate that snapshot, mutations
build a new mapping, persist it atomically (``tempfile`` + ``os.replace``), and
only then publish it.  A reader therefore always sees one consistent mapping, and
a reader can never observe a half-written file.
"""

import json
import logging
import os
import re
import tempfile
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from app.config import settings
from app.services.voice_registry import canonical_language, engine_locale

logger = logging.getLogger(__name__)

#: espeak-ng language-switch markers, e.g. ``(en)`` / ``(ja)`` / ``(en-gb)``.
#:
#: espeak emits these around a span it read in a different language. They are
#: metadata about *how* the phonemes were produced, never phonemes themselves: a
#: vocoder handed ``(en)`` speaks the parentheses and the language name. Every
#: marker is therefore removed before the phoneme string leaves this module.
LANGUAGE_SWITCH_MARKER = re.compile(r"\(([a-z]{2,3}(?:-[a-z]{2,4})?)\)")

#: Minimum number of language-switch marker *pairs* that means "this is not a
#: foreign word, this is espeak spelling the input out".
#:
#: A French sentence containing an English brand name produces a single
#: ``(en)...(fr)`` pair: the switch is a *word*, and one pair is legitimate
#: (measured: 1 pair for a French sentence containing three English brand names).
#: When espeak cannot read a script it instead names each character ("japanese
#: letter") and emits a pair per character — measured: 21 pairs for a 21-character
#: Japanese sentence. The guard is therefore length-relative; see
#: :func:`marks_spelling_fallback`.
SPELLING_FALLBACK_MARKER_PAIRS = 3

#: Characters per switch pair above which the switches are spelling, not words.
#:
#: Spelling fallback runs at roughly one pair per character; genuine loanword
#: switches are one pair per foreign *word*. One pair per four characters sits
#: between the two measured regimes with room for short sentences.
SPELLING_FALLBACK_CHARS_PER_PAIR = 4

#: Minimum number of phoneme tokens before the repeated-token signature is
#: considered at all. Short utterances legitimately repeat.
SPELLING_OUTPUT_MIN_TOKENS = 6

#: Distinct-token ratio at or below which the phoneme string is one syllable
#: cycling, not language. Measured on this repository's corpus: legitimate
#: sentences sit at 0.69-1.00; espeak spelling a script out sits at 0.05-0.10
#: (a 21-character Japanese sentence gave 21 tokens with 2 distinct values).
SPELLING_OUTPUT_MAX_DISTINCT_RATE = 0.35

#: Tokens per input character at or above which the output is character-for-
#: character, not word-for-word. Spelling fallback emits exactly one token per
#: source character; real text emits about one per word (measured maximum for a
#: legitimate sentence in its own language: 0.76, syllable-dense Mandarin).
SPELLING_OUTPUT_MIN_TOKENS_PER_CHAR = 0.6

#: Scripts that :data:`READABLE_SCRIPTS` maps onto languages.
#:
#: Latin is deliberately not listed: every language this service ships accepts
#: Latin characters (brand names, code identifiers, loanwords), which is why
#: ``Reactory`` is readable in Spanish, Italian, Portuguese and Hindi.
SCRIPT_PATTERNS: dict[str, re.Pattern[str]] = {
    "kana": re.compile(r"[\u3040-\u309f\u30a0-\u30ff\u31f0-\u31ff]"),
    "han": re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]"),
    "hangul": re.compile(r"[\u1100-\u11ff\u3130-\u318f\ua960-\ua97f\uac00-\ud7af]"),
    "devanagari": re.compile(r"[\u0900-\u097f\ua8e0-\ua8ff]"),
    "arabic": re.compile(r"[\u0600-\u06ff\u0750-\u077f\ufb50-\ufdff\ufe70-\ufefc]"),
    "hebrew": re.compile(r"[\u0590-\u05ff\ufb1d-\ufb4f]"),
    "cyrillic": re.compile(r"[\u0400-\u04ff\u0500-\u052f]"),
    "greek": re.compile(r"[\u0370-\u03ff\u1f00-\u1fff]"),
    "thai": re.compile(r"[\u0e00-\u0e7f]"),
}

#: Scripts each language's own orthography uses, by ISO 639-1 (or engine locale)
#: prefix, plus Latin because every shipped language reads Latin loanwords.
#:
#: This is the *deterministic* half of the SPEECH-010 guard: a Japanese sentence
#: sent to an English voice must be refused before a 21-second render, and for
#: ``en-us``/``es``/``it`` espeak emits no language-switch markers at all — it just
#: transliterates each character (``dʒˈæpəniːzlˌɛɾɚ`` twenty-one times), so the
#: marker guard alone could not see it. A language absent from this map is left to
#: the backend to accept or reject, so a custom voice pack is not broken by it.
READABLE_SCRIPTS: dict[str, frozenset[str]] = {
    **{
        code: frozenset({"latin"})
        for code in (
            "en", "es", "pt", "fr", "it", "de", "nl", "sv", "da", "no", "nb", "nn",
            "fi", "is", "pl", "cs", "sk", "sl", "hr", "bs", "tr", "az", "id", "ms",
            "vi", "ro", "hu", "ca", "gl", "eu", "et", "lv", "lt", "sq", "cy", "ga",
            "gd", "mt", "af", "sw", "tl", "fil",
        )
    },
    "hi": frozenset({"devanagari", "latin"}),
    "mr": frozenset({"devanagari", "latin"}),
    "ne": frozenset({"devanagari", "latin"}),
    "zh": frozenset({"han", "latin"}),
    "cmn": frozenset({"han", "latin"}),
    "yue": frozenset({"han", "latin"}),
    "ja": frozenset({"kana", "han", "latin"}),
    "ko": frozenset({"hangul", "han", "latin"}),
    "ar": frozenset({"arabic", "latin"}),
    "ur": frozenset({"arabic", "latin"}),
    "fa": frozenset({"arabic", "latin"}),
    "ps": frozenset({"arabic", "latin"}),
    "he": frozenset({"hebrew", "latin"}),
    "iw": frozenset({"hebrew", "latin"}),
    "ru": frozenset({"cyrillic", "latin"}),
    "uk": frozenset({"cyrillic", "latin"}),
    "bg": frozenset({"cyrillic", "latin"}),
    "sr": frozenset({"cyrillic", "latin"}),
    "mk": frozenset({"cyrillic", "latin"}),
    "be": frozenset({"cyrillic", "latin"}),
    "kk": frozenset({"cyrillic", "latin"}),
    "el": frozenset({"greek", "latin"}),
    "th": frozenset({"thai", "latin"}),
}

#: Language codes whose built-in lexicon, unit and abbreviation expansions apply.
#:
#: The built-in table is an *English* respelling dictionary ("Ree-actory",
#: "kilometers", "Mister"); applying it to French or Spanish splices English
#: phonemes into the utterance. Non-English languages get their own G2P, or an
#: explicit per-request / persistent lexicon entry (the ``ipa`` field is exact).
BUILTIN_LEXICON_LANGUAGES: frozenset[str] = frozenset({"en"})

#: Serialises every grapheme-to-phoneme call in this process.
#:
#: Reentrant so a caller that already holds it (an engine wrapping its own
#: phonemisation) can still reach :meth:`PhoneticConverter.convert_to_phonetic`
#: without deadlocking.
G2P_LOCK = threading.RLock()


class PhoneticConversionError(RuntimeError):
    """Grapheme-to-phoneme conversion failed and produced no usable phonemes.

    Raised instead of silently returning the normalised *prose*, which a caller
    that asked for phonemes would otherwise hand to the engine as if it were IPA.
    """


class UnsupportedLanguageError(ValueError):
    """The phonemizer cannot read this text in the requested language.

    A ``ValueError`` on purpose: the HTTP layer already maps that to ``422``, which
    is the honest status for "this voice cannot speak this language". Silence or
    fluent nonsense would both be worse than a clear refusal.
    """

    def __init__(self, language: str, reason: str) -> None:
        self.language = language
        self.reason = reason
        super().__init__(f"Language '{language}' cannot be phonemised: {reason}")


def strip_language_markers(phonemes: str) -> tuple[str, list[str]]:
    """Remove espeak language-switch markers, returning ``(clean, markers)``.

    The markers are returned so a caller can decide what their *presence* means
    (a loanword switch is fine, spelling fallback is not) without ever handing one
    to the vocoder. Punctuation-adjacent whitespace left behind is collapsed.
    """
    markers = LANGUAGE_SWITCH_MARKER.findall(phonemes or "")
    clean = LANGUAGE_SWITCH_MARKER.sub("", phonemes or "")
    clean = re.sub(r"\s{2,}", " ", clean).strip()
    return clean, markers


def is_english(language: str | None) -> bool:
    """Whether a language code denotes English (``en``, ``en-us``, ``en-gb``)."""
    if not language:
        return True
    normalised = language.strip().lower().replace("_", "-")
    return normalised.split("-")[0] in BUILTIN_LEXICON_LANGUAGES


def marks_spelling_fallback(text: str, marker_pairs: int) -> bool:
    """Whether espeak's language switches show it *spelled the input out*.

    Two regimes are measured on this repository:

    * a foreign word or brand name switches once — 1 pair for a 55-character
      French sentence containing three English brand names;
    * a script espeak cannot read falls back to naming every character — 21 pairs
      for a 21-character Japanese sentence.

    Counting pairs alone would refuse the first (a legitimate loanword) once a
    sentence names three products. The threshold therefore scales with the input:
    a fixed floor for short text, then one pair per
    :data:`SPELLING_FALLBACK_CHARS_PER_PAIR` characters.
    """
    threshold = max(
        SPELLING_FALLBACK_MARKER_PAIRS,
        len(text or "") // SPELLING_FALLBACK_CHARS_PER_PAIR,
    )
    return marker_pairs >= threshold


def unreadable_script(text: str, locale: str | None) -> str | None:
    """The script in ``text`` that ``locale`` cannot read, or ``None``.

    Deterministic, pre-phonemisation half of the SPEECH-010 guard. It exists
    because the marker guard cannot see the worst case: a Japanese sentence sent
    to an English voice makes espeak-ng transliterate each character without
    emitting a single language-switch marker (measured: ``dʒˈæpəniːzlˌɛɾɚ``
    repeated 21 times, 20.9 s of audio, HTTP 200).

    A language absent from :data:`READABLE_SCRIPTS` returns ``None`` — the backend
    keeps the final say, so an out-of-tree voice pack is not broken by a table it
    is not in.
    """
    prefix = (locale or "").strip().lower().replace("_", "-").split("-")[0]
    if not prefix:
        return None
    allowed = READABLE_SCRIPTS.get(prefix)
    if allowed is None:
        return None
    for name, pattern in SCRIPT_PATTERNS.items():
        if name in allowed:
            continue
        if pattern.search(text or ""):
            return name
    return None


def marks_spelling_output(text: str, phonemes: str) -> bool:
    """Whether phonemes show one syllable per character — spelling, not speech.

    Post-phonemisation half of the guard, for a script this module does not
    enumerate. Two conditions must both hold, so ordinary repetition in speech is
    not caught:

    * the token stream is mostly one value cycling (distinct-token rate at or
      below :data:`SPELLING_OUTPUT_MAX_DISTINCT_RATE`), and
    * there is about one token per source character (at or above
      :data:`SPELLING_OUTPUT_MIN_TOKENS_PER_CHAR`).

    Measured separation on this repository's corpus: legitimate sentences 0.69-1.00
    distinct and at most 0.76 tokens/char; spelling fallback 0.05-0.10 distinct and
    1.00 tokens/char.
    """
    tokens = (phonemes or "").split()
    if len(tokens) < SPELLING_OUTPUT_MIN_TOKENS:
        return False
    distinct_rate = len(set(tokens)) / len(tokens)
    if distinct_rate > SPELLING_OUTPUT_MAX_DISTINCT_RATE:
        return False
    characters = len("".join((text or "").split()))
    if not characters:
        return False
    return len(tokens) / characters >= SPELLING_OUTPUT_MIN_TOKENS_PER_CHAR


def ensure_text_is_readable(text: str, locale: str | None) -> None:
    """Raise :class:`UnsupportedLanguageError` when the script is unreadable.

    Called *before* phonemisation on every render path, so the refusal costs
    nothing and cannot be reached with a 20-second garbage render.
    """
    script = unreadable_script(text, locale)
    if script is not None:
        raise UnsupportedLanguageError(
            str(locale or ""),
            f"the text is written in {script} script, which the "
            f"'{locale}' voice cannot read",
        )


def validate_phoneme_output(text: str, locale: str, phonemes: str) -> str:
    """Strip markers and refuse a phoneme string that spells the text out.

    The single implementation of the output half of the marker policy, shared by
    the converter and by the engine's fused (self-phonemising) path so the two
    cannot drift. Returns the marker-free phonemes, or raises
    :class:`UnsupportedLanguageError`.
    """
    clean, markers = strip_language_markers(phonemes or "")
    if markers and marks_spelling_fallback(text, len(markers) // 2):
        raise UnsupportedLanguageError(
            locale,
            "the phonemizer fell back to spelling the input out "
            f"({len(markers)} language switches, e.g. {markers[:4]}); "
            "this voice cannot read that script",
        )
    if not clean and (phonemes or "").strip():
        raise UnsupportedLanguageError(
            locale,
            "the phonemizer produced only language-switch markers",
        )
    if marks_spelling_output(text, clean):
        sample = clean.split()[:3]
        raise UnsupportedLanguageError(
            locale,
            "the phonemizer emitted one syllable per character "
            f"(e.g. {' '.join(sample)}); this voice cannot read that script",
        )
    return clean


@contextmanager
def g2p_lock() -> Iterator[None]:
    """Hold the process-wide grapheme-to-phoneme lock.

    Exposed so an engine that phonemises internally (``kokoro_onnx`` does this when
    it is handed plain text rather than phonemes) can serialise that call against
    :meth:`PhoneticConverter.convert_to_phonetic`.
    """
    with G2P_LOCK:
        yield


class PhoneticConverter:
    """A phonetic converter that normalizes text and produces phonetic representations."""

    def __init__(self) -> None:
        self._tokenizer = None
        self._ready = False
        self._init_tokenizer()

        # Built-in platform & technology lexicon
        self.builtin_lexicon: dict[str, str] = {
            r"\bReactory\b": "Ree-actory",
            r"\breactory\b": "ree-actory",
            r"\bAPI\b": "A P I",
            r"\bAPIs\b": "A P Is",
            r"\bGraphQL\b": "Graph Q L",
            r"\bGQL\b": "G Q L",
            r"\bSQL\b": "sequel",
            r"\bNoSQL\b": "no-sequel",
            r"\bPostgres\b": "post-gress",
            r"\bPostgreSQL\b": "post-gress Q L",
            r"\bMySQL\b": "my sequel",
            r"\bSQLite\b": "sequel lite",
            r"\bMongoDB\b": "mongo D B",
            r"\bMongo\b": "mongo",
            r"\bONNX\b": "on-ix",
            r"\bJSON\b": "jay-son",
            r"\bYAML\b": "yam-el",
            r"\bHTML\b": "H T M L",
            r"\bCSS\b": "C S S",
            r"\bURL\b": "U R L",
            r"\bURLs\b": "U R Ls",
            r"\bURI\b": "U R I",
            r"\bURIs\b": "U R Is",
            r"\bHTTP\b": "H T T P",
            r"\bHTTPS\b": "H T T P S",
            r"\bREST\b": "rest",
            r"\bRESTful\b": "rest-ful",
            r"\bJWT\b": "jot",
            r"\bOAuth\b": "O-auth",
            r"\bCLI\b": "C L I",
            r"\bGUI\b": "gooey",
            r"\bSDK\b": "S D K",
            r"\bSDKs\b": "S D Ks",
            r"\bUI\b": "U I",
            r"\bUX\b": "U X",
            r"\bAI\b": "A I",
            r"\bML\b": "M L",
            r"\bTTS\b": "T T S",
            r"\bSTT\b": "S T T",
            r"\bWAV\b": "wave",
            r"\bMP3\b": "M P 3",
            r"\bCPU\b": "C P U",
            r"\bGPU\b": "G P U",
            r"\bRAM\b": "ram",
            r"\bROM\b": "rom",
            r"\bOS\b": "O S",
            r"\biOS\b": "eye O S",
            r"\bmacOS\b": "mac O S",
            r"\bFQN\b": "F Q N",
            r"\bUUID\b": "U U I D",
            r"\bAMQ\b": "A M Q",
            r"\bPWA\b": "P W A",
            r"\bWebSocket\b": "web-socket",
            r"\bWebSockets\b": "web-sockets",
        }

        #: Exact IPA for platform brand names that a language's own G2P would
        #: otherwise read with English phonemes. espeak-ng's French voice switches
        #: to English for any word ending in ``-y`` (measured: ``Reactory`` ->
        #: ``(en)ɹɪˈaktəɹi(fr)``), so the French pronunciation has to be supplied
        #: explicitly rather than left to that heuristic. Values are injected as
        #: protected IPA spans, so no G2P can re-interpret them. This is the
        #: per-language lexicon entry the SPEECH-010 scope calls for, kept beside
        #: the English builtins so it ships with every deployment.
        self.builtin_language_ipa: dict[str, dict[str, str]] = {
            "fr-fr": {"Reactory": "ʁeaktoʁi"},
        }

        # Common abbreviations
        self.abbreviations: dict[str, str] = {
            r"\bDr\.\s*": "Doctor ",
            r"\bMr\.\s*": "Mister ",
            r"\bMrs\.\s*": "Missus ",
            r"\bMs\.\s*": "Miss ",
            r"\bProf\.\s*": "Professor ",
            r"\bvs\.\s*": "versus ",
            r"\be\.g\.,?\s*": "for example, ",
            r"\bi\.e\.,?\s*": "that is, ",
            r"\betc\.\s*": "et cetera. ",
            r"\bdept\.\s*": "department ",
            r"\bapprox\.\s*": "approximately ",
        }

        # Units and symbols
        self.units: dict[str, str] = {
            r"(\d+)\s*km\b": r"\1 kilometers",
            r"(\d+)\s*kg\b": r"\1 kilograms",
            r"(\d+)\s*cm\b": r"\1 centimeters",
            r"(\d+)\s*mm\b": r"\1 millimeters",
            r"(\d+)\s*ms\b": r"\1 milliseconds",
            r"(\d+)\s*mph\b": r"\1 miles per hour",
            r"(\d+)\s*%\b": r"\1 percent",
            r"\$(\d+(?:\.\d{1,2})?)\b": r"\1 dollars",
            r"€(\d+(?:\.\d{1,2})?)\b": r"\1 euros",
            r"£(\d+(?:\.\d{1,2})?)\b": r"\1 pounds",
        }

        # Custom persistent lexicon, copy-on-write.  ``_custom_lexicon`` is
        # rebound, never mutated in place, so a reader that took a snapshot can
        # iterate it safely while a writer publishes a new mapping.
        self._custom_lexicon: dict[str, dict] = {}
        self._last_mtime: float | None = None
        #: Guards publication of ``_custom_lexicon`` / ``_last_mtime``.
        #:
        #: Deliberately separate from :data:`G2P_LOCK`: a lexicon mutation must
        #: never wait behind an espeak call, and a render must never wait behind
        #: a file write.
        self._lexicon_lock = threading.RLock()
        self.load_lexicon()

    def _init_tokenizer(self) -> None:
        """Initialize the underlying Kokoro tokenizer."""
        try:
            from kokoro_onnx.tokenizer import Tokenizer
            self._tokenizer = Tokenizer()
            self._ready = True
            logger.info("PhoneticConverter initialized successfully with Kokoro Tokenizer.")
        except Exception as e:  # noqa: BLE001 - optional espeak/tokenizer probe at startup
            logger.warning("Could not initialize Kokoro Tokenizer: %s", e)
            self._ready = False

    @property
    def is_available(self) -> bool:
        """Check if phonetic conversion is available."""
        return self._ready and self._tokenizer is not None

    # ------------------------------------------------------------------
    # Persistent lexicon: copy-on-write, snapshot reads
    # ------------------------------------------------------------------

    def lexicon_snapshot(self) -> dict[str, dict]:
        """Return a private copy of the custom lexicon, safe to iterate.

        This is the only supported way to read the entries from a render path.
        Iterating ``self._custom_lexicon`` directly is a data race: the HTTP
        handlers mutate it from the event loop while renders run in the thread
        pool.

        The copy is shallow on purpose.  Entry *values* are never mutated in
        place anywhere in this class — a mutation replaces the whole mapping — so
        a snapshot always sees consistent value dicts.
        """
        with self._lexicon_lock:
            return dict(self._custom_lexicon)

    def _publish(self, entries: dict[str, dict], mtime: float | None) -> None:
        """Replace the live mapping and its file stamp in one atomic step."""
        with self._lexicon_lock:
            self._custom_lexicon = entries
            self._last_mtime = mtime

    @staticmethod
    def _stat_mtime(path: Path) -> float | None:
        try:
            return path.stat().st_mtime
        except OSError:
            return None

    def _atomic_write_lexicon(self, entries: dict[str, dict]) -> None:
        """Write the lexicon to disk atomically (tempfile + ``os.replace``).

        Readers call :meth:`load_lexicon` on every render, so a non-atomic
        ``open(..., "w")`` let a render observe a half-written file, fail to
        parse it and silently keep the *old* entries — which the renderer
        fingerprint, computed from the *new* file, would then happily cache.
        """
        lexicon_file = Path(settings.lexicon_file_path)
        lexicon_file.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"entries": entries}, indent=2, ensure_ascii=False)
        handle_fd, temp_name = tempfile.mkstemp(
            dir=str(lexicon_file.parent), prefix=".lexicon-", suffix=".json"
        )
        try:
            with os.fdopen(handle_fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, lexicon_file)
        except BaseException:
            try:
                os.unlink(temp_name)
            except OSError:
                # Best effort: we are already unwinding, and a missing or
                # locked temp file must not mask the original exception.
                pass
            raise

    def load_lexicon(self) -> None:
        """Load the custom lexicon from disk, reloading only when it changed.

        The file is read and parsed *outside* the lock, so a slow filesystem does
        not stall concurrent renders; the mapping is published under the lock in
        a single rebind, so readers always see a complete mapping.
        """
        lexicon_file = Path(settings.lexicon_file_path)
        mtime = self._stat_mtime(lexicon_file)
        if mtime is None:
            with self._lexicon_lock:
                empty = not self._custom_lexicon
                changed = self._last_mtime is not None
            if changed or empty:
                self._publish({}, None)
            return

        with self._lexicon_lock:
            if mtime == self._last_mtime:
                return

        try:
            with open(lexicon_file, "r", encoding="utf-8") as handle:
                entries = json.load(handle).get("entries", {})
            if not isinstance(entries, dict):
                raise ValueError("'entries' must be a JSON object")
        except Exception as e:  # noqa: BLE001 - best-effort lexicon reload, never fatal
            logger.warning("Failed to read custom lexicon from %s: %s", lexicon_file, e)
            # Keep whatever is already published: a transient read failure must
            # not silently drop the operator's pronunciations.
            return

        self._publish(entries, mtime)
        logger.info("Loaded %d custom lexicon entries from %s", len(entries), lexicon_file)

    def save_lexicon(self) -> None:
        """Persist the current custom lexicon to disk, atomically."""
        entries = self.lexicon_snapshot()
        try:
            self._atomic_write_lexicon(entries)
            logger.info(
                "Saved %d custom lexicon entries to %s",
                len(entries),
                settings.lexicon_file_path,
            )
        except Exception as e:  # noqa: BLE001 - lexicon persistence is best-effort
            logger.error("Failed to save custom lexicon to %s: %s", settings.lexicon_file_path, e)

    def get_all_entries(self) -> dict[str, dict]:
        """Return all active lexicon entries (both built-in and custom)."""
        self.load_lexicon()
        result = {}
        # Builtins
        for pat, repl in self.builtin_lexicon.items():
            # Clean regex boundary for display
            clean_word = pat.replace(r"\b", "").replace(r"\B", "")
            result[clean_word] = {
                "replacement": repl,
                "source": "builtin",
                "languages": ["en"],
            }

        # Custom overrides take precedence
        for word, info in self.lexicon_snapshot().items():
            result[word] = {**info, "source": "custom"}

        return result

    def get_custom_entries(self) -> dict[str, dict]:
        """Return custom user-defined lexicon entries only.

        A copy: the caller must not be able to mutate the live mapping (or
        iterate one that a concurrent render is publishing over).
        """
        self.load_lexicon()
        return self.lexicon_snapshot()

    @staticmethod
    def _language_matches(candidate: str, language: str | None) -> bool:
        """Whether a lexicon entry's language scope covers the target language."""
        if not candidate or not language:
            return False
        if candidate.strip().lower() == language.strip().lower():
            return True
        if engine_locale(candidate) == engine_locale(language):
            return True
        wanted = canonical_language(language)
        return wanted is not None and canonical_language(candidate) == wanted

    @classmethod
    def entry_for_language(cls, info: dict, language: str | None) -> dict | None:
        """Resolve a lexicon entry for one target language, or ``None``.

        An entry with an explicit ``languages`` list applies only to those
        languages. An entry without one is treated as English: the shipped and
        built-in lexicons are English respellings, and applying one to French or
        Spanish splices English phonemes into the utterance — the SPEECH-010
        defect. An operator who wants a word pronounced in another language either
        adds that language to ``languages`` (with a target-language ``ipa``) or
        relies on that language's own G2P.
        """
        if not isinstance(info, dict):
            return None
        languages = info.get("languages")
        if languages:
            if any(cls._language_matches(str(entry), language) for entry in languages):
                return info
            return None
        return info if is_english(language) else None

    def get_entry(self, word: str) -> dict | None:
        """Get pronunciation entry for a word."""
        self.load_lexicon()
        entries = self.lexicon_snapshot()
        if word in entries:
            return {**entries[word], "source": "custom"}
        raw_pat = rf"\b{re.escape(word)}\b"
        if raw_pat in self.builtin_lexicon:
            return {"replacement": self.builtin_lexicon[raw_pat], "source": "builtin"}
        return None

    def set_entry(
        self,
        word: str,
        replacement: str | None = None,
        ipa: str | None = None,
        languages: list[str] | None = None,
    ) -> dict:
        """Add or update a custom lexicon entry.

        ``languages`` scopes the entry to one or more language tags (``["fr-FR"]``).
        An entry without it is English-only, matching the shipped lexicon: an
        unscoped respelling applied to another language splices English phonemes
        into it (SPEECH-010).

        Copy-on-write: build the next mapping, persist it atomically, then
        publish it.  Concurrent renders keep iterating the mapping they already
        hold, so they can neither see a partial update nor trip
        ``RuntimeError: dictionary changed size during iteration``.
        """
        entry: dict[str, object] = {}
        if replacement:
            entry["replacement"] = replacement.strip()
        if ipa:
            entry["ipa"] = ipa.strip()
        if languages:
            entry["languages"] = [str(language).strip() for language in languages if str(language).strip()]

        with self._lexicon_lock:
            updated = {**self._custom_lexicon, word: entry}
        try:
            self._atomic_write_lexicon(updated)
        except Exception as e:  # noqa: BLE001 - persistence stays best-effort
            # Match the previous behaviour: the in-memory change still takes
            # effect for this process, but report that it did not reach disk.
            logger.error(
                "Failed to persist custom lexicon to %s: %s",
                settings.lexicon_file_path,
                e,
            )
        self._publish(updated, self._stat_mtime(Path(settings.lexicon_file_path)))
        return {**entry, "word": word, "source": "custom"}

    def delete_entry(self, word: str) -> bool:
        """Delete a custom lexicon entry (copy-on-write, atomic persist)."""
        with self._lexicon_lock:
            if word not in self._custom_lexicon:
                return False
            updated = {k: v for k, v in self._custom_lexicon.items() if k != word}
        try:
            self._atomic_write_lexicon(updated)
        except Exception as e:  # noqa: BLE001 - persistence stays best-effort
            logger.error(
                "Failed to persist custom lexicon to %s: %s",
                settings.lexicon_file_path,
                e,
            )
        self._publish(updated, self._stat_mtime(Path(settings.lexicon_file_path)))
        return True

    def normalize_text(
        self,
        text: str,
        custom_lexicon: dict[str, str] | None = None,
        language: str | None = None,
    ) -> str:
        """Pre-process text for natural pronunciation before phonemization.

        The built-in units, abbreviations and technology lexicon are **English**
        respellings. They apply only when ``language`` is English; otherwise the
        word is left to the target language's own G2P, or to an explicit lexicon
        entry (``ipa`` produces exact output). ``language=None`` keeps the legacy
        behaviour of treating the text as English, so existing direct callers do
        not change.
        """
        # Auto-reload if file was edited externally on disk
        self.load_lexicon()

        normalized = text
        apply_builtins = is_english(language)

        # 1. Expand standard units & currency (English words: "kilometers")
        if apply_builtins:
            for pattern, replacement in self.units.items():
                normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)

        # 2. Expand common abbreviations (English words: "Mister", "versus")
        if apply_builtins:
            for pattern, replacement in self.abbreviations.items():
                normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)

        # 3. Handle inline bracket respellings [word|respelling] where respelling is NOT IPA (/.../)
        normalized = re.sub(r"\[([^\|\]]+)\|([^/\]][^\]]*)\]", r"\2", normalized)

        # 4. Apply per-request lexicon overrides (if supplied)
        if custom_lexicon:
            for word, override in custom_lexicon.items():
                if not (override.startswith("/") and override.endswith("/")):
                    pat = rf"\b{re.escape(word)}\b"
                    normalized = re.sub(pat, override, normalized)

        # 5. Apply persistent custom lexicon replacements for this language.
        # Iterate a snapshot: POST/DELETE /api/tts/lexicon mutate this mapping
        # from the event loop while this render runs in the inference pool.
        for word, info in self.lexicon_snapshot().items():
            effective = self.entry_for_language(info, language)
            repl = (effective or {}).get("replacement")
            if repl:
                pat = rf"\b{re.escape(word)}\b"
                normalized = re.sub(pat, repl, normalized)

        # 6. Apply built-in domain & tech lexicon (English respellings only)
        if apply_builtins:
            for pattern, replacement in self.builtin_lexicon.items():
                normalized = re.sub(pattern, replacement, normalized)

        # 7. Clean extra whitespace
        normalized = re.sub(r"\s+", " ", normalized).strip()

        return normalized

    def convert_to_phonetic(
        self,
        text: str,
        language: str = "en-us",
        custom_lexicon: dict[str, str] | None = None,
    ) -> str:
        """Convert text to phonetic (IPA) representation with exact phoneme protection.

        Supports:
        - Regular words (phonemized via espeak-ng)
        - Persistent custom IPA overrides
        - Per-request custom IPA overrides
        - Inline bracket notation: [word|/ipa/]
        - Inline SSML tag: <phoneme ph="ipa">word</phoneme>

        ``language`` is a caller language tag or an engine locale; it is resolved
        through the voice registry (``zh`` -> ``cmn``) before reaching espeak-ng.

        Language-switch markers are never returned. A span espeak read in another
        language is legitimate (a loanword); the markers are stripped as metadata.
        When the markers show espeak *spelling the input out* — the signature of a
        script it cannot read — the request is refused with
        :class:`UnsupportedLanguageError` (a ``ValueError``, so HTTP 422) rather
        than synthesising nonsense.
        """
        # Step 1: Pre-normalize text (expanding units, abbreviations, respellings)
        lang = engine_locale(language or "en-us")
        normalized = self.normalize_text(
            text, custom_lexicon=custom_lexicon, language=lang
        )

        if not self.is_available:
            logger.debug("Phonetic converter unavailable; returning normalized text.")
            return normalized

        # Step 2: Inject inline IPA markers for dictionary entries with 'ipa' configured
        # Check custom_lexicon first
        if custom_lexicon:
            for word, val in custom_lexicon.items():
                if val.startswith("/") and val.endswith("/"):
                    ipa_content = val.strip("/")
                    pat = rf"\b{re.escape(word)}\b"
                    normalized = re.sub(pat, f"[{word}|/{ipa_content}/]", normalized)

        # Check persistent custom lexicon with ipa (snapshot, see normalize_text).
        # Only entries scoped to this language apply; an English respelling must
        # not be spliced into a French or Spanish sentence.
        for word, info in self.lexicon_snapshot().items():
            effective = self.entry_for_language(info, lang)
            ipa_val = (effective or {}).get("ipa")
            if ipa_val:
                pat = rf"\b{re.escape(word)}\b"
                normalized = re.sub(pat, f"[{word}|/{ipa_val}/]", normalized)

        # Step 2b: inject the platform's per-language brand pronunciations. These
        # are exact IPA, wrapped so the target G2P cannot re-interpret them.
        for word, ipa_val in (self.builtin_language_ipa.get(lang) or {}).items():
            if f"[{word}|" in normalized:
                continue  # an explicit lexicon entry already supplied the IPA
            pat = rf"\b{re.escape(word)}\b"
            normalized = re.sub(pat, f"[{word}|/{ipa_val}/]", normalized)

        # Step 3: Segment text into ('text', ...) and ('ipa', ...) chunks
        pattern = re.compile(
            r"\[([^\|\]]+)\|/([^/\]]+)/\]|<phoneme\s+ph=\"([^\"]+)\">([^<]+)</phoneme>"
        )
        segments: list[tuple[str, str]] = []
        last_idx = 0

        for match in pattern.finditer(normalized):
            start, end = match.span()
            if start > last_idx:
                segments.append(("text", normalized[last_idx:start]))
            ipa_str = match.group(2) if match.group(2) is not None else match.group(3)
            segments.append(("ipa", ipa_str))
            last_idx = end

        if last_idx < len(normalized):
            segments.append(("text", normalized[last_idx:]))

        # Step 3b: the deterministic script guard. A span with explicit IPA needs
        # no G2P, so only the plain-text spans are checked — that keeps an
        # operator-supplied [word|/ipa/] usable while still refusing a Japanese
        # sentence sent to an English voice before it costs a render.
        ensure_text_is_readable(
            " ".join(chunk for kind, chunk in segments if kind == "text"), lang
        )

        # Step 4: Phonemize text chunks and preserve exact IPA chunks
        phoneme_parts: list[str] = []

        try:
            for kind, chunk in segments:
                if kind == "ipa":
                    clean_ipa = chunk.strip()
                    if clean_ipa:
                        phoneme_parts.append(clean_ipa)
                else:
                    if chunk.strip():
                        with G2P_LOCK:
                            p = self._tokenizer.phonemize(chunk.strip(), lang=lang)
                        if p:
                            phoneme_parts.append(p)

            raw = " ".join(phoneme_parts).strip()
            result = validate_phoneme_output(text, lang, raw)
            logger.debug("Phonetic result: '%s'", result[:60])
            return result
        except UnsupportedLanguageError:
            raise
        except Exception as e:  # noqa: BLE001 - a G2P failure is reported, never swallowed
            message = str(e)
            if "not supported by the espeak backend" in message:
                # The backend told us plainly that it has no voice for this code.
                raise UnsupportedLanguageError(lang, message) from e
            logger.error("Phonetic conversion failed: %s", e)
            raise PhoneticConversionError(message) from e

    def phonemize_raw(self, text: str, language: str = "en-us") -> str:
        """Phonemise one string *without* normalisation or marker cleanup.

        Diagnostics only: the language-capability probe needs to see espeak's raw
        output, markers included, to decide whether a language is usable. A render
        path must call :meth:`convert_to_phonetic` instead.
        """
        if not self.is_available:
            raise PhoneticConversionError("phonetic converter is not available")
        lang = engine_locale(language or "en-us")
        try:
            with G2P_LOCK:
                return (self._tokenizer.phonemize(text, lang=lang) or "").strip()
        except Exception as e:  # noqa: BLE001 - reported to the caller
            message = str(e)
            if "not supported by the espeak backend" in message:
                raise UnsupportedLanguageError(lang, message) from e
            raise PhoneticConversionError(message) from e


# Global singleton instance
phonetic_converter = PhoneticConverter()
