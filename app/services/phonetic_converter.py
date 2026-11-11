"""Phonetic converter for text-to-speech synthesis.

Provides text normalization, pronunciation dictionary mapping, persistent custom lexicon,
per-request lexicon overrides, inline phonetic markup ([word|pronunciation] and <phoneme ph="ipa">),
and grapheme-to-phoneme (G2P) conversion using espeak-ng / phonemizer via Kokoro's Tokenizer.

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

logger = logging.getLogger(__name__)

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
            result[clean_word] = {"replacement": repl, "source": "builtin"}

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

    def set_entry(self, word: str, replacement: str | None = None, ipa: str | None = None) -> dict:
        """Add or update a custom lexicon entry.

        Copy-on-write: build the next mapping, persist it atomically, then
        publish it.  Concurrent renders keep iterating the mapping they already
        hold, so they can neither see a partial update nor trip
        ``RuntimeError: dictionary changed size during iteration``.
        """
        entry: dict[str, str] = {}
        if replacement:
            entry["replacement"] = replacement.strip()
        if ipa:
            entry["ipa"] = ipa.strip()

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
        self, text: str, custom_lexicon: dict[str, str] | None = None
    ) -> str:
        """Pre-process text for natural pronunciation before phonemization."""
        # Auto-reload if file was edited externally on disk
        self.load_lexicon()

        normalized = text

        # 1. Expand standard units & currency
        for pattern, replacement in self.units.items():
            normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)

        # 2. Expand common abbreviations
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

        # 5. Apply persistent custom lexicon replacements.
        # Iterate a snapshot: POST/DELETE /api/tts/lexicon mutate this mapping
        # from the event loop while this render runs in the inference pool.
        for word, info in self.lexicon_snapshot().items():
            repl = info.get("replacement")
            if repl:
                pat = rf"\b{re.escape(word)}\b"
                normalized = re.sub(pat, repl, normalized)

        # 6. Apply built-in domain & tech lexicon
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
        """
        # Step 1: Pre-normalize text (expanding units, abbreviations, respellings)
        normalized = self.normalize_text(text, custom_lexicon=custom_lexicon)

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

        # Check persistent custom lexicon with ipa (snapshot, see normalize_text)
        for word, info in self.lexicon_snapshot().items():
            ipa_val = info.get("ipa")
            if ipa_val:
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

        # Step 4: Phonemize text chunks and preserve exact IPA chunks
        lang = language.lower() if language else "en-us"
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

            result = " ".join(phoneme_parts).strip()
            logger.debug("Phonetic result: '%s'", result[:60])
            return result
        except Exception as e:  # noqa: BLE001 - a G2P failure is reported, never swallowed
            logger.error("Phonetic conversion failed: %s", e)
            raise PhoneticConversionError(str(e)) from e


# Global singleton instance
phonetic_converter = PhoneticConverter()
