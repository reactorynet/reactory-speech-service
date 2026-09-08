"""Phonetic converter for text-to-speech synthesis.

Provides text normalization, pronunciation dictionary mapping, persistent custom lexicon,
per-request lexicon overrides, inline phonetic markup ([word|pronunciation] and <phoneme ph="ipa">),
and grapheme-to-phoneme (G2P) conversion using espeak-ng / phonemizer via Kokoro's Tokenizer.
"""

import json
import logging
import re
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)


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

        # Custom persistent lexicon loaded from disk
        self._custom_lexicon: dict[str, dict] = {}
        self._last_mtime: float = 0.0
        self.load_lexicon()

    def _init_tokenizer(self) -> None:
        """Initialize the underlying Kokoro tokenizer."""
        try:
            from kokoro_onnx.tokenizer import Tokenizer
            self._tokenizer = Tokenizer()
            self._ready = True
            logger.info("PhoneticConverter initialized successfully with Kokoro Tokenizer.")
        except Exception as e:
            logger.warning("Could not initialize Kokoro Tokenizer: %s", e)
            self._ready = False

    @property
    def is_available(self) -> bool:
        """Check if phonetic conversion is available."""
        return self._ready and self._tokenizer is not None

    def load_lexicon(self) -> None:
        """Load persistent custom lexicon from disk (auto-reloads if modified on disk)."""
        lexicon_file = Path(settings.lexicon_file_path)
        if lexicon_file.exists():
            try:
                mtime = lexicon_file.stat().st_mtime
                if mtime != self._last_mtime:
                    with open(lexicon_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        self._custom_lexicon = data.get("entries", {})
                        self._last_mtime = mtime
                        logger.info("Loaded %d custom lexicon entries from %s", len(self._custom_lexicon), lexicon_file)
            except Exception as e:
                logger.warning("Failed to read custom lexicon from %s: %s", lexicon_file, e)
                if not self._custom_lexicon:
                    self._custom_lexicon = {}
        else:
            self._custom_lexicon = {}

    def save_lexicon(self) -> None:
        """Persist custom lexicon to disk."""
        lexicon_file = Path(settings.lexicon_file_path)
        lexicon_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(lexicon_file, "w", encoding="utf-8") as f:
                json.dump({"entries": self._custom_lexicon}, f, indent=2, ensure_ascii=False)
            logger.info("Saved %d custom lexicon entries to %s", len(self._custom_lexicon), lexicon_file)
        except Exception as e:
            logger.error("Failed to save custom lexicon to %s: %s", lexicon_file, e)

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
        for word, info in self._custom_lexicon.items():
            result[word] = {**info, "source": "custom"}

        return result

    def get_custom_entries(self) -> dict[str, dict]:
        """Return custom user-defined lexicon entries only."""
        self.load_lexicon()
        return self._custom_lexicon

    def get_entry(self, word: str) -> dict | None:
        """Get pronunciation entry for a word."""
        self.load_lexicon()
        if word in self._custom_lexicon:
            return {**self._custom_lexicon[word], "source": "custom"}
        raw_pat = rf"\b{re.escape(word)}\b"
        if raw_pat in self.builtin_lexicon:
            return {"replacement": self.builtin_lexicon[raw_pat], "source": "builtin"}
        return None

    def set_entry(self, word: str, replacement: str | None = None, ipa: str | None = None) -> dict:
        """Add or update a custom lexicon entry."""
        entry: dict[str, str] = {}
        if replacement:
            entry["replacement"] = replacement.strip()
        if ipa:
            entry["ipa"] = ipa.strip()

        self._custom_lexicon[word] = entry
        self.save_lexicon()
        return {**entry, "word": word, "source": "custom"}

    def delete_entry(self, word: str) -> bool:
        """Delete a custom lexicon entry."""
        if word in self._custom_lexicon:
            del self._custom_lexicon[word]
            self.save_lexicon()
            return True
        return False

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

        # 5. Apply persistent custom lexicon replacements
        for word, info in self._custom_lexicon.items():
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

        # Check persistent custom lexicon with ipa
        for word, info in self._custom_lexicon.items():
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
                        p = self._tokenizer.phonemize(chunk.strip(), lang=lang)
                        if p:
                            phoneme_parts.append(p)

            result = " ".join(phoneme_parts).strip()
            logger.debug("Phonetic result: '%s'", result[:60])
            return result
        except Exception as e:
            logger.warning("Phonetic conversion failed: %s", e)
            return normalized


# Global singleton instance
phonetic_converter = PhoneticConverter()
