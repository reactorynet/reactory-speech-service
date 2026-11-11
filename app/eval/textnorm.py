"""Text normalisation and tokenisation for round-trip scoring.

Whisper does not return the input verbatim: it drops punctuation, changes case,
spells numbers differently ("1,284.50" -> "1284.50", "09:30" -> "9.30") and
inserts or omits filler words. Scoring a transcript without normalising both
sides measures formatting, not intelligibility.

Normalisation is deliberately conservative: Unicode NFKC, case folding,
punctuation stripped, whitespace collapsed. Nothing is *expanded* (no number to
words, no abbreviation to phrase) because that would need a language-specific
normaliser for every corpus language and would hide real mispronunciations.

Tokenisation is language-aware:

* space-delimited languages -> whitespace tokens (words);
* Japanese / Chinese / Thai -> one token per non-punctuation character, because
  those languages do not separate words with spaces. WER for them is therefore a
  character error rate; the report says so in ``meta.metric_definitions`` and the
  ``cer`` column remains the primary number.
"""

from __future__ import annotations

import re
import unicodedata

from app.eval.corpus import UNSPACED_LANGUAGES

#: Zero-width and bidi control characters that leak out of some TTS front-ends.
_CONTROL_CHARS = dict.fromkeys(
    map(ord, "\u200b\u200c\u200d\u2060\ufeff\u202a\u202b\u202c\u202d\u202e"), None
)

#: Everything Unicode classifies as punctuation or a symbol boundary.
_PUNCTUATION = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WHITESPACE = re.compile(r"\s+", flags=re.UNICODE)

#: Characters that are never spoken and therefore never scored.
_NON_SPEECH = re.compile(r"[\u3000\u3001\u3002\uff0c\uff1f\uff01\uff1b\uff1a\u300c\u300d\u300e\u300f]")


def normalise(text: str) -> str:
    """Fold a string into a comparable form.

    Keeps letters, digits and marks; drops punctuation, symbols and control
    characters; collapses whitespace; case-folds. The result is what both the
    reference and the hypothesis are scored on.
    """
    if not text:
        return ""
    folded = unicodedata.normalize("NFKC", text).translate(_CONTROL_CHARS)
    folded = folded.casefold()
    folded = _NON_SPEECH.sub(" ", folded)
    folded = _PUNCTUATION.sub(" ", folded)
    # ``\w`` keeps underscore; it is a symbol in prose, not a letter.
    folded = folded.replace("_", " ")
    return _WHITESPACE.sub(" ", folded).strip()


def tokenize(text: str, language: str | None = None) -> list[str]:
    """Split normalised text into scoring tokens.

    For :data:`app.eval.corpus.UNSPACED_LANGUAGES` the split is per character, so
    ``tokenize("你好世界", "zh")`` has four tokens. Everything else splits on
    whitespace.
    """
    normalised = normalise(text)
    if not normalised:
        return []
    if (language or "").lower() in UNSPACED_LANGUAGES:
        return [character for character in normalised if not character.isspace()]
    return normalised.split()


def characters(text: str) -> list[str]:
    """Character tokens with whitespace removed, for CER."""
    return [character for character in normalise(text) if not character.isspace()]


def spoken_units(text: str, language: str | None = None) -> int:
    """Number of tokens a listener would hear, used by the duration model.

    This is the same count as :func:`tokenize` but without requiring the caller to
    build the list.
    """
    return len(tokenize(text, language))


__all__ = ["characters", "normalise", "spoken_units", "tokenize"]
