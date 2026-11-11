"""Golden corpus loading and validation.

The corpus is data, not code: ``app/eval/data/golden_corpus.json`` is reviewed like
any other source file and identified by a content digest that the report records.
That digest is what makes a report diffable — a regression against a baseline is
only meaningful when both were measured on the same sentences.

Every entry carries the four fields the ticket requires (``id``, ``text``,
``language``, ``use``) plus optional ``tags``, ``max_wer`` and ``notes``.
``max_wer`` is the per-entry quality target used by ``--fail-on-wer``; it is a
target, not a promise. Several multilingual entries deliberately exceed it today
(see SPEECH-010) and reporting that is the point of the harness.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

#: Corpus file format version. Bump when the meaning of a field changes.
SCHEMA_VERSION = 1

#: Location of the committed corpus, resolved relative to this package so the
#: harness works regardless of the process working directory.
DEFAULT_CORPUS_PATH = Path(__file__).resolve().parent / "data" / "golden_corpus.json"

#: Languages written without spaces between words. WER for these is computed over
#: characters because a whitespace token is not a word there.
UNSPACED_LANGUAGES: frozenset[str] = frozenset({"ja", "zh", "yue", "th", "lo", "my", "km"})

REQUIRED_FIELDS: tuple[str, ...] = ("id", "text", "language", "use")


class CorpusError(ValueError):
    """The corpus file is missing, malformed or violates the schema."""


@dataclass(frozen=True)
class CorpusEntry:
    """One golden sentence.

    ``language`` is an ISO 639-1 code (``en``, ``fr``, ``ja``, ``zh``, ``es``) so it
    can be compared with a voice's locale prefix and handed to Whisper unchanged.
    """

    id: str
    text: str
    language: str
    use: str
    tags: tuple[str, ...] = ()
    max_wer: float | None = None
    notes: str | None = None

    @property
    def is_unspaced(self) -> bool:
        """Whether the language is written without spaces between words."""
        return self.language in UNSPACED_LANGUAGES


@dataclass(frozen=True)
class GoldenCorpus:
    """A validated corpus plus the identity of the file it came from."""

    entries: tuple[CorpusEntry, ...]
    path: Path
    digest: str
    schema_version: int = SCHEMA_VERSION
    description: str = ""
    metrics_note: str = ""
    _by_id: dict[str, CorpusEntry] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        # ``frozen=True`` still allows populating a private cache in __post_init__.
        object.__setattr__(self, "_by_id", {entry.id: entry for entry in self.entries})

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[CorpusEntry]:
        return iter(self.entries)

    def get(self, entry_id: str) -> CorpusEntry:
        """Return one entry, raising ``KeyError`` with a helpful message."""
        try:
            return self._by_id[entry_id]
        except KeyError as exc:  # pragma: no cover - message only
            raise KeyError(
                f"corpus entry '{entry_id}' not found; ids: {', '.join(sorted(self._by_id))}"
            ) from exc

    @property
    def languages(self) -> list[str]:
        """Distinct languages, sorted."""
        return sorted({entry.language for entry in self.entries})

    def for_language(self, language: str) -> list[CorpusEntry]:
        """Entries for one language, in corpus order."""
        return [entry for entry in self.entries if entry.language == language]

    def filtered(
        self,
        *,
        languages: Iterable[str] | None = None,
        ids: Iterable[str] | None = None,
    ) -> list[CorpusEntry]:
        """Entries matching an optional language and id filter, in corpus order."""
        wanted_languages = {language for language in languages or ()}
        wanted_ids = {entry_id for entry_id in ids or ()}
        return [
            entry
            for entry in self.entries
            if (not wanted_languages or entry.language in wanted_languages)
            and (not wanted_ids or entry.id in wanted_ids)
        ]

    def uses(self) -> list[str]:
        """Distinct ``use`` values, sorted — the coverage checklist."""
        return sorted({entry.use for entry in self.entries})


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CorpusError(message)


def _parse_entry(payload: dict, index: int) -> CorpusEntry:
    missing = [name for name in REQUIRED_FIELDS if name not in payload]
    _require(not missing, f"corpus entry #{index} is missing required fields: {missing}")

    entry_id = payload["id"]
    _require(
        isinstance(entry_id, str) and bool(entry_id.strip()),
        f"corpus entry #{index} has an empty id",
    )
    text = payload["text"]
    _require(isinstance(text, str) and bool(text.strip()), f"entry '{entry_id}' has no text")
    language = payload["language"]
    _require(
        isinstance(language, str) and len(language) >= 2,
        f"entry '{entry_id}' has an invalid language {language!r}",
    )
    use = payload["use"]
    _require(isinstance(use, str) and bool(use.strip()), f"entry '{entry_id}' has no use")
    tags = payload.get("tags", [])
    _require(isinstance(tags, list), f"entry '{entry_id}' tags must be a list")
    max_wer = payload.get("max_wer")
    _require(
        max_wer is None or (isinstance(max_wer, (int, float)) and 0.0 <= float(max_wer) <= 1.0),
        f"entry '{entry_id}' max_wer must be between 0 and 1",
    )

    return CorpusEntry(
        id=entry_id,
        text=text.strip(),
        language=language.lower(),
        use=use,
        tags=tuple(str(tag) for tag in tags),
        max_wer=None if max_wer is None else float(max_wer),
        notes=payload.get("notes"),
    )


def load_corpus(path: str | Path | None = None) -> GoldenCorpus:
    """Load and validate the golden corpus.

    Raises :class:`CorpusError` for a missing file, invalid JSON, a missing
    required field, a duplicate id or an out-of-range ``max_wer``. Failing loudly
    is deliberate: a half-loaded corpus would silently shrink the measured matrix
    and make a regression look like an improvement.
    """
    corpus_path = Path(path) if path is not None else DEFAULT_CORPUS_PATH
    try:
        raw = corpus_path.read_bytes()
    except FileNotFoundError as exc:
        raise CorpusError(f"golden corpus not found: {corpus_path}") from exc
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CorpusError(f"golden corpus is not valid UTF-8 JSON: {exc}") from exc

    _require(isinstance(payload, dict), "golden corpus must be a JSON object")
    schema_version = payload.get("schema_version", SCHEMA_VERSION)
    _require(
        schema_version == SCHEMA_VERSION,
        f"unsupported corpus schema_version {schema_version!r}; expected {SCHEMA_VERSION}",
    )
    raw_entries = payload.get("entries")
    _require(
        isinstance(raw_entries, list) and bool(raw_entries),
        "golden corpus has no 'entries' list",
    )

    entries: list[CorpusEntry] = []
    seen: set[str] = set()
    for index, entry_payload in enumerate(raw_entries):
        _require(isinstance(entry_payload, dict), f"corpus entry #{index} is not an object")
        entry = _parse_entry(entry_payload, index)
        _require(entry.id not in seen, f"duplicate corpus entry id '{entry.id}'")
        seen.add(entry.id)
        entries.append(entry)

    return GoldenCorpus(
        entries=tuple(entries),
        path=corpus_path,
        digest="sha256:" + hashlib.sha256(raw).hexdigest(),
        schema_version=int(schema_version),
        description=str(payload.get("description", "")),
        metrics_note=str(payload.get("metrics_note", "")),
    )


__all__ = [
    "CorpusEntry",
    "CorpusError",
    "DEFAULT_CORPUS_PATH",
    "GoldenCorpus",
    "SCHEMA_VERSION",
    "UNSPACED_LANGUAGES",
    "load_corpus",
]
