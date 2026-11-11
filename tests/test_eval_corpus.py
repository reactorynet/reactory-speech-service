"""The golden corpus must stay representative and runnable.

These tests are the guard rails on the measurement itself: if the corpus loses a
language, a use case or a domain term, every number the harness produces becomes
less meaningful without any test failing. So the corpus's coverage is asserted, not
assumed.
"""

from __future__ import annotations

import json

import pytest

from app.eval.corpus import CorpusError, load_corpus
from app.eval.runner import voice_language
from app.services import voice_registry

#: Use cases the ticket requires the corpus to cover.
REQUIRED_USES = (
    "short greeting",
    "long narration",
    "domain vocabulary",
    "numbers",
    "dates",
    "abbreviations",
)

#: Domain vocabulary the ticket names explicitly.
REQUIRED_TERMS = ("Reactory", "GraphQL", "Kubernetes", "ONNX", "FQDN")


def test_default_corpus_covers_the_ticket_scope() -> None:
    corpus = load_corpus()

    assert len(corpus) >= 20
    for use in REQUIRED_USES:
        assert use in corpus.uses(), f"corpus has no '{use}' entry"

    non_english = [language for language in corpus.languages if language != "en"]
    assert len(non_english) >= 2, f"corpus needs two non-English languages, has {non_english}"

    joined = " ".join(entry.text for entry in corpus)
    for term in REQUIRED_TERMS:
        assert term in joined, f"corpus never says '{term}'"


def test_every_entry_is_complete_and_unique() -> None:
    corpus = load_corpus()
    ids = [entry.id for entry in corpus]
    assert len(set(ids)) == len(ids)

    for entry in corpus:
        assert entry.id
        assert entry.text.strip()
        assert entry.language
        assert entry.use
        assert entry.max_wer is None or 0.0 <= entry.max_wer <= 1.0


def test_every_corpus_language_has_a_shipped_voice() -> None:
    """A language in the corpus with no voice is an unmeasurable sentence."""
    corpus = load_corpus()
    descriptors = [
        record.descriptor() for record in voice_registry.VOICE_REGISTRY.for_engine("kokoro")
    ]
    voice_languages = {voice_language(descriptor) for descriptor in descriptors}
    missing = sorted(set(corpus.languages) - voice_languages)
    assert not missing, f"corpus languages with no Kokoro voice: {missing}"


def test_corpus_lookup_helpers() -> None:
    corpus = load_corpus()
    english = corpus.for_language("en")
    assert english and all(entry.language == "en" for entry in english)
    assert corpus.get(english[0].id) is english[0]
    assert corpus.filtered(languages=["fr"]) == corpus.for_language("fr")
    assert corpus.filtered(ids=[english[0].id]) == [english[0]]
    with pytest.raises(KeyError):
        corpus.get("does-not-exist")


def test_digest_is_content_addressed(tmp_path) -> None:
    payload = json.loads(load_corpus().path.read_text(encoding="utf-8"))
    first = tmp_path / "corpus-a.json"
    second = tmp_path / "corpus-b.json"
    first.write_text(json.dumps(payload), encoding="utf-8")
    payload["entries"][0]["text"] += " Changed."
    second.write_text(json.dumps(payload), encoding="utf-8")

    assert load_corpus(first).digest == load_corpus(first).digest
    assert load_corpus(first).digest != load_corpus(second).digest


def test_duplicate_ids_are_rejected(tmp_path) -> None:
    payload = json.loads(load_corpus().path.read_text(encoding="utf-8"))
    payload["entries"][1]["id"] = payload["entries"][0]["id"]
    path = tmp_path / "duplicate.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CorpusError, match="duplicate"):
        load_corpus(path)


def test_missing_required_field_is_rejected(tmp_path) -> None:
    payload = json.loads(load_corpus().path.read_text(encoding="utf-8"))
    del payload["entries"][0]["language"]
    path = tmp_path / "missing.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CorpusError, match="missing required fields"):
        load_corpus(path)


def test_out_of_range_max_wer_is_rejected(tmp_path) -> None:
    payload = json.loads(load_corpus().path.read_text(encoding="utf-8"))
    payload["entries"][0]["max_wer"] = 1.5
    path = tmp_path / "max-wer.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CorpusError, match="max_wer"):
        load_corpus(path)


def test_missing_corpus_file_is_reported(tmp_path) -> None:
    with pytest.raises(CorpusError, match="not found"):
        load_corpus(tmp_path / "absent.json")
