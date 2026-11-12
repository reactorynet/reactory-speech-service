"""SPEECH-010 review revision 3: the quality signal, unselected evidence, guard scope.

Three things this module pins, all without models so the not-slow CI job runs them:

* **Quality is advertised, not just support.** Review finding F1: Mandarin was
  listed as a clean "yes" although a committed corpus sentence scores CER 0.92.
  ``language-quality.json`` now drives the catalogue and the capabilities response,
  so a language whose sentences do not all pass is reported as ``caveats`` with
  the failing entry and the measured figure named.
* **The record is the single source of truth.** The test reads the same file the
  docs table was generated from, so the catalogue cannot drift from the numbers.
* **The script guard does not refuse readable text** (review finding F2): Greek
  letters and mathematical symbols are ordinary developer copy and espeak reads
  them, so they must not become a 422.
* **The Hindi script prompt is a service behaviour**, not a harness-only trick
  (review finding F4).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config import settings
from app.engines.kokoro import KokoroEngine
from app.services.g2p_capability import (
    QUALITY_CAVEATS,
    QUALITY_PRODUCTION,
    QUALITY_RECORD_PATH,
    QUALITY_UNSUPPORTED,
    QUALITY_UNVERIFIED,
    PROBE_SENTENCES,
    g2p_capability,
    load_quality_record,
    reset_quality_cache,
)
from app.services.phonetic_converter import (
    UnsupportedLanguageError,
    phonetic_converter,
    unreadable_script,
)
from app.services.stt_service import SCRIPT_PROMPTS

CATALOGUE_DOC = Path(__file__).resolve().parent.parent / "docs" / "voices" / "kokoro-voice-catalogue.md"

#: Languages the committed corpus shows at least one failing sentence for. Pinned
#: so a corpus/model change that fixes them is noticed (the caveat is then stale
#: and must be removed deliberately, not silently).
KNOWN_CAVEAT_LANGUAGES = {"fr", "zh"}


# ---------------------------------------------------------------------------
# The committed record: present, well-formed, and honest by construction
# ---------------------------------------------------------------------------


def test_the_quality_record_is_committed_and_complete():
    record = load_quality_record()
    assert record, f"missing {QUALITY_RECORD_PATH}"

    raw = json.loads(QUALITY_RECORD_PATH.read_text(encoding="utf-8"))
    assert raw["corpus"]["entries"] > 1, "the record must cover a corpus, not one sentence"
    assert "generated_by" in raw and "record_language_quality" in raw["generated_by"]

    for language, row in record.items():
        assert row["quality"] in {
            QUALITY_PRODUCTION,
            QUALITY_CAVEATS,
            QUALITY_UNSUPPORTED,
        }, f"{language}: {row['quality']}"
        if row["quality"] == QUALITY_CAVEATS:
            assert row["entries_over_target"], f"{language}: caveats with no failing entry"
            assert row["worst_cer"] is not None
        if row["quality"] == QUALITY_PRODUCTION:
            assert row["entries_over_target"] == []


def test_the_known_caveats_are_recorded():
    """F1's evidence must survive in the committed artifact, not only in a report."""
    record = load_quality_record()
    assert set(record) >= KNOWN_CAVEAT_LANGUAGES | {"en", "ja"}

    chinese = record["zh"]
    assert chinese["quality"] == QUALITY_CAVEATS
    assert "zh-narration-long" in chinese["entries_over_target"]
    assert chinese["worst_cer"] > 0.5

    japanese = record["ja"]
    assert japanese["quality"] == QUALITY_UNSUPPORTED


def test_english_and_spanish_are_production():
    record = load_quality_record()
    assert record["en"]["quality"] == QUALITY_PRODUCTION
    assert record["es"]["quality"] == QUALITY_PRODUCTION


# ---------------------------------------------------------------------------
# The record reaches the capability response
# ---------------------------------------------------------------------------


def test_capability_rows_carry_quality_and_measured_figures():
    g2p_capability.report()  # warm: this test is about reporting, not probing cost
    capabilities = KokoroEngine().capabilities()

    chinese = next(
        row for row in capabilities.language_support if row["language"] == "zh"
    )
    assert chinese["supported"] is True
    assert chinese["quality"] == QUALITY_CAVEATS
    assert chinese["caveat"] and "zh-narration-long" in chinese["caveat"]
    assert chinese["measured"]["worst_cer"] > 0.5

    # The caveat list is the machine-readable form of the same fact.
    caveat_languages = {
        (row["language"] or "").split("-")[0]
        for row in capabilities.languages_with_caveats
    }
    assert "zh" in caveat_languages
    assert "fr" in caveat_languages

    # And a language with real coverage is not dragged into the caveat list.
    assert "es" not in caveat_languages
    assert "en" not in caveat_languages


def test_unsupported_language_is_classified_unsupported():
    g2p_capability.report()
    capabilities = KokoroEngine().capabilities()

    japanese = next(
        row for row in capabilities.language_support if row["language"] == "ja"
    )
    assert japanese["supported"] is False
    assert japanese["quality"] == QUALITY_UNSUPPORTED
    assert "ja" in capabilities.unsupported_languages


def test_a_language_with_no_record_is_unverified_not_production(monkeypatch):
    """An absent measurement must never be advertised as production quality."""
    from app.services import g2p_capability as module

    monkeypatch.setattr(module, "QUALITY_RECORD_PATH", Path("/nonexistent/record.json"))
    reset_quality_cache()
    monkeypatch.setattr(g2p_capability, "_results", {})
    monkeypatch.setattr(g2p_capability, "_last_report", {})
    try:
        support = g2p_capability.probe("es")
        assert support.supported is True
        assert support.quality == QUALITY_UNVERIFIED
        assert support.caveat and "not been measured" in support.caveat
    finally:
        reset_quality_cache()


# ---------------------------------------------------------------------------
# F3: capabilities() never pays a dictionary load
# ---------------------------------------------------------------------------


def test_capabilities_is_non_blocking_and_names_pending_languages(monkeypatch):
    """A cold ``capabilities()`` must not probe, and must not advertise the unprobed.

    Review finding F3: the first synthesis of a process paid 13.45 s because
    ``lower()`` reads ``capabilities()``, which used to run the whole probe. The
    cold answer now says "pending" instead of guessing.
    """
    calls: list[bool] = []

    def _boom(*args, **kwargs):  # pragma: no cover - must not be called
        calls.append(True)
        raise AssertionError("capabilities() ran the probe")

    monkeypatch.setattr(g2p_capability, "probe", _boom)
    monkeypatch.setattr(g2p_capability, "_results", {})
    monkeypatch.setattr(g2p_capability, "_last_report", {})

    capabilities = KokoroEngine().capabilities()

    assert calls == []
    assert capabilities.languages == [], "an unprobed language must not be advertised"
    assert capabilities.unsupported_languages == []
    assert len(capabilities.languages_pending) >= 8
    assert "ja" in capabilities.languages_pending


def test_report_block_false_returns_the_last_completed_report():
    g2p_capability.report()  # complete a report
    warm = g2p_capability.report(block=False)
    assert warm, "a completed report must be cached for non-blocking reads"
    assert all(row.language in PROBE_SENTENCES for row in warm.values())


# ---------------------------------------------------------------------------
# F2: the script guard must not refuse text espeak reads
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "The value of π is 3.14",
        "Use Δ for difference",
        "Alpha Ω beta function",
        "The sum ∑ of the series",
        "A square root √ symbol",
        "Option Д",
        "Café au lait",
        "Hi Werner — how can I help?",
    ],
)
def test_readable_text_is_not_refused_for_an_english_voice(text):
    """Review finding F2: these used to synthesise on master and must again."""
    assert phonetic_converter.convert_to_phonetic(text, language="en-us")


@pytest.mark.parametrize(
    "text",
    [
        "リアクトリーの音声サービスはとても便利です",
        "你好，今天我能帮你做什么？",
        "Привет, Werner",
        "Hello 世界",
    ],
)
def test_genuinely_unreadable_scripts_are_still_refused(text):
    """The guard must keep catching what espeak spells out letter by letter."""
    with pytest.raises(UnsupportedLanguageError):
        phonetic_converter.convert_to_phonetic(text, language="en-us")


def test_the_script_guard_reports_nothing_for_a_latin_sentence():
    assert unreadable_script("The value of π is 3.14", "en-us") is None
    assert unreadable_script("Привет", "en-us") == "cyrillic"
    assert unreadable_script("你好", "en-us") == "han"


# ---------------------------------------------------------------------------
# F4: the Hindi script prompt ships in the service
# ---------------------------------------------------------------------------


def test_script_prompts_are_defined_in_the_service():
    assert set(SCRIPT_PROMPTS) == {"hi", "zh"}


class _StopTranscription(Exception):
    """Raised by the fake to stop ``transcribe`` once the arguments are captured."""


def _capture_script_prompt(monkeypatch, service):
    """Capture the ``initial_prompt`` argument ``transcribe`` passes down."""
    captured: dict = {}

    def _fake_transcription(*args, **kwargs):
        # ``transcribe`` passes (audio, language, model, word_timestamps, vad,
        # temperature, initial_prompt) positionally.
        captured["initial_prompt"] = args[6] if len(args) > 6 else kwargs.get("initial_prompt")
        raise _StopTranscription

    monkeypatch.setattr(service, "_transcription", _fake_transcription)
    return captured


def test_stt_transcribe_applies_the_script_prompt_by_default(monkeypatch):
    """A plain (API-shaped) call gets Devanagari without the caller asking."""
    from app.services.stt_service import STTService

    monkeypatch.setattr(settings, "whisper_script_prompts_enabled", True)
    service = STTService()
    captured = _capture_script_prompt(monkeypatch, service)

    with pytest.raises(_StopTranscription):
        service.transcribe(b"wav", language="hi")

    assert captured["initial_prompt"] == SCRIPT_PROMPTS["hi"]


def test_stt_script_prompt_can_be_disabled(monkeypatch):
    from app.services.stt_service import STTService

    monkeypatch.setattr(settings, "whisper_script_prompts_enabled", False)
    service = STTService()
    captured = _capture_script_prompt(monkeypatch, service)

    with pytest.raises(_StopTranscription):
        service.transcribe(b"wav", language="hi")

    assert captured["initial_prompt"] is None


def test_an_explicit_prompt_wins_over_the_default(monkeypatch):
    from app.services.stt_service import STTService

    monkeypatch.setattr(settings, "whisper_script_prompts_enabled", True)
    service = STTService()
    captured = _capture_script_prompt(monkeypatch, service)

    with pytest.raises(_StopTranscription):
        service.transcribe(b"wav", language="hi", initial_prompt="explicit")

    assert captured["initial_prompt"] == "explicit"


def test_a_region_tag_still_selects_the_script_prompt(monkeypatch):
    from app.services.stt_service import STTService

    monkeypatch.setattr(settings, "whisper_script_prompts_enabled", True)
    service = STTService()
    captured = _capture_script_prompt(monkeypatch, service)

    with pytest.raises(_StopTranscription):
        service.transcribe(b"wav", language="zh-CN")

    assert captured["initial_prompt"] == SCRIPT_PROMPTS["zh"]


# ---------------------------------------------------------------------------
# The catalogue documents what the record measures
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not CATALOGUE_DOC.exists(), reason="catalogue doc is not on this branch")
def test_catalogue_states_the_mandarin_caveat():
    """F1's documentation half: the table must not say a bare 'yes' for zh."""
    text = CATALOGUE_DOC.read_text(encoding="utf-8")
    row = next(
        line
        for line in text.splitlines()
        if line.startswith("| Mandarin Chinese") and "cmn" in line
    )
    assert "caveats" in row.lower(), row
    assert "0.92" in row or "0.9231" in row, row
    assert "usable with caveats" in text.lower()
    # The old, rejected wording must not come back.
    assert "Chinese text returned; no espeak error" not in text


@pytest.mark.skipif(not CATALOGUE_DOC.exists(), reason="catalogue doc is not on this branch")
def test_catalogue_numbers_match_the_committed_record():
    """Every documented worst CER/WER must appear in the record it came from."""
    record = load_quality_record()
    text = CATALOGUE_DOC.read_text(encoding="utf-8")

    for code, row in record.items():
        if row["quality"] == QUALITY_UNSUPPORTED:
            continue
        worst_cer = row.get("worst_cer")
        if worst_cer is None:
            continue
        assert str(worst_cer) in text, (
            f"{code}: worst CER {worst_cer} is measured but not documented"
        )


def test_a_thinly_measured_language_says_so():
    """Passing one sentence is a smoke test, and the caveat says exactly that."""
    g2p_capability.report()
    capabilities = KokoroEngine().capabilities()

    hindi = next(
        row for row in capabilities.language_support if row["language"] == "hi"
    )
    assert hindi["supported"] is True
    assert hindi["quality"] == QUALITY_PRODUCTION
    assert hindi["caveat"], "a one-sentence language must not read as broad coverage"
    assert "small sample" in hindi["caveat"]
    assert hindi["measured"]["entries_measured"] == 1

    # English has real coverage, so it carries no such caveat.
    english = next(
        row for row in capabilities.language_support if row["language"] == "en-US"
    )
    assert english["measured"]["entries_measured"] > 10
    assert english["caveat"] is None
