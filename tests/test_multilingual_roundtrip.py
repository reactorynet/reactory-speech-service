"""SPEECH-010 acceptance: per-language round-trip intelligibility, unselected.

This runs the **committed** golden corpus — every entry, every representative
voice — through the shipped ``app/eval`` harness: synthesise with the real Kokoro
voice, transcribe with the local faster-whisper model, compare to the source. A
valid WAV is not evidence of anything; the ticket's Japanese case was 773 KB of
audio that said nothing.

Why the assertions look the way they do
---------------------------------------
Review finding F1 rejected an earlier revision of this file because it measured a
hand-picked subset (one short greeting for Mandarin, and the French greeting
replaced by the numbers sentence that scored better). That produced a reassuring
"0 quality failures" that the committed corpus does not support:
``zh-narration-long`` scores CER 0.92 with ``zf_xiaoxiao``.

So this module no longer asserts "every sentence is under target" — that is
false, and asserting it would only be passable by selecting sentences. It asserts
the thing that actually matters for the ticket: **the published evidence matches
the measurement**. Concretely:

* every entry the corpus has for a language is measured (nothing is skipped);
* the measured figures equal ``app/eval/data/language-quality.json``, which is
  what the catalogue table and ``/api/tts/capabilities`` report;
* a language with a failing sentence is classified ``caveats`` and named with its
  measured CER, instead of being advertised as a clean "yes";
* a language with no failing sentence is classified ``production``;
* Japanese is skipped with a measured reason rather than measured as nonsense.

Adding a sentence to the corpus therefore either passes (and stays
``production``) or fails (and forces the language into ``caveats`` everywhere it
is documented). There is no third option in which it quietly disappears.

Requires the real model files (``make download-models``): the Kokoro pack and a
mounted faster-whisper model. Without them the module skips with the reason, so a
machine without models cannot report a false pass.
"""

from __future__ import annotations

import json

import pytest

from app.config import settings
from app.eval.corpus import load_corpus
from app.eval.runner import EvaluationRunner, RunnerConfig
from app.eval.stt import LocalTranscriber
from app.services.g2p_capability import (
    QUALITY_CAVEATS,
    QUALITY_PRODUCTION,
    QUALITY_RECORD_PATH,
    g2p_capability,
    load_quality_record,
)
from app.services.stt_service import STTService
from app.services.tts_service import TTSService

pytestmark = pytest.mark.slow

MODEL_PATH = settings.kokoro_model_resolved
VOICES_PATH = settings.kokoro_voices_resolved

#: Explicit voices, one per language family, plus both Chinese voices the ticket
#: names in its acceptance criteria. Explicit rather than "best graded" so the
#: acceptance case cannot drift to a different voice when grades are re-tuned.
REPRESENTATIVE_VOICES = (
    "af_heart",  # en-US
    "bf_emma",  # en-GB
    "ef_dora",  # es
    "ff_siwis",  # fr-FR
    "hf_alpha",  # hi
    "if_sara",  # it
    "pf_dora",  # pt-BR
    "zf_xiaoxiao",  # zh
    "zm_yunjian",  # zh, male
    # Listed deliberately so the harness has to *decide* about Japanese: it must
    # be recorded as skipped-with-a-reason, never measured and never dropped.
    "jf_alpha",
)

#: A render may not take more than this multiple of the expected duration for its
#: text. The Japanese defect produced ~8x (16.11 s for a short sentence), so the
#: bound is the signal, not a quality judgement about pacing.
MAX_DURATION_RATIO = 2.2
MIN_DURATION_RATIO = 0.4

#: How far a live measurement may drift from the committed record before the
#: record is stale. Deterministic decoding (temperature 0) makes this tight; the
#: margin only absorbs model/CPU nondeterminism.
RECORD_TOLERANCE = 0.02


def _models_mounted() -> bool:
    return MODEL_PATH.exists() and VOICES_PATH.exists()


@pytest.fixture(scope="module")
def measured():
    """Measure the whole committed corpus once; return cells plus the skips."""
    pytest.importorskip("faster_whisper", reason="faster-whisper is not installed")
    if not _models_mounted():
        pytest.skip(
            "Kokoro model files are not present in models/; run `make download-models`"
        )

    import asyncio

    tts = TTSService()
    stt = STTService()
    asyncio.run(tts.initialize())
    asyncio.run(stt.initialize())

    if not stt.available_models:
        pytest.skip(
            "no faster-whisper model is mounted under models/ (expected whisper-<size>/); "
            "the round-trip measurement cannot be faked"
        )

    corpus = load_corpus()
    engine = tts.engine_manager.get("kokoro")
    runner = EvaluationRunner(
        corpus,
        [engine],
        LocalTranscriber(stt),
        config=RunnerConfig(repeats=1, probe_rate=1.5),
        voices_spec=",".join(REPRESENTATIVE_VOICES),
    )
    cells = runner.run()
    return {
        "corpus": corpus,
        "cells": cells,
        "skipped": list(runner.skipped),
        "supported": g2p_capability.supported_languages(),
        "record": load_quality_record(),
    }


def _cells_by_language(measured):
    grouped: dict[str, list] = {}
    for cell in measured["cells"]:
        grouped.setdefault(cell.language, []).append(cell)
    return grouped


# ---------------------------------------------------------------------------
# Nothing is skipped: the measurement covers every corpus sentence
# ---------------------------------------------------------------------------


def test_every_corpus_entry_for_a_language_is_measured(measured):
    """No sentence may be dropped from the acceptance measurement.

    This is the direct guard against review finding F1: a subset cannot be
    presented as the language's quality.
    """
    corpus = measured["corpus"]
    measured_ids = {
        item["id"] for cell in measured["cells"] for item in cell.entries
    }
    expected: set[str] = set()
    for language in {cell.language for cell in measured["cells"]}:
        expected.update(entry.id for entry in corpus.for_language(language))

    missing = expected - measured_ids
    assert not missing, f"corpus sentences were not measured: {sorted(missing)}"


def test_advertised_languages_all_have_a_measured_cell(measured):
    """Every language capabilities advertises was actually measured here."""
    measured_languages = {cell.language for cell in measured["cells"]}
    advertised = {(tag or "").split("-")[0] for tag in measured["supported"]}
    missing = advertised - measured_languages
    assert not missing, (
        f"advertised languages with no measured cell: {sorted(missing)} "
        f"(measured: {sorted(measured_languages)})"
    )


def test_no_render_is_structurally_broken(measured):
    """Silence, undecodable audio or a missing transcript is a failure, not a WER."""
    problems = []
    for cell in measured["cells"]:
        for item in cell.entries:
            if item["status"] != "ok":
                problems.append(f"{cell.voice}/{item['id']}: {item['status']}")
            elif "silent_output" in item["checks"]:
                problems.append(f"{cell.voice}/{item['id']}: silent")
            elif item["wer"] is None:
                problems.append(f"{cell.voice}/{item['id']}: no transcript")
    assert not problems, "\n".join(problems)


def test_duration_stays_proportional_to_the_sentence(measured):
    """The Japanese signal: 21 characters must not produce 16 seconds of audio."""
    offenders = []
    for cell in measured["cells"]:
        for item in cell.entries:
            ratio = item["duration_ratio"]
            if ratio is None:
                offenders.append(f"{cell.voice}/{item['id']}: no duration")
            elif not MIN_DURATION_RATIO <= ratio <= MAX_DURATION_RATIO:
                offenders.append(
                    f"{cell.voice}/{item['id']}: {item['duration_s']:.2f}s = {ratio:.2f}x"
                )
    assert not offenders, "\n".join(offenders)


# ---------------------------------------------------------------------------
# The published evidence equals the measurement — this is the real assertion
# ---------------------------------------------------------------------------


def test_measured_quality_matches_the_committed_record(measured):
    """The record that docs and capabilities read must match a live run.

    If this fails, either the corpus changed or the models did: regenerate the
    record with ``scripts/record_language_quality.py`` and update the catalogue.
    It must never be "fixed" by relaxing the assertion.
    """
    record = measured["record"]
    assert record, f"missing quality record at {QUALITY_RECORD_PATH}"

    grouped = _cells_by_language(measured)
    problems = []
    for language, cells in grouped.items():
        row = record.get(language)
        if row is None:
            problems.append(f"{language}: measured but absent from the record")
            continue
        live_wer = max(
            (item["wer"] for cell in cells for item in cell.entries if item["wer"] is not None),
            default=0.0,
        )
        live_cer = max(
            (item["cer"] for cell in cells for item in cell.entries if item["cer"] is not None),
            default=0.0,
        )
        recorded_wer = row.get("worst_wer")
        recorded_cer = row.get("worst_cer")
        if recorded_wer is None or abs(live_wer - recorded_wer) > RECORD_TOLERANCE + 1e-9:
            problems.append(
                f"{language}: worst WER measured {live_wer:.4f} != recorded {recorded_wer}"
            )
        if recorded_cer is None or abs(live_cer - recorded_cer) > RECORD_TOLERANCE + 1e-9:
            problems.append(
                f"{language}: worst CER measured {live_cer:.4f} != recorded {recorded_cer}"
            )
        live_failing = sorted(
            {
                item["id"]
                for cell in cells
                for item in cell.entries
                if "wer_threshold" in item["checks"]
            }
        )
        recorded_failing = sorted(row.get("entries_over_target") or [])
        if live_failing != recorded_failing:
            problems.append(
                f"{language}: failing entries measured {live_failing} != recorded "
                f"{recorded_failing}"
            )
    assert not problems, "\n".join(problems)
    assert json.dumps(record, sort_keys=True)  # the record is serialisable and stable


def test_each_language_is_classified_by_its_measured_evidence(measured):
    """``production`` means no sentence failed; ``caveats`` names the one that did.

    This is the rule that makes "advertised without evidence" impossible: the
    classification is derived from the failing-entry list, so it cannot be typed
    in by hand.
    """
    record = measured["record"]
    for language, row in record.items():
        quality = row.get("quality")
        failing = list(row.get("entries_over_target") or [])
        if quality == QUALITY_PRODUCTION:
            assert failing == [], f"{language} is production but has failures {failing}"
        elif quality == QUALITY_CAVEATS:
            assert failing, f"{language} is caveats but names no failing entry"
        else:
            assert quality not in (None, ""), f"{language} has no classification"


def test_the_known_hard_sentences_are_recorded_not_hidden(measured):
    """The three sentences that failed the reviewer's run are measured and named.

    Pinning them individually makes it obvious if a future edit quietly replaces
    the corpus sentence or drops a voice, and documents in the suite itself that
    these results are accepted limitations rather than unnoticed failures.
    """
    record = measured["record"]

    chinese = record["zh"]
    assert "zh-narration-long" in (chinese.get("entries_over_target") or [])
    assert chinese["quality"] == QUALITY_CAVEATS
    assert chinese["worst_cer"] > 0.5, (
        "the Chinese narration sentence measured under target; the caveat recorded in "
        "the catalogue is now stale and should be removed"
    )

    french = record["fr"]
    assert "fr-greeting-short" in (french.get("entries_over_target") or [])
    assert french["quality"] == QUALITY_CAVEATS


def test_japanese_is_skipped_with_a_reason_not_measured(measured):
    """Japanese cannot be phonemised here, so it must not be measured as nonsense."""
    assert all(cell.language != "ja" for cell in measured["cells"]), (
        "a Japanese cell was measured; the G2P guard did not fire"
    )
    japanese = [row for row in measured["skipped"] if row["language"] == "ja"]
    assert japanese, f"Japanese was dropped silently: skipped={measured['skipped']}"
    assert "spells" in (japanese[0]["reason"] or "").lower()
    assert "ja" not in measured["supported"]


# ---------------------------------------------------------------------------
# Script checks that the record cannot express
# ---------------------------------------------------------------------------


def test_chinese_returns_chinese_text(measured):
    """Both acceptance voices must come back as Han text (not an empty transcript)."""
    import re

    han = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
    for voice_id in ("zf_xiaoxiao", "zm_yunjian"):
        cell = next((c for c in measured["cells"] if c.voice == voice_id), None)
        assert cell is not None, f"{voice_id} produced no measurement cell"
        greeting = next(e for e in cell.entries if e["id"] == "zh-greeting-short")
        assert greeting["transcript"] and han.search(greeting["transcript"]), (
            f"{voice_id}: no Han characters in {greeting['transcript']!r}"
        )


def test_hindi_returns_devanagari(measured):
    """The Devanagari guarantee now holds in the service, not just the harness.

    ``SCRIPT_PROMPTS`` moved into ``STTService`` (review finding F4), so
    ``/api/stt/transcribe`` and this harness return the same script.
    """
    import re

    devanagari = re.compile(r"[\u0900-\u097f]")
    cell = next((c for c in measured["cells"] if c.voice == "hf_alpha"), None)
    assert cell is not None, "hf_alpha produced no measurement cell"
    item = cell.entries[0]
    assert item["transcript"] and devanagari.search(item["transcript"]), (
        f"expected Devanagari, got {item['transcript']!r}"
    )
