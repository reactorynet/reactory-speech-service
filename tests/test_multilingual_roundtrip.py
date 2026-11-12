"""SPEECH-010 acceptance: per-language round-trip intelligibility.

Synthesise a sentence with the real Kokoro voice for one language family, then
transcribe it with the local faster-whisper model and score the transcript
against the source text. This is exactly the technique that found the ticket: a
valid WAV is not evidence of anything — the ticket's Japanese sample was 773 KB
of perfectly valid silence-ish nonsense. The words that come back are the
evidence.

The tests deliberately drive the **shipped evaluation harness** (``app/eval``:
the golden corpus, the runner, the round-trip metrics and the local transcriber)
rather than a second harness of their own. SPEECH-007 owns that machinery; this
file only supplies the per-language regression cases and the acceptance
thresholds.

Requires the real model files (``make download-models``): the Kokoro pack and a
mounted faster-whisper model. When either is absent the module skips with the
reason, so a machine without models cannot report a false pass.
"""

from __future__ import annotations

import re

import pytest

from app.config import settings
from app.eval.corpus import GoldenCorpus, load_corpus
from app.eval.runner import EvaluationRunner, RunnerConfig
from app.eval.stt import LocalTranscriber
from app.services.g2p_capability import g2p_capability
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

#: One sentence per language, taken from the golden corpus by id so the measured
#: text is the committed text (no second, drifting copy).
#:
#: French uses ``fr-numbers`` rather than the greeting: the greeting's proper
#: name ("Werner") is misheard by the pinned ``base`` Whisper model as "Wayne",
#: which measures the recogniser, not the synthesiser. ``fr-numbers`` measured
#: WER 0.25 (target 0.50). The French *brand-name* defect is covered at the
#: phoneme level in ``test_multilingual_g2p.py``, where it belongs.
REGRESSION_ENTRY_IDS = (
    "en-greeting-short",
    "es-greeting-short",
    "fr-numbers",
    "hi-greeting-short",
    "it-greeting-short",
    "pt-greeting-short",
    "zh-greeting-short",
    "ja-greeting-short",
)

#: A render may not take more than this multiple of the expected duration for its
#: text. The Japanese defect produced ~8x (16.11 s for a short sentence), so the
#: bound is the signal, not a quality judgement about pacing.
MAX_DURATION_RATIO = 2.2
MIN_DURATION_RATIO = 0.4

HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
DEVANAGARI = re.compile(r"[\u0900-\u097f]")


def _models_mounted() -> bool:
    return MODEL_PATH.exists() and VOICES_PATH.exists()


@pytest.fixture(scope="module")
def measured():
    """Render the regression corpus once and return the harness report pieces.

    Module-scoped because loading Kokoro (325 MB) and Whisper (base) dominates
    the runtime, and none of the assertions mutate the services.
    """
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
    subset = GoldenCorpus(
        entries=tuple(
            sorted(
                corpus.filtered(ids=REGRESSION_ENTRY_IDS),
                key=lambda entry: entry.id,
            )
        ),
        path=corpus.path,
        digest=corpus.digest,
        schema_version=corpus.schema_version,
        description="SPEECH-010 per-language round-trip subset",
        metrics_note=corpus.metrics_note,
    )
    assert subset.entries, "the regression subset is empty; the corpus ids moved"

    engine = tts.engine_manager.get("kokoro")
    runner = EvaluationRunner(
        subset,
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
    }


def _cell_by_voice(measured, voice_id):
    return next((cell for cell in measured["cells"] if cell.voice == voice_id), None)


def _entry(cell, entry_id):
    return next((item for item in cell.entries if item["id"] == entry_id), None)


# ---------------------------------------------------------------------------
# Every measured language must actually round-trip
# ---------------------------------------------------------------------------


def test_every_supported_language_round_trips(measured):
    """Each measured cell must come back as recognisable text in its language.

    This is the acceptance criterion for every language ``/api/tts/capabilities``
    advertises: if it cannot pass here it must not be advertised.
    """
    measured_languages = {cell.language for cell in measured["cells"]}
    supported_languages = {
        canonical.split("-")[0] for canonical in measured["supported"]
    }
    missing = supported_languages - measured_languages
    assert not missing, (
        f"these advertised languages have no measured cell: {sorted(missing)} "
        f"(measured: {sorted(measured_languages)})"
    )

    problems = []
    for cell in measured["cells"]:
        for item in cell.entries:
            target = measured["corpus"].get(item["id"]).max_wer or 1.0
            if item["status"] != "ok":
                problems.append(f"{cell.language}/{cell.voice}: {item['status']} {item['error']}")
            elif "silent_output" in item["checks"]:
                problems.append(f"{cell.language}/{cell.voice}: silent output")
            elif item["wer"] is None:
                problems.append(f"{cell.language}/{cell.voice}: no transcript")
            elif item["wer"] > target:
                problems.append(
                    f"{cell.language}/{cell.voice}: WER {item['wer']:.2f} > {target:.2f} "
                    f"(heard {item['transcript']!r})"
                )
    assert not problems, "\n".join(problems)


def test_duration_stays_proportional_to_the_sentence(measured):
    """The Japanese signal: 21 characters must not produce 16 seconds of audio.

    Measured on the cells that *are* supported, so the bound is a real guard
    against a future G2P regression that spells text out rather than reading it.
    """
    offenders = []
    for cell in measured["cells"]:
        for item in cell.entries:
            ratio = item["duration_ratio"]
            if ratio is None:
                offenders.append(f"{cell.language}/{cell.voice}: no duration")
            elif not MIN_DURATION_RATIO <= ratio <= MAX_DURATION_RATIO:
                offenders.append(
                    f"{cell.language}/{cell.voice}: {item['duration_s']:.2f}s "
                    f"= {ratio:.2f}x expected"
                )
    assert not offenders, "\n".join(offenders)


def test_chinese_speaks_chinese(measured):
    """``zf_xiaoxiao`` and ``zm_yunjian`` must return Chinese text, not an error."""
    for voice_id in ("zf_xiaoxiao", "zm_yunjian"):
        cell = _cell_by_voice(measured, voice_id)
        assert cell is not None, f"{voice_id} produced no measurement cell"
        assert cell.language == "zh"
        item = _entry(cell, "zh-greeting-short")
        assert item is not None and item["status"] == "ok", f"{voice_id}: {item}"
        assert item["transcript"], f"{voice_id}: empty transcript"
        assert HAN.search(item["transcript"]), (
            f"{voice_id}: transcript has no Han characters: {item['transcript']!r}"
        )
        # Guard against the pre-fix failure mode: an espeak backend error would
        # have produced no audio at all.
        assert "synthesis_error" not in item["checks"]


def test_hindi_returns_devanagari(measured):
    """The round trip must come back in Devanagari, not Perso-Arabic (Urdu).

    The ticket saw Urdu script and suspected a language-tag mismatch. It is not:
    ``language="hi"`` reaches faster-whisper unchanged (see the investigation in
    the completion report). The pinned ``base`` model chooses the script from the
    acoustics and prefers Perso-Arabic; the harness pins the character set with a
    fixed Devanagari ``initial_prompt`` that carries no corpus content.
    """
    cell = _cell_by_voice(measured, "hf_alpha")
    assert cell is not None, "hf_alpha produced no measurement cell"
    item = _entry(cell, "hi-greeting-short")
    assert item is not None and item["status"] == "ok", f"hf_alpha: {item}"
    assert item["transcript"], "empty Hindi transcript"
    assert DEVANAGARI.search(item["transcript"]), (
        f"expected Devanagari, got {item['transcript']!r}"
    )


# ---------------------------------------------------------------------------
# The unsupported language is refused, documented and never measured
# ---------------------------------------------------------------------------


def test_japanese_is_skipped_with_a_reason_not_measured(measured):
    """Japanese is advertised by the acoustics but cannot be phonemised here.

    The honest outcome is an explicit skip with the measured reason — never a
    WER computed from fluent nonsense.
    """
    assert all(cell.language != "ja" for cell in measured["cells"]), (
        "a Japanese cell was measured; the G2P guard did not fire"
    )
    japanese = [row for row in measured["skipped"] if row["language"] == "ja"]
    assert japanese, f"Japanese was dropped silently: skipped={measured['skipped']}"
    reason = japanese[0]["reason"] or ""
    assert "spells" in reason.lower(), f"unhelpful reason: {reason!r}"
    assert "ja" not in measured["supported"]
