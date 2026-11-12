"""SPEECH-010 N2: the harness plumbing that reports unsupported languages.

The per-language acceptance evidence lives in ``test_multilingual_roundtrip.py``,
which is marked slow because it needs Kokoro and Whisper on disk. The *plumbing*
those slow tests rely on must not be slow, or the "no language is advertised
without evidence" rule is enforced only by a test that CI never runs. So the
glue is pinned here, with doubles:

* the runner records an unsupported language as skipped-with-a-reason and
  measures nothing for it;
* the report carries that skip list and its count;
* the transcriber applies the fixed script prompt for the right languages only;
* the routing matrix defaults to the corpus, not a hand-maintained list.
"""

from __future__ import annotations

import json

import pytest

from app.eval.corpus import load_corpus
from app.eval.engines import EvalEngineError
from app.eval.report import build_report
from app.eval.runner import EvaluationRunner, RunnerConfig
from app.eval.stt import SCRIPT_PROMPTS, LocalTranscriber
from tests.eval_fakes import EchoTranscriber, ToneEngine


class _G2PAwareToneEngine(ToneEngine):
    """``ToneEngine`` with a text front end that cannot read Japanese."""

    def __init__(self, unsupported: tuple[str, ...] = ("ja",), **kwargs) -> None:
        super().__init__(languages=("en", "ja"), **kwargs)
        self._unsupported = set(unsupported)

    def g2p_support(self, language: str | None) -> tuple[bool, str | None]:
        if language in self._unsupported:
            return False, "espeak-ng cannot read this script"
        return True, None


# ---------------------------------------------------------------------------
# The runner skips instead of measuring fluent nonsense
# ---------------------------------------------------------------------------


def test_runner_records_an_unsupported_language_as_skipped():
    corpus = load_corpus()
    runner = EvaluationRunner(
        corpus,
        [_G2PAwareToneEngine()],
        EchoTranscriber(),
        config=RunnerConfig(languages=("en", "ja")),
    )

    cells = runner.run()

    assert [cell.language for cell in cells] == ["en"]
    assert len(runner.skipped) == 1
    skip = runner.skipped[0]
    assert skip["language"] == "ja"
    assert skip["status"] == "unsupported_language"
    assert skip["reason"] == "espeak-ng cannot read this script"
    assert skip["voice"] == "tone_ja"
    # Nothing was measured for Japanese: no cell, so no WER can be conjured for it.
    assert all(cell.language != "ja" for cell in cells)


def test_runner_skips_every_voice_of_an_unsupported_language():
    corpus = load_corpus()
    runner = EvaluationRunner(
        corpus,
        [_G2PAwareToneEngine()],
        EchoTranscriber(),
        config=RunnerConfig(languages=("ja",)),
    )

    with pytest.raises(EvalEngineError) as excinfo:
        # Every cell is skipped, so there is nothing measurable. Raising is the
        # honest outcome: an empty report that says "passed" is the one failure
        # mode a measurement harness must never have.
        runner.run()

    assert "no engine/voice cell" in str(excinfo.value)
    assert [row["language"] for row in runner.skipped] == ["ja"]


def test_an_engine_without_g2p_support_is_measured_normally():
    """A third-party adapter that exposes no probe keeps the old behaviour."""
    corpus = load_corpus()
    runner = EvaluationRunner(
        corpus,
        [ToneEngine(languages=("en",))],
        EchoTranscriber(),
        config=RunnerConfig(languages=("en",)),
    )

    cells = runner.run()

    assert [cell.language for cell in cells] == ["en"]
    assert runner.skipped == []


# ---------------------------------------------------------------------------
# The report carries the skips
# ---------------------------------------------------------------------------


def _minimal_report(**kwargs):
    """A report payload with no cells, so only the bookkeeping is under test."""
    return build_report(corpus=load_corpus(), cells=[], **kwargs)


def test_report_records_skipped_cells_and_their_count():
    skipped = [
        {"engine": "kokoro", "voice": "jf_alpha", "language": "ja", "reason": "spelling"},
        {"engine": "kokoro", "voice": "jm_kumo", "language": "ja", "reason": "spelling"},
    ]

    report = _minimal_report(skipped=skipped)

    assert report["skipped"] == skipped
    assert report["status"]["cells_skipped"] == 2


def test_report_without_skips_still_reports_the_field():
    report = _minimal_report()

    assert report["skipped"] == []
    assert report["status"]["cells_skipped"] == 0


# ---------------------------------------------------------------------------
# The transcriber's script prompt wiring
# ---------------------------------------------------------------------------


class _RecordingSTTService:
    """An STT double that records the decoder kwargs it was given."""

    available_models = ["base"]
    default_model = "base"

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def transcribe(self, wav_bytes, language=None, **kwargs):
        self.calls.append({"language": language, **kwargs})
        return {"text": "ok"}


def test_script_prompt_is_applied_for_the_languages_that_need_it():
    service = _RecordingSTTService()
    transcriber = LocalTranscriber(service)

    transcriber.transcribe(b"wav", "hi")
    transcriber.transcribe(b"wav", "zh")

    assert service.calls[0]["initial_prompt"] == SCRIPT_PROMPTS["hi"]
    assert service.calls[1]["initial_prompt"] == SCRIPT_PROMPTS["zh"]


@pytest.mark.parametrize("language", ["en", "es", "fr", "it", "pt", "ja", None])
def test_script_prompt_is_not_applied_to_other_languages(language):
    """The prompt is a script selector, not a hint: other languages get none."""
    service = _RecordingSTTService()
    transcriber = LocalTranscriber(service)

    transcriber.transcribe(b"wav", language)

    assert service.calls[0]["initial_prompt"] is None


def test_script_prompts_are_fixed_and_carry_no_corpus_content():
    """A prompt that named corpus text would bias the measurement it scores."""
    corpus = load_corpus()
    for language, prompt in SCRIPT_PROMPTS.items():
        assert prompt, language
        for entry in corpus.entries:
            assert entry.text not in prompt
            assert prompt not in entry.text


def test_transcriber_describes_the_script_prompts():
    described = LocalTranscriber(_RecordingSTTService()).describe()

    assert described["script_prompts"] == sorted(SCRIPT_PROMPTS)
    assert "script_prompt_note" in described


# ---------------------------------------------------------------------------
# route --matrix defaults to the corpus
# ---------------------------------------------------------------------------


def test_route_matrix_defaults_to_every_corpus_language(capsys):
    from app.eval.cli import main

    assert main(["route", "--matrix", "--json"]) == 0
    decisions = json.loads(capsys.readouterr().out)

    assert {row["language"] for row in decisions} == set(load_corpus().languages)
    assert all(row["engine"] for row in decisions)
