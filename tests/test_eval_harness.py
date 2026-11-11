"""The harness must falsify: broken audio is a failure, not a pass.

Every test in this module runs the real pipeline — engine adapter, WAV bytes,
decode, transcript, metrics, report — against a deliberately broken engine double.
The only thing faked is the acoustic model, because a 325 MB Kokoro model does not
belong in a unit test. If these tests pass, a muted, truncated or resampled engine
cannot slip through the harness unnoticed.
"""

from __future__ import annotations

import pytest

from app.core.contracts import AudioResult
from app.eval.corpus import load_corpus
from app.eval.engines import (
    EvalEngineError,
    register_engine,
    resolve_engines,
    unregister_engine,
)
from app.eval.report import build_report, diff_reports, is_pass
from app.eval.runner import EvaluationRunner, RunnerConfig, select_voices
from tests.eval_fakes import (
    EchoTranscriber,
    NullTranscriberDouble,
    ScriptedTranscriber,
    ToneEngine,
)

#: Quality metrics are deterministic for a deterministic engine; timing is not.
QUALITY_METRICS = {"wer", "cer", "duration_s", "duration_ratio"}


def _report(engine, transcriber=None, **config_kwargs):
    """Run one engine through the harness and build the report payload."""
    corpus = load_corpus()
    config = RunnerConfig(languages=("en",), **config_kwargs)
    runner = EvaluationRunner(
        corpus,
        [engine],
        transcriber or EchoTranscriber(),
        config=config,
    )
    cells = [cell.as_dict() for cell in runner.run()]
    metadata = runner.metadata()
    return build_report(
        corpus=corpus,
        cells=cells,
        engines=metadata["engines"],
        transcriber=metadata["transcriber"],
        embedder=metadata["speaker_embedder"],
        config=config.as_dict(),
    )


def _checks(report):
    return {failure["check"] for failure in report["failures"]}


def _entry_values(report, key):
    return [
        entry[key]
        for cell in report["cells"]
        for entry in cell["entries"]
        if entry.get(key) is not None
    ]


# ----------------------------------------------------------------------------
# The happy path
# ----------------------------------------------------------------------------


def test_healthy_engine_passes_every_check() -> None:
    report = _report(ToneEngine())

    assert report["status"]["passed"] is True
    assert report["status"]["integrity_failures"] == 0
    assert report["status"]["entries"] == len(load_corpus().for_language("en"))

    for cell in report["cells"]:
        assert cell["engine"] == "tone"
        assert cell["voice"] == "tone_en"
        assert all(entry["wer"] == 0.0 for entry in cell["entries"])
        assert all(entry["checks"] == [] for entry in cell["entries"])
        # The streaming probe measured the first chunk.
        assert cell["streaming"]["ttfa_s"] is not None
        # Repeated renders of a deterministic engine are the same voice.
        assert cell["speaker_consistency"]["mean"] is not None
        assert cell["speaker_consistency"]["mean"] > 0.999
        # A 1.5x request produced ~1.5x the rate.
        assert cell["prosody"]["adherence"] == pytest.approx(1.0, abs=0.05)


def test_report_records_the_engine_and_embedder_identity() -> None:
    report = _report(ToneEngine())
    identity = report["meta"]["engines"][0]
    assert identity["id"] == "tone"
    assert identity["native_rate"] is True
    assert identity["sample_rates"] == [24_000]
    assert report["meta"]["speaker_embedder"]["id"].startswith("spectral")
    assert report["meta"]["transcriber"]["enabled"] is True
    assert report["meta"]["corpus"]["digest"].startswith("sha256:")


# ----------------------------------------------------------------------------
# Deliberate breakage: the falsification cases the ticket requires
# ----------------------------------------------------------------------------


def test_muted_engine_is_a_failure_not_a_pass() -> None:
    report = _report(ToneEngine(mode="muted"))

    checks = _checks(report)
    assert "silent_output" in checks
    assert "empty_transcript" in checks
    assert report["status"]["passed"] is False
    assert is_pass(report) is False
    # A dead audio path cannot score WER 0.
    assert set(_entry_values(report, "wer")) == {1.0}


def test_truncated_engine_is_caught_by_duration_and_wer() -> None:
    report = _report(ToneEngine(mode="truncated"))

    checks = _checks(report)
    assert "duration_out_of_bounds" in checks
    assert report["status"]["passed"] is False
    # Truncated audio also loses words, so WER independently reports the defect.
    assert min(_entry_values(report, "wer")) > 0.5


def test_wrong_sample_rate_is_caught_as_a_format_failure() -> None:
    report = _report(ToneEngine(mode="wrong_rate"))

    checks = _checks(report)
    assert "unexpected_sample_rate" in checks
    assert set(_entry_values(report, "sample_rate")) == {48_000}
    assert report["status"]["passed"] is False


def test_engine_exception_is_recorded_as_a_synthesis_failure() -> None:
    report = _report(ToneEngine(mode="error"))

    assert "synthesis_error" in _checks(report)
    assert report["status"]["passed"] is False
    assert all(entry["status"] == "error" for cell in report["cells"] for entry in cell["entries"])


def test_undecodable_audio_is_a_failure() -> None:
    class GarbageEngine(ToneEngine):
        def synthesize(self, req):
            return AudioResult(
                audio=b"this is not audio",
                duration=1.0,
                sample_rate=24_000,
                engine=self.id,
                voice=req.voice,
            )

    report = _report(GarbageEngine())
    assert "undecodable_audio" in _checks(report)
    assert report["status"]["passed"] is False


def test_engine_that_ignores_the_rate_is_visible_in_prosody() -> None:
    """A 1.5x request rendered at 1.0x is a rate adherence of ~0.67."""
    report = _report(ToneEngine(rate_applied=False, honour_rate=False))

    prosody = report["cells"][0]["prosody"]
    assert prosody["applied"] == {}
    assert prosody["ignored"] == ["rate"]
    assert prosody["adherence"] == pytest.approx(1 / 1.5, abs=0.05)


# ----------------------------------------------------------------------------
# Quality gate vs integrity gate
# ----------------------------------------------------------------------------


def test_wer_gate_is_opt_in_and_reported_as_quality() -> None:
    # Audio is fine, the transcript is not: a quality finding, not a broken render.
    report = _report(ToneEngine(), transcriber=ScriptedTranscriber(keep_words=1))

    assert "wer_threshold" in _checks(report)
    assert report["status"]["quality_failures"] > 0
    assert report["status"]["integrity_failures"] == 0
    assert is_pass(report) is True
    assert is_pass(report, fail_on_wer=True) is False


def test_disabled_stt_is_recorded_as_unmeasured_not_as_zero_wer() -> None:
    report = _report(ToneEngine(), transcriber=NullTranscriberDouble())

    assert report["meta"]["transcriber"]["enabled"] is False
    assert _entry_values(report, "wer") == []
    assert "wer_threshold" not in _checks(report)
    assert is_pass(report) is True


# ----------------------------------------------------------------------------
# Determinism
# ----------------------------------------------------------------------------


def test_reports_are_deterministic_enough_to_diff() -> None:
    first = _report(ToneEngine())
    second = _report(ToneEngine())

    def quality(report):
        return {
            (cell["engine"], cell["voice"], entry["id"]): tuple(
                entry[metric] for metric in sorted(QUALITY_METRICS)
            )
            for cell in report["cells"]
            for entry in cell["entries"]
        }

    assert quality(first) == quality(second)
    assert set(first["cells"][0]["entries"][0]) == set(second["cells"][0]["entries"][0])

    diff = diff_reports(second, first)
    assert diff["comparable"] is True
    assert all(delta["metric"] not in QUALITY_METRICS for delta in diff["regressions"])


# ----------------------------------------------------------------------------
# Extensibility: a new engine is a registration, not a harness change
# ----------------------------------------------------------------------------


def test_engine_can_be_registered_at_runtime() -> None:
    register_engine("tone-extra", lambda: ToneEngine("tone-extra"), replace=True)
    try:
        engines = resolve_engines(["tone-extra"])
        assert [engine.id for engine in engines] == ["tone-extra"]
        report = _report(engines[0])
        assert report["cells"][0]["engine"] == "tone-extra"
    finally:
        unregister_engine("tone-extra")


def test_engine_can_be_loaded_from_a_module_spec() -> None:
    engines = resolve_engines(["tests.eval_fakes:ToneEngine"])
    assert [engine.id for engine in engines] == ["tone"]


def test_registration_conflicts_are_loud() -> None:
    register_engine("tone-conflict", lambda: ToneEngine("tone-conflict"), replace=True)
    try:
        with pytest.raises(EvalEngineError, match="already registered"):
            register_engine("tone-conflict", lambda: ToneEngine("tone-conflict"))
    finally:
        unregister_engine("tone-conflict")


def test_unknown_engine_spec_is_rejected() -> None:
    with pytest.raises(EvalEngineError, match="unknown engine"):
        resolve_engines(["definitely-not-an-engine"])


# ----------------------------------------------------------------------------
# Voice selection
# ----------------------------------------------------------------------------


def test_default_voice_selection_is_one_best_voice_per_language() -> None:
    engine = ToneEngine(languages=("en", "fr"))

    default = select_voices(engine, ["en", "fr"], "default")
    assert [voice["id"] for voice in default] == ["tone_en", "tone_fr"]

    every = select_voices(engine, ["en", "fr"], "all")
    assert len(every) == 2

    explicit = select_voices(engine, ["en", "fr"], "tone_fr")
    assert [voice["id"] for voice in explicit] == ["tone_fr"]

    with pytest.raises(EvalEngineError, match="does not expose"):
        select_voices(engine, ["en"], "tone_zz")


def test_engine_with_no_matching_voice_fails_the_run() -> None:
    """A run that measures nothing must raise, not write an empty passing report."""
    runner = EvaluationRunner(
        load_corpus(),
        [ToneEngine(languages=("de",))],
        EchoTranscriber(),
        config=RunnerConfig(languages=("en",)),
    )
    with pytest.raises(EvalEngineError, match="no engine/voice cell"):
        runner.run()


def test_unknown_language_filter_is_rejected() -> None:
    runner = EvaluationRunner(
        load_corpus(),
        [ToneEngine()],
        EchoTranscriber(),
        config=RunnerConfig(languages=("xx",)),
    )
    with pytest.raises(EvalEngineError, match="not in the corpus"):
        runner.run()
