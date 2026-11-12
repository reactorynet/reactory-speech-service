"""Report rendering, baseline diffing and the command line contract.

The report is the artifact a reviewer reads in a pull request, and the CLI exit
codes are what a release gate consumes. Both are tested here: a diff must refuse to
compare incompatible reports, and a broken run must not exit zero.
"""

from __future__ import annotations

import json

import pytest

from app.eval.cli import main
from app.eval.corpus import load_corpus
from app.eval.engines import register_engine, unregister_engine
from app.eval.report import (
    SCHEMA,
    build_report,
    diff_reports,
    gate_regressions,
    is_pass,
    load_report,
    render_diff,
    render_failures,
    render_table,
    scores_by_engine,
    write_report,
)
from tests.eval_fakes import ToneEngine


def _synthetic_report(
    *,
    wer: float = 0.1,
    rtf: float = 2.0,
    duration: float = 1.5,
    embedder: str = "spectral-mel-stats-v1",
) -> dict:
    """A one-cell report, for diff/rendering tests that do not need an engine."""
    corpus = load_corpus()
    entry = {
        "id": "en-greeting-short",
        "use": "short greeting",
        "text_digest": "deadbeef0000",
        "status": "ok",
        "checks": [],
        "wer": wer,
        "cer": wer,
        "duration_s": duration,
        "duration_ratio": duration / 1.5,
        "rtf": rtf,
        "peak_memory_mb": 1.0,
    }
    cell = {
        "engine": "tone",
        "voice": "tone_en",
        "language": "en",
        "entries": [entry],
        "summary": {
            "wer_mean": wer,
            "cer_mean": wer,
            "duration_ratio_mean": 1.0,
            "rtf_mean": rtf,
            "peak_memory_mb_max": 1.0,
        },
        "streaming": {"ttfa_s": 0.01},
        "speaker_consistency": {"mean": 1.0, "min": 1.0},
        "prosody": {"adherence": 1.0},
    }
    return build_report(
        corpus=corpus,
        cells=[cell],
        engines=[{"id": "tone"}],
        transcriber={"enabled": True, "available": True, "model": "base"},
        embedder={"id": embedder},
        config={},
    )


# ----------------------------------------------------------------------------
# Report assembly and rendering
# ----------------------------------------------------------------------------


def test_report_is_rounded_and_classifies_findings() -> None:
    corpus = load_corpus()
    cell = {
        "engine": "tone",
        "voice": "tone_en",
        "language": "en",
        "entries": [
            {
                "id": "a",
                "checks": ["silent_output"],
                "wer": 1.0,
                "duration_s": 0.123456789,
                "status": "ok",
            },
            {"id": "b", "checks": ["wer_threshold"], "wer": 0.87654321, "status": "ok"},
        ],
        "summary": {"wer_mean": 0.87654321},
    }
    report = build_report(corpus=corpus, cells=[cell])

    assert report["schema"] == SCHEMA
    assert report["status"]["integrity_failures"] == 1
    assert report["status"]["quality_failures"] == 1
    assert report["status"]["passed"] is False
    assert report["cells"][0]["entries"][0]["duration_s"] == 0.123
    assert report["cells"][0]["entries"][1]["wer"] == 0.8765
    assert report["cells"][0]["summary"]["wer_mean"] == 0.8765
    assert is_pass(report) is False
    assert is_pass(report, fail_on_wer=True) is False


def test_render_table_and_failures_are_human_readable() -> None:
    report = _synthetic_report()
    table = render_table(report)
    assert "engine" in table and "wer" in table
    assert "tone" in table and "tone_en" in table
    assert "passed=yes" in table
    assert render_failures(report) == "no failures"


def test_report_round_trips_through_json(tmp_path) -> None:
    report = _synthetic_report()
    path = write_report(report, tmp_path / "report.json")
    assert load_report(path) == report
    assert path.read_text(encoding="utf-8").endswith("\n")

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema": "something-else"}), encoding="utf-8")
    with pytest.raises(ValueError, match="not a"):
        load_report(bad)


# ----------------------------------------------------------------------------
# Baseline diffing
# ----------------------------------------------------------------------------


def test_identical_reports_have_no_regressions() -> None:
    report = _synthetic_report()
    diff = diff_reports(report, report)
    assert diff["comparable"] is True
    assert diff["regressions"] == []
    assert diff["improvements"] == []
    assert all(delta["status"] == "unchanged" for delta in diff["deltas"])


def test_a_worse_wer_is_a_regression() -> None:
    diff = diff_reports(_synthetic_report(wer=0.4), _synthetic_report(wer=0.1))
    regressions = {delta["metric"] for delta in diff["regressions"]}
    assert "wer" in regressions
    assert "REGRESSIONS" in render_diff(diff)


def test_a_better_wer_is_an_improvement() -> None:
    diff = diff_reports(_synthetic_report(wer=0.01), _synthetic_report(wer=0.4))
    improved = {delta["metric"] for delta in diff["improvements"]}
    assert {"wer", "cer", "wer_mean", "cer_mean"} <= improved


def test_a_different_corpus_is_not_comparable() -> None:
    baseline = _synthetic_report()
    candidate = _synthetic_report()
    candidate["meta"]["corpus"]["digest"] = "sha256:different"
    diff = diff_reports(candidate, baseline)
    assert diff["comparable"] is False
    assert any("corpus digest" in reason for reason in diff["reasons"])


def test_a_different_embedder_is_not_comparable() -> None:
    baseline = _synthetic_report(embedder="ecapa-local-v1")
    candidate = _synthetic_report(embedder="spectral-mel-stats-v1")
    diff = diff_reports(candidate, baseline)
    assert diff["comparable"] is False
    assert any("speaker embedder" in reason for reason in diff["reasons"])


def test_a_different_whisper_model_is_not_comparable() -> None:
    baseline = _synthetic_report()
    candidate = _synthetic_report()
    candidate["meta"]["transcriber"]["model"] = "small"
    diff = diff_reports(candidate, baseline)
    assert diff["comparable"] is False
    assert any("Whisper model" in reason for reason in diff["reasons"])


def test_a_duration_shift_is_changed_and_still_gates() -> None:
    """Neither longer nor shorter is inherently better, but a shift is a finding."""
    diff = diff_reports(_synthetic_report(duration=3.0), _synthetic_report(duration=1.0))
    statuses = {delta["metric"]: delta["status"] for delta in diff["deltas"]}
    assert statuses["duration_s"] == "changed"
    assert "duration_s" not in {delta["metric"] for delta in diff["regressions"]}
    assert "duration_s" in {delta["metric"] for delta in gate_regressions(diff)}
    assert "CHANGED" in render_diff(diff)


def test_a_timing_regression_is_advisory_by_default() -> None:
    """RTF on a busy machine is not a reproducible signal, so it does not gate."""
    diff = diff_reports(_synthetic_report(rtf=6.0), _synthetic_report(rtf=2.0))
    assert any(delta["metric"] == "rtf" for delta in diff["regressions"])
    assert gate_regressions(diff) == []
    assert [delta["metric"] for delta in gate_regressions(diff, frozenset())] == ["rtf"]


def test_missing_and_new_metrics_are_reported() -> None:
    baseline = _synthetic_report()
    candidate = json.loads(json.dumps(baseline))
    candidate["cells"][0]["entries"][0]["id"] = "en-greeting-calm"
    diff = diff_reports(candidate, baseline)
    assert diff["new"] and diff["missing"]


def test_scores_by_engine_averages_across_voices() -> None:
    report = _synthetic_report(wer=0.2)
    extra = json.loads(json.dumps(report["cells"][0]))
    extra["voice"] = "tone_en_second"
    extra["summary"]["wer_mean"] = 0.4
    report["cells"].append(extra)

    scores = scores_by_engine(report)
    assert scores["tone"]["wer"] == pytest.approx(0.3)
    assert set(scores["tone"]["voices"]) == {"tone_en", "tone_en_second"}


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


@pytest.fixture
def registered_tone_engine():
    register_engine("tone", lambda: ToneEngine("tone"), replace=True)
    register_engine("tone-muted", lambda: ToneEngine("tone-muted", mode="muted"), replace=True)
    try:
        yield
    finally:
        unregister_engine("tone")
        unregister_engine("tone-muted")


def test_cli_run_writes_a_report_and_a_baseline(tmp_path, registered_tone_engine, capsys) -> None:
    out = tmp_path / "report.json"
    baseline = tmp_path / "baseline.json"
    code = main(
        [
            "run",
            "--engines",
            "tone",
            "--languages",
            "en",
            "--no-stt",
            "--memory-mode",
            "off",
            "--out",
            str(out),
            "--baseline-out",
            str(baseline),
        ]
    )

    assert code == 0
    assert out.exists() and baseline.exists()
    report = load_report(out)
    assert report["schema"] == SCHEMA
    assert report["meta"]["transcriber"]["enabled"] is False
    assert report["routing"]["decisions"], "the run must document the routing decision"
    assert report["cells"][0]["engine"] == "tone"
    printed = capsys.readouterr().out
    assert "report:" in printed
    assert "routing" in printed


def test_cli_run_fails_on_a_broken_engine(tmp_path, registered_tone_engine) -> None:
    code = main(
        [
            "run",
            "--engines",
            "tone-muted",
            "--languages",
            "en",
            "--no-stt",
            "--memory-mode",
            "off",
            "--out",
            str(tmp_path / "muted.json"),
            "--quiet",
        ]
    )
    assert code == 1
    report = load_report(tmp_path / "muted.json")
    assert report["status"]["passed"] is False
    assert any(failure["check"] == "silent_output" for failure in report["failures"])


def test_cli_run_rejects_an_unknown_engine(tmp_path, capsys) -> None:
    code = main(["run", "--engines", "not-an-engine", "--out", str(tmp_path / "x.json")])
    assert code == 2
    assert "error:" in capsys.readouterr().err


def test_cli_report_prints_the_table(tmp_path, capsys) -> None:
    path = write_report(_synthetic_report(), tmp_path / "report.json")
    assert main(["report", str(path)]) == 0
    assert "tone_en" in capsys.readouterr().out


def test_cli_compare_detects_a_regression(tmp_path) -> None:
    baseline = write_report(_synthetic_report(wer=0.1), tmp_path / "baseline.json")
    candidate_path = write_report(_synthetic_report(wer=0.5), tmp_path / "candidate.json")

    assert main(["compare", str(candidate_path), str(baseline)]) == 0
    assert main(["compare", str(candidate_path), str(baseline), "--fail-on-regression"]) == 1


def test_cli_compare_gate_ignores_timing_unless_asked(tmp_path) -> None:
    baseline = write_report(_synthetic_report(rtf=2.0), tmp_path / "baseline.json")
    candidate = write_report(_synthetic_report(rtf=8.0), tmp_path / "candidate.json")

    # Default gate: deterministic quality metrics only.
    assert main(["compare", str(candidate), str(baseline), "--fail-on-regression"]) == 0
    # Explicitly including performance metrics makes it a regression.
    assert (
        main(
            [
                "compare",
                str(candidate),
                str(baseline),
                "--fail-on-regression",
                "--metrics",
                "all",
            ]
        )
        == 1
    )
    assert (
        main(
            [
                "compare",
                str(candidate),
                str(baseline),
                "--fail-on-regression",
                "--metrics",
                "rtf",
            ]
        )
        == 1
    )


def test_cli_compare_refuses_incomparable_reports(tmp_path, capsys) -> None:
    baseline = _synthetic_report()
    candidate = _synthetic_report()
    candidate["meta"]["corpus"]["digest"] = "sha256:other"
    baseline_path = write_report(baseline, tmp_path / "baseline.json")
    candidate_path = write_report(candidate, tmp_path / "candidate.json")

    assert main(["compare", str(candidate_path), str(baseline_path)]) == 0
    assert main(["compare", str(candidate_path), str(baseline_path), "--fail-on-regression"]) == 2
    assert "NOT COMPARABLE" in capsys.readouterr().out


def test_cli_route_answers_and_documents_the_choice(capsys) -> None:
    assert main(["route", "--language", "ja", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["engine"] == "kokoro"
    assert payload["rule"] == "japanese-chinese-coverage"

    assert main(["route", "--language", "xx", "--style", "neutral", "--tier", "gold"]) == 0
    assert "engine: kokoro" in capsys.readouterr().out


def test_cli_route_matrix_covers_the_corpus(capsys) -> None:
    assert main(["route", "--matrix", "--json"]) == 0
    decisions = json.loads(capsys.readouterr().out)
    assert {decision["language"] for decision in decisions} >= set(load_corpus().languages)
    assert all(decision["engine"] for decision in decisions)
