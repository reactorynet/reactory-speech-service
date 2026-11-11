"""Command line interface: ``python -m app.eval <command>``.

Commands
--------

``run``
    Render the corpus through the selected engines/voices, measure, and write a
    JSON report. Exits 1 when an integrity check fails (broken audio) and, with
    ``--fail-on-wer``, also when a sentence misses its WER target.

``report``
    Render a human-readable comparison table from a report file.

``compare``
    Diff a report against a committed baseline. Exits 1 with
    ``--fail-on-regression`` when a metric regressed beyond its documented
    tolerance.

``route``
    Ask the routing policy which engine to use for a combination, or print the
    whole matrix.

Exit codes are part of the contract: ``0`` passed, ``1`` gate failed, ``2`` the
run could not be performed (bad corpus, no engine, no local STT model).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from app.eval.corpus import CorpusError, load_corpus
from app.eval.engines import EvalEngineError, available_engines, resolve_engines
from app.eval.report import (
    DEFAULT_GATE_METRICS,
    build_report,
    diff_reports,
    gate_regressions,
    is_pass,
    load_report,
    render_diff,
    render_failures,
    render_routing,
    render_table,
    scores_by_engine,
    write_report,
)
from app.eval.routing import RoutingError, RoutingRequest, load_policy
from app.eval.runner import EvaluationRunner, RunnerConfig
from app.eval.speaker import build_embedder
from app.eval.stt import TranscriberUnavailable, build_transcriber

#: Default location for an ad-hoc run; ignored by git.
DEFAULT_OUTPUT = Path("reports") / "evaluation.json"


def _engine_capabilities(engines: Sequence[Any]) -> dict[str, Any]:
    return {
        str(getattr(engine, "id", "unknown")): engine.capabilities() for engine in engines
    }


def _scores_from_cells(cells: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Mean measured quality per engine, in the shape the routing policy expects."""
    return scores_by_engine({"cells": list(cells)})


def _print(text: str) -> None:
    print(text)


def _add_common_engine_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--register",
        action="append",
        default=[],
        metavar="MODULE:ATTR",
        help="register an engine adapter that is not packaged (repeatable)",
    )


# ----------------------------------------------------------------------------
# run
# ----------------------------------------------------------------------------


def _build_run_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser(
        "run", help="render the corpus, measure it and write a JSON report"
    )
    parser.add_argument("--corpus", default=None, help="path to the golden corpus JSON")
    parser.add_argument(
        "--engines",
        default="kokoro",
        help="comma-separated engine ids, 'all', or module:attr specs (default: kokoro)",
    )
    parser.add_argument(
        "--voices",
        default="default",
        help="'default' (best-graded voice per language), 'all', or comma-separated ids",
    )
    parser.add_argument("--languages", default=None, help="comma-separated language filter")
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT), help="report path")
    parser.add_argument(
        "--baseline-out",
        default=None,
        help="also write the report here (use app/eval/baselines/<name>.json to promote a baseline)",
    )
    parser.add_argument("--stt-model", default=None, help="local Whisper model id")
    parser.add_argument(
        "--no-stt",
        action="store_true",
        help="skip round-trip WER/CER (the report records stt: disabled)",
    )
    parser.add_argument(
        "--speaker-embedder",
        default="spectral",
        help="'spectral' (default, dependency-free) or module:factory",
    )
    parser.add_argument(
        "--repeats", type=int, default=2, help="renders per voice for speaker consistency"
    )
    parser.add_argument(
        "--probe-rate", type=float, default=1.5, help="rate requested by the prosody probe"
    )
    parser.add_argument(
        "--memory-mode",
        default="tracemalloc",
        choices=["tracemalloc", "rss", "off"],
        help="peak-memory measurement mode (best effort)",
    )
    parser.add_argument(
        "--cross-language",
        action="store_true",
        help="render every entry through every voice (deliberately wrong pairs; diagnostic)",
    )
    parser.add_argument(
        "--fail-on-wer",
        action="store_true",
        help="exit 1 when a sentence exceeds its corpus WER target",
    )
    parser.add_argument("--policy", default=None, help="routing policy JSON")
    parser.add_argument(
        "--stamp", action="store_true", help="record generated_at (makes reports non-diffable)"
    )
    parser.add_argument("--quiet", action="store_true", help="write the report without printing")
    _add_common_engine_args(parser)


def _run(args: argparse.Namespace) -> int:
    corpus = load_corpus(args.corpus)
    engines = resolve_engines(
        [part.strip() for part in str(args.engines).split(",") if part.strip()],
        register=args.register,
    )
    transcriber = build_transcriber(enabled=not args.no_stt, model=args.stt_model)
    embedder = build_embedder(args.speaker_embedder)

    languages = (
        tuple(part.strip() for part in str(args.languages).split(",") if part.strip())
        if args.languages
        else None
    )
    config = RunnerConfig(
        repeats=max(1, int(args.repeats)),
        probe_rate=float(args.probe_rate),
        memory_mode=str(args.memory_mode),
        cross_language=bool(args.cross_language),
        fail_on_wer=bool(args.fail_on_wer),
        languages=languages,
    )
    runner = EvaluationRunner(
        corpus,
        engines,
        transcriber,
        embedder=embedder,
        config=config,
        voices_spec=str(args.voices),
    )
    cells = [cell.as_dict() for cell in runner.run()]
    metadata = runner.metadata()

    policy = load_policy(args.policy)
    capabilities = _engine_capabilities(engines)
    engine_ids = [str(getattr(engine, "id", "unknown")) for engine in engines]
    routing_languages = config.languages or tuple(corpus.languages)
    decisions = policy.matrix(
        list(routing_languages),
        available=engine_ids,
        capabilities=capabilities,
        scores=_scores_from_cells(cells),
    )

    report = build_report(
        corpus=corpus,
        cells=cells,
        routing={
            "policy": str(policy.path),
            "default_engine": policy.default_engine,
            "decisions": decisions,
        },
        engines=metadata["engines"],
        transcriber=metadata["transcriber"],
        embedder=metadata["speaker_embedder"],
        config={"voices": args.voices, **config.as_dict()},
        environment=metadata["environment"],
        generated_at=_timestamp() if args.stamp else None,
    )

    out_path = write_report(report, args.out)
    if args.baseline_out:
        write_report(report, args.baseline_out)

    if not args.quiet:
        _print(render_table(report))
        routing = render_routing(report)
        if routing:
            _print("")
            _print("routing")
            _print(routing)
        if report["failures"]:
            _print("")
            _print("failures")
            _print(render_failures(report))
        _print("")
        _print(f"report: {out_path}")
        if args.baseline_out:
            _print(f"baseline: {args.baseline_out}")

    if not is_pass(report, fail_on_wer=config.fail_on_wer):
        return 1
    return 0


def _timestamp() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ----------------------------------------------------------------------------
# report / compare / route
# ----------------------------------------------------------------------------


def _build_report_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("report", help="print a comparison table from a report")
    parser.add_argument("report", help="path to a report JSON")
    parser.add_argument("--json", action="store_true", help="print the report itself")
    parser.add_argument("--failures", action="store_true", help="print only the failures")


def _report(args: argparse.Namespace) -> int:
    report = load_report(args.report)
    if args.json:
        _print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
        return 0
    if args.failures:
        _print(render_failures(report))
        return 0
    _print(render_table(report))
    routing = render_routing(report)
    if routing:
        _print("")
        _print(routing)
    if report["failures"]:
        _print("")
        _print(render_failures(report))
    return 0


def _build_compare_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("compare", help="diff a report against a baseline")
    parser.add_argument("report", help="candidate report JSON")
    parser.add_argument("baseline", help="baseline report JSON")
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="exit 1 when a metric regressed beyond its documented tolerance",
    )
    parser.add_argument("--json", action="store_true", help="print the diff as JSON")
    parser.add_argument(
        "--metrics",
        default=None,
        help=(
            "comma-separated metric names that gate the exit code. Default: the "
            "deterministic quality metrics (wer, cer, duration_s, duration_ratio, "
            "speaker_consistency_*, prosody_adherence). Use 'all' to include rtf, "
            "ttfa_s and peak_memory_mb, which depend on the machine."
        ),
    )


def _gate_metric_filter(spec: str | None) -> frozenset[str] | None:
    """Translate ``--metrics`` into the gate's metric set."""
    if spec is None:
        return DEFAULT_GATE_METRICS
    if spec.strip().lower() == "all":
        return frozenset()
    return frozenset(part.strip() for part in spec.split(",") if part.strip())


def _compare(args: argparse.Namespace) -> int:
    candidate = load_report(args.report)
    baseline = load_report(args.baseline)
    diff = diff_reports(candidate, baseline)
    if args.json:
        _print(json.dumps(diff, indent=2, sort_keys=True, ensure_ascii=False))
    else:
        _print(render_diff(diff))
    if not diff["comparable"]:
        # Not a regression, but not a valid comparison either: refuse rather than
        # produce a reassuring number from different inputs.
        return 2 if args.fail_on_regression else 0
    gating = gate_regressions(diff, _gate_metric_filter(args.metrics))
    if args.fail_on_regression and gating:
        return 1
    return 0


def _build_route_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("route", help="ask the routing policy for an engine")
    parser.add_argument("--language", default="en")
    parser.add_argument("--style", default="neutral")
    parser.add_argument("--tier", default="standard")
    parser.add_argument("--rate", type=float, default=None)
    parser.add_argument(
        "--available",
        default=None,
        help="comma-separated engine ids available on the deployment (default: all known)",
    )
    parser.add_argument("--scores", default=None, help="report JSON to read measured scores from")
    parser.add_argument("--policy", default=None, help="routing policy JSON")
    parser.add_argument(
        "--matrix", action="store_true", help="print every corpus language x style x tier"
    )
    parser.add_argument("--languages", default=None, help="comma-separated languages for --matrix")
    parser.add_argument("--json", action="store_true", help="print JSON")


def _route(args: argparse.Namespace) -> int:
    policy = load_policy(args.policy)
    available = (
        [part.strip() for part in str(args.available).split(",") if part.strip()]
        if args.available
        else available_engines()
    )

    scores: dict[str, dict[str, Any]] = {}
    capabilities: dict[str, Any] = {}
    if args.scores:
        report = load_report(args.scores)
        scores = scores_by_engine(report)
        # The report's engine header carries the native capability flags a rule can
        # require; the full objects are not needed for a routing decision.
        for engine in (report.get("meta") or {}).get("engines") or []:
            capabilities[str(engine["id"])] = engine

    if args.matrix:
        languages = (
            [part.strip() for part in str(args.languages).split(",") if part.strip()]
            if args.languages
            else ["en", "fr", "ja", "zh", "es"]
        )
        decisions = policy.matrix(
            languages, available=available, capabilities=capabilities, scores=scores
        )
        if args.json:
            _print(json.dumps(decisions, indent=2, sort_keys=True, ensure_ascii=False))
        else:
            _print(render_routing({"routing": {"decisions": decisions}}))
        return 0

    decision = policy.select(
        RoutingRequest(
            language=args.language,
            style=args.style,
            tier=args.tier,
            rate=args.rate,
        ),
        available=available,
        capabilities=capabilities,
        scores=scores,
    )
    payload = {
        "request": {
            "language": args.language,
            "style": args.style,
            "tier": args.tier,
            "rate": args.rate,
        },
        **decision.as_dict(),
    }
    if args.json:
        _print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    else:
        _print(f"engine: {decision.engine}")
        _print(f"rule:   {decision.rule}")
        _print(f"reason: {decision.reason}")
        if decision.degraded:
            _print("degraded: yes")
        for item in decision.considered:
            if item["outcome"] != "no-match":
                _print(f"  - {item['rule']}: {item['outcome']} {item.get('reason', '')}".rstrip())
    return 0


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.eval",
        description=(
            "Local, offline evaluation harness for the Reactory speech platform. "
            "No network, no paid service: every model is local."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    _build_run_parser(subparsers)
    _build_report_parser(subparsers)
    _build_compare_parser(subparsers)
    _build_route_parser(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "run":
            return _run(args)
        if args.command == "report":
            return _report(args)
        if args.command == "compare":
            return _compare(args)
        if args.command == "route":
            return _route(args)
    except (CorpusError, EvalEngineError, TranscriberUnavailable, RoutingError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    parser.error(f"unknown command {args.command!r}")  # pragma: no cover
    return 2  # pragma: no cover


__all__ = ["build_parser", "main"]
