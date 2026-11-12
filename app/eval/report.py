"""Report schema, human-readable rendering and baseline diffing.

A report is the durable artifact of a run: JSON, no wall-clock unless ``--stamp``
is passed, all floats rounded to fixed precision, keys sorted. That is what makes
"is version N+1 better than N" a diff instead of an opinion.

Two classes of finding are separated on purpose:

* **integrity** — the render is broken: silent audio, truncated audio, a sample
  rate the engine does not advertise, undecodable bytes, a synthesis exception.
  An integrity finding is a harness failure and always exits non-zero.
* **quality** — the render is well-formed but unintelligible (WER above the
  entry's target). Recorded always, gates the exit code only with
  ``--fail-on-wer`` so the harness can be adopted before the multilingual bugs
  (SPEECH-010) are fixed without pretending they are not there.

Metric tolerances are declared in :data:`TOLERANCES` and copied into every report
so a diff is judged by the rules that were in force when it was written.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

#: Report format identifier. Bump when a consumer would have to change.
SCHEMA = "reactory.speech.eval.report/1"

#: Checks that mean the render is broken, not merely poor.
INTEGRITY_CHECKS: frozenset[str] = frozenset(
    {
        "synthesis_error",
        "undecodable_audio",
        "no_audio",
        "silent_output",
        "duration_out_of_bounds",
        "unexpected_sample_rate",
        "unexpected_channels",
        "empty_transcript",
        "transcription_error",
        "probe_error",
        "speaker_inconsistent",
    }
)

#: Checks that mean the render is poor but well-formed.
QUALITY_CHECKS: frozenset[str] = frozenset({"wer_threshold"})

#: Acceptable run-to-run variance, per metric.
#:
#: Quality metrics are deterministic for a deterministic engine: WER/CER only move
#: when the audio or the model changes, so their tolerance is one token of
#: measurement noise (0.02 of a rate). Timing and memory metrics depend on the
#: machine, the OS scheduler and the ONNX thread pool; they are allowed a relative
#: tolerance and are documented as not comparable across hosts. Audio duration is
#: deterministic in content but is compared relatively because a 4.2 s narration
#: may legitimately become 4.4 s after a model upgrade.
TOLERANCES: dict[str, dict[str, float | str]] = {
    "wer": {"mode": "absolute", "value": 0.02},
    "cer": {"mode": "absolute", "value": 0.02},
    "wer_mean": {"mode": "absolute", "value": 0.02},
    "cer_mean": {"mode": "absolute", "value": 0.02},
    "duration_s": {"mode": "relative", "value": 0.05},
    "duration_ratio": {"mode": "relative", "value": 0.08},
    "rtf": {"mode": "relative", "value": 0.15},
    "ttfa_s": {"mode": "relative", "value": 0.20},
    "peak_memory_mb": {"mode": "relative", "value": 0.15},
    "peak_memory_mb_max": {"mode": "relative", "value": 0.15},
    "speaker_consistency_mean": {"mode": "absolute", "value": 0.02},
    "speaker_consistency_min": {"mode": "absolute", "value": 0.02},
    "prosody_adherence": {"mode": "relative", "value": 0.05},
}

#: Per-entry metrics written into the report, in table order.
ENTRY_METRICS: tuple[str, ...] = ("wer", "cer", "duration_s", "duration_ratio", "rtf", "peak_memory_mb")

#: Metrics where neither direction is inherently better. A duration shift beyond
#: tolerance is a *change* worth a human look (the engine started dropping or adding
#: audio), but calling it an improvement or a regression would be a lie: a model
#: upgrade may legitimately change pacing.
NEUTRAL_METRICS: frozenset[str] = frozenset({"duration_s", "duration_ratio", "duration_ratio_mean"})

#: Metrics a regression gate checks by default.
#:
#: Only the metrics that are reproducible for a deterministic engine: audio content,
#: its duration, voice stability and rate adherence. ``rtf``, ``ttfa_s`` and
#: ``peak_memory_mb`` depend on the machine and its load — two runs on a busy laptop
#: differ by tens of percent — so they are reported and diffed but do not gate a
#: release unless ``--metrics all`` asks for them.
DEFAULT_GATE_METRICS: frozenset[str] = frozenset(
    {
        "wer",
        "cer",
        "wer_mean",
        "cer_mean",
        "duration_s",
        "duration_ratio",
        "speaker_consistency_mean",
        "speaker_consistency_min",
        "prosody_adherence",
    }
)

#: Cell-level metrics written into the report.
CELL_METRICS: tuple[str, ...] = (
    "wer_mean",
    "cer_mean",
    "duration_ratio_mean",
    "rtf_mean",
    "peak_memory_mb_max",
    "ttfa_s",
    "speaker_consistency_mean",
    "speaker_consistency_min",
    "prosody_adherence",
)

METRIC_DEFINITIONS: dict[str, str] = {
    "wer": (
        "Word error rate of the local STT transcript against the corpus sentence, after Unicode "
        "NFKC normalisation, case folding and punctuation removal. For Japanese/Chinese the "
        "tokens are characters, so WER equals a character-level rate there; prefer `cer`."
    ),
    "cer": "Character error rate of the transcript against the source, punctuation removed.",
    "duration_s": "Audio duration measured from the WAV frame count and header sample rate.",
    "duration_ratio": "duration_s divided by the nominal duration of the sentence at rate 1.0.",
    "rtf": (
        "Real-time factor: audio seconds produced per wall-clock synthesis second. Higher is "
        "faster; 1.0 is real time."
    ),
    "ttfa_s": (
        "Time to the first streamed audio chunk, measured on the streaming path. Kokoro streams a "
        "sentence at a time, so this is the latency to the first sentence, not to the first word."
    ),
    "peak_memory_mb": (
        "Peak Python heap allocation during one render (tracemalloc). Best effort: native "
        "allocations inside ONNX Runtime are not attributed. With --memory-mode rss this is the "
        "process high-water RSS instead, which is not per-render."
    ),
    "speaker_consistency_mean": (
        "Mean pairwise cosine similarity between the speaker embeddings of repeated renders of "
        "the same voice and text. 1.0 means a perfectly stable voice."
    ),
    "speaker_consistency_min": "Worst pairwise cosine similarity across the repeated renders.",
    "prosody_adherence": (
        "measured_rate / requested_rate, where measured_rate = base_duration / requested_duration. "
        "1.0 means the engine honoured the rate exactly; ~0.67 means a 1.5x request was ignored."
    ),
}


def _round(value: Any, digits: int = 4) -> Any:
    """Round a float, leaving ``None``, ints and non-numbers untouched."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return round(float(value), digits)
    return value


def build_report(
    *,
    corpus: Any,
    cells: list[dict[str, Any]],
    routing: dict[str, Any] | None = None,
    engines: Iterable[dict[str, Any]] = (),
    transcriber: dict[str, Any] | None = None,
    embedder: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    environment: dict[str, Any] | None = None,
    generated_at: str | None = None,
    skipped: Iterable[dict[str, Any]] = (),
) -> dict[str, Any]:
    """Assemble the report payload.

    Rounding happens here and nowhere else: every producer hands raw floats in, so
    there is one place to change the precision and no chance of a metric being
    rounded twice.
    """
    rounded_cells: list[dict[str, Any]] = []
    for cell in cells:
        entries = []
        for entry in cell.get("entries", []):
            entries.append(
                {
                    "id": entry.get("id"),
                    "use": entry.get("use"),
                    "text_digest": entry.get("text_digest"),
                    "status": entry.get("status", "ok"),
                    "error": entry.get("error"),
                    "checks": list(entry.get("checks", [])),
                    "transcript": entry.get("transcript"),
                    "wer": _round(entry.get("wer")),
                    "cer": _round(entry.get("cer")),
                    "wer_counts": entry.get("wer_counts"),
                    "cer_counts": entry.get("cer_counts"),
                    "duration_s": _round(entry.get("duration_s"), 3),
                    "expected_duration_s": _round(entry.get("expected_duration_s"), 3),
                    "duration_min_s": _round(entry.get("duration_min_s"), 3),
                    "duration_max_s": _round(entry.get("duration_max_s"), 3),
                    "duration_ratio": _round(entry.get("duration_ratio")),
                    "synthesis_s": _round(entry.get("synthesis_s")),
                    "rtf": _round(entry.get("rtf")),
                    "peak_memory_mb": _round(entry.get("peak_memory_mb"), 2),
                    "sample_rate": entry.get("sample_rate"),
                    "channels": entry.get("channels"),
                    "peak": _round(entry.get("peak")),
                    "rms": _round(entry.get("rms")),
                    "silent": entry.get("silent"),
                }
            )
        summary = {key: _round(value) for key, value in (cell.get("summary") or {}).items()}
        rounded_cells.append(
            {
                "engine": cell.get("engine"),
                "voice": cell.get("voice"),
                "language": cell.get("language"),
                "entries": entries,
                "summary": summary,
                "capabilities": cell.get("capabilities"),
                "streaming": _round_nested(cell.get("streaming")),
                "speaker_consistency": _round_nested(cell.get("speaker_consistency")),
                "prosody": _round_nested(cell.get("prosody")),
            }
        )

    failures: list[dict[str, Any]] = []
    for cell in rounded_cells:
        for entry in cell["entries"]:
            for check in entry["checks"]:
                severity = (
                    "integrity"
                    if check in INTEGRITY_CHECKS
                    else ("quality" if check in QUALITY_CHECKS else "unknown")
                )
                failures.append(
                    {
                        "engine": cell["engine"],
                        "voice": cell["voice"],
                        "entry": entry["id"],
                        "check": check,
                        "severity": severity,
                        "wer": entry["wer"],
                        "duration_s": entry["duration_s"],
                        "sample_rate": entry["sample_rate"],
                        "error": entry["error"],
                    }
                )
        for probe_name in ("streaming", "speaker_consistency", "prosody"):
            probe = cell.get(probe_name) or {}
            for check in probe.get("checks") or []:
                failures.append(
                    {
                        "engine": cell["engine"],
                        "voice": cell["voice"],
                        "entry": probe.get("entry"),
                        "check": check,
                        "severity": (
                            "integrity" if check in INTEGRITY_CHECKS else "quality"
                        ),
                        "wer": None,
                        "duration_s": None,
                        "sample_rate": None,
                        "error": probe.get("error") or probe.get("reason"),
                    }
                )

    # Two probes can fail for the same reason on the same cell (the speaker and
    # prosody probes both render the same sentence). Report the finding once, so the
    # failure count is the number of defects rather than the number of attempts.
    deduped: list[dict[str, Any]] = []
    seen_failures: set[tuple[Any, ...]] = set()
    for failure in failures:
        key = (failure["engine"], failure["voice"], failure["entry"], failure["check"])
        if key in seen_failures:
            continue
        seen_failures.add(key)
        deduped.append(failure)
    failures = deduped

    integrity = [failure for failure in failures if failure["severity"] == "integrity"]
    quality = [failure for failure in failures if failure["severity"] == "quality"]

    return {
        "schema": SCHEMA,
        "generated_at": generated_at,
        "meta": {
            "corpus": {
                "path": str(corpus.path),
                "schema_version": corpus.schema_version,
                "digest": corpus.digest,
                "entries": len(corpus),
                "languages": corpus.languages,
                "uses": corpus.uses(),
            },
            "engines": list(engines),
            "transcriber": transcriber or {"enabled": False},
            "speaker_embedder": embedder or {},
            "config": config or {},
            "environment": environment or {},
            "metric_definitions": METRIC_DEFINITIONS,
            "tolerances": TOLERANCES,
        },
        "cells": rounded_cells,
        "routing": routing,
        "failures": failures,
        "skipped": list(skipped),
        "status": {
            "cells": len(rounded_cells),
            "cells_skipped": len(list(skipped)),
            "entries": sum(len(cell["entries"]) for cell in rounded_cells),
            "integrity_failures": len(integrity),
            "quality_failures": len(quality),
            "passed": not integrity,
        },
    }


def _round_nested(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    result: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, float):
            result[key] = _round(value)
        elif isinstance(value, dict):
            result[key] = _round_nested(value)
        elif isinstance(value, list):
            result[key] = value
        else:
            result[key] = value
    return result


def write_report(report: dict[str, Any], path: str | Path) -> Path:
    """Write a report as stable, sorted, newline-terminated JSON."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return target


def load_report(path: str | Path) -> dict[str, Any]:
    """Load a report, rejecting a payload written by an incompatible schema."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ValueError(
            f"{path} is not a {SCHEMA} report (found {payload.get('schema') if isinstance(payload, dict) else type(payload).__name__})"
        )
    return payload


def is_pass(report: dict[str, Any], *, fail_on_wer: bool = False) -> bool:
    """Whether the run passed.

    Integrity failures always fail. Quality failures fail only when the caller
    opted into the WER gate.
    """
    status = report.get("status") or {}
    if status.get("integrity_failures"):
        return False
    if fail_on_wer and status.get("quality_failures"):
        return False
    return True


# ----------------------------------------------------------------------------
# Human-readable rendering
# ----------------------------------------------------------------------------


def _format(value: Any, width: int, digits: int | None = None) -> str:
    if value is None:
        return "-".rjust(width)
    if isinstance(value, bool):
        return ("yes" if value else "no").rjust(width)
    if isinstance(value, float):
        text = f"{value:.{digits}f}" if digits is not None else f"{value:.3f}"
    else:
        text = str(value)
    return text[:width].rjust(width)


def render_table(report: dict[str, Any]) -> str:
    """Render the per-voice comparison table.

    One row per engine+voice cell, so a reader can compare engines at a glance.
    The raw numbers stay in the JSON; this is for humans and for a PR comment.
    """
    headers = [
        ("engine", 10, None),
        ("voice", 14, None),
        ("lang", 5, None),
        ("n", 4, None),
        ("wer", 7, 3),
        ("cer", 7, 3),
        ("dur/x", 7, 2),
        ("rtf", 7, 2),
        ("ttfa ms", 8, 0),
        ("mem MB", 8, 1),
        ("spk cos", 8, 3),
        ("rate x", 7, 3),
        ("status", 10, None),
    ]
    lines = [" ".join(_format(name, width) for name, width, _ in headers)]
    lines.append(" ".join("-" * width for _, width, _ in headers))

    for cell in report.get("cells", []):
        summary = cell.get("summary") or {}
        streaming = cell.get("streaming") or {}
        speaker = cell.get("speaker_consistency") or {}
        prosody = cell.get("prosody") or {}
        checks = [check for entry in cell.get("entries", []) for check in entry.get("checks", [])]
        for probe_name in ("streaming", "speaker_consistency", "prosody"):
            checks.extend((cell.get(probe_name) or {}).get("checks") or [])
        status = "ok"
        if any(check in INTEGRITY_CHECKS for check in checks):
            status = "broken"
        elif any(check in QUALITY_CHECKS for check in checks):
            status = "poor"
        ttfa_ms = streaming.get("ttfa_s")
        values = [
            _format(cell.get("engine"), 10),
            _format(cell.get("voice"), 14),
            _format(cell.get("language"), 5),
            _format(len(cell.get("entries", [])), 4),
            _format(summary.get("wer_mean"), 7, 3),
            _format(summary.get("cer_mean"), 7, 3),
            _format(summary.get("duration_ratio_mean"), 7, 2),
            _format(summary.get("rtf_mean"), 7, 2),
            _format(None if ttfa_ms is None else ttfa_ms * 1000.0, 8, 0),
            _format(summary.get("peak_memory_mb_max"), 8, 1),
            _format(speaker.get("mean"), 8, 3),
            _format(prosody.get("adherence"), 7, 3),
            _format(status, 10),
        ]
        lines.append(" ".join(values))

    status = report.get("status") or {}
    lines.append("")
    # ``passed`` here means "the audio is structurally sound". Quality failures
    # are separate and are named explicitly: printing a bare ``passed=yes`` beside
    # ``quality_failures=3`` reads as an endorsement of languages that did not
    # round-trip, which is how SPEECH-010's Mandarin claim survived review.
    quality_failures = status.get("quality_failures", 0)
    verdict = "yes" if status.get("passed") else "NO"
    if status.get("passed") and quality_failures:
        verdict = f"yes (audio integrity only; {quality_failures} quality failures below)"
    lines.append(
        f"cells={status.get('cells', 0)} entries={status.get('entries', 0)} "
        f"integrity_failures={status.get('integrity_failures', 0)} "
        f"quality_failures={quality_failures} "
        f"passed={verdict}"
    )
    if status.get("cells_skipped"):
        lines.append(
            f"cells_skipped={status['cells_skipped']} (languages the text front end "
            "cannot phonemise; listed above)"
        )
    transcriber = (report.get("meta") or {}).get("transcriber") or {}
    if not transcriber.get("enabled", False):
        lines.append("round-trip WER/CER not measured: local STT disabled (--no-stt)")
    return "\n".join(lines)


def render_failures(report: dict[str, Any]) -> str:
    """Render the failure list, integrity findings first."""
    failures = report.get("failures") or []
    if not failures:
        return "no failures"
    ordered = sorted(
        failures,
        key=lambda failure: (
            failure.get("severity") != "integrity",
            str(failure.get("engine")),
            str(failure.get("voice")),
            str(failure.get("entry")),
            str(failure.get("check")),
        ),
    )
    lines = []
    for failure in ordered:
        detail = f"{failure.get('engine')}/{failure.get('voice')}"
        if failure.get("entry"):
            detail += f" {failure['entry']}"
        detail += f": {failure.get('check')}"
        if failure.get("wer") is not None:
            detail += f" (wer={failure['wer']})"
        if failure.get("duration_s") is not None:
            detail += f" (duration={failure['duration_s']}s)"
        if failure.get("error"):
            detail += f" — {failure['error']}"
        lines.append(f"[{failure.get('severity')}] {detail}")
    return "\n".join(lines)


def render_routing(report: dict[str, Any]) -> str:
    """Render the routing decisions embedded in a report."""
    routing = report.get("routing") or {}
    decisions = routing.get("decisions") or []
    if not decisions:
        return ""
    headers = [("language", 9), ("style", 12), ("tier", 9), ("engine", 12), ("rule", 24)]
    lines = [" ".join(_format(name, width) for name, width in headers)]
    lines.append(" ".join("-" * width for _, width in headers))
    for decision in decisions:
        lines.append(
            " ".join(
                [
                    _format(decision.get("language"), 9),
                    _format(decision.get("style"), 12),
                    _format(decision.get("tier"), 9),
                    _format(decision.get("engine"), 12),
                    _format(decision.get("rule"), 24),
                ]
            )
        )
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# Baseline diffing
# ----------------------------------------------------------------------------


def _cell_metric(cell: dict[str, Any], metric: str) -> Any:
    """Read one cell-level metric from the sub-object that owns it."""
    if metric in ("wer_mean", "cer_mean", "duration_ratio_mean", "rtf_mean", "peak_memory_mb_max"):
        return (cell.get("summary") or {}).get(metric)
    if metric == "ttfa_s":
        return (cell.get("streaming") or {}).get("ttfa_s")
    if metric == "speaker_consistency_mean":
        return (cell.get("speaker_consistency") or {}).get("mean")
    if metric == "speaker_consistency_min":
        return (cell.get("speaker_consistency") or {}).get("min")
    if metric == "prosody_adherence":
        return (cell.get("prosody") or {}).get("adherence")
    return None  # pragma: no cover - CELL_METRICS is exhaustive above


def gate_regressions(
    diff: dict[str, Any], metrics: frozenset[str] | set[str] | None = None
) -> list[dict[str, Any]]:
    """The regressions that should fail a release gate.

    ``metrics=None`` selects :data:`DEFAULT_GATE_METRICS`; an empty set selects all
    metrics (the caller asked for everything).
    """
    selected = DEFAULT_GATE_METRICS if metrics is None else metrics
    findings = [
        delta
        for delta in diff.get("deltas") or []
        if delta.get("status") in ("regression", "changed")
    ]
    if not selected:
        return findings
    return [delta for delta in findings if delta.get("metric") in selected]


def _metric_entries(report: dict[str, Any]) -> Iterator[tuple[str, str, float]]:
    """Yield ``(scope_key, metric, value)`` for every numeric metric in a report."""
    for cell in report.get("cells", []):
        prefix = f"{cell.get('engine')}/{cell.get('voice')}"
        for entry in cell.get("entries", []):
            for metric in ENTRY_METRICS:
                value = entry.get(metric)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    yield f"{prefix}/{entry.get('id')}", metric, float(value)
        for metric in CELL_METRICS:
            value = _cell_metric(cell, metric)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                yield prefix, metric, float(value)


def _tolerance(metric: str, tolerances: dict[str, dict[str, float | str]] | None = None) -> dict[str, float | str]:
    table = tolerances or TOLERANCES
    return table.get(metric, {"mode": "relative", "value": 0.10})


def _within_tolerance(
    metric: str,
    baseline: float,
    candidate: float,
    tolerances: dict[str, dict[str, float | str]] | None = None,
) -> bool:
    tolerance = _tolerance(metric, tolerances)
    if tolerance["mode"] == "absolute":
        return abs(candidate - baseline) <= float(tolerance["value"])
    scale = abs(baseline) if abs(baseline) > 1e-9 else 1.0
    return abs(candidate - baseline) / scale <= float(tolerance["value"])


def _comparability(candidate: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    candidate_meta = candidate.get("meta") or {}
    baseline_meta = baseline.get("meta") or {}

    candidate_corpus = candidate_meta.get("corpus") or {}
    baseline_corpus = baseline_meta.get("corpus") or {}
    if candidate_corpus.get("digest") != baseline_corpus.get("digest"):
        reasons.append(
            "corpus digest differs: the baseline measured different sentences. "
            "Regenerate the baseline for the new corpus."
        )

    def engine_ids(meta: dict[str, Any]) -> list[str]:
        return sorted(str(engine.get("id")) for engine in meta.get("engines") or [])

    if engine_ids(candidate_meta) != engine_ids(baseline_meta):
        reasons.append(
            f"engine set differs: candidate={engine_ids(candidate_meta)} "
            f"baseline={engine_ids(baseline_meta)}"
        )

    candidate_embedder = (candidate_meta.get("speaker_embedder") or {}).get("id")
    baseline_embedder = (baseline_meta.get("speaker_embedder") or {}).get("id")
    if candidate_embedder != baseline_embedder:
        reasons.append(
            f"speaker embedder differs: {candidate_embedder!r} vs {baseline_embedder!r}; "
            "cosine values are not comparable"
        )

    candidate_stt = candidate_meta.get("transcriber") or {}
    baseline_stt = baseline_meta.get("transcriber") or {}
    if bool(candidate_stt.get("enabled")) != bool(baseline_stt.get("enabled")):
        reasons.append("round-trip scoring enabled in one report but not the other")
    elif candidate_stt.get("enabled") and candidate_stt.get("model") != baseline_stt.get("model"):
        reasons.append(
            f"Whisper model differs: {candidate_stt.get('model')!r} vs "
            f"{baseline_stt.get('model')!r}; WER is not comparable"
        )
    return reasons


def diff_reports(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    *,
    tolerances: dict[str, dict[str, float | str]] | None = None,
) -> dict[str, Any]:
    """Compare a report with a baseline.

    Returns ``comparable`` plus per-metric deltas classified as ``regression``,
    ``improvement``, ``unchanged``, ``new`` or ``missing``. A regression is a
    candidate value that is worse than the baseline *beyond* its tolerance, where
    "worse" means higher for error rates, memory and latency and lower for
    consistency. A candidate metric outside tolerance in the *good* direction is an
    improvement, not a regression.
    """
    # Caller-supplied tolerances win per metric; the report already wrote the ones
    # that were in force when it was produced.
    effective_tolerances = {**TOLERANCES, **(tolerances or {})}

    reasons = _comparability(candidate, baseline)
    baseline_metrics = {(scope, metric): value for scope, metric, value in _metric_entries(baseline)}
    candidate_metrics = {(scope, metric): value for scope, metric, value in _metric_entries(candidate)}

    # Metrics where a larger number is worse.
    higher_is_worse = {
        "wer",
        "cer",
        "wer_mean",
        "cer_mean",
        "rtf",
        "ttfa_s",
        "peak_memory_mb",
        "peak_memory_mb_max",
    }

    deltas: list[dict[str, Any]] = []
    for key in sorted(set(baseline_metrics) | set(candidate_metrics)):
        scope, metric = key
        baseline_value = baseline_metrics.get(key)
        candidate_value = candidate_metrics.get(key)
        if baseline_value is None:
            deltas.append(
                {
                    "scope": scope,
                    "metric": metric,
                    "baseline": None,
                    "candidate": candidate_value,
                    "delta": None,
                    "status": "new",
                }
            )
            continue
        if candidate_value is None:
            deltas.append(
                {
                    "scope": scope,
                    "metric": metric,
                    "baseline": baseline_value,
                    "candidate": None,
                    "delta": None,
                    "status": "missing",
                }
            )
            continue

        delta = candidate_value - baseline_value
        if _within_tolerance(metric, baseline_value, candidate_value, effective_tolerances):
            status = "unchanged"
        elif metric in NEUTRAL_METRICS:
            status = "changed"
        elif metric in higher_is_worse:
            status = "regression" if delta > 0 else "improvement"
        else:
            status = "regression" if delta < 0 else "improvement"
        deltas.append(
            {
                "scope": scope,
                "metric": metric,
                "baseline": round(baseline_value, 4),
                "candidate": round(candidate_value, 4),
                "delta": round(delta, 4),
                "tolerance": _tolerance(metric, effective_tolerances),
                "status": status,
            }
        )

    regressions = [delta for delta in deltas if delta["status"] == "regression"]
    improvements = [delta for delta in deltas if delta["status"] == "improvement"]
    return {
        "comparable": not reasons,
        "reasons": reasons,
        "deltas": deltas,
        "regressions": regressions,
        "improvements": improvements,
        "changed": [delta for delta in deltas if delta["status"] == "changed"],
        "missing": [delta for delta in deltas if delta["status"] == "missing"],
        "new": [delta for delta in deltas if delta["status"] == "new"],
    }


def render_diff(diff: dict[str, Any], *, limit: int = 40) -> str:
    """Render a :func:`diff_reports` result for a human."""
    lines: list[str] = []
    if not diff["comparable"]:
        lines.append("NOT COMPARABLE:")
        lines.extend(f"  - {reason}" for reason in diff["reasons"])
    lines.append(
        f"regressions={len(diff['regressions'])} improvements={len(diff['improvements'])} "
        f"changed={len(diff.get('changed') or [])} "
        f"unchanged={sum(1 for delta in diff['deltas'] if delta['status'] == 'unchanged')} "
        f"missing={len(diff['missing'])} new={len(diff['new'])}"
    )
    for title, rows in (
        ("REGRESSIONS", diff["regressions"]),
        ("IMPROVEMENTS", diff["improvements"]),
        ("CHANGED", diff.get("changed") or []),
    ):
        if not rows:
            continue
        lines.append("")
        lines.append(title)
        for delta in rows[:limit]:
            lines.append(
                f"  {delta['scope']:<48} {delta['metric']:<24} "
                f"{delta['baseline']} -> {delta['candidate']} ({delta['delta']:+})"
            )
        if len(rows) > limit:
            lines.append(f"  ... {len(rows) - limit} more")
    return "\n".join(lines)


def scores_by_engine(report: dict[str, Any]) -> dict[str, dict[str, float]]:
    """Mean quality metrics per engine, for the routing policy's score input.

    Averages the cell ``wer_mean``/``cer_mean`` values across voices so one bad
    voice does not silently become the engine's headline number, and exposes the
    per-voice values under ``voices`` for a caller that wants them.
    """
    scores: dict[str, dict[str, Any]] = {}
    for cell in report.get("cells", []):
        engine = str(cell.get("engine"))
        summary = cell.get("summary") or {}
        row = scores.setdefault(engine, {"voices": {}})
        row["voices"][str(cell.get("voice"))] = {
            "wer": summary.get("wer_mean"),
            "cer": summary.get("cer_mean"),
        }

    for engine, row in scores.items():
        for metric in ("wer", "cer"):
            values = [
                voice[metric]
                for voice in row["voices"].values()
                if isinstance(voice[metric], (int, float))
            ]
            row[metric] = round(sum(values) / len(values), 4) if values else None
    return scores


__all__ = [
    "CELL_METRICS",
    "DEFAULT_GATE_METRICS",
    "ENTRY_METRICS",
    "INTEGRITY_CHECKS",
    "METRIC_DEFINITIONS",
    "NEUTRAL_METRICS",
    "QUALITY_CHECKS",
    "SCHEMA",
    "TOLERANCES",
    "build_report",
    "diff_reports",
    "gate_regressions",
    "is_pass",
    "load_report",
    "render_diff",
    "render_failures",
    "render_routing",
    "render_table",
    "scores_by_engine",
    "write_report",
]
