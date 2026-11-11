"""Data-driven routing policy.

Given ``(language, style, tier, requested rate)`` plus what the deployment can
actually run and what the harness measured, choose an engine. The policy itself is
JSON (``app/eval/policies/routing.json``), so changing a routing decision is a data
change a reviewer can read, not a code change buried in a service.

Structure
---------

* ``default_engine`` — the engine to use when no rule is usable. This is the
  documented degradation path.
* ``fallback_chain`` — ordered engines to try when the default is not deployed.
* ``rules`` — ordered ``{id, when, engine, require, rationale}`` objects.

``when`` predicates are all optional; an absent predicate is a wildcard:

===============  =========================================================
``language``     language codes to match
``style``        style names to match
``tier``         tiers to match
``rate_at_least`` requested rate must be >= this
``rate_at_most``  requested rate must be <= this
===============  =========================================================

``require`` gates are checked against the *candidate* engine and the measured
scores. A gate that cannot be evaluated (no measurement) fails, because a rule that
promotes an engine on measured quality must not fire blind:

===============  =========================================================
``native_rate``  engine reports native rate control
``native_pitch`` engine reports native pitch control
``native_volume``engine reports native volume control
``supports_streaming`` engine reports streaming support
``styles``       engine supports every listed style
``supports_requested_style`` engine supports the style in the request
``max_wer``      measured WER for the candidate must be <= this
``beat``         ``{metric, against, margin}``: candidate must beat the named engine
===============  =========================================================

A rule whose ``when`` matches but whose engine is not deployed (or whose gates fail)
is recorded in ``considered`` as *skipped* with a reason, and evaluation continues.
That "explain every decision" trail is what makes the routing table reviewable.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Policy file format version.
SCHEMA_VERSION = 1

#: Committed policy location, resolved relative to the package.
DEFAULT_POLICY_PATH = Path(__file__).resolve().parent / "policies" / "routing.json"

#: Metrics where a lower number is better, used by the ``beat`` gate.
_LOWER_IS_BETTER = frozenset({"wer", "cer", "wer_mean", "cer_mean", "ttfa_s", "peak_memory_mb"})


class RoutingError(ValueError):
    """The policy is malformed or no engine can serve the request."""


@dataclass(frozen=True)
class RoutingRequest:
    """What the caller wants, independent of which engine can deliver it."""

    language: str | None = "en"
    style: str | None = "neutral"
    tier: str = "standard"
    rate: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "style": self.style,
            "tier": self.tier,
            "rate": self.rate,
        }


@dataclass(frozen=True)
class RoutingDecision:
    """The chosen engine plus the trail that explains the choice."""

    engine: str
    rule: str
    reason: str
    degraded: bool
    considered: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "rule": self.rule,
            "reason": self.reason,
            "degraded": self.degraded,
            "considered": list(self.considered),
        }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RoutingError(message)


def _capability(capabilities: Any, name: str, default: Any = None) -> Any:
    """Read a capability field from an object or a mapping."""
    if capabilities is None:
        return default
    if isinstance(capabilities, Mapping):
        return capabilities.get(name, default)
    return getattr(capabilities, name, default)


class RoutingPolicy:
    """An ordered rule list plus a documented default."""

    def __init__(self, payload: dict[str, Any], path: Path | None = None) -> None:
        _require(isinstance(payload, dict), "routing policy must be a JSON object")
        _require(
            int(payload.get("schema_version", SCHEMA_VERSION)) == SCHEMA_VERSION,
            f"unsupported routing policy schema_version {payload.get('schema_version')!r}",
        )
        self.path = path or DEFAULT_POLICY_PATH
        self.description = str(payload.get("description", ""))
        self.default_engine = str(payload.get("default_engine") or "")
        _require(bool(self.default_engine), "routing policy has no default_engine")
        self.fallback_chain = [str(engine) for engine in payload.get("fallback_chain") or []]

        raw_rules = payload.get("rules") or []
        _require(isinstance(raw_rules, list), "routing policy 'rules' must be a list")
        rules: list[dict[str, Any]] = []
        seen: set[str] = set()
        for index, rule in enumerate(raw_rules):
            _require(isinstance(rule, dict), f"routing rule #{index} is not an object")
            rule_id = str(rule.get("id") or "")
            _require(bool(rule_id), f"routing rule #{index} has no id")
            _require(rule_id not in seen, f"duplicate routing rule id '{rule_id}'")
            seen.add(rule_id)
            _require(bool(rule.get("engine")), f"routing rule '{rule_id}' names no engine")
            rules.append(rule)
        self.rules = rules

    @classmethod
    def load(cls, path: str | Path | None = None) -> RoutingPolicy:
        """Load and validate a policy file."""
        policy_path = Path(path) if path is not None else DEFAULT_POLICY_PATH
        try:
            payload = json.loads(policy_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise RoutingError(f"routing policy not found: {policy_path}") from exc
        except json.JSONDecodeError as exc:
            raise RoutingError(f"routing policy is not valid JSON: {exc}") from exc
        return cls(payload, path=policy_path)

    # -- matching ----------------------------------------------------------

    @staticmethod
    def _when_matches(rule: dict[str, Any], request: RoutingRequest) -> bool:
        conditions = rule.get("when") or {}
        if not isinstance(conditions, dict):
            return False
        languages = conditions.get("language")
        if languages is not None and (request.language or "") not in {
            str(language).lower() for language in languages
        }:
            return False
        styles = conditions.get("style")
        if styles is not None and (request.style or "neutral").lower() not in {
            str(style).lower() for style in styles
        }:
            return False
        tiers = conditions.get("tier")
        if tiers is not None and request.tier.lower() not in {str(tier).lower() for tier in tiers}:
            return False
        rate = request.rate
        if "rate_at_least" in conditions:
            if rate is None or rate < float(conditions["rate_at_least"]):
                return False
        if "rate_at_most" in conditions:
            if rate is None or rate > float(conditions["rate_at_most"]):
                return False
        return True

    @staticmethod
    def _requirements_unmet(
        rule: dict[str, Any],
        request: RoutingRequest,
        capabilities: Any,
        scores: Mapping[str, Mapping[str, Any]] | None,
    ) -> str | None:
        """Return why the rule's gates fail, or ``None`` when they all pass."""
        require = rule.get("require") or {}
        if not isinstance(require, dict):
            return "rule has a non-object 'require' block"
        for flag in ("native_rate", "native_pitch", "native_volume", "supports_streaming"):
            if flag in require and bool(_capability(capabilities, flag, False)) != bool(
                require[flag]
            ):
                return f"engine does not report {flag}={require[flag]}"
        if "styles" in require:
            supported = {str(style).lower() for style in _capability(capabilities, "styles", []) or []}
            missing = [
                str(style) for style in require["styles"] if str(style).lower() not in supported
            ]
            if missing:
                return f"engine does not support style(s) {missing}"
        if require.get("supports_requested_style"):
            supported = {str(style).lower() for style in _capability(capabilities, "styles", []) or []}
            wanted = (request.style or "neutral").lower()
            if wanted not in supported:
                return f"engine does not support the requested style '{wanted}'"
        if "max_wer" in require:
            value = (scores or {}).get(str(rule["engine"]), {}).get("wer")
            if not isinstance(value, (int, float)):
                return "no measured WER for this engine; a score gate cannot fire blind"
            if float(value) > float(require["max_wer"]):
                return f"measured WER {value} exceeds the rule's {require['max_wer']}"
        if "beat" in require:
            beat = require["beat"] or {}
            metric = str(beat.get("metric") or "")
            against = str(beat.get("against") or "")
            margin = float(beat.get("margin") or 0.0)
            if not metric or not against:
                return "beat gate needs both a metric and an engine to beat"
            candidate_value = (scores or {}).get(str(rule["engine"]), {}).get(metric)
            other_value = (scores or {}).get(against, {}).get(metric)
            if not isinstance(candidate_value, (int, float)):
                return f"no measured {metric} for this engine; a score gate cannot fire blind"
            if not isinstance(other_value, (int, float)):
                return f"no measured {metric} for '{against}'; cannot decide the comparison"
            if metric in _LOWER_IS_BETTER:
                if float(candidate_value) > float(other_value) - margin:
                    return (
                        f"measured {metric} {candidate_value} does not beat {against} "
                        f"({other_value}) by {margin}"
                    )
            elif float(candidate_value) < float(other_value) + margin:
                return (
                    f"measured {metric} {candidate_value} does not beat {against} "
                    f"({other_value}) by {margin}"
                )
        return None

    # -- selection ---------------------------------------------------------

    def select(
        self,
        request: RoutingRequest,
        *,
        available: Iterable[str],
        capabilities: Mapping[str, Any] | None = None,
        scores: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> RoutingDecision:
        """Choose an engine for ``request``.

        Raises :class:`RoutingError` when the deployment has no engine at all — a
        routing policy cannot invent one. Every other path degrades to the default
        and explains itself in ``considered``.
        """
        available_list = [str(engine) for engine in available]
        if not available_list:
            raise RoutingError("no engine is available on this deployment")
        available_set = set(available_list)
        considered: list[dict[str, Any]] = []
        saw_skip = False

        for rule in self.rules:
            rule_id = str(rule["id"])
            if not self._when_matches(rule, request):
                considered.append({"rule": rule_id, "outcome": "no-match"})
                continue
            engine = str(rule["engine"])
            if engine not in available_set:
                saw_skip = True
                considered.append(
                    {
                        "rule": rule_id,
                        "outcome": "skipped",
                        "reason": f"engine '{engine}' is not deployed",
                    }
                )
                continue
            unmet = self._requirements_unmet(
                rule, request, (capabilities or {}).get(engine), scores
            )
            if unmet is not None:
                saw_skip = True
                considered.append({"rule": rule_id, "outcome": "skipped", "reason": unmet})
                continue
            considered.append({"rule": rule_id, "outcome": "selected"})
            return RoutingDecision(
                engine=engine,
                rule=rule_id,
                reason=str(rule.get("rationale") or ""),
                degraded=False,
                considered=tuple(considered),
            )

        engine = self._fallback_engine(available_list)
        reason = (
            "no rule matched; degraded to the default engine"
            if not saw_skip
            else "no matching rule could be honoured; degraded to the default engine"
        )
        considered.append({"rule": "default", "outcome": "selected", "reason": reason})
        return RoutingDecision(
            engine=engine,
            rule="default",
            reason=reason,
            degraded=saw_skip or engine != self.default_engine,
            considered=tuple(considered),
        )

    def _fallback_engine(self, available: Sequence[str]) -> str:
        """The documented degradation order: default, chain, then anything."""
        for engine in [self.default_engine, *self.fallback_chain]:
            if engine in available:
                return engine
        return sorted(available)[0]

    def matrix(
        self,
        languages: Sequence[str],
        *,
        available: Iterable[str],
        capabilities: Mapping[str, Any] | None = None,
        scores: Mapping[str, Mapping[str, Any]] | None = None,
        styles: Sequence[str] = ("neutral", "narration", "empathetic"),
        tiers: Sequence[str] = ("standard", "premium"),
        rates: Sequence[float | None] = (None,),
    ) -> list[dict[str, Any]]:
        """The routing decision for every corpus language x style x tier combination.

        This is the "documented engine for each combination" requirement: the run
        embeds the matrix in the report so the routing answer and the measurements
        that justify it live in one artifact.
        """
        decisions: list[dict[str, Any]] = []
        for language in languages:
            for style in styles:
                for tier in tiers:
                    for rate in rates:
                        request = RoutingRequest(
                            language=language, style=style, tier=tier, rate=rate
                        )
                        decision = self.select(
                            request,
                            available=available,
                            capabilities=capabilities,
                            scores=scores,
                        )
                        decisions.append({**request.as_dict(), **decision.as_dict()})
        return decisions


def load_policy(path: str | Path | None = None) -> RoutingPolicy:
    """Load the routing policy (committed default when ``path`` is omitted)."""
    return RoutingPolicy.load(path)


__all__ = [
    "DEFAULT_POLICY_PATH",
    "RoutingDecision",
    "RoutingError",
    "RoutingPolicy",
    "RoutingRequest",
    "SCHEMA_VERSION",
    "load_policy",
]
