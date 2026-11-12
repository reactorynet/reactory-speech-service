"""The routing policy is data, and it must always answer.

The policy is the artifact a reviewer argues with, so the tests assert the
documented behaviour: a rule match, a capability gate, a measured-score gate, a
clean degradation to the default engine, and an answer for every corpus
combination.
"""

from __future__ import annotations

import json

import pytest

from app.eval.corpus import load_corpus
from app.eval.routing import (
    RoutingError,
    RoutingPolicy,
    RoutingRequest,
    load_policy,
)

KOKORO_CAPABILITIES = {
    "native_rate": True,
    "native_pitch": False,
    "native_volume": False,
    "supports_streaming": True,
    "styles": ["neutral"],
}

CHATTERBOX_CAPABILITIES = {
    "native_rate": False,
    "native_pitch": False,
    "native_volume": False,
    "supports_streaming": True,
    "styles": ["neutral", "empathetic", "excited", "cheerful", "whisper"],
}


def test_committed_policy_loads_and_is_valid() -> None:
    policy = load_policy()
    assert policy.default_engine == "kokoro"
    assert policy.rules, "policy has no rules"
    assert len({rule["id"] for rule in policy.rules}) == len(policy.rules)
    for rule in policy.rules:
        assert rule["engine"]
        assert isinstance(rule.get("when") or {}, dict)


def test_no_matching_rule_degrades_to_the_default_engine() -> None:
    policy = load_policy()
    decision = policy.select(
        RoutingRequest(language="en", style="neutral", tier="standard"),
        available=["kokoro"],
        capabilities={"kokoro": KOKORO_CAPABILITIES},
    )
    assert decision.engine == "kokoro"
    assert decision.rule == "default"
    assert decision.degraded is False
    assert decision.reason


def test_rule_matches_by_language() -> None:
    policy = load_policy()
    decision = policy.select(
        RoutingRequest(language="ja"),
        available=["kokoro"],
        capabilities={"kokoro": KOKORO_CAPABILITIES},
    )
    assert decision.engine == "kokoro"
    assert decision.rule == "japanese-chinese-coverage"


def test_rule_matches_by_requested_rate() -> None:
    policy = load_policy()
    decision = policy.select(
        RoutingRequest(language="en", rate=1.5),
        available=["kokoro"],
        capabilities={"kokoro": KOKORO_CAPABILITIES},
    )
    assert decision.rule == "native-rate"


def test_expressive_style_routes_to_an_expressive_engine_when_deployed() -> None:
    policy = load_policy()
    decision = policy.select(
        RoutingRequest(language="en", style="empathetic"),
        available=["kokoro", "chatterbox"],
        capabilities={
            "kokoro": KOKORO_CAPABILITIES,
            "chatterbox": CHATTERBOX_CAPABILITIES,
        },
    )
    assert decision.engine == "chatterbox"
    assert decision.rule == "expressive-style"


def test_expressive_style_degrades_cleanly_when_the_engine_is_absent() -> None:
    policy = load_policy()
    decision = policy.select(
        RoutingRequest(language="en", style="empathetic"),
        available=["kokoro"],
        capabilities={"kokoro": KOKORO_CAPABILITIES},
    )
    assert decision.engine == "kokoro"
    assert decision.degraded is True
    skipped = [item for item in decision.considered if item["outcome"] == "skipped"]
    assert skipped and "not deployed" in skipped[0]["reason"]


def test_capability_gate_rejects_an_engine_that_cannot_honour_the_style() -> None:
    """A rule naming an engine whose capabilities do not match must not fire."""
    policy = load_policy()
    decision = policy.select(
        RoutingRequest(language="en", style="empathetic"),
        available=["kokoro", "chatterbox"],
        capabilities={
            "kokoro": KOKORO_CAPABILITIES,
            # Chatterbox is deployed but cannot express emotion in this profile.
            "chatterbox": {**CHATTERBOX_CAPABILITIES, "styles": ["neutral"]},
        },
    )
    assert decision.engine == "kokoro"
    assert decision.degraded is True


def test_score_gate_does_not_fire_without_measurements() -> None:
    policy = load_policy()
    decision = policy.select(
        RoutingRequest(language="en", tier="premium"),
        available=["kokoro", "chatterbox"],
        capabilities={
            "kokoro": KOKORO_CAPABILITIES,
            "chatterbox": CHATTERBOX_CAPABILITIES,
        },
    )
    assert decision.engine == "kokoro"
    skipped = [item for item in decision.considered if item["outcome"] == "skipped"]
    assert any("cannot fire blind" in item["reason"] for item in skipped)


def test_score_gate_promotes_a_measured_winner() -> None:
    policy = load_policy()
    decision = policy.select(
        RoutingRequest(language="en", tier="premium"),
        available=["kokoro", "chatterbox"],
        capabilities={
            "kokoro": KOKORO_CAPABILITIES,
            "chatterbox": CHATTERBOX_CAPABILITIES,
        },
        scores={"kokoro": {"wer": 0.22}, "chatterbox": {"wer": 0.11}},
    )
    assert decision.engine == "chatterbox"
    assert decision.rule == "premium-measured-winner"


def test_score_gate_keeps_the_default_when_the_candidate_is_worse() -> None:
    policy = load_policy()
    decision = policy.select(
        RoutingRequest(language="en", tier="premium"),
        available=["kokoro", "chatterbox"],
        capabilities={
            "kokoro": KOKORO_CAPABILITIES,
            "chatterbox": CHATTERBOX_CAPABILITIES,
        },
        scores={"kokoro": {"wer": 0.11}, "chatterbox": {"wer": 0.31}},
    )
    assert decision.engine == "kokoro"
    skipped = [item for item in decision.considered if item["outcome"] == "skipped"]
    assert any("does not beat" in item["reason"] for item in skipped)


def test_unknown_combination_falls_back_cleanly() -> None:
    policy = load_policy()
    decision = policy.select(
        RoutingRequest(language="xx", style="sarcastic", tier="platinum"),
        available=["kokoro"],
        capabilities={"kokoro": KOKORO_CAPABILITIES},
    )
    assert decision.engine == "kokoro"
    assert decision.rule == "default"


def test_falls_back_through_the_chain_when_the_default_is_absent() -> None:
    policy = load_policy()
    decision = policy.select(
        RoutingRequest(language="en"),
        available=["chatterbox"],
        capabilities={"chatterbox": CHATTERBOX_CAPABILITIES},
    )
    assert decision.engine == "chatterbox"
    assert decision.degraded is True


def test_no_available_engine_is_an_error() -> None:
    policy = load_policy()
    with pytest.raises(RoutingError, match="no engine is available"):
        policy.select(RoutingRequest(), available=[])


def test_matrix_covers_every_corpus_language_and_always_answers() -> None:
    policy = load_policy()
    languages = load_corpus().languages
    decisions = policy.matrix(
        languages,
        available=["kokoro"],
        capabilities={"kokoro": KOKORO_CAPABILITIES},
    )
    assert {decision["language"] for decision in decisions} == set(languages)
    for decision in decisions:
        assert decision["engine"]
        assert decision["rule"]
        assert decision["reason"]


def test_policy_change_is_a_data_change(tmp_path) -> None:
    payload = json.loads(load_policy().path.read_text(encoding="utf-8"))
    payload["rules"].insert(
        0,
        {
            "id": "french-uses-chatterbox",
            "when": {"language": ["fr"]},
            "engine": "chatterbox",
            "rationale": "A test rule: proves a routing change needs no code change.",
        },
    )
    custom = tmp_path / "routing.json"
    custom.write_text(json.dumps(payload), encoding="utf-8")

    decision = RoutingPolicy.load(custom).select(
        RoutingRequest(language="fr"),
        available=["kokoro", "chatterbox"],
        capabilities={
            "kokoro": KOKORO_CAPABILITIES,
            "chatterbox": CHATTERBOX_CAPABILITIES,
        },
    )
    assert decision.engine == "chatterbox"
    assert decision.rule == "french-uses-chatterbox"


def test_invalid_policies_are_rejected(tmp_path) -> None:
    base = json.loads(load_policy().path.read_text(encoding="utf-8"))

    no_default = dict(base)
    no_default.pop("default_engine")
    path = tmp_path / "no-default.json"
    path.write_text(json.dumps(no_default), encoding="utf-8")
    with pytest.raises(RoutingError, match="default_engine"):
        RoutingPolicy.load(path)

    duplicate = json.loads(json.dumps(base))
    duplicate["rules"] = [duplicate["rules"][0], dict(duplicate["rules"][0])]
    path = tmp_path / "duplicate.json"
    path.write_text(json.dumps(duplicate), encoding="utf-8")
    with pytest.raises(RoutingError, match="duplicate"):
        RoutingPolicy.load(path)

    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(RoutingError, match="valid JSON"):
        RoutingPolicy.load(path)
