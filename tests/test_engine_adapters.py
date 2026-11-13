"""Tests for the engine adapter layer: contracts, registry, lowering and routing."""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from app.config import settings
from app.core.contracts import Direction, EngineCapabilities, Prosody
from app.engines.base import (
    EngineUnavailableError,
    UnknownEngineError,
    lower_direction,
)
from app.engines.kokoro import KokoroEngine
from app.engines.manager import EngineManager
from app.engines.router import EngineRouter
from app.services import voice_registry
from app.services.voice_registry import (
    DEFAULT_REGISTRY_PATH,
    KOKORO_SAMPLE_RATE,
    REQUIRED_VOICE_FIELDS,
    SHIPPED_VOICE_IDS,
    VoiceRegistryError,
    load_voice_registry,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
APP_ROOT = REPO_ROOT / "app"
SCANNED_PACKAGES = ("services", "routers", "engines")


def _kokoro_caps(**overrides) -> EngineCapabilities:
    defaults = {
        "engine": "kokoro",
        "languages": ["en-US"],
        "native_rate": True,
        "native_pitch": False,
        "native_volume": False,
        "styles": ["neutral"],
        "supports_cloning": False,
        "supports_streaming": True,
        "supports_ssml": False,
        "sample_rates": [KOKORO_SAMPLE_RATE],
        "max_chars": 10000,
        "gpu_required": False,
    }
    defaults.update(overrides)
    return EngineCapabilities(**defaults)


class FakeKokoro:
    """Deterministic stand-in for ``kokoro_onnx.Kokoro``.

    The sample count is inversely proportional to speed, as real Kokoro is, so a
    rate change is observable in the rendered bytes without running inference.
    """

    def __init__(self, voice_ids=SHIPPED_VOICE_IDS):
        self.voices = {voice_id: None for voice_id in voice_ids}
        self.calls: list[dict] = []

    def create(self, text, voice, speed=1.0, lang="en-us", is_phonemes=False):
        if voice not in self.voices:
            raise ValueError(f"Voice {voice} not found in available voices")
        if not 0.5 <= speed <= 2.0:
            raise ValueError(f"Speed should be between 0.5 and 2.0, got {speed}")
        self.calls.append({"text": text, "voice": voice, "speed": speed, "lang": lang})
        samples = np.zeros(max(1, round(2400 / speed)), dtype=np.float32)
        return samples, KOKORO_SAMPLE_RATE


class UnavailableEngine:
    """A registered engine that cannot render — used to exercise the fallback list."""

    id = "stub-unavailable"

    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(engine=self.id)

    def voices(self):
        return []

    def synthesize(self, req):
        raise EngineUnavailableError("stub engine has no model")

    def stream(self, req):
        raise EngineUnavailableError("stub engine has no model")
        yield  # pragma: no cover - keeps this a generator


# ─────────────────────────────────────────────────────────────
# Registry loading
# ─────────────────────────────────────────────────────────────

def test_registry_file_is_the_source_and_loads():
    registry = load_voice_registry()

    assert DEFAULT_REGISTRY_PATH.is_file()
    assert registry.schema_version >= 1
    assert registry.default_voice in registry.voices
    # The Kokoro pack is 54 voices.  The optional expressive engine adds two
    # registry entries (standard/turbo) in the same file, but they are not pack
    # voices: SHIPPED_VOICE_IDS is deliberately scoped to the Kokoro engine so
    # every catalogue-size assertion keeps its meaning with the flag off.
    assert len(voice_registry.VOICE_REGISTRY.ids(engine="kokoro")) == 54
    assert len(voice_registry.VOICE_REGISTRY.ids(engine="chatterbox")) == 2
    assert set(SHIPPED_VOICE_IDS) <= set(registry.voices)


def test_every_registry_record_carries_the_required_fields():
    payload = json.loads(DEFAULT_REGISTRY_PATH.read_text(encoding="utf-8"))

    for entry in payload["voices"]:
        missing = [name for name in REQUIRED_VOICE_FIELDS if name not in entry]
        assert missing == [], f"{entry.get('id')} is missing {missing}"

        assert entry["license"] in voice_registry.PERMISSIVE_LICENSES
        assert entry["attribution"]
        assert entry["enabled"] is True
        assert entry["style_spec"]["type"] == "base"
        assert entry["style_spec"]["voice"]
        assert set(entry["supports"]) >= {"rate", "pitch", "style", "clone"}


def test_registry_default_and_quality_data_are_derived_not_duplicated():
    registry = voice_registry.VOICE_REGISTRY

    assert registry.default_voice == voice_registry.default_voice_id()
    assert len(voice_registry.QUALITY_GRADES) == 47
    assert len(voice_registry.UNGRADED_VOICE_IDS) == 7
    for voice_id, grade in voice_registry.QUALITY_GRADES.items():
        assert registry.get(voice_id).quality == grade
    for voice_id in voice_registry.UNGRADED_VOICE_IDS:
        assert registry.get(voice_id).quality is None


def _write_registry(tmp_path: Path, mutate) -> Path:
    payload = json.loads(DEFAULT_REGISTRY_PATH.read_text(encoding="utf-8"))
    mutate(payload)
    path = tmp_path / "voices.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_registry_rejects_missing_required_field(tmp_path):
    def mutate(payload):
        del payload["voices"][0]["license"]

    with pytest.raises(VoiceRegistryError, match="license"):
        load_voice_registry(_write_registry(tmp_path, mutate))


def test_registry_rejects_duplicate_ids(tmp_path):
    def mutate(payload):
        payload["voices"].append(dict(payload["voices"][0]))

    with pytest.raises(VoiceRegistryError, match="duplicate voice id"):
        load_voice_registry(_write_registry(tmp_path, mutate))


def test_registry_rejects_non_permissive_license(tmp_path):
    """The product direction is permissively licensed models only."""

    def mutate(payload):
        payload["voices"][0]["license"] = "CC-BY-NC-4.0"

    with pytest.raises(VoiceRegistryError, match="permissive"):
        load_voice_registry(_write_registry(tmp_path, mutate))


def test_registry_rejects_unknown_default_voice(tmp_path):
    def mutate(payload):
        payload["default_voice"] = "not_a_voice"

    with pytest.raises(VoiceRegistryError, match="default_voice"):
        load_voice_registry(_write_registry(tmp_path, mutate))


# ─────────────────────────────────────────────────────────────
# Effective default voice (KOKORO_DEFAULT_VOICE)
# ─────────────────────────────────────────────────────────────

def test_default_voice_without_configuration_is_the_registry_default():
    assert "af_heart" == voice_registry.VOICE_REGISTRY.default_voice
    assert voice_registry.default_voice_id() == voice_registry.VOICE_REGISTRY.default_voice


def test_configured_default_voice_is_honoured(monkeypatch):
    """B1: KOKORO_DEFAULT_VOICE must not be silently ignored.

    A deployment that configures a default keeps serving that voice; reverting to
    the registry default would change its audio on upgrade.
    """
    monkeypatch.setattr(settings, "kokoro_default_voice", "am_adam")

    assert voice_registry.default_voice_id() == "am_adam"


def test_unconfigured_default_voice_falls_back_to_the_registry(monkeypatch):
    monkeypatch.setattr(settings, "kokoro_default_voice", "")

    assert voice_registry.default_voice_id() == voice_registry.VOICE_REGISTRY.default_voice


def test_unknown_configured_default_voice_falls_back_with_a_warning(monkeypatch, caplog):
    monkeypatch.setattr(settings, "kokoro_default_voice", "zz_not_a_voice")

    with caplog.at_level(logging.WARNING):
        resolved = voice_registry.default_voice_id()

    assert resolved == voice_registry.VOICE_REGISTRY.default_voice
    assert any("KOKORO_DEFAULT_VOICE" in record.message for record in caplog.records)


def test_disabled_configured_default_voice_falls_back(monkeypatch):
    """A registry-disabled voice is a deliberate decision, not a usable default."""
    records = dict(voice_registry.VOICE_REGISTRY.voices)
    records["am_adam"] = dataclasses.replace(records["am_adam"], enabled=False)
    registry = dataclasses.replace(voice_registry.VOICE_REGISTRY, voices=records)
    monkeypatch.setattr(voice_registry, "VOICE_REGISTRY", registry)
    monkeypatch.setattr(settings, "kokoro_default_voice", "am_adam")

    assert voice_registry.default_voice_id() == registry.default_voice


def test_configured_default_voice_from_a_custom_pack_is_kept(monkeypatch):
    """A pack may ship ids the registry does not describe yet."""
    monkeypatch.setattr(settings, "kokoro_default_voice", "legacy_voice")

    assert voice_registry.default_voice_id(available=["legacy_voice"]) == "legacy_voice"
    assert voice_registry.default_voice_id(available=[]) == voice_registry.VOICE_REGISTRY.default_voice


def test_default_voice_env_var_reaches_the_registry():
    """End-to-end: the documented environment variable is actually wired up."""
    env = {**os.environ, "KOKORO_DEFAULT_VOICE": "am_adam"}
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from app.services.voice_registry import default_voice_id; print(default_voice_id())",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "am_adam"


def test_kokoro_adapter_resolves_the_configured_default_voice(monkeypatch):
    """The adapter must ask the registry, so the setting cannot bypass it."""
    monkeypatch.setattr(settings, "kokoro_default_voice", "am_adam")
    engine = KokoroEngine()
    engine.model = FakeKokoro()
    engine.set_ready(True)

    assert engine.resolve_voice(None) == ("am_adam", "en-us")


# ─────────────────────────────────────────────────────────────
# The registry is the only source of voice data
# ─────────────────────────────────────────────────────────────

def test_no_voice_id_literal_appears_in_services_routers_or_engines():
    """The acceptance criterion for this ticket.

    Engine adapters, services and HTTP handlers must ask the registry instead of
    hard-coding voice ids.  A voice id literal in any of those trees fails here.
    """
    offenders: list[str] = []
    for package in SCANNED_PACKAGES:
        for path in (APP_ROOT / package).rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            for voice_id in SHIPPED_VOICE_IDS:
                if re.search(rf"\b{re.escape(voice_id)}\b", source):
                    offenders.append(f"{path.relative_to(APP_ROOT)} contains '{voice_id}'")

    assert offenders == [], "voice ids must only live in app/data/voices.json: " + "; ".join(
        offenders
    )


def test_engine_adapters_do_not_know_about_http_or_request_models():
    forbidden = ("fastapi", "starlette", "app.models", "app.routers")
    offenders: list[str] = []
    for path in (APP_ROOT / "engines").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in source:
                offenders.append(f"{path.name} references {token}")

    assert offenders == []


# ─────────────────────────────────────────────────────────────
# Direction lowering
# ─────────────────────────────────────────────────────────────

def test_default_direction_reports_nothing():
    """A request with no direction must not invent applied controls."""
    plan = lower_direction(Direction(), _kokoro_caps())

    assert plan.applied == {}
    assert plan.ignored == []
    assert plan.warnings == []


def test_rate_is_applied_natively():
    direction = Direction(prosody=Prosody(rate=0.9))

    plan = lower_direction(direction, _kokoro_caps())

    assert plan.applied == {"rate": 0.9}
    assert plan.ignored == []
    assert plan.engine == "kokoro"


def test_pitch_is_ignored_with_a_human_readable_warning():
    direction = Direction(prosody=Prosody(pitch=-2))

    plan = lower_direction(direction, _kokoro_caps())

    assert "pitch" in plan.ignored
    assert "pitch" not in plan.applied
    assert any("pitch" in warning and "-2" in warning for warning in plan.warnings)


def test_volume_is_ignored_with_a_human_readable_warning():
    direction = Direction(prosody=Prosody(volume=3))

    plan = lower_direction(direction, _kokoro_caps())

    assert "volume" in plan.ignored
    assert any("volume" in warning for warning in plan.warnings)


def test_style_is_ignored_because_kokoro_has_no_emotion_control():
    direction = Direction(style="empathetic", intensity=0.8)

    plan = lower_direction(direction, _kokoro_caps())

    assert "style" in plan.ignored
    assert "intensity" in plan.ignored
    assert "style" not in plan.applied
    assert any("empathetic" in warning for warning in plan.warnings)


def test_neutral_style_is_a_no_op_not_a_claim():
    plan = lower_direction(Direction(style="neutral"), _kokoro_caps())

    assert plan.applied == {}
    assert plan.ignored == []


def test_natural_pauses_are_native_but_shaped_pauses_are_ignored():
    natural = lower_direction(Direction(prosody=Prosody(pauses="natural")), _kokoro_caps())
    dramatic = lower_direction(Direction(prosody=Prosody(pauses="dramatic")), _kokoro_caps())

    assert natural.applied == {"pauses": "natural"}
    assert "pauses" in dramatic.ignored
    assert any("pause" in warning for warning in dramatic.warnings)


def test_lowering_uses_native_controls_when_the_engine_has_them():
    caps = _kokoro_caps(native_pitch=True, native_volume=True, styles=["neutral", "calm"])

    plan = lower_direction(
        Direction(style="calm", prosody=Prosody(rate=1.1, pitch=2, volume=-3)),
        caps,
    )

    assert plan.applied == {"rate": 1.1, "pitch": 2, "volume": -3, "style": "calm"}
    assert plan.ignored == []


# ─────────────────────────────────────────────────────────────
# Kokoro adapter
# ─────────────────────────────────────────────────────────────

def test_kokoro_capabilities_are_the_documented_matrix():
    caps = KokoroEngine().capabilities()

    assert caps.engine == "kokoro"
    assert caps.native_rate is True
    assert caps.native_pitch is False
    assert caps.native_volume is False
    assert caps.styles == ["neutral"]
    assert caps.supports_cloning is False
    assert caps.supports_ssml is False
    assert caps.sample_rates == [KOKORO_SAMPLE_RATE]
    assert caps.gpu_required is False
    assert caps.max_chars == 10000


def test_kokoro_clamps_an_out_of_range_rate_and_says_so():
    engine = KokoroEngine()
    direction = Direction(prosody=Prosody.model_construct(rate=5.0))

    plan = engine.lower(direction)

    assert plan.applied["rate"] == 2.0
    assert any("clamped" in warning for warning in plan.warnings)


def test_kokoro_adapter_satisfies_the_tts_engine_protocol():
    from app.core.contracts import TTSEngine

    engine = KokoroEngine()

    assert isinstance(engine, TTSEngine)


# ─────────────────────────────────────────────────────────────
# Engine manager and router
# ─────────────────────────────────────────────────────────────

def test_engine_router_resolves_the_voice_owner():
    kokoro = KokoroEngine()
    router = EngineRouter(EngineManager([kokoro]), order=["kokoro"])

    assert router.resolve(voice=SHIPPED_VOICE_IDS[0]).id == "kokoro"
    assert router.resolve().id == "kokoro"


def test_engine_router_respects_an_explicit_override():
    kokoro = KokoroEngine()
    router = EngineRouter(EngineManager([kokoro]), order=["kokoro"])

    assert router.resolve(engine="kokoro").id == "kokoro"


def test_engine_router_rejects_an_unknown_override():
    router = EngineRouter(EngineManager([KokoroEngine()]), order=["kokoro"])

    with pytest.raises(UnknownEngineError, match="capabilities"):
        router.resolve(engine="chatterbox")


def test_engine_router_produces_an_ordered_fallback_list():
    stub = UnavailableEngine()
    kokoro = KokoroEngine()
    router = EngineRouter(EngineManager([stub, kokoro]), order=[stub.id, "kokoro"])

    default_order = router.candidates()
    voice_owner_first = router.candidates(voice=SHIPPED_VOICE_IDS[0])

    assert [engine.id for engine in default_order] == [stub.id, "kokoro"]
    assert [engine.id for engine in voice_owner_first] == ["kokoro", stub.id]


def test_orchestrator_falls_back_to_the_next_engine_when_one_is_unavailable():
    from app.services.tts_service import TTSService

    service = TTSService()
    service._kokoro = FakeKokoro()
    service._ready = True
    stub = service.engine_manager.register(UnavailableEngine())
    service._router = EngineRouter(service.engine_manager, order=[stub.id, "kokoro"])

    result = service.synthesize_detailed("Hello.")

    assert result.engine == "kokoro"
    assert result.audio.startswith(b"RIFF")
