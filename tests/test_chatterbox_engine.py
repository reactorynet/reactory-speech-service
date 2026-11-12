"""SPEECH-005 — Chatterbox expressive engine adapter.

Every test here runs **without the model and without torch**.  The adapter is
exercised with deterministic doubles; the one test that needs real weights is
marked ``slow`` and skips when the model is absent, so CI stays green with the
flag off and no model mounted.
"""

from __future__ import annotations

import asyncio
import builtins
import inspect
import json
import logging
import os
import subprocess
import sys
import types
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from app.config import settings
from app.core.contracts import AudioChunk, Direction, EngineRequest, Prosody
from app.engines.base import EngineUnavailableError, UnknownVoiceError
from app.engines.chatterbox import (
    CHATTERBOX_ATTRIBUTION,
    CHATTERBOX_LICENSE,
    EXAGGERATION_MAX,
    EXAGGERATION_MIN,
    MAX_CHARS_PER_CALL,
    VARIANT_FILES,
    VARIANT_REPO_BYTES,
    VARIANT_REQUIRED_BYTES,
    VARIANT_REPOS,
    VARIANT_SUPPORTS_EXAGGERATION,
    ChatterboxDependencyError,
    ChatterboxEngine,
    ChatterboxError,
    ChatterboxModelMissing,
    ChatterboxModelStore,
    _chunk_text,
    _import_backend,
    _instantiate_backend,
    download_model,
    exaggeration_for,
    main as chatterbox_main,
    neutralise_watermarker,
    sanitise_text,
)
from app.services import voice_registry
from app.services.tts_service import TTSService
from app.services.voice_registry import SHIPPED_VOICE_IDS

REPO_ROOT = Path(__file__).resolve().parent.parent
CHATTERBOX_STANDARD = "chatterbox_standard"
CHATTERBOX_TURBO = "chatterbox_turbo"


# ---------------------------------------------------------------------------
# Deterministic doubles
# ---------------------------------------------------------------------------


class _SentinelWatermarker:
    """A watermarker double that prepends one full-scale sample.

    A real Perth watermark is imperceptible; a sentinel is not, which is exactly
    what makes "was it neutralised?" assertable without audio analysis.
    """

    def apply_watermark(self, wav, *args, **kwargs):
        return np.concatenate([np.array([-1.0], dtype=wav.dtype), wav])


class _TensorLike:
    """Mimics the small slice of the torch.Tensor API the adapter touches."""

    def __init__(self, array):
        self._array = np.asarray(array, dtype=np.float32)

    def detach(self):
        return self

    def cpu(self):
        return self

    def float(self):
        return self

    def numpy(self):
        return self._array


class FakeChatterbox:
    """ChatterboxTTS double whose duration depends on exaggeration.

    The real model's exaggeration co-varies with pace; modelling that here makes
    "empathetic differs from neutral" a measurable assertion, not a claim.
    """

    sr = 24000

    def __init__(self, *, watermarker=True, exaggerates=True):
        self.watermarker = _SentinelWatermarker() if watermarker else None
        self.exaggerates = exaggerates
        self.calls: list[dict] = []

    def generate(self, text, exaggeration=None, audio_prompt_path=None):
        self.calls.append(
            {"text": text, "exaggeration": exaggeration, "audio_prompt_path": audio_prompt_path}
        )
        value = 0.5 if exaggeration is None else float(exaggeration)
        n = max(1, int(self.sr * (1.0 - 0.3 * value)))
        wav = np.full(n, 0.1, dtype=np.float32)
        if self.watermarker is not None:
            wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)
        return _TensorLike(wav)


class FakeTurbo(FakeChatterbox):
    """ChatterboxTurboTTS double with the *real* turbo signature.

    This is the shape that matters for the review finding: turbo's ``generate``
    accepts ``exaggeration`` (so a signature-name check sees a control) and then
    ignores it.  The double records the argument and renders identically
    regardless, and emits the upstream warning, so a test cannot pass by
    pretending the parameter does not exist.
    """

    def __init__(self):
        super().__init__()
        self.ignored_warnings = 0

    def generate(
        self,
        text,
        repetition_penalty=1.2,
        min_p=0.00,
        top_p=0.95,
        audio_prompt_path=None,
        exaggeration=0.0,
        cfg_weight=0.0,
        temperature=0.8,
        top_k=1000,
        norm_loudness=True,
    ):
        if cfg_weight > 0.0 or exaggeration > 0.0 or min_p > 0.0:
            self.ignored_warnings += 1
        # Same duration whatever exaggeration says: turbo has no emotion
        # embedding (hp.emotion_adv = False).
        return super().generate(text, exaggeration=0.0, audio_prompt_path=audio_prompt_path)


class FakeKokoro:
    """Minimal Kokoro double; only the calls the adapter makes are modelled."""

    def __init__(self):
        self.voices = {voice_id: None for voice_id in SHIPPED_VOICE_IDS}
        self.calls: list[dict] = []

    def create(self, text, voice, speed=1.0, lang="en-us", is_phonemes=False):
        if voice not in self.voices:
            raise ValueError(f"Voice {voice} not found in available voices")
        self.calls.append({"text": text, "voice": voice, "speed": speed, "lang": lang})
        return np.zeros(2400, dtype=np.float32), 24000


def _engine(tmp_path, *, variant="standard", watermarker=True) -> ChatterboxEngine:
    engine = ChatterboxEngine(
        model_store=ChatterboxModelStore(tmp_path, variant=variant), variant=variant
    )
    engine.model = FakeChatterbox(watermarker=watermarker)
    engine.set_ready(True)
    return engine


def _service(
    tmp_path,
    monkeypatch,
    *,
    expressive: bool,
    chatterbox_ready: bool = True,
    variant: str = "standard",
    kokoro_available: bool = True,
    model_dir: str | None = None,
):
    """A real TTSService with doubles, so routing and fallback run for real.

    ``kokoro_available=False`` is the deployment the graceful-degradation
    acceptance criterion describes: the expressive engine is the only one that
    can render, and every request must still succeed.
    ``variant`` selects the deployed Chatterbox variant, because voice
    substitution has to be variant-aware (F1).
    """
    monkeypatch.setattr(settings, "chatterbox_enabled", expressive)
    monkeypatch.setattr(settings, "chatterbox_variant", variant)
    monkeypatch.setattr(
        settings, "chatterbox_model_dir", model_dir or str(tmp_path / "chatterbox-model")
    )
    service = TTSService()
    if kokoro_available:
        service._kokoro = FakeKokoro()
        service._ready = True
    else:
        service._kokoro = None
        service._ready = False
    if service._chatterbox_engine is not None:
        service._chatterbox_engine.model = FakeChatterbox()
        service._chatterbox_engine.set_ready(chatterbox_ready)
    return service


def _write_model(
    root: Path,
    *,
    variant: str = "standard",
    payload: bytes = b"weights",
    license_name: str = "MIT",
    omit: tuple[str, ...] = (),
) -> dict:
    """Materialise a fake model directory with exactly the required files.

    The adapter now only hashes the files ``from_local`` reads, so a test fixture
    that writes a generic ``model.safetensors`` no longer describes a usable
    model — it describes the whole-repository download this ticket removed.
    """
    root.mkdir(parents=True, exist_ok=True)
    for name in VARIANT_FILES[variant]:
        if name in omit:
            continue
        (root / name).write_bytes(payload + name.encode("ascii"))
    store = ChatterboxModelStore(root, variant=variant)
    manifest = store.build_manifest()
    manifest["license"] = license_name
    store.write_manifest(manifest)
    return manifest


# ---------------------------------------------------------------------------
# Capability declaration and compute-cost advisory
# ---------------------------------------------------------------------------


def test_capabilities_declare_what_the_engine_can_actually_honour(tmp_path):
    caps = ChatterboxEngine(model_store=ChatterboxModelStore(tmp_path)).capabilities()

    assert caps.engine == "chatterbox"
    assert caps.languages == ["en-US"]
    assert "empathetic" in caps.styles
    assert "neutral" in caps.styles
    # No breathiness control: advertising whisper would repeat the SPEECH-010
    # mistake of promising a language the front end cannot read.
    assert "whisper" not in caps.styles
    assert caps.native_rate is False
    assert caps.supports_streaming is False
    assert caps.supports_cloning is False  # engine can clone; this service cannot yet
    assert caps.gpu_required is False
    assert caps.sample_rates == [24000]
    assert caps.attribution == CHATTERBOX_ATTRIBUTION
    assert caps.supports_paralinguistic_tags and "[laugh]" in caps.supports_paralinguistic_tags


def test_compute_cost_advisory_reports_the_measured_realtime_factor(tmp_path):
    caps = ChatterboxEngine(model_store=ChatterboxModelStore(tmp_path)).capabilities()
    advisory = caps.compute_cost

    assert advisory is not None
    assert advisory["measured_realtime_factor"] == 0.13
    assert "slower than realtime" in advisory["speed_vs_realtime"]
    # Must match docs/measurements/chatterbox-cpu.md ("resident memory 3,976 MB");
    # an operator-facing advisory that contradicts the page it cites is worse
    # than no advisory.
    assert advisory["resident_memory_mb"] == 3976
    measurement_doc = (REPO_ROOT / "docs" / "measurements" / "chatterbox-cpu.md").read_text(
        encoding="utf-8"
    )
    assert "3,976 MB" in measurement_doc
    assert advisory["load_seconds"] == 21.9
    assert "batch narration" in advisory["suitable_for"]
    assert "interactive turn-taking" in advisory["not_suitable_for"]
    assert "not native_rate" in advisory["routing_signal"]
    assert advisory["source"] == "docs/measurements/chatterbox-cpu.md"


def test_capabilities_endpoint_lists_only_kokoro_when_the_flag_is_off(tmp_path, monkeypatch):
    from app.main import app

    service = _service(tmp_path, monkeypatch, expressive=False)
    app.state.tts_service = service
    body = TestClient(app).get("/api/tts/capabilities").json()

    assert [row["engine"] for row in body["engines"]] == ["kokoro"]
    assert "compute_cost" not in body["engines"][0]


def test_capabilities_endpoint_publishes_the_advisory_when_enabled(tmp_path, monkeypatch):
    from app.main import app

    service = _service(tmp_path, monkeypatch, expressive=True)
    app.state.tts_service = service
    body = TestClient(app).get("/api/tts/capabilities").json()
    rows = {row["engine"]: row for row in body["engines"]}

    assert set(rows) == {"kokoro", "chatterbox"}
    assert rows["chatterbox"]["license"] == "MIT"
    assert rows["chatterbox"]["compute_cost"]["measured_realtime_factor"] == 0.13
    assert rows["chatterbox"]["gpu_required"] is False


# ---------------------------------------------------------------------------
# Direction lowering
# ---------------------------------------------------------------------------


def test_empathetic_is_applied_as_a_bounded_exaggeration(tmp_path):
    engine = ChatterboxEngine(model_store=ChatterboxModelStore(tmp_path))
    plan = engine.lower(Direction(style="empathetic"))

    assert plan.applied["style"] == "empathetic"
    assert EXAGGERATION_MIN <= plan.applied["exaggeration"] <= EXAGGERATION_MAX
    assert plan.ignored == []
    # The engine controls intensity, not valence; the warning says so rather
    # than letting the caller believe "empathetic" is modelled.
    assert any("not valence" in warning for warning in plan.warnings)


@pytest.mark.parametrize(
    "style", ["warm", "calm", "cheerful", "serious", "empathetic", "excited", "narration"]
)
@pytest.mark.parametrize("intensity", [0.0, 0.5, 1.0])
def test_every_style_mapping_stays_inside_the_hard_bounds(style, intensity):
    value = exaggeration_for(style, intensity)

    assert value is not None
    assert EXAGGERATION_MIN <= value <= EXAGGERATION_MAX
    # It co-varies with pace, so a style must not also be a rate multiplier.
    assert 0.3 <= value <= 0.7


def test_whisper_is_reported_as_ignored_not_approximated(tmp_path):
    engine = ChatterboxEngine(model_store=ChatterboxModelStore(tmp_path))
    plan = engine.lower(Direction(style="whisper"))

    assert plan.applied == {}
    assert "style" in plan.ignored
    assert any("whisper" in warning for warning in plan.warnings)


def test_rate_is_never_stacked_on_exaggeration(tmp_path):
    engine = ChatterboxEngine(model_store=ChatterboxModelStore(tmp_path))
    plan = engine.lower(Direction(style="empathetic", prosody=Prosody(rate=1.25)))

    assert plan.applied["style"] == "empathetic"
    assert "rate" not in plan.applied
    assert "rate" in plan.ignored
    assert any("co-varies with pace" in warning for warning in plan.warnings)


def test_unknown_style_is_ignored_with_a_warning(tmp_path):
    engine = ChatterboxEngine(model_store=ChatterboxModelStore(tmp_path))
    plan = engine.lower(Direction(style="sarcastic"))

    assert "style" in plan.ignored
    assert plan.applied == {}
    assert plan.warnings


def test_markup_the_engine_does_not_understand_is_translated_or_stripped():
    text, notes = sanitise_text('Hello <break time="500ms"/> there <emphasis>friend</emphasis>.')

    assert "<" not in text and ">" not in text
    assert "[" not in text
    assert notes


def test_paralinguistic_tags_are_preserved():
    text, _notes = sanitise_text("Well [laugh] that is funny.")

    assert "[laugh]" in text


# ---------------------------------------------------------------------------
# Watermark neutralisation (ADR-0003 decision 4)
# ---------------------------------------------------------------------------


def test_production_output_carries_no_watermark(tmp_path):
    engine = _engine(tmp_path, watermarker=True)
    result = engine.synthesize(
        EngineRequest(text="Hello there.", voice=CHATTERBOX_STANDARD, direction=Direction())
    )
    samples, _rate = sf.read(_wav(result.audio), dtype="float32")

    # The sentinel is -1.0; neutralised output starts at the real audio level.
    assert samples[0] == pytest.approx(0.1, abs=1e-4)
    assert not np.any(samples == pytest.approx(-1.0))
    assert type(engine.model.watermarker).__name__ == "_NoOpWatermarker"


def test_dev_mode_watermark_is_present_when_not_neutralised():
    """Proves the production assertion above is meaningful, not vacuous."""
    fake = FakeChatterbox(watermarker=True)
    raw = fake.generate("Hello there.", exaggeration=0.5)
    samples = np.asarray(raw.numpy(), dtype=np.float32)

    assert samples[0] == pytest.approx(-1.0)


def test_an_unneutralisable_watermarker_refuses_to_render(tmp_path):
    class LockedModel:
        sr = 24000

        @property
        def watermarker(self):
            return _SentinelWatermarker()

        def generate(self, text, exaggeration=None):
            return _TensorLike(np.zeros(2400, dtype=np.float32))

    engine = ChatterboxEngine(model_store=ChatterboxModelStore(tmp_path))
    engine.model = LockedModel()
    engine.set_ready(True)

    with pytest.raises(ChatterboxError, match="watermark"):
        neutralise_watermarker(engine.model)


# ---------------------------------------------------------------------------
# Model management: on-demand download, licence + SHA, never at build time
# ---------------------------------------------------------------------------


def test_missing_model_directory_is_an_actionable_error(tmp_path):
    store = ChatterboxModelStore(tmp_path / "absent")

    with pytest.raises(ChatterboxModelMissing) as excinfo:
        store.ensure_available()

    message = str(excinfo.value)
    assert str(store.root) in message
    assert "download_chatterbox.py" in message
    assert "never at container build time" in message
    assert "CHATTERBOX_ENABLED=false" in message
    # It must be a fallback trigger, never a 500.
    assert isinstance(excinfo.value, EngineUnavailableError)


def test_manifest_records_licence_attribution_and_sha(tmp_path):
    root = tmp_path / "model"
    manifest = _write_model(root)
    store = ChatterboxModelStore(root)

    assert manifest["license"] == "MIT"
    assert manifest["attribution"] == "Chatterbox (Resemble AI)"
    # Only the files from_local reads are recorded, not every weight on disk.
    assert set(manifest["files"]) == set(VARIANT_FILES["standard"])
    assert manifest["allow_patterns"] == list(VARIANT_FILES["standard"])
    assert manifest["download_bytes"] == sum(
        entry["bytes"] for entry in manifest["files"].values()
    )
    assert manifest["download_bytes"] < manifest["repo_total_bytes"]
    assert manifest["missing"] == []
    assert manifest["files"]["t3_cfg.safetensors"]["sha256"] == store.sha256(
        root / "t3_cfg.safetensors"
    )
    assert store.is_verified()
    assert store.ensure_available()["license"] == "MIT"


def test_corrupt_weights_are_rejected_with_the_hash(tmp_path):
    root = tmp_path / "model"
    _write_model(root)
    (root / "t3_cfg.safetensors").write_bytes(b"tampered")

    with pytest.raises(ChatterboxModelMissing, match="SHA-256"):
        ChatterboxModelStore(root).ensure_available()


def test_non_permissive_licence_is_refused(tmp_path):
    root = tmp_path / "model"
    _write_model(root, license_name="GPL-3.0")

    with pytest.raises(ChatterboxModelMissing, match="permitted"):
        ChatterboxModelStore(root).ensure_available()


def test_load_with_a_missing_model_is_unavailable_and_never_imports_torch(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "chatterbox_enabled", True)
    engine = ChatterboxEngine(
        model_store=ChatterboxModelStore(tmp_path / "absent"), variant="standard"
    )

    with pytest.raises(ChatterboxModelMissing):
        engine.load()

    # The adapter module never imports torch at module scope; the load failed
    # before the lazy backend import, so no backend was touched.
    assert engine.model is None
    assert engine.is_ready is False


def test_backend_construction_failure_is_mapped_to_an_actionable_error(tmp_path):
    root = tmp_path / "model"
    _write_model(root)

    class BrokenBackend:
        @staticmethod
        def from_pretrained(**kwargs):
            raise TypeError("'NoneType' object is not callable")

    with pytest.raises(ChatterboxDependencyError) as excinfo:
        _instantiate_backend(BrokenBackend, ChatterboxModelStore(root), "cpu")

    message = str(excinfo.value)
    assert "resemble-perth" in message
    assert "setuptools<81" in message
    assert isinstance(excinfo.value, EngineUnavailableError)


def test_load_neutralises_the_model_watermarker(tmp_path, monkeypatch):
    root = tmp_path / "model"
    _write_model(root)
    engine = ChatterboxEngine(model_store=ChatterboxModelStore(root))
    monkeypatch.setattr(settings, "chatterbox_enabled", True)
    monkeypatch.setattr(
        "app.engines.chatterbox._import_backend",
        lambda variant: type(
            "Backend",
            (),
            {"from_local": staticmethod(lambda path, device: FakeChatterbox())},
        ),
    )

    engine.load()

    assert engine.is_ready
    assert type(engine.model.watermarker).__name__ == "_NoOpWatermarker"


# ---------------------------------------------------------------------------
# Routing, fallback and never a 500
# ---------------------------------------------------------------------------


def test_flag_off_registers_only_kokoro_and_ignores_style(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch, expressive=False)

    assert service.engine_manager.ids == ["kokoro"]
    result = service.synthesize_detailed("Hello.", direction=Direction(style="empathetic"))

    assert result.engine == "kokoro"
    assert "style" in result.ignored
    assert service._chatterbox_engine is None


def test_flag_off_legacy_request_is_byte_identical_to_the_kokoro_engine(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch, expressive=False)
    direct = service._kokoro_engine.synthesize(EngineRequest(text="Hello."))
    legacy = service.synthesize_detailed("Hello.")

    assert legacy.audio == direct.audio
    assert legacy.engine == "kokoro"


def test_flag_on_routes_empathetic_to_the_expressive_engine(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch, expressive=True)
    result = service.synthesize_detailed("Hello there.", direction=Direction(style="empathetic"))

    assert result.engine == "chatterbox"
    assert result.applied.get("style") == "empathetic"
    assert "style" not in result.ignored


def test_empathetic_measurably_differs_from_neutral_on_the_same_voice(tmp_path, monkeypatch):
    """Without weights, prove the difference through the adapter's own controls.

    The real audible comparison is the ``slow`` test below; this one proves the
    request path sends different controls for the same text and voice, so the
    difference is not a coincidence of routing.
    """
    service = _service(tmp_path, monkeypatch, expressive=True)
    voice = CHATTERBOX_STANDARD
    neutral = service.synthesize_detailed("Hello there.", voice=voice, direction=Direction())
    empathetic = service.synthesize_detailed(
        "Hello there.", voice=voice, direction=Direction(style="empathetic")
    )

    assert neutral.engine == empathetic.engine == "chatterbox"
    assert neutral.applied.get("exaggeration") is None
    assert empathetic.applied["exaggeration"] > 0.5
    assert abs(empathetic.duration - neutral.duration) > 0.02


def test_missing_model_falls_back_to_kokoro_with_a_warning(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch, expressive=True, chatterbox_ready=False)
    result = service.synthesize_detailed("Hello.", direction=Direction(style="empathetic"))

    assert result.engine == "kokoro"
    assert "style" in result.ignored
    assert any("fell back" in warning for warning in result.warnings)


@pytest.mark.asyncio
async def test_async_missing_model_falls_back_and_never_raises_a_500(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch, expressive=True, chatterbox_ready=False)
    outcome = await service.synthesize_async("Hello.", direction=Direction(style="empathetic"))

    assert outcome.engine == "kokoro"
    assert any("fell back" in warning for warning in outcome.plan.warnings)
    assert "style" in outcome.plan.ignored


def test_fallback_substitutes_the_voice_and_reports_it(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch, expressive=True, chatterbox_ready=False)
    result = service.synthesize_detailed("Hello.", voice=CHATTERBOX_STANDARD)

    assert result.engine == "kokoro"
    assert "voice" in result.ignored
    assert any("default voice" in warning for warning in result.warnings)
    assert service._kokoro.calls[-1]["voice"] == voice_registry.default_voice_id()


def test_whisper_on_a_chatterbox_voice_succeeds_with_200(tmp_path, monkeypatch):
    from app.main import app

    service = _service(tmp_path, monkeypatch, expressive=True)
    app.state.tts_service = service
    response = TestClient(app).post(
        "/api/tts/synthesize/json",
        json={
            "text": "Speak softly.",
            "voice": CHATTERBOX_STANDARD,
            "direction": {"style": "whisper"},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["engine"] == "chatterbox"
    assert "style" in body["direction_ignored"]
    assert body["direction_warnings"]


def test_streaming_plan_does_not_style_route_to_a_non_streaming_engine(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch, expressive=True)
    plan = service.plan_direction(direction=Direction(style="empathetic"), streaming=True)

    assert plan.engine == "kokoro"
    assert "style" in plan.ignored


def test_voices_endpoint_surfaces_licence_and_attribution(tmp_path, monkeypatch):
    from app.main import app

    service = _service(tmp_path, monkeypatch, expressive=True)
    app.state.tts_service = service
    voices = TestClient(app).get("/api/tts/voices").json()["voices"]
    row = next(voice for voice in voices if voice["id"] == CHATTERBOX_STANDARD)

    assert row["license"] == CHATTERBOX_LICENSE == "MIT"
    assert row["attribution"] == CHATTERBOX_ATTRIBUTION


def test_voices_endpoint_hides_the_optional_engine_when_the_flag_is_off(tmp_path, monkeypatch):
    from app.main import app

    service = _service(tmp_path, monkeypatch, expressive=False)
    app.state.tts_service = service
    voices = TestClient(app).get("/api/tts/voices").json()["voices"]

    assert all(voice["engine"] == "kokoro" for voice in voices)


def test_turbo_accepts_exaggeration_and_ignores_it_so_styles_are_ignored(tmp_path):
    """The exact review finding: a signature check would advertise turbo styles.

    The double has turbo's real signature — ``exaggeration`` is accepted — so if
    the adapter inferred capability from the signature it would report
    ``style: empathetic`` as applied while the engine discards it.
    """
    engine = ChatterboxEngine(model_store=ChatterboxModelStore(tmp_path), variant="turbo")
    engine.model = FakeTurbo()
    engine.set_ready(True)

    assert "exaggeration" in inspect.signature(engine.model.generate).parameters
    assert engine.supports_exaggeration is False
    assert engine.capabilities().styles == ["neutral"]

    plan = engine.lower(Direction(style="empathetic"))
    assert plan.applied == {}
    assert "style" in plan.ignored
    assert "exaggeration" not in plan.applied
    assert "Turbo" in " ".join(plan.warnings)


def test_turbo_reports_the_style_in_the_response_contract(tmp_path):
    engine = ChatterboxEngine(model_store=ChatterboxModelStore(tmp_path), variant="turbo")
    engine.model = FakeTurbo()
    engine.set_ready(True)

    result = engine.synthesize(
        EngineRequest(
            text="Hello there.", voice=CHATTERBOX_TURBO, direction=Direction(style="empathetic")
        )
    )

    assert result.applied == {}
    assert "style" in result.ignored
    assert "intensity" not in result.ignored
    # The adapter never passes an exaggeration to a variant that would discard
    # it, so the engine's own "will be ignored" warning is never triggered.
    assert engine.model.calls[-1]["exaggeration"] == 0.0
    assert engine.model.ignored_warnings == 0


def test_standard_advertises_the_full_style_vocabulary(tmp_path):
    engine = ChatterboxEngine(model_store=ChatterboxModelStore(tmp_path), variant="standard")

    assert VARIANT_SUPPORTS_EXAGGERATION == {"standard": True, "turbo": False}
    assert engine.supports_exaggeration is True
    assert set(engine.capabilities().styles) == {
        "neutral", "warm", "calm", "cheerful", "serious", "empathetic", "excited", "narration"
    }


# ---------------------------------------------------------------------------
# Flag-off / no-torch proof (subprocess, because sys.modules is process-global)
# ---------------------------------------------------------------------------


def _run_python(code: str, **env_extra: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "CHATTERBOX_ENABLED": "false", **env_extra}
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_flag_off_never_imports_the_adapter_or_torch():
    code = (
        "import sys\n"
        "import app.main  # noqa: F401\n"
        "print('ADAPTER', 'app.engines.chatterbox' in sys.modules)\n"
        "print('TORCH', 'torch' in sys.modules)\n"
    )
    completed = _run_python(code)

    assert completed.returncode == 0, completed.stderr
    assert "ADAPTER False" in completed.stdout
    assert "TORCH False" in completed.stdout


def test_startup_works_with_torch_uninstalled_and_the_flag_on():
    code = (
        "import importlib.abc, sys\n"
        "class Block(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in {'torch', 'chatterbox'}:\n"
        "            raise ModuleNotFoundError(name)\n"
        "        return None\n"
        "sys.meta_path.insert(0, Block())\n"
        "from fastapi.testclient import TestClient\n"
        "from app.main import app\n"
        "with TestClient(app) as client:\n"
        "    response = client.get('/health')\n"
        "print('STATUS', response.status_code)\n"
        "print('ENGINES', ','.join(app.state.tts_service.engine_manager.ids))\n"
        "print('TORCH', 'torch' in sys.modules)\n"
    )
    completed = _run_python(code, CHATTERBOX_ENABLED="true")

    assert completed.returncode == 0, completed.stderr
    assert "STATUS 200" in completed.stdout
    assert "ENGINES kokoro,chatterbox" in completed.stdout
    assert "TORCH False" in completed.stdout


def test_flag_on_missing_model_request_falls_back_without_importing_torch():
    code = (
        "import importlib.abc, sys\n"
        "class Block(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in {'torch', 'chatterbox'}:\n"
        "            raise ModuleNotFoundError(name)\n"
        "        return None\n"
        "sys.meta_path.insert(0, Block())\n"
        "import asyncio\n"
        "from app.core.contracts import Direction\n"
        "from app.main import app\n"
        "service = app.state.tts_service\n"
        "service._kokoro = type('K', (), {'voices': {'af_heart': None}, 'create': lambda self, *a, **k: (__import__('numpy').zeros(2400, dtype='float32'), 24000)})()\n"
        "service._ready = True\n"
        "outcome = asyncio.run(service.synthesize_async('Hello.', direction=Direction(style='empathetic')))\n"
        "print('ENGINE', outcome.engine)\n"
        "print('WARN', any('fell back' in w for w in outcome.plan.warnings))\n"
        "print('TORCH', 'torch' in sys.modules)\n"
    )
    completed = _run_python(code, CHATTERBOX_ENABLED="true")

    assert completed.returncode == 0, completed.stderr
    assert "ENGINE kokoro" in completed.stdout
    assert "WARN True" in completed.stdout
    assert "TORCH False" in completed.stdout


# ---------------------------------------------------------------------------
# Deployment: default image unchanged, expressive profile never downloads
# ---------------------------------------------------------------------------


def test_default_dockerfile_does_not_grow_for_the_expressive_engine():
    dockerfile = (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8").lower()

    for forbidden in ("chatterbox", "torch", "tensorflow", "cuda", "nvidia"):
        assert forbidden not in dockerfile


def test_expressive_profile_extends_core_and_never_downloads_at_build_time():
    dockerfile = (REPO_ROOT / "Dockerfile.expressive").read_text(encoding="utf-8")
    # Comments may mention the operator command; build instructions may not run it.
    instructions = "\n".join(
        line for line in dockerfile.splitlines() if not line.strip().startswith("#")
    )

    assert "FROM reactory-speech-service:speech-core" in instructions
    assert "requirements-expressive.txt" in instructions
    assert "download_chatterbox" not in instructions
    assert "scripts/download_models.py" not in instructions


def test_expressive_requirements_pin_the_documented_dependency_traps():
    expressive = (REPO_ROOT / "requirements-expressive.txt").read_text(encoding="utf-8")
    default = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")

    assert "chatterbox-tts==0.1.7" in expressive
    assert "setuptools<81" in expressive
    assert "coverage>=7.6" in expressive
    # chatterbox-tts 0.1.7 requires torch==2.6.0 / torchaudio==2.6.0 exactly.
    # Asserting the pinned versions (not merely that "torch" appears) is what
    # would have caught the rejected torch==2.5.1+cpu pin: that string satisfied
    # a substring check while making the image uninstallable.
    assert "torch==2.6.0+cpu" in expressive
    assert "torchaudio==2.6.0+cpu" in expressive
    assert "download.pytorch.org/whl/cpu" in expressive
    assert "chatterbox" not in default.lower()
    assert "torch" not in default.lower()


def test_the_expressive_pins_satisfy_chatterboxs_own_requirement():
    """The pin must be a version chatterbox-tts actually accepts.

    ``torch==2.6.0`` (no local version) is what chatterbox declares; the CPU
    wheel ``2.6.0+cpu`` satisfies it because PEP 440 ignores a local version
    segment when the requirement names none.  Assert the base version matches.
    """
    from packaging.requirements import Requirement
    from packaging.version import Version

    expressive = (REPO_ROOT / "requirements-expressive.txt").read_text(encoding="utf-8")
    pins = {
        line.split("==")[0]: line.split("==")[1]
        for line in expressive.splitlines()
        if "==" in line and not line.lstrip().startswith("#")
    }

    for package in ("torch", "torchaudio"):
        assert package in pins, f"{package} must be pinned explicitly"
        requirement = Requirement(f"{package}==2.6.0")  # chatterbox-tts 0.1.7's pin
        assert Version(pins[package]) in requirement.specifier, (
            f"{package}=={pins[package]} does not satisfy chatterbox-tts 0.1.7's "
            f"{requirement}"
        )


def test_registry_entries_carry_mit_licence_and_attribution():
    for voice_id, variant in ((CHATTERBOX_STANDARD, "standard"), (CHATTERBOX_TURBO, "turbo")):
        record = voice_registry.get_record(voice_id)

        assert record is not None, voice_id
        assert record.engine == "chatterbox"
        assert record.license == "MIT"
        assert record.attribution == "Chatterbox (Resemble AI)"
        assert record.style_spec["variant"] == variant
        # Only the standard variant renders tone; turbo ignores exaggeration.
        assert record.supports["style"] is (variant == "standard")
        assert record.raw["gpu_required"] is False
        assert record.license == "MIT"
        assert record.attribution == "Chatterbox (Resemble AI)"


# ---------------------------------------------------------------------------
# Slow evidence: real weights only. Skipped in CI and when the model is absent.
# ---------------------------------------------------------------------------


def _real_model_available() -> bool:
    root = Path(settings.chatterbox_model_dir)
    store = ChatterboxModelStore(root, variant=settings.chatterbox_variant)
    if not store.is_present():
        return False
    try:
        import chatterbox  # noqa: F401
        import torch  # noqa: F401
    except ImportError:
        return False
    return True


@pytest.mark.slow
@pytest.mark.skipif(
    not _real_model_available(),
    reason="Chatterbox weights and torch are optional and absent in this environment",
)
def test_empathetic_audibly_differs_from_neutral_on_cpu(tmp_path, monkeypatch):
    """Acceptance evidence with real weights: same text, same voice, audible delta.

    Skipped unless the model is mounted and the expressive extras are installed,
    so this file never downloads 3.1 GB in CI.
    """
    monkeypatch.setattr(settings, "chatterbox_enabled", True)
    engine = ChatterboxEngine(model_store=ChatterboxModelStore(settings.chatterbox_model_dir))
    engine.load()  # raises if the manifest does not verify

    text = "We are sorry for the disruption to your service today."
    voice = CHATTERBOX_STANDARD
    neutral = engine.synthesize(
        EngineRequest(text=text, voice=voice, direction=Direction())
    )
    empathetic = engine.synthesize(
        EngineRequest(text=text, voice=voice, direction=Direction(style="empathetic"))
    )

    neutral_samples, rate = sf.read(_wav(neutral.audio), dtype="float32")
    empathetic_samples, _ = sf.read(_wav(empathetic.audio), dtype="float32")
    assert rate == 24000
    assert np.abs(neutral_samples).max() > 0.01

    # Generation is stochastic, so a single pair of renders can differ even at an
    # identical exaggeration.  Averaging several renders per condition keeps the
    # noise floor below the effect we are trying to prove.
    repeats = 3
    neutral_again = [
        engine.synthesize(EngineRequest(text=text, voice=voice, direction=Direction()))
        for _ in range(repeats - 1)
    ]
    empathetic_again = [
        engine.synthesize(
            EngineRequest(text=text, voice=voice, direction=Direction(style="empathetic"))
        )
        for _ in range(repeats - 1)
    ]

    def _mean_stats(results: list) -> tuple[float, float]:
        durations, rms_values = [], []
        for result in results:
            samples, _sr = sf.read(_wav(result.audio), dtype="float32")
            durations.append(len(samples) / rate)
            rms_values.append(float(np.sqrt(np.mean(samples**2))))
        return float(np.mean(durations)), float(np.mean(rms_values))

    neutral_duration, neutral_rms = _mean_stats(
        [neutral, *neutral_again]
    )
    empathetic_duration, empathetic_rms = _mean_stats([empathetic, *empathetic_again])

    duration_delta = abs(neutral_duration - empathetic_duration)
    rms_delta = abs(neutral_rms - empathetic_rms)

    # Thresholds derived from the measured exaggeration slope, not guessed:
    # 0.5 -> 1.0 is -17.6% duration / +18.6% RMS on the committed renders, so
    # 0.5 -> 0.62 should be about -4.2% / +4.5%.  Assert a fifth of that, which
    # is well above sampling noise once averaged over three renders but still
    # tolerant of a shorter sentence.
    assert duration_delta >= 0.01 or rms_delta >= 0.001, (
        f"neutral and empathetic renders are not measurably different after "
        f"{repeats} renders each (mean duration delta {duration_delta:.3f}s, "
        f"mean RMS delta {rms_delta:.4f})"
    )


def _wav(audio: bytes):
    import io

    return io.BytesIO(audio)


def test_registry_still_partitions_the_kokoro_pack():
    """The optional engine must not leak into Kokoro catalogue math."""
    assert len(SHIPPED_VOICE_IDS) == 54
    assert len(voice_registry.QUALITY_GRADES) == 47
    assert len(voice_registry.UNGRADED_VOICE_IDS) == 7
    assert len(voice_registry.ALL_REGISTERED_VOICE_IDS) == 56


def test_manifest_json_is_well_formed(tmp_path):
    root = tmp_path / "model"
    _write_model(root)
    payload = json.loads((root / "manifest.json").read_text(encoding="utf-8"))

    assert payload["engine"] == "chatterbox"
    assert payload["license"] == "MIT"
    assert payload["files"]
    assert payload["repo_total_bytes"] == 13_866_212_931
    assert payload["required_total_bytes"] == 3_191_966_992


# ---------------------------------------------------------------------------
# F1 — variant-aware fallback voice substitution
#
# A turbo deployment must degrade to *its own* default voice.  Reading the
# registry directly always returned `chatterbox_standard`, which the adapter
# refuses as a variant mismatch — a 422 on the fallback path.
# ---------------------------------------------------------------------------


def test_default_voice_for_is_variant_aware(tmp_path, monkeypatch):
    for variant, expected in (("standard", CHATTERBOX_STANDARD), ("turbo", CHATTERBOX_TURBO)):
        service = _service(tmp_path, monkeypatch, expressive=True, variant=variant)
        engine = service._chatterbox_engine

        assert service._default_voice_for(engine) == expected
        # And the substitute really is renderable by that engine.
        assert engine.resolve_voice(expected)[0] == voice_registry.get_record(
            expected
        ).engine_voice_id


@pytest.mark.parametrize("variant", ["standard", "turbo"])
def test_kokoro_unavailable_falls_back_to_the_deployed_variant(tmp_path, monkeypatch, variant):
    """The documented graceful-degradation path: no Kokoro, Kokoro voice pinned."""
    from app.main import app

    service = _service(
        tmp_path, monkeypatch, expressive=True, variant=variant, kokoro_available=False
    )
    app.state.tts_service = service

    response = TestClient(app).post(
        "/api/tts/synthesize/json",
        json={"text": "Hello there.", "voice": "bf_alice"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["engine"] == "chatterbox"
    assert "voice" in body["direction_ignored"]
    warnings = " | ".join(body["direction_warnings"])
    assert "variant" not in warnings
    assert f"chatterbox_{variant}" in warnings or "default voice" in warnings


@pytest.mark.parametrize("variant", ["standard", "turbo"])
def test_explicit_engine_override_with_a_foreign_voice_substitutes(tmp_path, monkeypatch, variant):
    from app.main import app

    service = _service(tmp_path, monkeypatch, expressive=True, variant=variant)
    app.state.tts_service = service

    request = EngineRequest(text="Hello there.", voice="bf_alice")
    prepared, notes, substituted = service._prepare_request(
        service._chatterbox_engine, request
    )

    assert substituted is True
    assert prepared.voice == f"chatterbox_{variant}"
    assert notes and "default voice" in notes[0]

    response = TestClient(app).post(
        "/api/tts/synthesize/json",
        json={"text": "Hello there.", "engine": "chatterbox", "voice": "bf_alice"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["engine"] == "chatterbox"
    assert "voice" in response.json()["direction_ignored"]


# ---------------------------------------------------------------------------
# F2 — the fallback warning must carry the actionable cause
# ---------------------------------------------------------------------------


def test_missing_model_warning_carries_path_and_download_command(tmp_path, monkeypatch):
    from app.main import app

    model_dir = tmp_path / "absent-chatterbox-model"
    service = _service(tmp_path, monkeypatch, expressive=True, model_dir=str(model_dir))
    # Drop the double so the real load path runs and fails on the absent model.
    service._chatterbox_engine.model = None
    service._chatterbox_engine.set_ready(False)
    app.state.tts_service = service

    response = TestClient(app).post(
        "/api/tts/synthesize/json",
        json={"text": "Hello.", "direction": {"style": "empathetic"}},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["engine"] == "kokoro"
    assert "style" in body["direction_ignored"]
    warnings = " | ".join(body["direction_warnings"])
    assert str(model_dir) in warnings
    assert "download_chatterbox.py" in warnings
    assert "CHATTERBOX_ENABLED=false" in warnings


def test_expected_model_absence_logs_without_a_traceback(tmp_path, monkeypatch, caplog):
    model_dir = tmp_path / "absent-chatterbox-model"
    service = _service(tmp_path, monkeypatch, expressive=True, model_dir=str(model_dir))
    service._chatterbox_engine.model = None
    service._chatterbox_engine.set_ready(False)

    with caplog.at_level(logging.WARNING):
        outcome = asyncio.run(
            service.synthesize_async("Hello.", direction=Direction(style="empathetic"))
        )

    assert outcome.engine == "kokoro"
    # One WARNING with the cause, and no traceback for a documented state.
    manager_records = [r for r in caplog.records if r.name == "app.engines.manager"]
    assert manager_records, "the failed load must still be reported"
    assert all(record.exc_info is None for record in manager_records)
    assert all(record.levelno == logging.WARNING for record in manager_records)


def test_manager_exposes_the_last_load_error(tmp_path, monkeypatch):
    service = _service(
        tmp_path,
        monkeypatch,
        expressive=True,
        model_dir=str(tmp_path / "absent-chatterbox-model"),
    )
    service._chatterbox_engine.model = None
    service._chatterbox_engine.set_ready(False)
    manager = service.engine_manager

    with pytest.raises(EngineUnavailableError) as excinfo:
        asyncio.run(manager.ensure_loaded("chatterbox"))

    cause = manager.last_load_error("chatterbox")
    assert isinstance(cause, ChatterboxModelMissing)
    assert "download_chatterbox.py" in str(cause)
    assert str(cause) in str(excinfo.value)
    assert excinfo.value.__cause__ is cause


# ---------------------------------------------------------------------------
# F4 — intensity is reported, never silently dropped
# ---------------------------------------------------------------------------


def test_intensity_without_a_style_is_reported_ignored(tmp_path):
    engine = ChatterboxEngine(model_store=ChatterboxModelStore(tmp_path))

    no_style = engine.lower(Direction(intensity=0.9))
    assert no_style.applied == {}
    assert "intensity" in no_style.ignored
    assert any("intensity" in warning for warning in no_style.warnings)

    neutral = engine.lower(Direction(style="neutral", intensity=0.9))
    assert neutral.applied == {}
    assert "intensity" in neutral.ignored
    assert any("intensity" in warning for warning in neutral.warnings)

    # An inexpressible style already reports intensity; it must not be doubled.
    whisper = engine.lower(Direction(style="whisper", intensity=0.9))
    assert whisper.ignored.count("intensity") == 1


def test_intensity_with_an_applied_style_is_applied_not_ignored(tmp_path):
    engine = ChatterboxEngine(model_store=ChatterboxModelStore(tmp_path))
    plan = engine.lower(Direction(style="empathetic", intensity=0.8))

    assert plan.applied["intensity"] == 0.8
    assert "intensity" not in plan.ignored


# ---------------------------------------------------------------------------
# F3 — on-demand download, dependency errors, chunking, streaming, guards
# ---------------------------------------------------------------------------


def _fake_hub(tmp_path, calls: list, *, payload: bytes = b"weights"):
    """Install a fake ``huggingface_hub`` that records and materialises the files.

    It honours ``allow_patterns`` exactly as the real ``snapshot_download`` does,
    so a test fails if the adapter forgets to scope the download.
    """
    module = types.ModuleType("huggingface_hub")

    def snapshot_download(
        repo_id, revision=None, local_dir=None, token=None, allow_patterns=None
    ):
        calls.append(
            {
                "repo_id": repo_id,
                "revision": revision,
                "local_dir": local_dir,
                "allow_patterns": allow_patterns,
            }
        )
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        for name in allow_patterns or ["t3_cfg.safetensors"]:
            Path(local_dir, name).write_bytes(payload)
        return local_dir

    module.snapshot_download = snapshot_download
    return module


def test_download_model_writes_mit_manifest_with_sha_and_nothing_at_import(tmp_path, monkeypatch):
    calls: list[dict] = []
    monkeypatch.setitem(sys.modules, "huggingface_hub", _fake_hub(tmp_path, calls))

    # Importing the adapter (already done) must never download anything.
    assert calls == []

    root = tmp_path / "chatterbox-model"
    manifest = download_model(root, variant="standard", revision="abc123")

    assert len(calls) == 1
    assert calls[0]["repo_id"] == VARIANT_REPOS["standard"]
    assert calls[0]["revision"] == "abc123"
    assert calls[0]["local_dir"] == str(root)

    on_disk = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    store = ChatterboxModelStore(root)
    assert manifest["license"] == CHATTERBOX_LICENSE == "MIT"
    assert manifest["attribution"] == CHATTERBOX_ATTRIBUTION
    assert on_disk["license"] == "MIT"
    assert on_disk["attribution"] == "Chatterbox (Resemble AI)"
    assert on_disk["revision"] == "abc123"
    assert calls[0]["allow_patterns"] == list(VARIANT_FILES["standard"])
    assert (
        on_disk["files"]["t3_cfg.safetensors"]["sha256"]
        == store.sha256(root / "t3_cfg.safetensors")
    )
    assert store.is_verified()


def test_operator_cli_downloads_and_reports_the_licence(tmp_path, monkeypatch, capsys):
    calls: list[dict] = []
    monkeypatch.setitem(sys.modules, "huggingface_hub", _fake_hub(tmp_path, calls))

    assert chatterbox_main(["--download", "--models-dir", str(tmp_path / "cb")]) == 0
    output = capsys.readouterr().out
    assert "MIT" in output
    assert "Chatterbox (Resemble AI)" in output
    assert (tmp_path / "cb" / "manifest.json").is_file()


def test_operator_cli_without_download_only_prints_help(capsys):
    assert chatterbox_main([]) == 0
    assert "download" in capsys.readouterr().out


def test_import_backend_reports_the_missing_optional_profile(monkeypatch):
    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.split(".")[0] in {"chatterbox", "torch"}:
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)

    with pytest.raises(ChatterboxDependencyError) as excinfo:
        _import_backend("standard")

    message = str(excinfo.value)
    assert "requirements-expressive.txt" in message
    assert "torch" in message
    assert "speech-expressive" in message


def test_long_narration_is_chunked_without_losing_words():
    text = ("The quick brown fox jumps over the lazy dog. " * 20).strip()

    assert len(text) > MAX_CHARS_PER_CALL
    chunks = _chunk_text(text)

    assert len(chunks) > 1
    assert all(len(chunk) <= MAX_CHARS_PER_CALL for chunk in chunks)
    assert " ".join(chunks).split() == text.split()


def test_a_single_overlong_sentence_is_hard_split():
    text = "A" * 1500

    chunks = _chunk_text(text)

    assert len(chunks) == 3
    assert all(len(chunk) <= MAX_CHARS_PER_CALL for chunk in chunks)
    assert "".join(chunks) == text


def test_chunk_text_of_empty_input_is_empty():
    assert _chunk_text("   ") == []


def test_stream_and_synthesize_span_yield_one_complete_chunk(tmp_path):
    engine = _engine(tmp_path)
    request = EngineRequest(text="Hello there. How are you?", voice=CHATTERBOX_STANDARD)

    span = engine.synthesize_span(request, "Hello there.")
    assert isinstance(span, AudioChunk)
    assert span.audio.startswith(b"RIFF")
    assert span.sample_rate == 24000
    assert span.duration > 0

    chunks = list(engine.stream(request))
    assert len(chunks) == 1
    assert isinstance(chunks[0], AudioChunk)
    assert chunks[0].audio.startswith(b"RIFF")


def test_split_sentences_matches_the_streaming_contract():
    assert ChatterboxEngine.split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]


def test_phoneme_input_is_refused_rather_than_misrendered(tmp_path):
    engine = _engine(tmp_path)

    with pytest.raises(ChatterboxError, match="phonemes"):
        engine.synthesize(
            EngineRequest(text="h@l'oU", voice=CHATTERBOX_STANDARD, is_phonemes=True)
        )


def test_empty_audio_from_the_backend_is_an_error(tmp_path):
    class Mute(FakeChatterbox):
        def generate(self, text, exaggeration=None, audio_prompt_path=None):
            return _TensorLike(np.zeros(0, dtype=np.float32))

    engine = ChatterboxEngine(model_store=ChatterboxModelStore(tmp_path))
    engine.model = Mute()
    engine.set_ready(True)

    with pytest.raises(ChatterboxError, match="empty audio"):
        engine.synthesize(EngineRequest(text="Hello.", voice=CHATTERBOX_STANDARD))


def test_text_that_is_only_markup_is_refused(tmp_path):
    engine = _engine(tmp_path)

    with pytest.raises(ChatterboxError, match="No speakable text"):
        engine.synthesize(
            EngineRequest(text="<emphasis></emphasis>", voice=CHATTERBOX_STANDARD)
        )


def test_an_unloaded_engine_is_unavailable_not_a_crash(tmp_path):
    engine = ChatterboxEngine(model_store=ChatterboxModelStore(tmp_path))

    with pytest.raises(ChatterboxError, match="not loaded"):
        engine.synthesize(EngineRequest(text="Hello.", voice=CHATTERBOX_STANDARD))
    assert isinstance(
        ChatterboxError("x"), EngineUnavailableError
    ), "the adapter error must stay a fallback trigger"


def test_load_refuses_when_the_flag_is_off(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "chatterbox_enabled", False)
    engine = ChatterboxEngine(model_store=ChatterboxModelStore(tmp_path))

    with pytest.raises(ChatterboxError, match="CHATTERBOX_ENABLED=false"):
        engine.load()


def test_turbo_still_renders_neutral_audio(tmp_path):
    engine = ChatterboxEngine(model_store=ChatterboxModelStore(tmp_path), variant="turbo")
    engine.model = FakeTurbo()
    engine.set_ready(True)

    result = engine.synthesize(EngineRequest(text="Hello.", voice=CHATTERBOX_TURBO))

    assert result.applied == {}
    assert result.audio.startswith(b"RIFF")


def test_resolve_voice_rejects_foreign_and_wrong_variant_ids(tmp_path):
    engine = ChatterboxEngine(model_store=ChatterboxModelStore(tmp_path), variant="turbo")

    with pytest.raises(UnknownVoiceError, match="variant"):
        engine.resolve_voice(CHATTERBOX_STANDARD)
    with pytest.raises(UnknownVoiceError, match="not a Chatterbox voice"):
        engine.resolve_voice("af_heart")

    assert engine.resolve_voice(CHATTERBOX_TURBO) == ("default_turbo", "en-us")
    assert engine.resolve_voice(None) == ("default_turbo", "en-us")


def test_reference_voice_is_refused_until_cloning_ships(tmp_path, monkeypatch):
    engine = ChatterboxEngine(model_store=ChatterboxModelStore(tmp_path))
    reference = replace(
        voice_registry.get_record(CHATTERBOX_STANDARD),
        style_spec={"type": "reference", "variant": "standard"},
    )
    monkeypatch.setattr(engine, "_records", lambda: [reference])
    monkeypatch.setattr(voice_registry, "get_record", lambda _voice_id: reference)

    with pytest.raises(UnknownVoiceError, match="SPEECH-006"):
        engine.resolve_voice(CHATTERBOX_STANDARD)


def test_no_registered_voice_for_the_variant_is_reported(tmp_path, monkeypatch):
    engine = ChatterboxEngine(model_store=ChatterboxModelStore(tmp_path))
    monkeypatch.setattr(engine, "_records", lambda: [])

    with pytest.raises(UnknownVoiceError, match="No Chatterbox voice is registered"):
        engine.resolve_voice(None)


def test_voice_catalogue_is_scoped_to_the_deployed_variant(tmp_path):
    engine = ChatterboxEngine(model_store=ChatterboxModelStore(tmp_path), variant="turbo")

    assert engine.available_voice_ids() == [CHATTERBOX_TURBO]
    assert engine.engine_voice_ids is None  # unloaded
    assert engine.get_voice(CHATTERBOX_STANDARD) is None

    engine.model = FakeChatterbox()
    engine.set_ready(True)

    assert engine.engine_voice_ids == [CHATTERBOX_TURBO]
    assert [voice["id"] for voice in engine.voices()] == [CHATTERBOX_TURBO]
    assert engine.get_voice(CHATTERBOX_TURBO)["license"] == "MIT"


def test_descriptor_reports_variant_licence_and_exaggeration_range(tmp_path):
    engine = _engine(tmp_path, variant="turbo")
    descriptor = engine.descriptor()

    assert descriptor["license"] == "MIT"
    assert descriptor["attribution"] == CHATTERBOX_ATTRIBUTION
    assert descriptor["variant"] == "turbo"
    assert descriptor["default_voice"] == CHATTERBOX_TURBO
    assert descriptor["exaggeration_range"] == [EXAGGERATION_MIN, EXAGGERATION_MAX]
    assert descriptor["watermark"] == "not-loaded"


@pytest.mark.asyncio
async def test_initialize_loads_off_the_loop_and_unload_releases(tmp_path, monkeypatch):
    root = tmp_path / "model"
    _write_model(root)
    monkeypatch.setattr(settings, "chatterbox_enabled", True)

    class Backend:
        @staticmethod
        def from_local(path, device):
            return FakeChatterbox()

    monkeypatch.setattr("app.engines.chatterbox._import_backend", lambda variant: Backend)
    engine = ChatterboxEngine(model_store=ChatterboxModelStore(root))

    await engine.initialize()
    assert engine.is_ready
    assert engine.sample_rate == 24000
    assert type(engine.model.watermarker).__name__ == "_NoOpWatermarker"

    # Idempotent: a second initialize is a no-op.
    await engine.initialize()
    assert engine.is_ready

    engine.unload()
    assert engine.is_ready is False
    assert engine.model is None
    assert engine.supports_exaggeration is True  # reset to "unknown"

    # Re-initialising after an explicit load-error reset must not raise.
    assert "download_chatterbox.py" in str(
        ChatterboxModelStore(tmp_path / "absent")._missing_message("absent")
    )


def test_unreadable_manifest_is_treated_as_absent(tmp_path):
    root = tmp_path / "model"
    _write_model(root)
    (root / "manifest.json").write_text("{not json", encoding="utf-8")
    store = ChatterboxModelStore(root)

    assert store.read_manifest() is None
    assert store.is_verified() is False
    with pytest.raises(ChatterboxModelMissing, match="manifest.json"):
        store.ensure_available()


def test_store_reports_absence_without_raising(tmp_path):
    store = ChatterboxModelStore(tmp_path / "absent")

    assert store.weight_files() == []
    assert store.is_present() is False
    assert store.read_manifest() is None


def test_weight_files_are_scoped_to_the_files_from_local_reads(tmp_path):
    """A whole-repository download must not be hashed end to end."""
    root = tmp_path / "model"
    _write_model(root)
    # Files a naive whole-repository download leaves behind, none of which
    # from_local opens.
    for extra in ("t3_cfg.pt", "s3gen.pt", "s3gen_v3.safetensors", "t3_23lang.safetensors"):
        (root / extra).write_bytes(b"x" * 1024)

    store = ChatterboxModelStore(root)

    assert [path.name for path in store.weight_files()] == sorted(VARIANT_FILES["standard"])
    # The extra files are NOT hashed: the manifest still covers only the 5.
    assert set(store.build_manifest()["files"]) == set(VARIANT_FILES["standard"])


def test_a_partial_download_is_rejected_with_the_missing_names(tmp_path):
    root = tmp_path / "model"
    _write_model(root, omit=("s3gen.safetensors",))

    with pytest.raises(ChatterboxModelMissing) as excinfo:
        ChatterboxModelStore(root).ensure_available()

    message = str(excinfo.value)
    assert "s3gen.safetensors" in message
    assert "download_chatterbox.py" in message


def test_verification_is_memoised_and_invalidated_by_a_rewrite(tmp_path, monkeypatch):
    root = tmp_path / "model"
    _write_model(root)
    store = ChatterboxModelStore(root)
    calls = {"n": 0}
    real = ChatterboxModelStore.sha256

    def counting(path):
        calls["n"] += 1
        return real(path)

    monkeypatch.setattr(ChatterboxModelStore, "sha256", staticmethod(counting))

    assert store.verify() is True
    first = calls["n"]
    assert first == len(VARIANT_FILES["standard"])

    # Unchanged files: no further hashing, so a reload does not pay 3.19 GB again.
    assert store.verify() is True
    assert calls["n"] == first

    # A rewrite changes size+mtime and forces re-verification.
    target = root / "s3gen.safetensors"
    target.write_bytes(b"changed")
    assert store.verify() is False
    assert calls["n"] > first


def test_a_persistent_model_absence_logs_one_warning_not_one_per_request(
    tmp_path, monkeypatch, caplog
):
    service = _service(
        tmp_path,
        monkeypatch,
        expressive=True,
        model_dir=str(tmp_path / "absent-chatterbox-model"),
    )
    service._chatterbox_engine.model = None
    service._chatterbox_engine.set_ready(False)

    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            asyncio.run(
                service.synthesize_async("Hello.", direction=Direction(style="empathetic"))
            )

    warnings = [
        record
        for record in caplog.records
        if record.name == "app.engines.manager" and record.levelno == logging.WARNING
    ]
    assert len(warnings) == 1, [record.getMessage() for record in warnings]
    assert "download_chatterbox.py" in warnings[0].getMessage()
    assert warnings[0].exc_info is None


# ---------------------------------------------------------------------------
# B3 — the download fetches only what from_local reads
# ---------------------------------------------------------------------------


def test_download_requests_only_the_files_from_local_reads(tmp_path, monkeypatch):
    recorded: list[dict] = []

    module = types.ModuleType("huggingface_hub")

    def snapshot_download(repo_id, revision=None, local_dir=None, token=None,
                          allow_patterns=None):
        recorded.append({"repo_id": repo_id, "allow_patterns": allow_patterns})
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        for name in allow_patterns or []:
            Path(local_dir, name).write_bytes(b"payload-" + name.encode("ascii"))
        return local_dir

    module.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)

    root = tmp_path / "cb"
    manifest = download_model(root, variant="standard")

    assert recorded[0]["allow_patterns"] == list(VARIANT_FILES["standard"])
    # 5 files, not the 16 in the repository.
    assert len(recorded[0]["allow_patterns"]) == 5
    assert set(manifest["files"]) == set(VARIANT_FILES["standard"])
    assert manifest["download_bytes"] < manifest["repo_total_bytes"]
    assert manifest["missing"] == []


def test_download_of_a_partial_pattern_set_fails_loudly(tmp_path, monkeypatch):
    module = types.ModuleType("huggingface_hub")

    def snapshot_download(repo_id, revision=None, local_dir=None, token=None,
                          allow_patterns=None):
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        # Simulates an interrupted / filtered download.
        Path(local_dir, "ve.safetensors").write_bytes(b"only one")
        return local_dir

    module.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)

    with pytest.raises(ChatterboxModelMissing, match="download finished"):
        download_model(tmp_path / "cb", variant="standard")


def test_turbo_download_requests_the_turbo_file_set(tmp_path, monkeypatch):
    recorded: list[dict] = []
    module = types.ModuleType("huggingface_hub")

    def snapshot_download(repo_id, revision=None, local_dir=None, token=None,
                          allow_patterns=None):
        recorded.append({"repo_id": repo_id, "allow_patterns": allow_patterns})
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        for name in allow_patterns or []:
            Path(local_dir, name).write_bytes(b"x")
        return local_dir

    module.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)

    manifest = download_model(tmp_path / "cbt", variant="turbo")

    assert "t3_turbo_v1.safetensors" in recorded[0]["allow_patterns"]
    assert "s3gen_meanflow.safetensors" in recorded[0]["allow_patterns"]
    # Turbo does not read the standard t3/s3gen checkpoints.
    assert "t3_cfg.safetensors" not in recorded[0]["allow_patterns"]
    assert manifest["variant"] == "turbo"
    assert manifest["repo_total_bytes"] == 4_044_184_736
    # The phantom tokenizer.json must never be requested again.
    assert "tokenizer.json" not in recorded[0]["allow_patterns"]


# ---------------------------------------------------------------------------
# N1 — internal phonetic markup and request-level controls
# ---------------------------------------------------------------------------


def test_inline_phonetic_markup_becomes_the_plain_word():
    text, notes = sanitise_text("Say [Kokoro|/koʊˈkɔːroʊ/] now.")

    assert text == "Say Kokoro now."
    assert any("phonetic markup" in note for note in notes)


def test_phoneme_tags_become_their_grapheme_text():
    text, notes = sanitise_text('Say <phoneme ph="koʊkɔroʊ">Kokoro</phoneme> now.')

    assert text == "Say Kokoro now."
    assert any("<phoneme>" in note for note in notes)


def test_paralinguistic_tags_survive_but_phonetic_markup_does_not():
    text, _notes = sanitise_text("Well [laugh] [Kokoro|/k/] there.")

    assert "[laugh]" in text
    assert "|" not in text
    assert "Kokoro" in text


def test_custom_lexicon_is_reported_ignored_not_dropped(tmp_path):
    engine = _engine(tmp_path)
    result = engine.synthesize(
        EngineRequest(
            text="Hello.",
            voice=CHATTERBOX_STANDARD,
            custom_lexicon={"Kokoro": "/koʊkɔroʊ/"},
        )
    )

    assert "custom_lexicon" in result.ignored
    assert any("custom_lexicon" in warning for warning in result.warnings)


def test_phonetic_preprocess_true_is_reported_ignored_not_dropped(tmp_path):
    engine = _engine(tmp_path)
    result = engine.synthesize(
        EngineRequest(text="Hello.", voice=CHATTERBOX_STANDARD, phonetic_preprocess=True)
    )

    assert "phonetic_preprocess" in result.ignored
    assert any("phonetic_preprocess" in warning for warning in result.warnings)


def test_phonetic_preprocess_false_is_not_reported_as_ignored(tmp_path):
    """Explicitly disabling preprocessing asks for the default; it is not ignored.

    Reporting it would tell a caller a control was dropped when there was
    nothing to drop.
    """
    engine = _engine(tmp_path)
    result = engine.synthesize(
        EngineRequest(text="Hello.", voice=CHATTERBOX_STANDARD, phonetic_preprocess=False)
    )

    assert "phonetic_preprocess" not in result.ignored
    assert not any("phonetic_preprocess" in w for w in result.warnings)


def test_markup_notes_reach_the_response_warnings(tmp_path):
    engine = _engine(tmp_path)
    result = engine.synthesize(
        EngineRequest(text="Say [Kokoro|/k/] now.", voice=CHATTERBOX_STANDARD)
    )

    assert engine.model.calls[-1]["text"] == "Say Kokoro now."
    assert any("phonetic markup" in warning for warning in result.warnings)


# ---------------------------------------------------------------------------
# B2 — capabilities table is the source of truth before load
# ---------------------------------------------------------------------------


def test_capabilities_are_correct_before_the_model_is_loaded(tmp_path):
    standard = ChatterboxEngine(model_store=ChatterboxModelStore(tmp_path), variant="standard")
    turbo = ChatterboxEngine(model_store=ChatterboxModelStore(tmp_path), variant="turbo")

    assert standard.is_ready is False and turbo.is_ready is False
    assert len(standard.capabilities().styles) == 8
    assert turbo.capabilities().styles == ["neutral"]
    assert turbo.descriptor()["supports_exaggeration"] is False
    assert standard.descriptor()["supports_exaggeration"] is True


def test_turbo_capability_note_explains_the_variant():
    engine = ChatterboxEngine(model_store=ChatterboxModelStore("/tmp/never"), variant="turbo")
    notes = engine.capabilities().capability_notes or []

    assert any("Turbo" in note and "ignor" in note for note in notes)


def test_descriptor_reports_the_download_scope(tmp_path):
    engine = ChatterboxEngine(model_store=ChatterboxModelStore(tmp_path))
    store = engine.model_store

    assert store.required_files == VARIANT_FILES["standard"]
    assert engine.variant == "standard"


def test_capabilities_endpoint_exposes_supports_exaggeration(tmp_path, monkeypatch):
    """The API must let a caller see that turbo cannot render tone."""
    from app.main import app

    for variant, expected in (("standard", True), ("turbo", False)):
        service = _service(tmp_path, monkeypatch, expressive=True, variant=variant)
        app.state.tts_service = service
        rows = {
            row["engine"]: row
            for row in TestClient(app).get("/api/tts/capabilities").json()["engines"]
        }
        assert rows["chatterbox"]["supports_exaggeration"] is expected
        if variant == "turbo":
            assert rows["chatterbox"]["styles"] == ["neutral"]
        else:
            assert "empathetic" in rows["chatterbox"]["styles"]


def test_turbo_style_request_via_the_api_is_ignored_not_applied(tmp_path, monkeypatch):
    from app.main import app

    service = _service(tmp_path, monkeypatch, expressive=True, variant="turbo")
    app.state.tts_service = service
    body = TestClient(app).post(
        "/api/tts/synthesize/json",
        json={
            "text": "We are sorry for the disruption.",
            "voice": CHATTERBOX_TURBO,
            "direction": {"style": "empathetic"},
        },
    ).json()

    assert body["engine"] == "chatterbox"
    assert body["direction_applied"] == {}
    assert "style" in body["direction_ignored"]
    assert any("Turbo" in warning for warning in body["direction_warnings"])


# ---------------------------------------------------------------------------
# Q1 — the required-file set must match the real upstream repository
#
# The B3 regression was invisible because the fake hub materialised whatever
# allow_pattern it was handed, so the fake repository was richer than the real
# one. These tests read a checked-in capture of the actual repository layout.
# ---------------------------------------------------------------------------

UPSTREAM_FIXTURE = REPO_ROOT / "tests" / "data" / "chatterbox-upstream.json"


def _upstream() -> dict:
    return json.loads(UPSTREAM_FIXTURE.read_text(encoding="utf-8"))


def _real_hub(monkeypatch, variant: str) -> list[dict]:
    """Install a fake hub that serves only files that exist upstream.

    Unlike the general-purpose fake, this one cannot be talked into creating a
    file the real repository does not contain, which is what makes it able to
    catch a phantom required file.
    """
    upstream = _upstream()["repos"][variant]["files"]
    calls: list[dict] = []
    module = types.ModuleType("huggingface_hub")

    def snapshot_download(
        repo_id, revision=None, local_dir=None, token=None, allow_patterns=None
    ):
        calls.append({"repo_id": repo_id, "allow_patterns": allow_patterns})
        Path(local_dir).mkdir(parents=True, exist_ok=True)
        for name in allow_patterns or []:
            if name in upstream:
                Path(local_dir, name).write_bytes(b"payload-" + name.encode("ascii"))
            # A name absent upstream simply is not produced, exactly as a real
            # download would behave.
        return local_dir

    module.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    return calls


@pytest.mark.parametrize("variant", ["standard", "turbo"])
def test_required_files_all_exist_in_the_real_upstream_repository(variant):
    """Every file the adapter demands must be in the repository it downloads from."""
    upstream = _upstream()["repos"][variant]

    missing_upstream = [name for name in VARIANT_FILES[variant] if name not in upstream["files"]]

    assert missing_upstream == [], (
        f"VARIANT_FILES['{variant}'] requires files that do not exist in "
        f"{upstream['repo_id']}: {missing_upstream}. A download can never produce them, "
        "so the variant would be permanently unusable."
    )


def test_turbo_does_not_require_a_tokenizer_json():
    """Turbo's tokenizer is a GPT2Tokenizer (vocab.json + merges.txt).

    ``tokenizer.json`` exists only in the standard repository, where
    ``EnTokenizer`` reads it. Requiring it here broke CHATTERBOX_VARIANT=turbo.
    """
    upstream = _upstream()["repos"]["turbo"]["files"]

    assert "tokenizer.json" not in upstream, "fixture is stale: turbo now has tokenizer.json"
    assert "tokenizer.json" not in VARIANT_FILES["turbo"]
    for required in (
        "vocab.json",
        "merges.txt",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
    ):
        assert required in VARIANT_FILES["turbo"], required


def test_upstream_fixture_matches_the_recorded_byte_totals():
    """The fixture is internally consistent with the published constants."""
    for variant, repo in _upstream()["repos"].items():
        assert sum(repo["files"].values()) == repo["total_bytes"], variant
        assert repo["total_bytes"] == VARIANT_REPO_BYTES[variant], variant
        required = sum(repo["files"][name] for name in VARIANT_FILES[variant])
        assert required == VARIANT_REQUIRED_BYTES[variant], (
            f"{variant}: VARIANT_FILES sums to {required:,} B but "
            f"VARIANT_REQUIRED_BYTES says {VARIANT_REQUIRED_BYTES[variant]:,} B"
        )


def test_standard_required_files_are_a_strict_subset_of_the_repository():
    """Scoping is still worth something: the required set is much smaller."""
    repo = _upstream()["repos"]["standard"]

    assert set(VARIANT_FILES["standard"]) < set(repo["files"])
    assert VARIANT_REQUIRED_BYTES["standard"] < repo["total_bytes"] / 4


@pytest.mark.parametrize("variant", ["standard", "turbo"])
def test_download_and_load_succeed_against_the_real_file_set(tmp_path, monkeypatch, variant):
    """End-to-end proof, per variant, that download -> verify -> load works.

    This is the test the previous revision lacked: with the phantom
    ``tokenizer.json`` present, the turbo case failed right here.
    """
    calls = _real_hub(monkeypatch, variant)
    root = tmp_path / f"cb-{variant}"

    manifest = download_model(root, variant=variant)

    assert calls[0]["repo_id"] == VARIANT_REPOS[variant]
    assert manifest["missing"] == []
    assert set(manifest["files"]) == set(VARIANT_FILES[variant])
    # ``download_bytes`` is what was actually written (fake payloads here);
    # ``required_total_bytes`` is the upstream truth from the constants.
    assert manifest["download_bytes"] == sum(e["bytes"] for e in manifest["files"].values())
    assert manifest["required_total_bytes"] == VARIANT_REQUIRED_BYTES[variant]

    # The store accepts exactly what the download produced.
    store = ChatterboxModelStore(root, variant=variant)
    assert store.missing_files() == []
    assert store.is_verified()
    assert store.ensure_available()["license"] == "MIT"

    # And the backend can be constructed from it.
    class Backend:
        @staticmethod
        def from_local(path, device):
            return FakeChatterbox()

    monkeypatch.setattr(settings, "chatterbox_enabled", True)
    monkeypatch.setattr("app.engines.chatterbox._import_backend", lambda _variant: Backend)
    engine = ChatterboxEngine(model_store=store, variant=variant)
    engine.load()

    assert engine.is_ready
    assert type(engine.model.watermarker).__name__ == "_NoOpWatermarker"


def test_a_whole_repository_download_still_satisfies_the_turbo_requirement(tmp_path, monkeypatch):
    """Operators who mirrored the full repo must not be rejected."""
    upstream = _upstream()["repos"]["turbo"]["files"]
    root = tmp_path / "full-turbo"
    root.mkdir()
    for name in upstream:
        (root / name).write_bytes(b"x")

    store = ChatterboxModelStore(root, variant="turbo")
    assert store.missing_files() == []
    # Only the required files are hashed, not all 13.
    manifest = store.build_manifest()
    # Only the 9 required files are hashed, not all 13 present on disk.
    assert set(manifest["files"]) == set(VARIANT_FILES["turbo"])
    assert manifest["required_total_bytes"] == VARIANT_REQUIRED_BYTES["turbo"]
    assert len(store.weight_files()) == len(VARIANT_FILES["turbo"]) < len(upstream)
    store.write_manifest(manifest)
    assert store.ensure_available()["missing"] == []
