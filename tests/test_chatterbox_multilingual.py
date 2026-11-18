"""Chatterbox Multilingual: a second Chatterbox engine beside the English one.

The model itself (torch + 3.2 GB of weights) is not available to the default
test suite, so these tests use doubles with the *real* upstream signature:
``ChatterboxMultilingualTTS.generate(text, language_id, ...)`` takes the
language as a required argument, and the English ``generate`` does not accept
it at all.  That makes "the right language reached the right model" an
assertion rather than an assumption.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from app.config import settings
from app.core.contracts import Direction, EngineRequest
from app.engines.base import UnknownVoiceError
from app.engines.chatterbox import (
    CHATTERBOX_ENGINE,
    CHATTERBOX_MULTILINGUAL_ENGINE,
    MULTILINGUAL_LANGUAGE_CAVEATS,
    MULTILINGUAL_LANGUAGES,
    MULTILINGUAL_VARIANT,
    STYLE_EXAGGERATION,
    VARIANT_FILES,
    ChatterboxEngine,
    ChatterboxError,
    ChatterboxModelStore,
    _offline_text_front_end,
)
from app.services import voice_registry
from app.services.tts_service import TTSService, VoiceLanguageUnavailableError
from app.services.voice_registry import SHIPPED_VOICE_IDS

TOKENIZER_MODULE = "chatterbox.models.tokenizers.tokenizer"


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------


class _Watermarker:
    def apply_watermark(self, wav, *args, **kwargs):
        return wav


class FakeMultilingual:
    """``ChatterboxMultilingualTTS`` double: ``language_id`` is required."""

    sr = 24000

    def __init__(self):
        self.watermarker = _Watermarker()
        self.calls: list[dict] = []

    def generate(self, text, language_id, audio_prompt_path=None, exaggeration=0.5):
        self.calls.append({"text": text, "language_id": language_id, "exaggeration": exaggeration})
        return np.full(2400, 0.1, dtype=np.float32)


class FakeEnglish:
    """``ChatterboxTTS`` double: passing ``language_id`` is a ``TypeError``."""

    sr = 24000

    def __init__(self):
        self.watermarker = _Watermarker()
        self.calls: list[dict] = []

    def generate(self, text, exaggeration=0.5, audio_prompt_path=None):
        self.calls.append({"text": text, "exaggeration": exaggeration})
        return np.full(2400, 0.1, dtype=np.float32)


class FakeKokoro:
    def __init__(self):
        self.voices = {voice_id: None for voice_id in SHIPPED_VOICE_IDS}
        self.calls: list[dict] = []

    def create(self, text, voice, speed=1.0, lang="en-us", is_phonemes=False):
        self.calls.append({"text": text, "voice": voice})
        return np.zeros(2400, dtype=np.float32), 24000


def _multilingual_engine(tmp_path) -> ChatterboxEngine:
    engine = ChatterboxEngine(
        model_store=ChatterboxModelStore(tmp_path, variant=MULTILINGUAL_VARIANT),
        variant=MULTILINGUAL_VARIANT,
    )
    engine.model = FakeMultilingual()
    engine.set_ready(True)
    return engine


def _service(tmp_path, monkeypatch, *, english: bool, multilingual: bool) -> TTSService:
    monkeypatch.setattr(settings, "chatterbox_enabled", english)
    monkeypatch.setattr(settings, "chatterbox_multilingual_enabled", multilingual)
    monkeypatch.setattr(settings, "chatterbox_variant", "standard")
    monkeypatch.setattr(settings, "chatterbox_model_dir", str(tmp_path / "chatterbox"))
    monkeypatch.setattr(
        settings, "chatterbox_multilingual_model_dir", str(tmp_path / "chatterbox-multilingual")
    )
    service = TTSService()
    service._kokoro = FakeKokoro()
    service._ready = True
    if service._chatterbox_engine is not None:
        service._chatterbox_engine.model = FakeEnglish()
        service._chatterbox_engine.set_ready(True)
    if service._chatterbox_multilingual_engine is not None:
        service._chatterbox_multilingual_engine.model = FakeMultilingual()
        service._chatterbox_multilingual_engine.set_ready(True)
    return service


def _multilingual_records() -> list:
    return voice_registry.VOICE_REGISTRY.for_engine(CHATTERBOX_MULTILINGUAL_ENGINE)


def _write_model(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name in VARIANT_FILES[MULTILINGUAL_VARIANT]:
        (root / name).write_bytes(b"payload-" + name.encode("ascii"))
    store = ChatterboxModelStore(root, variant=MULTILINGUAL_VARIANT)
    store.write_manifest(store.build_manifest())


# ---------------------------------------------------------------------------
# Identity and configuration
# ---------------------------------------------------------------------------


def test_multilingual_is_its_own_engine_with_its_own_model_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "chatterbox_model_dir", str(tmp_path / "english"))
    monkeypatch.setattr(settings, "chatterbox_multilingual_model_dir", str(tmp_path / "mtl"))

    english = ChatterboxEngine(variant="standard")
    multilingual = ChatterboxEngine(variant=MULTILINGUAL_VARIANT)

    assert english.id == CHATTERBOX_ENGINE
    assert multilingual.id == CHATTERBOX_MULTILINGUAL_ENGINE
    assert multilingual.is_multilingual and not english.is_multilingual
    # Separate directories: the two variants come from one repository and a
    # shared directory would be refused as a variant mismatch.
    assert english.model_store.root == tmp_path / "english"
    assert multilingual.model_store.root == tmp_path / "mtl"
    assert multilingual.descriptor()["name"] == "Chatterbox Multilingual (expressive, MIT)"


def test_both_flags_register_both_engines_english_first(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch, english=True, multilingual=True)

    assert service.engine_manager.ids == ["kokoro", "chatterbox", "chatterbox_multilingual"]


def test_the_multilingual_flag_alone_registers_only_that_engine(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch, english=False, multilingual=True)

    assert service.engine_manager.ids == ["kokoro", "chatterbox_multilingual"]
    assert service._chatterbox_engine is None


def test_chatterbox_variant_multilingual_is_refused_not_registered_twice(tmp_path, monkeypatch):
    # Accepting it registered two engines under one id; the English one vanished.
    monkeypatch.setattr(settings, "chatterbox_variant", "multilingual")
    monkeypatch.setattr(settings, "chatterbox_model_dir", str(tmp_path / "english"))

    with pytest.raises(ValueError, match="CHATTERBOX_MULTILINGUAL_ENABLED"):
        ChatterboxEngine()
    with pytest.raises(ValueError, match="CHATTERBOX_MULTILINGUAL_ENABLED"):
        TTSService(expressive=True, multilingual=True)


def test_the_multilingual_engine_is_off_by_default():
    assert type(settings).model_fields["chatterbox_multilingual_enabled"].default is False


# ---------------------------------------------------------------------------
# Languages
# ---------------------------------------------------------------------------


def test_capabilities_declare_23_languages_and_their_caveats(tmp_path):
    caps = _multilingual_engine(tmp_path).capabilities()

    assert caps.engine == CHATTERBOX_MULTILINGUAL_ENGINE
    assert len(caps.languages) == 23
    assert set(caps.languages) == set(MULTILINGUAL_LANGUAGES)
    assert {"de", "ko", "ru", "nl", "pl"} <= set(caps.languages)
    # Every language Kokoro covers is covered here too.
    assert caps.unsupported_languages == []
    rows = {row["language"]: row for row in caps.language_support}
    for code, row in rows.items():
        assert row["supported"] is True
        assert row["measured"] is False
        if code in MULTILINGUAL_LANGUAGE_CAVEATS:
            assert row["quality"] == "degraded"
            assert row["caveat"] == MULTILINGUAL_LANGUAGE_CAVEATS[code]
        else:
            assert row["quality"] == "unverified"
            assert row["caveat"] is None
    assert set(MULTILINGUAL_LANGUAGE_CAVEATS) == {"he", "ru", "zh"}
    # Same exaggeration control as the English model.
    assert "empathetic" in caps.styles
    # The English tag vocabulary is not claimed for an unverified tokenizer.
    assert caps.supports_paralinguistic_tags == []


@pytest.mark.parametrize(
    ("locale", "expected"),
    [("de", "de"), ("pt-br", "pt"), ("zh", "zh"), ("en-us", "en"), ("fr_FR", "fr")],
)
def test_a_voice_locale_becomes_the_model_language_id(tmp_path, locale, expected):
    assert _multilingual_engine(tmp_path).language_id_for(locale) == expected


def test_an_unsupported_locale_is_refused_not_rendered_in_another_language(tmp_path):
    with pytest.raises(ChatterboxError, match="cannot render locale 'cy'"):
        _multilingual_engine(tmp_path).language_id_for("cy")


def test_the_english_engine_still_refuses_other_languages(tmp_path):
    english = ChatterboxEngine(model_store=ChatterboxModelStore(tmp_path), variant="standard")

    assert english.language_id_for("en-us") is None
    with pytest.raises(ChatterboxError, match="chatterbox_multilingual"):
        english.language_id_for("fr-fr")


# ---------------------------------------------------------------------------
# Voices
# ---------------------------------------------------------------------------


def test_the_registry_has_one_voice_per_language():
    records = _multilingual_records()

    assert sorted(record.language for record in records) == sorted(MULTILINGUAL_LANGUAGES)
    for record in records:
        assert record.id == f"chatterbox_multilingual_{record.language}"
        assert record.locale == record.language
        assert record.style_spec["variant"] == MULTILINGUAL_VARIANT
        assert record.license == "MIT"
        assert record.enabled
    # The synthesis cache keys on the engine voice id, not the locale, so two
    # languages sharing one would serve a German render to a French request.
    keys = [record.engine_voice_id for record in records]
    assert len(set(keys)) == len(keys)


def test_every_language_voice_reaches_the_model_with_its_language(tmp_path):
    engine = _multilingual_engine(tmp_path)

    for record in _multilingual_records():
        result = engine.synthesize(EngineRequest(text="Hallo.", voice=record.id))

        assert result.engine == CHATTERBOX_MULTILINGUAL_ENGINE
        assert engine.model.calls[-1]["language_id"] == record.language


def test_a_style_is_rendered_as_exaggeration_in_any_language(tmp_path):
    engine = _multilingual_engine(tmp_path)

    result = engine.synthesize(
        EngineRequest(
            text="Guten Morgen.",
            voice="chatterbox_multilingual_de",
            direction=Direction(style="empathetic"),
        )
    )

    assert result.applied["style"] == "empathetic"
    assert engine.model.calls[-1] == {
        "text": "Guten Morgen.",
        "language_id": "de",
        "exaggeration": STYLE_EXAGGERATION["empathetic"],
    }


def test_the_default_voice_is_listed_from_the_multilingual_records(tmp_path):
    engine = _multilingual_engine(tmp_path)

    assert set(engine.available_voice_ids()) == {r.id for r in _multilingual_records()}
    with pytest.raises(UnknownVoiceError):
        engine.resolve_voice("chatterbox_standard")


def test_the_english_engine_does_not_claim_the_multilingual_voices(tmp_path):
    english = ChatterboxEngine(model_store=ChatterboxModelStore(tmp_path), variant="standard")

    assert not english.can_render_voice("chatterbox_multilingual_de")
    assert "chatterbox_multilingual_de" not in english.available_voice_ids()


# ---------------------------------------------------------------------------
# Routing through the service
# ---------------------------------------------------------------------------


def test_a_pinned_voice_routes_to_the_model_that_owns_it(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch, english=True, multilingual=True)

    german = service.synthesize_detailed("Guten Morgen.", voice="chatterbox_multilingual_de")
    english = service.synthesize_detailed("Good morning.", voice="chatterbox_standard")

    assert german.engine == CHATTERBOX_MULTILINGUAL_ENGINE
    assert english.engine == CHATTERBOX_ENGINE
    assert service._chatterbox_multilingual_engine.model.calls[-1]["language_id"] == "de"
    # FakeEnglish would raise TypeError had language_id been passed to it.
    assert service._chatterbox_engine.model.calls[-1]["text"] == "Good morning."


def test_a_style_with_no_voice_still_goes_to_the_english_engine(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch, english=True, multilingual=True)

    result = service.synthesize_detailed("Hello.", direction=Direction(style="empathetic"))

    assert result.engine == CHATTERBOX_ENGINE


def test_voices_endpoint_lists_the_multilingual_voices(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from app.main import app

    service = _service(tmp_path, monkeypatch, english=True, multilingual=True)
    app.state.tts_service = service
    response = TestClient(app).get("/api/tts/voices")

    assert response.status_code == 200, response.text
    payload = response.json()
    voices = payload["voices"] if isinstance(payload, dict) else payload
    listed = {voice["id"] for voice in voices}
    assert "chatterbox_multilingual_de" in listed
    assert "chatterbox_standard" in listed


def test_voices_endpoint_omits_the_multilingual_voices_when_the_flag_is_off(
    tmp_path, monkeypatch
):
    from fastapi.testclient import TestClient

    from app.main import app

    app.state.tts_service = _service(tmp_path, monkeypatch, english=True, multilingual=False)
    payload = TestClient(app).get("/api/tts/voices").json()

    voices = payload["voices"] if isinstance(payload, dict) else payload
    listed = {voice["id"] for voice in voices}
    assert "chatterbox_standard" in listed
    assert not {v for v in listed if v.startswith("chatterbox_multilingual_")}


def test_streaming_a_multilingual_voice_passes_its_language(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch, english=False, multilingual=True)

    chunks = list(service.synthesize_streaming("Bonjour.", voice="chatterbox_multilingual_fr"))

    assert chunks
    assert service._chatterbox_multilingual_engine.model.calls[-1]["language_id"] == "fr"


# ---------------------------------------------------------------------------
# The default voice is English, never the first language alphabetically
# ---------------------------------------------------------------------------


def test_the_engine_default_voice_is_english(tmp_path):
    engine = _multilingual_engine(tmp_path)

    voice_key, locale = engine.resolve_voice(None)

    assert engine.language_id_for(locale) == "en"
    assert engine.available_voice_ids()[0] == "chatterbox_multilingual_en"
    assert engine.descriptor()["default_voice"] == "chatterbox_multilingual_en"


def test_naming_the_engine_without_a_voice_renders_english(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch, english=False, multilingual=True)

    result = service.synthesize_detailed("Hello there.", engine=CHATTERBOX_MULTILINGUAL_ENGINE)

    assert result.engine == CHATTERBOX_MULTILINGUAL_ENGINE
    assert service._chatterbox_multilingual_engine.model.calls[-1]["language_id"] == "en"


# ---------------------------------------------------------------------------
# A substituted voice keeps the language
# ---------------------------------------------------------------------------


def test_a_multilingual_voice_without_its_engine_is_refused_not_answered_in_english(
    tmp_path, monkeypatch
):
    service = _service(tmp_path, monkeypatch, english=True, multilingual=False)

    with pytest.raises(VoiceLanguageUnavailableError, match="chatterbox_multilingual"):
        service.synthesize_detailed("Guten Morgen.", voice="chatterbox_multilingual_de")
    assert service._kokoro.calls == []


def test_the_api_answers_a_refused_language_with_422(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from app.main import app

    app.state.tts_service = _service(tmp_path, monkeypatch, english=True, multilingual=False)
    response = TestClient(app).post(
        "/api/tts/synthesize",
        json={"text": "Guten Morgen.", "voice": "chatterbox_multilingual_de", "cache": False},
    )

    assert response.status_code == 422, response.text
    assert "does not answer in another language" in response.text


def test_an_english_multilingual_voice_still_falls_back_with_a_warning(tmp_path, monkeypatch):
    # Same language, different timbre: the pre-existing reported substitution.
    service = _service(tmp_path, monkeypatch, english=False, multilingual=False)

    result = service.synthesize_detailed("Hello.", voice="chatterbox_multilingual_en")

    assert result.engine == "kokoro"
    assert "voice" in result.ignored
    assert any("chatterbox_multilingual_en" in w for w in result.warnings)


def test_a_substitute_is_chosen_in_the_pinned_voices_language(tmp_path, monkeypatch):
    # A French Kokoro voice sent to the multilingual engine is rendered by its
    # French voice, not by its (English) default.
    service = _service(tmp_path, monkeypatch, english=False, multilingual=True)
    french = next(
        r.id for r in voice_registry.VOICE_REGISTRY.for_engine("kokoro") if r.language.startswith("fr")
    )

    result = service.synthesize_detailed(
        "Bonjour.", voice=french, engine=CHATTERBOX_MULTILINGUAL_ENGINE
    )

    assert result.engine == CHATTERBOX_MULTILINGUAL_ENGINE
    assert service._chatterbox_multilingual_engine.model.calls[-1]["language_id"] == "fr"
    assert any("chatterbox_multilingual_fr" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# Cloning stays on the English model
# ---------------------------------------------------------------------------


def test_clones_are_never_created_on_the_multilingual_engine(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch, english=False, multilingual=True)

    assert service._clone_engine() is None
    assert service._clone_backend() is None


def test_clones_use_the_english_engine_when_both_run(tmp_path, monkeypatch):
    service = _service(tmp_path, monkeypatch, english=True, multilingual=True)

    assert service._clone_engine() is service._chatterbox_engine


# ---------------------------------------------------------------------------
# Loading never reaches the network
# ---------------------------------------------------------------------------


def _install_fake_front_end(monkeypatch) -> tuple[types.ModuleType, types.ModuleType]:
    """Stand-ins for the two upstream modules the loader reaches the network through."""

    def network_download(*args, **kwargs):
        raise AssertionError("hf_hub_download reached the network")

    def network_segmenter(*args, **kwargs):
        raise AssertionError("pkuseg downloaded its model")

    tokenizer = types.ModuleType(TOKENIZER_MODULE)
    tokenizer.hf_hub_download = network_download
    segmenter = types.ModuleType("spacy_pkuseg")
    segmenter.pkuseg = network_segmenter
    for name in ("chatterbox", "chatterbox.models", "chatterbox.models.tokenizers"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, TOKENIZER_MODULE, tokenizer)
    monkeypatch.setitem(sys.modules, "spacy_pkuseg", segmenter)
    return tokenizer, segmenter


def test_the_cangjie_table_is_served_from_the_model_directory(tmp_path, monkeypatch):
    tokenizer, segmenter = _install_fake_front_end(monkeypatch)
    original_download, original_segmenter = tokenizer.hf_hub_download, segmenter.pkuseg
    (tmp_path / "Cangjie5_TC.json").write_text("[]", encoding="utf-8")

    with _offline_text_front_end(tmp_path):
        served = tokenizer.hf_hub_download(
            repo_id="ResembleAI/chatterbox", filename="Cangjie5_TC.json", cache_dir=tmp_path
        )
        assert served == str(tmp_path / "Cangjie5_TC.json")
        with pytest.raises(FileNotFoundError, match="never downloads"):
            tokenizer.hf_hub_download(repo_id="ResembleAI/chatterbox", filename="other.json")
        # ImportError is the one failure the upstream tokenizer handles itself.
        with pytest.raises(ImportError):
            segmenter.pkuseg()

    assert tokenizer.hf_hub_download is original_download
    assert segmenter.pkuseg is original_segmenter


def test_loading_the_multilingual_model_stays_offline(tmp_path, monkeypatch):
    tokenizer, _segmenter = _install_fake_front_end(monkeypatch)
    root = tmp_path / "mtl"
    _write_model(root)
    seen: list[str] = []

    class Backend:
        @staticmethod
        def from_local(path, device):
            # What MTLTokenizer does during construction.
            seen.append(tokenizer.hf_hub_download(repo_id="x", filename="Cangjie5_TC.json"))
            return FakeMultilingual()

    monkeypatch.setattr("app.engines.chatterbox._import_backend", lambda _variant: Backend)
    engine = ChatterboxEngine(
        model_store=ChatterboxModelStore(root, variant=MULTILINGUAL_VARIANT),
        variant=MULTILINGUAL_VARIANT,
    )
    engine.load()

    assert engine.is_ready
    assert seen == [str(root / "Cangjie5_TC.json")]
    assert type(engine.model.watermarker).__name__ == "_NoOpWatermarker"


def test_a_failed_load_still_restores_the_upstream_functions(tmp_path, monkeypatch):
    tokenizer, segmenter = _install_fake_front_end(monkeypatch)
    original_download, original_segmenter = tokenizer.hf_hub_download, segmenter.pkuseg
    root = tmp_path / "mtl"
    _write_model(root)

    class Backend:
        @staticmethod
        def from_local(path, device):
            raise RuntimeError("corrupt checkpoint")

    monkeypatch.setattr("app.engines.chatterbox._import_backend", lambda _variant: Backend)
    engine = ChatterboxEngine(
        model_store=ChatterboxModelStore(root, variant=MULTILINGUAL_VARIANT),
        variant=MULTILINGUAL_VARIANT,
    )
    with pytest.raises(RuntimeError, match="corrupt checkpoint"):
        engine.load()

    assert tokenizer.hf_hub_download is original_download
    assert segmenter.pkuseg is original_segmenter


def test_a_missing_upstream_target_is_logged_not_silently_skipped(tmp_path, monkeypatch, caplog):
    tokenizer, _segmenter = _install_fake_front_end(monkeypatch)
    del tokenizer.hf_hub_download  # an upstream rename
    # ``None`` in sys.modules makes the import raise ImportError.
    monkeypatch.setitem(sys.modules, "spacy_pkuseg", None)

    with caplog.at_level("WARNING", logger="app.engines.chatterbox"):
        with _offline_text_front_end(tmp_path):
            pass

    messages = [record.getMessage() for record in caplog.records]
    assert any(f"{TOKENIZER_MODULE}.hf_hub_download does not exist" in m for m in messages)
    assert any("spacy_pkuseg is not importable" in m for m in messages)


def test_the_upstream_language_list_matches_ours():
    """Runs where chatterbox-tts is installed (the speech-expressive image)."""
    mtl_tts = pytest.importorskip("chatterbox.mtl_tts")

    assert set(mtl_tts.SUPPORTED_LANGUAGES) == set(MULTILINGUAL_LANGUAGES)
