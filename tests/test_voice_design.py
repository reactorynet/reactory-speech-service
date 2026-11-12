"""SPEECH-003 Phase 2 — designed voices by blending Kokoro style tensors.

Everything here runs on CPU against a deterministic engine double: the real
Kokoro model is not required to prove the recipe rules, the tensor resolver,
persistence, preview caching or the HTTP contract.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.engines.kokoro import (
    KokoroEngine,
    blend_cache_bytes,
    blend_cache_size,
    clear_blend_cache,
)
from app.main import app as fastapi_app
from app.services import designed_voices, voice_registry
from app.services.designed_voices import (
    PREVIEW_TEXT,
    VoiceRecipeError,
    normalised_components,
    preset_voice_ids,
    recipe_hash,
    validate_components,
)
from app.services.tts_service import TTSService
from app.services.voice_design import VoiceDesignService
from app.services.voice_registry import KOKORO_SAMPLE_RATE, SHIPPED_VOICE_IDS

APP_ROOT = Path(__file__).resolve().parent.parent
DESIGNED_DOC = APP_ROOT / "docs" / "voices" / "designed-voices.md"

MODEL_PATH = Path(settings.kokoro_model_resolved)
VOICES_PATH = Path(settings.kokoro_voices_resolved)
requires_models = pytest.mark.skipif(
    not (MODEL_PATH.exists() and VOICES_PATH.exists()),
    reason="Kokoro model files are not present in models/",
)

#: Small synthetic style shape: only *equality* of component shapes matters for
#: the resolver.  One test uses the real (510, 1, 256) shape explicitly.
STYLE_SHAPE = (8, 1, 4)


class FakeKokoro:
    """Deterministic stand-in for ``kokoro_onnx.Kokoro``.

    Exposes ``get_voice_style`` (the real API) and accepts either a voice name or
    a style ``ndarray`` in ``create``, exactly like the real engine.
    """

    def __init__(self, voice_ids=SHIPPED_VOICE_IDS, shape: tuple[int, ...] = STYLE_SHAPE):
        self.voices = {voice_id: None for voice_id in voice_ids}
        self.shape = shape
        self.calls: list[dict] = []
        self._styles: dict[str, np.ndarray] = {}

    def get_voice_style(self, name: str) -> np.ndarray:
        if name not in self.voices:
            raise KeyError(name)
        if name not in self._styles:
            index = float(len(self._styles) + 1)
            base = np.arange(np.prod(self.shape), dtype=np.float32).reshape(self.shape)
            self._styles[name] = base + index * 100.0
        return self._styles[name]

    def create(self, text, voice, speed=1.0, lang="en-us", is_phonemes=False):
        if not isinstance(voice, np.ndarray) and voice not in self.voices:
            raise ValueError(f"Voice {voice} not found in available voices")
        self.calls.append({"text": text, "voice": voice, "speed": speed, "lang": lang})
        return np.zeros(2400, dtype=np.float32), KOKORO_SAMPLE_RATE


def _service(voice_ids=SHIPPED_VOICE_IDS, shape: tuple[int, ...] = STYLE_SHAPE) -> TTSService:
    service = TTSService()
    service._kokoro = FakeKokoro(voice_ids, shape=shape)
    service._ready = True
    return service


def _client(service: TTSService) -> TestClient:
    fastapi_app.state.tts_service = service
    fastapi_app.state.stt_service = MagicMock(is_ready=False)
    return TestClient(fastapi_app)


@pytest.fixture
def design_service() -> VoiceDesignService:
    return VoiceDesignService(_service())


@pytest.fixture
def service() -> TTSService:
    return _service()


@pytest.fixture
def api(service) -> TestClient:
    client = _client(service)
    yield client
    fastapi_app.state.tts_service = TTSService()


def _recipe(*pairs: tuple[str, float]) -> list[dict]:
    return [{"voice": voice, "weight": weight} for voice, weight in pairs]


# ─────────────────────────────────────────────────────────────
# Recipe rules → 422 with an actionable message
# ─────────────────────────────────────────────────────────────

def test_a_single_component_recipe_is_rejected(api):
    response = api.post(
        "/api/tts/voices/design",
        json={"name": "Solo", "components": _recipe(("af_heart", 1.0))},
    )

    assert response.status_code == 422
    assert "at least 2" in response.json()["detail"]


def test_five_components_are_rejected(api):
    response = api.post(
        "/api/tts/voices/design",
        json={
            "name": "Too Many",
            "components": _recipe(
                ("af_heart", 0.6),
                ("af_bella", 0.1),
                ("af_nicole", 0.1),
                ("af_sarah", 0.1),
                ("af_sky", 0.1),
            ),
        },
    )

    assert response.status_code == 422
    assert "at most 4" in response.json()["detail"]


def test_a_zero_sum_recipe_is_rejected(api):
    response = api.post(
        "/api/tts/voices/design",
        json={"name": "Zero", "components": _recipe(("af_heart", 0.0), ("af_bella", 0.0))},
    )

    assert response.status_code == 422
    assert "greater than zero" in response.json()["detail"]


def test_negative_weights_are_rejected(api):
    response = api.post(
        "/api/tts/voices/design",
        json={"name": "Negative", "components": _recipe(("af_heart", 1.2), ("af_bella", -0.2))},
    )

    assert response.status_code == 422
    assert "non-negative" in response.json()["detail"]


def test_non_finite_weights_are_rejected(design_service):
    with pytest.raises(VoiceRecipeError, match="finite"):
        design_service.validate_blend(_recipe(("af_heart", float("nan")), ("af_bella", 0.2)))


def test_flat_mix_without_a_dominant_component_is_rejected(api):
    response = api.post(
        "/api/tts/voices/design",
        json={
            "name": "Flat",
            "components": _recipe(("af_heart", 0.4), ("af_bella", 0.35), ("af_nicole", 0.25)),
        },
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "dominant" in detail
    assert "60%" in detail


def test_cross_language_family_blend_names_the_offending_voices(api):
    response = api.post(
        "/api/tts/voices/design",
        json={
            "name": "Cross Family",
            "components": _recipe(("af_heart", 0.7), ("bf_emma", 0.3)),
        },
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "af_heart" in detail and "bf_emma" in detail
    assert "en-US" in detail and "en-GB" in detail
    assert "language famil" in detail


def test_unknown_component_ids_are_rejected_with_the_valid_ids(api):
    response = api.post(
        "/api/tts/voices/design",
        json={
            "name": "Unknown Ingredient",
            "components": _recipe(("af_heart", 0.7), ("zz_ghost", 0.3)),
        },
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "zz_ghost" in detail
    assert "af_heart" in detail  # the valid-id list is actionable, not just an error
    assert "Valid base voice ids" in detail


def test_a_designed_voice_cannot_be_a_component(api):
    response = api.post(
        "/api/tts/voices/design",
        json={
            "name": "Blend Of Blend",
            "components": _recipe(("warm_narrator", 0.7), ("af_bella", 0.3)),
        },
    )

    assert response.status_code == 422
    assert "designed voice" in response.json()["detail"]


# ─────────────────────────────────────────────────────────────
# F1: ingredients resolve against the shipped pack only
# ─────────────────────────────────────────────────────────────

def test_base_voice_ids_returns_pack_ids_only(api):
    """A designed voice (blend *or* base alias) is never a valid ingredient."""
    api.post("/api/tts/voices", json={"name": "Heart Alias", "base_voice": "af_heart"})
    api.post(
        "/api/tts/voices/design",
        json={
            "name": "Some Blend",
            "components": _recipe(("af_heart", 0.7), ("af_bella", 0.3)),
        },
    )

    ids = VoiceDesignService(_service()).base_voice_ids()

    assert set(ids) == set(SHIPPED_VOICE_IDS)
    assert "heart_alias" not in ids
    assert "some_blend" not in ids
    assert "warm_narrator" not in ids


def test_a_base_alias_cannot_be_a_blend_component(api):
    """An alias is a designed voice, so it cannot be an ingredient.

    Accepting it produced a 201 for a voice that could never render (the alias
    id is not a pack key), was invisible in discovery, and could not be fixed
    because there is no update endpoint.
    """
    alias = api.post(
        "/api/tts/voices", json={"name": "Heart Alias", "base_voice": "af_heart"}
    )
    assert alias.status_code == 201

    response = api.post(
        "/api/tts/voices/design",
        json={
            "name": "Alias Blend",
            "components": _recipe(("heart_alias", 0.7), ("af_bella", 0.3)),
        },
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "heart_alias" in detail
    assert "designed voice" in detail
    assert "base" in detail.lower()
    assert "Valid base voice ids" in detail

    # Nothing was persisted, so the rejected voice cannot linger as a broken row.
    assert "alias_blend" not in {v["id"] for v in api.get("/api/tts/voices").json()["voices"]}


def test_a_base_alias_cannot_be_an_alias_target(api):
    """An alias of an alias would advertise a name for a deletable recipe."""
    assert (
        api.post(
            "/api/tts/voices", json={"name": "Alias One", "base_voice": "af_heart"}
        ).status_code
        == 201
    )

    response = api.post(
        "/api/tts/voices", json={"name": "Alias Two", "base_voice": "alias_one"}
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "alias_one" in detail
    assert "Alias target" in detail
    assert "designed voice" in detail
    assert "Valid base voice ids" in detail

    assert "alias_two" not in {v["id"] for v in api.get("/api/tts/voices").json()["voices"]}


def test_a_pack_voice_alias_still_works_after_the_ingredient_rule(api):
    """The rule narrows ingredients, not the alias feature itself."""
    response = api.post(
        "/api/tts/voices", json={"name": "Heart Alias", "base_voice": "af_heart"}
    )

    assert response.status_code == 201
    assert response.json()["style_spec"] == {"type": "base", "voice": "af_heart"}


def test_a_zero_weight_component_is_accepted_and_normalised(api):
    """Documented behaviour: 1.0 + 0.0 passes the literal rules.

    The weight is non-negative and the sum is positive, so it is a valid
    (if pointless) recipe.  See ``docs/voices/designed-voices.md`` — documented
    rather than rejected, so a caller can scale weights freely.
    """
    response = api.post(
        "/api/tts/voices/design",
        json={"name": "Weightless", "components": _recipe(("af_heart", 1.0), ("af_bella", 0.0))},
    )

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["normalised_weights"] == {"af_bella": 0.0, "af_heart": 1.0}
    assert sum(body["normalised_weights"].values()) == pytest.approx(1.0)


def test_duplicate_components_are_rejected(api):
    response = api.post(
        "/api/tts/voices/design",
        json={"name": "Duplicate", "components": _recipe(("af_heart", 0.7), ("af_heart", 0.3))},
    )

    assert response.status_code == 422
    assert "Duplicate" in response.json()["detail"]


def test_language_assertion_must_match_the_components(api):
    response = api.post(
        "/api/tts/voices/design",
        json={
            "name": "Wrong Language",
            "language": "ja",
            "components": _recipe(("af_heart", 0.7), ("af_bella", 0.3)),
        },
    )

    assert response.status_code == 422
    assert "en-US" in response.json()["detail"]


def test_weights_are_normalised_internally():
    """Arbitrary-scale weights are normalised to 1, not rejected."""
    components = _recipe(("af_heart", 3.0), ("af_bella", 2.0))
    service = VoiceDesignService(_service())
    resolved = service.validate_blend(components)

    assert resolved["language"] == "en-US"
    assert sum(component["weight"] for component in resolved["components"]) == pytest.approx(5.0)
    normalised = normalised_components(
        {"type": "blend", "components": resolved["components"]}
    )
    weights = {component["voice"]: component["weight"] for component in normalised}
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["af_heart"] == pytest.approx(0.6)


def test_validate_components_is_pure_and_reusable():
    assert validate_components(_recipe(("af_heart", 0.7), ("af_bella", 0.3))) == [
        ("af_heart", 0.7),
        ("af_bella", 0.3),
    ]


# ─────────────────────────────────────────────────────────────
# Creation and descriptor
# ─────────────────────────────────────────────────────────────

def test_design_blend_persists_the_recipe_and_returns_the_descriptor(api, service):
    response = api.post(
        "/api/tts/voices/design",
        json={
            "name": "Evening Host",
            "components": _recipe(("af_heart", 0.65), ("af_bella", 0.35)),
            "tags": ["host", "evening"],
            "default_direction": {"prosody": {"rate": 0.9}},
        },
    )

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["id"] == "evening_host"
    assert body["style_type"] == "blend"
    assert body["resolved_language"] == "en-US"
    assert body["language"] == "en-US"
    assert body["created_at"]
    assert body["preview_url"] == "/api/tts/voices/evening_host/preview"
    assert body["default_direction"] == {"prosody": {"rate": 0.9}}
    assert body["quality"] is None
    assert body["preset"] is False
    assert "designed" in body["tags"] and "host" in body["tags"]

    # Components carry both the authored and the normalised weight.
    assert [(c["voice"], c["weight"]) for c in body["components"]] == [
        ("af_heart", 0.65),
        ("af_bella", 0.35),
    ]
    assert {
        c["voice"]: c["normalised_weight"] for c in body["components"]
    } == pytest.approx({"af_heart": 0.65, "af_bella": 0.35})
    assert sum(body["normalised_weights"].values()) == pytest.approx(1.0)

    # The durable artefact is the recipe.  The tensor is never stored.
    persisted = json.loads(Path(settings.designed_voices_path).read_text(encoding="utf-8"))
    entry = next(v for v in persisted["voices"] if v["id"] == "evening_host")
    assert entry["style_spec"] == {
        "type": "blend",
        "components": [
            {"voice": "af_heart", "weight": 0.65},
            {"voice": "af_bella", "weight": 0.35},
        ],
    }
    serialised = json.dumps(entry).lower()
    assert "tensor" not in serialised
    assert "ndarray" not in serialised

    # And the voice is selectable everywhere a pack voice is.
    listing = api.get("/api/tts/voices").json()
    assert "evening_host" in {voice["id"] for voice in listing["voices"]}
    assert service.get_voice("evening_host")["name"] == "Evening Host"


def test_duplicate_name_returns_409(api):
    payload = {
        "name": "Twin Voice",
        "components": _recipe(("af_heart", 0.7), ("af_bella", 0.3)),
    }

    assert api.post("/api/tts/voices/design", json=payload).status_code == 201
    response = api.post("/api/tts/voices/design", json=payload)

    assert response.status_code == 409
    assert "already exists" in response.json()["detail"]


def test_design_does_not_require_a_loaded_model():
    """Validation uses the registry manifest before the model is resident."""
    service = VoiceDesignService(TTSService())
    assert service.base_voice_ids()
    assert "warm_narrator" not in service.base_voice_ids()

    resolved = service.validate_blend(_recipe(("af_heart", 0.7), ("af_bella", 0.3)))

    assert resolved["language"] == "en-US"


# ─────────────────────────────────────────────────────────────
# Preview cache
# ─────────────────────────────────────────────────────────────

def test_create_synthesises_a_preview_and_get_serves_wav(api, service):
    created = api.post(
        "/api/tts/voices/design",
        json={
            "name": "Preview Voice",
            "components": _recipe(("af_heart", 0.7), ("af_bella", 0.3)),
        },
    ).json()
    assert len(service._kokoro.calls) == 1  # warmed on create
    assert service._kokoro.calls[0]["text"]  # the fixed preview sentence was rendered
    entry = designed_voices.get_store().get_entry(created["id"])
    assert entry["preview"]["text"] == PREVIEW_TEXT

    response = api.get(created["preview_url"])

    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert response.content.startswith(b"RIFF")
    assert response.headers["x-preview-cache"] == "hit"
    assert len(service._kokoro.calls) == 1  # a cache hit must not re-synthesise


def test_preview_regenerates_when_the_recipe_changes(api, service):
    created = api.post(
        "/api/tts/voices/design",
        json={
            "name": "Changing Voice",
            "components": _recipe(("af_heart", 0.7), ("af_bella", 0.3)),
        },
    ).json()
    calls_before = len(service._kokoro.calls)

    store = designed_voices.get_store()
    store.update(
        created["id"],
        {
            "style_spec": {
                "type": "blend",
                "components": [
                    {"voice": "af_heart", "weight": 0.75},
                    {"voice": "af_bella", "weight": 0.25},
                ],
            }
        },
    )

    response = api.get(created["preview_url"])

    assert response.status_code == 200
    assert response.headers["x-preview-cache"] == "miss"
    assert len(service._kokoro.calls) == calls_before + 1


def test_preview_survives_a_restart(api, service):
    created = api.post(
        "/api/tts/voices/design",
        json={
            "name": "Durable Voice",
            "components": _recipe(("af_heart", 0.7), ("af_bella", 0.3)),
        },
    ).json()
    calls_before = len(service._kokoro.calls)

    # Simulate a new process: the in-memory store is rebuilt from the file.
    designed_voices.reset_store(settings.designed_voices_path)

    response = api.get(created["preview_url"])

    assert response.status_code == 200
    assert response.headers["x-preview-cache"] == "hit"
    assert len(service._kokoro.calls) == calls_before

    # The reloaded recipe is still a usable voice, not just a readable record.
    synthesised = api.post(
        "/api/tts/synthesize",
        json={
            "text": "Still here after a restart.",
            "voice": created["id"],
            "phonetic_preprocess": False,
        },
    )
    assert synthesised.status_code == 200
    assert isinstance(service._kokoro.calls[-1]["voice"], np.ndarray)


def test_preview_of_an_unknown_voice_returns_404(api):
    response = api.get("/api/tts/voices/nope_not_here/preview")

    assert response.status_code == 404
    assert "nope_not_here" in response.json()["detail"]


def test_the_recipe_is_readable_after_a_restart(api):
    created = api.post(
        "/api/tts/voices/design",
        json={
            "name": "Readable Recipe",
            "components": _recipe(("af_heart", 0.7), ("af_bella", 0.3)),
        },
    ).json()
    designed_voices.reset_store(settings.designed_voices_path)

    response = api.get(f"/api/tts/voices/{created['id']}/design")

    assert response.status_code == 200
    body = response.json()
    assert body["style_spec"] == created["style_spec"]
    assert body["resolved_language"] == "en-US"
    assert {c["voice"] for c in body["components"]} == {"af_heart", "af_bella"}
    assert body["created_at"] == created["created_at"]

    assert api.get("/api/tts/voices/af_heart/design").status_code == 404
    assert api.get("/api/tts/voices/never_existed/design").status_code == 404


# ─────────────────────────────────────────────────────────────
# Synthesis with a designed voice
# ─────────────────────────────────────────────────────────────

def test_designed_voice_synthesises_through_both_endpoints(api, service):
    voice_id = api.post(
        "/api/tts/voices/design",
        json={
            "name": "Synth Voice",
            "components": _recipe(("af_heart", 0.7), ("af_bella", 0.3)),
        },
    ).json()["id"]
    service._kokoro.calls.clear()

    wav = api.post(
        "/api/tts/synthesize",
        json={
            "text": "Hello from a designed voice.",
            "voice": voice_id,
            "phonetic_preprocess": False,
        },
    )
    assert wav.status_code == 200
    assert wav.headers["content-type"] == "audio/wav"
    assert wav.content.startswith(b"RIFF")
    assert isinstance(service._kokoro.calls[-1]["voice"], np.ndarray)

    payload = api.post(
        "/api/tts/synthesize/json",
        json={"text": "Hello again.", "voice": voice_id, "phonetic_preprocess": False},
    )
    assert payload.status_code == 200
    assert payload.json()["engine"] == "kokoro"
    assert isinstance(service._kokoro.calls[-1]["voice"], np.ndarray)


def test_designed_voice_applies_its_default_direction(api, service):
    """A preset's default rate is used only when the request expresses nothing."""
    service._kokoro.calls.clear()

    response = api.post(
        "/api/tts/synthesize",
        json={"text": "Preset please.", "voice": "warm_narrator", "phonetic_preprocess": False},
    )

    assert response.status_code == 200
    assert service._kokoro.calls[-1]["speed"] == pytest.approx(0.95)
    assert response.headers["x-direction-applied"] == '{"rate":0.95}'

    # An explicit speed wins over the stored default.
    service._kokoro.calls.clear()
    api.post(
        "/api/tts/synthesize",
        json={
            "text": "Faster please.",
            "voice": "warm_narrator",
            "speed": 1.3,
            "phonetic_preprocess": False,
        },
    )
    assert service._kokoro.calls[-1]["speed"] == pytest.approx(1.3)


def test_base_alias_does_not_duplicate_pack_data(api, service):
    pack_before = (APP_ROOT / "app" / "data" / "voices.json").read_bytes()

    response = api.post(
        "/api/tts/voices",
        json={"name": "Heart Alias", "base_voice": "af_heart"},
    )

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["style_type"] == "base"
    assert body["style_spec"] == {"type": "base", "voice": "af_heart"}
    assert body["resolved_language"] == "en-US"

    persisted = json.loads(Path(settings.designed_voices_path).read_text(encoding="utf-8"))
    alias = next(v for v in persisted["voices"] if v["id"] == "heart_alias")
    assert alias["style_spec"] == {"type": "base", "voice": "af_heart"}

    # The shipped pack is untouched and the alias synthesises through the base id.
    assert (APP_ROOT / "app" / "data" / "voices.json").read_bytes() == pack_before
    service._kokoro.calls.clear()
    wav = api.post(
        "/api/tts/synthesize",
        json={"text": "Alias.", "voice": "heart_alias", "phonetic_preprocess": False},
    )
    assert wav.status_code == 200
    assert service._kokoro.calls[-1]["voice"] == "af_heart"


def test_base_alias_accepts_the_voice_field_alias(api):
    """The body may name the pack voice as ``voice`` as well as ``base_voice``."""
    response = api.post(
        "/api/tts/voices",
        json={"name": "Alias Field", "voice": "af_heart"},
    )

    assert response.status_code == 201, response.text
    assert response.json()["style_spec"] == {"type": "base", "voice": "af_heart"}


def test_base_alias_rejects_an_unknown_base_voice(api):
    response = api.post(
        "/api/tts/voices",
        json={"name": "Ghost Alias", "base_voice": "zz_ghost"},
    )

    assert response.status_code == 422
    assert "zz_ghost" in response.json()["detail"]


# ─────────────────────────────────────────────────────────────
# Deletion
# ─────────────────────────────────────────────────────────────

def test_deleting_a_pack_voice_returns_409(api):
    response = api.delete("/api/tts/voices/af_heart")

    assert response.status_code == 409
    assert "pack voice" in response.json()["detail"]
    assert api.get("/api/tts/voices/af_heart").status_code == 200


def test_deleting_a_preset_returns_409(api):
    response = api.delete("/api/tts/voices/warm_narrator")

    assert response.status_code == 409
    assert "preset" in response.json()["detail"]


def test_deleting_a_custom_voice_returns_200_and_removes_it(api):
    voice_id = api.post(
        "/api/tts/voices/design",
        json={
            "name": "Throwaway",
            "components": _recipe(("af_heart", 0.7), ("af_bella", 0.3)),
        },
    ).json()["id"]
    assert voice_id in {v["id"] for v in api.get("/api/tts/voices").json()["voices"]}
    preview_dir = designed_voices.resolve_preview_dir(Path(settings.designed_voices_path))
    assert list(preview_dir.glob(f"{voice_id}-*.wav"))

    response = api.delete(f"/api/tts/voices/{voice_id}")

    assert response.status_code == 200
    assert response.json() == {"status": "deleted", "id": voice_id}
    assert voice_id not in {v["id"] for v in api.get("/api/tts/voices").json()["voices"]}
    assert api.get(f"/api/tts/voices/{voice_id}").status_code == 404
    assert list(preview_dir.glob(f"{voice_id}-*.wav")) == []


def test_deleting_an_unknown_voice_returns_404(api):
    response = api.delete("/api/tts/voices/never_existed")

    assert response.status_code == 404


# ─────────────────────────────────────────────────────────────
# The tensor resolver and its cache
# ─────────────────────────────────────────────────────────────

def test_blend_resolves_to_a_weighted_sum_of_style_tensors():
    engine = KokoroEngine()
    fake = FakeKokoro(shape=(510, 1, 256))
    engine.model = fake
    engine.set_ready(True)

    style_spec = {
        "type": "blend",
        "components": [
            {"voice": "af_heart", "weight": 0.7},
            {"voice": "af_bella", "weight": 0.3},
        ],
    }
    tensor = engine.resolve_blend(style_spec)
    expected = (
        0.7 * fake.get_voice_style("af_heart") + 0.3 * fake.get_voice_style("af_bella")
    ).astype(np.float32)

    assert tensor.shape == (510, 1, 256)
    assert tensor.dtype == np.float32
    np.testing.assert_allclose(tensor, expected, rtol=1e-6)
    assert blend_cache_size() == 1


def test_recipe_hash_is_order_independent():
    forward = {
        "type": "blend",
        "components": [
            {"voice": "af_heart", "weight": 0.7},
            {"voice": "af_bella", "weight": 0.3},
        ],
    }
    reversed_order = {
        "type": "blend",
        "components": [
            {"voice": "af_bella", "weight": 0.3},
            {"voice": "af_heart", "weight": 0.7},
        ],
    }

    assert recipe_hash(forward) == recipe_hash(reversed_order)


def test_tensor_cache_is_bounded_and_clearable(monkeypatch):
    monkeypatch.setattr(settings, "designed_voice_tensor_cache_size", 2)
    engine = KokoroEngine()
    engine.model = FakeKokoro()
    engine.set_ready(True)

    for index in range(3):
        engine.resolve_blend(
            {
                "type": "blend",
                "components": [
                    {"voice": "af_heart", "weight": 0.7 + index * 0.01},
                    {"voice": "af_bella", "weight": 0.3 - index * 0.01},
                ],
            }
        )

    assert blend_cache_size() == 2
    assert blend_cache_bytes() == 2 * int(np.prod(STYLE_SHAPE)) * 4

    clear_blend_cache()
    assert blend_cache_size() == 0
    assert blend_cache_bytes() == 0


def test_twenty_designed_voices_do_not_copy_or_reload_the_model(api, service):
    model_before = service._kokoro
    pack_size = len(model_before.voices)
    clear_blend_cache()

    created = []
    for index in range(20):
        dominant = 0.70 + index * 0.01
        response = api.post(
            "/api/tts/voices/design",
            json={
                "name": f"Batch Voice {index}",
                "components": _recipe(
                    ("af_heart", round(dominant, 2)),
                    ("af_bella", round(1.0 - dominant, 2)),
                ),
            },
        )
        assert response.status_code == 201, response.text
        created.append(response.json()["id"])

    assert service._kokoro is model_before  # same model instance
    assert len(model_before.voices) == pack_size  # nothing copied into the pack
    assert len(created) == 20
    assert len(designed_voices.get_store()) == 24  # 4 presets + 20 designed

    # Memory is bounded by the tensor cache, one tensor per distinct recipe.
    assert blend_cache_size() == 20
    assert blend_cache_bytes() <= 20 * 510 * 1 * 256 * 4  # real-shape upper bound

    # The cache can be cleared, and synthesis re-derives the tensor on demand.
    clear_blend_cache()
    assert blend_cache_size() == 0
    wav = api.post(
        "/api/tts/synthesize",
        json={"text": "Still works.", "voice": created[0], "phonetic_preprocess": False},
    )
    assert wav.status_code == 200
    assert blend_cache_size() == 1


# ─────────────────────────────────────────────────────────────
# Persistence tolerance
# ─────────────────────────────────────────────────────────────

def test_a_missing_registry_file_is_tolerated(tmp_path, monkeypatch):
    missing = tmp_path / "does-not-exist.json"
    monkeypatch.setattr(settings, "designed_voices_path", str(missing))
    store = designed_voices.reset_store(missing)

    assert len(store) == 0
    assert store.ids() == ()


def test_a_malformed_registry_file_is_tolerated(tmp_path, monkeypatch, caplog):
    path = tmp_path / "designed_voices.json"
    path.write_text("{ this is not json", encoding="utf-8")
    monkeypatch.setattr(settings, "designed_voices_path", str(path))

    with caplog.at_level("WARNING"):
        store = designed_voices.reset_store(path)

    assert len(store) == 0
    assert any("not valid JSON" in record.message for record in caplog.records)

    # Pack voices still work: the bad file never blocks startup or discovery.
    client = _client(_service())
    body = client.get("/api/tts/voices").json()
    assert body["total"] == len(SHIPPED_VOICE_IDS)
    assert "af_heart" in {voice["id"] for voice in body["voices"]}


def test_a_corrupt_registry_is_preserved_before_the_first_write(tmp_path, monkeypatch, caplog):
    """F2: a tolerant load must not let the next write destroy the file."""
    path = tmp_path / "designed_voices.json"
    corrupt = "{ this is not json"
    path.write_text(corrupt, encoding="utf-8")
    monkeypatch.setattr(settings, "designed_voices_path", str(path))
    designed_voices.reset_store(path)

    client = _client(_service())
    with caplog.at_level("WARNING"):
        response = client.post(
            "/api/tts/voices/design",
            json={
                "name": "After Corruption",
                "components": _recipe(("af_heart", 0.7), ("af_bella", 0.3)),
            },
        )

    assert response.status_code == 201, response.text
    backups = sorted(tmp_path.glob("designed_voices.json.corrupt-*"))
    assert len(backups) == 1
    # The original bytes are preserved verbatim, so nothing is destroyed silently.
    assert backups[0].read_text(encoding="utf-8") == corrupt
    assert any(str(backups[0]) in record.message for record in caplog.records)

    # A fresh, valid registry now holds the new voice.
    fresh = json.loads(path.read_text(encoding="utf-8"))
    assert [entry["id"] for entry in fresh["voices"]] == ["after_corruption"]
    assert "after_corruption" in {v["id"] for v in client.get("/api/tts/voices").json()["voices"]}

    # The shipped presets were never touched on disk and remain available.
    assert designed_voices.preset_voice_ids() == preset_voice_ids()


def test_presets_are_recoverable_from_the_backup_of_a_partly_corrupt_file(
    tmp_path, monkeypatch
):
    """A file with a broken row loses it in memory, never on disk."""
    seed = json.loads(
        designed_voices.DEFAULT_DESIGNED_VOICES_PATH.read_text(encoding="utf-8")
    )
    seed["voices"].append({"id": "broken", "name": "Broken"})
    path = tmp_path / "designed_voices.json"
    path.write_text(json.dumps(seed), encoding="utf-8")
    monkeypatch.setattr(settings, "designed_voices_path", str(path))
    designed_voices.reset_store(path)

    client = _client(_service())
    response = client.post(
        "/api/tts/voices/design",
        json={"name": "Survivor", "components": _recipe(("af_heart", 0.7), ("af_bella", 0.3))},
    )
    assert response.status_code == 201, response.text

    backups = sorted(tmp_path.glob("designed_voices.json.corrupt-*"))
    assert len(backups) == 1
    # The preserved file still contains every original entry, presets included.
    assert designed_voices.preset_voice_ids(backups[0]) == preset_voice_ids()
    preserved = json.loads(backups[0].read_text(encoding="utf-8"))
    assert {entry["id"] for entry in preserved["voices"]} >= set(preset_voice_ids())


def test_a_good_registry_is_rewritten_without_a_backup(tmp_path, monkeypatch):
    """The preservation path must not fire on a healthy file."""
    path = tmp_path / "designed_voices.json"
    seed = designed_voices.DEFAULT_DESIGNED_VOICES_PATH.read_text(encoding="utf-8")
    path.write_text(seed, encoding="utf-8")
    monkeypatch.setattr(settings, "designed_voices_path", str(path))
    designed_voices.reset_store(path)

    client = _client(_service())
    assert (
        client.post(
            "/api/tts/voices/design",
            json={
                "name": "Healthy Write",
                "components": _recipe(("af_heart", 0.7), ("af_bella", 0.3)),
            },
        ).status_code
        == 201
    )

    assert list(tmp_path.glob("designed_voices.json.corrupt-*")) == []
    assert designed_voices.preset_voice_ids(path) == preset_voice_ids()


def test_invalid_entries_are_skipped_without_dropping_valid_ones(tmp_path, monkeypatch):
    seed = json.loads(
        designed_voices.DEFAULT_DESIGNED_VOICES_PATH.read_text(encoding="utf-8")
    )
    seed["voices"].append({"id": "broken", "name": "Broken"})
    path = tmp_path / "designed_voices.json"
    path.write_text(json.dumps(seed), encoding="utf-8")
    monkeypatch.setattr(settings, "designed_voices_path", str(path))

    store = designed_voices.reset_store(path)

    assert "broken" not in store.ids()
    assert set(store.ids()) == set(preset_voice_ids())


# ─────────────────────────────────────────────────────────────
# Shipped presets and documentation
# ─────────────────────────────────────────────────────────────

def test_four_presets_ship_as_registry_data():
    ids = preset_voice_ids()

    assert len(ids) == 4
    names = {voice_registry.get_record(voice_id).name for voice_id in ids}
    assert names == {"Warm Narrator", "Calm Assistant", "Bright Support", "Steady Narration"}

    service = VoiceDesignService(_service())
    for voice_id in ids:
        record = voice_registry.get_record(voice_id)
        assert record is not None
        assert record.style_type == "blend"
        resolved = service.validate_blend(
            [
                {"voice": component["voice"], "weight": component["weight"]}
                for component in record.style_spec["components"]
            ]
        )
        assert resolved["language"] in {"en-US", "en-GB"}
        assert resolved["dominant_weight"] >= 0.6


@pytest.mark.slow
@requires_models
async def test_real_engine_renders_a_designed_blend():
    """F3: pin the real-tensor hand-off with the shipped model.

    Everything else in this module uses a deterministic double, so a change to
    ``_render``, the ndarray hand-off or ``kokoro_onnx`` itself could break real
    blending with the fast suite green.  Skips without the model files, exactly
    like the other slow tests.
    """
    clear_blend_cache()
    service = TTSService()
    await service.initialize()
    assert service.is_ready, "the real Kokoro model should load"

    design = VoiceDesignService(service)
    created = await design.create_blend(
        name="Real Blend",
        components=[
            {"voice": "af_heart", "weight": 0.65},
            {"voice": "am_adam", "weight": 0.35},
        ],
    )
    voice_id = created["id"]

    # Preview: real inference, real WAV, cached by recipe hash.
    preview = await design.ensure_preview(voice_id)
    assert preview["audio"].startswith(b"RIFF")
    assert preview["audio"] == (await design.ensure_preview(voice_id))["audio"]

    # Synthesis: the ndarray reaches Kokoro.create and produces audio.
    wav, duration = service.synthesize(
        "A designed voice rendered by the real model.", voice=voice_id, speed=1.0
    )
    assert wav.startswith(b"RIFF")
    assert len(wav) > 1000
    assert duration > 0

    # The blend really is a derived tensor, and the model was loaded once.
    assert KokoroEngine.blend_cache_size() == 1
    assert service.engine_manager.any_ready()

    # A single base voice still renders through the unchanged string path.
    base_wav, base_duration = service.synthesize(
        "A base voice from the same model.", voice="af_heart", speed=1.0
    )
    assert base_wav.startswith(b"RIFF")
    assert base_duration > 0


def test_designed_voices_documentation_states_timbre_not_emotion():
    assert DESIGNED_DOC.is_file(), "docs/voices/designed-voices.md must document the presets"
    doc = DESIGNED_DOC.read_text(encoding="utf-8").lower()
    readme = (APP_ROOT / "README.md").read_text(encoding="utf-8").lower()

    for text in (doc, readme):
        assert "timbre" in text
        assert "not change emotion" in text
    for name in ("warm narrator", "calm assistant", "bright support", "steady narration"):
        assert name in doc


def test_documentation_covers_the_operational_trade_offs():
    """F6/F7: the docs must name the recovery path and the single-process limit."""
    doc = DESIGNED_DOC.read_text(encoding="utf-8").lower()
    catalogue = (APP_ROOT / "docs" / "voices" / "kokoro-voice-catalogue.md").read_text(
        encoding="utf-8"
    )

    assert ".corrupt-" in doc
    assert "single-process" in doc
    assert "zero-weight" in doc
    assert "designed-voices.md" in catalogue
