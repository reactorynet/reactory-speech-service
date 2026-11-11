"""Tests for the Kokoro voice catalogue, the voice API and synthesis guards."""

import logging
import re
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app as fastapi_app
from app.services.tts_service import TTSService
from app.services import voice_registry
from app.services.voice_registry import (
    KOKORO_SAMPLE_RATE,
    LANGUAGE_FAMILIES,
    QUALITY_GRADES,
    SHIPPED_VOICE_IDS,
    UNGRADED_VOICE_IDS,
    build_catalogue,
    describe_voice,
    filter_catalogue,
)

MODEL_PATH = Path(settings.kokoro_model_resolved)
VOICES_PATH = Path(settings.kokoro_voices_resolved)
CATALOGUE_DOC = Path(__file__).resolve().parent.parent / "docs" / "voices" / "kokoro-voice-catalogue.md"

requires_models = pytest.mark.skipif(
    not (MODEL_PATH.exists() and VOICES_PATH.exists()),
    reason="Kokoro model files are not present in models/",
)
requires_catalogue_doc = pytest.mark.skipif(
    not CATALOGUE_DOC.exists(),
    reason="docs/voices/kokoro-voice-catalogue.md is not on this branch (owned by PR #6)",
)

# One representative voice and sample text per language family.
LANGUAGE_SAMPLES = {
    "en-US": ("af_heart", "Hello from Reactor."),
    "en-GB": ("bm_george", "Hello from Reactor."),
    "ja": ("jf_alpha", "こんにちは、リアクターです。"),
    "zh": ("zf_xiaoxiao", "你好，我是反应堆。"),
    "es": ("ef_dora", "Hola, soy un asistente de voz."),
    "fr-FR": ("ff_siwis", "Bonjour, je suis un assistant vocal."),
    "hi": ("hf_alpha", "नमस्ते, मैं एक सहायक हूँ।"),
    "it": ("if_sara", "Ciao, sono un assistente vocale."),
    "pt-BR": ("pf_dora", "Olá, sou um assistente de voz."),
}

# Grades pinned against docs/voices/kokoro-voice-catalogue.md. The four
# ``corrected`` entries are the ones an earlier revision got wrong; they are
# listed explicitly so the table cannot silently drift again.
DOCUMENTED_GRADES = {
    # corrected entries — assert these first if this test ever fails
    "bf_lily": "C",
    "ff_siwis": "B-",
    "jf_nezumi": "C-",
    "jm_kumo": "C-",
    # remaining grade buckets, one or more representatives each
    "af_heart": "A",
    "af_bella": "A-",
    "af_nicole": "B-",
    "bf_emma": "B-",
    "af_aoede": "C+",
    "af_kore": "C+",
    "af_sarah": "C+",
    "am_fenrir": "C+",
    "am_michael": "C+",
    "am_puck": "C+",
    "jf_alpha": "C+",
    "af_alloy": "C",
    "af_nova": "C",
    "bf_isabella": "C",
    "bm_fable": "C",
    "bm_george": "C",
    "hf_alpha": "C",
    "hf_beta": "C",
    "hm_omega": "C",
    "hm_psi": "C",
    "if_sara": "C",
    "im_nicola": "C",
    "jf_gongitsune": "C",
    "jf_tebukuro": "C",
    "zf_xiaoxiao": "C",
    "zm_yunyang": "C",
    "af_sky": "C-",
    "af_jessica": "D",
    "af_river": "D",
    "am_echo": "D",
    "am_eric": "D",
    "am_liam": "D",
    "am_onyx": "D",
    "bf_alice": "D",
    "bm_daniel": "D",
    "am_santa": "D-",
    "am_adam": "F+",
}


class FakeKokoro:
    """Minimal stand-in for ``kokoro_onnx.Kokoro``.

    Exposes a ``voices`` mapping (as the real engine does) and enforces the same
    contract for unknown voices and out-of-range speeds, without running inference.
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
        return np.zeros(2400, dtype=np.float32), KOKORO_SAMPLE_RATE


def _service_with_engine(voice_ids=SHIPPED_VOICE_IDS) -> TTSService:
    service = TTSService()
    service._kokoro = FakeKokoro(voice_ids)
    service._ready = True
    return service


def _client_for(service: TTSService) -> TestClient:
    fastapi_app.state.tts_service = service
    fastapi_app.state.stt_service = MagicMock(is_ready=False)
    return TestClient(fastapi_app)


@pytest.fixture
def catalogue_client():
    """Client backed by a real (not yet loaded) TTSService."""
    client = _client_for(TTSService())
    yield client
    fastapi_app.state.tts_service = TTSService()


@pytest.fixture
def fake_engine_service() -> TTSService:
    return _service_with_engine()


@pytest.fixture
def fake_engine_client(fake_engine_service):
    yield _client_for(fake_engine_service)


# ─────────────────────────────────────────────────────────────
# Voice registry
# ─────────────────────────────────────────────────────────────

def test_catalogue_covers_all_shipped_voices():
    """The manifest and the catalogue describe the same 54-voice, 9-language pack."""
    voices = build_catalogue(SHIPPED_VOICE_IDS)

    assert len(voices) == 54
    assert len({voice["id"] for voice in voices}) == 54
    assert len({voice["language"] for voice in voices}) == 9
    assert {voice["id"] for voice in voices} == set(SHIPPED_VOICE_IDS)


def test_quality_grades_match_the_catalogue_of_record():
    """Pin the grades transcribed from docs/voices/kokoro-voice-catalogue.md.

    Includes the four ids an earlier revision had wrong (bf_lily C, ff_siwis B-,
    jf_nezumi C-, jm_kumo C-).
    """
    drifted = {
        voice_id: {"expected": grade, "actual": QUALITY_GRADES.get(voice_id)}
        for voice_id, grade in DOCUMENTED_GRADES.items()
        if QUALITY_GRADES.get(voice_id) != grade
    }
    assert drifted == {}, f"quality grades drifted from the catalogue of record: {drifted}"


def test_ungraded_voices_are_not_given_invented_grades():
    """The 7 voices the catalogue does not grade report ``quality: None``."""
    assert sorted(UNGRADED_VOICE_IDS) == [
        "bm_lewis",
        "ef_dora",
        "em_alex",
        "em_santa",
        "pf_dora",
        "pm_alex",
        "pm_santa",
    ]

    for voice_id in UNGRADED_VOICE_IDS:
        assert voice_id not in QUALITY_GRADES, f"{voice_id} must not carry an invented grade"
        descriptor = describe_voice(voice_id)
        assert descriptor is not None
        assert descriptor["quality"] is None
        assert "unrated" in descriptor["tags"]
        assert not [tag for tag in descriptor["tags"] if tag.startswith("grade-")]


def test_grade_table_and_manifest_partition_the_pack():
    """Graded ∪ ungraded == shipped, and the two sets are disjoint.

    The manifest is a literal, deliberately not derived from the grade table, so a
    voice added to a future pack without a grade row still appears in discovery.
    """
    graded = set(QUALITY_GRADES)
    ungraded = set(UNGRADED_VOICE_IDS)

    assert graded & ungraded == set()
    assert graded | ungraded == set(SHIPPED_VOICE_IDS)
    assert len(graded) == 47
    assert len(ungraded) == 7


def test_grade_table_has_no_duplicate_or_orphan_entries():
    for voice_id in QUALITY_GRADES:
        assert voice_id in SHIPPED_VOICE_IDS, f"{voice_id} is graded but not shipped"


@pytest.mark.parametrize(
    ("voice_id", "language", "locale", "gender"),
    [
        ("af_heart", "en-US", "en-us", "female"),
        ("am_adam", "en-US", "en-us", "male"),
        ("bm_george", "en-GB", "en-gb", "male"),
        ("jf_alpha", "ja", "ja", "female"),
        ("zm_yunxi", "zh", "cmn", "male"),
        ("em_santa", "es", "es", "male"),
        ("ff_siwis", "fr-FR", "fr-fr", "female"),
        ("hf_beta", "hi", "hi", "female"),
        ("im_nicola", "it", "it", "male"),
        ("pf_dora", "pt-BR", "pt-br", "female"),
    ],
)
def test_describe_voice_derives_metadata_from_id(voice_id, language, locale, gender):
    descriptor = describe_voice(voice_id)

    assert descriptor is not None
    assert descriptor["language"] == language
    assert descriptor["locale"] == locale
    assert descriptor["gender"] == gender
    assert descriptor["engine"] == "kokoro"
    assert descriptor["name"]


def test_every_shipped_voice_parses():
    for voice_id in SHIPPED_VOICE_IDS:
        descriptor = describe_voice(voice_id)
        assert descriptor is not None, voice_id
        assert descriptor["language"] in {family.language for family in LANGUAGE_FAMILIES.values()}
        assert descriptor["gender"] in {"female", "male"}
        assert descriptor["tags"]


def test_mandarin_uses_the_espeak_cmn_alias():
    """espeak-ng (kokoro-onnx's G2P backend) rejects 'zh' but accepts 'cmn'."""
    descriptor = describe_voice("zf_xiaoxiao")

    assert descriptor is not None
    assert descriptor["language"] == "zh"
    assert descriptor["locale"] == "cmn"


def test_unknown_ids_are_not_described():
    assert describe_voice("") is None
    assert describe_voice("nope") is None
    assert describe_voice("xx_foo") is None


def test_off_scheme_engine_voices_are_exposed_not_dropped():
    """A voice the engine exposes must always be visible, even off-scheme."""
    voices = build_catalogue(["legacy_voice"])

    assert [voice["id"] for voice in voices] == ["legacy_voice"]
    assert voices[0]["language"] == "und"
    assert voices[0]["quality"] is None


def test_build_catalogue_sorted_by_language_then_id():
    voices = build_catalogue(["zm_yunxi", "af_heart", "bm_george", "jf_alpha"])

    assert [(voice["language"], voice["id"]) for voice in voices] == [
        ("en-GB", "bm_george"),
        ("en-US", "af_heart"),
        ("ja", "jf_alpha"),
        ("zh", "zm_yunxi"),
    ]


def test_filter_catalogue_is_case_insensitive():
    voices = build_catalogue(SHIPPED_VOICE_IDS)

    assert len(filter_catalogue(voices, language="en-us")) == 20
    assert len(filter_catalogue(voices, language="EN-US")) == 20
    assert len(filter_catalogue(voices, gender="MALE")) == 25
    assert len(filter_catalogue(voices, engine="KOKORO")) == 54
    assert filter_catalogue(voices, engine="piper") == []


def test_grade_tags_distinguish_plus_and_minus():
    """``grade-a`` must not silently mean both A and A-."""
    voices = {voice["id"]: voice for voice in build_catalogue(SHIPPED_VOICE_IDS)}

    assert "grade-a" in voices["af_heart"]["tags"]
    assert "grade-a-minus" in voices["af_bella"]["tags"]
    assert "grade-a" not in voices["af_bella"]["tags"]
    assert "grade-c" in voices["bf_lily"]["tags"]
    assert "grade-c-minus" in voices["jf_nezumi"]["tags"]
    assert "grade-b-minus" in voices["ff_siwis"]["tags"]

    grade_a = filter_catalogue(voices.values(), tag="grade-a")
    assert [voice["id"] for voice in grade_a] == ["af_heart"]


def test_recommended_tag_matches_the_documented_customer_facing_threshold():
    """The catalogue says "treat C+ and above as candidates for customer-facing defaults"."""
    voices = filter_catalogue(build_catalogue(SHIPPED_VOICE_IDS), tag="recommended")

    assert sorted(voice["id"] for voice in voices) == [
        "af_aoede",
        "af_bella",
        "af_heart",
        "af_kore",
        "af_nicole",
        "af_sarah",
        "am_fenrir",
        "am_michael",
        "am_puck",
        "bf_emma",
        "ff_siwis",
        "jf_alpha",
    ]
    assert len(voices) == 12
    assert all(voice["quality"] in {"A", "A-", "B-", "C+"} for voice in voices)


def test_unrated_tag_filter_returns_the_ungraded_voices():
    voices = filter_catalogue(build_catalogue(SHIPPED_VOICE_IDS), tag="unrated")

    assert sorted(voice["id"] for voice in voices) == sorted(UNGRADED_VOICE_IDS)


@requires_catalogue_doc
def test_grade_table_matches_the_catalogue_document_on_disk():
    """Guard against drift once docs/voices/kokoro-voice-catalogue.md lands (PR #6)."""
    text = CATALOGUE_DOC.read_text(encoding="utf-8")
    section = text.split("## Quality grades")[1].split("## ")[0]

    documented: dict[str, str] = {}
    blocks: dict[str, str] = {}
    for line in section.splitlines():
        match = re.match(r"\|\s*([A-F][+-]?)\s*\|\s*(.+?)\s*\|", line)
        if not match:
            continue
        grade, listed = match.group(1), match.group(2)
        for token in re.findall(r"`([a-z]{2}_[a-z]+|[a-z]{2}_\*)`", listed):
            if token.endswith("_*"):
                blocks[token[:-1]] = grade
            else:
                documented[token] = grade

    for prefix, grade in blocks.items():
        for voice_id in SHIPPED_VOICE_IDS:
            if voice_id.startswith(prefix):
                documented[voice_id] = grade

    assert documented == QUALITY_GRADES


# ─────────────────────────────────────────────────────────────
# GET /api/tts/voices
# ─────────────────────────────────────────────────────────────

def test_list_voices_returns_all_54_voices(catalogue_client):
    response = catalogue_client.get("/api/tts/voices")

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 54
    assert len(body["voices"]) == 54
    assert len({voice["language"] for voice in body["voices"]}) == 9


def test_list_voices_descriptor_shape(catalogue_client):
    response = catalogue_client.get("/api/tts/voices")
    heart = next(voice for voice in response.json()["voices"] if voice["id"] == "af_heart")

    assert heart == {
        "id": "af_heart",
        "name": "Heart",
        "language": "en-US",
        "engine": "kokoro",
        "gender": "female",
        "locale": "en-us",
        "quality": "A",
        "tags": ["female", "english", "american", "grade-a", "recommended"],
    }


def test_list_voices_reports_corrected_and_unrated_grades(catalogue_client):
    response = catalogue_client.get("/api/tts/voices")
    voices = {voice["id"]: voice for voice in response.json()["voices"]}

    assert voices["bf_lily"]["quality"] == "C"
    assert voices["ff_siwis"]["quality"] == "B-"
    assert voices["jf_nezumi"]["quality"] == "C-"
    assert voices["jm_kumo"]["quality"] == "C-"

    for voice_id in UNGRADED_VOICE_IDS:
        assert voices[voice_id]["quality"] is None
        assert "unrated" in voices[voice_id]["tags"]


def test_list_voices_reports_engine_availability(catalogue_client):
    """Discovery before load says so, rather than pretending synthesis will work."""
    body = catalogue_client.get("/api/tts/voices").json()

    assert body["engine"] == "kokoro"
    assert body["available"] is False


def test_list_voices_reports_availability_when_engine_is_loaded(fake_engine_client):
    body = fake_engine_client.get("/api/tts/voices").json()

    assert body["available"] is True


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("?language=en-US", 20),
        ("?language=en-us", 20),
        ("?language=en-GB", 8),
        ("?language=ja", 5),
        ("?language=zh", 8),
        ("?gender=female", 29),
        ("?gender=male", 25),
        ("?tag=recommended", 12),
        ("?tag=unrated", 7),
        ("?tag=grade-c-minus", 3),
        ("?engine=kokoro", 54),
        ("?engine=piper", 0),
        ("?engine=kokoro&language=ja", 5),
        ("?language=ja&gender=male", 1),
    ],
)
def test_list_voices_filters(catalogue_client, query, expected):
    response = catalogue_client.get(f"/api/tts/voices{query}")

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == expected
    assert len(body["voices"]) == expected


def test_list_voices_works_against_the_shared_mock_fixture(client):
    """The standard fixture returns the full descriptor shape (no KeyError)."""
    response = client.get("/api/tts/voices?gender=female")

    assert response.status_code == 200
    assert [voice["id"] for voice in response.json()["voices"]] == ["af_heart"]


# ─────────────────────────────────────────────────────────────
# GET /api/tts/voices/{voice_id}
# ─────────────────────────────────────────────────────────────

def test_get_voice_returns_descriptor(catalogue_client):
    response = catalogue_client.get("/api/tts/voices/af_heart")

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == "af_heart"
    assert body["name"] == "Heart"
    assert body["language"] == "en-US"
    assert body["engine"] == "kokoro"
    assert body["quality"] == "A"


def test_get_ungraded_voice_returns_null_quality(catalogue_client):
    response = catalogue_client.get("/api/tts/voices/bm_lewis")

    assert response.status_code == 200
    assert response.json()["quality"] is None


def test_get_unknown_voice_returns_404(catalogue_client):
    response = catalogue_client.get("/api/tts/voices/zz_unknown")

    assert response.status_code == 404
    detail = response.json()["detail"]
    assert "zz_unknown" in detail
    assert "GET /api/tts/voices" in detail


# ─────────────────────────────────────────────────────────────
# GET /api/tts/capabilities
# ─────────────────────────────────────────────────────────────

def test_capabilities_reports_kokoro_engine(catalogue_client):
    response = catalogue_client.get("/api/tts/capabilities")

    assert response.status_code == 200
    engines = response.json()["engines"]
    assert len(engines) == 1

    kokoro = engines[0]
    assert kokoro["id"] == "kokoro"
    assert kokoro["gpu_required"] is False
    assert kokoro["native_rate"] is True
    assert kokoro["native_pitch"] is False
    assert kokoro["native_volume"] is False
    assert kokoro["styles"] == ["neutral"]
    assert kokoro["supports_cloning"] is False
    assert kokoro["supports_streaming"] is True
    assert kokoro["sample_rates"] == [24000]
    assert kokoro["voice_count"] == 54
    assert kokoro["languages"] == [
        "en-GB",
        "en-US",
        "es",
        "fr-FR",
        "hi",
        "it",
        "ja",
        "pt-BR",
        "zh",
    ]


def test_capabilities_reports_custom_pack_voices():
    client = _client_for(_service_with_engine(["af_heart", "jf_alpha"]))

    kokoro = client.get("/api/tts/capabilities").json()["engines"][0]

    assert kokoro["voice_count"] == 2
    assert kokoro["languages"] == ["en-US", "ja"]
    assert kokoro["available"] is True


def test_capabilities_reports_unavailable_engine(catalogue_client):
    kokoro = catalogue_client.get("/api/tts/capabilities").json()["engines"][0]

    assert kokoro["available"] is False


# ─────────────────────────────────────────────────────────────
# Speed validation
# ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("speed", [0.4, 0.25, 2.5, 4.0])
def test_out_of_range_speed_returns_422(client, speed):
    response = client.post("/api/tts/synthesize", json={"text": "Test", "speed": speed})

    assert response.status_code == 422


@pytest.mark.parametrize("speed", [0.5, 1.0, 2.0])
def test_speed_bounds_are_accepted(client, speed):
    response = client.post("/api/tts/synthesize", json={"text": "Test", "speed": speed})

    assert response.status_code == 200


def test_service_clamps_speed_defensively():
    """Callers that bypass the schema (WebSocket) still cannot reach the engine out of range."""
    assert TTSService.clamp_speed(0.1) == 0.5
    assert TTSService.clamp_speed(4.0) == 2.0
    assert TTSService.clamp_speed(1.0) == 1.0
    assert TTSService.clamp_speed(None) == pytest.approx(settings.kokoro_default_speed)


def test_websocket_speed_is_clamped_before_inference(fake_engine_client, fake_engine_service):
    import json

    with fake_engine_client.websocket_connect("/api/tts/stream") as ws:
        ws.send_text(json.dumps({"text": "Hello world.", "speed": 5.0}))
        ws.receive_bytes()
        assert ws.receive_json()["done"] is True

    assert fake_engine_service._kokoro.calls[0]["speed"] == 2.0


# ─────────────────────────────────────────────────────────────
# Synthesis headers
# ─────────────────────────────────────────────────────────────

def test_binary_synthesis_emits_both_sample_rate_headers(fake_engine_client):
    response = fake_engine_client.post(
        "/api/tts/synthesize",
        json={"text": "Hello.", "voice": "af_heart", "phonetic_preprocess": False},
    )

    assert response.status_code == 200
    assert response.headers["x-sample-rate"] == "24000"
    assert response.headers["x-audio-sample-rate"] == "24000"
    assert response.headers["x-duration"] == response.headers["x-audio-duration"]
    assert response.headers["x-format"] == "wav"
    assert response.headers["x-audio-format"] == "wav"


def test_json_synthesis_emits_both_sample_rate_headers(fake_engine_client):
    """The JSON endpoint carries the same metadata, so the guarantee is endpoint-agnostic."""
    response = fake_engine_client.post(
        "/api/tts/synthesize/json",
        json={"text": "Hello.", "voice": "af_heart", "phonetic_preprocess": False},
    )

    assert response.status_code == 200
    assert response.headers["x-sample-rate"] == "24000"
    assert response.headers["x-audio-sample-rate"] == "24000"
    assert response.headers["x-duration"] == "0.1"
    assert response.headers["x-audio-duration"] == "0.1"
    assert response.headers["x-format"] == "wav"
    assert response.headers["x-audio-format"] == "wav"
    # The body still carries the same values.
    body = response.json()
    assert body["sample_rate"] == 24000
    assert body["duration"] == 0.1


# ─────────────────────────────────────────────────────────────
# Synthesis resolution
# ─────────────────────────────────────────────────────────────

def test_every_catalogued_voice_resolves_to_a_synthesis_call(fake_engine_service):
    voices = fake_engine_service.get_voices()
    assert len(voices) == 54

    for voice in voices:
        wav_bytes, duration = fake_engine_service.synthesize(
            "Hello.", voice=voice["id"], speed=1.0, phonetic_preprocess=False
        )
        assert wav_bytes.startswith(b"RIFF")
        assert duration > 0

    calls = fake_engine_service._kokoro.calls
    assert [call["voice"] for call in calls] == [voice["id"] for voice in voices]
    assert [call["lang"] for call in calls] == [voice["locale"] for voice in voices]


@pytest.mark.parametrize(("language", "sample"), LANGUAGE_SAMPLES.items())
def test_synthesis_accepts_one_voice_per_language_family(language, sample, fake_engine_service):
    voice_id, text = sample
    descriptor = fake_engine_service.get_voice(voice_id)

    assert descriptor is not None
    assert descriptor["language"] == language

    wav_bytes, duration = fake_engine_service.synthesize(
        text, voice=voice_id, phonetic_preprocess=False
    )
    assert wav_bytes.startswith(b"RIFF")
    assert duration > 0
    assert fake_engine_service._kokoro.calls[-1]["lang"] == descriptor["locale"]


def test_synthesis_rejects_voice_absent_from_engine():
    service = _service_with_engine(["af_heart"])

    with pytest.raises(ValueError, match="Unknown voice 'jf_alpha'"):
        service.synthesize("Hello.", voice="jf_alpha", phonetic_preprocess=False)

    assert service._kokoro.calls == []


def test_engine_keys_are_the_source_of_truth():
    """A pack with different voices is reflected as-is; no hard-coded allow-list."""
    service = _service_with_engine(["af_heart", "zm_yunyang"])

    assert service.voice_ids == ["af_heart", "zm_yunyang"]
    assert [voice["id"] for voice in service.get_voices()] == ["af_heart", "zm_yunyang"]
    assert service.get_voice("jf_alpha") is None

    service.synthesize("Hello.", voice="zm_yunyang", phonetic_preprocess=False)
    assert service._kokoro.calls[-1]["lang"] == "cmn"


# ─────────────────────────────────────────────────────────────
# Slow tests — require the real model files
# ─────────────────────────────────────────────────────────────

@pytest.mark.slow
@requires_models
def test_shipped_pack_matches_catalogue_manifest():
    pack_ids = set(np.load(VOICES_PATH).keys())

    assert len(pack_ids) == 54
    assert pack_ids == set(SHIPPED_VOICE_IDS)


@pytest.mark.slow
@requires_models
async def test_real_engine_exposes_whole_pack_and_synthesizes_every_language(caplog):
    service = TTSService()
    await service.initialize()

    assert service.is_ready
    assert len(service.voice_ids) == 54
    assert len({voice["language"] for voice in service.get_voices()}) == 9

    with caplog.at_level(logging.WARNING):
        for language, (voice_id, text) in LANGUAGE_SAMPLES.items():
            wav_bytes, duration = service.synthesize(text, voice=voice_id, speed=1.0)
            assert wav_bytes.startswith(b"RIFF"), language
            assert duration > 0, language
            assert service.get_voice(voice_id)["language"] == language

    assert not [record for record in caplog.records if "does not follow" in record.message]


# ─────────────────────────────────────────────────────────────
# F4: the cache key records the engine voice id, not the locale
# ─────────────────────────────────────────────────────────────
#
# ``KokoroEngine.resolve_voice`` returns ``(engine_key, locale)`` and only the
# engine key enters the cache key.  That is safe only while one engine never maps
# two registered voices onto the same ``engine_voice_id`` — otherwise two voices
# that differ only by locale would share a cache entry.  The invariant holds
# today (0 duplicates); this test is what keeps it from being broken silently.

def test_no_engine_maps_two_voices_onto_the_same_engine_voice_id():
    records = voice_registry.VOICE_REGISTRY.voices
    assert records, "the registry is empty"

    seen: dict[tuple[str, str], str] = {}
    collisions: list[str] = []
    for voice_id, record in records.items():
        key = (record.engine, record.engine_voice_id)
        if key in seen:
            collisions.append(
                f"{seen[key]!r} and {voice_id!r} both map to {record.engine_voice_id!r} "
                f"on engine {record.engine!r}"
            )
        seen[key] = voice_id

    assert collisions == [], (
        "two voices share an engine_voice_id, so the cache key (which excludes "
        "the locale) cannot distinguish them: " + "; ".join(collisions)
    )
