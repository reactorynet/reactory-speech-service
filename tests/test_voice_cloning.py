"""SPEECH-006 — consent-first voice cloning (ADR-0003).

The cloning itself is one library call; almost every test here is about the
governance around it: approver authority, transient reference audio, signed
consent records, request-time expiry, takedown, and signed provenance manifests.

No test needs torch or the 3.1 GB Chatterbox weights.  A deterministic clone
backend stands in for the model; the adapter's watermark neutralisation is still
exercised through the real ``ChatterboxEngine`` code path.
"""

from __future__ import annotations

import asyncio
import io
import json
import threading
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from app.config import settings
from app.core.contracts import EngineCapabilities
from app.engines.base import EngineUnavailableError
from app.engines.manager import EngineManager
from app.services import clone_crypto, provenance, voice_cloning
from app.services.clone_store import (
    MAX_REFERENCE_RETENTION_DAYS,
    REQUIRED_CONSENT_FIELDS,
    SCOPES,
    CloneStore,
    format_timestamp,
    parse_timestamp,
)
from app.services.tts_service import TTSService
from app.services.voice_cloning import (
    ApproverNotAllowedError,
    CloneInactiveError,
    CloneValidationError,
    CloningUnavailableError,
    VoiceCloneService,
    build_identity,
    reset_clone_service,
)

T0 = 1_800_000_000.0  # 2027-01-15T08:00:00Z — an arbitrary fixed clock

CLONE_TEXT = "Hi Vin, this is a private sentence used only inside the test."


# ---------------------------------------------------------------------------
# Deterministic doubles
# ---------------------------------------------------------------------------


class _SentinelWatermarker:
    """Prepends one full-scale sample, so \"was it neutralised?\" is assertable."""

    def apply_watermark(self, wav, *args, **kwargs):
        return np.concatenate([np.array([-1.0], dtype=wav.dtype), wav])


class _FakeModel:
    sr = 24000

    def __init__(self) -> None:
        self.watermarker = _SentinelWatermarker()
        self.calls: list[dict] = []

    def generate(self, text, exaggeration=None, audio_prompt_path=None):
        self.calls.append({"text": text, "exaggeration": exaggeration})
        n = max(1, int(self.sr * 0.05))
        wav = np.full(n, 0.1, dtype=np.float32)
        if self.watermarker is not None:
            wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)
        return wav

    def prepare_conditionals(self, path, exaggeration=0.5):  # pragma: no cover - unused
        self.conds = object()


class FakeCloneBackend:
    """Extraction is a deterministic function of the clip; rendering is a tone."""

    def __init__(self) -> None:
        self.extract_calls = 0
        self.render_calls = 0
        self.started = threading.Event()
        self.release = threading.Event()
        self.block_render = False

    def extract_embedding(self, audio: bytes) -> bytes:
        self.extract_calls += 1
        return b"embedding:" + clone_crypto.sha256_hex(audio)[:16].encode("ascii")

    def render(self, text: str, embedding: bytes, *, exaggeration: float):
        self.render_calls += 1
        if self.block_render:
            self.started.set()
            self.release.wait(timeout=10)
        return np.full(1200, 0.1, dtype=np.float32), 24000


class FakeKokoro:
    voices = {"af_heart": None}

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, text, voice, speed=1.0, lang="en-us", is_phonemes=False):
        self.calls.append({"voice": voice})
        return np.zeros(1200, dtype=np.float32), 24000


def _wav_bytes(samples: np.ndarray | None = None, *, sample_rate: int = 24000) -> bytes:
    data = np.zeros(2400, dtype=np.float32) if samples is None else samples
    buffer = io.BytesIO()
    sf.write(buffer, data, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def _consent(**overrides) -> dict:
    payload = {
        "subjectName": "Vin",
        "purpose": "Internal narration demo",
        "capturedAt": format_timestamp(T0 - 3600),
        "expiresAt": format_timestamp(T0 + 365 * 86400),
        "approver": "approver@example.com",
        "evidence": "consent-form-2026-11-10.pdf",
        "scope": "internal-only",
    }
    payload.update(overrides)
    return payload


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


def _make_service(
    *,
    backend: FakeCloneBackend | None = None,
    clock=T0,
    approvers_path: str | Path | None = None,
) -> VoiceCloneService:
    """Build a clone service against the isolated keys/store from conftest."""
    identity = build_identity(settings)
    assert identity is not None, "conftest must provision an instance key"
    reference_key = clone_crypto.load_reference_key(settings.clone_reference_key_path)
    now = clock if callable(clock) else (lambda: float(clock))
    store = CloneStore(settings.clone_store_dir, reference_key=reference_key, clock=now)
    service = VoiceCloneService(
        store=store,
        identity=identity,
        reference_key=reference_key,
        approvers_path=approvers_path or settings.clone_approvers_path,
        backend_resolver=lambda: backend,
        clock=now,
    )
    reset_clone_service(service)
    return service


def _create(service: VoiceCloneService, backend: FakeCloneBackend, **kwargs):
    return service.create(
        _wav_bytes(),
        _consent(**kwargs.pop("consent", {})),
        created_by=kwargs.pop("created_by", "creator@example.com"),
        **kwargs,
    )


def _tts_service_with_chatterbox(render_backend=None) -> TTSService:
    """A real TTSService with a loadable Chatterbox double and a clone renderer."""
    service = TTSService(expressive=True)
    service._kokoro = FakeKokoro()
    service._ready = True
    engine = service._chatterbox_engine
    assert engine is not None
    engine.model = _FakeModel()
    engine.set_ready(True)
    renderer = render_backend or FakeCloneBackend()
    engine.clone_backend = lambda: renderer  # type: ignore[method-assign]
    return service


def _client(tts_service: TTSService) -> TestClient:
    from app.main import app

    app.state.tts_service = tts_service
    return TestClient(app)


# ---------------------------------------------------------------------------
# 1. Approver authority is configuration, and it fails closed
# ---------------------------------------------------------------------------


def test_an_approver_in_the_deployed_config_is_accepted():
    backend = FakeCloneBackend()
    service = _make_service(backend=backend)

    descriptor = _create(service, backend)

    assert descriptor["clone"] is True
    assert descriptor["approved_by"] == "approver@example.com"


def test_an_approver_not_in_the_deployed_config_is_refused():
    backend = FakeCloneBackend()
    service = _make_service(backend=backend)

    with pytest.raises(ApproverNotAllowedError):
        _create(service, backend, consent={"approver": "stranger@example.com"})

    assert service.list_clones(include_inactive=True) == []


def test_an_absent_approver_config_makes_cloning_unavailable(tmp_path):
    backend = FakeCloneBackend()
    service = _make_service(
        backend=backend, approvers_path=tmp_path / "does-not-exist.json"
    )

    assert service.availability().available is False
    with pytest.raises(CloningUnavailableError, match="unavailable"):
        _create(service, backend)


def test_an_empty_approver_config_makes_cloning_unavailable_not_unrestricted(tmp_path):
    backend = FakeCloneBackend()
    empty = tmp_path / "empty-approvers.json"
    empty.write_text(json.dumps({"clone_approvers": []}), encoding="utf-8")
    service = _make_service(backend=backend, approvers_path=empty)

    with pytest.raises(CloningUnavailableError):
        _create(service, backend)


def test_self_approval_works_only_through_the_same_allowlist():
    backend = FakeCloneBackend()
    service = _make_service(backend=backend)

    # The creator is the approver and is in the list: permitted.
    granted = _create(
        service,
        backend,
        created_by="approver@example.com",
        consent={"approver": "approver@example.com"},
    )
    assert granted["approved_by"] == "approver@example.com"

    # The same self-approval is refused when the address is not configured.
    with pytest.raises(ApproverNotAllowedError):
        _create(
            service,
            backend,
            created_by="someone@example.com",
            consent={"approver": "someone@example.com"},
        )


def test_the_allowlist_is_read_from_configuration_not_baked_into_code(tmp_path):
    backend = FakeCloneBackend()
    path = tmp_path / "approvers.json"
    path.write_text(json.dumps({"clone_approvers": ["first@example.com"]}), encoding="utf-8")
    service = _make_service(backend=backend, approvers_path=path)

    _create(service, backend, created_by="first@example.com", consent={"approver": "first@example.com"})
    with pytest.raises(ApproverNotAllowedError):
        _create(service, backend, consent={"approver": "second@example.com"})

    # Deploying a new list takes effect without a restart: it is configuration.
    path.write_text(
        json.dumps({"clone_approvers": ["first@example.com", "second@example.com"]}),
        encoding="utf-8",
    )
    _create(service, backend, consent={"approver": "second@example.com"})


# ---------------------------------------------------------------------------
# 2. Incomplete consent: 422 naming every missing field, one test per field
# ---------------------------------------------------------------------------


def _post_clone(client: TestClient, consent: dict, *, filename: str = "reference.wav"):
    return client.post(
        "/api/tts/voices/clone",
        files={"reference_audio": (filename, _wav_bytes(), "audio/wav")},
        data={"consent": json.dumps(consent), "created_by": "creator@example.com"},
    )


def _http_harness():
    tts = _tts_service_with_chatterbox()
    backend = FakeCloneBackend()
    _make_service(backend=backend)
    return _client(tts), backend


@pytest.mark.parametrize("field", REQUIRED_CONSENT_FIELDS)
def test_a_missing_consent_field_returns_422_naming_that_field(field):
    client, _backend = _http_harness()
    consent = _consent()
    consent.pop(field)

    response = _post_clone(client, consent)

    assert response.status_code == 422, response.text
    detail = response.json()["detail"]
    assert isinstance(detail, str)
    assert field in detail
    # Every other required field was supplied, so the message names only this one.
    for other in REQUIRED_CONSENT_FIELDS:
        if other != field:
            assert other not in detail


def test_a_consent_record_reporting_every_missing_field_at_once():
    client, _backend = _http_harness()

    response = _post_clone(client, {})

    assert response.status_code == 422
    detail = response.json()["detail"]
    for field in REQUIRED_CONSENT_FIELDS:
        assert field in detail


def test_a_scope_outside_the_enumeration_is_refused():
    client, _backend = _http_harness()

    response = _post_clone(client, _consent(scope="whatever-we-like"))

    assert response.status_code == 422
    assert "scope" in response.json()["detail"]
    assert all(scope in response.json()["detail"] for scope in SCOPES)


def test_an_expired_consent_is_refused_at_creation():
    client, _backend = _http_harness()

    response = _post_clone(client, _consent(expiresAt=format_timestamp(T0 - 60)))

    assert response.status_code == 422
    assert "expiresAt" in response.json()["detail"]


def test_capture_cannot_be_later_than_expiry():
    client, _backend = _http_harness()

    response = _post_clone(
        client,
        _consent(
            capturedAt=format_timestamp(T0 + 7200),
            expiresAt=format_timestamp(T0 + 3600),
        ),
    )

    assert response.status_code == 422
    assert "capturedAt" in response.json()["detail"]


def test_an_undecodable_reference_clip_is_422_not_500():
    client, _backend = _http_harness()

    response = client.post(
        "/api/tts/voices/clone",
        files={"reference_audio": ("ref.wav", b"not audio at all", "audio/wav")},
        data={"consent": json.dumps(_consent())},
    )

    assert response.status_code == 422
    assert "decode" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# 3. Reference audio is transient and encrypted at rest
# ---------------------------------------------------------------------------


def test_reference_audio_is_absent_from_disk_after_a_successful_clone():
    backend = FakeCloneBackend()
    service = _make_service(backend=backend)

    descriptor = _create(service, backend)

    reference = service.store.reference_path(descriptor["id"])
    assert not reference.exists(), "the reference clip survived the default deletion"
    assert service.store.embedding_path(descriptor["id"]).exists()


def test_a_retained_reference_is_encrypted_at_rest(tmp_path):
    backend = FakeCloneBackend()
    service = _make_service(backend=backend)

    descriptor = _create(
        service,
        backend,
        retain_reference=True,
        retention_reason="manual review of a disputed render",
        retention_days=7,
    )

    path = service.store.reference_path(descriptor["id"])
    assert path.exists()
    blob = path.read_bytes()
    assert b"RIFF" not in blob, "reference audio was stored in the clear"
    assert clone_crypto.decrypt(
        service.store._reference_key, blob, aad=descriptor["id"].encode("utf-8")
    ) == _wav_bytes()


def test_retention_requires_a_reason():
    backend = FakeCloneBackend()
    service = _make_service(backend=backend)

    with pytest.raises(CloneValidationError, match="retentionReason"):
        _create(service, backend, retain_reference=True, retention_days=7)


def test_retention_is_capped_at_ninety_days_and_is_visible_on_the_consent():
    backend = FakeCloneBackend()
    service = _make_service(backend=backend)

    descriptor = _create(
        service,
        backend,
        retain_reference=True,
        retention_reason="re-derivation debugging",
        retention_days=365,
    )

    retention = service.consent_record(descriptor["id"])["record"]["referenceRetention"]
    assert retention["retained"] is True
    assert retention["reason"] == "re-derivation debugging"
    assert retention["capped"] is True
    delete_at = parse_timestamp(retention["deleteAt"])
    assert delete_at == pytest.approx(T0 + MAX_REFERENCE_RETENTION_DAYS * 86400)


def test_the_ninety_day_boundary_purges_a_retained_reference_without_sleeping():
    now = {"t": T0}
    backend = FakeCloneBackend()
    service = _make_service(backend=backend, clock=lambda: now["t"])
    descriptor = _create(
        service,
        backend,
        retain_reference=True,
        retention_reason="review window",
        retention_days=90,
    )
    reference = service.store.reference_path(descriptor["id"])
    assert reference.exists()

    now["t"] = T0 + 89 * 86400
    service.list_clones()
    assert reference.exists(), "the clip was deleted before the ceiling"

    now["t"] = T0 + 90 * 86400 + 1
    service.list_clones()
    assert not reference.exists(), "the 90-day ceiling did not delete the clip"
    # The signed consent record is immutable (editing it would invalidate the
    # signature); the purge is recorded on the registry entry and in the audit.
    entry = service.store.get(descriptor["id"])
    assert entry["reference"]["retained"] is False
    assert entry["reference"]["purgedAt"]
    assert any(
        item["action"] == "reference-expired" and item["voiceId"] == descriptor["id"]
        for item in service.audit_entries()
    )


def test_deletion_succeeds_when_the_reference_file_is_already_missing():
    backend = FakeCloneBackend()
    service = _make_service(backend=backend)
    descriptor = _create(
        service,
        backend,
        retain_reference=True,
        retention_reason="review window",
        retention_days=30,
    )
    service.store.reference_path(descriptor["id"]).unlink()

    result = service.delete(
        descriptor["id"], requested_by="gateway:takedown", reason="subject request"
    )

    assert result["status"] == "deleted"
    assert service.store.get(descriptor["id"]) is None
    assert not service.store.consent_path(descriptor["id"]).parent.joinpath(
        "embedding.enc"
    ).exists()


# ---------------------------------------------------------------------------
# 4. The consent record is instance-signed and tamper-evident
# ---------------------------------------------------------------------------


def test_the_consent_record_is_stored_beside_the_voice_and_verifies():
    backend = FakeCloneBackend()
    service = _make_service(backend=backend)
    descriptor = _create(service, backend)

    consent_path = service.store.consent_path(descriptor["id"])
    assert consent_path.parent == service.store.voice_dir(descriptor["id"])
    result = service.consent_record(descriptor["id"])
    assert result["signatureValid"] is True
    assert result["verifiedAgainstInstanceId"] == "test-instance"
    record = result["record"]
    for field in (*REQUIRED_CONSENT_FIELDS, "signature", "consentId"):
        assert record.get(field)


def test_tampering_with_a_signed_consent_record_invalidates_the_signature():
    backend = FakeCloneBackend()
    service = _make_service(backend=backend)
    descriptor = _create(service, backend)

    path = service.store.consent_path(descriptor["id"])
    record = json.loads(path.read_text(encoding="utf-8"))
    record["purpose"] = "A purpose that was never approved"
    path.write_text(json.dumps(record), encoding="utf-8")

    assert service.consent_record(descriptor["id"])["signatureValid"] is False


def test_the_consent_record_does_not_claim_that_the_subject_consented():
    backend = FakeCloneBackend()
    service = _make_service(backend=backend)
    descriptor = _create(service, backend)

    record = service.consent_record(descriptor["id"])["record"]
    assert "does NOT prove" in record["notice"]
    assert "does not prove" in record["notice"].lower()
    assert "subject consented" in record["notice"].lower()


# ---------------------------------------------------------------------------
# 5. Expiry and takedown are request-time, without a restart
# ---------------------------------------------------------------------------


async def test_expired_consent_makes_the_clone_inert_at_request_time():
    now = {"t": T0}
    backend = FakeCloneBackend()
    clone_service = _make_service(backend=backend, clock=lambda: now["t"])
    tts = _tts_service_with_chatterbox()
    descriptor = _create(
        clone_service,
        backend,
        consent={"expiresAt": format_timestamp(T0 + 86400)},
    )

    first = await tts.synthesize_async("Hello.", voice=descriptor["id"])
    assert first.engine == "chatterbox"

    now["t"] = T0 + 86400 + 1  # consent expires; no restart, no background job
    with pytest.raises(CloneInactiveError, match="expired"):
        await tts.synthesize_async("Hello.", voice=descriptor["id"])


def test_takedown_removes_the_voice_from_the_catalogue_immediately():
    tts = _tts_service_with_chatterbox()
    backend = FakeCloneBackend()
    clone_service = _make_service(backend=backend)
    descriptor = _create(clone_service, backend)

    assert descriptor["id"] in [voice["id"] for voice in tts.get_voices()]

    clone_service.takedown(
        descriptor["id"], requested_by="gateway:takedown", reason="subject request"
    )

    assert descriptor["id"] not in [voice["id"] for voice in tts.get_voices()]
    assert clone_service.is_active(descriptor["id"]) is False


def test_a_disabled_clone_returns_410_and_is_absent_from_voices():
    client, backend = _http_harness()
    descriptor = _create(voice_cloning.get_clone_service(), backend)
    voice_id = descriptor["id"]
    assert voice_id in [voice["id"] for voice in client.get("/api/tts/voices").json()["voices"]]

    takedown = client.post(
        f"/api/tts/voices/{voice_id}/takedown",
        json={"requestedBy": "gateway:takedown", "reason": "subject request"},
    )
    assert takedown.status_code == 200, takedown.text

    assert voice_id not in [voice["id"] for voice in client.get("/api/tts/voices").json()["voices"]]
    assert client.get(f"/api/tts/voices/{voice_id}").status_code == 410
    assert client.post("/api/tts/synthesize/json", json={"text": "Hi", "voice": voice_id}).status_code == 410


async def test_an_in_flight_request_fails_closed_after_takedown():
    backend = FakeCloneBackend()
    clone_service = _make_service(backend=backend)
    descriptor = _create(clone_service, backend)

    tts = _tts_service_with_chatterbox()
    rendering = FakeCloneBackend()
    rendering.block_render = True
    tts._chatterbox_engine.clone_backend = lambda: rendering  # type: ignore[method-assign]

    task = asyncio.create_task(tts.synthesize_async("Hello.", voice=descriptor["id"]))
    assert await asyncio.to_thread(rendering.started.wait, 5)

    clone_service.takedown(
        descriptor["id"], requested_by="gateway:takedown", reason="urgent"
    )
    rendering.release.set()

    with pytest.raises(CloneInactiveError):
        # Bind the result so the await is unambiguously a statement with an
        # effect: we assert only that it raises.
        _ = await task


async def test_a_cached_clone_render_is_never_served_after_takedown():
    backend = FakeCloneBackend()
    clone_service = _make_service(backend=backend)
    tts = _tts_service_with_chatterbox()
    descriptor = _create(clone_service, backend)

    first = await tts.synthesize_async(CLONE_TEXT, voice=descriptor["id"])
    assert first.cache == "miss"
    second = await tts.synthesize_async(CLONE_TEXT, voice=descriptor["id"])
    assert second.cache == "hit"  # same text, same voice: the cache is warm

    clone_service.takedown(
        descriptor["id"], requested_by="gateway:takedown", reason="subject request"
    )

    with pytest.raises(CloneInactiveError):
        await tts.synthesize_async(CLONE_TEXT, voice=descriptor["id"])


async def test_a_clone_never_falls_back_to_another_voice_when_the_engine_is_unavailable():
    backend = FakeCloneBackend()
    clone_service = _make_service(backend=backend)
    tts = _tts_service_with_chatterbox()
    tts._chatterbox_engine.set_ready(False)
    tts._chatterbox_engine.model = None
    descriptor = _create(clone_service, backend)

    with pytest.raises((CloningUnavailableError, EngineUnavailableError)):
        await tts.synthesize_async("Hello.", voice=descriptor["id"])
    assert tts._kokoro.calls == [], "a clone was rendered in a different voice"


# ---------------------------------------------------------------------------
# 6. The ordinary synthesis path, and no watermark in production audio
# ---------------------------------------------------------------------------


def test_a_cloned_voice_synthesises_through_the_ordinary_endpoint():
    client, backend = _http_harness()
    descriptor = _create(voice_cloning.get_clone_service(), backend)

    response = client.post(
        "/api/tts/synthesize/json",
        json={"text": CLONE_TEXT, "voice": descriptor["id"]},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["engine"] == "chatterbox"
    assert body["provenance"]["voiceId"] == descriptor["id"]
    assert body["provenance"]["audioSha256"]
    assert response.headers["x-provenance-id"] == body["provenance"]["audioSha256"]


def test_production_clone_output_carries_no_watermark():
    backend = FakeCloneBackend()
    clone_service = _make_service(backend=backend)
    tts = _tts_service_with_chatterbox()
    descriptor = _create(clone_service, backend)

    outcome = asyncio.run(tts.synthesize_async("Hello there.", voice=descriptor["id"]))
    samples, _rate = sf.read(io.BytesIO(outcome.audio), dtype="float32")

    # The model's sentinel watermarker is replaced before any render is returned.
    assert type(tts._chatterbox_engine.model.watermarker).__name__ == "_NoOpWatermarker"
    assert not np.any(samples == pytest.approx(-1.0))
    assert samples[0] == pytest.approx(0.1, abs=1e-4)


async def test_repeated_synthesis_does_not_re_extract_or_re_decrypt_the_embedding():
    backend = FakeCloneBackend()
    clone_service = _make_service(backend=backend)
    tts = _tts_service_with_chatterbox(backend)
    descriptor = _create(clone_service, backend)

    # A fresh store has an empty in-memory embedding cache.
    fresh = CloneStore(settings.clone_store_dir, reference_key=clone_service._reference_key)
    assert fresh.embedding_reads == 0
    fresh.load_embedding(descriptor["id"])
    fresh.load_embedding(descriptor["id"])
    assert fresh.embedding_reads == 1

    await tts.synthesize_async(CLONE_TEXT, voice=descriptor["id"], cache=False)
    await tts.synthesize_async(CLONE_TEXT, voice=descriptor["id"], cache=False)
    assert backend.extract_calls == 1, "the embedding was re-extracted from a clip"
    assert backend.render_calls == 2


# ---------------------------------------------------------------------------
# 7. Provenance: signed manifest, registry lookup, honest negative
# ---------------------------------------------------------------------------


def _rendered_manifest(service: VoiceCloneService, descriptor: dict) -> tuple[bytes, dict]:
    audio = _wav_bytes(np.full(2400, 0.1, dtype=np.float32))
    manifest = service.record_synthesis(
        descriptor["id"], audio, text_length=len(CLONE_TEXT), engine="chatterbox"
    )
    return audio, manifest


def test_a_signed_manifest_verifies_over_audio_and_its_signed_fields():
    backend = FakeCloneBackend()
    service = _make_service(backend=backend)
    descriptor = _create(service, backend)
    _audio, manifest = _rendered_manifest(service, descriptor)

    assert provenance.verify_manifest(manifest, service.identity.public_key) is True

    for field in ("audioSha256", "voiceId", "consentId", "instanceId", "createdAt"):
        tampered = dict(manifest)
        tampered[field] = f"tampered-{field}"
        assert (
            provenance.verify_manifest(tampered, service.identity.public_key) is False
        ), field


def test_different_audio_does_not_match_the_manifest():
    backend = FakeCloneBackend()
    service = _make_service(backend=backend)
    descriptor = _create(service, backend)
    _audio, manifest = _rendered_manifest(service, descriptor)

    other = _wav_bytes(np.full(2400, 0.2, dtype=np.float32))
    result = service.lookup_provenance(other)

    assert result["status"] == provenance.NO_RECORD_STATUS
    assert result["audioSha256"] != manifest["audioSha256"]


def test_tampering_with_a_stored_manifest_is_reported_as_an_invalid_signature():
    backend = FakeCloneBackend()
    service = _make_service(backend=backend)
    descriptor = _create(service, backend)
    _audio, manifest = _rendered_manifest(service, descriptor)
    digest = manifest["audioSha256"]

    payload = json.loads(service.store._manifests_path.read_text(encoding="utf-8"))
    payload["manifests"][digest]["voiceId"] = "clone_someone_else"
    service.store._manifests_path.write_text(json.dumps(payload), encoding="utf-8")
    service.store.load()

    result = service.lookup_by_digest(digest)
    assert result["status"] == provenance.INVALID_SIGNATURE_STATUS


def _rewrap_with_metadata_chunk(wav: bytes) -> bytes:
    """Insert a RIFF LIST/INFO chunk without touching a single audio sample."""
    data = bytearray(wav)
    position = 12
    insert_at = len(data)
    while position + 8 <= len(data):
        chunk_id = bytes(data[position : position + 4])
        size = int.from_bytes(data[position + 4 : position + 8], "little")
        if chunk_id == b"data":
            insert_at = position
            break
        position += 8 + size + (size & 1)
    info = b"INFOICMT" + (4).to_bytes(4, "little") + b"copy"
    chunk = b"LIST" + len(info).to_bytes(4, "little") + info
    result = data[:insert_at] + chunk + data[insert_at:]
    result[4:8] = (len(result) - 8).to_bytes(4, "little")
    return bytes(result)


def test_audio_sha256_is_over_decoded_pcm_so_a_container_change_still_resolves():
    original = _wav_bytes(np.full(2400, 0.1, dtype=np.float32))
    rewrapped = _rewrap_with_metadata_chunk(original)

    assert rewrapped != original
    assert provenance.audio_sha256(rewrapped) == provenance.audio_sha256(original)

    other = _wav_bytes(np.full(2400, 0.2, dtype=np.float32))
    assert provenance.audio_sha256(other) != provenance.audio_sha256(original)


def test_an_unknown_file_reads_no_record_found_and_never_not_ours():
    service = _make_service(backend=FakeCloneBackend())

    result = service.lookup_provenance(_wav_bytes(np.full(2400, 0.3, dtype=np.float32)))

    assert result["status"] == provenance.NO_RECORD_STATUS
    assert result["manifest"] is None
    serialised = json.dumps(result).lower()
    assert "not ours" not in serialised
    assert "foreign" not in serialised


def test_the_provenance_endpoints_report_a_miss_with_the_safe_wording():
    client, backend = _http_harness()
    descriptor = _create(voice_cloning.get_clone_service(), backend)

    rendered = client.post(
        "/api/tts/synthesize/json", json={"text": CLONE_TEXT, "voice": descriptor["id"]}
    ).json()
    digest = rendered["provenance"]["audioSha256"]

    found = client.get(f"/api/tts/provenance/{digest}")
    assert found.status_code == 200
    assert found.json()["status"] == provenance.VERIFIED_STATUS

    missing = client.get("/api/tts/provenance/" + "0" * 64)
    assert missing.status_code == 404
    assert missing.json()["status"] == "no record found"
    assert "not ours" not in missing.text.lower()

    unknown_audio = client.post(
        "/api/tts/provenance/verify",
        files={"file": ("x.wav", _wav_bytes(np.full(2400, 0.4, dtype=np.float32)), "audio/wav")},
    )
    assert unknown_audio.status_code == 404
    assert unknown_audio.json()["status"] == "no record found"


def test_the_manifest_registry_outlives_the_voice():
    backend = FakeCloneBackend()
    service = _make_service(backend=backend)
    descriptor = _create(service, backend)
    _audio, manifest = _rendered_manifest(service, descriptor)

    service.delete(descriptor["id"], requested_by="gateway", reason="subject request")

    assert service.store.get_manifest(manifest["audioSha256"]) is not None


# ---------------------------------------------------------------------------
# 8. Audit: create, use and takedown; a synthesis entry has the count, not text
# ---------------------------------------------------------------------------


async def test_audit_records_create_use_and_takedown_without_the_text():
    backend = FakeCloneBackend()
    service = _make_service(backend=backend)
    tts = _tts_service_with_chatterbox()
    descriptor = _create(service, backend)

    await tts.synthesize_async(CLONE_TEXT, voice=descriptor["id"])
    service.takedown(descriptor["id"], requested_by="gateway", reason="subject request")

    entries = service.audit_entries()
    actions = [entry["action"] for entry in entries]
    assert "create" in actions
    assert "use" in actions
    assert "takedown" in actions

    use = next(entry for entry in entries if entry["action"] == "use")
    assert use["charCount"] == len(CLONE_TEXT)
    raw = service.store._audit_path.read_text(encoding="utf-8")
    assert CLONE_TEXT not in raw, "the synthesis text leaked into the audit log"


def test_a_deletion_audit_entry_carries_no_audio_or_matching_hash():
    backend = FakeCloneBackend()
    service = _make_service(backend=backend)
    descriptor = _create(service, backend)
    audio = _wav_bytes()

    service.delete(descriptor["id"], requested_by="gateway", reason="subject request")

    entry = next(e for e in service.audit_entries() if e["action"] == "delete")
    assert entry["requestedBy"] == "gateway"
    assert entry["reason"] == "subject request"
    serialised = json.dumps(entry).lower()
    assert "audiosha256" not in serialised
    assert clone_crypto.sha256_hex(audio) not in serialised


# ---------------------------------------------------------------------------
# 9. Startup tolerance and the embedding-only clone
# ---------------------------------------------------------------------------


def _key_script():
    import importlib.util

    path = Path(__file__).resolve().parent.parent / "scripts" / "generate_clone_keys.py"
    spec = importlib.util.spec_from_file_location("generate_clone_keys_script", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_the_operator_script_generates_separate_keys_and_a_fail_closed_allowlist(tmp_path):
    script = _key_script()
    output = tmp_path / "operator"

    assert script.main(["--output-dir", str(output)]) == 0

    signing = output / "keys" / "clone_signing_key.pem"
    reference = output / "keys" / "clone_reference.key"
    approvers = output / "clone_approvers.json"
    assert signing.exists() and reference.exists() and approvers.exists()
    assert signing.read_bytes() != reference.read_bytes()
    assert len(reference.read_bytes()) == 32
    assert json.loads(approvers.read_text(encoding="utf-8"))["clone_approvers"] == []
    # Running again must refuse: a new signing key would invalidate every signature.
    assert script.main(["--output-dir", str(output)]) == 1
    assert script.main(["--output-dir", str(output), "--force"]) == 0


def test_health_reports_cloning_as_unavailable_without_an_allowlist(tmp_path, monkeypatch):
    from app.main import app

    monkeypatch.setattr(settings, "clone_approvers_path", str(tmp_path / "absent.json"))
    reset_clone_service()

    payload = TestClient(app).get("/health").json()
    assert payload["cloning"]["available"] is False
    assert "approver" in payload["cloning"]["reason"]


def test_a_corrupt_consent_store_degrades_to_no_clones_rather_than_failing_boot():
    service = _make_service(backend=FakeCloneBackend())
    service.store.registry_path.parent.mkdir(parents=True, exist_ok=True)
    service.store.registry_path.write_text("{not json", encoding="utf-8")

    voice_cloning.reset_clone_service()  # what startup does

    rebuilt = voice_cloning.get_clone_service()
    assert rebuilt.list_clones(include_inactive=True) == []
    with pytest.raises(CloneInactiveError):
        rebuilt.authorize_synthesis("clone_ghost_00000000")


def test_a_clone_is_unusable_without_the_instance_signing_key():
    """Provenance cannot be emitted unsigned, so the clone fails closed (503)."""
    backend = FakeCloneBackend()
    service = _make_service(backend=backend)
    descriptor = _create(service, backend)

    service.identity = None
    with pytest.raises(CloningUnavailableError, match="signing key"):
        service.authorize_synthesis(descriptor["id"])


def test_a_clone_id_with_no_record_fails_closed_instead_of_falling_back():
    service = _make_service(backend=FakeCloneBackend())

    with pytest.raises(CloneInactiveError, match="No clone record"):
        service.authorize_synthesis("clone_deleted_12345678")


def test_the_clone_engine_and_licence_are_permissive_and_no_prohibited_model_is_wired():
    """ADR-0001/ADR-0003: approved engines only; XTTS v2 and F5-TTS are prohibited."""
    from app.services import voice_registry

    backend = FakeCloneBackend()
    service = _make_service(backend=backend)
    descriptor = _create(service, backend)
    entry = service.store.get(descriptor["id"])

    assert entry["engine"] == "chatterbox"
    assert entry["license"] in voice_registry.PERMISSIVE_LICENSES
    adapter_source = (
        Path(__file__).resolve().parent.parent / "app" / "engines" / "chatterbox.py"
    ).read_text(encoding="utf-8").lower()
    # "chatterboxtts" contains the letters "xtts"; check the model names, not
    # an arbitrary substring.
    assert "xtts_v2" not in adapter_source
    assert "f5-tts" not in adapter_source
    assert "coqui" not in adapter_source


def test_the_clone_is_re_derivable_from_the_embedding_only():
    backend = FakeCloneBackend()
    service = _make_service(backend=backend)
    descriptor = _create(service, backend)

    # The reference is gone; the embedding is the only input a render receives.
    assert not service.store.reference_path(descriptor["id"]).exists()
    context = service.authorize_synthesis(descriptor["id"])
    assert context is not None
    assert context.embedding.startswith(b"embedding:")
    assert context.entry["referenceSamples"] > 0


# ---------------------------------------------------------------------------
# 10. Known defect in the same path: a degraded render must never be cached
# ---------------------------------------------------------------------------


class _UnavailableEngine:
    id = "unavailable"
    lazy_load = False
    is_ready = False

    def capabilities(self):
        return EngineCapabilities(engine=self.id, sample_rates=[24000])

    def voices(self):
        return []

    def synthesize(self, req):  # pragma: no cover - always unavailable
        raise EngineUnavailableError("weights are not mounted")

    def stream(self, req):  # pragma: no cover
        yield


class _WorkingEngine:
    id = "working"
    lazy_load = False
    is_ready = True

    def __init__(self) -> None:
        self.calls = 0

    def capabilities(self):
        return EngineCapabilities(engine=self.id, sample_rates=[24000], native_rate=True)

    def voices(self):
        return []

    def synthesize(self, req):
        from app.core.contracts import AudioResult

        self.calls += 1
        audio = _wav_bytes(np.full(2400, 0.05, dtype=np.float32))
        return AudioResult(
            audio=audio, duration=0.1, sample_rate=24000, engine=self.id
        )

    def stream(self, req):  # pragma: no cover
        yield


async def test_a_degraded_fallback_render_is_not_cacheable():
    unavailable = _UnavailableEngine()
    working = _WorkingEngine()
    tts = TTSService(
        manager=EngineManager([unavailable, working]),
        cache=None,
    )

    first = await tts.synthesize_async("Hello.", engine="unavailable")
    second = await tts.synthesize_async("Hello.", engine="unavailable")

    assert first.engine == second.engine == "working"
    assert first.cache == second.cache == "miss"
    assert tts.cache.stats.hits == 0, "a fallback render was served from cache"
    assert working.calls == 2

# ---------------------------------------------------------------------------
# 11. The consent record is an enforcement input, not only a readable artefact
#
# ADR-0003 #2 says a signed record proves which instance recorded the consent
# and that it is unaltered.  That is only a control if altering it has a
# consequence in the request path; reporting `signatureValid: false` on a read
# endpoint while still rendering the voice would make the signature decorative.
# ---------------------------------------------------------------------------


def test_a_tampered_consent_record_makes_the_clone_inert_not_just_unverified():
    backend = FakeCloneBackend()
    service = _make_service(backend=backend)
    tts = _tts_service_with_chatterbox(backend)
    client = _client(tts)
    descriptor = _create(service, backend)

    # The untampered clone renders through the ordinary endpoint.
    before = client.post(
        "/api/tts/synthesize/json", json={"text": "Hello.", "voice": descriptor["id"]}
    )
    assert before.status_code == 200, before.text

    # Extend the consent's expiry without re-signing it: the plausible tamper.
    path = service.store.consent_path(descriptor["id"])
    record = json.loads(path.read_text(encoding="utf-8"))
    record["expiresAt"] = format_timestamp(T0 + 10 * 365 * 86400)
    path.write_text(json.dumps(record), encoding="utf-8")

    assert service.consent_record(descriptor["id"])["signatureValid"] is False
    assert service.is_active(descriptor["id"]) is False
    with pytest.raises(CloneInactiveError, match="failed verification"):
        service.authorize_synthesis(descriptor["id"])

    assert client.get(f"/api/tts/voices/{descriptor['id']}").status_code == 410
    assert (
        client.post(
            "/api/tts/synthesize/json", json={"text": "Hello.", "voice": descriptor["id"]}
        ).status_code
        == 410
    )
    assert descriptor["id"] not in [
        voice["id"] for voice in client.get("/api/tts/voices").json()["voices"]
    ]


def test_a_missing_consent_record_makes_the_clone_inert():
    """No authorising evidence means no authorisation, not 'render anyway'."""
    backend = FakeCloneBackend()
    service = _make_service(backend=backend)
    descriptor = _create(service, backend)

    service.store.consent_path(descriptor["id"]).unlink()

    assert service.is_active(descriptor["id"]) is False
    with pytest.raises(CloneInactiveError, match="missing"):
        service.authorize_synthesis(descriptor["id"])


@pytest.mark.parametrize("scope", SCOPES)
def test_scope_is_recorded_and_does_not_gate_the_request_path(scope):
    """ADR-0003 #6: scope is recorded, not enforced here.

    Every enumerated scope is stored on the record and on the descriptor, and
    none of them changes whether the service will render the voice.  Enforcing
    a scope would require knowing who is asking, which this service cannot do.
    """
    backend = FakeCloneBackend()
    service = _make_service(backend=backend)

    descriptor = _create(service, backend, consent={"scope": scope})

    assert descriptor["scope"] == scope
    assert service.consent_record(descriptor["id"])["record"]["scope"] == scope
    assert service.is_active(descriptor["id"]) is True


# ---------------------------------------------------------------------------
# 12. HTTP mapping of the fail-closed configuration rules
# ---------------------------------------------------------------------------


def test_an_approver_not_in_the_config_is_refused_over_http():
    client, _backend = _http_harness()

    response = _post_clone(client, _consent(approver="stranger@example.com"))

    assert response.status_code == 403, response.text
    assert "clone_approvers" in response.json()["detail"]


def test_an_absent_allowlist_makes_cloning_503_not_unrestricted_over_http(tmp_path):
    backend = FakeCloneBackend()
    tts = _tts_service_with_chatterbox()
    _make_service(backend=backend, approvers_path=tmp_path / "absent.json")
    client = _client(tts)

    response = _post_clone(client, _consent())

    assert response.status_code == 503, response.text
    assert "unavailable" in response.json()["detail"].lower()


# ---------------------------------------------------------------------------
# 13. Expiry is request-time for the catalogue and the read endpoint too
# ---------------------------------------------------------------------------


def test_an_expired_clone_is_absent_from_voices_and_410_without_a_restart():
    now = {"t": T0}
    backend = FakeCloneBackend()
    service = _make_service(backend=backend, clock=lambda: now["t"])
    tts = _tts_service_with_chatterbox(backend)
    client = _client(tts)
    descriptor = _create(
        service, backend, consent={"expiresAt": format_timestamp(T0 + 86400)}
    )
    voice_id = descriptor["id"]

    assert voice_id in [voice["id"] for voice in client.get("/api/tts/voices").json()["voices"]]

    now["t"] = T0 + 86400 + 1  # the consent expires; the process is not restarted

    assert voice_id not in [
        voice["id"] for voice in client.get("/api/tts/voices").json()["voices"]
    ]
    assert client.get(f"/api/tts/voices/{voice_id}").status_code == 410
    assert (
        client.post("/api/tts/synthesize/json", json={"text": "Hi", "voice": voice_id}).status_code
        == 410
    )


def test_a_disabled_clone_returns_410_on_the_wav_endpoint():
    client, backend = _http_harness()
    descriptor = _create(voice_cloning.get_clone_service(), backend)
    voice_id = descriptor["id"]

    voice_cloning.get_clone_service().takedown(
        voice_id, requested_by="gateway:takedown", reason="subject request"
    )

    response = client.post("/api/tts/synthesize", json={"text": "Hi", "voice": voice_id})

    assert response.status_code == 410


# ---------------------------------------------------------------------------
# 14. Provenance completeness: every signed field, and one manifest per render
# ---------------------------------------------------------------------------


def test_every_signed_manifest_field_is_covered_by_the_signature():
    backend = FakeCloneBackend()
    service = _make_service(backend=backend)
    descriptor = _create(service, backend)
    _audio, manifest = _rendered_manifest(service, descriptor)

    assert provenance.verify_manifest(manifest, service.identity.public_key) is True

    # Type-appropriate substitutions, so the test fails for a real reason (the
    # payload changed), not because the field held an unexpected value.
    replacements = {
        "audioSha256": "0" * 64,
        "sampleRate": 48000,
        "channels": 2,
    }
    for field in provenance.SIGNED_MANIFEST_FIELDS:
        tampered = dict(manifest)
        tampered[field] = replacements.get(field, f"tampered-{field}")
        assert tampered[field] != manifest[field], field
        assert (
            provenance.verify_manifest(tampered, service.identity.public_key) is False
        ), field

    # Dropping the signature is a failure, not a pass.
    unsigned = {key: value for key, value in manifest.items() if key != "signature"}
    assert provenance.verify_manifest(unsigned, service.identity.public_key) is False


class _TextAwareBackend(FakeCloneBackend):
    """Returns a different tone per text, so distinct renders have distinct PCM."""

    def render(self, text: str, embedding: bytes, *, exaggeration: float):
        self.render_calls += 1
        level = 0.1 + (len(text) % 5) / 100
        return np.full(1200, level, dtype=np.float32), 24000


async def test_each_distinct_synthesis_emits_its_own_signed_manifest():
    backend = _TextAwareBackend()
    service = _make_service(backend=backend)
    tts = _tts_service_with_chatterbox(backend)
    descriptor = _create(service, backend)

    first = await tts.synthesize_async("First sentence.", voice=descriptor["id"], cache=False)
    second = await tts.synthesize_async(
        "A second, different sentence.", voice=descriptor["id"], cache=False
    )

    assert first.manifest is not None and second.manifest is not None
    assert first.manifest["audioSha256"] != second.manifest["audioSha256"]
    assert first.manifest["voiceId"] == second.manifest["voiceId"] == descriptor["id"]
    assert first.manifest["consentId"] == descriptor["consent_id"]
    assert provenance.verify_manifest(first.manifest, service.identity.public_key) is True
    assert provenance.verify_manifest(second.manifest, service.identity.public_key) is True
    assert service.store.manifest_count() == 2


# ---------------------------------------------------------------------------
# 15. Deletion removes every artefact and demands an auditable assertion
# ---------------------------------------------------------------------------


def test_delete_removes_a_clone_preview(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "designed_voices_preview_dir", str(tmp_path / "previews"))
    backend = FakeCloneBackend()
    service = _make_service(backend=backend)
    descriptor = _create(service, backend)

    preview_dir = tmp_path / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    preview = preview_dir / f"{descriptor['id']}.wav"
    preview.write_bytes(b"RIFF....not-really-a-preview")

    service.delete(descriptor["id"], requested_by="gateway", reason="subject request")

    assert not preview.exists(), "DELETE left a preview behind"


def test_deletion_requires_an_actor_and_a_reason():
    backend = FakeCloneBackend()
    service = _make_service(backend=backend)
    descriptor = _create(service, backend)

    with pytest.raises(CloneValidationError, match="requestedBy"):
        service.delete(descriptor["id"], requested_by="", reason="")
    with pytest.raises(CloneValidationError, match="reason"):
        service.delete(descriptor["id"], requested_by="gateway", reason="")

    # Refusals must not have deleted anything.
    assert service.store.get(descriptor["id"]) is not None


def test_reference_retention_is_purged_on_startup_as_well_as_on_request():
    """A clone nobody touches must still lose its clip at the ceiling.

    The ceiling is enforced without a background job: on request *and* when the
    service is constructed.  Neither mechanism can silently re-enable a voice,
    which is why expiry itself is only ever checked in the request path.
    """
    now = {"t": T0}
    backend = FakeCloneBackend()
    service = _make_service(backend=backend, clock=lambda: now["t"])
    descriptor = _create(
        service,
        backend,
        retain_reference=True,
        retention_reason="review window",
        retention_days=30,
    )
    reference = service.store.reference_path(descriptor["id"])
    assert reference.exists()

    now["t"] = T0 + 31 * 86400  # past the ceiling, no request has been made

    identity = build_identity(settings)
    reference_key = clone_crypto.load_reference_key(settings.clone_reference_key_path)
    rebuilt = VoiceCloneService(
        store=CloneStore(
            settings.clone_store_dir, reference_key=reference_key, clock=lambda: now["t"]
        ),
        identity=identity,
        reference_key=reference_key,
        approvers_path=settings.clone_approvers_path,
        backend_resolver=lambda: backend,
        clock=lambda: now["t"],
    )
    reset_clone_service(rebuilt)

    assert not reference.exists(), "startup did not purge an expired retained reference"
    assert rebuilt.store.get(descriptor["id"])["reference"]["retained"] is False
# ---------------------------------------------------------------------------
# 16. The operator contract: descriptor fields, gitignored secrets, no
#     implicit key generation
# ---------------------------------------------------------------------------


def test_the_voice_descriptor_surfaces_every_governance_fact():
    """ADR-0003: engine, licence, consent id, timestamps, approver, scope, expiry."""
    client, backend = _http_harness()
    descriptor = _create(voice_cloning.get_clone_service(), backend)

    body = client.get(f"/api/tts/voices/{descriptor['id']}").json()

    assert body["engine"] == "chatterbox"
    assert body["license"] == "MIT"
    assert body["consent_id"] == descriptor["consent_id"]
    assert body["created_at"] and body["created_by"] == "creator@example.com"
    assert body["approved_by"] == "approver@example.com"
    assert body["scope"] == "internal-only"
    assert body["expires_at"]
    assert body["clone"] is True and body["active"] is True


def test_the_deployment_secrets_and_runtime_store_are_gitignored():
    """A committed allowlist is public and stale; a committed key is a breach."""
    gitignore = (Path(__file__).resolve().parents[1] / ".gitignore").read_text(
        encoding="utf-8"
    )
    for entry in ("data/clone_approvers.json", "data/keys/", "data/clones/"):
        assert entry in gitignore, f"{entry} is not gitignored"


def test_an_absent_signing_key_is_never_generated_at_boot(tmp_path, monkeypatch):
    """A per-boot key would invalidate every existing signature."""
    missing = tmp_path / "absent" / "clone_signing_key.pem"
    monkeypatch.setattr(settings, "clone_signing_key_path", str(missing))

    assert build_identity(settings) is None
    assert not missing.exists(), "a signing key was generated implicitly"
    assert not missing.parent.exists()
# ---------------------------------------------------------------------------
# 17. The enforced expiry is the signed one (review finding B2)
#
# registry.json is not signed.  Taking the authorising expiry from it let anyone
# who could edit that file extend a clone years past the consent it was granted
# under, while the co-located signature still verified — the same fail-open hole
# the signature check closed, one layer down.
# ---------------------------------------------------------------------------


def _rewrite_registry_expiry(service, voice_id: str, expires_at: str) -> None:
    """Edit only the unsigned registry copy, leaving the signed record alone."""
    payload = json.loads(service.store.registry_path.read_text(encoding="utf-8"))
    for entry in payload["voices"]:
        if entry["id"] == voice_id:
            entry["expiresAt"] = expires_at
    service.store.registry_path.write_text(json.dumps(payload), encoding="utf-8")
    service.store.load()


def test_a_registry_only_expiry_edit_does_not_extend_a_clone():
    """The reviewer's repro: the signature still verifies, the clone must not."""
    now = {"t": T0}
    backend = FakeCloneBackend()
    service = _make_service(backend=backend, clock=lambda: now["t"])
    tts = _tts_service_with_chatterbox(backend)
    client = _client(tts)
    descriptor = _create(
        service, backend, consent={"expiresAt": format_timestamp(T0 + 86400)}
    )
    voice_id = descriptor["id"]

    # Past the signed consent, edit only registry.json to ten years out.
    now["t"] = T0 + 86400 + 1
    _rewrite_registry_expiry(service, voice_id, format_timestamp(T0 + 10 * 365 * 86400))

    # The signed artefact was not touched, so it still verifies — which is
    # exactly why the registry copy must never be the authority.
    assert service.consent_record(voice_id)["signatureValid"] is True
    assert service.is_active(voice_id) is False
    with pytest.raises(CloneInactiveError, match="does not match the signed consent"):
        service.authorize_synthesis(voice_id)

    assert client.get(f"/api/tts/voices/{voice_id}").status_code == 410
    assert voice_id not in [
        voice["id"] for voice in client.get("/api/tts/voices").json()["voices"]
    ]
    assert (
        client.post(
            "/api/tts/synthesize/json", json={"text": "Hi", "voice": voice_id}
        ).status_code
        == 410
    )


def test_a_registry_expiry_that_disagrees_with_the_signed_record_is_inert():
    """Disagreement fails closed even before the signed consent expires."""
    backend = FakeCloneBackend()
    service = _make_service(backend=backend)
    descriptor = _create(
        service, backend, consent={"expiresAt": format_timestamp(T0 + 365 * 86400)}
    )
    voice_id = descriptor["id"]
    assert service.is_active(voice_id) is True

    _rewrite_registry_expiry(service, voice_id, format_timestamp(T0 + 20 * 365 * 86400))

    assert service.is_active(voice_id) is False
    assert "does not match the signed consent" in (
        service.active_reason(voice_id) or ""
    )
    with pytest.raises(CloneInactiveError):
        service.authorize_synthesis(voice_id)


def test_the_signed_record_and_the_registry_entry_agree_at_creation():
    """Normalising both means a legitimate clone is never inert by accident."""
    backend = FakeCloneBackend()
    service = _make_service(backend=backend)
    # A caller-supplied rendering of the same instant must not look like tampering.
    descriptor = _create(
        service,
        backend,
        consent={"expiresAt": "2027-01-16T08:00:00Z", "capturedAt": "2027-01-15T07:00:00Z"},
    )

    record = service.consent_record(descriptor["id"])["record"]
    entry = service.store.get(descriptor["id"])
    assert parse_timestamp(record["expiresAt"]) == parse_timestamp(entry["expiresAt"])
    assert service.is_active(descriptor["id"]) is True
    assert descriptor["expires_at"] == record["expiresAt"]
# ---------------------------------------------------------------------------
# 18. Cloning NOT configured must not disable ordinary synthesis
#
# This is the profile every existing deployment runs: no signing key, no
# reference key, no approver allowlist (all three are gitignored and absent
# until an operator generates them). Cloning being unavailable must cost exactly
# cloning — not every named voice in the service.
#
# The autouse `isolated_voice_cloning` fixture provisions all three for every
# test, which is precisely why this configuration was unobservable before. The
# tests below clear the three settings explicitly instead of relying on it.
# ---------------------------------------------------------------------------


@pytest.fixture
def cloning_unconfigured(tmp_path, monkeypatch):
    """A deployment with cloning switched off, exactly as shipped."""
    absent = tmp_path / "not-configured"
    monkeypatch.setattr(settings, "clone_signing_key_path", str(absent / "signing.pem"))
    monkeypatch.setattr(settings, "clone_reference_key_path", str(absent / "reference.key"))
    monkeypatch.setattr(settings, "clone_approvers_path", str(absent / "approvers.json"))
    monkeypatch.setattr(settings, "clone_store_dir", str(absent / "clones"))
    voice_cloning.reset_clone_service()
    yield absent
    voice_cloning.reset_clone_service()


def _unconfigured_harness():
    tts = _tts_service_with_chatterbox()
    return tts, _client(tts)


def test_a_pack_voice_synthesises_when_cloning_is_not_configured(cloning_unconfigured):
    """The regression: an unconfigured deployment still serves every pack voice."""
    tts, client = _unconfigured_harness()

    for voice in ("af_heart", "chatterbox_standard"):
        response = client.post(
            "/api/tts/synthesize/json", json={"text": "Hello world.", "voice": voice}
        )
        assert response.status_code == 200, f"{voice}: {response.status_code} {response.text}"

    response = client.post(
        "/api/tts/synthesize", json={"text": "Hello world.", "voice": "chatterbox_standard"}
    )
    assert response.status_code == 200, response.text

    assert tts.get_voice("chatterbox_standard") is not None


def test_authorize_synthesis_ignores_a_plain_voice_when_no_key_is_deployed(
    cloning_unconfigured,
):
    """Service-layer repro of the reviewer's probe."""
    service = voice_cloning.get_clone_service()
    assert service.identity is None
    assert service.store._reference_key is None

    for plain in ("af_heart", "kokoro-af-heart", "chatterbox_standard"):
        assert service.authorize_synthesis(plain) is None, plain

    # A clone-shaped id still fails closed rather than falling back.
    with pytest.raises(CloneInactiveError):
        service.authorize_synthesis("clone_nope_00000000")


def test_clone_creation_is_503_and_health_says_unavailable_when_unconfigured(
    cloning_unconfigured,
):
    """Cloning is unavailable, loudly — and only cloning."""
    _tts, client = _unconfigured_harness()

    health = client.get("/health")
    assert health.status_code == 200
    payload = health.json()["cloning"]
    assert payload["available"] is False
    assert payload["signingKeyConfigured"] is False
    assert payload["referenceKeyConfigured"] is False
    assert "approver" in payload["reason"]

    # The read endpoint for a pack voice is normal too.
    assert client.get("/api/tts/voices/chatterbox_standard").status_code == 200

    # Clone creation refuses, naming the missing configuration.
    response = _post_clone(client, _consent())
    assert response.status_code == 503, response.text
    assert "unavailable" in response.json()["detail"].lower()


def test_health_and_voices_are_normal_while_cloning_is_disabled(cloning_unconfigured):
    """/health reports cloning unavailable; the catalogue is untouched."""
    _tts, client = _unconfigured_harness()

    voices = client.get("/api/tts/voices")
    assert voices.status_code == 200
    ids = [voice["id"] for voice in voices.json()["voices"]]
    assert "chatterbox_standard" in ids
    assert all(not voice_id.startswith("clone_") for voice_id in ids)

    health = client.get("/health").json()
    assert health["cloning"]["available"] is False


def test_a_deployed_clone_still_works_once_cloning_is_configured():
    """Guard against ordering bugs: configuring cloning later still works.

    Deliberately does NOT use the unconfigured fixture — it proves the configured
    path still works after the unconfigured tests have run.
    """
    backend = FakeCloneBackend()
    service = _make_service(backend=backend)
    descriptor = _create(service, backend)

    assert service.is_active(descriptor["id"]) is True
    context = service.authorize_synthesis(descriptor["id"])
    assert context is not None and context.embedding


# ---------------------------------------------------------------------------
# 19. The retention ceiling is the signed one (review finding 3)
# ---------------------------------------------------------------------------


def test_a_registry_only_retention_edit_cannot_extend_the_ceiling():
    """The 90-day ceiling is an invariant, so registry.json cannot move it."""
    now = {"t": T0}
    backend = FakeCloneBackend()
    service = _make_service(backend=backend, clock=lambda: now["t"])
    descriptor = _create(
        service,
        backend,
        retain_reference=True,
        retention_reason="review window",
        retention_days=90,
    )
    voice_id = descriptor["id"]
    reference = service.store.reference_path(voice_id)
    assert reference.exists()

    # Rewrite only the unsigned registry copy to a thousand days out.
    payload = json.loads(service.store.registry_path.read_text(encoding="utf-8"))
    for entry in payload["voices"]:
        if entry["id"] == voice_id:
            entry["reference"]["deleteAt"] = format_timestamp(T0 + 1000 * 86400)
    service.store.registry_path.write_text(json.dumps(payload), encoding="utf-8")
    service.store.load()

    now["t"] = T0 + 200 * 86400  # well past the signed 90-day ceiling

    purged = service._purge_expired_references()

    assert voice_id in purged, "the edited registry extended biometric retention"
    assert not reference.exists(), "the reference clip survived the ceiling"


def test_a_retention_deadline_beyond_the_ceiling_is_clamped_even_if_signed():
    """A correctly signed record cannot authorise more than 90 days."""
    now = {"t": T0}
    backend = FakeCloneBackend()
    service = _make_service(backend=backend, clock=lambda: now["t"])
    descriptor = _create(
        service,
        backend,
        retain_reference=True,
        retention_reason="review window",
        retention_days=30,
    )
    voice_id = descriptor["id"]
    four_years = format_timestamp(T0 + 4 * 365 * 86400)

    # Forge a *validly signed* record claiming a four-year window, as a
    # compromised writer with the instance key could.
    path = service.store.consent_path(voice_id)
    record = json.loads(path.read_text(encoding="utf-8"))
    record["referenceRetention"]["deleteAt"] = four_years
    payload = {key: value for key, value in record.items() if key != "signature"}
    record["signature"] = service.identity.sign_payload(payload)
    path.write_text(json.dumps(record), encoding="utf-8")

    # Make the registry agree too, so only the ceiling stands in the way.
    reg = json.loads(service.store.registry_path.read_text(encoding="utf-8"))
    for entry in reg["voices"]:
        if entry["id"] == voice_id:
            entry["reference"]["deleteAt"] = four_years
    service.store.registry_path.write_text(json.dumps(reg), encoding="utf-8")
    service.store.load()

    now["t"] = T0 + MAX_REFERENCE_RETENTION_DAYS * 86400 + 1

    assert voice_id in service._purge_expired_references()
    assert not service.store.reference_path(voice_id).exists()


def test_retention_is_purged_when_the_consent_record_cannot_justify_it():
    """Unjustifiable retention is deleted, not kept."""
    backend = FakeCloneBackend()
    service = _make_service(backend=backend)
    descriptor = _create(
        service,
        backend,
        retain_reference=True,
        retention_reason="review window",
        retention_days=30,
    )
    voice_id = descriptor["id"]
    assert service.store.reference_path(voice_id).exists()

    # Remove the authorising record: the exception is now unrecorded.
    service.store.consent_path(voice_id).unlink()

    assert voice_id in service._purge_expired_references()
    assert not service.store.reference_path(voice_id).exists()


# ---------------------------------------------------------------------------
# 20. Non-blocking review findings that are cheap to pin
# ---------------------------------------------------------------------------


def test_inactive_reason_reaches_the_wire():
    """Review finding 4: the camelCase key must not be dropped by Pydantic."""
    client, backend = _http_harness()
    descriptor = _create(voice_cloning.get_clone_service(), backend)
    voice_id = descriptor["id"]

    voice_cloning.get_clone_service().takedown(
        voice_id, requested_by="gateway", reason="subject request"
    )

    described = voice_cloning.get_clone_service().describe(voice_id)
    assert described["inactiveReason"] == "the clone has been disabled"

    from app.models.schemas import CloneVoiceInfo

    assert CloneVoiceInfo(**described).inactive_reason == "the clone has been disabled"


def test_the_legacy_sync_stream_refuses_a_clone(tmp_path, monkeypatch):
    """Review finding 5: no path may render a clone without its embedding."""
    backend = FakeCloneBackend()
    service = _make_service(backend=backend)
    tts = _tts_service_with_chatterbox(backend)
    descriptor = _create(service, backend)

    with pytest.raises(CloneInactiveError, match="cloned reference voice"):
        list(tts.synthesize_streaming("Hello.", voice=descriptor["id"]))


def test_provenance_verify_caps_the_upload_and_narrows_its_handler():
    """Review finding 6: a huge file is 413, not a decode error."""
    from app.routers.tts import MAX_REFERENCE_BYTES

    client, _backend = _http_harness()

    oversized = b"RIFF" + b"\x00" * (MAX_REFERENCE_BYTES + 1)
    response = client.post(
        "/api/tts/provenance/verify",
        files={"file": ("big.wav", oversized, "audio/wav")},
    )

    assert response.status_code == 413, response.text
    assert "larger than" in response.json()["detail"]
# ---------------------------------------------------------------------------
# 21. Clone ids cannot escape the store root (py/path-injection, alerts 81-89)
#
# `voice_dir()` is the only place a caller-supplied id is joined into a path, and
# every other helper inherits it. The confinement is enforced in the store rather
# than by caller discipline, because `CloneStore` is a service-level API and
# `read_consent()` is reached with a raw request-path id.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_id",
    [
        "../../../etc/passwd",
        "..",
        "clone_a/../../b",
        "clone_a/b",
        "clone_a\\b",
        "/etc/passwd",
        "clone_a\x00b",
        "",
        "-leading-dash",
        "Clone_Upper",
        "clone_" + "a" * 200,
        # `$` in a regex also matches just before a trailing newline, so these
        # passed `.match()` and needed `.fullmatch()` plus the basename check.
        "clone_x\n",
        "clone_x/",
        "clone_x/.",
        "clone_x\r\n",
        "clone_x\nclone_y",
    ],
)
def test_a_malformed_voice_id_is_rejected_before_a_path_is_built(bad_id):
    from app.services.clone_store import InvalidVoiceId, validate_voice_id

    with pytest.raises(InvalidVoiceId):
        validate_voice_id(bad_id)


def test_a_trailing_newline_would_have_passed_a_naive_match():
    """The specific gap: `.match()` accepted it, `.fullmatch()` does not.

    Pinned as a test in its own right so the reason the id sanitiser uses
    fullmatch cannot be "simplified" back to match() later.
    """
    from app.services.clone_store import CLONE_ID_PATTERN, InvalidVoiceId, validate_voice_id

    assert CLONE_ID_PATTERN.match("clone_x\n") is not None
    assert CLONE_ID_PATTERN.fullmatch("clone_x\n") is None

    with pytest.raises(InvalidVoiceId):
        validate_voice_id("clone_x\n")


@pytest.mark.parametrize(
    "value,expected",
    [("clone_x", "clone_x"), ("clone_vin_1a2b3c4d", "clone_vin_1a2b3c4d")],
)
def test_a_valid_id_is_returned_unchanged_so_callers_use_the_checked_value(
    value, expected
):
    """The sanitiser returns the component; callers must build paths from it."""
    from app.services.clone_store import validate_voice_id

    assert validate_voice_id(value) == expected


@pytest.mark.parametrize(
    "method",
    ["voice_dir", "consent_path", "embedding_path", "reference_path"],
)
def test_every_path_helper_refuses_to_escape_the_store_root(method, tmp_path):
    from app.services.clone_store import InvalidVoiceId, CloneStore

    store = CloneStore(tmp_path / "store", reference_key=b"0" * 32)
    with pytest.raises(InvalidVoiceId):
        getattr(store, method)("../../outside")


def test_a_well_formed_id_resolves_under_the_store_root(tmp_path):
    from app.services.clone_store import CloneStore

    root = tmp_path / "store"
    store = CloneStore(root, reference_key=b"0" * 32)

    for path in (
        store.voice_dir("clone_vin_1a2b3c4d"),
        store.consent_path("clone_vin_1a2b3c4d"),
        store.embedding_path("clone_vin_1a2b3c4d"),
        store.reference_path("clone_vin_1a2b3c4d"),
    ):
        assert path.resolve().is_relative_to(root.resolve())


def test_delete_refuses_to_unlink_outside_the_store(tmp_path):
    """A traversal id must not delete a file that happens to live above the root."""
    from app.services.clone_store import InvalidVoiceId, CloneStore

    store = CloneStore(tmp_path / "store" / "clones", reference_key=b"0" * 32)
    victim = tmp_path / "store" / "victim.wav"
    victim.parent.mkdir(parents=True, exist_ok=True)
    victim.write_bytes(b"must survive")

    with pytest.raises(InvalidVoiceId):
        store.delete_artefacts("../victim")

    assert victim.exists(), "a traversal id unlinked a file outside the store"


def test_the_consent_endpoint_returns_404_for_a_traversal_id():
    """`consent_record()` is the live unguarded caller CodeQL flagged.

    Over HTTP the encoded separator is decoded before routing, so the route does
    not match at all and the caller gets a plain 404. That is the first line of
    defence, not the only one: the store-level guard below covers the case where
    a future route or a direct service call passes the id through.
    """
    client, _backend = _http_harness()

    response = client.get("/api/tts/voices/..%2f..%2fetc%2fpasswd/consent")

    assert response.status_code == 404
    assert "passwd" not in response.text.lower()


def test_the_consent_lookup_treats_a_traversal_id_as_no_record():
    """The service-level caller: an unusable id is indistinguishable from unknown."""
    service = _make_service(backend=FakeCloneBackend())

    for bad in ("../../../etc/passwd", "..", "clone_a/b"):
        assert service.consent_record(bad) is None
        assert service.describe(bad) is None


def test_synthesis_with_a_traversal_voice_id_does_not_touch_the_filesystem():
    client, _backend = _http_harness()

    response = client.post(
        "/api/tts/synthesize/json",
        json={"text": "Hello.", "voice": "../../../etc/passwd"},
    )

    # Nothing to render and nothing to leak: a 4xx, never a 5xx from a failed read.
    assert 400 <= response.status_code < 500, response.text