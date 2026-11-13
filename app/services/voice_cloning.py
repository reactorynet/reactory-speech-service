"""Consent-first voice cloning: the governance layer around a single model call.

The clone itself is one library call.  Everything in this module exists because
of the asymmetry in ADR-0003: the subject of a reference clip is not a party to
our system, cannot revoke anything, and may never learn their voice was used.
The six invariants are enforced here, not by configuration:

1. **Approver authority is deployed configuration.** ``clone_approvers.json`` is
   read on every approval.  An absent or empty file makes cloning *unavailable*,
   never unrestricted.  Self-approval works because the same allowlist applies.
2. **Consent evidence is co-located and instance-signed.**  The signature proves
   which instance recorded the record and that it is unaltered.  It does **not**
   prove the subject consented, and no docstring or HTTP field here says it does.
3. **Reference audio is transient.**  It is written only in encrypted form,
   deleted as soon as embedding extraction completes, and retention is a visible,
   reasoned exception hard-capped at 90 days.
4. **No watermark in production audio** is enforced by the Chatterbox adapter;
   clone synthesis goes through the same path, so the same guard applies.
5. **Takedown is an external privilege.**  The service executes an assertion it
   is given; it cannot model privilege.  Its own obligation is bounded: once
   applied, the voice is inert immediately, without a redeploy.
6. **The service does not model identity.**  There is no role check anywhere in
   this module; ``scope`` is recorded, not enforced.
"""

from __future__ import annotations

import io
import logging
import re
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np
import soundfile as sf

from app.config import settings
from app.services import clone_crypto, provenance, voice_registry
from app.services.clone_crypto import Ed25519PrivateKey, Ed25519PublicKey, KeyMaterialError
from app.services.clone_store import (
    CLONE_ID_PREFIX,
    MAX_REFERENCE_RETENTION_DAYS,
    REQUIRED_CONSENT_FIELDS,
    SCOPES,
    CloneStore,
    CloneStoreError,
    format_timestamp,
    is_clone_id,
    parse_timestamp,
)
from app.services.voice_registry import VoiceRecord

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]

#: What a signed record proves.  Carried verbatim in every consent payload so a
#: reader of the artefact cannot mistake chain-of-custody for subject consent.
CONSENT_NOTICE = (
    "Signed by this service instance. Proves which instance recorded this consent "
    "and that it is unaltered since. It does NOT prove that the subject consented."
)

#: Prompt text for the retention exception, shown on the consent record.
DEFAULT_RETENTION_DAYS = MAX_REFERENCE_RETENTION_DAYS


class CloningUnavailableError(RuntimeError):
    """Cloning is not configured on this deployment (maps to HTTP 503)."""


class CloneValidationError(ValueError):
    """Consent input is incomplete or invalid (maps to HTTP 422).

    ``missing_fields`` names each absent required field individually so the HTTP
    layer can report all of them, not just the first.
    """

    def __init__(self, missing_fields: list[str] | None = None, message: str | None = None):
        self.missing_fields = list(missing_fields or [])
        if message is None:
            if self.missing_fields:
                message = (
                    "Missing required consent fields: " + ", ".join(self.missing_fields)
                )
            else:
                message = "The consent record is invalid."
        elif self.missing_fields:
            # Keep the field name visible in the message as well as in
            # ``missing_fields``: the HTTP layer reports the string, and a caller
            # must be able to fix the request without guessing.
            message = f"{message} (field: {', '.join(self.missing_fields)})"
        super().__init__(message)


class CloneAudioError(CloneValidationError):
    """The uploaded reference audio is absent, empty or undecodable."""


class ApproverNotAllowedError(PermissionError):
    """The named approver is not in this deployment's configured allowlist."""


class CloneInactiveError(ValueError):
    """A known clone is disabled, taken down or expired (maps to HTTP 410)."""


class CloneNotFoundError(LookupError):
    """The requested clone does not exist (maps to HTTP 404)."""


class CloneBackendUnavailable(CloningUnavailableError):
    """No engine that can clone is loaded (maps to HTTP 503)."""


class CloneBackend(Protocol):
    """The two operations a cloning engine must expose.

    Extraction happens once, at creation.  Synthesis happens on every request
    and receives the **stored embedding**, never the reference clip: the clip is
    deleted by default, so a "re-clone from reference" flow must not exist.
    """

    # Bodies are docstrings rather than ``...``: a bare ``...`` is an expression
    # statement with no effect, and the static analyzer reports every one of them.
    # A Protocol member is never called, so the body has no runtime meaning.
    def extract_embedding(self, audio: bytes) -> bytes:
        """Derive the speaker conditioning blob from a reference clip."""

    def render(
        self, text: str, embedding: bytes, *, exaggeration: float
    ) -> tuple[np.ndarray, int]:
        """Render ``text`` in the speaker carried by ``embedding``."""


@dataclass(frozen=True)
class CloneAvailability:
    """Whether cloning may be used, and why not when it may not."""

    available: bool
    reason: str | None = None
    approver_count: int = 0
    signing_ready: bool = False
    reference_key_ready: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "reason": self.reason,
            "approverCount": self.approver_count,
            "signingKeyConfigured": self.signing_ready,
            "referenceKeyConfigured": self.reference_key_ready,
        }


@dataclass(frozen=True)
class InstanceIdentity:
    """The instance's signing identity.

    ``instance_id`` travels inside every signature payload, so a record can only
    verify against the key of the instance that wrote it.
    """

    instance_id: str
    private_key: Ed25519PrivateKey
    key_fingerprint: str

    @property
    def public_key(self) -> Ed25519PublicKey:
        return self.private_key.public_key()

    def sign_payload(self, payload: dict[str, Any]) -> str:
        return clone_crypto.sign(self.private_key, provenance.canonical_json(payload))

    def verify_payload(self, payload: dict[str, Any], signature: str) -> bool:
        return clone_crypto.verify(
            self.public_key, provenance.canonical_json(payload), signature
        )


@dataclass(frozen=True)
class CloneSynthesisContext:
    """What a synthesis needs to know about an active clone."""

    voice_id: str
    engine: str
    embedding: bytes
    consent_id: str | None
    entry: dict[str, Any] = field(repr=False)


# ---------------------------------------------------------------------------
# Configuration: approvers and key material
# ---------------------------------------------------------------------------


def load_approvers(path: str | Path) -> tuple[str, ...]:
    """Read ``clone_approvers`` from the deployed config file.

    Tolerant on purpose: a missing or malformed file yields an **empty** tuple,
    which makes cloning unavailable.  Raising here would turn a configuration
    omission into a startup failure; returning an empty list is fail-closed and
    keeps the rest of the service up.
    """
    candidate = Path(path)
    try:
        import json

        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return ()
    except (OSError, ValueError) as exc:
        logger.warning(
            "Clone approver config %s could not be read (%s); cloning is UNAVAILABLE.",
            candidate,
            exc,
        )
        return ()
    raw = payload.get("clone_approvers") if isinstance(payload, dict) else None
    if not isinstance(raw, list):
        return ()
    return tuple(
        str(item).strip().lower()
        for item in raw
        if isinstance(item, str) and str(item).strip()
    )


def build_identity(settings_obj: Any = settings) -> InstanceIdentity | None:
    """Load the instance signing key, or return ``None`` (never generate one)."""
    path = getattr(settings_obj, "clone_signing_key_path", "")
    if not path:
        return None
    try:
        private_key = clone_crypto.load_signing_key(path)
    except KeyMaterialError as exc:
        logger.warning(
            "Voice cloning UNAVAILABLE: %s The service will NOT generate a key at boot "
            "because a regenerated key invalidates every existing consent signature.",
            exc,
        )
        return None
    instance_id = (
        str(getattr(settings_obj, "clone_instance_id", "") or "").strip()
        or socket.gethostname()
    )
    return InstanceIdentity(
        instance_id=instance_id,
        private_key=private_key,
        key_fingerprint=clone_crypto.fingerprint(private_key),
    )


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def availability_of(
    *,
    approvers: tuple[str, ...],
    identity: InstanceIdentity | None,
    reference_key: bytes | None,
) -> CloneAvailability:
    """Assess cloning availability, failing closed and naming every gap."""
    missing: list[str] = []
    if not approvers:
        missing.append(
            "no clone approvers configured (CLONE_APPROVERS_PATH is absent or empty)"
        )
    if identity is None:
        missing.append("no instance signing key (CLONE_SIGNING_KEY_PATH)")
    if reference_key is None:
        missing.append("no reference-audio key (CLONE_REFERENCE_KEY_PATH)")
    return CloneAvailability(
        available=not missing,
        reason="; ".join(missing) if missing else None,
        approver_count=len(approvers),
        signing_ready=identity is not None,
        reference_key_ready=reference_key is not None,
    )


# ---------------------------------------------------------------------------
# Backend resolution
#
# Rendering belongs to the engine adapter: it is the only place that owns the
# loaded model *and* the watermark neutralisation (ADR-0003 #4).  The resolver is
# registered by TTSService, so this module never imports torch or chatterbox.
# ---------------------------------------------------------------------------

_backend_resolver: Callable[[], CloneBackend | None] | None = None


def set_clone_backend_resolver(resolver: Callable[[], CloneBackend | None] | None) -> None:
    global _backend_resolver
    _backend_resolver = resolver


def _resolve_backend() -> CloneBackend | None:
    if _backend_resolver is None:
        return None
    try:
        return _backend_resolver()
    except Exception:  # pragma: no cover - a broken adapter must not break the API
        logger.warning("Clone backend resolver failed.", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# The service
# ---------------------------------------------------------------------------


class VoiceCloneService:
    """Creation, synthesis authorisation, provenance and takedown for clones."""

    def __init__(
        self,
        *,
        store: CloneStore,
        identity: InstanceIdentity | None,
        reference_key: bytes | None,
        approvers_path: str | Path,
        backend_resolver: Callable[[], CloneBackend | None] | None = None,
        clock: Callable[[], float] = time.time,
        default_variant: str = "standard",
    ) -> None:
        self.store = store
        self.identity = identity
        self._reference_key = reference_key
        self._approvers_path = Path(approvers_path)
        self._backend_resolver = backend_resolver or _resolve_backend
        self._clock = clock
        self._variant = default_variant
        self._lock = threading.RLock()
        #: Number of times the approval policy file has been read.  Exposed so a
        #: test can prove the allowlist is configuration, not a cached constant.
        self.approver_reads = 0
        # A retained reference past its ceiling is deleted on boot as well as at
        # request time.  Neither is a background job: a job that stops running
        # must not silently keep biometric-adjacent data *or* re-enable a voice.
        self._purge_expired_references()
        self._register_with_voice_registry()

    # -- availability --------------------------------------------------

    @property
    def approvers(self) -> tuple[str, ...]:
        self.approver_reads += 1
        return load_approvers(self._approvers_path)

    def availability(self) -> CloneAvailability:
        return availability_of(
            approvers=self.approvers,
            identity=self.identity,
            reference_key=self._reference_key,
        )

    def _require_available(self) -> CloneAvailability:
        state = self.availability()
        if not state.available:
            raise CloningUnavailableError(
                "Voice cloning is unavailable on this deployment: "
                f"{state.reason}. An absent or empty configuration makes cloning "
                "unavailable, never unrestricted (ADR-0003 decision 1)."
            )
        return state

    # -- voice registry integration ------------------------------------

    def _register_with_voice_registry(self) -> None:
        voice_registry.set_reference_provider(self._active_records)
        voice_registry.set_reference_lookup(self._lookup_record)
        voice_registry.set_reference_authorizer(self.is_active)

    def _active_records(self, engine_id: str) -> list[VoiceRecord]:
        records: list[VoiceRecord] = []
        for entry in self.store.entries():
            if str(entry.get("engine")) != engine_id:
                continue
            if not self._entry_active(entry)[0]:
                continue
            records.append(self._entry_to_record(entry))
        return records

    def _lookup_record(self, voice_id: str) -> VoiceRecord | None:
        entry = self.store.get(voice_id)
        if entry is None or not self._entry_active(entry)[0]:
            return None
        return self._entry_to_record(entry)

    def _entry_to_record(self, entry: dict[str, Any]) -> VoiceRecord:
        return VoiceRecord(
            id=str(entry["id"]),
            name=str(entry.get("name") or entry["id"]),
            engine=str(entry.get("engine") or "chatterbox"),
            language=str(entry.get("language") or "en-US"),
            gender=str(entry.get("gender") or "unspecified"),
            tags=tuple(str(tag) for tag in entry.get("tags") or ("clone", "reference")),
            quality=None,
            license=str(entry.get("license") or "MIT"),
            attribution=str(entry.get("attribution") or "Chatterbox (Resemble AI)"),
            style_spec={
                "type": "reference",
                "variant": str(entry.get("variant") or self._variant),
                "engine": str(entry.get("engine") or "chatterbox"),
            },
            supports={
                "rate": False,
                "pitch": False,
                "volume": False,
                "style": True,
                "clone": True,
            },
            enabled=True,
            locale=str(entry.get("locale") or "en-us"),
            raw=dict(entry),
        )

    # -- lifecycle: active / expired / disabled ------------------------

    def _verified_consent(self, voice_id: str) -> tuple[dict[str, Any] | None, str | None]:
        """Read and verify the consent record beside the voice.

        Returns the record on success, or ``(None, reason)``.  This is checked in
        the same request-time path as takedown and expiry, so tampering with the
        record has a consequence beyond a ``signatureValid`` flag on a read
        endpoint: the voice stops rendering.  A missing or unreadable record is a
        failure too — a clone whose authorising evidence cannot be produced is not
        authorised.

        The signature proves *which instance* recorded the consent and that the
        record is unaltered.  It does not prove the subject consented; see
        :data:`CONSENT_NOTICE`.
        """
        record = self.store.read_consent(voice_id)
        if record is None:
            return None, "the consent record is missing or unreadable"
        if self.identity is None:
            return (
                None,
                "no instance signing key is configured, so the consent record cannot be verified",
            )
        signature = str(record.get("signature") or "")
        if not signature:
            return None, "the consent record is not signed"
        payload = {key: value for key, value in record.items() if key != "signature"}
        if not self.identity.verify_payload(payload, signature):
            return (
                None,
                "the consent record failed verification: it was altered, or it was "
                "signed by a different instance",
            )
        return record, None

    def _signed_expiry(self, entry: dict[str, Any]) -> tuple[float | None, str | None]:
        """The expiry that authorises synthesis, taken from the **signed** record.

        ``registry.json`` is not signed.  Reading ``entry['expiresAt']`` would let
        anyone who can edit that file extend a clone years past the consent it was
        granted under, while the co-located signature still verified — the same
        fail-open hole the signature check was added to close, one layer down.

        The registry value is therefore only a consistency witness: if it
        disagrees with the signed record the clone is inert, because the two are
        no longer describing the same authority.  Timestamps are compared as
        parsed instants, not strings, so a different but equivalent rendering of
        the same moment is not mistaken for tampering.
        """
        record, reason = self._verified_consent(str(entry["id"]))
        if record is None:
            return None, reason
        signed_expiry = parse_timestamp(str(record.get("expiresAt") or ""))
        if signed_expiry is None:
            return None, "the signed consent has no valid expiry"
        registry_raw = entry.get("expiresAt")
        registry_expiry = parse_timestamp(str(registry_raw)) if registry_raw else None
        if registry_expiry is None:
            return None, "the clone has no valid consent expiry"
        if abs(registry_expiry - signed_expiry) > 0.001:
            return None, (
                "the registry expiry does not match the signed consent record; the "
                "clone is inert until they agree (the signed record is authoritative)"
            )
        return signed_expiry, None

    def _retention_deadline(
        self, entry: dict[str, Any]
    ) -> tuple[float | None, str | None]:
        """When a retained reference must be gone, derived from the **signed** record.

        The 90-day ceiling is an invariant, not a setting, so the value that
        enforces it cannot come from ``registry.json`` — that file is unsigned and
        an edit to one field there would extend biometric-adjacent retention
        indefinitely.  Same reasoning as :meth:`_signed_expiry`, one field over.

        The signed deadline is additionally clamped to ``createdAt + 90 days``, so
        even a correctly signed record cannot authorise a longer window than the
        ceiling allows.

        Returns ``(delete_at, problem)``.  ``problem`` is non-``None`` when
        retention cannot be justified from the signed record at all, which the
        caller treats as "purge now" — deleting early is the safe direction for
        biometric-adjacent data.
        """
        reference = entry.get("reference") or {}
        record, reason = self._verified_consent(str(entry["id"]))
        if record is None:
            return None, reason
        retention = record.get("referenceRetention") or {}
        if not retention.get("retained"):
            return None, "the signed consent records no retention exception"
        signed_delete = parse_timestamp(str(retention.get("deleteAt") or ""))
        if signed_delete is None:
            return None, "the signed retention has no valid deleteAt"
        created_raw = entry.get("createdAt")
        created_at = parse_timestamp(str(created_raw)) if created_raw else None
        if created_at is None:
            return None, "the clone has no valid creation time"
        # The ceiling is absolute: clamp regardless of what the record says.
        hard_cap = created_at + MAX_REFERENCE_RETENTION_DAYS * 86400
        ceiling = min(signed_delete, hard_cap)
        registry_delete = parse_timestamp(str(reference.get("deleteAt") or ""))
        if registry_delete is not None and abs(registry_delete - ceiling) > 0.001:
            # Disagreement means the unsigned copy is describing a different
            # authority than the signed one. Treat it as tampering: purge now.
            return None, (
                "the registry retention deadline does not match the signed consent record"
            )
        return ceiling, None

    def _purge_expired_references(self) -> list[str]:
        """Delete retained references that are past their signed deadline.

        Called at request time and at construction, never from a background job.
        A retained clip whose deadline cannot be established from a verified
        consent record is deleted rather than kept: retention is an exception that
        must be justified, not a default that survives losing its justification.
        """
        now = self._clock()
        purged: list[str] = []
        for entry in self.store.entries():
            reference = entry.get("reference") or {}
            if not reference.get("retained"):
                continue
            voice_id = str(entry["id"])
            deadline, problem = self._retention_deadline(entry)
            if deadline is None:
                logger.warning(
                    "Deleting retained reference audio for %s: %s. Retention must be "
                    "justified by the instance-signed consent record.",
                    voice_id,
                    problem,
                )
                expired = True
            else:
                expired = now >= deadline
            if not expired:
                continue
            self.store.delete_reference(voice_id)
            self.store.update(
                voice_id,
                reference={
                    **reference,
                    "retained": False,
                    "purgedAt": format_timestamp(now),
                },
            )
            self.store.audit(
                "reference-expired",
                voiceId=voice_id,
                reason=(
                    "retention ceiling reached"
                    if problem is None
                    else "retention could not be justified from the signed consent"
                ),
            )
            purged.append(voice_id)
        return purged

    def _entry_state(
        self, entry: dict[str, Any]
    ) -> tuple[bool, str | None, float | None]:
        """Activity plus the signed expiry that decided it.

        One pass, so a caller that needs both the verdict and the authoritative
        expiry (``_describe``) does not verify the signature twice.
        """
        if not bool(entry.get("enabled", True)):
            return False, "the clone has been disabled", None
        if entry.get("takedown"):
            return False, "the clone has been taken down", None
        signed_expiry, reason = self._signed_expiry(entry)
        if signed_expiry is None:
            return False, reason, None
        if self._clock() >= signed_expiry:
            return False, "the consent authorising this clone has expired", signed_expiry
        return True, None, signed_expiry

    def _entry_active(self, entry: dict[str, Any]) -> tuple[bool, str | None]:
        active, reason, _signed_expiry = self._entry_state(entry)
        return active, reason

    def is_active(self, voice_id: str | None) -> bool:
        entry = self.store.get(voice_id)
        return bool(entry) and self._entry_active(entry)[0]

    def active_reason(self, voice_id: str | None) -> str | None:
        entry = self.store.get(voice_id)
        if entry is None:
            return "no clone record"
        return self._entry_active(entry)[1]

    # -- creation ------------------------------------------------------

    def create(
        self,
        reference_audio: bytes,
        consent: dict[str, Any],
        *,
        created_by: str,
        retain_reference: bool = False,
        retention_reason: str | None = None,
        retention_days: int | None = None,
    ) -> dict[str, Any]:
        """Create a clone from a reference clip plus a structured consent record.

        Every refusal names the field or rule that caused it.  Nothing is written
        until the approver, the scope and the expiry have all been checked.
        """
        self._require_available()
        approvers = self.approvers

        missing = [
            field
            for field in REQUIRED_CONSENT_FIELDS
            if not str(consent.get(field) or "").strip()
        ]
        if missing:
            raise CloneValidationError(missing)

        scope = str(consent["scope"]).strip()
        if scope not in SCOPES:
            raise CloneValidationError(
                ["scope"],
                f"scope '{scope}' is not one of: {', '.join(SCOPES)}.",
            )

        captured_at = parse_timestamp(str(consent["capturedAt"]))
        if captured_at is None:
            raise CloneValidationError(
                ["capturedAt"], "capturedAt must be an ISO-8601 timestamp."
            )
        expires_at = parse_timestamp(str(consent["expiresAt"]))
        if expires_at is None:
            raise CloneValidationError(
                ["expiresAt"], "expiresAt must be an ISO-8601 timestamp."
            )
        now = self._clock()
        if expires_at <= now:
            raise CloneValidationError(
                ["expiresAt"],
                "expiresAt is in the past; a consent that has already expired cannot "
                "authorise a clone.",
            )
        if captured_at > expires_at:
            raise CloneValidationError(
                ["capturedAt", "expiresAt"],
                "capturedAt must not be later than expiresAt.",
            )

        approver = str(consent["approver"]).strip().lower()
        if approver not in approvers:
            # Deliberately a refusal, not a warning (ADR-0003 #1).  Self-approval
            # needs no special path: the same allowlist check applies.
            raise ApproverNotAllowedError(
                f"Approver '{consent['approver']}' is not in this deployment's configured "
                "clone_approvers list. An approver outside the deployment configuration "
                "is refused; there is no separate self-approval path."
            )

        audio = bytes(reference_audio or b"")
        samples, sample_rate = _decode_reference(audio)
        if samples == 0:
            raise CloneAudioError([], "The reference audio contains no samples.")

        backend = self._backend_resolver()
        if backend is None:
            raise CloneBackendUnavailable(
                "No cloning-capable engine is loaded. Voice cloning runs locally on "
                "Chatterbox; enable and load it before creating a clone."
            )

        retention = self._retention_plan(
            retain_reference=retain_reference,
            retention_reason=retention_reason,
            retention_days=retention_days,
            now=now,
        )
        voice_id = self._allocate_id(str(consent["subjectName"]))
        consent_id = f"consent_{uuid.uuid4().hex[:16]}"

        # Encrypted at rest *before* extraction touches it, so a crash never
        # leaves plaintext reference audio on disk *in the store*.  Extraction
        # makes its own short-lived copy in a private 0700 directory because the
        # upstream API takes a path; that copy is removed in a finally.
        self.store.write_reference(voice_id, audio)
        try:
            embedding = backend.extract_embedding(audio)
        except Exception as exc:
            self.store.delete_reference(voice_id)
            raise CloneAudioError(
                [],
                "The reference audio could not be converted into a speaker embedding: "
                f"{exc}",
            ) from exc

        # Default and preferred behaviour: the clip is gone as soon as the
        # embedding exists (ADR-0003 #3).
        if not retention["retained"]:
            self.store.delete_reference(voice_id)

        self.store.write_embedding(voice_id, embedding)

        consent_payload: dict[str, Any] = {
            "consentId": consent_id,
            "subjectName": str(consent["subjectName"]).strip(),
            "purpose": str(consent["purpose"]).strip(),
            "capturedAt": str(consent["capturedAt"]).strip(),
            # Normalised so the signed record and the registry entry carry the
            # same rendering of the instant; _signed_expiry cross-checks them.
            "expiresAt": format_timestamp(expires_at),
            "approver": str(consent["approver"]).strip(),
            "evidence": str(consent["evidence"]).strip(),
            "scope": scope,
            "instanceId": self.identity.instance_id if self.identity else "",
            "createdAt": format_timestamp(now),
            "createdBy": str(created_by or "unspecified"),
            "referenceRetention": retention,
            "notice": CONSENT_NOTICE,
        }
        signature = self.identity.sign_payload(consent_payload) if self.identity else ""
        consent_record = {**consent_payload, "signature": signature}
        self.store.write_consent(voice_id, consent_record)

        tags = ["clone", "reference", scope]
        if retention["retained"]:
            tags.append("reference-retained")
        entry: dict[str, Any] = {
            "id": voice_id,
            "name": str(consent.get("name") or f"{consent['subjectName']} (cloned)"),
            "engine": "chatterbox",
            "variant": str(getattr(settings, "chatterbox_variant", self._variant)),
            "language": "en-US",
            "locale": "en-us",
            "gender": "unspecified",
            "license": "MIT",
            "attribution": "Chatterbox (Resemble AI)",
            "tags": tags,
            "createdAt": format_timestamp(now),
            "createdBy": str(created_by or "unspecified"),
            "approvedBy": str(consent["approver"]).strip(),
            "consentId": consent_id,
            "scope": scope,
            "expiresAt": format_timestamp(expires_at),
            "enabled": True,
            "takedown": None,
            "reference": retention,
            "referenceSamples": samples,
            "referenceSampleRate": sample_rate,
            "embeddingBytes": len(embedding),
        }
        try:
            stored = self.store.add(entry)
        except CloneStoreError:
            self.store.delete_artefacts(voice_id)
            raise
        self.store.audit(
            "create",
            voiceId=voice_id,
            consentId=consent_id,
            actor=created_by,
            approvedBy=entry["approvedBy"],
            scope=scope,
            referenceRetained=retention["retained"],
        )
        return self._describe(stored)

    def _retention_plan(
        self,
        *,
        retain_reference: bool,
        retention_reason: str | None,
        retention_days: int | None,
        now: float,
    ) -> dict[str, Any]:
        """Build the visible retention exception, honouring the hard 90-day cap."""
        if not retain_reference:
            return {"retained": False, "reason": None, "deleteAt": None, "days": 0}
        reason = str(retention_reason or "").strip()
        if not reason:
            raise CloneValidationError(
                ["retentionReason"],
                "Retaining the reference clip is an exception and needs a reason on the "
                "consent record. Omit retain_reference to delete it after extraction.",
            )
        requested = DEFAULT_RETENTION_DAYS if retention_days is None else int(retention_days)
        if requested < 1:
            raise CloneValidationError(
                ["retentionDays"], "retentionDays must be at least 1 when retaining."
            )
        effective = min(requested, MAX_REFERENCE_RETENTION_DAYS)
        return {
            "retained": True,
            "reason": reason,
            "requestedDays": requested,
            "days": effective,
            "capped": requested > MAX_REFERENCE_RETENTION_DAYS,
            "deleteAt": format_timestamp(now + effective * 86400),
        }

    def _allocate_id(self, subject_name: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", subject_name.strip().lower()).strip("-") or "voice"
        slug = slug[:24].strip("-") or "voice"
        for _ in range(16):
            candidate = f"{CLONE_ID_PREFIX}{slug}_{uuid.uuid4().hex[:8]}"
            if self.store.get(candidate) is None:
                return candidate
        raise CloningUnavailableError("Could not allocate a unique clone id.")  # pragma: no cover

    # -- reads ---------------------------------------------------------

    def list_clones(self, *, include_inactive: bool = False) -> list[dict[str, Any]]:
        self._purge_expired_references()
        described = [self._describe(entry) for entry in self.store.entries()]
        if include_inactive:
            return described
        return [entry for entry in described if entry["active"]]

    def describe(self, voice_id: str) -> dict[str, Any] | None:
        self._purge_expired_references()
        entry = self.store.get(voice_id)
        return self._describe(entry) if entry is not None else None

    def _describe(self, entry: dict[str, Any]) -> dict[str, Any]:
        active, reason, signed_expiry = self._entry_state(entry)
        # The signed record is the authority for what was agreed, so it is what
        # gets reported.  The registry copy is only shown when no signed expiry
        # could be established, in which case the clone is already inert.
        expires_at = (
            format_timestamp(signed_expiry)
            if signed_expiry is not None
            else entry.get("expiresAt")
        )
        return {
            "id": entry["id"],
            "name": entry.get("name") or entry["id"],
            "language": entry.get("language") or "en-US",
            "locale": entry.get("locale") or "en-us",
            "gender": entry.get("gender") or "unspecified",
            "engine": entry.get("engine") or "chatterbox",
            "quality": None,
            "tags": list(entry.get("tags") or []),
            "license": entry.get("license") or "MIT",
            "attribution": entry.get("attribution") or "Chatterbox (Resemble AI)",
            "clone": True,
            "active": active,
            "inactiveReason": reason,
            "consent_id": entry.get("consentId"),
            "created_at": entry.get("createdAt"),
            "created_by": entry.get("createdBy"),
            "approved_by": entry.get("approvedBy"),
            "scope": entry.get("scope"),
            "expires_at": expires_at,
            "reference_retained": bool((entry.get("reference") or {}).get("retained")),
        }

    def consent_record(self, voice_id: str) -> dict[str, Any] | None:
        record = self.store.read_consent(voice_id)
        if record is None:
            return None
        # Same verification the request path uses, so this endpoint cannot report
        # a record as valid that authorisation would reject.
        _verified, _reason = self._verified_consent(voice_id)
        valid = _verified is not None
        return {
            "record": record,
            "signatureValid": valid,
            "verifiedAgainstInstanceId": self.identity.instance_id if self.identity else None,
            "notice": CONSENT_NOTICE,
        }

    # -- synthesis -----------------------------------------------------

    def authorize_synthesis(self, voice_id: str | None) -> CloneSynthesisContext | None:
        """Authorise a synthesis at **request time**.

        Expiry and takedown are checked here, in the request path, not by a
        background sweep: a job that stops running must not silently re-enable
        voices (ADR-0003 #6).  An id that looks like a clone but has no record
        fails closed rather than falling back to another voice.
        """
        if not voice_id:
            return None
        # Look the voice up FIRST.  Cloning is optional per deployment: an
        # instance with no signing key, no reference key and no allowlist is a
        # perfectly normal deployment that simply cannot clone.  Refusing here
        # before the lookup made every named voice — pack, designed, everything —
        # fail with 503 on such an instance, which is the opposite of "cloning is
        # unavailable but the rest of the service keeps running".
        entry = self.store.get(voice_id)
        if entry is None:
            if is_clone_id(voice_id):
                # A clone id is never allowed to fall back to another voice.
                raise CloneInactiveError(
                    f"No clone record for '{voice_id}': it was deleted or the clone store "
                    "is unreadable. A known clone id never falls back to another voice."
                )
            # Not a clone: the request is none of this service's business.
            return None
        # From here the voice IS a clone, so clone configuration is required.
        if self.identity is None:
            # Manifests and consent records are instance-signed, and the consent
            # record cannot be verified without the key. Refuse the clone rather
            # than render it unsigned.
            raise CloningUnavailableError(
                "The instance signing key is not configured, so the consent record "
                "cannot be verified and no provenance manifest can be emitted; cloned "
                "voices are unavailable on this deployment."
            )
        self._purge_expired_references()
        active, reason = self._entry_active(entry)
        if not active:
            raise CloneInactiveError(
                f"Voice '{voice_id}' is not active: {reason}. It cannot be synthesised."
            )
        try:
            embedding = self.store.load_embedding(voice_id)
        except (CloneStoreError, KeyMaterialError) as exc:
            raise CloningUnavailableError(
                f"Voice '{voice_id}' cannot be synthesised: {exc}"
            ) from exc
        return CloneSynthesisContext(
            voice_id=voice_id,
            engine=str(entry.get("engine") or "chatterbox"),
            embedding=embedding,
            consent_id=entry.get("consentId"),
            entry=entry,
        )

    def record_synthesis(
        self,
        voice_id: str,
        audio: bytes,
        *,
        text_length: int,
        engine: str,
    ) -> dict[str, Any]:
        """Emit (or reuse) the signed manifest for one rendered file, and audit it."""
        if self.identity is None:
            raise CloningUnavailableError(
                "No instance signing key is configured; a provenance manifest cannot be "
                "signed."
            )
        entry = self.store.get(voice_id) or {}
        manifest = provenance.build_manifest(
            private_key=self.identity.private_key,
            audio=audio,
            voice_id=voice_id,
            consent_id=entry.get("consentId"),
            instance_id=self.identity.instance_id,
            created_at=format_timestamp(self._clock()),
        )
        stored = self.store.register_manifest(manifest)
        # Character count only.  The synthesis text is never written to the audit
        # trail: it is the caller's content, not a governance fact.
        self.store.audit(
            "use",
            voiceId=voice_id,
            consentId=entry.get("consentId"),
            engine=engine,
            charCount=int(text_length),
        )
        return stored

    # -- provenance lookup ---------------------------------------------

    def lookup_provenance(self, audio: bytes) -> dict[str, Any]:
        """Verify a file by registry lookup; never asserts a file is foreign."""
        digest = provenance.audio_sha256(audio)
        return self.lookup_by_digest(digest)

    def lookup_by_digest(self, digest: str) -> dict[str, Any]:
        manifest = self.store.get_manifest(digest)
        if manifest is None:
            # The wording is a safety property.  Absence of a record is not
            # evidence of absence: the file may be transcoded or never registered.
            return {
                "status": provenance.NO_RECORD_STATUS,
                "audioSha256": digest,
                "manifest": None,
            }
        if self.identity is None:
            status = "verification unavailable"
        else:
            status = (
                provenance.VERIFIED_STATUS
                if provenance.verify_manifest(manifest, self.identity.public_key)
                else provenance.INVALID_SIGNATURE_STATUS
            )
        return {"status": status, "audioSha256": digest, "manifest": manifest}

    # -- takedown and deletion -----------------------------------------

    def takedown(self, voice_id: str, *, requested_by: str, reason: str) -> dict[str, Any]:
        """Disable a clone and remove its artefacts, effective immediately.

        The caller is an authorising gateway; the service records the assertion
        it is given and does not verify it (ADR-0003 #5).
        """
        entry = self.store.get(voice_id)
        if entry is None:
            raise CloneNotFoundError(f"Unknown clone '{voice_id}'.")
        requested_by, reason = _require_actor_and_reason(requested_by, reason)
        self.store.delete_artefacts(voice_id, _preview_dir())
        updated = self.store.update(
            voice_id,
            enabled=False,
            takedown={
                "at": format_timestamp(self._clock()),
                "requestedBy": requested_by,
                "reason": reason,
            },
        )
        self.store.audit(
            "takedown",
            voiceId=voice_id,
            consentId=entry.get("consentId"),
            requestedBy=requested_by,
            reason=reason,
        )
        return self._describe(updated)

    def delete(self, voice_id: str, *, requested_by: str, reason: str) -> dict[str, Any]:
        """Remove audio, embedding, registry entry and any preview.

        Succeeds when a file is already missing.  The audit entry records who,
        when and why, and deliberately contains no audio and no hash of it.
        """
        entry = self.store.get(voice_id)
        if entry is None:
            raise CloneNotFoundError(f"Unknown clone '{voice_id}'.")
        requested_by, reason = _require_actor_and_reason(requested_by, reason)
        self.store.delete_artefacts(voice_id, _preview_dir())
        self.store.remove(voice_id)
        self.store.audit(
            "delete",
            voiceId=voice_id,
            consentId=entry.get("consentId"),
            requestedBy=requested_by,
            reason=reason,
        )
        return {"id": voice_id, "status": "deleted"}

    # -- audit ---------------------------------------------------------

    def audit_entries(self) -> list[dict[str, Any]]:
        return self.store.read_audit()


def _require_actor_and_reason(requested_by: str, reason: str) -> tuple[str, str]:
    missing: list[str] = []
    actor = str(requested_by or "").strip()
    why = str(reason or "").strip()
    if not actor:
        missing.append("requestedBy")
    if not why:
        missing.append("reason")
    if missing:
        raise CloneValidationError(
            missing,
            "A deletion or takedown must record who requested it and why: "
            + ", ".join(missing),
        )
    return actor, why


def _preview_dir() -> Path | None:
    try:
        return _resolve_path(settings.designed_voices_preview_resolved)
    except Exception:  # pragma: no cover - defensive
        return None


def _decode_reference(audio: bytes) -> tuple[int, int]:
    """Return ``(sample_count, sample_rate)`` or raise a caller-facing 422."""
    if not audio:
        raise CloneAudioError([], "The reference audio is empty.")
    try:
        data, sample_rate = sf.read(io.BytesIO(audio), dtype="float32", always_2d=True)
    except Exception as exc:
        raise CloneAudioError(
            [], f"The reference audio could not be decoded as WAV: {exc}"
        ) from exc
    return int(data.shape[0]), int(sample_rate)


# ---------------------------------------------------------------------------
# Process-wide service
# ---------------------------------------------------------------------------

#: The process-wide service and the configuration signature it was built from.
#: Held in a mutable mapping rather than two module globals so the accessors need
#: no ``global`` statement: rebinding a name is easy to get wrong under a lock,
#: and a module-level variable that is only ever written inside a function reads
#: as unused to static analysis.
_service_state: dict[str, Any] = {"service": None, "signature": None}
_service_lock = threading.Lock()


def _signature() -> tuple[Any, ...]:
    return (
        str(getattr(settings, "clone_store_dir", "")),
        str(getattr(settings, "clone_signing_key_path", "")),
        str(getattr(settings, "clone_reference_key_path", "")),
        str(getattr(settings, "clone_approvers_path", "")),
        str(getattr(settings, "clone_instance_id", "")),
        str(getattr(settings, "chatterbox_variant", "standard")),
    )


def build_service() -> VoiceCloneService:
    """Construct a service from ``Settings`` without caching it."""
    identity = build_identity(settings)
    reference_key: bytes | None = None
    key_path = str(getattr(settings, "clone_reference_key_path", "") or "")
    if key_path:
        try:
            reference_key = clone_crypto.load_reference_key(key_path)
        except KeyMaterialError as exc:
            logger.warning("Voice cloning UNAVAILABLE (reference key): %s", exc)
    return VoiceCloneService(
        store=CloneStore(_resolve_path(settings.clone_store_dir), reference_key=reference_key),
        identity=identity,
        reference_key=reference_key,
        approvers_path=_resolve_path(settings.clone_approvers_path),
        default_variant=str(getattr(settings, "chatterbox_variant", "standard")),
    )


def get_clone_service() -> VoiceCloneService:
    """The process-wide service, rebuilt when the configured paths change."""
    signature = _signature()
    with _service_lock:
        if _service_state["service"] is None or _service_state["signature"] != signature:
            _service_state["service"] = build_service()
            _service_state["signature"] = signature
        return _service_state["service"]


def reset_clone_service(service: VoiceCloneService | None = None) -> VoiceCloneService:
    """Replace the process-wide service (tests, and operators clearing state)."""
    with _service_lock:
        _service_state["service"] = service if service is not None else build_service()
        # Record the current configuration signature even for an injected service:
        # otherwise the very next get_clone_service() compares None against the
        # signature and silently rebuilds, discarding the caller's backend/clock.
        _service_state["signature"] = _signature()
        return _service_state["service"]


def log_startup_status() -> CloneAvailability:
    """Log, once at startup, whether cloning is usable and why not.

    An operator who deploys without the approver config gets a clear warning
    instead of a silent 503 on the first clone request.
    """
    try:
        state = get_clone_service().availability()
    except Exception as exc:  # pragma: no cover - never block boot
        logger.warning("Could not assess voice-cloning availability: %s", exc)
        return CloneAvailability(available=False, reason=str(exc))
    if state.available:
        logger.info(
            "Voice cloning available: %d configured approver(s), instance-signed records.",
            state.approver_count,
        )
    else:
        logger.warning(
            "Voice cloning UNAVAILABLE: %s. The service is running; clone creation "
            "returns 503 until the deployment supplies the missing configuration. "
            "See docs/voices/voice-cloning.md.",
            state.reason,
        )
    return state


__all__ = [
    "CONSENT_NOTICE",
    "ApproverNotAllowedError",
    "CloneAudioError",
    "CloneAvailability",
    "CloneBackend",
    "CloneBackendUnavailable",
    "CloneInactiveError",
    "CloneNotFoundError",
    "CloneSynthesisContext",
    "CloneValidationError",
    "CloningUnavailableError",
    "InstanceIdentity",
    "VoiceCloneService",
    "availability_of",
    "build_identity",
    "build_service",
    "get_clone_service",
    "load_approvers",
    "log_startup_status",
    "reset_clone_service",
    "set_clone_backend_resolver",
]
