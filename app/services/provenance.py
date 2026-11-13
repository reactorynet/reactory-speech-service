"""Provenance manifests: hash decoded PCM, sign the record, verify by lookup.

ADR-0003 decision 4a is the provenance control that decision 4 declines to place
in the audio.  Three details are load-bearing and are enforced here rather than
by convention:

* ``audioSha256`` is taken over the **decoded PCM samples**, not the container
  bytes, so re-wrapping identical samples in another container still resolves.
* the manifest is signed over every field it claims, so tampering with the audio
  hash, the voice, the consent id, the instance or the timestamp invalidates it.
* an unknown hash is reported as **"no record found"**.  It is never reported as
  "not ours": absence of a record is not evidence of absence (the file may have
  been transcoded, cropped or simply never registered here).
"""

from __future__ import annotations

import hashlib
import io
import json
from typing import Any

import numpy as np
import soundfile as sf

from app.services.clone_crypto import Ed25519PrivateKey, Ed25519PublicKey, sign, verify

#: The fields covered by the manifest signature, in canonical order.
SIGNED_MANIFEST_FIELDS: tuple[str, ...] = (
    "audioSha256",
    "voiceId",
    "consentId",
    "instanceId",
    "createdAt",
    "sampleRate",
    "channels",
)

#: The only wording a negative lookup may use.  Exported so tests and support
#: tooling assert the exact string rather than a paraphrase.
NO_RECORD_STATUS = "no record found"

VERIFIED_STATUS = "verified"
INVALID_SIGNATURE_STATUS = "signature invalid"


def canonical_json(payload: dict[str, Any]) -> bytes:
    """Deterministic bytes for signing (sorted keys, no whitespace)."""
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def decode_pcm(wav_bytes: bytes) -> tuple[np.ndarray, int, int]:
    """Decode WAV bytes to ``(int16 samples, sample_rate, channels)``.

    Integer PCM is canonical: the container's header, metadata chunks and
    wrapping are discarded, so two containers carrying the same samples hash
    identically.
    """
    data, sample_rate = sf.read(io.BytesIO(wav_bytes), dtype="int16", always_2d=True)
    return np.ascontiguousarray(data), int(sample_rate), int(data.shape[1])


def audio_sha256(wav_bytes: bytes) -> str:
    """SHA-256 over the decoded PCM samples of ``wav_bytes``."""
    samples, _sample_rate, _channels = decode_pcm(wav_bytes)
    return hashlib.sha256(samples.tobytes()).hexdigest()


def build_manifest(
    *,
    private_key: Ed25519PrivateKey,
    audio: bytes,
    voice_id: str,
    consent_id: str | None,
    instance_id: str,
    created_at: str,
) -> dict[str, Any]:
    """Build and sign a manifest for one rendered file."""
    samples, sample_rate, channels = decode_pcm(audio)
    payload: dict[str, Any] = {
        "audioSha256": hashlib.sha256(samples.tobytes()).hexdigest(),
        "voiceId": voice_id,
        "consentId": consent_id,
        "instanceId": instance_id,
        "createdAt": created_at,
        "sampleRate": sample_rate,
        "channels": channels,
    }
    manifest = dict(payload)
    manifest["signature"] = sign(private_key, canonical_json(payload))
    return manifest


def manifest_payload(manifest: dict[str, Any]) -> dict[str, Any]:
    """The signed subset of a manifest, as a new dict."""
    return {field: manifest.get(field) for field in SIGNED_MANIFEST_FIELDS}


def verify_manifest(manifest: dict[str, Any], public_key: Ed25519PublicKey) -> bool:
    """Whether the manifest's signature covers its claimed fields."""
    signature = manifest.get("signature")
    if not isinstance(signature, str) or not signature:
        return False
    return verify(public_key, canonical_json(manifest_payload(manifest)), signature)


__all__ = [
    "INVALID_SIGNATURE_STATUS",
    "NO_RECORD_STATUS",
    "SIGNED_MANIFEST_FIELDS",
    "VERIFIED_STATUS",
    "audio_sha256",
    "build_manifest",
    "canonical_json",
    "decode_pcm",
    "manifest_payload",
    "verify_manifest",
]
