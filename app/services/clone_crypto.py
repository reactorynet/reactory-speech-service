"""Key material and authenticated encryption for consent-first voice cloning.

ADR-0003 decisions 2, 3 and 4a need three cryptographic operations and nothing
else:

* a **signature** over a canonical JSON payload, so a consent record or a
  provenance manifest proves *which instance* wrote it and that it is unaltered;
* **authenticated encryption** for reference audio (and the derived embedding),
  under a key that is separate from ordinary application data;
* a stable **instance identity** derived from the signing key.

Ed25519 signatures and AES-256-GCM are used directly from ``cryptography``
(Apache-2.0/BSD-3-Clause).  No cipher is implemented here.

Two properties matter more than the primitives:

* **Nothing is generated at boot.**  A regenerated signing key silently
  invalidates every existing consent signature and manifest, so an absent key is
  a configuration error that makes cloning unavailable, never a reason to
  generate one.  :func:`load_signing_key` raises
  :class:`KeyMaterialError` with the path and the fix.
* **Signing and encryption keys are separate files.**  A database or backup that
  leaks application data must not also expose voice clips.

What a signature proves is deliberately narrow: it proves the named instance
produced this payload and that the payload has not changed.  It does **not**
prove that the subject consented, and no caller may describe it that way.
"""

from __future__ import annotations

import base64
import hashlib
import os
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

#: AES-256 requires a 32-byte key.
REFERENCE_KEY_BYTES = 32

#: AES-GCM nonce length recommended by NIST SP 800-38D for random nonces.
NONCE_BYTES = 12

#: Leading bytes of an encrypted blob, so a wrong-key read fails loudly rather
#: than being interpreted as audio.
ENCRYPTION_MAGIC = b"RSPEECH1"


class KeyMaterialError(RuntimeError):
    """A required signing or encryption key is absent or unusable.

    This is a **configuration** error, not a request error: the caller must
    supply the key (or accept that cloning is unavailable).
    """


def _read_bytes(path: str | Path) -> bytes:
    candidate = Path(path)
    try:
        return candidate.read_bytes()
    except FileNotFoundError as exc:
        raise KeyMaterialError(
            f"key material not found at {candidate}. Supply it with the deployment and "
            "never commit it; see docs/voices/voice-cloning.md."
        ) from exc
    except OSError as exc:  # pragma: no cover - permission errors are environmental
        raise KeyMaterialError(f"key material at {candidate} could not be read: {exc}") from exc


def load_signing_key(path: str | Path) -> Ed25519PrivateKey:
    """Load the instance signing key from a PEM file.

    Detection of a missing key is what makes cloning **fail closed**: the
    service keeps running, but :func:`app.services.voice_cloning.availability`
    reports why cloning is unavailable.
    """
    raw = _read_bytes(path)
    try:
        key = serialization.load_pem_private_key(raw, password=None)
    except (ValueError, TypeError) as exc:
        raise KeyMaterialError(
            f"signing key at {path} is not an unencrypted PEM private key: {exc}"
        ) from exc
    if not isinstance(key, Ed25519PrivateKey):
        raise KeyMaterialError(
            f"signing key at {path} is {type(key).__name__}, but cloning requires Ed25519"
        )
    return key


def load_reference_key(path: str | Path) -> bytes:
    """Load the reference-audio encryption key.

    Accepted forms are 32 raw bytes or a base64/hex encoding of them.  The key
    is deliberately a *different file* from the signing key (ADR-0003 #3).
    """
    raw = _read_bytes(path)
    # Raw bytes first, and *without* stripping: a random 32-byte key can begin or
    # end with a whitespace byte, and trimming it would silently change the key.
    candidates: list[bytes] = []
    if len(raw) == REFERENCE_KEY_BYTES:
        candidates.append(raw)
    text = raw.strip()
    for decoder in (base64.b64decode, bytes.fromhex):
        try:
            decoded = decoder(text.decode("ascii"))
        except (ValueError, UnicodeDecodeError):
            continue
        if len(decoded) == REFERENCE_KEY_BYTES:
            candidates.append(decoded)
    if not candidates:
        raise KeyMaterialError(
            f"reference-audio key at {path} must be {REFERENCE_KEY_BYTES} bytes "
            "(raw, base64 or hex). Generate one with scripts/generate_clone_keys.py."
        )
    return candidates[0]


def public_key_for(private_key: Ed25519PrivateKey) -> Ed25519PublicKey:
    return private_key.public_key()


def fingerprint(private_key: Ed25519PrivateKey) -> str:
    """A short, stable identifier for the instance key (never the key itself)."""
    raw = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return hashlib.sha256(raw).hexdigest()[:16]


def sign(private_key: Ed25519PrivateKey, payload: bytes) -> str:
    """Sign raw bytes, returning URL-safe base64 without padding."""
    return base64.urlsafe_b64encode(private_key.sign(payload)).decode("ascii").rstrip("=")


def verify(public_key: Ed25519PublicKey, payload: bytes, signature: str) -> bool:
    """Verify a base64 signature over ``payload``.

    A malformed signature is a failed verification, never an exception: the
    callers are verification endpoints and startup checks.
    """
    try:
        padding = "=" * (-len(signature) % 4)
        raw = base64.urlsafe_b64decode(signature + padding)
        public_key.verify(raw, payload)
        return True
    except (InvalidSignature, ValueError, TypeError):
        return False


def encrypt(key: bytes, plaintext: bytes, *, aad: bytes = b"") -> bytes:
    """AES-256-GCM encrypt, returning ``MAGIC | nonce | ciphertext``."""
    if len(key) != REFERENCE_KEY_BYTES:
        raise KeyMaterialError(
            f"reference-audio key must be {REFERENCE_KEY_BYTES} bytes, got {len(key)}"
        )
    nonce = os.urandom(NONCE_BYTES)
    sealed = AESGCM(key).encrypt(nonce, plaintext, aad)
    return ENCRYPTION_MAGIC + nonce + sealed


def decrypt(key: bytes, blob: bytes, *, aad: bytes = b"") -> bytes:
    """Reverse :func:`encrypt`; raises :class:`KeyMaterialError` on tampering."""
    if not blob.startswith(ENCRYPTION_MAGIC):
        raise KeyMaterialError("encrypted blob has no recognised header")
    nonce = blob[len(ENCRYPTION_MAGIC) : len(ENCRYPTION_MAGIC) + NONCE_BYTES]
    sealed = blob[len(ENCRYPTION_MAGIC) + NONCE_BYTES :]
    try:
        return AESGCM(key).decrypt(nonce, sealed, aad)
    except Exception as exc:  # cryptography raises InvalidTag, a bare Exception subclass
        raise KeyMaterialError(
            "encrypted reference data could not be decrypted: the reference key is "
            "wrong or the file was altered"
        ) from exc


def generate_signing_key_pem() -> bytes:
    """Generate a new Ed25519 private key as PEM (used by the operator script).

    Never called at boot: see the module docstring.
    """
    key = Ed25519PrivateKey.generate()
    return key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


def generate_reference_key() -> bytes:
    """Generate 32 random bytes for reference-audio encryption."""
    return os.urandom(REFERENCE_KEY_BYTES)


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


__all__ = [
    "ENCRYPTION_MAGIC",
    "KeyMaterialError",
    "NONCE_BYTES",
    "REFERENCE_KEY_BYTES",
    "decrypt",
    "encrypt",
    "fingerprint",
    "generate_reference_key",
    "generate_signing_key_pem",
    "load_reference_key",
    "load_signing_key",
    "public_key_for",
    "sha256_hex",
    "sign",
    "verify",
]
