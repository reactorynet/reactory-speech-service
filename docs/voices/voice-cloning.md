# Consent-first voice cloning (SPEECH-006)

Cloning reuses a **private individual's voice**. The person whose clip is used is
not a party to this system, cannot revoke anything, and may never learn their
voice was used. That asymmetry is why the controls below are invariants, not
settings. The authoritative specification is
[ADR-0003](../adr/0003-consent-first-voice-cloning.md); this page is the
operations guide.

Product direction: open source, self-hosted, edge-first. Cloning runs locally on
an approved engine. **Chatterbox (MIT)** is the engine wired here. XTTS v2 (CPML)
and F5-TTS (CC-BY-NC) are prohibited by ADR-0001 and are not present.

## What the service can and cannot claim

| Claim | True? |
|---|---|
| "This instance recorded this consent at this time, and the record is unaltered." | **Yes** — Ed25519 signature over the record. |
| "The subject consented." | **No.** The signature is chain of custody, not subject verification. |
| "If you give us this file, we can look for the authorising record." | **Yes** — registry lookup on `audioSha256`. |
| "We can prove any file came from us." | **No.** There is no watermark in production audio. |
| "A file with no record is not ours." | **Never say this.** Absence of a record is not evidence of absence. The only wording is **"no record found"**. |

## Deployment (required before cloning is available)

Cloning fails closed. With any of the files below missing, **clone creation**
returns `503` and `/health` reports why. Ordinary synthesis — pack voices,
designed voices, previews — is unaffected: cloning being unconfigured costs you
cloning and nothing else. A clone-shaped id (`clone_…`) with no record still
fails closed rather than falling back to another voice.

```bash
python scripts/generate_clone_keys.py --output-dir data
# edit data/clone_approvers.json -> {"clone_approvers": ["ops@example.com"]}
```

| Setting | Default | Meaning |
|---|---|---|
| `CLONE_APPROVERS_PATH` | `data/clone_approvers.json` | Deployed allowlist. **Never commit.** Empty or absent => cloning unavailable. |
| `CLONE_SIGNING_KEY_PATH` | `data/keys/clone_signing_key.pem` | Ed25519 key signing consent records and manifests. Never generated at boot. |
| `CLONE_REFERENCE_KEY_PATH` | `data/keys/clone_reference.key` | 32 bytes for AES-256-GCM, separate from ordinary application data. |
| `CLONE_STORE_DIR` | `data/clones` | Consent records, encrypted embeddings, manifest registry, audit log. |
| `CLONE_INSTANCE_ID` | hostname | Recorded inside every signature payload. |

All four paths are in `.gitignore`. A regenerated signing key invalidates every
existing consent record and manifest — rotate deliberately, never implicitly.

## Create a clone

```bash
curl -sS -X POST http://localhost:8765/api/tts/voices/clone \
  -F reference_audio=@vin.wav \
  -F created_by=ops@example.com \
  -F consent='{
        "subjectName": "Vin",
        "purpose": "Internal narration demo",
        "capturedAt": "2026-11-10T09:00:00Z",
        "expiresAt": "2027-11-10T00:00:00Z",
        "approver": "ops@example.com",
        "evidence": "consent-form-2026-11-10.pdf",
        "scope": "internal-only"
      }'
```

Refusals:

* `422` — each missing consent field named individually; an unparseable
  timestamp; `expiresAt` in the past; a `scope` outside
  `internal-only | customer-facing-approved | demo | sandbox`.
* `403` — the approver is not in this deployment's allowlist. There is no
  separate self-approval path: self-approval works only because the same
  allowlist check applies.
* `503` — cloning is not configured on this instance.

`scope` is **recorded, not enforced**. The service cannot tell a customer from an
operator, so there is no role check anywhere in the clone path.

## Reference audio: transient by default

The clip is written only in encrypted form, and deleted as soon as the embedding
is extracted. The 90-day ceiling is enforced from the **signed** consent record and
clamped to `createdAt + 90 days`, so editing `registry.json` cannot extend it; a
retained clip whose deadline cannot be justified from a verified record is deleted
rather than kept. During extraction the engine also writes the clip to a private
mode-0700 temporary directory (the upstream API takes a path); that directory is
removed in a `finally`, so the only lasting copy is the encrypted one in the
store. A hard kill during extraction can leave that temporary file behind, which
is why the directory is private and short-lived rather than the system temp dir. To keep it, send `retain_reference=true` **with**
`retention_reason`; the exception is recorded on the consent record and capped at
**90 days**, independent of consent expiry. Expired references are purged during
request handling, not by a background job.

The clone is re-derivable from the embedding alone. There is deliberately no
"re-clone from reference" flow, because the reference is normally gone.

## Synthesise

Clones use the ordinary endpoint with no special case:

```bash
curl -sS -X POST http://localhost:8765/api/tts/synthesize/json \
  -H 'content-type: application/json' \
  -d '{"text": "Hi Vin, welcome back.", "voice": "clone_vin_1a2b3c4d"}'
```

The JSON body carries `provenance`; the WAV response carries
`X-Provenance-Id` (the registry key) plus `X-Provenance-Lookup`. Cloned voices are
rendered as complete buffers: the streaming endpoint refuses them rather than
streaming a substitute voice.

## Provenance verification

```bash
curl -sS -X POST http://localhost:8765/api/tts/provenance/verify -F file=@maybe-ours.wav
curl -sS http://localhost:8765/api/tts/provenance/<audioSha256>
```

`audioSha256` is over **decoded PCM**, not container bytes: re-wrapping the same
samples in another container still resolves. A lossy transcode changes the
samples and therefore does not — correctly, because it is no longer the file we
made.

The manifest registry is a durable store. Its retention is deliberately **not**
bounded by the 90-day reference ceiling: a manifest records that a *file was
made* and must outlive the voice.

States: `verified`, `signature invalid`, `verification unavailable`,
**`no record found`**. The last is not a statement about ownership.

## Takedown and deletion

`POST /api/tts/voices/{id}/takedown` is **internal-only** and must sit behind the
authorising gateway. The service cannot model privilege; it executes the
assertion it is given and records who, when and why.

```bash
curl -sS -X POST http://localhost:8765/api/tts/voices/clone_vin_1a2b3c4d/takedown \
  -H 'content-type: application/json' \
  -d '{"requestedBy": "gateway:takedown-service", "reason": "subject request"}'
```

Once applied: `410` on synthesis, the voice is absent from `/voices`, and an
in-flight render is discarded — all without a redeploy. An **expired** consent is
inert by the same mechanism, checked at request time. A background sweeper that
stopped running must never silently re-enable a voice.

`registry.json` is **not** re-read from disk: the store loads it at construction
and keeps it in memory, so hand-editing that file does nothing until the process
restarts. Use the takedown endpoint for anything that must take effect now. The
registry file is also unsigned, so it is only a consistency witness — the expiry
that authorises synthesis is read from the signed consent record, and if the two
disagree the clone is inert.

`DELETE /api/tts/voices/{id}?requestedBy=...&reason=...` removes audio, embedding,
registry entry and any preview. It succeeds when a file is already missing. The
audit entry records who, when and why, and contains no audio and no hash of it.

**This endpoint is internal-only too.** It is exactly as destructive as takedown —
it removes the embedding and the registry entry — and it relies on the same
externally supplied `requestedBy` assertion. Like takedown, it must sit behind the
authorising gateway and must not be exposed to untrusted callers (ADR-0003
decision 5). The service records the assertion; it cannot verify the authority
behind it.

The signed consent record is deliberately **retained** after deletion: it is the
chain of custody for what was done, and removing it would destroy the evidence
the audit log refers to. Deleting or altering it makes the clone inert, because
the record is verified in the request path rather than only when it is read. It contains the subject name, so a retention review of
the store is part of the same data-protection process as the audit log.

## Audit

`CLONE_STORE_DIR/audit.jsonl` is append-only and contains entries for **create**,
**use** and **takedown**/delete. A synthesis entry records the character count and
the engine — **never the text**.

## Dependencies

`cryptography` (Apache-2.0/BSD-3-Clause) is pinned in `requirements.txt`. It
provides AES-256-GCM and Ed25519; no cipher is implemented in this repository.

Engine: Chatterbox (MIT) remains the only cloning engine wired in. Its Perth
watermarker is neutralised before any audio is returned, including cloned audio
(`neutralise_watermarker`). Production output carries no watermark — that is an
accepted risk recorded in ADR-0003 decision 4, not an oversight.

## Known limitations

* **No role checks.** By design; authorisation belongs at the gateway.
* **No bulk cloning.** There is deliberately no endpoint that creates many
  clones from a manifest.
* **No URL or streaming reference sources.** File upload only.
* **No fine-tuning.** Cloning is zero-shot from a clip; training a voice model is
  a separate decision with a separate data-protection review.
