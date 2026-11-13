# Voice cloning — operator setup

Short operational companion to [`voice-cloning.md`](voice-cloning.md) and
[ADR-0003](../adr/0003-consent-first-voice-cloning.md). That pair explains *why* the controls
exist; this page is the checklist for turning cloning on for an instance.

Cloning is **off by default and off unless fully configured**. An instance that has never been
set up behaves exactly like one from before this feature existed: pack voices and designed voices
synthesise normally, and clone creation returns `503` naming what is missing.

## What to provision

Three paths, all configurable, all gitignored:

| Setting | Default | Purpose |
|---|---|---|
| `clone_approvers_path` | `data/clone_approvers.json` | who may approve a clone |
| `clone_signing_key_path` | `data/keys/clone_signing_key.pem` | Ed25519 key that signs consent records and provenance manifests |
| `clone_reference_key_path` | `data/keys/clone_reference.key` | 32 bytes, AES-256-GCM, for reference audio at rest |

Generate all three in one step:

```bash
python scripts/generate_clone_keys.py --output-dir data
```

That writes the two key files plus a `clone_approvers.json` **template** containing an empty
array. Note the wording the script prints: *add approver emails before cloning is available*. An
empty allowlist is the deliberate default, not an oversight.

Useful flags: `--instance-id` sets the id recorded inside signatures (defaults to the hostname at
runtime), and `--force` overwrites existing keys — which invalidates every consent record and
manifest signature already issued, so it is not a routine operation.

The signing key is deliberately separate from the reference key. A backup or database compromise
that exposes one must not also expose the other: the signing key proves provenance, the reference
key protects biometric-adjacent audio.

## Then name the approvers

Edit `data/clone_approvers.json`:

```json
{ "clone_approvers": ["the.person.who.approves@example.org"] }
```

An absent file, an empty array, or a malformed file all mean **nobody can approve**, which means
cloning is unavailable. That is the intended reading of fail-closed: a configuration omission
removes the capability rather than leaving it unrestricted.

## Never commit these

`.gitignore` already covers `data/clone_approvers.json`, `data/keys/` and `data/clones/`. Do not
work around it. An allowlist in the repository is public and stale; a signing key in the repository
is a compromise.

If the signing key is missing at startup the service does **not** generate one. It reports cloning
as unavailable and carries on serving everything else. That is intentional — a key generated per
boot would silently invalidate the provenance chain on every restart.

## Verifying an instance is configured

`GET /api/health` carries a `cloning` block from `CloneAvailability.as_dict()`:

```
available               true when all three pieces resolved
reason                  why not, when available is false
approverCount           how many emails the allowlist resolved to
signingKeyConfigured    Ed25519 key loaded
referenceKeyConfigured  AES-256-GCM key loaded
```

Check it after provisioning, so a missing allowlist is visible before the first clone request 503s:

```bash
curl -s localhost:8765/api/health | python3 -m json.tool
```

Cloning available does not mean a clone exists. It means the instance *could* accept one, subject
to the consent record.

## What a caller must supply

A clone request is a multipart upload: the reference clip plus every consent field. Missing fields
are reported individually in the `422`, not as a single generic failure, so a caller can fix the
form in one pass. The request is also refused when:

- the approver is not in the deployed allowlist,
- the `scope` is outside the enumeration,
- `expiresAt` is already in the past.

## After a clone exists

- The reference clip is deleted as soon as the speaker embedding has been extracted. Retention is
  an explicit, reasoned exception capped at 90 days, and the deadline is derived from the signed
  consent record rather than from the registry — the registry is not signed, so reading it would
  let an unsigned edit extend retention.
- Expired consent and takedown both take effect **at request time**. There is no background sweep,
  so nothing has to keep running for a withdrawn voice to stay withdrawn, and nothing has to be
  restarted for a change to take effect.

## Provenance lookups

Every synthesis from a cloned voice emits a signed manifest keyed by the SHA-256 of the **decoded
PCM samples**, not of the container file. That means re-wrapping the same audio in a different
container still resolves.

A lookup that finds nothing reports **"no record found"**. It must never be read as "not ours" —
absence of a record is not evidence of absence, because the file may have been transcoded, cropped,
or simply never registered. Support tooling built on this endpoint should preserve that wording.
