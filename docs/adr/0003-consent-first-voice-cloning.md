# ADR-0003 — Consent-first voice cloning with watermarking and provenance

- **Status:** **Proposed** — blocks SPEECH-006. Not accepted until the open questions at the end are answered by the platform owner.
- **Date:** 2026-11-12
- **Deciders:** Reactory platform owners; requires a named human approver per voice
- **Supersedes:** none
- **Blocks:** SPEECH-006 (`REACTORY-302016408/20261110/166562983`)
- **Constrained by:** ADR-0001 (open source, self-hosted, edge-first; permissive licences only)

## Context

Zero-shot voice cloning is now easy enough to be a checkbox in a speech library. ADR-0001 already
committed us to permissively licensed, self-hosted engines, and two candidates can clone from a
short reference clip on CPU: Chatterbox (MIT) and Orpheus (Apache-2.0). The engines are not the
obstacle. Governance is.

The capability creates three distinct exposures that our current architecture cannot answer:

1. **Consent.** A clip of someone speaking is enough to synthesise new speech in their voice. The
   subject is not a party to our system, cannot revoke anything, and may never learn their voice was
   used. This is unlike every other asset we process, where the data subject is the account holder.

2. **Provenance and deniability.** Once a clone exists, audio generated from it is
   indistinguishable from that person speaking — absent watermarking. If we cannot show that a
   given file came from our system, or which consent record authorised it, we cannot answer the
   first question anyone will ask when something goes wrong: *"did you make this?"*

3. **Irreversible retention.** Reference audio is biometric-adjacent data. It cannot be reissued,
   rotated, or re-derived if leaked. A password breach is recoverable; a voice-clip breach is not.

The existing service already treats voices as registry records with a `style_spec` that names base
voices or blends of them. Cloning adds a third style type — `reference` — whose content is
supplied by a customer rather than shipped with the model. That single change moves the service
from "redistributing a public model's voices" to "storing and reusing a private individual's
voice". Everything below exists because of that shift.

### What this ADR is not

It does not decide *whether* to build cloning. It defines the conditions under which the existing
SPEECH-006 scope may start, and the invariants the implementation must satisfy. If the conditions
cannot be met, the correct outcome is to leave SPEECH-006 blocked and say so, rather than to
implement it with the guardrails bolted on afterwards.

## Decision

Voice cloning may be implemented **only** as consent-first: a clone cannot exist without a consent
record, cannot be used without a watermark, and cannot outlive its consent. The four invariants:

### 1. A clone cannot be created without a complete consent record

`POST /api/tts/voices/clone` accepts multipart reference audio **and** a structured consent record.
Incomplete input returns `422` naming each missing field. Required fields:

| Field | Purpose |
|---|---|
| `subjectName` | identifies the natural person whose voice this is |
| `purpose` | the specific use this clone is authorised for, in free text |
| `capturedAt` | when consent was obtained |
| `expiresAt` | when the consent lapses; a clone past this date is inert |
| `approver` | the named human who accepted the consent, as a user account id |
| `evidence` | a reference to the consent artefact (signed form id, uploaded document id) |
| `scope` | surfaces the clone may appear in — not a free-text promise but an enumeration |

**Self-consent is allowed and must be recorded as such.** An operator cloning their own voice still
supplies a record with `subjectName` matching their own identity; the difference is auditable rather
than implicit. There is no code path that skips the record.

The API must refuse, not warn, on: missing fields, `expiresAt` in the past, an `approver` that is
not an active user, and a `scope` value outside the enumeration.

### 2. Reference audio is stored encrypted, retained for a stated period, and deletable

- Reference audio is encrypted at rest with a key that is **not** the same key protecting ordinary
  application data, so that a database compromise does not also expose voice clips.
- Retention is bounded and stated in the consent record. The default is the consent expiry; a
  shorter retention is allowed, a longer one requires a new consent record.
- `DELETE /api/tts/voices/{id}` on a cloned voice removes, in one transaction or with a recorded
  partial-failure state: the reference audio, the cached embedding, the registry entry, and any
  generated preview. Deletion must **not** require the audio to be readable first — a corrupt or
  missing clip must not block removal of the reference to it.
- A deletion is soft-deleted in the audit log (who, when, why) while the audio itself is
  hard-deleted. The audit record must not contain the audio or a hash that could be used to match it.

### 3. Watermarking is on by default and cannot be silently disabled

- Where the engine supports it, synthesis from a `reference`-style voice applies an imperceptible
  watermark by default.
- Disabling it requires an explicit per-deployment setting **and** a per-request justification
  recorded in the audit log. It is never a silent default and never a query parameter a caller can
  simply set.
- If an engine used for cloning cannot watermark, that engine must not be used for cloning. The
  watermark requirement constrains engine choice; it does not bend to it.
- We must be able to demonstrate the watermark exists on a generated file. An "on by default" claim
  with no verification is a policy statement, not a control.

### 4. Every use is attributable, and every voice is withdrawable

- The registry entry for a cloned voice carries: engine, licence, consent record id, `createdAt`,
  `createdBy`, `approvedBy`. `GET /api/tts/voices/{id}` surfaces all of it.
- An audit log entry is written for **creation, every synthesis, and deletion**. Synthesis entries
  record the voice id, consent id and character count — never the text.
- **Takedown is a registry operation, not a deployment.** Disabling a cloned voice takes effect
  without a redeploy: a disabled voice returns `404` or `410`, is absent from `/voices`, and any
  in-flight request fails closed. This mirrors the pack-voice immutability already enforced
  (deleting a pack voice returns `409`).
- An expired consent makes the clone inert by the same mechanism as takedown. Expiry is checked at
  request time, not swept by a background job, because a job that stops running must not silently
  re-enable its voices.

### Scope of the enumeration (needed by invariant 1)

`scope` values are deliberately not free text, because "authorised for use" is unenforceable if each
record phrases it differently. Proposed initial set: `internal-only`, `customer-facing-approved`,
`demo`, `sandbox`. Any surface that renders cloned audio must declare one, and a mismatch between
the surface's declared scope and the clone's permitted scope is a refusal.

## Consequences

**What becomes possible:** a clone that can be defended. If a generated file is questioned, we can
produce the consent record, the approver, the watermark, and the synthesis audit entry. That is the
difference between a capability we can ship and one we cannot.

**What becomes slower:** cloning is not a one-call convenience. Creating a usable clone requires a
consent artefact, an approver, and a scope decision before any audio is synthesised. That friction
is the control — removing it is the same as removing the control.

**What becomes constrained:** engine choice is narrowed to engines that watermark. If neither
Chatterbox nor Orpheus reliably watermarks on CPU, SPEECH-006 has no compliant engine and must not
ship until one exists. This must be verified, not assumed.

**What we are choosing to make hard:** bulk cloning. The design deliberately has no endpoint that
creates many clones from a manifest, because the plausible use case for that is the one we least
want to enable.

**Residual risk we accept:** a determined party with enough reference audio and a local model can
clone anyone, with or without this service. Nothing here prevents that. What it prevents is *our
service* becoming the convenient, auditable-looking path to do it. We should not claim more than
that.

## Rejected alternatives

**Rely on terms of service alone.** A prohibition without a technical control is a statement of
intent, not a safeguard. Every parameter is optional in practice if the API does not refuse.

**Watermarking as an option, defaulting off.** Watermarking that can be quietly turned off protects
nobody, and cannot be cited as a mitigation after an incident.

**Unbounded retention of reference audio.** Keeping clips indefinitely "for quality" converts a
bounded consent into an indefinite biometric store. The quality argument does not survive the
breach scenario.

**Cloning via a hosted provider.** Violates ADR-0001, and moves the consent problem to a third party
who has no relationship with the subject.

**Ship cloning first, add governance later.** Not possible in the order promised: the first cloned
voice would exist without a consent record, and the retrofit would have to decide what to do about
clones already in use. There is no clean migration from "ungoverned" to "governed" for this data.

## Verification plan

These are the checks that make the invariants measurable rather than aspirational:

1. `422` for each individually missing consent field; a test per field, not one combined test.
2. `422` for `expiresAt` in the past, an inactive `approver`, and an out-of-enumeration `scope`.
3. A clone synthesised through the ordinary `/synthesize` path with no special case in the API.
4. Deletion removes audio, embedding, registry entry and preview; a test that asserts the audio file
   is gone from disk, and one where the audio file is already missing (deletion must still succeed).
5. Watermark present on generated audio, verified by an actual detector or extractor, with a test
   that fails if watermarking is disabled without the recorded justification.
6. A disabled clone returns `404`/`410` and is absent from `/voices`; an expired clone does the same.
7. Audit entries exist for create/use/delete; a synthesis audit entry contains the character count
   and **does not** contain the text (asserted, not reviewed by eye).
8. Startup with a corrupt consent store degrades to "no clones" rather than failing to boot — the
   same tolerance the designed-voice store already has.

## Open questions — must be answered before status becomes Accepted

1. **Who is the named human approver**, and is one person sufficient, or is a second reviewer
   required for `customer-facing-approved` scope?
2. **Where does consent evidence live**, and who can audit it? A reference id is only useful if the
   artefact behind it is retained and retrievable.
3. **What is the retention period** for reference audio, and does it differ by scope? The ADR
   proposes consent-expiry as the default ceiling; the owner must confirm or override.
4. **Does the watermark survive** the audio pipeline end-to-end — encode, transcode, and playback on
   the surfaces in `scope`? This is an empirical question that must be answered with the chosen
   engine before SPEECH-006 starts, because it determines whether a compliant engine exists at all.
5. **What is the takedown SLA**, and who executes it? "Without a redeploy" is a technical property;
   it still needs an owner and a time bound.
6. **Does cloning customer reference audio change our data-processing posture enough** to require a
   privacy review or a DPA change? If yes, that review precedes implementation, not follows it.

Until questions 1, 3, 4 and 5 are answered, SPEECH-006 remains blocked regardless of engineering
readiness.
