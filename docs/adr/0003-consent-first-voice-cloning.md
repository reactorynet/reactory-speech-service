# ADR-0003 — Consent-first voice cloning with provenance and withdrawal

- **Status:** **Accepted** — decisions recorded 2026-11-12. Unblocks SPEECH-006 subject to the implementation constraints below.
- **Date:** 2026-11-12
- **Deciders:** Reactory platform owners
- **Supersedes:** none
- **Unblocks:** SPEECH-006 (`REACTORY-302016408/20261110/166562983`)
- **Constrained by:** ADR-0001 (open source, self-hosted, edge-first; permissive licences only)

## Context

Zero-shot voice cloning is now easy enough to be a checkbox in a speech library. ADR-0001 already
committed us to permissively licensed, self-hosted engines, and two candidates can clone from a
short reference clip on CPU: Chatterbox (MIT) and Orpheus (Apache-2.0). The engines are not the
obstacle. Governance is.

The capability creates three exposures our current architecture cannot answer:

1. **Consent.** A clip of someone speaking is enough to synthesise new speech in their voice. The
   subject is not a party to our system, cannot revoke anything, and may never learn their voice was
   used. This is unlike every other asset we process, where the data subject is the account holder.

2. **Provenance.** Once a clone exists, audio generated from it is indistinguishable from that person
   speaking. If we cannot show that a given file came from our system and which consent authorised
   it, we cannot answer the first question anyone asks when something goes wrong: *"did you make
   this?"*

3. **Irreversible retention.** Reference audio is biometric-adjacent data. It cannot be reissued,
   rotated, or re-derived if leaked. A password breach is recoverable; a voice-clip breach is not.

The service already treats voices as registry records with a `style_spec` naming base voices or
blends. Cloning adds a third style type — `reference` — whose content is supplied by a caller rather
than shipped with the model. That single change moves the service from *"redistributing a public
model's voices"* to *"reusing a private individual's voice"*. Everything below exists because of
that shift.

### Scope boundary: this service does not model identity

The service does **not** know who is a customer, who is an operator, or who holds which privilege.
Identity, roles and authorisation are externally managed, and the service cannot distinguish them.
That decision shapes each control below: the service **records** what it is told and **enforces only
what it can observe locally**. Where a rule requires knowing who someone is, the enforcement point is
an external gateway, and this ADR says so explicitly rather than pretending the service enforces it.

## Decision

### 1. Approval authority is configuration, not code

The set of people who may approve a clone is a **configuration value**, not a hard-coded list and not
a database table the service owns:

- A service config file supplies `clone_approvers: [<email>, ...]` — an **array of email addresses**.
- This file is **deployed with each service instance and MUST NOT be committed to git.** It joins the
  existing set of per-deployment settings (model paths, keys) rather than the repository.
- The service validates at approval time that the supplied approver email is in the configured array.
  An approver not in the list is a refusal, not a warning.
- **Self-approval is permitted.** An operator cloning their own voice supplies their own address; if
  it is in the configured array the record is valid. There is no separate "self" code path — the
  same check applies, so self-approval is a configured permission rather than a special case.
- An empty or absent approver config makes cloning **unavailable**, not unrestricted. Fail closed.

Because identity is external (see scope boundary), the approver field is a **recorded fact the caller
asserts** and the service checks against local configuration. The service cannot verify that the
named person actually consented — only that they are permitted to approve.

### 2. Consent evidence is co-located with the voice and signed by the instance

- The consent record is stored **alongside the generated voice**, not in a separate audit system the
  service cannot reach. If the clone artefact is exported or migrated, its authorising record moves
  with it.
- Each record is **signed with the service instance's key at the time of approval**. That signature
  gives **provenance of the server instance**: it proves *which instance* created this record and
  that the record has not been altered since.
- Required fields: `subjectName`, `purpose`, `capturedAt`, `expiresAt`, `approver`, `evidence`
  (reference to the consent artefact), `scope`, and the instance signature.

**What this proves and what it does not.** It is a chain-of-custody control: it establishes that a
specific instance recorded a specific consent at a specific time. It does **not** prove the subject
consented, and it does not attach to audio after the fact — see the watermark decision below. A
signed record is only useful when you also hold the file it authorises.

### 3. Reference audio is transient: deleted after cloning, 90-day hard ceiling

- **Default and preferred behaviour: the reference clip is deleted as soon as embedding extraction
  completes.** A clone is a derived embedding, not a retained recording. This is the intended state
  for every clone.
- If an implementation needs to retain the reference — for re-derivation, quality debugging, or a
  manual review step — retention **must not exceed 90 days**, after which it is deleted
  automatically, independent of consent expiry. The 90-day ceiling is a hard limit, not a default to
  be tuned upward.
- Retention beyond immediate deletion must be recorded on the consent record with a reason, so it is
  a visible exception rather than an invisible setting.
- Deletion removes the reference audio, and must succeed when the file is already missing — a corrupt
  or absent clip must not block removal of the reference to it.
- Reference audio is encrypted under a key **separate from ordinary application data**, so a database
  compromise does not also expose voice clips.

### 4. Watermarking is a development diagnostic only — no watermark in production audio

**Production audio carries no watermark.** Watermarking is enabled only in development environments,
where it exists as a diagnostic to confirm the synthesis path is the one you think it is.

This was decided deliberately, and it **removes the technical provenance control** that an earlier
draft of this ADR proposed making mandatory. The consequence is recorded here rather than left
implicit:

- There is **no technical marker in shipped audio** tying a file back to this service. A leaked or
  disputed file cannot be identified as ours by inspecting it.
- Provenance therefore rests entirely on the signed consent record (decision 2), which can only be
  matched to a file when you already hold the file's hash or the file itself.
- The claim we can make is narrow and should be stated as such: *"if you give us this file and a
  reason to check, we can look for the authorising record."* We cannot claim *"we can prove any file
  came from us."*

This is an accepted risk, recorded explicitly. If the risk posture changes — for example a customer
contract requires detectable provenance — the decision to revisit is this section, not a hidden
config flag.

### 5. Takedown is a privileged external operation

- The service exposes a takedown operation (disable a clone, delete its artefacts) and **executes it
  on request**.
- **Only a privileged account may invoke it.** Because the service cannot model privilege (see scope
  boundary), that authorisation is enforced **externally**, at the gateway or calling system that
  fronts this service. The service trusts the assertion it is given.
- The service must not be exposed directly to untrusted callers for this operation. It is an internal
  endpoint reached through an authorising layer.
- The **takedown SLA is owned by external systems.** The service's obligation is bounded and
  testable: once a takedown has been applied, the voice is inert **immediately and without a
  redeploy** — `404`/`410`, absent from `/voices`, in-flight requests fail closed.
- Every takedown records who requested it (the assertion it was given), when, and why. The service
  records the claim; it does not verify the authority behind it.

### 6. Identity and scope are external; the service records and does not adjudicate

- The service does not model customer, operator, tenant or role. Nothing in the clone path branches
  on who the caller is.
- The `scope` field (`internal-only`, `customer-facing-approved`, `demo`, `sandbox`) is **recorded,
  not enforced by the service**, because enforcing it would require knowing who is asking. Surfaces
  that render cloned audio are responsible for honouring the scope they are given.
- An expired consent makes the clone inert by the **same mechanism** as takedown: checked at request
  time, not swept by a background job, because a job that stops running must not silently re-enable
  its voices.

## Consequences

**What becomes possible:** a clone with a defensible audit trail — who approved it, on which
instance, under which stated purpose and scope, and when it expires. That is enough to answer "do
you have a record for this voice?" and to act on a takedown quickly.

**What we cannot claim:** that shipped audio is detectably ours, or that we verified the subject's
consent. Both rest on external processes and signed records rather than on the audio itself. Any
customer-facing statement about provenance must be written to that limit.

**What becomes slower:** creating a usable clone needs an approver from a deployed config plus a
complete consent record before any synthesis. That friction is the control.

**What we are choosing not to build:** bulk cloning. There is deliberately no endpoint that creates
many clones from a manifest, because the plausible use for that is the one we least want to enable.

**Residual risk we accept:** a determined party with enough reference audio and a local model can
clone anyone, with or without this service. Nothing here prevents that. What it prevents is *our
service* becoming the convenient, auditable-looking path.

## Rejected alternatives

**Approver list in code or in the database.** A committed list is public and stale on redeploy; a
database table puts approval authority inside the system whose authority is being checked. A
deployed config file separates them.

**Retaining reference audio by default.** Default retention turns a bounded consent into an
indefinite biometric store. Deletion after extraction is the default because the embedding is what
the clone needs.

**Mandatory production watermarking.** Rejected by the platform owner: production audio must be
clean. Recorded above as an accepted risk with its consequence stated, rather than implemented
half-heartedly or silently dropped.

**Service-side role checks.** The service cannot distinguish a customer from an operator, so a
role check inside the service would be a check against nothing. Authorisation belongs at the gateway.

**Ship cloning first, add governance later.** Not possible in the promised order: the first clone
would exist without a consent record, and the retrofit would have to decide what to do about clones
already in use.

## Verification plan

These make the invariants measurable rather than aspirational:

1. An approver **not** in the deployed config is refused; one **in** the config is accepted; an
   **absent/empty** config makes cloning unavailable rather than unrestricted.
2. A missing field returns `422` naming each field individually — a test per field.
3. An `expiresAt` in the past is refused at creation; an expired clone is inert at synthesis.
4. The consent record is stored alongside the voice artefact and carries an instance signature;
   a test asserts the signature verifies against the instance key and that tampering invalidates it.
5. Reference audio is **absent from disk** after a successful clone by default; with retention
   enabled it is present and removed at the 90-day boundary (clock-injected test, not a real wait).
6. Deletion succeeds when the reference file is already missing.
7. A disabled clone returns `404`/`410`, is absent from `/voices`, and an in-flight request fails
   closed — tested without a service restart.
8. Synthesising with a clone works through the ordinary `/synthesize` path with no special case.
9. **No watermark is present in production-mode output** — asserted positively, so the decision is
   enforced rather than assumed. A dev-mode test asserts the watermark *is* present.
10. Audit entries exist for create, use and takedown; a synthesis entry contains the character count
    and **does not** contain the text (asserted, not reviewed by eye).
11. Startup with a corrupt consent store degrades to "no clones" rather than failing to boot — the
    same tolerance the designed-voice store already has.

## Implementation notes for SPEECH-006

- The deployed approver config joins the existing per-instance settings; it must be added to
  `.gitignore` and documented in the operations section so an operator knows to supply it. A clone
  endpoint that silently cannot approve anyone is a support incident waiting to happen.
- Because reference audio is deleted after extraction by default, the clone must be re-derivable
  only from the embedding. Do not design a "re-clone from reference" flow that assumes the clip is
  still there.
- The takedown endpoint must be documented as **internal-only, behind the authorising gateway**. An
  exposed takedown is worse than no takedown.
- Engine choice is now unconstrained by watermarking. Chatterbox (MIT) and Orpheus (Apache-2.0) both
  remain acceptable on licence grounds; the remaining question is CPU feasibility, which SPEECH-006
  must measure rather than assume.
