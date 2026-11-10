# ADR-0001 — Open-source, self-hosted, edge-first speech platform

- **Status:** Accepted
- **Date:** 2026-11-10
- **Deciders:** Reactory platform owners
- **Supersedes:** none

## Context

The speech service currently provides TTS through Kokoro-82M (ONNX) and STT through
faster-whisper. It works, runs locally, and costs nothing per request. It also exposes only 11
voices from a pack that contains 54, and offers no control beyond a single `speed` multiplier.

Requirements are expanding: more voices, more languages, control of pace and tone, and eventually
expressive delivery for assistant and support experiences. The obvious shortcut is to adopt a
hosted TTS API, which would deliver tone and voice variety quickly but would introduce:

- per-character cost that grows with usage and cannot be capped by infrastructure planning,
- a hard dependency on a third party for a core product capability,
- voice libraries and model behaviour we cannot inspect, self-host, or run offline,
- a data path where user text leaves our infrastructure by default,
- vendor lock-in for voice identities that customers may build into their products.

At the same time, the open-weight TTS field has matured. Permissively licensed models now cover
the full range we need: compact CPU models for the default path, and expressive models with
zero-shot cloning for a premium path.

## Decision

We build the speech platform **open source, self-hosted and edge-first**.

1. **Permissive licences only, for anything on the default path.**
   Apache-2.0, MIT and BSD are acceptable. Copyleft-restricted or non-commercial weights are not
   used on commercial paths. Concretely: Kokoro (Apache-2.0), Chatterbox (MIT), Orpheus
   (Apache-2.0) and Piper (MIT) are approved; XTTS v2 (CPML) and F5-TTS (CC-BY-NC) are not
   approved for commercial deployment.

2. **CPU and edge are the reference target.**
   The default image must run a full TTS + STT round trip on a developer laptop and on a single
   small node, with no GPU. GPU-only models are allowed only as an opt-in tier.

3. **No paid API is required for any capability.**
   Every capability — voices, languages, pace, tone, streaming, cloning — must have a
   self-hostable implementation. A hosted provider may be added later as an optimisation or a
   premium tier, never as the only implementation.

4. **Voice identity is data, not code.**
   Voices live in a registry with licence, provenance and capability metadata. Adding a voice is
   a data change and must not require a code change or a redeploy of the inference code.

5. **Models are replaceable behind an adapter boundary.**
   The public API describes *what* to say and *how* it should sound, not *which model* renders it.
   Replacing an engine is an adapter change.

6. **Cost is a first-class constraint.**
   Infrastructure cost, image size, memory footprint and time-to-first-audio are tracked per
   engine. A model that cannot run within the edge budget does not become the default, regardless
   of benchmark quality.

## Consequences

### Positive

- No per-character spend; cost scales with hardware we already run, not with usage.
- Text stays on our infrastructure by default, which simplifies privacy and data-handling
  obligations.
- Voice identities we design or clone remain ours; no vendor can withdraw them.
- Offline and air-gapped deployments become possible.
- We can inspect, benchmark and fix model behaviour ourselves.
- The same stack runs on a laptop, in CI and in production, which shortens the debugging loop.

### Negative

- We own the operational burden: model downloads, image size, memory, cold start, upgrades.
- The absolute quality ceiling of a CPU-friendly model is below the best hosted models. This is
  accepted for the default tier and mitigated by an opt-in expressive tier.
- Expressive models bring heavier dependencies (PyTorch) and larger images. They must live in a
  separate deployment profile rather than bloating the default image.
- We must actively track licences as new models appear.

### Risks and mitigations

| Risk | Mitigation |
|---|---|
| A chosen model changes licence | Registry records the licence; CI asserts an allow-list; the adapter boundary makes replacement a contained change |
| Model quality regresses on upgrade | Evaluation harness with a golden corpus and per-engine scores (SPEECH-007) |
| Image size creeps past edge budgets | Separate `core` and `expressive` profiles; size is asserted in CI |
| Voice cloning is misused | Consent records, watermarking and provenance are mandatory before cloning ships (SPEECH-006) |
| Someone adds a hosted dependency to the default path | Hosted adapters are opt-in and disabled by default; `/capabilities` reports what is local |

## Non-goals

- Training a TTS model from scratch.
- Matching the absolute quality of the best hosted model on CPU hardware.
- Supporting every language at launch. Coverage grows per engine, reported by `/capabilities`.

## Alternatives considered

| Alternative | Why rejected |
|---|---|
| Hosted TTS as the primary engine | Per-character cost, vendor lock-in, user text leaving our infrastructure, voices not owned |
| Single large GPU model for everything | Violates the edge baseline; unacceptable default footprint and cost floor |
| Keep Kokoro but hard-code a longer voice list | Solves the symptom, not the cause; no tone control and no path to new engines |
| Build a bespoke model | Years of work, no licence to train on, no advantage over adapting open weights |
