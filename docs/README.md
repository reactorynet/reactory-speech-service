# Reactory Speech Service — Documentation

Local, open-source text-to-speech and speech-to-text for the Reactory platform.

This folder holds the **product direction, architecture decisions and delivery roadmap** for the
speech service. It is the reference for every ticket raised against this repository.

## Direction (non-negotiable)

1. **Open source first.** Every default and reference capability is provided by permissively
   licensed models (Apache-2.0, MIT, BSD) running on our own infrastructure. No paid API is
   required to get a complete, good-sounding, multilingual speech stack.
2. **Edge is the baseline.** The reference deployment must run on a developer laptop and a single
   small node with no GPU. CPU-friendly models and ONNX runtimes are preferred over GPU-only
   models with better benchmarks.
3. **Hosted providers are optional.** OpenAI / ElevenLabs / Azure / Cartesia may be added as an
   adapter for a premium tier. They are never the default, never required, and never the only
   path to a capability.
4. **Models are replaceable.** Voice identity, prosody and engine choice sit behind an adapter
   boundary and a data-driven voice registry. Swapping a model must not change the public API.
5. **Cost stays low and predictable.** Per-character billing and vendor lock-in are treated as
   defects, not as acceptable trade-offs.

## Documents

| Document | Purpose |
|---|---|
| [`adr/0001-open-source-edge-first-speech-platform.md`](adr/0001-open-source-edge-first-speech-platform.md) | The decision that governs every later choice |
| [`adr/0002-engine-adapter-and-voice-registry.md`](adr/0002-engine-adapter-and-voice-registry.md) | How voices, prosody and engines are modelled |
| [`adr/0003-consent-first-voice-cloning.md`](adr/0003-consent-first-voice-cloning.md) | Consent, retention and provenance rules for cloning — **Accepted, unblocks Phase 5** |
| [`architecture/speech-platform.md`](architecture/speech-platform.md) | Target architecture, contracts and capability matrix |
| [`architecture/evaluation-harness.md`](architecture/evaluation-harness.md) | Offline evaluation harness, golden corpus, baseline and routing policy |
| [`roadmap/voice-expression-roadmap.md`](roadmap/voice-expression-roadmap.md) | Phased plan and per-ticket scope |
| [`voices/kokoro-voice-catalogue.md`](voices/kokoro-voice-catalogue.md) | The 54 shipped voices and how to design new ones |
| [`voices/designed-voices.md`](voices/designed-voices.md) | Blending rules, the four shipped presets, previews and persistence |

## Tickets

Each phase maps to one support ticket. The `SPEECH-nnn` label is the stable planning id; the
reference is the immutable ticket key in the support system.

| Phase | Planning id | Scope | State | Ticket reference |
|---|---|---|---|---|
| 0 | `SPEECH-001` | Unlock the 54-voice catalogue, expose a voice API, fix schema/header defects | In progress | `REACTORY-302016408/20261110/2432217528` |
| 1 | `SPEECH-002` | Engine adapter layer, direction model, capabilities API | Backlog | `REACTORY-302016408/20261110/1663424173` |
| 2 | `SPEECH-003` | Voice design by style-vector blending + presets + previews | In progress | `REACTORY-302016408/20261110/1712540096` |
| 3 | `SPEECH-004` | Non-blocking inference, per-engine concurrency, synthesis cache, engine lifecycle | Backlog | `REACTORY-302016408/20261110/3808700412` |
| 4 | `SPEECH-005` | Expressive engine adapter (Chatterbox, MIT) for emotion and paralinguistic tags | Backlog | `REACTORY-302016408/20261110/4004237369` |
| 5 | `SPEECH-006` | Consent-first voice cloning with watermarking and provenance | Blocked on policy | `REACTORY-302016408/20261110/166562983` |
| 6 | `SPEECH-007` | Evaluation harness, golden corpus and routing policy | Backlog | `REACTORY-302016408/20261110/1252552414` |
| 7 | `SPEECH-008` | STT streaming, word timestamps, temp-file cleanup, model tiering | Backlog | `REACTORY-302016408/20261110/2269269873` |
| 8 | `SPEECH-009` | Optional hosted premium adapter with cost guardrails | Deferred | `REACTORY-302016408/20261110/1925583329` |

## Quick orientation

```
app/
├── main.py          FastAPI app and model lifecycle
├── config.py        pydantic-settings configuration
├── routers/         health, tts, stt HTTP + WebSocket endpoints
├── services/        tts_service (Kokoro), stt_service (faster-whisper), phonetic_converter
└── models/          pydantic request/response schemas
```

Today `TTSService` is a Kokoro wrapper, and `KOKORO_VOICES` is a hard-coded allow-list of 11 of the
54 voices shipped in `models/voices-v1.0.bin`. Phase 0 removes that allow-list; Phase 1 makes
Kokoro one adapter among several.
