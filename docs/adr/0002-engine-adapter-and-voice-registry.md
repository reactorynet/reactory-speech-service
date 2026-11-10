# ADR-0002 — Engine adapter layer, voice registry and direction model

- **Status:** Accepted
- **Date:** 2026-11-10
- **Depends on:** ADR-0001

## Context

`TTSService` currently conflates four separate concerns:

1. **Voice identity** — which speaker the audio sounds like (`KOKORO_VOICES`, a hard-coded dict).
2. **Prosody** — the mechanics of delivery (`speed`, and nothing else).
3. **Style** — the emotional colour of the delivery (not supported at all).
4. **Engine** — the model that renders the audio (Kokoro, hard-wired).

Because these are entangled, every new requirement turns into a change to `tts_service.py`, and
every consumer hard-codes a copy of the voice list. `LocalSpeechProvider` in the server module
already keeps two independent hard-coded voice lists that disagree with the service, and silently
rewrites unknown voices to `af_heart`.

If we add a second TTS model without first separating these concerns, we will duplicate the same
mistake twice.

## Decision

Separate the four concerns, and describe controls in a **provider-neutral** vocabulary that each
engine lowers to its own native parameters.

### 1. Voice registry (data)

Voices are records, loaded at startup from a registry file, writable at runtime, and served by the
API. Each record carries identity, language, licence, provenance, capability flags and a
`style_spec` describing how to render it.

```json
{
  "id": "af_heart",
  "name": "Heart",
  "engine": "kokoro",
  "language": "en-US",
  "gender": "female",
  "tags": ["warm", "narration", "conversational"],
  "quality": "A",
  "license": "Apache-2.0",
  "attribution": "Kokoro-82M v1.0",
  "style_spec": { "type": "base", "voice": "af_heart" },
  "supports": { "rate": true, "pitch": false, "style": false, "clone": false },
  "enabled": true
}
```

`style_spec` has three forms:

- `{"type": "base", "voice": "<engine voice id>"}` — an engine's own voice.
- `{"type": "blend", "components": [{"voice": "af_heart", "weight": 0.65}, ...]}` — a designed
  voice composed from base style vectors (see ADR note below).
- `{"type": "reference", "audio_path": "...", "embedding_path": "..."}` — a cloned voice backed
  by reference audio.

### 2. Direction model (provider-neutral)

```python
class Prosody(BaseModel):
    rate: float | None = 1.0            # 0.5 .. 2.0
    pitch: float | None = 0.0           # semitones, -12 .. 12
    volume: float | None = 0.0          # dB, -12 .. 12
    pauses: Literal["none", "natural", "dramatic"] = "natural"

class Direction(BaseModel):
    style: str | None = "neutral"       # neutral, warm, calm, cheerful, serious,
                                        # empathetic, excited, whisper, narration
    intensity: float | None = 0.5       # 0 .. 1
    prosody: Prosody = Prosody()
```

Callers express intent. They never name a model parameter.

### 3. Engine adapters

One contract per engine:

```python
class TTSEngine(Protocol):
    id: str
    def capabilities(self) -> EngineCapabilities: ...
    def voices(self) -> list[VoiceDescriptor]: ...
    def synthesize(self, req: EngineRequest) -> AudioResult: ...
    def stream(self, req: EngineRequest) -> Iterator[AudioChunk]: ...
```

The orchestrator resolves a voice to an engine, lowers `Direction` into that engine's native
controls, and returns what it applied and what it could not:

```json
{
  "engine": "kokoro",
  "applied": { "rate": 0.95, "pauses": "natural" },
  "ignored": ["pitch", "style"],
  "warnings": ["kokoro does not support emotion; style 'empathetic' was not applied"]
}
```

### 4. Capability negotiation

`GET /api/tts/capabilities` reports, per engine: languages, native rate/pitch/volume support,
available styles, cloning, streaming, SSML, sample rates and limits. Callers use it to decide
whether to route elsewhere, or to tell the user a control is unavailable, instead of silently
receiving neutral speech.

### 5. Degradation policy

Unsupported controls degrade in a fixed order:

1. Use the engine's native control if it exists.
2. Otherwise post-process (time-stretch for rate, pitch-shift for pitch, gain for volume).
3. Otherwise ignore it and report it in `ignored`.

Never silently pretend a control was applied.

## Note on Kokoro style vectors

Kokoro voices are style tensors of shape `(510, 1, 256)`, identical for all 54 voices, and
`kokoro_onnx.Kokoro.create()` accepts either a voice name or an `ndarray`. Blending style tensors
therefore produces a new voice identity with no training and no new model. This was verified in
this repository with a 65 % `af_heart` / 35 % `am_adam` blend, which synthesised correctly.

Blending changes **timbre**, not **emotion**. A blend cannot make a neutral voice sound
empathetic. That requires an expressive engine, which is why ADR-0001 keeps the adapter boundary.

## Consequences

- Adding a voice is a registry edit; adding an engine is a new adapter.
- The Python service becomes the only source of truth for voices; the TypeScript clients fetch
  rather than duplicate.
- The `applied` / `ignored` contract is part of the public API and must be tested.
- `speed` remains accepted as a legacy alias for `direction.prosody.rate` so existing callers keep
  working.
- Post-processing is a compatibility layer, not the strategy. Native model control always sounds
  better and should be preferred when available.

## Alternatives considered

| Alternative | Why rejected |
|---|---|
| Add a `pitch` and `style` field to `TTSRequest` and pass them to Kokoro | Kokoro has no such inputs; the fields would be decorative |
| Separate endpoints per engine (`/kokoro/...`, `/chatterbox/...`) | Leaks the implementation into the public API; clients must know models |
| Distil style into SSML only | Kokoro cannot consume SSML; would need a parser anyway, and SSML lacks a portable emotion vocabulary |
| Keep voice lists in each client | Already caused three disagreeing lists; the drift is the defect |
