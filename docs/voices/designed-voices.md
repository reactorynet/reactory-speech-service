# Designed voices — blending Kokoro style vectors

A *designed voice* is a recipe, not a model. Kokoro exposes every voice as a style
tensor of shape `(510, 1, 256)`; `kokoro_onnx.Kokoro.create()` accepts that tensor where it
accepts a voice name. A normalised weighted sum of two to four base tensors is therefore a
valid new style input — a new vocal identity with **no training, no second model, no GPU and
no paid API**. The model is loaded once and shared by every designed voice.

> **Blending changes timbre. It does not change emotion.**
> A blend cannot make a voice sound empathetic, calm, cheerful or excited. Emotion needs an
> expressive engine (SPEECH-005). A voice picker that implied otherwise would be a product
> lie, so the API description says the same thing.

## Recipe rules

| Rule | Why |
|---|---|
| 2 to 4 components | One component is just that voice; more than four produces mud rather than a new identity |
| Weights finite, non-negative, sum > 0 | Stored weights are relative; they are normalised to 1 internally |
| All components in **one** language family | The phonemiser is chosen from the voice's language, not per component, so a cross-family blend degrades pronunciation |
| Dominant component ≥ 0.6 of the total | Flatter mixes sound like neither ingredient |
| Ingredients are **shipped pack voices** | A designed voice is never an ingredient — neither a blend nor a base alias. Composing user-managed entries makes one recipe depend on another, and deleting the inner voice would silently break every recipe naming it. It also keeps the derivation one level deep and guarantees every accepted ingredient renders |
| Store the **recipe**, never the tensor | The tensor is derived and cached (keyed by a stable hash of the normalised recipe); the recipe is the durable artefact |

Violations return `422` with an actionable message; unknown component ids are rejected with
the valid ids listed, and a designed voice used as a component or alias target is named as
such rather than reported as merely unknown.

**Zero-weight components are accepted.** `1.0 + 0.0` passes the literal rules (non-negative,
positive sum) and normalises to `{a: 1.0, b: 0.0}`. It is a pointless recipe — the result is
the dominant voice — but rejecting it would force callers to special-case a scale they may
compute programmatically. Documented rather than refused; use a base alias if you want one
voice under a product name.

## API

```bash
# Design a blended voice. The preview WAV is synthesised and cached on create.
curl -X POST http://localhost:8765/api/tts/voices/design \
  -H 'Content-Type: application/json' \
  -d '{
        "name": "Evening Host",
        "components": [
          {"voice": "af_heart", "weight": 0.65},
          {"voice": "af_bella", "weight": 0.20},
          {"voice": "af_nicole", "weight": 0.15}
        ],
        "direction": {"prosody": {"rate": 0.95}},
        "tags": ["narration", "warm"]
      }'
# -> 201, id "evening_host"

# Preview audio (audio/wav; a cache hit does not re-synthesise)
curl http://localhost:8765/api/tts/voices/evening_host/preview --output preview.wav

# The persisted recipe and design metadata
curl http://localhost:8765/api/tts/voices/evening_host/design

# Synthesise with the designed voice, exactly like a pack voice
curl -X POST http://localhost:8765/api/tts/synthesize \
  -H 'Content-Type: application/json' \
  -d '{"text": "Hello from a designed voice.", "voice": "evening_host"}' \
  --output hello.wav

# Register a product alias for a base voice (no pack data is duplicated)
curl -X POST http://localhost:8765/api/tts/voices \
  -H 'Content-Type: application/json' \
  -d '{"name": "Heart Alias", "base_voice": "af_heart"}'
# -> 201, style_spec {"type": "base", "voice": "af_heart"}

# Delete the voice designed above (200, and it leaves /voices).
# Shipped pack voices and the four presets are immutable and return 409.
curl -X DELETE http://localhost:8765/api/tts/voices/evening_host
```

`POST /api/tts/voices/design` returns the descriptor with the design metadata:
`components` (each with its authored `weight` and `normalised_weight`),
`normalised_weights`, `resolved_language`, `created_at`, `preview_url`, `style_type` and the
persisted `style_spec` recipe.

## Shipped preset library

Four designed voices ship as registry data in `data/designed_voices.json`. Each is a
documented recipe, not a binary artefact.

| Id | Name | Language | Recipe | Default rate |
|---|---|---|---|---|
| `warm_narrator` | Warm Narrator | en-US | `af_heart` 0.65 + `af_bella` 0.20 + `af_nicole` 0.15 | 0.95 |
| `calm_assistant` | Calm Assistant | en-US | `af_nicole` 0.70 + `af_sarah` 0.20 + `af_kore` 0.10 | 0.95 |
| `bright_support` | Bright Support | en-US | `af_bella` 0.60 + `af_sky` 0.25 + `af_nova` 0.15 | 1.05 |
| `steady_narration` | Steady Narration | en-GB | `bm_george` 0.70 + `bm_fable` 0.20 + `bf_emma` 0.10 | 0.95 |

The names describe **timbre** — the character of the voice — not an emotional performance.
"Calm Assistant" is a softer-sounding vocal identity; it is still neutral delivery. Rate is
part of the default direction, which is applied only when a synthesis request expresses no
speed or direction of its own.

Presets are immutable: `DELETE` returns `409`. Design a new voice when you need a different
recipe. A preset whose components are missing from a custom voice pack is not advertised,
because the service will not offer a voice it cannot render.

## Persistence

Custom voices are one JSON file, configurable with `DESIGNED_VOICES_PATH` (default
`data/designed_voices.json`). Previews cache under `DESIGNED_VOICES_PREVIEW_DIR` (default
`data/previews/`). Loading is tolerant: a missing or malformed file logs a warning and the
service continues with the shipped pack voices only — a bad file never fails startup and
never hides the 54 pack voices.

Tolerance must not become data loss, so before the **first write** after a bad load the
unusable file is moved aside to `<name>.corrupt-<UTC timestamp>` and its path is logged. That
covers an unparseable file and a partly corrupt one (where a broken row is skipped in memory).
If the file cannot be preserved, the write is refused rather than overwriting it. The shipped
presets live in the repository's `data/designed_voices.json`, so they can always be restored.

The registry is a **single-process store**: the whole file is rewritten on each change, so
with `uvicorn --workers N` a voice created in one worker is invisible to the others and
concurrent writes are last-writer-wins. Run one worker for the edge profile, or put the
registry behind a shared external store before scaling out horizontally.

## Memory

The resolved tensor cache is bounded by `DESIGNED_VOICE_TENSOR_CACHE_SIZE` (default 64
entries). One tensor is 510 × 1 × 256 float32 ≈ 522 KB, so the worst case is ~33 MB
regardless of how many voices are designed. `KokoroEngine.clear_blend_cache()` drops every
tensor; recipes on disk are unaffected and the next synthesis re-derives them.

## What a blend is not

- Not emotion, style acting or prosody.
- Not voice cloning (that is the `reference` style spec, out of scope here).
- Not a new model download, a GPU workload or a hosted dependency. Everything runs on CPU
  with the existing Apache-2.0 Kokoro model.
