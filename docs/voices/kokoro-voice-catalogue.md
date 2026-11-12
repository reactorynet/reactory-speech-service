# Kokoro v1.0 voice catalogue

The service ships `models/voices-v1.0.bin`, which contains **54 voices across 9 language
families**. Every voice is a style tensor of shape `(510, 1, 256)`, identical for all entries —
which is what makes blending possible.

Before Phase 0 the service exposed 11 of these through a hard-coded allow-list.

## Catalogue

Language is derived from the voice-id prefix. Quality grades are the upstream estimates from the
Kokoro model card and reflect both reference-audio quality and training duration.

| Language | Prefix | Voices | Count |
|---|---|---|---|
| American English | `a` | `af_heart`, `af_alloy`, `af_aoede`, `af_bella`, `af_jessica`, `af_kore`, `af_nicole`, `af_nova`, `af_river`, `af_sarah`, `af_sky`, `am_adam`, `am_echo`, `am_eric`, `am_fenrir`, `am_liam`, `am_michael`, `am_onyx`, `am_puck`, `am_santa` | 20 |
| British English | `b` | `bf_alice`, `bf_emma`, `bf_isabella`, `bf_lily`, `bm_daniel`, `bm_fable`, `bm_george`, `bm_lewis` | 8 |
| Japanese | `j` | `jf_alpha`, `jf_gongitsune`, `jf_nezumi`, `jf_tebukuro`, `jm_kumo` | 5 |
| Mandarin Chinese | `z` | `zf_xiaobei`, `zf_xiaoni`, `zf_xiaoxiao`, `zf_xiaoyi`, `zm_yunjian`, `zm_yunxi`, `zm_yunxia`, `zm_yunyang` | 8 |
| Spanish | `e` | `ef_dora`, `em_alex`, `em_santa` | 3 |
| French | `f` | `ff_siwis` | 1 |
| Hindi | `h` | `hf_alpha`, `hf_beta`, `hm_omega`, `hm_psi` | 4 |
| Italian | `i` | `if_sara`, `im_nicola` | 2 |
| Brazilian Portuguese | `p` | `pf_dora`, `pm_alex`, `pm_santa` | 3 |

Naming convention: `<language><gender>_<name>`, where gender is `f` or `m`.
`af_heart` = American English, female, "Heart".

## Quality grades

Grades come from the upstream `VOICES.md` and are a rough guide to production suitability.

| Grade | Voices |
|---|---|
| A | `af_heart` |
| A- | `af_bella` |
| B- | `af_nicole`, `bf_emma`, `ff_siwis` |
| C+ | `af_aoede`, `af_kore`, `af_sarah`, `am_fenrir`, `am_michael`, `am_puck`, `jf_alpha` |
| C | `af_alloy`, `af_nova`, `bf_isabella`, `bf_lily` (C), `bm_fable`, `bm_george`, `hf_alpha`, `hf_beta`, `hm_omega`, `hm_psi`, `if_sara`, `im_nicola`, `jf_gongitsune`, `jf_tebukuro`, `zf_*`, `zm_*` |
| C- | `af_sky`, `jf_nezumi`, `jm_kumo` |
| D | `af_jessica`, `af_river`, `am_echo`, `am_eric`, `am_liam`, `am_onyx`, `bf_alice`, `bm_daniel` |
| D- | `am_santa` |
| F+ | `am_adam` |

Treat C+ and above as candidates for customer-facing defaults. Lower grades are usable for
internal, placeholder or rare-language cases but should not be a product's primary voice without
listening tests.

Some languages have very little training data (French is a single voice; Mandarin and Hindi voices
are graded C). Pronunciation quality for those languages may be limited by grapheme-to-phoneme
coverage rather than by the model. Verify with native speakers before promising a locale.

## Using a voice

```bash
curl -X POST http://localhost:8765/api/tts/synthesize \
  -H 'Content-Type: application/json' \
  -d '{"text": "Hello from Reactory.", "voice": "af_bella"}' \
  --output hello.wav
```

Any id in the catalogue above is valid after Phase 0. Before Phase 0, only the 11-voice allow-list
is accepted.

## Designing new voices by blending

Because every voice is a tensor of the same shape, a weighted sum is a valid style input:

```python
import numpy as np
from kokoro_onnx import Kokoro

kokoro = Kokoro("models/kokoro-v1.0.onnx", "models/voices-v1.0.bin")

ingredients = {"af_heart": 0.65, "bf_emma": 0.25, "af_nicole": 0.10}
blend = sum(w * kokoro.get_voice_style(v) for v, w in ingredients.items()).astype(np.float32)

samples, sample_rate = kokoro.create(
    "Reactory speech now supports designed voice identities.",
    voice=blend,
    speed=0.95,
    lang="en-us",
)
```

This was verified in this repository with a 65 % `af_heart` / 35 % `am_adam` blend, which
synthesised correctly through the existing API.

**Rules that keep blends usable**

- Normalise weights so they sum to 1.
- Blend within one language family. Cross-language blends degrade pronunciation because the
  phonemiser is chosen from the voice's language, not per component.
- Keep at least 60 % weight on one component; flatter mixes tend to sound like neither voice.
- Use at most four components; more produces mud rather than a new identity.
- Prefer components graded C+ or better for anything customer-facing.
- Store the **recipe**, not the tensor. The tensor is derived and cached; the recipe is the
  durable artefact.

The productised form of this — the design endpoint, the recipe registry, the four shipped
presets and the preview cache — is documented in
[`designed-voices.md`](designed-voices.md). Read that document before implementing against
the blend API; this section remains the underlying model fact.

**What blending does not do**

Blending changes timbre. It does not change emotion, add breathiness, or make a voice sound
empathetic. Those require an expressive engine (Phase 4). Do not let a voice picker imply
otherwise.

## Sources

- Kokoro-82M model card: <https://huggingface.co/hexgrad/Kokoro-82M>
- Voice grades and language notes: `VOICES.md` in the same repository
- Licence: Apache-2.0
