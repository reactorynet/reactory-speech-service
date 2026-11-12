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

## Text front end (G2P) support — measured, not assumed

The voice pack is an **acoustic** asset. It says which languages the model can render; it says
nothing about whether the text front end can phonemise them, and on this deployment the two
disagree. Until SPEECH-010 the service advertised every catalogue language, so a request for a
language espeak-ng could not read produced a valid WAV of nonsense (the Japanese case: 773 KB /
16.11 s for a 21-character sentence, made of the words "english japanese letter").

`GET /api/tts/capabilities` now reports **what the phonemiser was measured able to produce**, and a
request for a language it cannot handle returns **422** instead of audio. The matrix below is not
hand-written: it is `app/eval/data/language-quality.json`, produced by
`scripts/record_language_quality.py` from **every entry in the committed golden corpus** for every
shipped voice, through the local faster-whisper model. `tests/test_multilingual_roundtrip.py`
re-measures the corpus and fails if the file and the measurement disagree, so a number here cannot
drift from reality or be replaced by a sentence that happens to pass.

| Language | Voices | Engine locale | Advertised | Evidence (committed corpus) |
|---|---|---|---|---|
| American English | `af_*`, `am_*` | `en-us` | yes — production | 22 sentences, worst WER 0.2632 / CER 0.2105 |
| British English | `bf_*`, `bm_*` | `en-gb` | yes — production | 22 sentences, worst WER 0.2632 / CER 0.2105 |
| Spanish | `ef_*`, `em_*` | `es` | yes — production | 2 sentences, worst WER 0.1053 / CER 0.0303 |
| Brazilian Portuguese | `pf_*`, `pm_*` | `pt-br` | yes — **small sample** | 1 sentence (WER 0.2857 / CER 0.0312); passes, but one sentence is a smoke test |
| French | `ff_siwis` | `fr-fr` | yes, **usable with caveats** | `fr-greeting-short` exceeded target; worst WER 0.6667 / CER 0.2444 |
| Italian | `if_*`, `im_*` | `it` | yes — **small sample** | 1 sentence (WER 0.0 / CER 0.0); passes, but one sentence is a smoke test |
| Hindi | `hf_*`, `hm_*` | `hi` | yes — **small sample** | 1 sentence (WER 0.3333 / CER 0.0435, Devanagari); a smoke test, not coverage |
| Mandarin Chinese | `zf_*`, `zm_*` | `cmn` | yes, **usable with caveats** | `zh-narration-long` exceeded target; worst WER 0.9231 / CER 0.9231 |
| Japanese | `jf_*`, `jm_*` | `ja` | **no** | espeak-ng spells the script out (8 language switches for the probe sentence, e.g. ['en', 'ja', 'en', 'ja']) |

**Sample size is part of the evidence.** The corpus has 22 English sentences but only one or two
for Portuguese, Italian, Hindi and Spanish: those languages pass, and their row says so, but one
sentence is a smoke test rather than a quality claim. `GET /api/tts/capabilities` attaches the same
caveat, so a client does not have to read this document to learn that. Only English has anything
approaching coverage, which is why the grade table above — not these pass/fail bands — is what to
use when picking a customer-facing voice.

**What "usable with caveats" means.** Two advertised languages do not round-trip
cleanly on the committed corpus, and saying so is the point of this table:

* **Mandarin Chinese** — short and technical sentences round-trip (greeting CER 0.50 with
  `zf_xiaoxiao`, 0.36 with `zm_yunjian`; the domain-vocabulary sentence 0.28 / 0.38), but the
  narration sentence measures **CER 0.92 / 0.72**. Espeak's Mandarin phonemisation is good enough
  for short utterances and degrades with length; that is a text-front-end limitation, not the
  acoustic model, and it is why Mandarin is not listed as production-ready. Treat long Chinese
  narration as unsupported in practice.
* **French** — `ff_siwis` round-trips narration (0.04) and numbers (0.25), but the greeting scores
  WER 0.67 because the local `base` Whisper model mishears the proper name ("Werner" → "Wayne").
  That is a recogniser limitation on a name, not a synthesis defect, and it is recorded rather than
  averaged away.

Both appear in `GET /api/tts/capabilities` as a measured caveat with the failing entry named, so a
caller can decide before rendering.

**Japanese is unsupported on this build.** The voices are still present and still listed by
`GET /api/tts/voices` — the acoustic model renders them — but espeak-ng has no usable Japanese
dictionary, so it names each character ("japanese letter", "chinese letter") instead of reading
it. Advertising a language is therefore tied to evidence, not to the pack.

**Two guards stop a script a voice cannot read**, because the marker guard alone was not enough:
espeak emits language-switch markers when it *knows* it is changing language, but when a Japanese
sentence is sent to an English voice it simply transliterates each character with no markers at
all (measured: `dʒˈæpəniːzlˌɛɾɚ` twenty-one times, 20.9 s of audio, HTTP 200).

1. **Before phonemisation — script vs locale.** If the text contains a script the voice's locale
   does not use (kana/Han for `en-*`, `es`, `it`, `fr`, `pt`…), the request is refused immediately
   with `UnsupportedLanguageError` → **422**, with no render. Latin is accepted by every shipped
   language, which is why `Reactory` works as a loanword in Spanish, Italian, Portuguese and Hindi.
   Inline `[word|/ipa/]` spans are exempt: they need no G2P.

   Two deliberate allowances keep this from refusing text the phonemizer reads correctly:

   * **Greek letters and mathematical symbols are readable alongside Latin.**
     `The value of π is 3.14` → `ðə vˈæljuː ʌv pˈaɪ ɪz θɹˈiː.fˈɔːɹtiːn`, and `Use Δ for
     difference` → `jˈuːs dˈɛltə fɔːɹ dˈɪfɹəns`. Developer copy containing π, Δ, Ω, ∑ or √ is
     normal input and used to synthesise; refusing it was a regression, not a fix.
   * **A single foreign character is tolerated** and falls through to the output check, because
     one letter is read as a letter name (`Option Д` → `ˈɑːpʃən dˈɛː`) while a foreign *word* is
     spelt out (`Привет` → `pˈɛː ˈɛɹ ˈɪː vˈɛː jˈɛː tˈɛː`, refused).

   **The remaining risk is mixed-script text:** a Chinese or Japanese *word* inside an English
   sentence still shortens the utterance to a 422 rather than being transliterated. That is
   deliberate — the alternative is the "chinese letter" garbage this ticket is about — and the
   workaround is to split the utterance or supply an explicit `[word|/ipa/]` span.
2. **After phonemisation — the spelling signature.** A phoneme string that emits about one token
   per source character, and whose tokens are nearly all one value cycling, is espeak reading the
   text out letter by letter. Measured separation: legitimate sentences 0.69–1.00 distinct-token
   ratio at ≤0.76 tokens/char; spelling fallback 0.05–0.10 distinct at 1.00 tokens/char. Language
   switches are also counted, relative to input length, so a legitimate loanword is not refused.

Both guards run on the converter path *and* on the engine's fused (self-phonemising) path, so the
streaming and `phonetic_preprocess=false` routes cannot reach the vocoder with garbage either.

Two related rules the same change pins down:

- **The voice's locale is authoritative.** A caller language tag is resolved through the voice
  registry (`zh` → `cmn`, `pt` → `pt-br`), never passed to espeak-ng verbatim. The registry is the
  only mapping table; there is no second copy to drift.
- **The built-in lexicon is English.** `Ree-actory`, `jay-son`, `kilometers` and the abbreviation
  expansions apply to English voices only. Applying an English respelling to French spliced an
  English diphthong into the sentence (`(en)ɹˈiːˈaktəɹi(fr)`). Other languages use their own G2P or
  an explicit lexicon entry, whose `ipa` field is exact.

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
