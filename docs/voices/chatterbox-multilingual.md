# Chatterbox Multilingual

Chatterbox Multilingual (`ChatterboxMultilingualTTS` in chatterbox-tts 0.1.7, MIT)
is one model that speaks 23 languages. It runs as a second engine,
`chatterbox_multilingual`, beside the English Chatterbox engine, not instead
of it. Both are optional and off by default.

| Code | Language | Code | Language | Code | Language |
|---|---|---|---|---|---|
| `ar` | Arabic | `fi` | Finnish | `no` | Norwegian |
| `da` | Danish | `fr` | French | `pl` | Polish |
| `de` | German | `he` | Hebrew ¹ | `pt` | Portuguese |
| `el` | Greek | `hi` | Hindi | `ru` | Russian ¹ |
| `en` | English | `it` | Italian | `sv` | Swedish |
| `es` | Spanish | `ja` | Japanese | `sw` | Swahili |
| | | `ko` | Korean | `tr` | Turkish |
| | | `ms` | Malay | `zh` | Chinese ¹ |
| | | `nl` | Dutch | | |

¹ Rendered with a reduced text front end; `/api/tts/capabilities` reports these
as `degraded`:

- **Hebrew** is read without vowel marks, because the optional `dicta-onnx`
  diacritiser is not installed.
- **Russian** is read without stress marks, because the optional
  `russian-text-stresser` package is not installed.
- **Chinese** is converted through the bundled Cangjie table but not
  word-segmented. `spacy-pkuseg` downloads its segmentation model the first
  time it is used, and the adapter never allows a download while loading.

None of the 23 languages has been measured yet: every row is `unverified` or
`degraded`, and `measured` is `false`.

## Voices

Each language is its own voice, `chatterbox_multilingual_<code>`, for example
`chatterbox_multilingual_de`. A request picks the language by picking the voice:

```bash
curl -X POST http://localhost:8765/api/tts/synthesize \
  -H 'Content-Type: application/json' \
  -d '{"text": "Guten Morgen. Wie geht es Ihnen heute?", "voice": "chatterbox_multilingual_de"}' \
  -o german.wav
```

The voice decides the engine, so `chatterbox_standard` still goes to the English
model and a style-only request with no voice still prefers the English engine.
A request that names `"engine": "chatterbox_multilingual"` but no voice is
rendered in English (`chatterbox_multilingual_en` is the engine's default voice).

When a voice cannot be rendered by its own engine (for example the multilingual
engine is not enabled), the service substitutes another voice **in the same
language** and reports it in `X-Direction-Ignored` / `X-Direction-Warnings`. If
the engine that would serve the request has no voice in that language, the
request is refused with 422 rather than answered in another language: on a
deployment without the multilingual engine, `chatterbox_multilingual_de` is a
422, while `chatterbox_multilingual_en` falls back to an English Kokoro voice.
The same rule works the other way: a French Kokoro voice sent to
`"engine": "chatterbox_multilingual"` is rendered by `chatterbox_multilingual_fr`.
If the owning engine is enabled but failed to load, the answer is the 503 for
that engine, since a retry can succeed.
Styles (`empathetic`, `cheerful` and the rest) work the same way as on the
English model, through `exaggeration`. Paralinguistic tags such as `[laugh]` are
not claimed: it has not been checked whether the multilingual tokenizer keeps
them.

## Enabling it

1. **Weights.** About 3.2 GB from the standard Chatterbox repository on Hugging
   Face (`ve.pt`, `t3_mtl23ls_v2.safetensors`, `s3gen.pt`,
   `grapheme_mtl_merged_expanded_v1.json`, `Cangjie5_TC.json`, `conds.pt`), in
   their own directory: the files share a repository with the English model,
   and the engine refuses a directory whose manifest names another variant.

   ```bash
   python scripts/download_chatterbox.py --variant multilingual
   ```

   On a network that blocks Hugging Face, use the offline bundle on a connected
   machine and restore it on the restricted host (see the README, "Restricted
   networks"):

   ```bash
   scripts/models-download.sh --include-chatterbox --include-chatterbox-multilingual
   scripts/models-archive.sh --include-chatterbox --include-chatterbox-multilingual
   ```

2. **Run.** The speech-expressive image already contains the model code. Mount
   the weights and turn the engine on (`CHATTERBOX_VARIANT=multilingual` is
   refused at startup: that setting selects the English model):

   ```bash
   make expressive-run CHATTERBOX_MULTILINGUAL=1
   ```

   which mounts `models/chatterbox-multilingual` and sets
   `CHATTERBOX_MULTILINGUAL_ENABLED=true` and `SPEECH_MAX_LOADED_ENGINES=3`.

## Memory

Each Chatterbox model holds about 4 GB once loaded, so Kokoro and both
Chatterbox models resident need about 9 GB. With `SPEECH_MAX_LOADED_ENGINES=2`
(the default) the manager unloads the least recently used idle engine to make
room instead, and reloads it when it is next needed (about 20–30 s on CPU).

## Compute cost

The capabilities advisory is the English model's measurement (RTF 0.13 on the
reference 8-core node). The multilingual model has the same architecture and is
expected to be similar, but that is an assumption until it is measured.

## Cloning

Voice clones are created on and rendered by the English engine only. A speaker
embedding taken from one model and replayed through another is not something the
consent and quality work for SPEECH-006 covered, so the service never offers the
multilingual engine to the clone flow.

## Loading offline

The upstream tokenizer reaches the network twice while it is being built: it
fetches `Cangjie5_TC.json` from Hugging Face, and `spacy-pkuseg` downloads its
segmentation model. While loading, the adapter answers the first from the
verified model directory and turns the second into the `ImportError` the
tokenizer already handles. CI checks, inside the built image, that both upstream
functions still exist under those names and that the package's language list
matches the adapter's. If either is missing at runtime anyway, loading logs a
`Chatterbox offline guard not applied` warning naming it.
