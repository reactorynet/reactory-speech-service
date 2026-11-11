# Reactory Speech Service

Local TTS and STT microservice for the Reactory platform, providing text-to-speech (Kokoro-82M ONNX) and speech-to-text (faster-whisper) capabilities.

> **Direction and roadmap:** this service is built open source, self-hosted and edge-first — no paid
> TTS API is required for any capability. Architecture decisions, the delivery roadmap and the full
> voice catalogue live in [`docs/`](docs/README.md).

## Quick Start

### Prerequisites
- Python 3.11+
- macOS: `brew install espeak-ng ffmpeg`
- Or Docker

### Development (virtualenv)

```bash
# Setup (creates venv, installs deps, downloads models ~350MB)
make setup

# Start development server with hot reload
make dev
# Service runs at http://localhost:8765

# Run tests
make test

# Lint
make lint
```

### Docker

```bash
make docker-build
make docker-run
```

Or via the Reactory docker-compose:

```bash
cd $REACTORY_SERVER/config/reactory
docker compose up reactory_speech_service
```

## Configuration

Copy `.env.example` to `.env` and customize:

| Variable | Default | Description |
|---|---|---|
| `SPEECH_SERVICE_PORT` | `8765` | Server port |
| `KOKORO_DEFAULT_VOICE` | `af_heart` | Default TTS voice. Must be an enabled voice id from `GET /api/tts/voices`; an unknown or disabled id falls back to the registry default |
| `KOKORO_DEFAULT_SPEED` | `1.0` | Default speech speed |
| `WHISPER_MODEL_SIZE` | `base` | Whisper model: `tiny`, `base`, `small`, `medium`, `large-v3` |
| `WHISPER_DEVICE` | `auto` | Device: `auto`, `cpu`, `cuda` |
| `PHONETIC_PROCESSING_ENABLED` | `True` | Enable phonetic text processing (phonemise before synthesis) |
| `SPEECH_THREAD_POOL_SIZE` | `4` | Bounded worker threads for inference; the event loop never runs a model |
| `SPEECH_ENGINE_MAX_CONCURRENCY` | `{"kokoro": 2, "default": 1}` | Per-engine concurrency limit as a JSON map |
| `SPEECH_MAX_LOADED_ENGINES` | `2` | Engines resident at once; further engines load lazily on first use |
| `SPEECH_ENGINE_IDLE_TIMEOUT_SECONDS` | `300` | Unload an engine after this long without work (one engine always stays resident) |
| `SPEECH_ENGINE_REAPER_INTERVAL_SECONDS` | `30` | How often the idle reaper runs |
| `SPEECH_CACHE_ENABLED` | `true` | Enable the two-tier synthesis cache |
| `SPEECH_CACHE_DIR` | `.cache/tts` | Disk tier location (local to the node; no Redis) |
| `SPEECH_CACHE_TTL_SECONDS` | `86400` | Entry lifetime, enforced on **both** the in-process LRU and the disk tier |
| `SPEECH_CACHE_LRU_ENTRIES` | `256` | In-process LRU entry cap |
| `SPEECH_CACHE_LRU_MAX_BYTES` | `16777216` | In-process LRU byte cap (16 MiB) |
| `SPEECH_CACHE_LRU_MAX_ENTRY_BYTES` | `524288` | Entries larger than this go to disk only |
| `SPEECH_CACHE_DISK_MAX_BYTES` | `536870912` | Disk tier byte cap (512 MiB) |

## API

### Health
- `GET /health` — Service status, per-engine readiness and queue depth, plus cache counters
- `GET /ready` — Readiness probe (200 only when models loaded) with the same per-engine detail

The legacy `tts` / `stt` booleans and the `ready` key are unchanged. `engines.<id>` reports
`ready`, `loaded`, `lifecycle`, `queue_depth`, `in_flight`, `max_concurrency` and
`peak_concurrency`; the top-level `queue_depth` is the sum across engines.

### TTS (Text-to-Speech)
- `POST /api/tts/synthesize` — Returns WAV audio bytes
- `POST /api/tts/synthesize/json` — Returns base64-encoded audio in JSON
- `POST /api/tts/phonemize` — Preview text normalization & phonetic (IPA) transcription
- `WS /api/tts/stream` — WebSocket streaming (sentence-by-sentence)

The synthesis endpoints accept `speed` in the range **0.5–2.0** (the range kokoro-onnx
supports); anything outside it is rejected with a 422 before inference. The WebSocket
endpoint clamps the same range defensively so the engine is never handed a bad value.

**Both** synthesis endpoints — binary and JSON — carry the audio metadata as headers under
both spellings, so a client can read either one:

| Canonical | Alias |
|---|---|
| `X-Sample-Rate` | `X-Audio-Sample-Rate` |
| `X-Duration` | `X-Audio-Duration` |
| `X-Format` | `X-Audio-Format` |

(`/api/tts/synthesize/json` also returns `duration`, `format` and `sample_rate` in the body.)

Both endpoints also return `X-Cache: hit|miss` and `X-Cache-Tier: lru|disk|none`. A repeated
identical request is served from the cache, is measurably faster and returns byte-identical
audio. Send `"cache": false` in the body to force a fresh render for that request;
`SPEECH_CACHE_ENABLED=false` disables the cache for the deployment.

Inference always runs on a bounded worker pool under a per-engine concurrency limit, so a long
synthesis never blocks `/health` or another request. That holds on every render path, including
`"phonetic_preprocess": false`: grapheme-to-phoneme conversion is serialised (espeak-ng is not
thread-safe) but the lock is released before the model runs, so it never caps concurrency. See
[`docs/architecture/speech-platform.md`](docs/architecture/speech-platform.md) for the limits,
the cache key and the engine lifecycle.

### TTS Voices and Capabilities
- `GET /api/tts/voices` — the full voice catalogue as
  `{"voices": [...], "total": n, "engine": "kokoro", "available": true}`.
  Optional filters: `engine`, `language`, `gender`, `tag`.
- `GET /api/tts/voices/{voice_id}` — one voice descriptor; `404` with a clear message
  when the id is not present in the voice pack.
- `GET /api/tts/capabilities` — per-engine capabilities (languages, `native_rate`,
  `native_pitch`, `native_volume`, `styles`, `supports_cloning`, `supports_streaming`,
  `sample_rates`, `gpu_required`).

`available` on the voices response tells you whether the engine is actually loaded. Voice
discovery works *before* the model is loaded (the shipped-pack manifest is used as the
catalogue), so that a picker can populate at boot — but when `available` is `false`,
synthesis returns `503`. `GET /api/tts/capabilities` reports the same flag per engine.

### Pronunciation Lexicon Management
- `GET /api/tts/lexicon` — List all active pronunciation dictionary entries (built-in + custom)
- `GET /api/tts/lexicon/custom` — List custom user-defined dictionary entries
- `GET /api/tts/lexicon/{word}` — Get pronunciation for a specific word
- `POST /api/tts/lexicon` — Add or update a word pronunciation (persisted to `data/lexicon.json`)
- `DELETE /api/tts/lexicon/{word}` — Remove a custom pronunciation entry

### Customizing Pronunciation

You can customize pronunciation in four ways:

1. **Persistent Dictionary via API**:
   ```bash
   curl -X POST http://localhost:8765/api/tts/lexicon \
     -H 'Content-Type: application/json' \
     -d '{"word": "Kubernetes", "replacement": "koo-ber-net-eez", "ipa": "kˈuːbɚnˌɛtiːz"}'
   ```
2. **Persistent Dictionary via File**: Edit `data/lexicon.json` (or mount into `/app/data/lexicon.json`).
3. **Per-Request Overrides**: Pass `custom_lexicon` in `POST /api/tts/synthesize`:
   ```json
   {
     "text": "Welcome Werner to the platform.",
     "voice": "af_heart",
     "custom_lexicon": {
       "Werner": "Vair-ner"
     }
   }
   ```
4. **Inline Text Markup**:
   - Respellings: `"Welcome [Werner|Vair-ner] to [K8s|koo-ber-net-eez]."`
   - Direct IPA: `"Welcome [Werner|/vˈɛərnər/]."`
   - SSML: `'Welcome <phoneme ph="vˈɛərnər">Werner</phoneme>.'`

### STT (Speech-to-Text)
- `POST /api/stt/transcribe` — Upload an audio file, get a transcription
- `GET /api/stt/capabilities` — Mounted models, device, and enabled features
- `WS /api/stt/stream` — WebSocket streaming transcription

The STT path is **faster-whisper only**: locally mounted CTranslate2 models, CPU by
default, no cloud transcription and no GPU requirement. Audio is written to a
temporary file for the duration of the request and removed on success, on error and
when a streaming client disconnects.

#### `POST /api/stt/transcribe`

`multipart/form-data` fields (all optional except `file`):

| Field | Type | Default | Meaning |
|---|---|---|---|
| `file` | file | — | WAV, WebM, OGG, MP3, FLAC (max 25 MB) |
| `language` | string | auto | Force a language code (e.g. `en`); omit to auto-detect |
| `model` | string | deployment default | A mounted model id (see capabilities) |
| `word_timestamps` | bool | `false` | Add `words` to every segment |
| `vad` | bool | deployment default | Override voice-activity detection |

Response:

```json
{
  "text": "Hello world",
  "language": "en",
  "language_confidence": 0.94,
  "language_probabilities": [
    {"language": "en", "probability": 0.94},
    {"language": "nl", "probability": 0.03}
  ],
  "needs_language_confirmation": false,
  "segments": [
    {"start": 0.0, "end": 1.52, "text": "Hello world"}
  ],
  "duration": 1.52,
  "model": "base",
  "vad_applied": true,
  "vad_failed": false,
  "speech_duration": 1.4
}
```

`duration` is the duration of the **submitted** audio and is never shortened by VAD
trimming; `speech_duration` reports how much of it was speech.
`vad_failed` is `true` when detection could not run (see below).

`language_probability` is Whisper's confidence in the detected language. When it is
below `WHISPER_LANGUAGE_CONFIDENCE_THRESHOLD` (default `0.6`),
`needs_language_confirmation` is `true` and a client should ask the user rather than
silently trusting the guess. `language_probabilities` carries the top three
candidates. Null fields are omitted from the payload.

With `word_timestamps=true` each segment also carries `words`:

```json
{"start": 0.0, "end": 1.52, "text": "Hello world",
 "words": [
   {"word": "Hello", "start": 0.0, "end": 0.62, "probability": 0.93},
   {"word": "world", "start": 0.68, "end": 1.48, "probability": 0.88}
 ]}
```

Word timestamps are monotonic within a segment (start ≤ end, each word after the
previous one). They are off by default so the payload stays small.

**Precedence:** omitting `word_timestamps` uses the deployment setting
`WHISPER_WORD_TIMESTAMPS`; sending `word_timestamps=true` or `false` always wins. The
WebSocket `word_timestamps` config key behaves the same way.

#### Voice-activity detection

Before decoding, the audio is passed through the Silero VAD model that ships inside
`faster-whisper` (MIT, ONNX, CPU). Silence is removed and the remaining speech is
concatenated back onto the original timeline, so timestamps still point at the
source audio. Long silences therefore cost nothing and clips with pauses transcribe
faster.

There are three distinct outcomes, and they are not interchangeable:

| Situation | Response |
|---|---|
| Speech trimmed | `vad_applied: true`, `speech_duration` set, transcript from the trimmed audio |
| Detection ran, no speech at all | `vad_applied: true`, `speech_duration: 0.0`, `text: ""` (nothing to decode) |
| Detection could not run | `vad_applied: false`, `vad_failed: true`, and the **original audio is transcribed normally** |

The third row covers a missing VAD module/asset, an ONNX runtime mismatch, a decoder
that cannot open the file, and any exception raised while detecting. It is reported
explicitly (`vad_failed`) instead of being turned into an empty transcript, so a
broken detector degrades to slower-but-correct transcription and is visible in
monitoring. Setting `vad=false` (per request) or `WHISPER_VAD_ENABLED=false`
(deployment) skips detection entirely and also leaves the audio untouched.

#### Model tiering

Any faster-whisper model mounted under `MODELS_DIR` is selectable per request. A
directory counts as a model when it contains `model.bin`; the id is the directory
name with `whisper-` or `faster-whisper-` stripped:

```
models/
  whisper-tiny/          -> "tiny"
  whisper-base/          -> "base"
  whisper-small/         -> "small"
  faster-whisper-distil-large-v3/  -> "distil-large-v3"
```

The catalogue is re-scanned on every `/api/stt/capabilities` request, so a model
mounted after boot appears without restarting the service. If `WHISPER_MODEL_PATH`
points at a generically named directory (`.../models`), the model id comes from
`WHISPER_MODEL_SIZE` rather than the folder name.

`GET /api/stt/capabilities` reports what is mounted:

```json
{
  "ready": true,
  "default_model": "base",
  "available_models": ["base", "small", "tiny"],
  "loaded_models": ["base"],
  "device": "cpu",
  "compute_type": "int8",
  "streaming": true,
  "word_timestamps": true,
  "language_confidence": true,
  "vad": {"enabled": true, "available": true, "threshold": 0.5,
          "min_silence_ms": 500, "min_speech_ms": 250, "pad_ms": 200}
}
```

`WHISPER_MODEL_SIZE` picks the model loaded at startup; models are loaded lazily
when first requested. Models are only ever loaded from disk
(`local_files_only=True`) — nothing is downloaded at runtime. Requesting a model
that is not mounted returns `400` with the list of mounted ids:

```json
{"detail": {"error": "model_not_available", "message": "...",
            "model": "large-v3", "available_models": ["base", "small", "tiny"]}}
```

#### `WS /api/stt/stream` — streaming protocol

1. **Configure once** (optional; every key is optional):

   ```json
   {"language": "en", "encoding": "pcm_s16le", "sample_rate": 16000,
    "channels": 1, "model": "base", "word_timestamps": false,
    "min_window_seconds": 2.0, "partial_interval_seconds": 1.0}
   ```

   The server replies `{"status": "configured", ...}` echoing the effective
   settings. `encoding` is `pcm_s16le` for raw little-endian 16-bit PCM — the
   recommended streaming encoding, because any byte boundary is a valid cut — or
   `container` (default) for WAV/WebM/OGG/MP3.

2. **Send binary audio chunks.** As soon as `min_window_seconds` of audio is
   buffered, the server re-decodes the window and emits the segments that are
   stable — those ending at least `WHISPER_STREAM_STABILITY_MARGIN_SECONDS` before
   the end of the buffer. The unstable tail is held back until the next window or
   until `done`, so text is not emitted twice with different timings.

3. **Send `{"done": true}`** to flush the tail and finish.

Every segment uses the original message contract plus additive fields:

```json
{"text": "Hello world", "start": 0.0, "end": 1.5,
 "partial_text": "Hello world", "language": "en", "is_partial": true}
```

`words` is present when `word_timestamps` was requested. The final message is:

```json
{"done": true, "text": "Hello world", "language": "en", "language_confidence": 0.94}
```

Existing clients that only read `text`/`start`/`end`/`partial_text`/`language` and
the final `{"done": true}` keep working unchanged. Errors arrive as
`{"error": "..."}`; an unknown model is reported as
`{"error": "model_not_available", "available_models": [...]}` without closing the
socket.

**Partial results need a length-derivable encoding.** Partials are emitted for
`encoding: "pcm_s16le"` and for WAV, where the server can tell how much audio it has.
With `encoding: "container"` (the default) and a WebM/OGG/MP3 payload — which is what
a browser `MediaRecorder` produces — the server cannot time a prefix, so it transcribes
once at `done` and returns the full result without intermediate partials. Send
`pcm_s16le` from an `AudioWorklet` if you need live captions.

Duplicate suppression: each segment is emitted once. If a re-decode of the growing
buffer moves a boundary outward (`"hello"` -> `"hello world"`), the newer segment
supersedes the earlier one, so `partial_text` and the final `text` never repeat words.

#### STT configuration

| Variable | Default | Description |
|---|---|---|
| `WHISPER_MODEL_SIZE` | `base` | Model id loaded at startup |
| `WHISPER_DEVICE` | `auto` | `auto`, `cpu` or `cuda`; CPU is the supported default |
| `WHISPER_COMPUTE_TYPE` | `auto` | CTranslate2 compute type (e.g. `int8` on CPU) |
| `WHISPER_WORD_TIMESTAMPS` | `false` | Default when `word_timestamps` is omitted (an explicit `true`/`false` wins) |
| `WHISPER_VAD_ENABLED` | `true` | Run Silero VAD before transcription |
| `WHISPER_VAD_THRESHOLD` | `0.5` | Speech probability threshold |
| `WHISPER_VAD_MIN_SILENCE_MS` | `500` | Silence needed to split regions |
| `WHISPER_VAD_MIN_SPEECH_MS` | `250` | Shortest accepted speech region |
| `WHISPER_VAD_PAD_MS` | `200` | Padding kept around each region |
| `WHISPER_STREAM_MIN_WINDOW_SECONDS` | `2.0` | Buffered audio before the first partial |
| `WHISPER_STREAM_PARTIAL_INTERVAL_SECONDS` | `1.0` | Minimum gap between partial passes |
| `WHISPER_STREAM_STABILITY_MARGIN_SECONDS` | `0.35` | Tail withheld from partial output |
| `WHISPER_STREAM_MAX_BUFFER_SECONDS` | `300` | Per-session streaming limit |
| `WHISPER_LANGUAGE_CONFIDENCE_THRESHOLD` | `0.6` | Below this, ask the user to confirm |

**Not implemented (follow-up):** speaker diarisation. It needs a separate model on
top of the transcription, so it is tracked separately rather than smuggled into the
STT payload.

### Examples

```bash
# Synthesize speech
curl -X POST http://localhost:8765/api/tts/synthesize \
  -H 'Content-Type: application/json' \
  -d '{"text": "Hello from Reactory!", "voice": "af_heart"}' \
  --output hello.wav

# Play it (macOS)
afplay hello.wav

# Transcribe audio
curl -X POST http://localhost:8765/api/stt/transcribe \
  -F 'file=@hello.wav'

# List available at /docs (Swagger UI)
open http://localhost:8765/docs
```

## Configuration

Copy `.env.example` to `.env` and customize:

| Variable | Default | Description |
|---|---|---|
| `SPEECH_SERVICE_PORT` | `8765` | Server port |
| `KOKORO_DEFAULT_VOICE` | `af_heart` | Default TTS voice. Must be an enabled voice id from `GET /api/tts/voices`; an unknown or disabled id falls back to the registry default |
| `KOKORO_DEFAULT_SPEED` | `1.0` | Default speech speed |
| `WHISPER_MODEL_SIZE` | `base` | Whisper model: `tiny`, `base`, `small`, `medium`, `large-v3` |
| `WHISPER_DEVICE` | `auto` | Device: `auto`, `cpu`, `cuda` |

## Voices

The catalogue is derived from the loaded voice pack (`models/voices-v1.0.bin`): the pack
is the single source of truth for which voices exist, and the service never filters it
through a hard-coded list. The shipped pack contains **54 voices across 9 languages**
(29 female, 25 male), each a style tensor of shape `(510, 1, 256)`:

| Language | Tag | Voices | Example ids |
|---|---|---|---|
| American English | `en-US` | 20 | `af_heart` (grade A), `af_bella` (A-), `am_michael` |
| British English | `en-GB` | 8 | `bf_emma` (B-), `bm_george` |
| Japanese | `ja` | 5 | `jf_alpha`, `jm_kumo` |
| Mandarin Chinese | `zh` | 8 | `zf_xiaoxiao`, `zm_yunxi` |
| Spanish | `es` | 3 | `ef_dora`, `em_alex` |
| French | `fr-FR` | 1 | `ff_siwis` (B-) |
| Hindi | `hi` | 4 | `hf_alpha`, `hm_psi` |
| Italian | `it` | 2 | `if_sara`, `im_nicola` |
| Brazilian Portuguese | `pt-BR` | 3 | `pf_dora`, `pm_santa` |

### Descriptor fields

| Field | Meaning |
|---|---|
| `id` | Voice id as it appears in the pack (e.g. `af_heart`) |
| `name` | Display name derived from the id (e.g. `Heart`) |
| `language` | Canonical language tag (e.g. `en-US`) |
| `locale` | Code handed to the engine for phonemization (e.g. `en-us`; `cmn` for Chinese) |
| `gender` | `female` or `male` |
| `engine` | Engine id (`kokoro`) |
| `quality` | Published quality grade, or `null` when the catalogue does not grade the voice |
| `tags` | Filterable tags, see below |

### Quality grades and tags

Grades reproduce the Kokoro-82M estimates from the catalogue of record: **47 of the 54
voices are graded**, and the remaining 7 (`bm_lewis`, `ef_dora`, `em_alex`, `em_santa`,
`pf_dora`, `pm_alex`, `pm_santa`) are reported as `quality: null` rather than being given
an invented grade.

| Grade | Voices |
|---|---|
| A | `af_heart` |
| A- | `af_bella` |
| B- | `af_nicole`, `bf_emma`, `ff_siwis` |
| C+ | `af_aoede`, `af_kore`, `af_sarah`, `am_fenrir`, `am_michael`, `am_puck`, `jf_alpha` |
| C | 22 voices, including `af_alloy`, `bf_lily`, `bm_george`, `hf_*`, `hm_*`, `zf_*`, `zm_*` |
| C- | `af_sky`, `jf_nezumi`, `jm_kumo` |
| D | 8 voices, including `af_jessica`, `am_echo`, `bf_alice`, `bm_daniel` |
| D- | `am_santa` |
| F+ | `am_adam` |

Tags are lower-case and filterable via `?tag=`:

- `female` / `male`
- `english` / `japanese` / `chinese` / `spanish` / `french` / `hindi` / `italian` / `portuguese`
- `american` / `british` / `mandarin` (regional variants)
- one grade tag per graded voice, with the sign spelled out: `grade-a`, `grade-a-minus`,
  `grade-b-minus`, `grade-c-plus`, `grade-c`, `grade-c-minus`, `grade-d`, `grade-d-minus`,
  `grade-f-plus`
- `recommended` — the catalogue's customer-facing guidance ("treat C+ and above as
  candidates for customer-facing defaults"), so 12 voices
- `unrated` — the 7 voices with no published grade

```bash
# All 54 voices
curl http://localhost:8765/api/tts/voices

# Filter by language, gender, tag or engine
curl 'http://localhost:8765/api/tts/voices?language=ja'
curl 'http://localhost:8765/api/tts/voices?language=en-us&gender=female'
curl 'http://localhost:8765/api/tts/voices?tag=recommended'
curl 'http://localhost:8765/api/tts/voices?tag=unrated'

# One voice (404 when unknown)
curl http://localhost:8765/api/tts/voices/af_heart

# Engine capabilities
curl http://localhost:8765/api/tts/capabilities
```

The per-voice catalogue of record — language families, the full grade table, blending
recipes and licensing notes — lives in `docs/voices/kokoro-voice-catalogue.md`. That
document is introduced by the speech-direction branch (PR #6); until that merges, the table
above and `GET /api/tts/voices` are the available references.

## Architecture

```
app/
├── main.py              # FastAPI app, lifespan (model loading)
├── config.py            # Settings via pydantic-settings
├── routers/
│   ├── health.py        # Health/readiness endpoints
│   ├── tts.py           # TTS REST + WebSocket
│   └── stt.py           # STT REST + WebSocket
├── services/
│   ├── tts_service.py   # Kokoro ONNX wrapper
│   ├── voice_registry.py # Voice catalogue metadata + capabilities
│   └── stt_service.py   # faster-whisper wrapper
└── models/
    └── schemas.py       # Pydantic request/response models
```
