# Reactory Speech Service

Local TTS and STT microservice for the Reactory platform, providing text-to-speech (Kokoro-82M ONNX) and speech-to-text (faster-whisper) capabilities.

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
| `KOKORO_DEFAULT_VOICE` | `af_heart` | Default TTS voice |
| `KOKORO_DEFAULT_SPEED` | `1.0` | Default speech speed |
| `WHISPER_MODEL_SIZE` | `base` | Whisper model: `tiny`, `base`, `small`, `medium`, `large-v3` |
| `WHISPER_DEVICE` | `auto` | Device: `auto`, `cpu`, `cuda` |
| `PHONETIC_PROCESSING_ENABLED` | `False` | Enable phonetic text processing for improved pronunciation |

## API

### Health
- `GET /health` — Service status and model availability
- `GET /ready` — Readiness probe (200 only when models loaded)

### TTS (Text-to-Speech)
- `POST /api/tts/synthesize` — Returns WAV audio bytes
- `POST /api/tts/synthesize/json` — Returns base64-encoded audio in JSON
- `POST /api/tts/phonemize` — Preview text normalization & phonetic (IPA) transcription
- `WS /api/tts/stream` — WebSocket streaming (sentence-by-sentence)

The synthesis endpoints accept `speed` in the range **0.5–2.0** (the range kokoro-onnx
supports); anything outside it is rejected with a 422 before inference. Responses carry
the audio metadata under both header names — `X-Sample-Rate` / `X-Audio-Sample-Rate`,
`X-Duration` / `X-Audio-Duration`, `X-Format` / `X-Audio-Format` — so either spelling
works for clients.

### TTS Voices and Capabilities
- `GET /api/tts/voices` — full voice catalogue as `{"voices": [...], "total": n}`;
  optional filters `engine`, `language`, `gender`, `tag`
- `GET /api/tts/voices/{voice_id}` — one voice descriptor; `404` with a clear message
  when the id is not present in the voice pack
- `GET /api/tts/capabilities` — per-engine capabilities (languages, `native_rate`,
  `native_pitch`, `native_volume`, `styles`, `supports_cloning`, `supports_streaming`,
  `sample_rates`, `gpu_required`)

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
- `POST /api/stt/transcribe` — Upload audio file, get transcription
- `WS /api/stt/stream` — WebSocket streaming transcription

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
| `KOKORO_DEFAULT_VOICE` | `af_heart` | Default TTS voice |
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
| French | `fr-FR` | 1 | `ff_siwis` |
| Hindi | `hi` | 4 | `hf_alpha`, `hm_psi` |
| Italian | `it` | 2 | `if_sara`, `im_nicola` |
| Brazilian Portuguese | `pt-BR` | 3 | `pf_dora`, `pm_santa` |

Each descriptor reports `id`, `name`, `language`, `locale` (the code handed to the
engine), `gender`, `engine`, `quality` (published Kokoro-82M grade) and `tags`
(`female`/`male`, language, regional variant, `grade-a`…`grade-f`, `recommended`).
The full per-voice table is in
[`docs/voices/kokoro-voice-catalogue.md`](docs/voices/kokoro-voice-catalogue.md).

```bash
# All 54 voices
curl http://localhost:8765/api/tts/voices

# Filter by language, gender, tag or engine
curl 'http://localhost:8765/api/tts/voices?language=ja'
curl 'http://localhost:8765/api/tts/voices?language=en-us&gender=female'
curl 'http://localhost:8765/api/tts/voices?tag=recommended'

# One voice (404 when unknown)
curl http://localhost:8765/api/tts/voices/af_heart

# Engine capabilities
curl http://localhost:8765/api/tts/capabilities
```

Voice discovery works before the model is loaded (the shipped-pack manifest is used as
the catalogue); synthesis still requires the engine.

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
