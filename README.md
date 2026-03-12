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

## API

### Health
- `GET /health` — Service status and model availability
- `GET /ready` — Readiness probe (200 only when models loaded)

### TTS (Text-to-Speech)
- `POST /api/tts/synthesize` — Returns WAV audio bytes
- `POST /api/tts/synthesize/json` — Returns base64-encoded audio in JSON
- `WS /api/tts/stream` — WebSocket streaming (sentence-by-sentence)

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

## Available Voices

| ID | Name | Language |
|---|---|---|
| `af_heart` | Heart (Female) | en-us |
| `af_bella` | Bella (Female) | en-us |
| `af_nicole` | Nicole (Female) | en-us |
| `af_sarah` | Sarah (Female) | en-us |
| `af_sky` | Sky (Female) | en-us |
| `am_adam` | Adam (Male) | en-us |
| `am_michael` | Michael (Male) | en-us |
| `bf_emma` | Emma (Female, British) | en-gb |
| `bf_isabella` | Isabella (Female, British) | en-gb |
| `bm_george` | George (Male, British) | en-gb |
| `bm_lewis` | Lewis (Male, British) | en-gb |

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
│   └── stt_service.py   # faster-whisper wrapper
└── models/
    └── schemas.py       # Pydantic request/response models
```
