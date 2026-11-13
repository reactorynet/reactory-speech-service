.PHONY: setup dev test lint docker-build docker-run image-size clean download-models \
	expressive-build expressive-run download-chatterbox

VENV := .venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
UVICORN := $(VENV)/bin/uvicorn
PYTEST := $(VENV)/bin/pytest
RUFF := $(VENV)/bin/ruff

setup: $(VENV)/bin/activate download-models
	@echo "✅ Setup complete. Run 'make dev' to start the service."

$(VENV)/bin/activate:
	python3.13 -m venv $(VENV) || python3.11 -m venv $(VENV) || python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	touch $(VENV)/bin/activate

download-models:
	$(PYTHON) scripts/download_models.py

dev:
	$(UVICORN) app.main:app --host 0.0.0.0 --port 8765 --reload

run:
	$(UVICORN) app.main:app --host 0.0.0.0 --port 8765

test:
	$(PYTEST) -v -m "not slow"

test-all:
	$(PYTEST) -v

lint:
	$(RUFF) check app/ tests/

lint-fix:
	$(RUFF) check --fix app/ tests/

clean:
	rm -rf $(VENV) __pycache__ .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

docker-build:
	docker build -t reactory-speech-service .

docker-run:
	docker run -p 8765:8765 --env-file .env reactory-speech-service

# Build the speech-core image and assert it stays inside the documented edge
# budget (docs/architecture/speech-platform.md). Fails the build when exceeded.
image-size:
	./scripts/check_image_size.sh --build

# Optional expressive profile (SPEECH-005). A separate build, NOT layered on
# speech-core: it needs Python 3.13 while speech-core is 3.11 (chatterbox-tts
# caps numpy<2.0.0 below 3.13; kokoro-onnx needs numpy>=2.0.2). The default
# image and its size assertion are unchanged. Chatterbox weights are downloaded
# on demand into the models volume, never at build time.
expressive-build: image-size
	docker build -f Dockerfile.expressive -t reactory-speech-service:speech-expressive .

expressive-run:
	docker run -p 8765:8765 --env-file .env \
		-v "$(PWD)/models/chatterbox:/app/models/chatterbox" \
		reactory-speech-service:speech-expressive

download-chatterbox:
	$(PYTHON) scripts/download_chatterbox.py --variant standard
