.PHONY: setup dev test lint docker-build docker-run clean download-models

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
