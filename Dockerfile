# Python 3.14, not 3.15: onnxruntime (required by both kokoro-onnx and
# faster-whisper) publishes no cp315 wheel and no sdist as of 1.32.0, so on
# 3.15 `pip install -r requirements.txt` cannot resolve and the image does not
# build. Dependabot is told to ignore python 3.15 on reactorynet PR 15; run
# `@dependabot unignore python` once onnxruntime ships cp315 wheels.
# This interpreter must stay in the CI test matrix (tests/test_image_budget.py).
#
# Base image is a build arg so each environment can use its own approved
# registry. Override with `--build-arg PYTHON_BASE_IMAGE=<image>`; it must stay
# a [<registry>/]python:X.Y-slim image (Debian, for apt-get below). Dependabot
# does not follow an ARG, so bump the version on this line by hand.
ARG PYTHON_BASE_IMAGE=python:3.14-slim

FROM ${PYTHON_BASE_IMAGE}

# Some hardened base images end on a non-root user. The steps below install
# system packages, so switch to root explicitly.
USER root

# System dependencies for espeak-ng (Kokoro phonemization) and ffmpeg (audio conversion)
RUN apt-get update && apt-get install -y --no-install-recommends \
    espeak-ng \
    ffmpeg \
    ca-certificates \
    curl \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Ensure Python / OpenSSL / requests / httpx / huggingface_hub use system CA certificates
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
ENV CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy models and scripts first so heavy model layers are cached
COPY models/ models/
COPY scripts/ scripts/

# Download models at build time if not present
RUN python scripts/download_models.py

# Copy application code and data
COPY app/ app/
COPY data/ data/

EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8765/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8765"]
