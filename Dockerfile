FROM python:3.11-slim

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
