#!/usr/bin/env bash
#
# Download the speech-core models (Kokoro TTS + faster-whisper STT) into a
# models directory, ready to be packed by scripts/models-archive.sh.
#
# This is step 1 of the offline model bundle, for hosts whose network blocks a
# model source (a corporate web filter that refuses huggingface.co, which is
# where the Whisper weights live). Run steps 1 and 2 on a machine that can reach
# GitHub and Hugging Face, copy the archive across, then run step 3 on the
# restricted host:
#
#   1. scripts/models-download.sh                 # on the connected machine
#   2. scripts/models-archive.sh                  # on the connected machine
#   3. scripts/models-restore.sh <archive>        # on the restricted host
#
# It wraps scripts/download_models.py (the same downloader the Dockerfile runs),
# so the files and layout are exactly what the image build expects. That script
# skips files that are already present, so an image built after a restore does
# not touch the network for models.
#
# Python 3.10+ is required. When the chosen interpreter cannot import httpx and
# huggingface_hub, a throwaway virtualenv is created for the run and removed
# afterwards; the repository's own .venv is never modified.
#
# Usage:
#   scripts/models-download.sh [--models-dir DIR] [--whisper-size SIZE]
#                              [--python PYTHON]
#
#   --models-dir DIR     where to download (default: <repo>/models)
#   --whisper-size SIZE  faster-whisper size: tiny, base, small, ... (default:
#                        $WHISPER_MODEL_SIZE, else base — the service default)
#   --python PYTHON      interpreter to run the downloader (default: python3)
#
# Exit codes:
#   0  every expected model file is present
#   1  the download ran but expected files are missing
#   2  the download could not run (bad usage, no suitable Python)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MODELS_DIR="$REPO_ROOT/models"
WHISPER_SIZE="${WHISPER_MODEL_SIZE:-base}"
PYTHON="python3"

#: The files faster-whisper needs in a model directory; mirrors WHISPER_FILES in
#: scripts/download_models.py.
WHISPER_FILES="config.json model.bin tokenizer.json vocabulary.txt"

usage() {
    sed -n '2,37p' "$0" | sed 's/^# \{0,1\}//'
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --models-dir)
            [ "$#" -ge 2 ] || { echo "error: --models-dir needs a value" >&2; exit 2; }
            MODELS_DIR="$2"
            shift 2
            ;;
        --whisper-size)
            [ "$#" -ge 2 ] || { echo "error: --whisper-size needs a value" >&2; exit 2; }
            WHISPER_SIZE="$2"
            shift 2
            ;;
        --python)
            [ "$#" -ge 2 ] || { echo "error: --python needs a value" >&2; exit 2; }
            PYTHON="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown argument '$1'" >&2
            usage >&2
            exit 2
            ;;
    esac
done

case "$WHISPER_SIZE" in
    ''|*/*|*..*)
        echo "error: --whisper-size must be a model size such as 'base', got '$WHISPER_SIZE'" >&2
        exit 2
        ;;
esac

if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "error: '$PYTHON' not found. Install Python 3.10+ or pass --python." >&2
    exit 2
fi

if ! "$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
    echo "error: $("$PYTHON" -V 2>&1) is too old; scripts/download_models.py needs 3.10+." >&2
    exit 2
fi

RUN_PYTHON="$PYTHON"
TMP_VENV=""
cleanup() {
    if [ -n "$TMP_VENV" ]; then
        rm -rf "$TMP_VENV"
    fi
}
trap cleanup EXIT

if ! "$PYTHON" -c 'import httpx, huggingface_hub' >/dev/null 2>&1; then
    TMP_VENV="$(mktemp -d "${TMPDIR:-/tmp}/speech-models-venv.XXXXXX")"
    echo "Creating a temporary virtualenv for the downloader in $TMP_VENV ..."
    "$PYTHON" -m venv "$TMP_VENV"
    "$TMP_VENV/bin/python" -m pip install --quiet --disable-pip-version-check \
        httpx huggingface_hub
    RUN_PYTHON="$TMP_VENV/bin/python"
fi

mkdir -p "$MODELS_DIR"
MODELS_DIR="$(cd "$MODELS_DIR" && pwd)"

echo "Downloading speech-core models into $MODELS_DIR (whisper: $WHISPER_SIZE) ..."
if ! MODELS_DIR="$MODELS_DIR" WHISPER_MODEL_SIZE="$WHISPER_SIZE" \
    "$RUN_PYTHON" "$SCRIPT_DIR/download_models.py"; then
    echo >&2
    echo "error: the model download failed (see above). Files already fetched are kept" >&2
    echo "       and skipped on the next run. If this network blocks github.com or" >&2
    echo "       huggingface.co, run this script on a machine that can reach both." >&2
    exit 1
fi

# Confirm the result rather than trusting the downloader's exit status alone:
# the archive step packs whatever is here, and a partial set would only fail
# much later, on the restricted host.
missing=""
set -- "$MODELS_DIR"/kokoro*.onnx
[ -f "$1" ] || missing="$missing kokoro*.onnx"
set -- "$MODELS_DIR"/voices*.bin
[ -f "$1" ] || missing="$missing voices*.bin"
for f in $WHISPER_FILES; do
    [ -s "$MODELS_DIR/whisper-$WHISPER_SIZE/$f" ] || missing="$missing whisper-$WHISPER_SIZE/$f"
done

if [ -n "$missing" ]; then
    echo "error: download finished but these files are missing from $MODELS_DIR:$missing" >&2
    exit 1
fi

echo
echo "Models ready in $MODELS_DIR."
echo "Next: scripts/models-archive.sh --models-dir \"$MODELS_DIR\""
