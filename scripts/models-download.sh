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
# With --include-chatterbox it also fetches the optional Chatterbox weights (the
# speech-expressive engine used for voice cloning) into <models-dir>/chatterbox
# through scripts/download_chatterbox.py: the standard variant, about 3.2 GB,
# with a manifest.json of SHA-256 hashes the engine checks before loading.
#
# Python 3.10+ is required. When the chosen interpreter cannot import what the
# downloaders need (httpx and huggingface_hub; plus numpy, soundfile and
# pydantic-settings for Chatterbox, never torch), a throwaway virtualenv is
# created for the run and removed afterwards; the repository's own .venv is
# never modified.
#
# Usage:
#   scripts/models-download.sh [--models-dir DIR] [--whisper-size SIZE]
#                              [--include-chatterbox] [--python PYTHON]
#
#   --models-dir DIR      where to download (default: <repo>/models)
#   --whisper-size SIZE   faster-whisper size: tiny, base, small, ... (default:
#                         $WHISPER_MODEL_SIZE, else base — the service default)
#   --include-chatterbox  also download the Chatterbox weights (about 3.2 GB)
#   --python PYTHON       interpreter to run the downloader (default: python3)
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
INCLUDE_CHATTERBOX=0

#: The files faster-whisper needs in a model directory; mirrors WHISPER_FILES in
#: scripts/download_models.py.
WHISPER_FILES="config.json model.bin tokenizer.json vocabulary.txt"

#: The files the standard Chatterbox variant loads, plus the manifest that
#: records their hashes; mirrors VARIANT_FILES["standard"] in
#: app/engines/chatterbox.py.
CHATTERBOX_FILES="manifest.json ve.safetensors t3_cfg.safetensors s3gen.safetensors tokenizer.json conds.pt"

usage() {
    sed -n '2,45p' "$0" | sed 's/^# \{0,1\}//'
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
        --include-chatterbox)
            INCLUDE_CHATTERBOX=1
            shift
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

# download_chatterbox.py imports app.engines.chatterbox for the variant's file
# list and manifest format; that import needs these, but not torch.
IMPORTS="httpx, huggingface_hub"
PACKAGES="httpx huggingface_hub"
if [ "$INCLUDE_CHATTERBOX" -eq 1 ]; then
    IMPORTS="$IMPORTS, numpy, soundfile, pydantic_settings"
    PACKAGES="$PACKAGES numpy soundfile pydantic-settings"
fi

if ! "$PYTHON" -c "import $IMPORTS" >/dev/null 2>&1; then
    TMP_VENV="$(mktemp -d "${TMPDIR:-/tmp}/speech-models-venv.XXXXXX")"
    echo "Creating a temporary virtualenv for the downloader in $TMP_VENV ..."
    "$PYTHON" -m venv "$TMP_VENV"
    # shellcheck disable=SC2086  # PACKAGES is a word list
    "$TMP_VENV/bin/python" -m pip install --quiet --disable-pip-version-check $PACKAGES
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

if [ "$INCLUDE_CHATTERBOX" -eq 1 ]; then
    echo "Downloading Chatterbox (standard, about 3.2 GB) into $MODELS_DIR/chatterbox ..."
    if ! "$RUN_PYTHON" "$SCRIPT_DIR/download_chatterbox.py" \
        --variant standard --models-dir "$MODELS_DIR/chatterbox"; then
        echo >&2
        echo "error: the Chatterbox download failed (see above). The Kokoro and Whisper" >&2
        echo "       models above are in place; re-run with --include-chatterbox to retry." >&2
        exit 1
    fi
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
if [ "$INCLUDE_CHATTERBOX" -eq 1 ]; then
    for f in $CHATTERBOX_FILES; do
        [ -s "$MODELS_DIR/chatterbox/$f" ] || missing="$missing chatterbox/$f"
    done
fi

if [ -n "$missing" ]; then
    echo "error: download finished but these files are missing from $MODELS_DIR:$missing" >&2
    exit 1
fi

echo
echo "Models ready in $MODELS_DIR."
if [ "$INCLUDE_CHATTERBOX" -eq 1 ]; then
    echo "Next: scripts/models-archive.sh --models-dir \"$MODELS_DIR\" --include-chatterbox"
else
    echo "Next: scripts/models-archive.sh --models-dir \"$MODELS_DIR\""
fi
