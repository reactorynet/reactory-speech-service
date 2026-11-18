#!/usr/bin/env bash
#
# Pack the speech-core models into one checksummed archive that can be copied to
# a host whose network blocks the model sources (step 2 of the offline model
# bundle; see scripts/models-download.sh for the whole flow).
#
# The archive holds a single top-level directory:
#
#   reactory-speech-models/
#     MANIFEST.txt          when, which whisper models, file count and bytes
#     SHA256SUMS            "<sha256>  <path>" for every model file below
#     kokoro-v1.0.onnx
#     voices-v1.0.bin
#     whisper-base/config.json, model.bin, tokenizer.json, vocabulary.txt
#     chatterbox/manifest.json, *.safetensors, ...   with --include-chatterbox
#     chatterbox-multilingual/manifest.json, ...     with --include-chatterbox-multilingual
#
# and a sibling "<archive>.sha256" is written next to it, so a truncated or
# corrupted copy is caught before anything is unpacked. scripts/models-restore.sh
# checks both.
#
# Packed: kokoro*.onnx and voices*.bin at the top of the models directory, and
# every whisper-*/ and faster-whisper-*/ model directory (a directory that
# contains model.bin). Hidden files (the .cache/ huggingface_hub leaves behind)
# are never packed. The optional Chatterbox weights in chatterbox/ (about
# 3.2 GB, used by the speech-expressive image for voice cloning) are packed only
# with --include-chatterbox, and then chatterbox/manifest.json must be present:
# it is the hash list the engine verifies before loading, so weights without it
# would be refused on the restored host anyway. The same holds for the
# multilingual model in chatterbox-multilingual/ with
# --include-chatterbox-multilingual.
#
# Files are staged as hard links where the filesystem allows it, so packing the
# Chatterbox weights does not first need a second 3.2 GB copy on disk.
#
# Usage:
#   scripts/models-archive.sh [--models-dir DIR] [--output FILE]
#                             [--include-chatterbox]
#                             [--include-chatterbox-multilingual]
#
#   --models-dir DIR      models to pack (default: <repo>/models)
#   --output FILE         archive path (default:
#                         ./reactory-speech-models-<YYYYMMDD>.tar.gz)
#   --include-chatterbox  also pack <models-dir>/chatterbox/
#   --include-chatterbox-multilingual
#                         also pack <models-dir>/chatterbox-multilingual/
#
# Exit codes:
#   0  archive and checksum written
#   1  the models directory is incomplete (no Kokoro model, voices or Whisper;
#      with an --include-chatterbox* flag, no manifest.json for that model)
#   2  the archive could not be written (bad usage, output exists, no tools)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

#: Top-level directory inside the archive; scripts/models-restore.sh expects it.
BUNDLE_NAME="reactory-speech-models"

MODELS_DIR="$REPO_ROOT/models"
OUTPUT=""
INCLUDE_CHATTERBOX=0
INCLUDE_MULTILINGUAL=0

usage() {
    sed -n '2,52p' "$0" | sed 's/^# \{0,1\}//'
}

sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        shasum -a 256 "$1" | awk '{print $1}'
    fi
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --models-dir)
            [ "$#" -ge 2 ] || { echo "error: --models-dir needs a value" >&2; exit 2; }
            MODELS_DIR="$2"
            shift 2
            ;;
        --output)
            [ "$#" -ge 2 ] || { echo "error: --output needs a value" >&2; exit 2; }
            OUTPUT="$2"
            shift 2
            ;;
        --include-chatterbox)
            INCLUDE_CHATTERBOX=1
            shift
            ;;
        --include-chatterbox-multilingual)
            INCLUDE_MULTILINGUAL=1
            shift
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

for tool in tar gzip awk; do
    command -v "$tool" >/dev/null 2>&1 || { echo "error: '$tool' is required" >&2; exit 2; }
done
if ! command -v sha256sum >/dev/null 2>&1 && ! command -v shasum >/dev/null 2>&1; then
    echo "error: sha256sum or shasum is required" >&2
    exit 2
fi

if [ ! -d "$MODELS_DIR" ]; then
    echo "error: models directory '$MODELS_DIR' does not exist." >&2
    echo "       Run scripts/models-download.sh first." >&2
    exit 1
fi
MODELS_DIR="$(cd "$MODELS_DIR" && pwd)"

[ -n "$OUTPUT" ] || OUTPUT="$PWD/$BUNDLE_NAME-$(date -u +%Y%m%d).tar.gz"
case "$OUTPUT" in
    *.tar.gz) ;;
    *) echo "error: --output must end in .tar.gz, got '$OUTPUT'" >&2; exit 2 ;;
esac
if [ -e "$OUTPUT" ]; then
    echo "error: '$OUTPUT' already exists; remove it or pass a different --output." >&2
    exit 2
fi
OUTPUT_DIR="$(dirname "$OUTPUT")"
mkdir -p "$OUTPUT_DIR"
OUTPUT="$(cd "$OUTPUT_DIR" && pwd)/$(basename "$OUTPUT")"

# Collect the files to pack, as paths relative to MODELS_DIR, one per line.
FILES="$(
    cd "$MODELS_DIR"
    for f in kokoro*.onnx voices*.bin; do
        [ -f "$f" ] && printf '%s\n' "$f"
    done
    for d in whisper-* faster-whisper-*; do
        [ -d "$d" ] && [ -f "$d/model.bin" ] || continue
        # -L: a model directory may itself be a symlink (see .gitignore).
        find -L "$d" -type f ! -path '*/.*' | LC_ALL=C sort
    done
    if [ "$INCLUDE_CHATTERBOX" -eq 1 ] && [ -d chatterbox ]; then
        find -L chatterbox -type f ! -path '*/.*' | LC_ALL=C sort
    fi
    if [ "$INCLUDE_MULTILINGUAL" -eq 1 ] && [ -d chatterbox-multilingual ]; then
        find -L chatterbox-multilingual -type f ! -path '*/.*' | LC_ALL=C sort
    fi
    true
)"

has() { printf '%s\n' "$FILES" | grep -q "$1"; }
missing=""
has '^kokoro.*\.onnx$' || missing="$missing kokoro*.onnx"
has '^voices.*\.bin$' || missing="$missing voices*.bin"
has '^\(faster-\)\{0,1\}whisper-[^/]*/model\.bin$' || missing="$missing whisper-<size>/model.bin"
if [ "$INCLUDE_CHATTERBOX" -eq 1 ]; then
    has '^chatterbox/manifest\.json$' || missing="$missing chatterbox/manifest.json (run scripts/models-download.sh --include-chatterbox)"
fi
if [ "$INCLUDE_MULTILINGUAL" -eq 1 ]; then
    has '^chatterbox-multilingual/manifest\.json$' || missing="$missing chatterbox-multilingual/manifest.json (run scripts/models-download.sh --include-chatterbox-multilingual)"
fi
if [ -n "$missing" ]; then
    echo "error: $MODELS_DIR is incomplete; missing:$missing" >&2
    echo "       Run scripts/models-download.sh first." >&2
    exit 1
fi

STAGE="$(mktemp -d "${TMPDIR:-/tmp}/speech-models-archive.XXXXXX")"
trap 'rm -rf "$STAGE"' EXIT
BUNDLE="$STAGE/$BUNDLE_NAME"
mkdir -p "$BUNDLE"

count=0
bytes=0
: > "$BUNDLE/SHA256SUMS"
while IFS= read -r rel; do
    [ -n "$rel" ] || continue
    mkdir -p "$BUNDLE/$(dirname "$rel")"
    # The archive must hold real files only (models-restore.sh refuses links).
    # A hard link is a real file to tar, and it saves copying gigabytes of
    # weights; a symlink is copied through, and so is anything ln refuses (a
    # different filesystem, for example).
    src="$MODELS_DIR/$rel"
    if [ -L "$src" ] || ! ln "$src" "$BUNDLE/$rel" 2>/dev/null; then
        cp -L "$src" "$BUNDLE/$rel"
    fi
    printf '%s  %s\n' "$(sha256_of "$BUNDLE/$rel")" "$rel" >> "$BUNDLE/SHA256SUMS"
    size=$(wc -c < "$BUNDLE/$rel" | tr -d ' ')
    count=$((count + 1))
    bytes=$((bytes + size))
done <<EOF
$FILES
EOF

whisper_models="$(printf '%s\n' "$FILES" | sed -n 's#^\([^/]*\)/model\.bin$#\1#p' | tr '\n' ' ')"
# manifest.json is written by json.dumps(indent=2): one key per line.
manifest_variant() {
    if [ -f "$1" ]; then
        variant="$(sed -n 's/^ *"variant": *"\([^"]*\)".*/\1/p' "$1" | head -n 1)"
        printf '%s' "${variant:-unknown variant}"
    else
        printf 'none'
    fi
}
chatterbox="$(manifest_variant "$BUNDLE/chatterbox/manifest.json")"
chatterbox_multilingual="$(manifest_variant "$BUNDLE/chatterbox-multilingual/manifest.json")"
cat > "$BUNDLE/MANIFEST.txt" <<EOF
Reactory speech-core model bundle
created_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)
whisper_models: ${whisper_models% }
chatterbox: $chatterbox
chatterbox-multilingual: $chatterbox_multilingual
files: $count
bytes: $bytes
restore: scripts/models-restore.sh <this archive>
EOF

echo "Packing $count files ($((bytes / 1024 / 1024)) MiB) into $OUTPUT ..."
# COPYFILE_DISABLE stops macOS tar adding AppleDouble (._*) entries.
COPYFILE_DISABLE=1 tar -C "$STAGE" -czf "$OUTPUT" "$BUNDLE_NAME"

printf '%s  %s\n' "$(sha256_of "$OUTPUT")" "$(basename "$OUTPUT")" > "$OUTPUT.sha256"

echo
echo "Wrote:"
echo "  $OUTPUT"
echo "  $OUTPUT.sha256"
echo
echo "Copy both files to the target host, then run there:"
echo "  scripts/models-restore.sh $(basename "$OUTPUT")"
