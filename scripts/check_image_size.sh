#!/usr/bin/env bash
#
# Assert the speech-core image stays inside the documented edge budget.
#
# The product direction is edge-first: the reference deployment must run on a
# developer laptop and one small node.  The image therefore has a hard,
# documented size ceiling.  This script is the enforcement point — run it in CI
# after `docker build`, and the build fails if the ceiling is exceeded.
#
# Ceiling: 2048 MiB (2 GiB) uncompressed, per docs/architecture/speech-platform.md
# ("Deployment profiles").  The constant lives here and is read by
# tests/test_image_budget.py so the documentation and the check cannot drift.
# Tighten it when a build lands comfortably below the ceiling; never raise it
# without recording why in the document.
#
# Usage:
#   scripts/check_image_size.sh                     # inspect the default image
#   scripts/check_image_size.sh --build             # build first, then inspect
#   scripts/check_image_size.sh --image NAME        # inspect a specific image
#   scripts/check_image_size.sh --max-mb 1900       # override the ceiling
#
# Exit codes:
#   0  the image is at or below the ceiling
#   1  the image exceeds the ceiling
#   2  the check could not run (no docker, no image, bad usage)
#
set -euo pipefail

DEFAULT_CEILING_MB=2048
CEILING_MB="${SPEECH_CORE_MAX_IMAGE_MB:-$DEFAULT_CEILING_MB}"
IMAGE="reactory-speech-service:speech-core"
BUILD=0

usage() {
    sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --image)
            [ "$#" -ge 2 ] || { echo "error: --image needs a value" >&2; exit 2; }
            IMAGE="$2"
            shift 2
            ;;
        --max-mb)
            [ "$#" -ge 2 ] || { echo "error: --max-mb needs a value" >&2; exit 2; }
            CEILING_MB="$2"
            shift 2
            ;;
        --build)
            BUILD=1
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

case "$CEILING_MB" in
    ''|*[!0-9]*)
        echo "error: --max-mb must be a positive integer, got '$CEILING_MB'" >&2
        exit 2
        ;;
esac

if ! command -v docker >/dev/null 2>&1; then
    echo "error: docker is not available, so the image size cannot be checked." >&2
    echo "       Install docker or run this step in a CI job that has it." >&2
    exit 2
fi

if [ "$BUILD" -eq 1 ]; then
    echo "Building $IMAGE ..."
    docker build -t "$IMAGE" .
fi

if ! SIZE_BYTES=$(docker image inspect --format '{{.Size}}' "$IMAGE" 2>/dev/null); then
    echo "error: image '$IMAGE' was not found locally. Build it first or pass --build." >&2
    exit 2
fi

case "$SIZE_BYTES" in
    ''|*[!0-9]*)
        echo "error: could not read a numeric size for '$IMAGE' (got '$SIZE_BYTES')" >&2
        exit 2
        ;;
esac

# Round up to whole MiB so the reported number is never flattering.
SIZE_MB=$(( (SIZE_BYTES + 1048575) / 1048576 ))

printf 'speech-core image: %s\n' "$IMAGE"
printf 'size:              %s MiB (%s bytes)\n' "$SIZE_MB" "$SIZE_BYTES"
printf 'ceiling:           %s MiB\n' "$CEILING_MB"

if [ "$SIZE_MB" -gt "$CEILING_MB" ]; then
    printf 'FAIL: image exceeds the documented edge budget by %s MiB\n' \
        "$(( SIZE_MB - CEILING_MB ))" >&2
    echo "      See docs/architecture/speech-platform.md (Deployment profiles)." >&2
    exit 1
fi

echo "OK: image is within the documented edge budget."
