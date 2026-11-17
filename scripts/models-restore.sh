#!/usr/bin/env bash
#
# Unpack a model bundle made by scripts/models-archive.sh into the models
# directory, verifying it first (step 3 of the offline model bundle; see
# scripts/models-download.sh for the whole flow).
#
# Nothing is written to the models directory until every check passes:
#
#   1. "<archive>.sha256", when it sits next to the archive, must match — this
#      catches a truncated or corrupted copy. Without it a warning is printed
#      and the per-file checks below still run.
#   2. Every archive entry must be a plain file or directory under
#      reactory-speech-models/. Absolute paths, ".." components, symlinks,
#      hard links and device files are refused.
#   3. After unpacking into a temporary directory, every file listed in
#      SHA256SUMS must match its checksum. Only files listed there are copied.
#   4. A file already in the models directory with different content is a
#      conflict and stops the restore, unless --force is given. Identical files
#      are left alone, so re-running a restore is harmless.
#
# The image build then finds the models in place and does not download them:
#
#   podman build -t reactory-speech-service:speech-core .
#
# Usage:
#   scripts/models-restore.sh ARCHIVE [--models-dir DIR] [--force]
#
#   ARCHIVE           reactory-speech-models-<date>.tar.gz
#   --models-dir DIR  where to restore (default: <repo>/models)
#   --force           replace model files whose content differs
#
# Exit codes:
#   0  models restored (or already identical)
#   1  verification failed or a conflict was found; nothing was written
#   2  the restore could not run (bad usage, missing archive, no tools)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

#: Top-level directory inside the archive; written by scripts/models-archive.sh.
BUNDLE_NAME="reactory-speech-models"

MODELS_DIR="$REPO_ROOT/models"
ARCHIVE=""
FORCE=0

usage() {
    sed -n '2,35p' "$0" | sed 's/^# \{0,1\}//'
}

fail() {
    echo "error: $*" >&2
    echo "       Nothing was written to $MODELS_DIR." >&2
    exit 1
}

sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{print $1}'
    else
        shasum -a 256 "$1" | awk '{print $1}'
    fi
}

#: A relative path that stays inside its root: no leading "/", no ".." part.
is_safe_relpath() {
    case "$1" in
        ''|/*|..|../*|*/..|*/../*) return 1 ;;
    esac
    return 0
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --models-dir)
            [ "$#" -ge 2 ] || { echo "error: --models-dir needs a value" >&2; exit 2; }
            MODELS_DIR="$2"
            shift 2
            ;;
        --force)
            FORCE=1
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        -*)
            echo "error: unknown argument '$1'" >&2
            usage >&2
            exit 2
            ;;
        *)
            [ -z "$ARCHIVE" ] || { echo "error: only one archive may be given" >&2; exit 2; }
            ARCHIVE="$1"
            shift
            ;;
    esac
done

if [ -z "$ARCHIVE" ]; then
    echo "error: no archive given" >&2
    usage >&2
    exit 2
fi
if [ ! -f "$ARCHIVE" ]; then
    echo "error: archive '$ARCHIVE' not found" >&2
    exit 2
fi
for tool in tar gzip awk; do
    command -v "$tool" >/dev/null 2>&1 || { echo "error: '$tool' is required" >&2; exit 2; }
done
if ! command -v sha256sum >/dev/null 2>&1 && ! command -v shasum >/dev/null 2>&1; then
    echo "error: sha256sum or shasum is required" >&2
    exit 2
fi

# 1. Whole-archive checksum.
if [ -f "$ARCHIVE.sha256" ]; then
    expected="$(awk 'NR==1 {print $1}' "$ARCHIVE.sha256")"
    actual="$(sha256_of "$ARCHIVE")"
    [ "$expected" = "$actual" ] || fail "archive checksum mismatch (expected $expected, got $actual); the copy is corrupt or incomplete."
    echo "Archive checksum OK."
else
    echo "warning: $ARCHIVE.sha256 not found; skipping the whole-archive check." >&2
    echo "         Per-file checksums inside the archive are still verified." >&2
fi

# 2. Entry names and types, before anything is extracted.
names="$(tar -tzf "$ARCHIVE")" || fail "'$ARCHIVE' is not a readable .tar.gz archive."
while IFS= read -r entry; do
    [ -n "$entry" ] || continue
    is_safe_relpath "$entry" || fail "archive entry '$entry' escapes the bundle directory."
    case "$entry" in
        "$BUNDLE_NAME"|"$BUNDLE_NAME"/*) ;;
        *) fail "archive entry '$entry' is outside $BUNDLE_NAME/; not a model bundle." ;;
    esac
done <<EOF
$names
EOF

# The first character of each verbose listing line is the entry type in both
# GNU tar and bsdtar: '-' file, 'd' directory, 'l' symlink, 'h' hard link, ...
bad_types="$(tar -tvzf "$ARCHIVE" | awk 'substr($0,1,1) != "-" && substr($0,1,1) != "d"')"
[ -z "$bad_types" ] || fail "archive contains links or special files, which a model bundle never has:
$bad_types"

STAGE="$(mktemp -d "${TMPDIR:-/tmp}/speech-models-restore.XXXXXX")"
trap 'rm -rf "$STAGE"' EXIT
tar -C "$STAGE" -xzf "$ARCHIVE"
BUNDLE="$STAGE/$BUNDLE_NAME"
[ -f "$BUNDLE/SHA256SUMS" ] || fail "the archive has no $BUNDLE_NAME/SHA256SUMS; not a model bundle."

# 3. Per-file checksums.
count=0
while IFS= read -r line; do
    [ -n "$line" ] || continue
    hash="${line%%  *}"
    rel="${line#*  }"
    is_safe_relpath "$rel" || fail "SHA256SUMS lists an unsafe path '$rel'."
    [ -f "$BUNDLE/$rel" ] || fail "SHA256SUMS lists '$rel', which is not in the archive."
    [ "$(sha256_of "$BUNDLE/$rel")" = "$hash" ] || fail "checksum mismatch for '$rel'."
    count=$((count + 1))
done < "$BUNDLE/SHA256SUMS"
[ "$count" -gt 0 ] || fail "SHA256SUMS is empty; the archive holds no models."
echo "Verified $count model files."

# 4. Conflicts with what is already installed.
mkdir -p "$MODELS_DIR"
MODELS_DIR="$(cd "$MODELS_DIR" && pwd)"
conflicts=""
while IFS= read -r line; do
    [ -n "$line" ] || continue
    hash="${line%%  *}"
    rel="${line#*  }"
    if [ -e "$MODELS_DIR/$rel" ] && [ "$(sha256_of "$MODELS_DIR/$rel")" != "$hash" ]; then
        conflicts="$conflicts
  $rel"
    fi
done < "$BUNDLE/SHA256SUMS"
if [ -n "$conflicts" ] && [ "$FORCE" -ne 1 ]; then
    fail "these files already exist in $MODELS_DIR with different content:$conflicts
       Re-run with --force to replace them."
fi

# Install. Each file goes to a temporary name first and is renamed into place,
# so an interrupted restore never leaves a half-written model.bin behind.
installed=0
unchanged=0
while IFS= read -r line; do
    [ -n "$line" ] || continue
    hash="${line%%  *}"
    rel="${line#*  }"
    dest="$MODELS_DIR/$rel"
    if [ -f "$dest" ] && [ "$(sha256_of "$dest")" = "$hash" ]; then
        unchanged=$((unchanged + 1))
        continue
    fi
    mkdir -p "$(dirname "$dest")"
    cp "$BUNDLE/$rel" "$dest.restore-tmp"
    mv -f "$dest.restore-tmp" "$dest"
    installed=$((installed + 1))
done < "$BUNDLE/SHA256SUMS"

echo
[ ! -f "$BUNDLE/MANIFEST.txt" ] || sed 's/^/  /' "$BUNDLE/MANIFEST.txt"
echo
echo "Restored into $MODELS_DIR: $installed installed, $unchanged already up to date."
echo "Build the image with the models in place:"
echo "  podman build -t reactory-speech-service:speech-core ."
if grep -q '  chatterbox/' "$BUNDLE/SHA256SUMS"; then
    # Never baked into an image (.dockerignore); the expressive container mounts it.
    echo "Chatterbox weights are in $MODELS_DIR/chatterbox. They are mounted into the"
    echo "speech-expressive container rather than built in:"
    echo "  make expressive-build && make expressive-run"
fi
