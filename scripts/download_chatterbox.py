#!/usr/bin/env python3
"""Download the optional Chatterbox weights on demand (SPEECH-005).

This is deliberately an *operator command*, not a build step:

* the default `speech-core` image must not grow or gain torch;
* the weights are 3.1 GB and are never downloaded at container build time;
* the service must start (and serve Kokoro) with the model absent.

The download writes the weights into the models volume and records the licence
(MIT), the upstream repository and a SHA-256 per file in `manifest.json`. The
adapter verifies that manifest before loading, so a corrupt or replaced file is a
clear error and a Kokoro fallback, not a crash.

Usage:
    python scripts/download_chatterbox.py --variant standard
    python scripts/download_chatterbox.py --variant turbo --models-dir /models/chatterbox-turbo
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.config import settings  # noqa: E402
from app.engines.chatterbox import VARIANT_REPOS, download_model  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        default="standard",
        choices=sorted(VARIANT_REPOS),
        help="Chatterbox variant to download (default: standard)",
    )
    parser.add_argument(
        "--models-dir",
        default=None,
        help="Target directory (default: CHATTERBOX_MODEL_DIR, models/chatterbox)",
    )
    parser.add_argument("--repo-id", default=None, help="Override the upstream repository")
    parser.add_argument("--revision", default=None, help="Pin an upstream revision")
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "overwrite a models directory that already holds a different variant "
            "(the default refuses, because the two variants share file names)"
        ),
    )
    args = parser.parse_args(argv)

    root = args.models_dir or str(settings.chatterbox_model_dir_resolved)
    manifest = download_model(
        root,
        variant=args.variant,
        repo_id=args.repo_id,
        revision=args.revision,
        force=args.force,
    )
    print(
        f"Chatterbox {args.variant} downloaded to {root}\n"
        f"  license:     {manifest['license']}\n"
        f"  attribution: {manifest['attribution']}\n"
        f"  files:       {len(manifest['files'])} (SHA-256 recorded in manifest.json)\n"
        "Mount this directory into the speech-expressive container and set "
        "CHATTERBOX_ENABLED=true."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
