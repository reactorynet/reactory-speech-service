#!/usr/bin/env python3
"""Generate the per-deployment key material for consent-first voice cloning.

Run this **once per instance**, store the output in the deployment's secret
volume, and never commit it.  The service deliberately does not generate these
at boot: a regenerated signing key invalidates every existing consent signature
and provenance manifest (ADR-0003 #2 and #4a).

Usage::

    python scripts/generate_clone_keys.py --output-dir data

creates::

    data/keys/clone_signing_key.pem   Ed25519 private key (0600)
    data/keys/clone_reference.key     32 random bytes for AES-256-GCM (0600)
    data/clone_approvers.json         {"clone_approvers": []} template (0600)

Then edit ``clone_approvers.json`` to name the people allowed to approve a
clone.  An empty list leaves cloning **unavailable**, which is the intended
fail-closed default.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services import clone_crypto  # noqa: E402


def _write_private(path: Path, payload: bytes, *, overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | (os.O_TRUNC if overwrite else os.O_EXCL)
    descriptor = os.open(path, flags, stat.S_IRUSR | stat.S_IWUSR)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:  # pragma: no cover - cleanup on a failed write
        path.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory that will receive keys/ and clone_approvers.json (default: data)",
    )
    parser.add_argument(
        "--instance-id",
        default="",
        help="Instance id recorded in signatures; defaults to key-<signing key fingerprint> at runtime",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing key files. This invalidates every existing signature.",
    )
    args = parser.parse_args(argv)

    root = Path(args.output_dir)
    key_dir = root / "keys"
    signing_path = key_dir / "clone_signing_key.pem"
    reference_path = key_dir / "clone_reference.key"
    approvers_path = root / "clone_approvers.json"

    existing = [path for path in (signing_path, reference_path) if path.exists()]
    if existing and not args.force:
        print(
            "Refusing to overwrite existing key material: "
            + ", ".join(str(path) for path in existing)
            + "\nA new signing key invalidates every existing consent record and manifest. "
            "Pass --force only if that is what you intend.",
            file=sys.stderr,
        )
        return 1

    if not signing_path.exists() or args.force:
        _write_private(
            signing_path,
            clone_crypto.generate_signing_key_pem(),
            overwrite=args.force,
        )
    if not reference_path.exists() or args.force:
        _write_private(
            reference_path,
            clone_crypto.generate_reference_key(),
            overwrite=args.force,
        )

    template = {
        "clone_approvers": [],
        "_comment": (
            "Deployed per instance and never committed. List the email addresses "
            "allowed to approve a clone. An empty list makes cloning unavailable."
        ),
    }
    if args.instance_id:
        template["_instance_id"] = args.instance_id
    if not approvers_path.exists() or args.force:
        _write_private(
            approvers_path,
            (json.dumps(template, indent=2) + "\n").encode("utf-8"),
            overwrite=args.force,
        )

    print(f"Wrote {signing_path} (Ed25519 signing key)")
    print(f"Wrote {reference_path} (AES-256-GCM reference key)")
    print(f"Wrote {approvers_path} (add approver emails before cloning is available)")
    print(
        "\nThese files are gitignored. Supplying an empty approver list means cloning is\n"
        "UNAVAILABLE, not unrestricted. See docs/voices/voice-cloning.md."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
