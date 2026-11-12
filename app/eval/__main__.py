"""``python -m app.eval`` entry point."""

from __future__ import annotations

import sys

from app.eval.cli import main

if __name__ == "__main__":
    sys.exit(main())
