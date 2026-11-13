"""Pipeline scratch state must never be tracked in git.

`.ai-agent/` holds ProcessAITicket's working state: the completion report, the
review verdict, the publish decision. It is written inside a worktree and is
meaningful only for the run that produced it.

It was tracked once and the consequences were not cosmetic. A later worktree
checked it out from the base branch and inherited the previous ticket's
`review-normalised.json` — including `"verdict": "approved"` — so a fresh run
presented another ticket's approval as its own. Any consumer that trusted that
file could have published unreviewed code. See PROCESS-003.

This test exists so the directory cannot be re-tracked by a stray `git add -A`.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRATCH_PATHS = (".ai-agent",)


def _tracked(path: str) -> list[str]:
    """Files under *path* that git tracks. Empty when nothing is tracked."""
    try:
        out = subprocess.run(
            ["git", "ls-files", "--", path],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover
        pytest.skip(f"git unavailable: {exc}")
    if out.returncode != 0:
        pytest.skip(f"not a git checkout: {out.stderr.strip()[:120]}")
    return [line for line in out.stdout.splitlines() if line.strip()]


@pytest.mark.parametrize("path", SCRATCH_PATHS)
def test_pipeline_scratch_state_is_not_tracked(path: str) -> None:
    tracked = _tracked(path)
    assert tracked == [], (
        f"{path}/ is pipeline scratch state and must not be committed; found "
        f"{len(tracked)} tracked file(s): {tracked[:5]}. A tracked {path}/ is "
        "checked out into every later worktree, where a stale "
        "review-normalised.json can present another ticket's approval as this "
        "run's verdict (PROCESS-003)."
    )


@pytest.mark.parametrize("path", SCRATCH_PATHS)
def test_pipeline_scratch_state_is_ignored(path: str) -> None:
    """Ignoring it is the durable fix: git add -A must not stage it."""
    try:
        out = subprocess.run(
            ["git", "check-ignore", "-v", f"{path}/report.json"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover
        pytest.skip(f"git unavailable: {exc}")
    if out.returncode not in (0, 1):
        pytest.skip(f"check-ignore unusable: {out.stderr.strip()[:120]}")
    assert out.returncode == 0, (
        f"{path}/ is not covered by .gitignore, so `git add -A` can stage it. "
        f"Offending path: {path}/report.json"
    )
