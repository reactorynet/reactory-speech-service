"""The speech-core image-size budget is real, documented and enforced.

The edge profile has a hard image-size ceiling: if the image grows, `make
docker-build` starts producing something that no longer fits a small node.  This
file asserts that

* the ceiling constant exists in ``scripts/check_image_size.sh`` and matches the
  value documented in ``docs/architecture/speech-platform.md``;
* the script actually fails when the image is over the ceiling (tested with a
  stubbed ``docker`` executable, so the assertion itself is exercised without a
  docker daemon);
* CI runs the check, so the budget cannot silently regress.
"""

from __future__ import annotations

import os
import re
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "check_image_size.sh"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"
ARCHITECTURE = REPO_ROOT / "docs" / "architecture" / "speech-platform.md"
DOCKERFILE = REPO_ROOT / "Dockerfile"


def _ceiling_mb() -> int:
    match = re.search(r"DEFAULT_CEILING_MB=(\d+)", SCRIPT.read_text(encoding="utf-8"))
    assert match, "scripts/check_image_size.sh must declare DEFAULT_CEILING_MB"
    return int(match.group(1))


def _run(args, *, env_extra=None) -> subprocess.CompletedProcess:
    env = {**os.environ, **(env_extra or {})}
    return subprocess.run(
        [str(SCRIPT), *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_ceiling_is_declared_documented_and_used_by_ci():
    ceiling = _ceiling_mb()

    assert ceiling > 0
    assert f"{ceiling} MiB" in ARCHITECTURE.read_text(encoding="utf-8")
    assert "scripts/check_image_size.sh" in WORKFLOW.read_text(encoding="utf-8")


def test_the_image_size_check_runs_in_ci_after_a_build():
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "docker build" in workflow
    assert "--image reactory-speech-service:speech-core" in workflow
    assert "scripts/check_image_size.sh" in workflow


def test_the_script_is_executable_and_syntactically_valid():
    assert SCRIPT.is_file()
    assert SCRIPT.stat().st_mode & stat.S_IXUSR, "the script must be executable"

    completed = subprocess.run(
        ["bash", "-n", str(SCRIPT)], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr


def test_the_help_text_states_the_documented_ceiling():
    completed = _run(["--help"])

    assert completed.returncode == 0, completed.stderr
    assert f"{_ceiling_mb()} MiB" in completed.stdout
    assert "edge budget" in completed.stdout


def test_the_build_fails_when_the_image_exceeds_the_ceiling(tmp_path):
    completed = _run(["--max-mb", "100"], env_extra=_fake_docker_env(tmp_path, "200000000"))

    assert completed.returncode == 1, completed.stdout + completed.stderr
    assert "FAIL" in completed.stderr
    assert "exceeds the documented edge budget" in completed.stderr


def test_the_build_passes_when_the_image_is_within_the_ceiling(tmp_path):
    completed = _run(["--max-mb", "500"], env_extra=_fake_docker_env(tmp_path, "200000000"))

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "OK" in completed.stdout
    assert "191 MiB" in completed.stdout


def test_a_missing_image_is_a_harness_error_not_a_pass(tmp_path):
    completed = _run(
        ["--image", "no-such-image"],
        env_extra=_fake_docker_env(tmp_path, "", status=1),
    )

    assert completed.returncode == 2
    assert "not found" in completed.stderr


def test_the_default_profile_does_not_pull_in_a_gpu_runtime():
    """speech-core is the CPU/ONNX profile; torch belongs to speech-expressive."""
    dockerfile = DOCKERFILE.read_text(encoding="utf-8").lower()

    for forbidden in ("torch", "tensorflow", "cuda", "nvidia"):
        assert forbidden not in dockerfile, f"the default image must not install {forbidden}"


def test_the_makefile_exposes_the_image_size_check():
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")

    assert "image-size:" in makefile
    assert "check_image_size.sh" in makefile


def _fake_docker_env(tmp_path: Path, size_bytes: str, *, status: int = 0) -> dict[str, str]:
    """Put a stub ``docker`` on PATH that answers ``image inspect``."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    docker = bin_dir / "docker"
    docker.write_text(
        "#!/usr/bin/env bash\n"
        'if [ "$1" = "image" ] && [ "$2" = "inspect" ]; then\n'
        f"  echo '{size_bytes}'\n"
        f"  exit {status}\n"
        "fi\n"
        "exit 0\n",
        encoding="utf-8",
    )
    docker.chmod(0o755)
    return {"PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"}


@pytest.mark.parametrize("path", [SCRIPT, WORKFLOW, ARCHITECTURE])
def test_the_budget_files_exist(path):
    assert path.is_file(), f"{path} is part of the image-budget contract"
