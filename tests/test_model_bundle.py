"""The offline model bundle round-trips and refuses anything it cannot verify.

``scripts/models-download.sh`` / ``models-archive.sh`` / ``models-restore.sh``
carry the speech-core models to a host whose network blocks the model sources.
The restore runs on that host against a file that has been copied around, so
these tests pin that it

* reproduces the models byte for byte;
* rejects a corrupted archive, a tampered file, path traversal and links
  before writing anything to the models directory;
* never silently overwrites a different model.

The models here are small fake files: no network, no real weights.
"""

from __future__ import annotations

import hashlib
import io
import os
import subprocess
import tarfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"
DOWNLOAD = SCRIPTS / "models-download.sh"
ARCHIVE = SCRIPTS / "models-archive.sh"
RESTORE = SCRIPTS / "models-restore.sh"
BUNDLE = "reactory-speech-models"

FAKE_MODELS = {
    "kokoro-v1.0.onnx": b"onnx-weights",
    "voices-v1.0.bin": b"voice-pack",
    "whisper-base/config.json": b"{}",
    "whisper-base/model.bin": b"whisper-weights",
    "whisper-base/tokenizer.json": b'{"tok": 1}',
    "whisper-base/vocabulary.txt": b"a\nb\n",
}


def _run(script: Path, *args, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(script), *map(str, args)],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def _write_models(root: Path, files: dict[str, bytes] = FAKE_MODELS) -> Path:
    for rel, data in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return root


def _tree(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()
    }


@pytest.fixture
def archive(tmp_path: Path) -> Path:
    models = _write_models(tmp_path / "src-models")
    # Noise that must not be packed.
    _write_models(models, {"whisper-base/.cache/huggingface/lock": b"x", ".gitkeep": b""})
    out = tmp_path / "out" / "bundle.tar.gz"

    completed = _run(ARCHIVE, "--models-dir", models, "--output", out, cwd=tmp_path)

    assert completed.returncode == 0, completed.stderr
    return out


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hand_made_archive(path: Path, members: list[tarfile.TarInfo], payloads: dict[str, bytes]):
    """Write a .tar.gz directly, for archives the archive script would never make."""
    with tarfile.open(path, "w:gz") as tar:
        for info in members:
            data = payloads.get(info.name)
            if data is not None:
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
            else:
                tar.addfile(info)


def _file_member(name: str) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.type = tarfile.REGTYPE
    info.mode = 0o644
    return info


@pytest.mark.parametrize("script", [DOWNLOAD, ARCHIVE, RESTORE])
def test_scripts_are_executable_valid_and_documented(script: Path):
    assert os.access(script, os.X_OK), f"{script.name} must be executable"
    assert subprocess.run(["bash", "-n", str(script)], check=False).returncode == 0

    completed = _run(script, "--help", cwd=REPO_ROOT)

    assert completed.returncode == 0, completed.stderr
    assert "Usage:" in completed.stdout
    assert "Exit codes:" in completed.stdout


def test_archive_then_restore_reproduces_the_models_exactly(archive: Path, tmp_path: Path):
    target = tmp_path / "restored"

    completed = _run(RESTORE, archive, "--models-dir", target, cwd=tmp_path)

    assert completed.returncode == 0, completed.stderr
    assert _tree(target) == FAKE_MODELS
    assert "6 installed" in completed.stdout


def test_archive_holds_one_bundle_directory_with_checksums_and_manifest(archive: Path):
    with tarfile.open(archive) as tar:
        names = tar.getnames()
        sums = tar.extractfile(f"{BUNDLE}/SHA256SUMS").read().decode()
        manifest = tar.extractfile(f"{BUNDLE}/MANIFEST.txt").read().decode()

    assert all(n == BUNDLE or n.startswith(f"{BUNDLE}/") for n in names)
    assert not any("/." in n or "._" in n for n in names), "hidden files must not be packed"
    assert sorted(line.split("  ", 1)[1] for line in sums.splitlines()) == sorted(FAKE_MODELS)
    for line in sums.splitlines():
        digest, rel = line.split("  ", 1)
        assert digest == _sha256(FAKE_MODELS[rel])
    assert "whisper_models: whisper-base" in manifest
    assert "files: 6" in manifest

    sidecar = Path(f"{archive}.sha256").read_text().split()
    assert sidecar == [_sha256(archive.read_bytes()), archive.name]


def test_archive_refuses_an_incomplete_models_directory(tmp_path: Path):
    incomplete = dict(FAKE_MODELS)
    del incomplete["whisper-base/model.bin"]
    models = _write_models(tmp_path / "models", incomplete)

    completed = _run(
        ARCHIVE, "--models-dir", models, "--output", tmp_path / "b.tar.gz", cwd=tmp_path
    )

    assert completed.returncode == 1
    assert "whisper-<size>/model.bin" in completed.stderr
    assert not (tmp_path / "b.tar.gz").exists()


def test_archive_never_overwrites_an_existing_archive(archive: Path, tmp_path: Path):
    before = archive.read_bytes()

    completed = _run(
        ARCHIVE, "--models-dir", tmp_path / "src-models", "--output", archive, cwd=tmp_path
    )

    assert completed.returncode == 2
    assert archive.read_bytes() == before


def test_restore_rejects_a_corrupted_copy_and_writes_nothing(archive: Path, tmp_path: Path):
    data = bytearray(archive.read_bytes())
    data[len(data) // 2] ^= 0xFF
    archive.write_bytes(bytes(data))
    target = tmp_path / "restored"

    completed = _run(RESTORE, archive, "--models-dir", target, cwd=tmp_path)

    assert completed.returncode == 1
    assert "archive checksum mismatch" in completed.stderr
    assert not target.exists()


def test_restore_rejects_a_file_that_does_not_match_its_checksum(tmp_path: Path):
    sums = f"{_sha256(b'the real weights')}  kokoro-v1.0.onnx\n".encode()
    bad = tmp_path / "tampered.tar.gz"
    _hand_made_archive(
        bad,
        [_file_member(f"{BUNDLE}/SHA256SUMS"), _file_member(f"{BUNDLE}/kokoro-v1.0.onnx")],
        {f"{BUNDLE}/SHA256SUMS": sums, f"{BUNDLE}/kokoro-v1.0.onnx": b"something else"},
    )
    target = tmp_path / "restored"

    completed = _run(RESTORE, bad, "--models-dir", target, cwd=tmp_path)

    assert completed.returncode == 1
    assert "checksum mismatch for 'kokoro-v1.0.onnx'" in completed.stderr
    assert not (target / "kokoro-v1.0.onnx").exists()


@pytest.mark.parametrize(
    "name",
    [
        f"{BUNDLE}/../escaped.onnx",
        "/tmp/absolute.onnx",
        "not-a-bundle/kokoro-v1.0.onnx",
    ],
)
def test_restore_rejects_entries_outside_the_bundle(tmp_path: Path, name: str):
    bad = tmp_path / "evil.tar.gz"
    _hand_made_archive(bad, [_file_member(name)], {name: b"payload"})
    target = tmp_path / "models" / "restored"

    completed = _run(RESTORE, bad, "--models-dir", target, cwd=tmp_path)

    assert completed.returncode == 1
    assert not (tmp_path / "models" / "escaped.onnx").exists()
    assert not target.exists()


@pytest.mark.parametrize("link_type", [tarfile.SYMTYPE, tarfile.LNKTYPE])
def test_restore_rejects_links(tmp_path: Path, link_type: bytes):
    link = tarfile.TarInfo(f"{BUNDLE}/kokoro-v1.0.onnx")
    link.type = link_type
    link.linkname = "/etc/passwd" if link_type == tarfile.SYMTYPE else f"{BUNDLE}/SHA256SUMS"
    bad = tmp_path / "links.tar.gz"
    _hand_made_archive(
        bad, [_file_member(f"{BUNDLE}/SHA256SUMS"), link], {f"{BUNDLE}/SHA256SUMS": b""}
    )
    target = tmp_path / "restored"

    completed = _run(RESTORE, bad, "--models-dir", target, cwd=tmp_path)

    assert completed.returncode == 1
    assert "links or special files" in completed.stderr
    assert not target.exists()


def test_restore_refuses_to_overwrite_a_different_model_without_force(
    archive: Path, tmp_path: Path
):
    target = _write_models(tmp_path / "restored", {"whisper-base/model.bin": b"a different model"})

    refused = _run(RESTORE, archive, "--models-dir", target, cwd=tmp_path)

    assert refused.returncode == 1
    assert "whisper-base/model.bin" in refused.stderr
    assert _tree(target) == {"whisper-base/model.bin": b"a different model"}

    forced = _run(RESTORE, archive, "--models-dir", target, "--force", cwd=tmp_path)

    assert forced.returncode == 0, forced.stderr
    assert _tree(target) == FAKE_MODELS


def test_restoring_twice_is_harmless(archive: Path, tmp_path: Path):
    target = tmp_path / "restored"
    assert _run(RESTORE, archive, "--models-dir", target, cwd=tmp_path).returncode == 0

    again = _run(RESTORE, archive, "--models-dir", target, cwd=tmp_path)

    assert again.returncode == 0, again.stderr
    assert "0 installed, 6 already up to date" in again.stdout
    assert _tree(target) == FAKE_MODELS


def test_restore_without_the_sidecar_warns_but_still_verifies_every_file(
    archive: Path, tmp_path: Path
):
    Path(f"{archive}.sha256").unlink()
    target = tmp_path / "restored"

    completed = _run(RESTORE, archive, "--models-dir", target, cwd=tmp_path)

    assert completed.returncode == 0, completed.stderr
    assert "skipping the whole-archive check" in completed.stderr
    assert "Verified 6 model files" in completed.stdout


def test_download_rejects_a_whisper_size_that_is_a_path(tmp_path: Path):
    completed = _run(DOWNLOAD, "--whisper-size", "../etc", "--models-dir", tmp_path, cwd=tmp_path)

    assert completed.returncode == 2
    assert "must be a model size" in completed.stderr
