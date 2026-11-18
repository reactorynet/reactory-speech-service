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
import re
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


# ---------------------------------------------------------------------------
# Optional Chatterbox weights (--include-chatterbox)
# ---------------------------------------------------------------------------

FAKE_CHATTERBOX = {
    "chatterbox/manifest.json": b'{\n  "engine": "chatterbox",\n  "variant": "standard"\n}\n',
    "chatterbox/ve.safetensors": b"ve",
    "chatterbox/t3_cfg.safetensors": b"t3",
    "chatterbox/s3gen.safetensors": b"s3gen",
    "chatterbox/tokenizer.json": b"{}",
    "chatterbox/conds.pt": b"conds",
}


FAKE_MULTILINGUAL = {
    "chatterbox-multilingual/manifest.json": (
        b'{\n  "engine": "chatterbox",\n  "variant": "multilingual"\n}\n'
    ),
    "chatterbox-multilingual/ve.pt": b"ve",
    "chatterbox-multilingual/t3_mtl23ls_v2.safetensors": b"t3",
    "chatterbox-multilingual/s3gen.pt": b"s3gen",
    "chatterbox-multilingual/grapheme_mtl_merged_expanded_v1.json": b"{}",
    "chatterbox-multilingual/Cangjie5_TC.json": b"[]",
    "chatterbox-multilingual/conds.pt": b"conds",
}


def _archive_names(path: Path) -> list[str]:
    with tarfile.open(path) as tar:
        return tar.getnames()


def test_archive_leaves_chatterbox_out_unless_asked(tmp_path: Path):
    models = _write_models(tmp_path / "models", {**FAKE_MODELS, **FAKE_CHATTERBOX})
    out = tmp_path / "b.tar.gz"

    completed = _run(ARCHIVE, "--models-dir", models, "--output", out, cwd=tmp_path)

    assert completed.returncode == 0, completed.stderr
    assert not any("chatterbox" in n for n in _archive_names(out))
    with tarfile.open(out) as tar:
        assert "chatterbox: none" in tar.extractfile(f"{BUNDLE}/MANIFEST.txt").read().decode()


def test_chatterbox_round_trips_when_included(tmp_path: Path):
    models = _write_models(tmp_path / "models", {**FAKE_MODELS, **FAKE_CHATTERBOX})
    _write_models(models, {"chatterbox/.cache/huggingface/lock": b"x"})
    out = tmp_path / "b.tar.gz"

    packed = _run(
        ARCHIVE, "--models-dir", models, "--output", out, "--include-chatterbox", cwd=tmp_path
    )

    assert packed.returncode == 0, packed.stderr
    with tarfile.open(out) as tar:
        manifest = tar.extractfile(f"{BUNDLE}/MANIFEST.txt").read().decode()
    assert "chatterbox: standard" in manifest
    assert "files: 12" in manifest

    target = tmp_path / "restored"
    restored = _run(RESTORE, out, "--models-dir", target, cwd=tmp_path)

    assert restored.returncode == 0, restored.stderr
    assert _tree(target) == {**FAKE_MODELS, **FAKE_CHATTERBOX}
    assert "make expressive-build" in restored.stdout
    # Staging may hard-link the sources; none may be left with an extra link.
    assert all(p.stat().st_nlink == 1 for p in models.rglob("*") if p.is_file())


def test_including_chatterbox_requires_its_manifest(tmp_path: Path):
    weights = {k: v for k, v in FAKE_CHATTERBOX.items() if not k.endswith("manifest.json")}
    models = _write_models(tmp_path / "models", {**FAKE_MODELS, **weights})
    out = tmp_path / "b.tar.gz"

    completed = _run(
        ARCHIVE, "--models-dir", models, "--output", out, "--include-chatterbox", cwd=tmp_path
    )

    assert completed.returncode == 1
    assert "chatterbox/manifest.json" in completed.stderr
    assert not out.exists()


def test_restore_without_chatterbox_does_not_mention_it(archive: Path, tmp_path: Path):
    completed = _run(RESTORE, archive, "--models-dir", tmp_path / "restored", cwd=tmp_path)

    assert completed.returncode == 0, completed.stderr
    assert "Chatterbox" not in completed.stdout


def test_download_script_checks_the_files_the_engine_loads():
    """The shell list must track VARIANT_FILES, or a partial download passes."""
    from app.engines.chatterbox import MANIFEST_NAME, VARIANT_FILES

    script = DOWNLOAD.read_text(encoding="utf-8")
    listed = re.search(r'^CHATTERBOX_FILES="([^"]*)"', script, re.M)

    assert listed, "models-download.sh must declare CHATTERBOX_FILES"
    assert sorted(listed.group(1).split()) == sorted([MANIFEST_NAME, *VARIANT_FILES["standard"]])


def test_chatterbox_weights_are_never_built_into_an_image():
    """Both Dockerfiles COPY models/; restored weights must not ride along."""
    ignored = (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()

    assert "models/chatterbox/" in ignored


def _fake_python(tmp_path: Path, *, skip_chatterbox_file: str = "") -> tuple[Path, Path]:
    """A ``--python`` stand-in that writes what the real downloaders would."""
    log = tmp_path / "calls.log"
    stub = tmp_path / "fake-python"
    whisper = " ".join(p.split("/", 1)[1] for p in FAKE_MODELS if p.startswith("whisper-"))

    def names(files: dict[str, bytes]) -> str:
        return " ".join(
            p.split("/", 1)[1] for p in files if not p.endswith(skip_chatterbox_file or "\0")
        )

    chatterbox = names(FAKE_CHATTERBOX)
    multilingual = names(FAKE_MULTILINGUAL)
    stub.write_text(
        "#!/usr/bin/env bash\n"
        '[ "$1" = "-c" ] && exit 0\n'
        f'echo "$(basename "$1") ${{*:2}}" >> "{log}"\n'
        'case "$(basename "$1")" in\n'
        "  download_models.py)\n"
        '    touch "$MODELS_DIR/kokoro-v1.0.onnx" "$MODELS_DIR/voices-v1.0.bin"\n'
        '    mkdir -p "$MODELS_DIR/whisper-$WHISPER_MODEL_SIZE"\n'
        f'    for f in {whisper}; do echo x > "$MODELS_DIR/whisper-$WHISPER_MODEL_SIZE/$f"; done ;;\n'
        "  download_chatterbox.py)\n"
        '    dir="${*: -1}"; mkdir -p "$dir"\n'
        f'    files="{chatterbox}"\n'
        f'    case " $* " in *" --variant multilingual "*) files="{multilingual}" ;; esac\n'
        '    for f in $files; do echo x > "$dir/$f"; done ;;\n'
        "esac\n",
        encoding="utf-8",
    )
    stub.chmod(0o755)
    return stub, log


def test_download_fetches_chatterbox_only_when_asked(tmp_path: Path):
    stub, log = _fake_python(tmp_path)
    models = tmp_path / "models"

    plain = _run(DOWNLOAD, "--models-dir", models, "--python", stub, cwd=tmp_path)

    assert plain.returncode == 0, plain.stderr
    assert "download_chatterbox.py" not in log.read_text()
    assert "--include-chatterbox" not in plain.stdout

    full = _run(
        DOWNLOAD, "--models-dir", models, "--python", stub, "--include-chatterbox", cwd=tmp_path
    )

    assert full.returncode == 0, full.stderr
    assert (
        f"download_chatterbox.py --variant standard --models-dir {models.resolve()}/chatterbox"
        in log.read_text()
    )
    assert "models-archive.sh" in full.stdout and "--include-chatterbox" in full.stdout


def test_download_reports_a_partial_chatterbox_download(tmp_path: Path):
    stub, _ = _fake_python(tmp_path, skip_chatterbox_file="conds.pt")

    completed = _run(
        DOWNLOAD,
        "--models-dir",
        tmp_path / "models",
        "--python",
        stub,
        "--include-chatterbox",
        cwd=tmp_path,
    )

    assert completed.returncode == 1
    assert "chatterbox/conds.pt" in completed.stderr


# ---------------------------------------------------------------------------
# Optional Chatterbox Multilingual weights (--include-chatterbox-multilingual)
# ---------------------------------------------------------------------------


def test_archive_leaves_multilingual_out_unless_asked(tmp_path: Path):
    models = _write_models(
        tmp_path / "models", {**FAKE_MODELS, **FAKE_CHATTERBOX, **FAKE_MULTILINGUAL}
    )
    out = tmp_path / "b.tar.gz"

    completed = _run(
        ARCHIVE, "--models-dir", models, "--output", out, "--include-chatterbox", cwd=tmp_path
    )

    assert completed.returncode == 0, completed.stderr
    assert not any("chatterbox-multilingual" in n for n in _archive_names(out))
    with tarfile.open(out) as tar:
        manifest = tar.extractfile(f"{BUNDLE}/MANIFEST.txt").read().decode()
    assert "chatterbox: standard" in manifest
    assert "chatterbox-multilingual: none" in manifest


def test_multilingual_round_trips_alongside_the_english_model(tmp_path: Path):
    models = _write_models(
        tmp_path / "models", {**FAKE_MODELS, **FAKE_CHATTERBOX, **FAKE_MULTILINGUAL}
    )
    out = tmp_path / "b.tar.gz"

    packed = _run(
        ARCHIVE,
        "--models-dir",
        models,
        "--output",
        out,
        "--include-chatterbox",
        "--include-chatterbox-multilingual",
        cwd=tmp_path,
    )

    assert packed.returncode == 0, packed.stderr
    with tarfile.open(out) as tar:
        manifest = tar.extractfile(f"{BUNDLE}/MANIFEST.txt").read().decode()
    assert "chatterbox: standard" in manifest
    assert "chatterbox-multilingual: multilingual" in manifest

    target = tmp_path / "restored"
    restored = _run(RESTORE, out, "--models-dir", target, cwd=tmp_path)

    assert restored.returncode == 0, restored.stderr
    assert _tree(target) == {**FAKE_MODELS, **FAKE_CHATTERBOX, **FAKE_MULTILINGUAL}
    assert "CHATTERBOX_MULTILINGUAL=1" in restored.stdout


def test_including_multilingual_requires_its_manifest(tmp_path: Path):
    weights = {k: v for k, v in FAKE_MULTILINGUAL.items() if not k.endswith("manifest.json")}
    models = _write_models(tmp_path / "models", {**FAKE_MODELS, **weights})
    out = tmp_path / "b.tar.gz"

    completed = _run(
        ARCHIVE,
        "--models-dir",
        models,
        "--output",
        out,
        "--include-chatterbox-multilingual",
        cwd=tmp_path,
    )

    assert completed.returncode == 1
    assert "chatterbox-multilingual/manifest.json" in completed.stderr
    assert not out.exists()


def test_download_script_checks_the_files_the_multilingual_engine_loads():
    from app.engines.chatterbox import MANIFEST_NAME, MULTILINGUAL_VARIANT, VARIANT_FILES

    script = DOWNLOAD.read_text(encoding="utf-8")
    listed = re.search(r'^MULTILINGUAL_FILES="([^"]*)"', script, re.M)

    assert listed, "models-download.sh must declare MULTILINGUAL_FILES"
    assert sorted(listed.group(1).split()) == sorted(
        [MANIFEST_NAME, *VARIANT_FILES[MULTILINGUAL_VARIANT]]
    )
    assert set(FAKE_MULTILINGUAL) == {
        f"chatterbox-multilingual/{name}"
        for name in (MANIFEST_NAME, *VARIANT_FILES[MULTILINGUAL_VARIANT])
    }


def test_multilingual_weights_are_never_built_into_an_image():
    ignored = (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()

    assert "models/chatterbox-multilingual/" in ignored


def test_download_fetches_multilingual_only_when_asked(tmp_path: Path):
    stub, log = _fake_python(tmp_path)
    models = tmp_path / "models"

    english = _run(
        DOWNLOAD, "--models-dir", models, "--python", stub, "--include-chatterbox", cwd=tmp_path
    )

    assert english.returncode == 0, english.stderr
    assert "--variant multilingual" not in log.read_text()

    both = _run(
        DOWNLOAD,
        "--models-dir",
        models,
        "--python",
        stub,
        "--include-chatterbox",
        "--include-chatterbox-multilingual",
        cwd=tmp_path,
    )

    assert both.returncode == 0, both.stderr
    assert (
        "download_chatterbox.py --variant multilingual "
        f"--models-dir {models.resolve()}/chatterbox-multilingual"
    ) in log.read_text()
    assert "--include-chatterbox --include-chatterbox-multilingual" in both.stdout


def test_download_reports_a_partial_multilingual_download(tmp_path: Path):
    stub, _ = _fake_python(tmp_path, skip_chatterbox_file="Cangjie5_TC.json")

    completed = _run(
        DOWNLOAD,
        "--models-dir",
        tmp_path / "models",
        "--python",
        stub,
        "--include-chatterbox-multilingual",
        cwd=tmp_path,
    )

    assert completed.returncode == 1
    assert "chatterbox-multilingual/Cangjie5_TC.json" in completed.stderr
