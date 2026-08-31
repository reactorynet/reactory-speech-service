#!/usr/bin/env python3
"""Download required models for the Reactory Speech Service."""

import os
import sys
import ssl
import urllib.request
from pathlib import Path

# Disable SSL verification for model downloads in environments with TLS-decrypting proxies / self-signed CAs
_orig_create_default_context = ssl.create_default_context


def _unverified_create_default_context(*args, **kwargs):
    ctx = _orig_create_default_context(*args, **kwargs)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


ssl.create_default_context = _unverified_create_default_context
ssl._create_default_https_context = _unverified_create_default_context
os.environ["PYTHONHTTPSVERIFY"] = "0"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

KOKORO_ONNX_URLS = [
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
    "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/kokoro-v0_19.onnx",
    "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/kokoro-v1_0.onnx",
    "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/kokoro.onnx",
]

KOKORO_VOICES_URLS = [
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
    "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/voices-v1.0.bin",
    "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/voices.bin",
]

WHISPER_FILES = ["config.json", "model.bin", "tokenizer.json", "vocabulary.txt"]


def download_with_fallback(urls: list[str] | str, dest: Path) -> None:
    """Download a file from a list of candidate URLs using httpx with unverified SSL and stream to disk."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest.with_suffix(dest.suffix + ".tmp")

    if isinstance(urls, str):
        urls = [urls]

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }

    import httpx

    last_error = None
    for url in urls:
        print(f"⬇️ Streaming {dest.name} from {url}...")
        try:
            with httpx.Client(verify=False, follow_redirects=True, timeout=300.0, headers=headers) as client:
                with client.stream("GET", url) as response:
                    response.raise_for_status()
                    with open(temp_path, "wb") as out_file:
                        for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                            out_file.write(chunk)

            temp_path.replace(dest)
            print(f"✅ {dest.name} downloaded ({dest.stat().st_size / (1024 * 1024):.1f} MB)")
            return
        except Exception as e:
            print(f"⚠️ httpx failed for {url} ({e}), trying next source...")
            last_error = e

    if temp_path.exists():
        temp_path.unlink()
    raise RuntimeError(f"Failed to download {dest.name} from all candidate URLs. Last error: {last_error}")


KOKORO_BASE_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
)


def _download_file(url: str, dest: Path) -> None:
    """Download a file from *url* to *dest* with a progress indicator."""
    import urllib.request
    import shutil

    with urllib.request.urlopen(url) as resp, open(dest, "wb") as out:  # noqa: S310
        shutil.copyfileobj(resp, out)


def download_kokoro_models(models_dir: Path) -> None:
    """Download Kokoro TTS ONNX model and voice pack."""
    onnx_path = models_dir / "kokoro-v1.0.onnx"
    voices_path = models_dir / "voices-v1.0.bin"

    # Also check if any kokoro-*.onnx or voices-*.bin exists
    existing_onnx = list(models_dir.glob("kokoro*.onnx"))
    existing_voices = list(models_dir.glob("voices*.bin"))

    if existing_onnx and existing_voices:
        print(f"✅ Kokoro models already present: {existing_onnx[0].name}, {existing_voices[0].name}")
        return

    if not existing_onnx:
        download_with_fallback(KOKORO_ONNX_URLS, onnx_path)

    if not existing_voices:
        download_with_fallback(KOKORO_VOICES_URLS, voices_path)


def download_whisper_model(model_size: str, models_dir: Path) -> None:
    """Download faster-whisper model files to models_dir / whisper-{model_size}."""
    whisper_dir = models_dir / f"whisper-{model_size}"
    whisper_dir.mkdir(parents=True, exist_ok=True)

    all_exist = all((whisper_dir / f).exists() for f in WHISPER_FILES)
    if all_exist:
        print(f"✅ faster-whisper '{model_size}' model already present in {whisper_dir}.")
        return

    repo_id = f"Systran/faster-whisper-{model_size}"
    print(f"⬇️ Downloading faster-whisper '{model_size}' model from {repo_id}...")

    try:
        from faster_whisper import download_model

        download_model(model_size, output_dir=str(whisper_dir))
        print(f"✅ faster-whisper '{model_size}' model downloaded via faster_whisper.")
        return
    except Exception as e:
        print(f"⚠️ faster_whisper download_model failed ({e}), trying snapshot_download...")

    try:
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id=repo_id, local_dir=str(whisper_dir))
        print(f"✅ faster-whisper '{model_size}' model downloaded via snapshot_download.")
        return
    except Exception as e:
        print(f"⚠️ snapshot_download failed ({e}), trying direct URL downloads...")

    # Fallback to direct HTTP download per file
    for filename in WHISPER_FILES:
        dest_file = whisper_dir / filename
        if not dest_file.exists():
            candidate_urls = [
                f"https://huggingface.co/{repo_id}/resolve/main/{filename}?download=true",
                f"https://huggingface.co/{repo_id}/raw/main/{filename}",
                f"https://huggingface.co/{repo_id}/resolve/main/{filename}",
            ]
            download_with_fallback(candidate_urls, dest_file)

    print(f"✅ faster-whisper '{model_size}' model files downloaded to {whisper_dir}.")


def main() -> None:
    models_dir = Path(os.environ.get("MODELS_DIR", "models"))
    models_dir.mkdir(parents=True, exist_ok=True)

    model_size = os.environ.get("WHISPER_MODEL_SIZE", "base")

    print("🔧 Reactory Speech Service - Model Setup")
    print(f"   Models directory: {models_dir.resolve()}")
    print(f"   Whisper model size: {model_size}")
    print()

    download_kokoro_models(models_dir)
    print()
    download_whisper_model(model_size, models_dir)

    print()
    print("🎉 Model setup complete!")


if __name__ == "__main__":
    main()
