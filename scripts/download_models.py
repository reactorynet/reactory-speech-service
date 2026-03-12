#!/usr/bin/env python3
"""Download required models for the Reactory Speech Service."""

import os
from pathlib import Path


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

    if onnx_path.exists() and voices_path.exists():
        print("✅ Kokoro models already present.")
        return

    if not onnx_path.exists():
        print("⬇️  Downloading Kokoro ONNX model (~310MB)...")
        _download_file(f"{KOKORO_BASE_URL}/kokoro-v1.0.onnx", onnx_path)
        print("✅ Kokoro ONNX model downloaded.")

    if not voices_path.exists():
        print("⬇️  Downloading Kokoro voice pack (~4MB)...")
        _download_file(f"{KOKORO_BASE_URL}/voices-v1.0.bin", voices_path)
        print("✅ Kokoro voice pack downloaded.")


def download_whisper_model(model_size: str) -> None:
    """Ensure faster-whisper model is available (auto-downloads on first use)."""
    print(f"ℹ️  faster-whisper '{model_size}' model will auto-download on first use.")
    print("   To pre-download, run: python -c \"from faster_whisper import WhisperModel; WhisperModel('{model_size}')\"")


def main() -> None:
    models_dir = Path(os.environ.get("MODELS_DIR", "models"))
    models_dir.mkdir(parents=True, exist_ok=True)

    model_size = os.environ.get("WHISPER_MODEL_SIZE", "base")

    print("🔧 Reactory Speech Service - Model Setup")
    print(f"   Models directory: {models_dir.resolve()}")
    print()

    download_kokoro_models(models_dir)
    print()
    download_whisper_model(model_size)

    print()
    print("🎉 Model setup complete!")


if __name__ == "__main__":
    main()
