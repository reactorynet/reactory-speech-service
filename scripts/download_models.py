#!/usr/bin/env python3
"""Download required models for the Reactory Speech Service."""

import os
import sys
from pathlib import Path


def download_kokoro_models(models_dir: Path) -> None:
    """Download Kokoro TTS ONNX model and voice pack."""
    onnx_path = models_dir / "kokoro-v1_9.onnx"
    voices_path = models_dir / "voices-v1_0.bin"

    if onnx_path.exists() and voices_path.exists():
        print("✅ Kokoro models already present.")
        return

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Installing huggingface_hub for model download...")
        os.system(f"{sys.executable} -m pip install huggingface_hub")
        from huggingface_hub import hf_hub_download

    if not onnx_path.exists():
        print("⬇️  Downloading Kokoro ONNX model (~330MB)...")
        hf_hub_download(
            repo_id="hexgrad/Kokoro-82M",
            filename="kokoro-v1_9.onnx",
            local_dir=str(models_dir),
        )
        print("✅ Kokoro ONNX model downloaded.")

    if not voices_path.exists():
        print("⬇️  Downloading Kokoro voice pack (~4MB)...")
        hf_hub_download(
            repo_id="hexgrad/Kokoro-82M",
            filename="voices-v1_0.bin",
            local_dir=str(models_dir),
        )
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
