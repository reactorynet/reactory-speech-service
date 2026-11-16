"""``.env.example`` must point at the model files that actually get installed.

It named ``kokoro-v1_9.onnx`` / ``voices-v1_0.bin`` while
``scripts/download_models.py`` writes ``kokoro-v1.0.onnx`` / ``voices-v1.0.bin``
and the settings default to the same, so copying the example to ``.env`` left
the service unable to load Kokoro.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings

ENV_EXAMPLE = Path(__file__).resolve().parent.parent / ".env.example"


def _example_values() -> dict[str, str]:
    values = {}
    for line in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


@pytest.mark.parametrize("field", ["kokoro_model_path", "kokoro_voices_path"])
def test_env_example_model_paths_match_the_installed_files(field: str):
    default = Settings.model_fields[field].default

    assert _example_values()[field.upper()] == default
