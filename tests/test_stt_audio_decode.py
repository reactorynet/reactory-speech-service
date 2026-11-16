"""faster-whisper can decode real audio with the PyAV version that gets installed.

The STT tests fake the model, so none of them decode audio. That hid a break
where faster-whisper 1.2.1 passes ``metadata_errors=`` to ``av.open`` and
PyAV 19 no longer accepts it: every transcription on the Python 3.14 image
failed with a 500 while the suite stayed green. This decodes a real WAV through
faster-whisper's own decoder, which needs no model weights.
"""

from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

from faster_whisper.audio import decode_audio

SAMPLE_RATE = 16000


def _write_tone(path: Path, seconds: float = 0.5, rate: int = 24000) -> None:
    frames = int(seconds * rate)
    samples = (int(12000 * math.sin(2 * math.pi * 440 * i / rate)) for i in range(frames))
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(b"".join(struct.pack("<h", s) for s in samples))


def test_faster_whisper_decodes_a_wav_with_the_installed_pyav(tmp_path: Path):
    path = tmp_path / "tone.wav"
    _write_tone(path)

    audio = decode_audio(str(path), sampling_rate=SAMPLE_RATE)

    # Resampled from 24 kHz to 16 kHz: 0.5 s is ~8000 samples.
    assert abs(len(audio) - SAMPLE_RATE // 2) < 200
    assert float(abs(audio).max()) > 0.1, "decoded audio is silent"
