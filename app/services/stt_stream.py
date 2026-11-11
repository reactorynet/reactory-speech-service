"""Audio buffering for the STT WebSocket streaming path.

Clients may stream either a container format (WAV/WebM/OGG/MP3 — the format they
would upload to ``POST /api/stt/transcribe``) or raw little-endian 16-bit PCM.
PCM is what a browser ``AudioWorklet``/``MediaRecorder`` bridge naturally produces
and is the only encoding that can be cut at an arbitrary byte boundary, so it is
the recommended streaming encoding. Raw PCM is wrapped in a minimal WAV container
before it reaches faster-whisper, which keeps the decoder happy without writing a
partial container to disk.

The buffer never touches the filesystem: it lives in memory for the lifetime of
the WebSocket session and is dropped when the session ends.
"""

from __future__ import annotations

import io
import wave

#: Accepted values of the ``encoding`` config message.
PCM_ENCODINGS = frozenset({"pcm", "pcm16", "pcm_s16le", "pcm-s16le"})
CONTAINER_ENCODINGS = frozenset({"container", "wav", "webm", "ogg", "mp3", "audio", "auto"})

#: Default sample rate for raw PCM. Whisper resamples to 16 kHz anyway; 16 kHz is
#: what a streaming client should send to avoid an unnecessary resample.
DEFAULT_PCM_SAMPLE_RATE = 16000


def normalise_encoding(encoding: str | None) -> str:
    """Return a canonical encoding name, raising ``ValueError`` when unknown."""
    value = (encoding or "container").strip().lower()
    if value in PCM_ENCODINGS:
        return "pcm_s16le"
    if value in CONTAINER_ENCODINGS:
        return "container"
    raise ValueError(
        f"Unsupported encoding '{encoding}'. Use one of: pcm_s16le, container (wav/webm/ogg/mp3)"
    )


class StreamingAudioBuffer:
    """Accumulates streamed audio chunks and reports how much speech is buffered."""

    def __init__(
        self,
        *,
        encoding: str = "container",
        sample_rate: int = DEFAULT_PCM_SAMPLE_RATE,
        channels: int = 1,
    ) -> None:
        self.encoding = normalise_encoding(encoding)
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if channels <= 0:
            raise ValueError("channels must be positive")
        self.sample_rate = int(sample_rate)
        self.channels = int(channels)
        self._buffer = bytearray()

    # -- state -------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._buffer)

    def __bool__(self) -> bool:
        return bool(self._buffer)

    @property
    def is_pcm(self) -> bool:
        return self.encoding == "pcm_s16le"

    @property
    def bytes_per_second(self) -> int:
        return self.sample_rate * self.channels * 2

    def append(self, chunk: bytes) -> int:
        """Append a chunk and return the new total size in bytes."""
        if not chunk:
            return len(self._buffer)
        self._buffer.extend(chunk)
        return len(self._buffer)

    def clear(self) -> None:
        self._buffer.clear()

    # -- views -------------------------------------------------------------
    @property
    def duration_seconds(self) -> float:
        """Seconds of audio currently buffered.

        Exact for PCM and for WAV (the RIFF header is parsed), ``0.0`` for other
        containers, whose length cannot be derived from a prefix.
        """
        if not self._buffer:
            return 0.0
        if self.is_pcm:
            return round(len(self._buffer) / self.bytes_per_second, 3)
        return self._wav_duration()

    def audio_bytes(self) -> bytes:
        """Bytes to hand to the transcriber (raw PCM is wrapped in a WAV header)."""
        if not self.is_pcm:
            return bytes(self._buffer)
        return self._wrap_pcm(bytes(self._buffer))

    def _wrap_pcm(self, pcm: bytes) -> bytes:
        frame = self.channels * 2
        usable = len(pcm) - (len(pcm) % frame)
        out = io.BytesIO()
        with wave.open(out, "wb") as handle:
            handle.setnchannels(self.channels)
            handle.setsampwidth(2)
            handle.setframerate(self.sample_rate)
            handle.writeframes(pcm[:usable])
        return out.getvalue()

    def _wav_duration(self) -> float:
        buffer = self._buffer
        if len(buffer) < 44 or buffer[:4] != b"RIFF" or buffer[8:12] != b"WAVE":
            return 0.0
        try:
            with wave.open(io.BytesIO(bytes(buffer)), "rb") as handle:
                frame_rate = handle.getframerate()
                if frame_rate <= 0:
                    return 0.0
                declared = handle.getnframes()
        except (wave.Error, EOFError):
            # Truncated header/chunk — the length is not derivable from the prefix.
            return 0.0
        if declared <= 0:
            return 0.0
        # ``getnframes`` trusts the declared data-chunk size; a live stream has not
        # sent all of it yet, so clamp to the bytes actually buffered.
        available = max(0, len(buffer) - 44)
        frame_width = max(1, self.channels * 2)
        frames = min(declared, available // frame_width)
        return round(frames / frame_rate, 3)
