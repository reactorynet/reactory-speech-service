"""Local, CPU-only voice-activity detection for the STT path.

Voice activity detection is used *before* transcription to drop leading, trailing
and internal silence. This has two effects:

* it makes transcription measurably faster on clips with long pauses, because
  Whisper is never asked to decode silence, and
* it splits long recordings into contiguous speech regions, which is what a
  streaming client wants to render.

The detector is the Silero VAD ONNX model that ships inside the
``faster-whisper`` wheel (MIT licensed, ``faster_whisper/assets/silero_vad_v6.onnx``).
No network access, no GPU and no cloud service is involved.

Failure is **not** silence
--------------------------
A detector that cannot run (module missing, ONNX runtime mismatch, model asset
missing, ``get_speech_timestamps`` raising) must never be mistaken for a clip with
no speech. ``detect_regions`` therefore returns ``None`` for "I could not tell" and
``[]`` for "I listened and there was nothing". ``trim_samples`` turns ``None`` into
``TrimResult(applied=False, failed=True)``, which leaves the original audio
untouched for the transcriber; only a genuine ``[]`` produces an empty signal.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

#: Silero VAD (and faster-whisper) operate on 16 kHz mono audio.
SAMPLE_RATE = 16000

#: If the speech regions cover at least this fraction of the clip there is nothing
#: worth trimming and we skip the re-encode (saves a full copy of the audio).
MIN_TRIM_RATIO = 0.98


@dataclass(frozen=True)
class SpeechRegion:
    """One contiguous speech region, in seconds."""

    start: float
    end: float


@dataclass
class TrimResult:
    """Outcome of a VAD pass.

    ``applied`` is True only when the audio was actually re-cut. ``samples`` holds
    the concatenated speech (16 kHz mono float32) when ``applied``; ``offsets`` maps
    a position in the trimmed signal back to the original timeline as
    ``(trimmed_start, original_start)`` pairs, so timestamps handed to clients stay
    on the source timeline.

    ``failed`` means detection could not be performed. It is mutually exclusive with
    ``applied``: on failure the caller must transcribe the original audio.
    """

    applied: bool = False
    samples: np.ndarray | None = None
    original_duration: float = 0.0
    speech_duration: float = 0.0
    regions: list[SpeechRegion] = field(default_factory=list)
    offsets: list[tuple[float, float]] = field(default_factory=list)
    sample_rate: int = SAMPLE_RATE
    failed: bool = False

    @property
    def is_silence(self) -> bool:
        """True only when detection ran and genuinely found no speech."""
        return (
            self.applied
            and not self.failed
            and self.samples is not None
            and self.samples.size == 0
        )

    def map_time(self, value: float) -> float:
        """Map a time in the trimmed signal back onto the original timeline."""
        if not self.applied or not self.offsets:
            return value
        chosen: tuple[float, float] | None = None
        for trimmed_start, original_start in self.offsets:
            if trimmed_start <= value + 1e-9:
                chosen = (trimmed_start, original_start)
            else:
                break
        if chosen is None:
            return value
        trimmed_start, original_start = chosen
        return original_start + (value - trimmed_start)


class VoiceActivityDetector:
    """Silero VAD wrapper.

    Every capability is best-effort: when detection cannot run the wrapper reports
    failure and the caller transcribes the untrimmed audio.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        threshold: float = 0.5,
        min_silence_ms: int = 500,
        min_speech_ms: int = 250,
        pad_ms: int = 200,
        sample_rate: int = SAMPLE_RATE,
    ) -> None:
        self.enabled = enabled
        self.threshold = threshold
        self.min_silence_ms = min_silence_ms
        self.min_speech_ms = min_speech_ms
        self.pad_ms = pad_ms
        self.sample_rate = sample_rate

    # -- introspection -----------------------------------------------------
    @property
    def available(self) -> bool:
        """Whether the Silero VAD model can actually be loaded."""
        try:
            from faster_whisper.vad import get_vad_model

            get_vad_model()
        except Exception:  # noqa: BLE001 - optional dependency probe
            return False
        return True

    def describe(self) -> dict:
        """Machine-readable description, used by the capabilities endpoint."""
        return {
            "enabled": self.enabled,
            "available": self.available,
            "threshold": self.threshold,
            "min_silence_ms": self.min_silence_ms,
            "min_speech_ms": self.min_speech_ms,
            "pad_ms": self.pad_ms,
        }

    # -- trimming ----------------------------------------------------------
    def trim_file(self, audio_path: str | Path) -> TrimResult:
        """Decode ``audio_path`` and return the speech-only signal.

        Decoding failure (including a faster-whisper/PyAV version mismatch) is a
        detection failure, not silence: the caller transcribes the original file.
        """
        if not self.enabled:
            return TrimResult(applied=False, sample_rate=self.sample_rate)

        try:
            from faster_whisper.audio import decode_audio
        except Exception:  # noqa: BLE001 - optional dependency probe
            logger.warning(
                "faster-whisper audio decoding unavailable; transcribing untrimmed audio"
            )
            return TrimResult(applied=False, failed=True, sample_rate=self.sample_rate)

        try:
            samples = decode_audio(str(audio_path), sampling_rate=self.sample_rate)
        except Exception:
            logger.exception("Audio decoding for VAD failed; transcribing untrimmed audio")
            return TrimResult(applied=False, failed=True, sample_rate=self.sample_rate)

        if samples is None or len(samples) == 0:
            # Nothing decoded: not a statement about speech content.
            return TrimResult(applied=False, failed=True, sample_rate=self.sample_rate)
        return self.trim_samples(np.asarray(samples))

    def trim_samples(self, samples: np.ndarray) -> TrimResult:
        """Return the speech regions of a 16 kHz mono float32 signal."""
        original_duration = round(float(len(samples)) / self.sample_rate, 3)
        if not self.enabled or len(samples) == 0:
            return TrimResult(applied=False, original_duration=original_duration)

        regions = self.detect_regions(samples)
        if regions is None:
            # Detection could not run. Never report this as silence.
            return TrimResult(
                applied=False,
                failed=True,
                original_duration=original_duration,
                sample_rate=self.sample_rate,
            )

        speech_duration = round(sum(r.end - r.start for r in regions), 3)

        if not regions:
            # Genuine silence: hand back an empty signal so the caller returns an
            # empty transcript instead of asking Whisper to decode nothing.
            return TrimResult(
                applied=True,
                samples=np.zeros(0, dtype=np.float32),
                original_duration=original_duration,
                speech_duration=0.0,
                regions=[],
                offsets=[],
                sample_rate=self.sample_rate,
            )

        if speech_duration >= original_duration * MIN_TRIM_RATIO:
            # Almost all of the clip is speech: re-cutting would only cost a copy.
            return TrimResult(
                applied=False,
                original_duration=original_duration,
                speech_duration=speech_duration,
                regions=regions,
                sample_rate=self.sample_rate,
            )

        chunks: list[np.ndarray] = []
        offsets: list[tuple[float, float]] = []
        cursor = 0
        for region in regions:
            start_sample = max(0, int(region.start * self.sample_rate))
            end_sample = min(len(samples), int(region.end * self.sample_rate))
            if end_sample <= start_sample:
                continue
            offsets.append((round(cursor / self.sample_rate, 3), region.start))
            chunks.append(samples[start_sample:end_sample])
            cursor += end_sample - start_sample

        if not chunks:
            # Regions were reported but none survived clamping to the signal.
            return TrimResult(
                applied=False,
                original_duration=original_duration,
                speech_duration=speech_duration,
                regions=regions,
                sample_rate=self.sample_rate,
            )

        return TrimResult(
            applied=True,
            samples=np.concatenate(chunks).astype(np.float32, copy=False),
            original_duration=original_duration,
            speech_duration=speech_duration,
            regions=regions,
            offsets=offsets,
            sample_rate=self.sample_rate,
        )

    def detect_regions(self, samples: np.ndarray) -> list[SpeechRegion] | None:
        """Run Silero VAD and return speech regions in seconds.

        Returns ``[]`` when detection ran and found no speech, and ``None`` when
        detection could not run at all. Callers must distinguish the two.
        """
        try:
            from faster_whisper.vad import VadOptions, get_speech_timestamps
        except Exception:  # noqa: BLE001 - optional dependency probe
            logger.warning(
                "Silero VAD unavailable; transcribing untrimmed audio instead "
                "(voice-activity detection skipped)"
            )
            return None

        options = VadOptions(
            threshold=self.threshold,
            min_speech_duration_ms=self.min_speech_ms,
            min_silence_duration_ms=self.min_silence_ms,
            speech_pad_ms=self.pad_ms,
        )
        try:
            timestamps = get_speech_timestamps(samples, options, sampling_rate=self.sample_rate)
        except Exception:
            logger.exception("Voice-activity detection failed; transcribing untrimmed audio")
            return None

        if timestamps is None:
            return None
        try:
            return [
                SpeechRegion(
                    start=round(float(stamp["start"]) / self.sample_rate, 3),
                    end=round(float(stamp["end"]) / self.sample_rate, 3),
                )
                for stamp in timestamps
                if stamp["end"] > stamp["start"]
            ]
        except (KeyError, TypeError, ValueError):
            logger.exception("Voice-activity detection returned malformed timestamps")
            return None
