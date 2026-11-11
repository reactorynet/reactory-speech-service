"""Tests for the real :class:`VoiceActivityDetector` (Phase 7 / SPEECH-008).

The service-level tests inject ``NoVAD``/``StubVAD`` doubles, which is exactly how
the "detector failure looks like silence" bug slipped through review. These tests
drive ``app.services.vad`` itself, with the Silero dependency monkeypatched only at
the boundary (``get_speech_timestamps`` / ``decode_audio``) so the production code
paths under test are the real ones.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from app.services import vad as vad_module
from app.services.vad import MIN_TRIM_RATIO, SAMPLE_RATE, SpeechRegion, VoiceActivityDetector

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def tone(seconds: float, frequency: int = 220, amplitude: float = 0.3) -> np.ndarray:
    """A deterministic non-silent signal of the requested length."""
    n = int(SAMPLE_RATE * seconds)
    return (np.sin(2 * np.pi * frequency * np.arange(n) / SAMPLE_RATE) * amplitude).astype(
        np.float32
    )


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(SAMPLE_RATE * seconds), dtype=np.float32)


def patch_timestamps(monkeypatch, stamps, recorder: list | None = None):
    """Replace ``get_speech_timestamps`` with canned sample-index ranges."""
    import faster_whisper.vad as fw_vad

    def fake(audio, vad_options=None, sampling_rate=16000, **kwargs):
        if recorder is not None:
            recorder.append(
                {
                    "samples": len(audio),
                    "sampling_rate": sampling_rate,
                    "threshold": getattr(vad_options, "threshold", None),
                    "min_silence": getattr(vad_options, "min_silence_duration_ms", None),
                }
            )
        if isinstance(stamps, Exception):
            raise stamps
        return stamps

    monkeypatch.setattr(fw_vad, "get_speech_timestamps", fake)


def patch_decode(monkeypatch, result):
    """Replace ``faster_whisper.audio.decode_audio`` with a canned result or error."""
    import faster_whisper.audio as fw_audio

    def fake(path, sampling_rate=16000, **kwargs):
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(fw_audio, "decode_audio", fake)


def hide(monkeypatch, module: str) -> None:
    """Make ``import <module>`` fail, as on an install without the dependency."""
    monkeypatch.setitem(sys.modules, module, None)


# ----------------------------------------------------------------------
# Detection failure must never be reported as silence (RC-1)
# ----------------------------------------------------------------------
def test_detect_regions_returns_none_when_detector_raises(monkeypatch):
    """An exception from Silero returns None (unknown), not [] (silence)."""
    patch_timestamps(monkeypatch, RuntimeError("onnxruntime exploded"))
    detector = VoiceActivityDetector()

    assert detector.detect_regions(tone(3)) is None


def test_detect_regions_returns_none_when_import_fails(monkeypatch):
    """A missing/broken faster-whisper VAD module returns None, not []."""
    hide(monkeypatch, "faster_whisper.vad")
    detector = VoiceActivityDetector()

    assert detector.detect_regions(tone(3)) is None


def test_detect_regions_returns_empty_list_for_genuine_silence(monkeypatch):
    """Genuine silence is [] and stays distinguishable from failure."""
    patch_timestamps(monkeypatch, [])
    detector = VoiceActivityDetector()

    assert detector.detect_regions(silence(2)) == []


def test_real_silero_reports_silence_not_failure():
    """The shipped Silero model returns [] (not None) for a silent clip."""
    detector = VoiceActivityDetector()
    if not detector.available:  # pragma: no cover - depends on the install
        pytest.skip("Silero VAD asset unavailable in this environment")

    result = detector.trim_samples(silence(2))

    assert result.failed is False
    assert result.is_silence is True
    assert result.speech_duration == 0.0


def test_trim_samples_marks_failure_and_keeps_audio_untouched(monkeypatch):
    """A failed detector yields applied=False/failed=True (transcribe untrimmed)."""
    patch_timestamps(monkeypatch, RuntimeError("boom"))
    detector = VoiceActivityDetector()
    samples = tone(3)

    result = detector.trim_samples(samples)

    assert result.failed is True
    assert result.applied is False
    assert result.is_silence is False
    assert result.samples is None
    assert result.original_duration == pytest.approx(3.0)


def test_trim_file_reports_failure_when_import_fails(monkeypatch):
    """A missing faster-whisper audio module is a failure, not silence."""
    hide(monkeypatch, "faster_whisper.audio")
    detector = VoiceActivityDetector()

    result = detector.trim_file("/tmp/nonexistent.wav")

    assert result.failed is True
    assert result.is_silence is False
    assert result.applied is False


def test_trim_file_reports_failure_when_decoder_raises(monkeypatch):
    """A PyAV/faster-whisper mismatch inside decode_audio is a failure, not silence."""
    patch_decode(monkeypatch, TypeError("open() got an unexpected keyword argument"))
    detector = VoiceActivityDetector()

    result = detector.trim_file("/tmp/whatever.wav")

    assert result.failed is True
    assert result.is_silence is False
    assert result.applied is False


def test_trim_file_reports_failure_when_nothing_decodes(monkeypatch):
    """An empty decode is inconclusive, so it must not be called silence."""
    patch_decode(monkeypatch, np.zeros(0, dtype=np.float32))
    detector = VoiceActivityDetector()

    result = detector.trim_file("/tmp/whatever.wav")

    assert result.failed is True
    assert result.is_silence is False


def test_trim_file_trims_decoded_audio(monkeypatch):
    """A successful decode is handed to the detector and trimmed."""
    patch_decode(monkeypatch, tone(3))
    patch_timestamps(monkeypatch, [{"start": 16000, "end": 32000}])
    detector = VoiceActivityDetector()

    result = detector.trim_file("/tmp/whatever.wav")

    assert result.applied is True
    assert result.failed is False
    assert result.original_duration == pytest.approx(3.0)
    assert result.speech_duration == pytest.approx(1.0)
    assert result.samples is not None
    assert len(result.samples) == SAMPLE_RATE


def test_trim_file_is_a_noop_when_disabled(monkeypatch):
    """``enabled=False`` never touches the decoder."""
    patch_decode(monkeypatch, RuntimeError("should not be called"))
    detector = VoiceActivityDetector(enabled=False)

    result = detector.trim_file("/tmp/whatever.wav")

    assert result.applied is False
    assert result.failed is False


# ----------------------------------------------------------------------
# Trimming behaviour
# ----------------------------------------------------------------------
def test_short_silent_gap_is_trimmed():
    """Leading and trailing silence are removed from the returned signal."""
    samples = np.concatenate([silence(1), tone(1), silence(2)])
    detector = VoiceActivityDetector()
    detector.detect_regions = lambda _samples: [SpeechRegion(1.0, 2.0)]  # type: ignore[method-assign]

    result = detector.trim_samples(samples)

    assert result.applied is True
    assert result.failed is False
    assert result.original_duration == pytest.approx(4.0)
    assert result.speech_duration == pytest.approx(1.0)
    assert result.samples is not None
    assert len(result.samples) == SAMPLE_RATE


def test_min_trim_ratio_skips_the_recut():
    """When nearly everything is speech, the audio is not copied."""
    # 99% speech coverage: above MIN_TRIM_RATIO (0.98).
    duration = 10.0
    covered = duration * MIN_TRIM_RATIO + 0.05
    detector = VoiceActivityDetector()
    detector.detect_regions = lambda _samples: [SpeechRegion(0.0, covered)]  # type: ignore[method-assign]

    result = detector.trim_samples(tone(duration))

    assert result.applied is False
    assert result.failed is False
    assert result.samples is None
    assert result.regions == [SpeechRegion(0.0, covered)]
    assert result.speech_duration == pytest.approx(covered)


def test_multi_region_trim_builds_offset_map():
    """Two speech regions are concatenated and mapped back to source times."""
    detector = VoiceActivityDetector()
    detector.detect_regions = lambda _samples: [  # type: ignore[method-assign]
        SpeechRegion(1.0, 2.0),
        SpeechRegion(4.0, 5.0),
    ]

    result = detector.trim_samples(tone(6))

    assert result.applied is True
    assert result.speech_duration == pytest.approx(2.0)
    assert result.samples is not None
    assert len(result.samples) == 2 * SAMPLE_RATE
    assert result.offsets == [(0.0, 1.0), (1.0, 4.0)]
    # A time in the trimmed signal maps back onto the original timeline.
    assert result.map_time(0.0) == pytest.approx(1.0)
    assert result.map_time(0.9) == pytest.approx(1.9)
    assert result.map_time(1.0) == pytest.approx(4.0)
    assert result.map_time(1.5) == pytest.approx(4.5)


def test_map_time_is_identity_when_not_applied():
    """An untrimmed result never shifts timestamps."""
    result = vad_module.TrimResult(applied=False)

    assert result.map_time(2.5) == 2.5


def test_regions_outside_the_signal_are_ignored():
    """Regions clamped to nothing fall back to the untrimmed audio."""
    detector = VoiceActivityDetector()
    detector.detect_regions = lambda _samples: [SpeechRegion(5.0, 9.0)]  # type: ignore[method-assign]

    result = detector.trim_samples(tone(2))

    assert result.applied is False
    assert result.regions == [SpeechRegion(5.0, 9.0)]


def test_malformed_timestamps_report_failure(monkeypatch):
    """A detector returning junk is a failure, not silence."""
    patch_timestamps(monkeypatch, [{"begin": 0, "finish": 100}])
    detector = VoiceActivityDetector()

    assert detector.detect_regions(tone(1)) is None


def test_detector_options_are_forwarded(monkeypatch):
    """Configured thresholds reach the Silero call."""
    recorder: list[dict] = []
    patch_timestamps(monkeypatch, [], recorder=recorder)
    detector = VoiceActivityDetector(
        threshold=0.7, min_silence_ms=800, min_speech_ms=120, pad_ms=50
    )

    detector.detect_regions(tone(1))

    assert recorder == [
        {
            "samples": SAMPLE_RATE,
            "sampling_rate": SAMPLE_RATE,
            "threshold": 0.7,
            "min_silence": 800,
        }
    ]


# ----------------------------------------------------------------------
# describe() / available
# ----------------------------------------------------------------------
def test_describe_reports_configuration():
    """describe() exposes the tuning knobs used by the capabilities endpoint."""
    detector = VoiceActivityDetector(threshold=0.4, min_silence_ms=300, pad_ms=10)

    described = detector.describe()

    assert described["enabled"] is True
    assert described["threshold"] == 0.4
    assert described["min_silence_ms"] == 300
    assert described["min_speech_ms"] == 250
    assert described["pad_ms"] == 10
    assert isinstance(described["available"], bool)


def test_available_is_false_when_the_dependency_is_missing(monkeypatch):
    """An unloadable detector advertises itself as unavailable."""
    hide(monkeypatch, "faster_whisper.vad")

    assert VoiceActivityDetector().available is False


def test_available_is_true_when_silero_loads():
    """In this environment the real Silero model loads."""
    detector = VoiceActivityDetector()
    if not detector.available:  # pragma: no cover - depends on the install
        pytest.skip("Silero VAD asset unavailable in this environment")
    assert detector.available is True
