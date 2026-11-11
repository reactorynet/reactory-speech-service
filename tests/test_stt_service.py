"""Service-level tests for the STT path (Phase 7 / SPEECH-008).

These tests do not need a Whisper model on disk: the faster-whisper model object is
replaced with a fake that returns canned segments. Everything the service itself
owns — temp-file lifetime, VAD trimming, timestamp mapping, word timestamps, model
tiering and language confidence — is exercised for real.
"""

from __future__ import annotations

import itertools
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

from app.config import settings
from app.services.stt_service import ModelNotAvailableError, STTService
from app.services.vad import SpeechRegion, TrimResult
from tests.stt_fakes import (
    FakeInfo,
    FakeModel,
    FakeSegment,
    FakeWord,
    NoVAD,
    StubVAD,
    build_service,
)


@pytest.fixture
def temp_dir(tmp_path, monkeypatch):
    """Point ``tempfile`` at an empty directory and return it."""
    directory = tmp_path / "tempdir"
    directory.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(directory))
    return directory


# ----------------------------------------------------------------------
# Temp-file lifetime
# ----------------------------------------------------------------------


def test_transcribe_leaves_no_temp_file_on_success(temp_dir):
    """A successful request removes its temporary audio file."""
    service = build_service(FakeModel())

    result = service.transcribe(b"RIFF....WAVEfmt ")

    assert result["text"] == "Hello world"
    assert list(temp_dir.iterdir()) == []


def test_transcribe_leaves_no_temp_file_when_transcribe_raises(temp_dir):
    """A decoder error still removes the temporary audio file."""
    service = build_service(FakeModel(fail_at="call"))

    with pytest.raises(RuntimeError, match="decoder exploded"):
        service.transcribe(b"RIFF....WAVEfmt ")

    assert list(temp_dir.iterdir()) == []


def test_transcribe_leaves_no_temp_file_when_iteration_raises(temp_dir):
    """An error while consuming segments still removes the temp file."""
    service = build_service(FakeModel(fail_at="iterate"))

    with pytest.raises(RuntimeError, match="decode failed"):
        service.transcribe(b"RIFF....WAVEfmt ")

    assert list(temp_dir.iterdir()) == []


def test_transcribe_file_exists_while_decoding(temp_dir):
    """The temp file is alive for the duration of the decode (and gone after)."""
    seen: list[int] = []
    service = build_service(FakeModel(observer=lambda: seen.append(len(list(temp_dir.iterdir())))))

    service.transcribe(b"RIFF....WAVEfmt ")

    assert seen == [1]
    assert list(temp_dir.iterdir()) == []


def test_streaming_generator_releases_temp_file_when_closed(temp_dir):
    """Closing mid-iteration (what a client disconnect does) removes the temp file."""
    service = build_service(
        FakeModel(segments=[FakeSegment(0.0, 0.5, "one"), FakeSegment(0.5, 1.0, "two")])
    )

    stream = service.transcribe_streaming(b"RIFF....WAVEfmt ")
    first = next(stream)
    assert first["text"] == "one"
    assert len(list(temp_dir.iterdir())) == 1, "the temp file must exist while streaming"

    stream.close()

    assert list(temp_dir.iterdir()) == []


def test_streaming_generator_releases_temp_file_when_exhausted(temp_dir):
    """A fully consumed stream also removes its temp file."""
    service = build_service(FakeModel(segments=[FakeSegment(0.0, 1.0, "one")]))

    items = list(service.transcribe_streaming(b"RIFF....WAVEfmt "))

    assert [item["text"] for item in items] == ["one"]
    assert items[0]["partial_text"] == "one"
    assert list(temp_dir.iterdir()) == []


def test_repeated_requests_do_not_accumulate_temp_files(temp_dir):
    """Twenty requests leave the temp directory exactly as they found it."""
    service = build_service(FakeModel())

    for _ in range(20):
        service.transcribe(b"RIFF....WAVEfmt ")

    assert list(temp_dir.iterdir()) == []


# ----------------------------------------------------------------------
# Word timestamps
# ----------------------------------------------------------------------


def test_words_are_absent_by_default():
    """The default payload stays small: no ``words`` key."""
    model = FakeModel(
        segments=[FakeSegment(0.0, 1.0, "hello world", [FakeWord("hello", 0.0, 0.4)])]
    )
    service = build_service(model)

    result = service.transcribe(b"RIFF....WAVEfmt ")

    assert "words" not in result["segments"][0]
    assert model.calls[0]["kwargs"]["word_timestamps"] is False


def test_words_returned_when_requested():
    """``word_timestamps=True`` exposes word, start, end and probability."""
    words = [FakeWord("hello", 0.0, 0.4, 0.91), FakeWord("world", 0.4, 1.0, 0.87)]
    service = build_service(FakeModel(segments=[FakeSegment(0.0, 1.0, "hello world", words)]))

    result = service.transcribe(b"RIFF....WAVEfmt ", word_timestamps=True)

    segment_words = result["segments"][0]["words"]
    assert [word["word"] for word in segment_words] == ["hello", "world"]
    assert segment_words[0]["probability"] == 0.91
    assert set(segment_words[0]) == {"word", "start", "end", "probability"}


def test_word_timestamps_are_monotonic_within_a_segment():
    """Out-of-order decoder output is clamped, never emitted non-monotonically."""
    words = [
        FakeWord("one", 0.0, 0.5),
        FakeWord("two", 0.3, 0.6),
        FakeWord("three", 0.6, 0.6),
        FakeWord("four", 0.55, 0.9),
    ]
    service = build_service(FakeModel(segments=[FakeSegment(0.0, 1.0, "one two three four", words)]))

    result = service.transcribe(b"RIFF....WAVEfmt ", word_timestamps=True)
    segment_words = result["segments"][0]["words"]

    starts = [word["start"] for word in segment_words]
    ends = [word["end"] for word in segment_words]
    assert starts == sorted(starts)
    assert ends == sorted(ends)
    for previous, current in itertools.pairwise(segment_words):
        assert current["start"] >= previous["end"]


# ----------------------------------------------------------------------
# Voice-activity detection
# ----------------------------------------------------------------------


def test_vad_trim_remaps_segment_times_to_source_timeline():
    """Trimming silence shifts segment times back onto the original timeline."""
    trim = TrimResult(
        applied=True,
        samples=np.zeros(16000, dtype=np.float32),
        original_duration=3.0,
        speech_duration=1.0,
        regions=[SpeechRegion(1.5, 2.5)],
        offsets=[(0.0, 1.5)],
    )
    model = FakeModel(segments=[FakeSegment(0.0, 1.0, "hello")])
    service = build_service(model, vad=StubVAD(trim))

    result = service.transcribe(b"RIFF....WAVEfmt ")

    assert model.calls[0]["source"] is trim.samples
    assert result["segments"][0]["start"] == 1.5
    assert result["segments"][0]["end"] == 2.5
    assert result["vad_applied"] is True
    assert result["speech_duration"] == 1.0


def test_vad_disabled_per_request_passes_the_file_path():
    """``vad=False`` skips detection entirely and hands faster-whisper the path."""
    vad = NoVAD()
    model = FakeModel()
    service = build_service(model, vad=vad)

    result = service.transcribe(b"RIFF....WAVEfmt ", vad=False)

    assert vad.calls == 0
    assert isinstance(model.calls[0]["source"], str)
    assert result["vad_applied"] is False


def test_vad_is_used_by_default():
    """Detection runs when the caller does not override it."""
    vad = NoVAD()
    service = build_service(FakeModel(), vad=vad)

    service.transcribe(b"RIFF....WAVEfmt ")

    assert vad.calls == 1


def test_silence_only_clip_returns_empty_transcript_without_decoding():
    """A clip with no speech returns an empty result and never calls the model."""
    trim = TrimResult(
        applied=True,
        samples=np.zeros(0, dtype=np.float32),
        original_duration=60.0,
        speech_duration=0.0,
    )
    model = FakeModel()
    service = build_service(model, vad=StubVAD(trim))

    result = service.transcribe(b"RIFF....WAVEfmt ")

    assert model.calls == []
    assert result["text"] == ""
    assert result["segments"] == []
    assert result["language"] is None
    assert result["duration"] == 60.0


# ----------------------------------------------------------------------
# Language confidence
# ----------------------------------------------------------------------


def test_language_confidence_and_alternatives_are_returned():
    """Detected language probability and the top alternatives are exposed."""
    info = FakeInfo(
        language="nl",
        language_probability=0.42,
        all_language_probs=[("nl", 0.42), ("en", 0.35), ("de", 0.11), ("fr", 0.05)],
    )
    service = build_service(FakeModel(info=info))

    result = service.transcribe(b"RIFF....WAVEfmt ")

    assert result["language"] == "nl"
    assert result["language_confidence"] == 0.42
    assert result["needs_language_confirmation"] is True
    assert len(result["language_probabilities"]) == 3
    assert result["language_probabilities"][0] == {"language": "nl", "probability": 0.42}


def test_confident_language_does_not_ask_for_confirmation():
    """A high-probability detection is not flagged for confirmation."""
    info = FakeInfo(language="en", language_probability=0.97, all_language_probs=[("en", 0.97)])
    service = build_service(FakeModel(info=info))

    result = service.transcribe(b"RIFF....WAVEfmt ")

    assert result["needs_language_confirmation"] is False
    assert result["language_confidence"] == 0.97


def test_language_confidence_threshold_is_configurable(monkeypatch):
    """The confirmation threshold comes from settings."""
    monkeypatch.setattr(settings, "whisper_language_confidence_threshold", 0.95)
    info = FakeInfo(language_probability=0.9, all_language_probs=[("en", 0.9)])
    service = build_service(FakeModel(info=info))

    result = service.transcribe(b"RIFF....WAVEfmt ")

    assert result["needs_language_confirmation"] is True


# ----------------------------------------------------------------------
# Model tiering
# ----------------------------------------------------------------------


def test_unknown_model_raises_with_available_models():
    """An unavailable model id raises with the list of mounted models."""
    service = build_service(FakeModel(), model_id="base")

    with pytest.raises(ModelNotAvailableError) as excinfo:
        service.transcribe(b"RIFF....WAVEfmt ", model="distil-large-v3")

    assert excinfo.value.available == ["base"]
    assert "distil-large-v3" in str(excinfo.value)


def test_model_selection_is_forwarded_to_resolution():
    """A per-request model id selects the matching mounted model."""
    base_model = FakeModel()
    tiny_model = FakeModel(segments=[FakeSegment(0.0, 1.0, "tiny says hi")])
    service = build_service(base_model, model_id="base")
    service._paths["tiny"] = Path("/models/tiny")
    service._models["tiny"] = tiny_model

    result = service.transcribe(b"RIFF....WAVEfmt ", model="tiny")

    assert result["model"] == "tiny"
    assert result["text"] == "tiny says hi"
    assert base_model.calls == []
    assert len(tiny_model.calls) == 1


def test_discover_models_finds_mounted_directories(tmp_path, monkeypatch):
    """Model discovery maps mounted directories to stable model ids."""
    for name in ("whisper-small", "whisper-tiny", "not-a-model"):
        directory = tmp_path / name
        directory.mkdir()
        if name != "not-a-model":
            (directory / "model.bin").write_bytes(b"x")
    monkeypatch.setattr(settings, "models_dir", str(tmp_path))
    monkeypatch.setattr(settings, "whisper_model_path", None)

    service = STTService(vad=NoVAD())

    assert service.available_models == ["small", "tiny"]


def test_capabilities_reports_models_and_features():
    """The capabilities payload advertises models, streaming, words and VAD."""
    service = build_service(FakeModel(), model_id="base")
    service._paths["tiny"] = Path("/models/tiny")

    capabilities = service.capabilities()

    assert capabilities["available_models"] == ["base", "tiny"]
    assert capabilities["loaded_models"] == ["base"]
    assert capabilities["default_model"] == "base"
    assert capabilities["streaming"] is True
    assert capabilities["word_timestamps"] is True
    assert capabilities["language_confidence"] is True
    assert capabilities["vad"]["enabled"] is True


# ----------------------------------------------------------------------
# Streaming shape
# ----------------------------------------------------------------------


def test_streaming_yields_legacy_message_shape():
    """The streaming generator keeps the contract existing clients use."""
    service = build_service(
        FakeModel(segments=[FakeSegment(0.0, 0.5, "one"), FakeSegment(0.5, 1.0, "two")])
    )

    items = list(service.transcribe_streaming(b"RIFF....WAVEfmt "))

    assert [item["text"] for item in items] == ["one", "two"]
    assert [item["partial_text"] for item in items] == ["one", "one two"]
    assert set(items[0]) == {"start", "end", "text", "partial_text", "language"}
    assert items[0]["language"] == "en"


def test_streaming_can_include_words():
    """Word timestamps flow through the streaming generator when requested."""
    words = [FakeWord("hello", 0.0, 1.0)]
    service = build_service(FakeModel(segments=[FakeSegment(0.0, 1.0, "hello", words)]))

    items = list(service.transcribe_streaming(b"RIFF....WAVEfmt ", word_timestamps=True))

    assert items[0]["words"][0]["word"] == "hello"


# ----------------------------------------------------------------------
# Slow integration check (needs a real model on disk)
# ----------------------------------------------------------------------


@pytest.mark.slow
def test_vad_trimming_is_faster_than_no_vad_on_a_silent_clip():
    """A 60s clip with long silences transcribes faster with VAD than without.

    Requires a mounted faster-whisper model; skipped otherwise (see the report:
    this worktree has no model mounted, so this acceptance check is skipped).

    The clip is synthetic (tone bursts between 20s silences), so text equality is
    only asserted when both runs recognise something — a pure tone has no words to
    compare, and the measurable claims are trimming and wall-clock time.
    """
    pytest.importorskip("faster_whisper")

    service = STTService()
    if not service.available_models:
        pytest.skip("No faster-whisper model mounted; cannot measure real inference")

    import io
    import wave

    tone_seconds = 1
    silence = np.zeros(16000 * 20, dtype=np.int16)
    tone = (
        np.sin(2 * np.pi * 220 * np.arange(16000 * tone_seconds) / 16000) * 8000
    ).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        # 20s silence, 1s tone, 18s silence, 1s tone, 20s silence = 60s.
        for block in (silence, tone, silence, tone, silence):
            handle.writeframes(block.tobytes())
    audio = buffer.getvalue()

    started = time.perf_counter()
    with_vad = service.transcribe(audio, vad=True)
    vad_seconds = time.perf_counter() - started

    started = time.perf_counter()
    without_vad = service.transcribe(audio, vad=False)
    plain_seconds = time.perf_counter() - started

    assert with_vad["vad_applied"] is True
    assert without_vad["vad_applied"] is False
    assert with_vad["speech_duration"] is not None
    assert with_vad["speech_duration"] < 10.0
    assert vad_seconds < plain_seconds
    if with_vad["text"] and without_vad["text"]:
        assert with_vad["text"] == without_vad["text"]


# ----------------------------------------------------------------------
# RC-1: detector failure must fall back to the original audio
# ----------------------------------------------------------------------
def _broken_silero_detector(monkeypatch):
    """The real detector with only the Silero call broken (production code path)."""
    import faster_whisper.vad as fw_vad

    from app.services.vad import VoiceActivityDetector

    def boom(*args, **kwargs):
        raise RuntimeError("onnxruntime exploded")

    monkeypatch.setattr(fw_vad, "get_speech_timestamps", boom)
    detector = VoiceActivityDetector()
    samples = np.zeros(16000, dtype=np.float32)
    detector.trim_file = lambda _path: detector.trim_samples(samples)
    return detector


def test_failing_detector_still_calls_the_model(temp_dir, monkeypatch):
    """Detector failure transcribes the untrimmed audio instead of returning ''."""
    model = FakeModel(segments=[FakeSegment(0.0, 1.0, "Hello world")])
    service = build_service(model, vad=_broken_silero_detector(monkeypatch))

    result = service.transcribe(b"RIFF....WAVEfmt ")

    assert len(model.calls) == 1, "the model must be called when VAD cannot run"
    assert result["text"] == "Hello world"
    assert result["segments"]
    assert result["vad_failed"] is True
    assert result["vad_applied"] is False
    assert list(temp_dir.iterdir()) == []


def test_failing_detector_reports_failure_but_no_empty_transcript(temp_dir, monkeypatch):
    """The failure is surfaced (``vad_failed``) without losing the transcript."""
    model = FakeModel(segments=[FakeSegment(0.0, 0.5, "still here")])
    service = build_service(model, vad=_broken_silero_detector(monkeypatch))

    result = service.transcribe(b"RIFF....WAVEfmt ")

    assert result["text"] == "still here"
    assert result["vad_failed"] is True
    assert result["speech_duration"] is None


def test_failed_detection_is_not_silence():
    """``TrimResult.failed`` and ``is_silence`` are mutually exclusive."""
    failed = TrimResult(applied=False, failed=True, original_duration=3.0)
    silent = TrimResult(
        applied=True, samples=np.zeros(0, dtype=np.float32), original_duration=3.0
    )

    assert failed.is_silence is False
    assert silent.is_silence is True
    assert silent.failed is False


def test_streaming_falls_back_too(temp_dir, monkeypatch):
    """The streaming generator also transcribes through a failed detector."""
    model = FakeModel(segments=[FakeSegment(0.0, 1.0, "hello")])
    service = build_service(model, vad=_broken_silero_detector(monkeypatch))

    items = list(service.transcribe_streaming(b"RIFF....WAVEfmt "))

    assert [item["text"] for item in items] == ["hello"]
    assert len(model.calls) == 1


# ----------------------------------------------------------------------
# Duration semantics (F6b)
# ----------------------------------------------------------------------
def test_duration_is_the_source_audio_duration_under_vad():
    """``duration`` describes the submitted audio; ``speech_duration`` the speech."""
    trim = TrimResult(
        applied=True,
        samples=np.zeros(16000, dtype=np.float32),
        original_duration=30.0,
        speech_duration=2.0,
        regions=[SpeechRegion(10.0, 12.0)],
        offsets=[(0.0, 10.0)],
    )
    model = FakeModel(info=FakeInfo(duration=2.0), segments=[FakeSegment(0.0, 2.0, "hello")])
    service = build_service(model, vad=StubVAD(trim))

    result = service.transcribe(b"RIFF....WAVEfmt ")

    assert result["duration"] == 30.0, "duration must not shrink to the trimmed length"
    assert result["speech_duration"] == 2.0


def test_duration_falls_back_to_decoder_duration_without_vad():
    """With VAD disabled, the decoder's duration is used."""
    model = FakeModel(info=FakeInfo(duration=7.5), segments=[FakeSegment(0.0, 7.5, "hi")])
    service = build_service(model)

    result = service.transcribe(b"RIFF....WAVEfmt ", vad=False)

    assert result["duration"] == 7.5
    assert result["speech_duration"] is None


# ----------------------------------------------------------------------
# Temp-file robustness (F6a)
# ----------------------------------------------------------------------
def test_temp_file_removed_when_initial_write_fails(temp_dir, monkeypatch):
    """A failing write (e.g. disk full) must not leave the file behind."""
    real_factory = tempfile.NamedTemporaryFile

    def failing_factory(**kwargs):
        handle = real_factory(**kwargs)

        def boom(_data):
            raise OSError("No space left on device")

        handle.write = boom
        return handle

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", failing_factory)

    with (
        pytest.raises(OSError, match="No space left"),
        # The body never runs: the failing write happens on context entry.
        STTService._temporary_audio_file(b"RIFF....WAVEfmt "),
    ):
        pass

    assert list(temp_dir.iterdir()) == []


# ----------------------------------------------------------------------
# Model discovery details (F6d)
# ----------------------------------------------------------------------
def test_generic_model_path_uses_the_configured_size_as_id(tmp_path, monkeypatch):
    """A generically named directory does not become the model id."""
    generic = tmp_path / "models"
    generic.mkdir()
    (generic / "model.bin").write_bytes(b"x")
    monkeypatch.setattr(settings, "models_dir", str(tmp_path / "empty"))
    monkeypatch.setattr(settings, "whisper_model_path", str(generic))
    monkeypatch.setattr(settings, "whisper_model_size", "small")

    assert STTService(vad=NoVAD()).available_models == ["small"]


def test_named_model_path_supplies_its_own_id(tmp_path, monkeypatch):
    """``.../whisper-tiny`` still maps to ``tiny``."""
    named = tmp_path / "whisper-tiny"
    named.mkdir()
    (named / "model.bin").write_bytes(b"x")
    monkeypatch.setattr(settings, "models_dir", str(tmp_path / "empty"))
    monkeypatch.setattr(settings, "whisper_model_path", str(named))

    assert STTService(vad=NoVAD()).available_models == ["tiny"]
