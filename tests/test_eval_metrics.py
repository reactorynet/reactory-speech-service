"""Unit tests for the objective metrics.

These are the numbers every other conclusion rests on, so they are tested directly:
the alignment counts (not just the rate), the normalisation that makes a transcript
comparable, the duration model's two failure directions, and the prosody ratio's
ability to detect an ignored rate.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf

from app.eval import metrics, speaker, textnorm


# ----------------------------------------------------------------------------
# Text normalisation
# ----------------------------------------------------------------------------


def test_normalise_folds_case_punctuation_and_width() -> None:
    assert textnorm.normalise("Hello, WORLD!") == "hello world"
    assert textnorm.normalise("１，２３４．５０") == "1 234 50"  # full-width NFKC
    assert textnorm.normalise("\u200bzero\u200bwidth") == "zerowidth"
    assert textnorm.normalise("Reactory_platform") == "reactory platform"


def test_tokenize_is_language_aware() -> None:
    assert textnorm.tokenize("Hi Werner, how can I help?") == [
        "hi",
        "werner",
        "how",
        "can",
        "i",
        "help",
    ]
    # Japanese and Chinese have no spaces; a whitespace "word" would be the whole line.
    assert textnorm.tokenize("こんにちは、ワーナーさん", "ja") == list("こんにちはワーナーさん")
    assert len(textnorm.tokenize("你好，维尔纳。", "zh")) == 5
    assert textnorm.spoken_units("你好，维尔纳。", "zh") == 5


# ----------------------------------------------------------------------------
# WER / CER
# ----------------------------------------------------------------------------


def test_identical_text_scores_zero() -> None:
    result = metrics.wer("Hi Werner, how can I help?", "Hi Werner, how can I help?", "en")
    assert result.rate == 0.0
    assert result.errors == 0


def test_case_and_punctuation_do_not_count_as_errors() -> None:
    result = metrics.wer("Hi Werner, how can I help?", "hi werner how can i help", "en")
    assert result.rate == 0.0


def test_substitution_counted() -> None:
    result = metrics.wer("one two three four", "one two three five")
    assert result.rate == pytest.approx(0.25)
    assert result.substitutions == 1
    assert result.errors == 1


def test_insertion_counted() -> None:
    result = metrics.wer("one two three", "one two three four")
    assert result.rate == pytest.approx(1 / 3)
    assert result.insertions == 1


def test_deletion_counted() -> None:
    result = metrics.wer("one two three four", "one three four")
    assert result.rate == pytest.approx(0.25)
    assert result.deletions == 1


def test_empty_hypothesis_is_a_total_failure() -> None:
    result = metrics.wer("one two three", "")
    assert result.rate == 1.0
    assert result.deletions == 3


def test_none_hypothesis_is_a_total_failure_not_a_pass() -> None:
    assert metrics.wer("one two three", None).rate == 1.0


def test_cer_is_character_level() -> None:
    result = metrics.cer("kitten", "sitten")
    assert result.rate == pytest.approx(1 / 6)
    assert result.substitutions == 1


def test_wer_for_unspaced_language_uses_characters() -> None:
    result = metrics.wer("你好世界", "你好世", "zh")
    assert result.rate == pytest.approx(0.25)


# ----------------------------------------------------------------------------
# Duration model
# ----------------------------------------------------------------------------


def test_duration_bounds_bracket_a_plausible_render() -> None:
    text = "The Reactory platform turns a support ticket into a reviewable change."
    shortest, longest = metrics.duration_bounds(text, "en")
    expected = metrics.expected_duration_seconds(text, "en")
    assert 0 < shortest < expected < longest
    assert longest > shortest


def test_duration_bounds_follow_the_requested_rate() -> None:
    text = "The Reactory platform turns a support ticket into a reviewable change."
    slow = metrics.duration_bounds(text, "en", rate=0.5)
    fast = metrics.duration_bounds(text, "en", rate=2.0)
    assert fast[0] < slow[0]
    assert fast[1] < slow[1]


def test_truncated_audio_falls_outside_the_lower_bound() -> None:
    text = "The Reactory platform turns a support ticket into a reviewable change."
    shortest, _ = metrics.duration_bounds(text, "en")
    assert 0.2 * metrics.expected_duration_seconds(text, "en") < shortest


def test_empty_text_has_a_small_bounded_window() -> None:
    assert metrics.duration_bounds("", "en") == (0.0, metrics.DURATION_SLACK_SECONDS)


def test_real_time_factor() -> None:
    assert metrics.real_time_factor(4.0, 2.0) == 2.0
    assert metrics.real_time_factor(4.0, 0.0) is None
    assert metrics.real_time_factor(None, 2.0) is None


def test_prosody_adherence_detects_an_ignored_rate() -> None:
    # A respected 1.5x request halves the duration by a factor of 1.5.
    assert metrics.prosody_rate_adherence(3.0, 2.0, 1.5) == pytest.approx(1.0)
    # An ignored request renders the same duration twice.
    assert metrics.prosody_rate_adherence(3.0, 3.0, 1.5) == pytest.approx(1 / 1.5)
    assert metrics.prosody_rate_adherence(None, 3.0, 1.5) is None
    assert metrics.prosody_rate_adherence(3.0, 2.0, 0.0) is None


# ----------------------------------------------------------------------------
# Audio inspection
# ----------------------------------------------------------------------------


def _wav(samples: np.ndarray, sample_rate: int = 24_000) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, samples, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def test_inspect_audio_measures_format_and_level() -> None:
    samples = (0.5 * np.sin(np.linspace(0, 40, 24_000))).astype(np.float32)
    info = metrics.inspect_audio(_wav(samples))
    assert info.sample_rate == 24_000
    assert info.channels == 1
    assert info.duration_s == pytest.approx(1.0, abs=0.01)
    assert info.silent is False
    assert 0.3 < info.peak <= 0.6


def test_inspect_audio_flags_silence() -> None:
    info = metrics.inspect_audio(_wav(np.zeros(24_000, dtype=np.float32)))
    assert info.silent is True
    assert info.peak < metrics.SILENCE_PEAK_THRESHOLD


def test_inspect_audio_rejects_empty_and_garbage() -> None:
    with pytest.raises(ValueError):
        metrics.inspect_audio(b"")
    with pytest.raises(ValueError):
        metrics.inspect_audio(b"not a wav file at all")


# ----------------------------------------------------------------------------
# Speaker embedding
# ----------------------------------------------------------------------------


def test_embedding_is_deterministic_and_identical_for_identical_audio() -> None:
    embedder = speaker.SpectralEmbedder()
    timeline = np.arange(24_000) / 24_000
    samples = (0.3 * np.sin(2 * np.pi * 180 * timeline)).astype(np.float32)
    first = embedder.embed(samples, 24_000)
    second = embedder.embed(samples.copy(), 24_000)
    assert speaker.cosine_similarity(first, second) == pytest.approx(1.0, abs=1e-9)


def test_embedding_differs_for_different_signals() -> None:
    embedder = speaker.SpectralEmbedder()
    timeline = np.arange(24_000) / 24_000
    low = (0.3 * np.sin(2 * np.pi * 180 * timeline)).astype(np.float32)
    high = (0.3 * np.sin(2 * np.pi * 900 * timeline)).astype(np.float32)
    similarity = speaker.cosine_similarity(embedder.embed(low, 24_000), embedder.embed(high, 24_000))
    assert similarity is not None
    assert similarity < 0.95


def test_consistency_reports_mean_and_worst_pair() -> None:
    timeline = np.arange(24_000) / 24_000
    embedder = speaker.SpectralEmbedder()
    same = embedder.embed((0.3 * np.sin(2 * np.pi * 180 * timeline)).astype(np.float32), 24_000)
    other = embedder.embed((0.3 * np.sin(2 * np.pi * 900 * timeline)).astype(np.float32), 24_000)
    stats = speaker.consistency([same, same, other])
    assert stats["renders"] == 3
    assert stats["mean"] is not None and stats["min"] is not None
    assert stats["min"] < stats["mean"]


def test_cosine_handles_zero_vectors() -> None:
    assert speaker.cosine_similarity(np.zeros(4), np.zeros(4)) is None
    assert speaker.cosine_similarity(np.ones(4), np.ones(5)) is None


def test_build_embedder_defaults_and_rejects_junk() -> None:
    assert speaker.build_embedder(None).id == speaker.EMBEDDER_ID
    assert speaker.build_embedder("spectral").id == speaker.EMBEDDER_ID
    with pytest.raises(ValueError):
        speaker.build_embedder("not-a-spec")
