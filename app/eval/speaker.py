"""Speaker-consistency measurement.

The ticket asks for "cosine similarity between speaker embeddings of repeated
renders of the same voice". That answers a narrow, useful question: *is this voice
stable?* A voice whose embedding drifts between identical requests will sound like
a different person mid-conversation, and no WER score notices.

Choice of embedder (documented because the ticket requires it)
--------------------------------------------------------------

:class:`SpectralEmbedder` is the default. It is a **deterministic, dependency-free
spectral-statistics embedder**: 25 ms Hamming-windowed frames at a 10 ms hop, a
26-band mel filterbank, log magnitude, then mean and standard deviation across
time. It needs no download, no licence beyond numpy, and no network — which is the
binding product direction for this harness.

What it measures, precisely: timbre and channel stability of *repeated renders*.
Two renders of the same voice with the same text that differ only in prosody have
near-identical log-mel statistics; a different voice (or a resampled/re-encoded
render) does not. Cosine similarity is therefore high (≈1.0) for a stable engine
and lower for an unstable one.

What it is not: a biometric speaker-verification model. It is not trained on
speakers, so its absolute values are not comparable to ECAPA/x-vector numbers, and
the harness must never be used to claim "these two clips are the same person".
Absolute values are only meaningful as a regression signal for one engine/voice
across runs, which is exactly how the baseline uses them.

A trained local model can replace it without touching the harness: pass
``--speaker-embedder module:factory`` (the factory returns an object with
``embed(samples, sample_rate) -> vector`` and an ``id``). The report records which
embedder produced the numbers, so a baseline cannot be silently compared across
different embedders — :func:`app.eval.report.diff_reports` refuses that.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np

#: Analysis sample rate the mel filterbank is defined at. Renders are resampled to
#: this rate with linear interpolation so two engines at different native rates are
#: still comparable.
ANALYSIS_SAMPLE_RATE = 16_000

FRAME_SECONDS = 0.025
HOP_SECONDS = 0.010
MEL_BANDS = 26
#: Floor added before the log so digital silence maps to a finite number.
LOG_FLOOR = 1e-10

EMBEDDER_ID = "spectral-mel-stats-v1"


@runtime_checkable
class SpeakerEmbedder(Protocol):
    """Turns audio into a fixed-length vector for cosine comparison."""

    id: str

    def embed(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """Return a 1-D embedding for a mono or multi-channel signal."""
        raise NotImplementedError


def _mel_filterbank(sample_rate: int, fft_size: int, bands: int) -> np.ndarray:
    """Triangular mel filterbank, ``(bands, fft_size // 2 + 1)``.

    Implemented inline so the harness has no dependency beyond numpy. The mel
    scale is the HTK convention (``2595 * log10(1 + f / 700)``) over 80 Hz-7.6 kHz.
    """
    low_hz, high_hz = 80.0, min(7_600.0, sample_rate / 2)
    low_mel = 2595.0 * np.log10(1.0 + low_hz / 700.0)
    high_mel = 2595.0 * np.log10(1.0 + high_hz / 700.0)
    mel_points = np.linspace(low_mel, high_mel, bands + 2)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
    bins = np.floor((fft_size + 1) * hz_points / sample_rate).astype(int)
    bins = np.clip(bins, 0, fft_size // 2)

    bank = np.zeros((bands, fft_size // 2 + 1), dtype=np.float64)
    for band in range(bands):
        left, centre, right = bins[band], bins[band + 1], bins[band + 2]
        if centre == left:
            centre = min(centre + 1, fft_size // 2)
        if right == centre:
            right = min(right + 1, fft_size // 2)
        for index in range(left, centre):
            bank[band, index] = (index - left) / max(1, centre - left)
        for index in range(centre, right):
            bank[band, index] = (right - index) / max(1, right - centre)
    return bank


def _to_mono(samples: np.ndarray) -> np.ndarray:
    array = np.asarray(samples, dtype=np.float64)
    if array.ndim == 2:
        array = array.mean(axis=1)
    return array.reshape(-1)


def _resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Linear resampling to the analysis rate.

    Good enough for a coarse spectral envelope: the embedding averages over 26
    bands, so interpolation error is far below the band width. If the rate already
    matches, the array is returned unchanged.
    """
    if source_rate == target_rate or samples.size == 0:
        return samples
    duration = samples.size / source_rate
    target_length = max(1, int(round(duration * target_rate)))
    source_positions = np.arange(samples.size) / source_rate
    target_positions = np.arange(target_length) / target_rate
    return np.interp(target_positions, source_positions, samples)


class SpectralEmbedder:
    """Deterministic log-mel statistics embedder; see the module docstring."""

    id = EMBEDDER_ID

    def __init__(self, sample_rate: int = ANALYSIS_SAMPLE_RATE) -> None:
        self.sample_rate = int(sample_rate)
        self._frame_length = int(round(FRAME_SECONDS * self.sample_rate))
        self._hop = int(round(HOP_SECONDS * self.sample_rate))
        self._fft_size = 1 << (self._frame_length - 1).bit_length()
        self._window = np.hamming(self._frame_length)
        self._bank = _mel_filterbank(self.sample_rate, self._fft_size, MEL_BANDS)

    def log_mel(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """``(frames, bands)`` log-mel spectrogram at the analysis rate."""
        signal = _resample(_to_mono(samples), int(sample_rate), self.sample_rate)
        if signal.size < self._frame_length:
            signal = np.pad(signal, (0, self._frame_length - signal.size))
        frame_count = 1 + (signal.size - self._frame_length) // self._hop
        frames = np.lib.stride_tricks.as_strided(
            signal,
            shape=(frame_count, self._frame_length),
            strides=(signal.strides[0] * self._hop, signal.strides[0]),
            writeable=False,
        )
        spectrum = np.abs(np.fft.rfft(frames * self._window, n=self._fft_size, axis=1)) ** 2
        mel = spectrum @ self._bank.T
        return np.log(np.maximum(mel, LOG_FLOOR))

    def embed(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """Mean and standard deviation per mel band, L2-normalised."""
        mel = self.log_mel(samples, sample_rate)
        if mel.size == 0:
            return np.zeros(MEL_BANDS * 2, dtype=np.float64)
        vector = np.concatenate([mel.mean(axis=0), mel.std(axis=0)])
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm > 0 else vector


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float | None:
    """Cosine similarity of two embeddings, ``None`` for a zero vector."""
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    if a.size != b.size or a.size == 0:
        return None
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0.0:
        return None
    return float(np.dot(a, b) / denominator)


def build_embedder(spec: str | None = None) -> SpeakerEmbedder:
    """Resolve an embedder spec.

    ``None`` / ``spectral`` -> :class:`SpectralEmbedder`.
    ``module:factory`` -> call the factory (``module`` may also be
    ``module:attribute`` where the attribute is the embedder object itself).
    """
    if spec in (None, "", "spectral", EMBEDDER_ID):
        return SpectralEmbedder()
    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError(
            f"invalid speaker embedder spec {spec!r}; expected 'module:factory' or 'spectral'"
        )
    module = importlib.import_module(module_name)
    target: Any = getattr(module, attribute)
    # An embedder instance/class exposes ``embed``; a factory is called.
    embedder = target() if callable(target) and not hasattr(target, "embed") else target
    if not hasattr(embedder, "embed"):
        raise ValueError(f"speaker embedder {spec!r} does not expose embed(samples, sample_rate)")
    return embedder


def consistency(embeddings: Sequence[np.ndarray]) -> dict[str, float | int | None]:
    """Pairwise cosine statistics for repeated renders of one voice.

    ``min`` is the regression signal: one unstable repeat is enough to make a voice
    unsuitable, so the worst pair is reported next to the mean.
    """
    pairs: list[float] = []
    for index, left in enumerate(embeddings):
        for right in embeddings[index + 1 :]:
            similarity = cosine_similarity(left, right)
            if similarity is not None:
                pairs.append(similarity)
    if not pairs:
        return {"renders": len(embeddings), "mean": None, "min": None}
    return {
        "renders": len(embeddings),
        "mean": float(np.mean(pairs)),
        "min": float(np.min(pairs)),
    }


__all__ = [
    "ANALYSIS_SAMPLE_RATE",
    "EMBEDDER_ID",
    "MEL_BANDS",
    "SpeakerEmbedder",
    "SpectralEmbedder",
    "build_embedder",
    "consistency",
    "cosine_similarity",
]
