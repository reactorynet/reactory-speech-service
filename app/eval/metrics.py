"""Objective metrics used by the evaluation harness.

Everything here is pure: text in, numbers out, or WAV bytes in, measurements out.
No model, no network, no clock unless the caller passes one. That is what makes
the report diffable.

Metrics
-------

* :func:`wer` / :func:`cer` — round-trip intelligibility. The reference is the
  corpus sentence, the hypothesis is the transcript the *local* STT service
  produced from the rendered audio. Both sides are normalised first (see
  :mod:`app.eval.textnorm`).
* :func:`duration_bounds` — the plausible duration band for a sentence, derived
  from a speaking-rate model and the requested rate. It exists to falsify a muted
  or truncated engine without needing a transcript. Subtle truncation is caught by
  WER instead: a render that lost words cannot transcribe back to the source.
* :func:`real_time_factor` — audio seconds produced per wall-clock second of
  synthesis. Above 1.0 is faster than real time.
* :func:`inspect_audio` — header-level format facts (sample rate, channels,
  frames, peak, RMS) plus the silence verdict.

Sample rates, durations and RTF are rounded by the report, not here.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import soundfile as sf

from app.eval import textnorm
from app.eval.corpus import UNSPACED_LANGUAGES

#: Peak amplitude below which a render is treated as silent, in normalised units.
#: Speech peaks at -3 to -12 dBFS (0.25-0.7); silence is several orders below.
SILENCE_PEAK_THRESHOLD = 0.01

#: Nominal articulation rates, in spoken units per second. A spoken unit is a word
#: for space-delimited languages and a character for Japanese/Chinese. The values
#: sit in the middle of the range reported for conversational speech, so a normal
#: render lands near a duration ratio of 1.0.
NOMINAL_RATES: dict[str, float] = {"spaced": 3.4, "unspaced": 5.2}

#: A rate-1.0 render may legitimately be this many times faster than nominal
#: (a brisk narrator) before the lower bound declares the audio truncated.
MAX_SPEEDUP_FACTOR = 2.0

#: The slowest plausible rate, as a fraction of nominal, before the upper bound
#: declares the audio padded or hung.
MIN_RATE_FACTOR = 0.35

#: Leading/trailing silence allowance, in seconds, on the upper bound only.
DURATION_SLACK_SECONDS = 1.0


@dataclass(frozen=True)
class ErrorRate:
    """Levenshtein alignment counts for one reference/hypothesis pair."""

    errors: int
    substitutions: int
    insertions: int
    deletions: int
    reference_length: int
    rate: float | None

    def as_dict(self) -> dict[str, int | float | None]:
        return {
            "errors": self.errors,
            "substitutions": self.substitutions,
            "insertions": self.insertions,
            "deletions": self.deletions,
            "reference_length": self.reference_length,
            "rate": None if self.rate is None else round(self.rate, 4),
        }


def _align(reference: list[str], hypothesis: list[str]) -> ErrorRate:
    """Levenshtein distance with operation counts.

    ``errors == substitutions + insertions + deletions`` always holds, so the
    report can explain a score instead of only stating it.
    """
    reference_length = len(reference)
    if reference_length == 0:
        insertions = len(hypothesis)
        return ErrorRate(
            errors=insertions,
            substitutions=0,
            insertions=insertions,
            deletions=0,
            reference_length=0,
            # A hypothesis against an empty reference is a total failure; 1.0 is
            # the only rate that cannot be mistaken for a good score.
            rate=None if insertions == 0 else 1.0,
        )

    # dp[i][j] = (cost, substitutions, insertions, deletions) for the first i
    # reference tokens and first j hypothesis tokens.
    previous: list[tuple[int, int, int, int]] = [(j, 0, j, 0) for j in range(len(hypothesis) + 1)]
    for i, reference_token in enumerate(reference, start=1):
        current: list[tuple[int, int, int, int]] = [(i, 0, 0, i)]
        for j, hypothesis_token in enumerate(hypothesis, start=1):
            if reference_token == hypothesis_token:
                cost, subs, ins, dels = previous[j - 1]
                current.append((cost, subs, ins, dels))
                continue
            substitute = previous[j - 1]
            insert = current[j - 1]
            delete = previous[j]
            best = min(
                (substitute[0] + 1, substitute[1] + 1, substitute[2], substitute[3]),
                (insert[0] + 1, insert[1], insert[2] + 1, insert[3]),
                (delete[0] + 1, delete[1], delete[2], delete[3] + 1),
            )
            current.append(best)
        previous = current

    errors, substitutions, insertions, deletions = previous[-1]
    return ErrorRate(
        errors=errors,
        substitutions=substitutions,
        insertions=insertions,
        deletions=deletions,
        reference_length=reference_length,
        rate=errors / reference_length,
    )


def error_rate(reference: list[str], hypothesis: list[str]) -> ErrorRate:
    """Error rate over pre-tokenised sequences."""
    return _align(reference, hypothesis)


def wer(reference_text: str, hypothesis_text: str | None, language: str | None = None) -> ErrorRate:
    """Word error rate (character error rate for unspaced languages).

    A ``None`` or empty hypothesis scores 1.0, never 0.0: a render that produced
    no transcript is a total failure and must not read as a perfect one.
    """
    reference = textnorm.tokenize(reference_text, language)
    hypothesis = textnorm.tokenize(hypothesis_text or "", language)
    return _align(reference, hypothesis)


def cer(reference_text: str, hypothesis_text: str | None) -> ErrorRate:
    """Character error rate, always over characters regardless of language."""
    reference = textnorm.characters(reference_text)
    hypothesis = textnorm.characters(hypothesis_text or "")
    return _align(reference, hypothesis)


def _rate_bucket(language: str | None) -> str:
    return "unspaced" if (language or "").lower() in UNSPACED_LANGUAGES else "spaced"


def expected_duration_seconds(text: str, language: str | None = None) -> float:
    """The nominal duration of a sentence at the nominal articulation rate."""
    units = textnorm.spoken_units(text, language)
    return units / NOMINAL_RATES[_rate_bucket(language)]


def duration_bounds(
    text: str,
    language: str | None = None,
    *,
    rate: float = 1.0,
) -> tuple[float, float]:
    """The plausible duration window for a render of ``text``.

    The window widens with the requested ``rate``: asking for 1.5x means the audio
    should be shorter, and the bounds follow so a correct fast render is not
    flagged. A render outside the window is either truncated (too short: the
    engine dropped audio) or padded/hung (too long).
    """
    units = textnorm.spoken_units(text, language)
    if units == 0:
        return 0.0, DURATION_SLACK_SECONDS
    bucket = _rate_bucket(language)
    nominal = NOMINAL_RATES[bucket]
    safe_rate = max(0.1, float(rate))
    shortest = units / (nominal * safe_rate * MAX_SPEEDUP_FACTOR)
    longest = units / (nominal * safe_rate * MIN_RATE_FACTOR) + DURATION_SLACK_SECONDS
    return max(0.15, shortest), max(shortest, longest)


def real_time_factor(duration_seconds: float | None, synthesis_seconds: float | None) -> float | None:
    """Audio seconds per wall-clock synthesis second.

    ``None`` when either input is missing or the synthesis clock is not positive,
    so a division by a zero-length measurement never becomes an infinite score.
    """
    if duration_seconds is None or synthesis_seconds is None:
        return None
    if synthesis_seconds <= 0:
        return None
    return duration_seconds / synthesis_seconds


@dataclass(frozen=True)
class AudioInfo:
    """Format and level facts read from the rendered WAV bytes."""

    sample_rate: int
    channels: int
    frames: int
    duration_s: float
    peak: float
    rms: float
    silent: bool

    def as_dict(self) -> dict[str, int | float | bool]:
        return {
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "frames": self.frames,
            "duration_s": round(self.duration_s, 3),
            "peak": round(self.peak, 4),
            "rms": round(self.rms, 4),
            "silent": self.silent,
        }


def inspect_audio(wav_bytes: bytes) -> AudioInfo:
    """Read a WAV buffer and measure its format and level.

    Raises ``ValueError`` for bytes that are not a readable audio container: a
    render that cannot be decoded is a failure of the engine, not of the metric,
    and the caller records it as such.
    """
    if not wav_bytes:
        raise ValueError("engine returned no audio")
    try:
        with sf.SoundFile(io.BytesIO(wav_bytes)) as handle:
            sample_rate = int(handle.samplerate)
            channels = int(handle.channels)
            frames = int(handle.frames)
            data = handle.read(dtype="float32", always_2d=True)
    except Exception as exc:
        # libsndfile raises its own exception hierarchy; callers only need to know
        # that the bytes are not decodable audio.
        raise ValueError(f"cannot decode audio: {exc}") from exc
    if data.size == 0:
        raise ValueError("engine returned an empty audio buffer")
    peak = float(np.max(np.abs(data)))
    rms = float(np.sqrt(np.mean(np.square(data, dtype=np.float64))))
    return AudioInfo(
        sample_rate=sample_rate,
        channels=channels,
        frames=frames,
        duration_s=frames / sample_rate if sample_rate else 0.0,
        peak=peak,
        rms=rms,
        silent=peak < SILENCE_PEAK_THRESHOLD,
    )


def prosody_rate_adherence(
    base_duration_s: float | None,
    requested_duration_s: float | None,
    requested_rate: float,
) -> float | None:
    """How closely the engine honoured a requested rate multiplier.

    ``measured_rate = base_duration / requested_duration`` and adherence is
    ``measured_rate / requested_rate``. 1.0 means the render is exactly as much
    faster as asked; 0.67 means the engine ignored a 1.5x request (it rendered the
    same duration twice). ``None`` when either render is missing or degenerate.
    """
    if not base_duration_s or not requested_duration_s:
        return None
    if requested_rate <= 0:
        return None
    measured_rate = base_duration_s / requested_duration_s
    return measured_rate / requested_rate


__all__ = [
    "AudioInfo",
    "DURATION_SLACK_SECONDS",
    "ErrorRate",
    "MAX_SPEEDUP_FACTOR",
    "MIN_RATE_FACTOR",
    "NOMINAL_RATES",
    "SILENCE_PEAK_THRESHOLD",
    "cer",
    "duration_bounds",
    "error_rate",
    "expected_duration_seconds",
    "inspect_audio",
    "prosody_rate_adherence",
    "real_time_factor",
    "wer",
]
