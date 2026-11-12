"""Deterministic engine and transcriber doubles for the evaluation-harness tests.

The point of these doubles is to let the tests prove the harness can *fail*. A
broken engine must be detected by a check, not wave through as a pass, and that
claim is only credible when the breakage is real audio rather than a mocked metric.

``ToneEngine`` synthesises a speech-shaped tone whose duration follows the sentence
length. The source words are appended to the WAV as a trailing RIFF chunk, and
``EchoTranscriber`` reads them back — but only the words the audio could still
contain. A truncated render therefore returns a truncated transcript and a muted
render returns nothing, which is exactly how a real STT model behaves and what
makes the WER check meaningful without a 100 MB Whisper download.
"""

from __future__ import annotations

import io
import json
from typing import Any

import numpy as np
import soundfile as sf

from app.core.contracts import (
    AudioChunk,
    AudioResult,
    Direction,
    EngineCapabilities,
    EngineRequest,
)
from app.eval import metrics

#: Trailing chunk magic. Real WAV readers ignore bytes after the data chunk, so the
#: tone stays a valid WAV for the harness and for soundfile.
_MARKER = b"REACTORYEVALTEXT"

#: Approximate articulation rate used to size a render, in units per second.
_UNITS_PER_SECOND = 3.4

_LANGUAGE_META: dict[str, tuple[str, str]] = {
    "en": ("en-US", "en-us"),
    "fr": ("fr-FR", "fr-fr"),
    "ja": ("ja", "ja"),
    "zh": ("zh", "cmn"),
    "es": ("es", "es"),
}

#: Full scale amplitude of the synthetic tone: comfortably above the silence floor.
_AMPLITUDE = 0.3


def _sentence_duration(text: str, language: str) -> float:
    units = max(1, len([token for token in text.split() if token]))
    if language in {"ja", "zh"}:
        units = max(1, len([character for character in text if not character.isspace()]))
    return max(0.3, units / _UNITS_PER_SECOND)


def _encode_marker(text: str, full_duration: float) -> bytes:
    payload = json.dumps({"text": text, "duration": full_duration}).encode("utf-8")
    return _MARKER + len(payload).to_bytes(4, "little") + payload


def read_marker(wav_bytes: bytes) -> tuple[str, float] | None:
    """Recover ``(text, intended_duration)`` from a :class:`ToneEngine` render."""
    index = wav_bytes.find(_MARKER)
    if index < 0:
        return None
    start = index + len(_MARKER)
    length = int.from_bytes(wav_bytes[start : start + 4], "little")
    payload = json.loads(wav_bytes[start + 4 : start + 4 + length].decode("utf-8"))
    return str(payload["text"]), float(payload["duration"])


class ToneEngine:
    """A deterministic engine double with selectable brokenness.

    ``mode`` values:

    ``ok``             a normal render
    ``muted``          digital silence (a dead audio path)
    ``truncated``      only the first 20 percent of the samples
    ``wrong_rate``     samples generated at 24 kHz but the header says 48 kHz
    ``error``          ``synthesize`` raises
    """

    license = "MIT"
    attribution = "evaluation test double"
    native_rate = True

    def __init__(
        self,
        engine_id: str = "tone",
        *,
        mode: str = "ok",
        languages: tuple[str, ...] = ("en",),
        sample_rate: int = 24_000,
        supports_streaming: bool = True,
        styles: tuple[str, ...] = ("neutral",),
        rate_applied: bool = True,
        honour_rate: bool = True,
    ) -> None:
        self.id = engine_id
        self.name = f"Tone engine ({mode})"
        self.mode = mode
        self.languages = tuple(languages)
        self.sample_rate = sample_rate
        self.supports_streaming = supports_streaming
        self.styles = list(styles)
        #: Whether the render honours the requested rate, and whether the engine
        #: *reports* that it did. ``False`` on both is an engine that silently
        #: ignores rate control.
        self.honour_rate = honour_rate
        self.rate_applied = rate_applied
        self.is_ready = True
        self.calls: list[EngineRequest] = []

    # -- contract ----------------------------------------------------------

    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            engine=self.id,
            languages=[_LANGUAGE_META[language][0] for language in self.languages],
            native_rate=self.native_rate,
            native_pitch=False,
            native_volume=False,
            styles=list(self.styles),
            supports_cloning=False,
            supports_streaming=self.supports_streaming,
            supports_ssml=False,
            sample_rates=[self.sample_rate],
            max_chars=10_000,
            gpu_required=False,
        )

    def voices(self) -> list[dict[str, Any]]:
        catalogue: list[dict[str, Any]] = []
        for language in self.languages:
            tag, locale = _LANGUAGE_META.get(language, (language, language))
            catalogue.append(
                {
                    "id": f"{self.id}_{language}",
                    "name": f"{language.upper()} tone",
                    "language": tag,
                    "locale": locale,
                    "gender": "female",
                    "engine": self.id,
                    "quality": "A",
                    "tags": ["test-double"],
                }
            )
        return catalogue

    def _language_for(self, voice: str | None) -> str:
        if voice and "_" in voice:
            candidate = voice.rsplit("_", 1)[-1]
            if candidate in _LANGUAGE_META:
                return candidate
        return self.languages[0]

    def _samples(self, text: str, language: str, rate: float) -> np.ndarray:
        duration = _sentence_duration(text, language) / max(0.1, rate)
        if self.mode == "truncated":
            duration *= 0.2
        frames = max(1, int(duration * self.sample_rate))
        timeline = np.arange(frames, dtype=np.float64) / self.sample_rate
        # Syllabic amplitude modulation gives the embedder something speech-like:
        # a flat sine has no spectral variation for the statistics to latch onto.
        envelope = 0.5 * (1.0 - np.cos(2.0 * np.pi * 3.0 * timeline))
        samples = _AMPLITUDE * np.sin(2.0 * np.pi * 180.0 * timeline) * envelope
        if self.mode == "muted":
            samples = np.zeros_like(samples)
        return samples.astype(np.float32)

    def synthesize(self, req: EngineRequest) -> AudioResult:
        self.calls.append(req)
        if self.mode == "error":
            raise RuntimeError("engine audio path is broken on purpose")

        language = self._language_for(req.voice)
        rate = float(req.direction.prosody.rate or 1.0)
        samples = self._samples(req.text, language, rate if self.honour_rate else 1.0)
        # ``wrong_rate`` writes the same frame count under a doubled header rate,
        # which is exactly the defect that makes a 2 s clip play in 1 s.
        header_rate = self.sample_rate * 2 if self.mode == "wrong_rate" else self.sample_rate
        buffer = io.BytesIO()
        sf.write(buffer, samples, header_rate, format="WAV", subtype="PCM_16")
        audio = buffer.getvalue() + _encode_marker(
            req.text, _sentence_duration(req.text, language)
        )

        applied: dict[str, Any] = {"rate": rate} if self.rate_applied else {}
        ignored = [] if self.rate_applied else ["rate"]
        return AudioResult(
            audio=audio,
            duration=len(samples) / header_rate,
            sample_rate=header_rate,
            engine=self.id,
            voice=req.voice,
            applied=applied,
            ignored=ignored,
        )

    def stream(self, req: EngineRequest):
        first_sentence = req.text.split(".")[0] or req.text
        chunk_request = EngineRequest(
            text=first_sentence,
            voice=req.voice,
            direction=Direction(prosody=req.direction.prosody),
        )
        result = self.synthesize(chunk_request)
        yield AudioChunk(
            audio=result.audio, duration=result.duration, sample_rate=result.sample_rate
        )


class EchoTranscriber:
    """A transcriber double that recovers the words the audio still contains."""

    available = True
    model = "echo-test-double"

    def describe(self) -> dict:
        return {
            "enabled": True,
            "available": True,
            "model": self.model,
            "description": "test double: reads the render's word marker",
        }

    def transcribe(self, wav_bytes: bytes, language: str | None = None) -> str | None:
        try:
            audio = metrics.inspect_audio(wav_bytes)
        except Exception:
            return None
        if audio.silent:
            # A real model has nothing to decode in silence.
            return ""
        marker = read_marker(wav_bytes)
        if marker is None:
            return ""
        text, intended_duration = marker
        if intended_duration <= 0:
            return text
        fraction = min(1.0, audio.duration_s / intended_duration)
        if fraction >= 0.95:
            return text
        words = text.split()
        keep = max(1, int(round(len(words) * fraction)))
        return " ".join(words[:keep])


class ScriptedTranscriber(EchoTranscriber):
    """Returns only the first ``keep_words`` words of a correct render.

    Used to prove the WER gate fires on a well-formed render that is simply
    unintelligible, without breaking the audio.
    """

    model = "scripted-test-double"

    def __init__(self, keep_words: int = 1) -> None:
        self.keep_words = keep_words

    def transcribe(self, wav_bytes: bytes, language: str | None = None) -> str | None:
        full = super().transcribe(wav_bytes, language)
        if not full:
            return full
        return " ".join(full.split()[: self.keep_words])


class NullTranscriberDouble:
    """The ``--no-stt`` shape, for report/CLI tests."""

    available = False
    model = None

    def describe(self) -> dict:
        return {"enabled": False, "available": False, "model": None}

    def transcribe(self, wav_bytes: bytes, language: str | None = None) -> str | None:
        return None


__all__ = [
    "EchoTranscriber",
    "NullTranscriberDouble",
    "ScriptedTranscriber",
    "ToneEngine",
    "read_marker",
]
