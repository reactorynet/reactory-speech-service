"""The evaluation runner: render the corpus, measure it, falsify it.

For every (engine, voice) pair the runner renders the corpus sentences for that
voice's language, transcribes each render with the local STT service and records
the objective metrics. It also runs three cell-level probes:

* **time to first audio** — the streaming path, timed until the first chunk;
* **speaker consistency** — the same voice and text rendered ``--repeats`` times,
  compared by cosine similarity of speaker embeddings;
* **prosody adherence** — the same text at rate 1.0 and at ``--probe-rate``, so a
  requested rate can be checked against the rate actually produced.

Design rules that make the numbers trustworthy:

* A render that cannot be decoded, is silent, is too short/long for the sentence,
  or arrives at a sample rate the engine does not advertise is recorded as an
  **integrity failure**. The harness exits non-zero; it never reports a pass for
  broken audio.
* A missing transcript is never a WER of 0.0. If transcription is enabled and
  produces nothing, the entry fails ``empty_transcript`` *and* scores 1.0.
* Every measurement is taken on a freshly rendered buffer: no synthesis cache is
  involved (the harness calls the adapter directly), so a cached clip can never
  hide a regression.
"""

from __future__ import annotations

import hashlib
import io
import os
import statistics
import sys
import time
import tracemalloc
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from app.config import settings
from app.core.contracts import Direction, EngineRequest, Prosody
from app.eval import metrics, speaker as speaker_module, textnorm
from app.eval.corpus import CorpusEntry, GoldenCorpus
from app.eval.engines import EvalEngineError, engine_identity
from app.eval.stt import Transcriber

#: Published quality grades, best first. Voices the catalogue does not grade sort
#: last; the id breaks ties so voice selection is reproducible.
GRADE_ORDER: dict[str, int] = {
    "A": 0,
    "A-": 1,
    "B-": 2,
    "C+": 3,
    "C": 4,
    "C-": 5,
    "D": 6,
    "D-": 7,
    "F+": 8,
}

#: Engine locale codes that are not the ISO 639-1 prefix of their language tag.
_LOCALE_ALIASES: dict[str, str] = {"cmn": "zh", "yue": "zh", "nb": "no", "iw": "he"}

#: Below this pairwise cosine, repeated renders of one voice are not the same voice
#: any more. Deliberately coarse: the value is a stability signal, and a stochastic
#: engine may legitimately sit well below the 1.0 a deterministic engine produces.
SPEAKER_CONSISTENCY_FLOOR = 0.5


@dataclass
class RunnerConfig:
    """Knobs that change what is measured, not how it is scored."""

    repeats: int = 2
    probe_rate: float = 1.5
    memory_mode: str = "tracemalloc"
    cross_language: bool = False
    fail_on_wer: bool = False
    speaker_floor: float = SPEAKER_CONSISTENCY_FLOOR
    #: Restrict measurement to these languages; ``None`` means every language in the
    #: corpus. An unknown language is an error, not an empty cell.
    languages: tuple[str, ...] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "repeats": self.repeats,
            "probe_rate": self.probe_rate,
            "memory_mode": self.memory_mode,
            "cross_language": self.cross_language,
            "fail_on_wer": self.fail_on_wer,
            "speaker_floor": self.speaker_floor,
            "languages": list(self.languages) if self.languages else None,
        }


@dataclass
class Render:
    """One synthesis call: the result plus what it cost."""

    result: Any
    synthesis_s: float
    peak_memory_mb: float | None


@dataclass
class Cell:
    """One (engine, voice, language) measurement, before report rounding."""

    engine: str
    voice: str
    language: str
    voice_descriptor: dict[str, Any]
    entries: list[dict[str, Any]] = field(default_factory=list)
    streaming: dict[str, Any] | None = None
    speaker_consistency: dict[str, Any] | None = None
    prosody: dict[str, Any] | None = None
    capabilities: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "voice": self.voice,
            "language": self.language,
            "entries": self.entries,
            "summary": summarise(self.entries),
            "streaming": self.streaming,
            "speaker_consistency": self.speaker_consistency,
            "prosody": self.prosody,
            "capabilities": self.capabilities,
            "voice_descriptor": self.voice_descriptor,
        }


# ----------------------------------------------------------------------------
# Voice selection
# ----------------------------------------------------------------------------


def voice_language(descriptor: dict[str, Any]) -> str:
    """ISO 639-1 language of a voice, from its engine locale then its tag.

    ``cmn`` (the Kokoro/Mandarin locale code) maps to ``zh`` so the corpus and the
    voice catalogue agree on one code. ``und``/missing metadata stays ``und`` and
    never matches a corpus language, which is the honest outcome: a voice with no
    declared language is not evidence about any language.
    """
    locale = str(descriptor.get("locale") or "").strip().lower()
    if locale:
        prefix = locale.split("-")[0]
        return _LOCALE_ALIASES.get(prefix, prefix)
    language = str(descriptor.get("language") or "").strip().lower()
    if language:
        prefix = language.split("-")[0]
        return _LOCALE_ALIASES.get(prefix, prefix)
    return "und"


def voice_rank(descriptor: dict[str, Any]) -> tuple[int, str]:
    """Sort key: best published grade first, then id."""
    grade = descriptor.get("quality")
    return (GRADE_ORDER.get(str(grade), len(GRADE_ORDER)), str(descriptor.get("id")))


def select_voices(
    engine: Any,
    languages: Sequence[str],
    spec: str = "default",
) -> list[dict[str, Any]]:
    """Choose the voices to evaluate for one engine.

    * ``default`` — the best-graded voice per language, so a four-language corpus
      does not become a fifty-four voice run.
    * ``all`` — every voice whose language is in ``languages``.
    * an explicit comma-separated id list — those voices, in the given order.

    Explicit ids are validated against the engine: a typo must fail the run rather
    than silently measure nothing.
    """
    catalogue = list(engine.voices())
    by_id = {str(voice["id"]): voice for voice in catalogue}
    wanted = {language for language in languages}

    if spec in ("", "default"):
        selected: list[dict[str, Any]] = []
        for language in sorted(wanted):
            candidates = [
                voice for voice in catalogue if voice_language(voice) == language
            ]
            if not candidates:
                continue
            selected.append(sorted(candidates, key=voice_rank)[0])
        return selected

    if spec == "all":
        return sorted(
            (voice for voice in catalogue if voice_language(voice) in wanted),
            key=lambda voice: (voice_language(voice), *voice_rank(voice)),
        )

    ids = [part.strip() for part in spec.split(",") if part.strip()]
    if not ids:
        raise EvalEngineError("--voices was given but contains no voice ids")
    missing = [voice_id for voice_id in ids if voice_id not in by_id]
    if missing:
        raise EvalEngineError(
            f"engine '{getattr(engine, 'id', '?')}' does not expose voice(s) {missing}. "
            f"Available: {', '.join(sorted(by_id)) or 'none'}"
        )
    return [by_id[voice_id] for voice_id in ids]


# ----------------------------------------------------------------------------
# Measurement helpers
# ----------------------------------------------------------------------------


def _digest(text: str) -> str:
    """Short content digest of a sentence, so a text edit is visible in a diff."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _mean(values: Iterable[Any]) -> float | None:
    numbers = [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    return statistics.fmean(numbers) if numbers else None


def summarise(entries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-entry metrics into one row per cell."""
    peaks = [
        float(entry["peak_memory_mb"])
        for entry in entries
        if isinstance(entry.get("peak_memory_mb"), (int, float))
    ]
    return {
        "entries": len(entries),
        "errors": sum(1 for entry in entries if entry.get("status") != "ok"),
        "integrity_failures": sum(
            1
            for entry in entries
            for check in entry.get("checks", [])
            if check != "wer_threshold"
        ),
        "quality_failures": sum(
            1 for entry in entries for check in entry.get("checks", []) if check == "wer_threshold"
        ),
        "wer_mean": _mean(entry.get("wer") for entry in entries),
        "cer_mean": _mean(entry.get("cer") for entry in entries),
        "duration_ratio_mean": _mean(entry.get("duration_ratio") for entry in entries),
        "rtf_mean": _mean(entry.get("rtf") for entry in entries),
        "peak_memory_mb_max": max(peaks) if peaks else None,
    }


def _process_peak_rss_mb() -> float | None:
    """Process high-water RSS in MiB, or ``None`` where ``resource`` is absent.

    ``ru_maxrss`` is bytes on macOS and kibibytes on Linux. It is a process
    high-water mark, not a per-render delta, which is why the report labels the
    ``rss`` memory mode separately.
    """
    try:
        import resource
    except ImportError:  # pragma: no cover - Windows only
        return None
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
    return usage / divisor


def _close(generator: Any) -> None:
    close = getattr(generator, "close", None)
    if callable(close):
        close()


# ----------------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------------


class EvaluationRunner:
    """Renders a corpus through engines and voices, and measures the result."""

    def __init__(
        self,
        corpus: GoldenCorpus,
        engines: Sequence[Any],
        transcriber: Transcriber,
        *,
        embedder: Any | None = None,
        config: RunnerConfig | None = None,
        voices_spec: str = "default",
    ) -> None:
        self.corpus = corpus
        self.engines = list(engines)
        self.transcriber = transcriber
        self.embedder = embedder or speaker_module.SpectralEmbedder()
        self.config = config or RunnerConfig()
        self.voices_spec = voices_spec
        #: Cells not measured because the engine's text front end cannot phonemise
        #: the language. Recorded, never silently dropped: an unmeasured language
        #: must not look like a passing one.
        self.skipped: list[dict[str, Any]] = []

    # -- rendering ---------------------------------------------------------

    def _render(self, engine: Any, text: str, voice: str, rate: float) -> Render:
        """Synthesise one buffer and measure its wall-clock cost and peak heap."""
        request = EngineRequest(
            text=text,
            voice=voice,
            direction=Direction(prosody=Prosody(rate=rate)),
        )
        tracing = self.config.memory_mode == "tracemalloc"
        peak_mb: float | None = None
        started = time.perf_counter()
        try:
            if tracing:
                tracemalloc.start()
                tracemalloc.reset_peak()
            result = engine.synthesize(request)
            elapsed = time.perf_counter() - started
            if tracing:
                peak_bytes = tracemalloc.get_traced_memory()[1]
                peak_mb = peak_bytes / (1024.0 * 1024.0)
            elif self.config.memory_mode == "rss":
                peak_mb = _process_peak_rss_mb()
        finally:
            if tracing:
                tracemalloc.stop()
        return Render(result=result, synthesis_s=elapsed, peak_memory_mb=peak_mb)

    def _measure_entry(
        self,
        engine: Any,
        voice: str,
        entry: CorpusEntry,
        capabilities: Any,
    ) -> dict[str, Any]:
        """Render and score one corpus sentence for one voice."""
        record: dict[str, Any] = {
            "id": entry.id,
            "use": entry.use,
            "text_digest": _digest(entry.text),
            "status": "ok",
            "error": None,
            "checks": [],
            "transcript": None,
            "wer": None,
            "cer": None,
            "wer_counts": None,
            "cer_counts": None,
            "duration_s": None,
            "expected_duration_s": metrics.expected_duration_seconds(entry.text, entry.language),
            "duration_min_s": None,
            "duration_max_s": None,
            "duration_ratio": None,
            "synthesis_s": None,
            "rtf": None,
            "peak_memory_mb": None,
            "sample_rate": None,
            "channels": None,
            "peak": None,
            "rms": None,
            "silent": None,
        }

        try:
            render = self._render(engine, entry.text, voice, 1.0)
        except Exception as exc:  # a broken adapter is a finding, not a crash
            record["status"] = "error"
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["checks"].append("synthesis_error")
            return record

        record["synthesis_s"] = render.synthesis_s
        record["peak_memory_mb"] = render.peak_memory_mb

        try:
            audio = metrics.inspect_audio(render.result.audio)
        except Exception as exc:
            record["status"] = "error"
            record["error"] = f"undecodable audio: {type(exc).__name__}: {exc}"
            record["checks"].append(
                "no_audio" if not getattr(render.result, "audio", b"") else "undecodable_audio"
            )
            return record

        record.update(
            {
                "duration_s": audio.duration_s,
                "sample_rate": audio.sample_rate,
                "channels": audio.channels,
                "peak": audio.peak,
                "rms": audio.rms,
                "silent": audio.silent,
                "rtf": metrics.real_time_factor(audio.duration_s, render.synthesis_s),
            }
        )

        shortest, longest = metrics.duration_bounds(entry.text, entry.language, rate=1.0)
        expected = metrics.expected_duration_seconds(entry.text, entry.language)
        record["duration_min_s"] = shortest
        record["duration_max_s"] = longest
        record["duration_ratio"] = audio.duration_s / expected if expected else None

        if audio.silent:
            record["checks"].append("silent_output")
        if not shortest <= audio.duration_s <= longest:
            record["checks"].append("duration_out_of_bounds")
        native_rates = list(getattr(capabilities, "sample_rates", []) or [])
        if native_rates and audio.sample_rate not in native_rates:
            record["checks"].append("unexpected_sample_rate")
        if audio.channels != 1:
            record["checks"].append("unexpected_channels")

        if self.transcriber.available:
            try:
                transcript = self.transcriber.transcribe(render.result.audio, entry.language)
            except Exception as exc:
                record["checks"].append("transcription_error")
                record["error"] = f"{type(exc).__name__}: {exc}"
                transcript = None
            if transcript is None:
                record["checks"].append("transcription_error")
            else:
                record["transcript"] = transcript
                wer = metrics.wer(entry.text, transcript, entry.language)
                cer = metrics.cer(entry.text, transcript)
                record["wer"] = wer.rate
                record["cer"] = cer.rate
                record["wer_counts"] = wer.as_dict()
                record["cer_counts"] = cer.as_dict()
                if not textnorm.normalise(transcript):
                    record["checks"].append("empty_transcript")
                if (
                    entry.max_wer is not None
                    and wer.rate is not None
                    and wer.rate > entry.max_wer
                ):
                    record["checks"].append("wer_threshold")

        return record

    # -- probes ------------------------------------------------------------

    def _measure_ttfa(self, engine: Any, text: str, voice: str, capabilities: Any) -> dict[str, Any]:
        """Timed first chunk of the streaming path, or a documented skip."""
        if not getattr(capabilities, "supports_streaming", False):
            return {
                "ttfa_s": None,
                "entry": None,
                "reason": "engine reports no streaming support",
                "checks": [],
            }
        request = EngineRequest(
            text=text,
            voice=voice,
            direction=Direction(prosody=Prosody(rate=1.0)),
        )
        started = time.perf_counter()
        stream = None
        try:
            stream = engine.stream(request)
            first = next(iter(stream))
            elapsed = time.perf_counter() - started
        except StopIteration:
            return {
                "ttfa_s": None,
                "entry": None,
                "reason": "stream produced no audio",
                "checks": ["probe_error"],
            }
        except Exception as exc:
            return {
                "ttfa_s": None,
                "entry": None,
                "reason": f"{type(exc).__name__}: {exc}",
                "checks": ["probe_error"],
            }
        finally:
            if stream is not None:
                _close(stream)
        return {
            "ttfa_s": elapsed,
            "entry": None,
            "chunk_duration_s": getattr(first, "duration", None),
            "sample_rate": getattr(first, "sample_rate", None),
            "checks": [],
        }

    def _measure_probes(
        self,
        engine: Any,
        voice: str,
        text: str,
        capabilities: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Run the streaming, speaker-consistency and prosody probes for one cell.

        The rate-1.0 render is shared: it is both the first speaker-consistency
        repeat and the prosody baseline, so enabling all three probes costs
        ``repeats + 1`` renders plus one streaming render, not five.
        """
        checks: list[str] = []
        probe_error: str | None = None
        embeddings: list[Any] = []
        base_duration: float | None = None

        for index in range(max(1, self.config.repeats)):
            try:
                render = self._render(engine, text, voice, 1.0)
            except Exception as exc:
                checks.append("probe_error")
                probe_error = f"{type(exc).__name__}: {exc}"
                break
            try:
                audio = metrics.inspect_audio(render.result.audio)
            except Exception as exc:
                checks.append("probe_error")
                probe_error = f"undecodable audio: {type(exc).__name__}: {exc}"
                continue
            if index == 0:
                base_duration = audio.duration_s
            samples, sample_rate = _decode(render.result.audio)
            embeddings.append(self.embedder.embed(samples, sample_rate))

        consistency = speaker_module.consistency(embeddings)
        consistency_payload: dict[str, Any] = {
            "embedder": getattr(self.embedder, "id", "unknown"),
            "entry": None,
            "error": probe_error,
            **consistency,
            "checks": checks,
        }
        if (
            consistency.get("min") is not None
            and float(consistency["min"]) < self.config.speaker_floor
        ):
            consistency_payload["checks"] = [*checks, "speaker_inconsistent"]

        prosody_payload: dict[str, Any] = {
            "requested_rate": self.config.probe_rate,
            "base_duration_s": base_duration,
            "requested_duration_s": None,
            "measured_rate": None,
            "adherence": None,
            "applied": None,
            "ignored": None,
            "checks": [],
        }
        try:
            requested_render = self._render(engine, text, voice, self.config.probe_rate)
            requested_audio = metrics.inspect_audio(requested_render.result.audio)
            prosody_payload["requested_duration_s"] = requested_audio.duration_s
            prosody_payload["measured_rate"] = (
                base_duration / requested_audio.duration_s
                if base_duration and requested_audio.duration_s
                else None
            )
            prosody_payload["adherence"] = metrics.prosody_rate_adherence(
                base_duration, requested_audio.duration_s, self.config.probe_rate
            )
            prosody_payload["applied"] = getattr(requested_render.result, "applied", None)
            prosody_payload["ignored"] = getattr(requested_render.result, "ignored", None)
        except Exception as exc:
            prosody_payload["error"] = f"{type(exc).__name__}: {exc}"
            prosody_payload["checks"] = ["probe_error"]

        return consistency_payload, prosody_payload

    # -- cells -------------------------------------------------------------

    def run_cell(
        self,
        engine: Any,
        voice_descriptor: dict[str, Any],
        entries: Sequence[CorpusEntry],
    ) -> Cell:
        """Measure one engine + voice over the entries for its language."""
        capabilities = engine.capabilities()
        voice_id = str(voice_descriptor["id"])
        cell = Cell(
            engine=str(getattr(engine, "id", "unknown")),
            voice=voice_id,
            language=voice_language(voice_descriptor),
            voice_descriptor=dict(voice_descriptor),
            capabilities={
                "sample_rates": list(getattr(capabilities, "sample_rates", []) or []),
                "native_rate": bool(getattr(capabilities, "native_rate", False)),
                "styles": list(getattr(capabilities, "styles", []) or []),
                "supports_streaming": bool(getattr(capabilities, "supports_streaming", False)),
            },
        )
        for entry in entries:
            cell.entries.append(self._measure_entry(engine, voice_id, entry, capabilities))

        if not entries:
            return cell

        # Probes want the longest sentence (the most stable rate measurement), but a
        # sentence this engine cannot render must not take the whole probe down: the
        # entry-level failure is already reported. Try the longest few in turn.
        ordered = sorted(entries, key=lambda entry: (-len(entry.text), entry.id))
        consistency: dict[str, Any] = {}
        prosody: dict[str, Any] = {}
        streaming: dict[str, Any] = {}
        probe_entry = ordered[0]
        for probe_entry in ordered[:3]:
            consistency, prosody = self._measure_probes(
                engine, voice_id, probe_entry.text, capabilities
            )
            streaming = self._measure_ttfa(engine, probe_entry.text, voice_id, capabilities)
            if not consistency.get("checks") and not prosody.get("checks"):
                break

        if streaming.get("ttfa_s") is not None:
            streaming["entry"] = probe_entry.id
        consistency["entry"] = probe_entry.id
        prosody["entry"] = probe_entry.id
        cell.streaming = streaming
        cell.speaker_consistency = consistency
        cell.prosody = prosody
        return cell

    def run(self) -> list[Cell]:
        """Measure every engine/voice cell in this run.

        Raises :class:`app.eval.engines.EvalEngineError` when the selection produces
        no measurable cell. An empty report that says ``passed`` is the one failure
        mode a measurement harness must never have.
        """
        languages = self._selected_languages()
        cells: list[Cell] = []
        self.skipped = []
        for engine in self.engines:
            support = getattr(engine, "g2p_support", None)
            for voice_descriptor in select_voices(engine, languages, self.voices_spec):
                language = voice_language(voice_descriptor)
                if callable(support):
                    ok, reason = support(language)
                    if not ok:
                        # Measuring a language the pipeline cannot phonemise
                        # produces fluent nonsense and a misleading WER. Record
                        # the skip with the engine's own reason instead.
                        self.skipped.append(
                            {
                                "engine": str(getattr(engine, "id", "unknown")),
                                "voice": str(voice_descriptor.get("id")),
                                "language": language,
                                "status": "unsupported_language",
                                "reason": reason,
                            }
                        )
                        continue
                entries = (
                    list(self.corpus)
                    if self.config.cross_language
                    else self.corpus.for_language(language)
                )
                if not entries:
                    continue
                cells.append(self.run_cell(engine, voice_descriptor, entries))
        if not cells:
            raise EvalEngineError(
                "no engine/voice cell matched the corpus languages "
                f"({', '.join(languages) or 'none'}); check --engines/--voices"
            )
        return cells

    def _selected_languages(self) -> list[str]:
        """Corpus languages this run measures, validating an explicit filter."""
        available = self.corpus.languages
        if not self.config.languages:
            return available
        unknown = [language for language in self.config.languages if language not in available]
        if unknown:
            raise EvalEngineError(
                f"language(s) {unknown} are not in the corpus (available: {', '.join(available)})"
            )
        return [language for language in available if language in set(self.config.languages)]

    # -- metadata ----------------------------------------------------------

    def metadata(self) -> dict[str, Any]:
        """Engine identities and environment facts for the report header."""
        return {
            "engines": [engine_identity(engine) for engine in self.engines],
            "transcriber": self.transcriber.describe(),
            "speaker_embedder": {
                "id": getattr(self.embedder, "id", "unknown"),
                "note": (
                    "log-mel statistics over repeated renders; a stability proxy for one "
                    "engine/voice, not a biometric speaker-verification model"
                ),
            },
            "environment": {
                "python": sys.version.split()[0],
                "platform": sys.platform,
                "cpu_count": _cpu_count(),
                # Deployment inputs that change the audio but are not per-request:
                # recorded so a baseline reader can see which render path produced
                # the numbers. Model files are identified by name and size, not by
                # path, so a committed baseline does not embed one machine's layout.
                "models": _model_inventory(),
                "phonetic_preprocessing": settings.phonetic_processing_enabled,
                "kokoro_default_voice": settings.kokoro_default_voice,
                "kokoro_default_speed": settings.kokoro_default_speed,
            },
        }


def _cpu_count() -> int | None:
    return os.cpu_count()


def _model_inventory() -> dict[str, int]:
    """Model files the harness can see, by name and size in bytes.

    Name plus size is enough to tell two model builds apart for the purpose of
    reading a baseline, and it keeps an absolute local path out of the report.
    """
    inventory: dict[str, int] = {}
    models_dir = Path(settings.models_dir)
    for pattern in ("kokoro*.onnx", "voices*.bin"):
        for path in sorted(models_dir.glob(pattern)):
            try:
                inventory[path.name] = path.stat().st_size
            except OSError:  # pragma: no cover - a model removed mid-run
                continue
    return inventory


def _decode(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    """Decode WAV bytes to float32 samples for the embedder."""
    with sf.SoundFile(io.BytesIO(wav_bytes)) as handle:
        sample_rate = int(handle.samplerate)
        data = handle.read(dtype="float32", always_2d=True)
    return np.asarray(data, dtype=np.float32), sample_rate


__all__ = [
    "Cell",
    "EvaluationRunner",
    "GRADE_ORDER",
    "Render",
    "RunnerConfig",
    "select_voices",
    "summarise",
    "voice_language",
    "voice_rank",
]
