"""STT service wrapping faster-whisper for speech-to-text transcription.

Phase 7 (SPEECH-008) additions:

* **Temp-file ownership** — every request writes its audio to a single temporary
  file which is removed on success, on error and when the caller abandons the
  streaming generator (client disconnect). See :meth:`STTService._temporary_audio_file`.
* **Voice-activity detection** — silence is trimmed locally (Silero, via
  faster-whisper) before transcription. Configurable and disableable per request.
* **Word timestamps** — opt-in ``words`` per segment.
* **Model tiering** — any faster-whisper model mounted under ``models_dir`` can be
  selected per request; unknown ids raise :class:`ModelNotAvailableError`.
* **Language confidence** — detected language probability and the top alternatives
  are returned so a client can ask the user to confirm.
"""

from __future__ import annotations

import contextlib
import logging
import os
import tempfile
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from app.config import settings
from app.services.vad import TrimResult, VoiceActivityDetector

logger = logging.getLogger(__name__)

#: Languages whose base-tier Whisper output picks the wrong *script*.
#:
#: Spelling is not content: Whisper's decoder chooses between Devanagari and
#: Perso-Arabic for Hindi, and between Simplified and Traditional for Chinese,
#: from acoustic evidence alone. On the pinned ``base`` model it picks Perso-Arabic
#: for Hindi and Traditional for Chinese, so a correct render scores as a total
#: miss against a Devanagari/Simplified reference purely because the characters
#: differ. Each prompt is a fixed, language-generic sentence that names the script
#: and carries no content from any corpus; it steers character set only. Measured
#: on ``base``: Hindi `نمستی ورنر...` -> `नमस्टी वारनर...`, Chinese `你要為Ena...`
#: -> `你要为Ena...`. Same audio, same model, same beam width.
#:
#: Review finding F4: this lived only in the evaluation harness, so
#: ``/api/stt/transcribe`` still returned Perso-Arabic for Hindi while the harness
#: reported Devanagari. It now lives here, is applied by :meth:`STTService.transcribe`
#: (config-gated by ``whisper_script_prompts_enabled``) and the harness simply
#: relies on it, so the API and the acceptance measurement agree.
SCRIPT_PROMPTS: dict[str, str] = {
    "hi": "यह एक हिन्दी में लिखा हुआ वाक्य है।",
    "zh": "这是一个用简体中文写的句子。",
}

#: Prefix for temporary audio files so any leak is attributable in the temp dir.
TEMP_AUDIO_PREFIX = "reactory-stt-"

#: Directory-name prefixes stripped when deriving a model id (``whisper-base`` ->
#: ``base``), so a mounted folder name and the API model id agree.
_MODEL_DIR_PREFIXES = ("faster-whisper-", "whisper-")

#: Directory names that do not identify a model themselves. When ``whisper_model_path``
#: points at one of these the id comes from ``whisper_model_size`` instead.
_GENERIC_MODEL_DIR_NAMES = frozenset({"models", "model", "whisper", "faster-whisper", "ct2"})

#: How many alternative languages to report.
LANGUAGE_PROBABILITY_LIMIT = 3


class ModelNotAvailableError(RuntimeError):
    """Raised when a caller asks for a model that is not mounted locally."""

    def __init__(self, model: str | None, available: list[str]) -> None:
        self.model = model
        self.available = list(available)
        available_text = ", ".join(self.available) if self.available else "none"
        super().__init__(
            f"Model '{model}' is not available on this deployment. Available models: {available_text}"
        )


class _TranscriptionRun:
    """State for one in-flight transcription (valid only inside its temp-file scope)."""

    def __init__(
        self,
        model_id: str,
        trim: TrimResult,
        raw_segments: Iterator[Any] | None,
        info: Any,
        word_timestamps: bool,
    ) -> None:
        self.model_id = model_id
        self.trim = trim
        self._raw_segments = raw_segments
        self.info = info
        self.word_timestamps = word_timestamps

    def __iter__(self) -> Iterator[dict]:
        if self._raw_segments is None:
            return
        for segment in self._raw_segments:
            yield self._segment_payload(segment)

    def _segment_payload(self, segment: Any) -> dict:
        start = self._map_time(getattr(segment, "start", 0.0) or 0.0)
        end = self._map_time(getattr(segment, "end", 0.0) or 0.0)
        payload: dict = {
            "start": round(start, 3),
            "end": round(max(end, start), 3),
            "text": str(getattr(segment, "text", "") or "").strip(),
        }
        if self.word_timestamps:
            payload["words"] = self._words(getattr(segment, "words", None))
        return payload

    def _words(self, words: Any) -> list[dict]:
        payload: list[dict] = []
        previous_end = 0.0
        for word in words or []:
            start = self._map_time(getattr(word, "start", 0.0) or 0.0)
            end = self._map_time(getattr(word, "end", 0.0) or 0.0)
            # Guard the contract: timestamps must increase monotonically. A word
            # that starts before the previous one ends is clamped, never reordered.
            start = max(start, previous_end)
            end = max(end, start)
            previous_end = end
            probability = getattr(word, "probability", None)
            payload.append(
                {
                    "word": str(getattr(word, "word", "") or "").strip(),
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "probability": (
                        round(float(probability), 4) if probability is not None else None
                    ),
                }
            )
        return payload

    def _map_time(self, value: float) -> float:
        if not self.trim.applied:
            return float(value)
        return float(self.trim.map_time(float(value)))


class STTService:
    """Speech-to-text service using faster-whisper (CTranslate2)."""

    def __init__(self, vad: VoiceActivityDetector | None = None) -> None:
        self._models: dict[str, Any] = {}
        self._paths: dict[str, Path] = {}
        self._default_model: str | None = None
        self._ready = False
        self._lock = threading.Lock()
        self._injected_vad = vad

    # ------------------------------------------------------------------
    # Model discovery and loading
    # ------------------------------------------------------------------
    @staticmethod
    def _normalise_model_id(name: str) -> str:
        for prefix in _MODEL_DIR_PREFIXES:
            if name.startswith(prefix):
                return name[len(prefix) :]
        return name

    def _model_id_for_path(self, path: Path) -> str:
        """Derive the model id for an explicitly configured model directory.

        A directory named after the model (``whisper-tiny``) supplies its own id. A
        generic directory (``models``, ``whisper``, ...) does not describe a model,
        so the configured ``whisper_model_size`` is used instead of leaking the
        folder name into the API.
        """
        name = self._normalise_model_id(path.name)
        if not name or name.lower() in _GENERIC_MODEL_DIR_NAMES:
            return self._normalise_model_id(settings.whisper_model_size)
        return name

    def discover_models(self) -> dict[str, Path]:
        """Map model id -> local directory for every model mounted on this host.

        A model is "available" when a directory under ``models_dir`` (or the
        explicitly configured ``whisper_model_path``) contains ``model.bin``. Only
        local files are considered — faster-whisper is always constructed with
        ``local_files_only=True``, so nothing is ever downloaded at runtime.
        """
        found: dict[str, Path] = {}
        configured = Path(settings.whisper_model_path) if settings.whisper_model_path else None
        if configured is not None and configured.exists() and (configured / "model.bin").exists():
            found[self._model_id_for_path(configured)] = configured

        root = Path(settings.models_dir)
        if root.exists():
            # Legacy layout: the models directory itself is a single model.
            if (root / "model.bin").exists():
                found.setdefault(self._normalise_model_id(settings.whisper_model_size), root)
            for candidate in sorted(root.iterdir()):
                if candidate.is_dir() and (candidate / "model.bin").exists():
                    found.setdefault(self._normalise_model_id(candidate.name), candidate)
        return found

    def _refresh_models(self) -> dict[str, Path]:
        with self._lock:
            self._paths = self.discover_models()
            return self._paths

    def _sync_models(self) -> dict[str, Path]:
        """Merge freshly discovered models into the path map.

        The scan runs on every call so a model mounted after boot shows up in
        ``available_models``/``/api/stt/capabilities``. Merging is monotonic: an entry
        is never dropped while the process runs (a directory removed underneath a
        loaded model must not break the loaded instance), and explicitly registered
        paths survive when the filesystem scan finds nothing.
        """
        try:
            discovered = self.discover_models()
        except OSError:
            logger.warning("Could not scan %s for models", settings.models_dir, exc_info=True)
            discovered = {}
        with self._lock:
            self._paths.update(discovered)
            return dict(self._paths)

    @property
    def available_models(self) -> list[str]:
        """Ids of the models mounted locally, sorted for stable output."""
        return sorted(self._sync_models())

    @property
    def loaded_models(self) -> list[str]:
        return sorted(self._models)

    @property
    def default_model(self) -> str | None:
        return self._default_model

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def vad(self) -> VoiceActivityDetector:
        if self._injected_vad is not None:
            return self._injected_vad
        return VoiceActivityDetector(
            enabled=settings.whisper_vad_enabled,
            threshold=settings.whisper_vad_threshold,
            min_silence_ms=settings.whisper_vad_min_silence_ms,
            min_speech_ms=settings.whisper_vad_min_speech_ms,
            pad_ms=settings.whisper_vad_pad_ms,
        )

    async def initialize(self) -> None:
        """Load the default faster-whisper model. Call once at startup."""
        paths = self._refresh_models()
        if not paths:
            logger.warning(
                "No local Whisper model files found in %s. STT service disabled. "
                "Mount a faster-whisper model (a directory containing model.bin) into "
                "%s/whisper-%s to enable.",
                settings.models_dir,
                settings.models_dir,
                settings.whisper_model_size,
            )
            self._ready = False
            return

        preferred = self._normalise_model_id(settings.whisper_model_size)
        name = preferred if preferred in paths else min(paths)
        if name != preferred:
            logger.warning(
                "Configured Whisper model '%s' is not mounted; falling back to '%s' "
                "(available: %s)",
                preferred,
                name,
                ", ".join(sorted(paths)),
            )
        try:
            self._load_model(name)
            self._default_model = name
            self._ready = True
            logger.info(
                "faster-whisper model '%s' loaded (available: %s)",
                name,
                ", ".join(sorted(paths)),
            )
        except Exception:
            logger.exception("Failed to load faster-whisper model '%s'", name)
            self._ready = False

    def _load_model(self, model_id: str) -> Any:
        """Load (and cache) one local model, raising when it is not available."""
        cached = self._models.get(model_id)
        if cached is not None:
            return cached

        path = self._sync_models().get(model_id)
        if path is None:
            raise ModelNotAvailableError(model_id, self.available_models)

        from faster_whisper import WhisperModel

        with self._lock:
            cached = self._models.get(model_id)
            if cached is not None:
                return cached
            logger.info(
                "Loading faster-whisper model '%s' from %s, device=%s, compute=%s",
                model_id,
                path,
                settings.whisper_device,
                settings.whisper_compute_type,
            )
            model = WhisperModel(
                str(path),
                device=settings.whisper_device,
                compute_type=settings.whisper_compute_type,
                local_files_only=True,
            )
            self._models[model_id] = model
            return model

    def _resolve_model(self, model: str | None) -> tuple[str, Any]:
        available = self.available_models
        if not available:
            raise ModelNotAvailableError(model or settings.whisper_model_size, available)
        requested = model or self._default_model or available[0]
        requested = self._normalise_model_id(requested)
        if requested not in available:
            raise ModelNotAvailableError(requested, available)
        return requested, self._load_model(requested)

    def capabilities(self) -> dict:
        """Machine-readable description of the STT surface."""
        return {
            "ready": self._ready,
            "default_model": self._default_model,
            "available_models": self.available_models,
            "loaded_models": self.loaded_models,
            "device": settings.whisper_device,
            "compute_type": settings.whisper_compute_type,
            "streaming": True,
            "word_timestamps": True,
            "language_confidence": True,
            "vad": self.vad.describe(),
        }

    # ------------------------------------------------------------------
    # Transcription
    # ------------------------------------------------------------------
    def transcribe(
        self,
        audio_bytes: bytes,
        language: str | None = None,
        *,
        model: str | None = None,
        word_timestamps: bool | None = None,
        vad: bool | None = None,
        temperature: float | None = None,
        initial_prompt: str | None = None,
    ) -> dict:
        """Transcribe audio bytes to text.

        Args:
            audio_bytes: Audio file bytes (WAV, WebM, OGG, MP3, …).
            language: Optional language code. Auto-detects if ``None``.
            model: Optional model id; must be mounted locally.
            word_timestamps: Emit ``words`` per segment (defaults to the deployment setting).
            vad: Override voice-activity detection for this request.
            temperature: Optional decoding temperature. ``None`` keeps faster-whisper's
                default fallback ladder (greedy, then sampled retries at 0.2-1.0), which
                is right for a live request and wrong for a measurement: on garbage
                audio the fallback samples, so two identical inputs can transcribe
                differently. The evaluation harness passes ``0.0`` to make round-trip
                WER reproducible.
            initial_prompt: Optional decoder prompt. The evaluation harness uses a
                fixed, language-generic sentence to select the script Whisper should
                emit for Hindi (Devanagari, not Perso-Arabic) and Chinese (Simplified,
                not Traditional). It biases character set, not content.

        Returns:
            Dict with keys: text, language, language_confidence,
            language_probabilities, needs_language_confirmation, segments,
            duration, model, vad_applied, vad_failed, speech_duration.

            ``duration`` is the duration of the audio the caller sent and is not
            shortened by VAD trimming; ``speech_duration`` reports the speech
            portion. ``vad_failed`` marks a detector that could not run, in which
            case the original audio was transcribed.
        """
        if initial_prompt is None and settings.whisper_script_prompts_enabled:
            # A script prompt selects the character set, not the content. The
            # caller can always override it (the evaluation harness does not need
            # to) and an operator can disable it wholesale.
            initial_prompt = SCRIPT_PROMPTS.get(
                (language or "").strip().lower().split("-")[0]
            )

        with self._transcription(
            audio_bytes, language, model, word_timestamps, vad, temperature, initial_prompt
        ) as run:
            segments: list[dict] = []
            text_parts: list[str] = []
            for payload in run:
                segments.append(payload)
                if payload["text"]:
                    text_parts.append(payload["text"])
            return self._build_result(run, segments, text_parts)

    def transcribe_streaming(
        self,
        audio_bytes: bytes,
        language: str | None = None,
        *,
        model: str | None = None,
        word_timestamps: bool | None = None,
        vad: bool | None = None,
        temperature: float | None = None,
    ) -> Iterator[dict]:
        """Yield one dict per segment as the decoder produces it.

        Each yield is ``{start, end, text, partial_text, language}`` (plus ``words``
        when requested). The temporary audio file is removed when the generator is
        exhausted *or* closed — closing is what happens on a client disconnect.
        """
        with self._transcription(
            audio_bytes, language, model, word_timestamps, vad, temperature, None
        ) as run:
            accumulated: list[str] = []
            for payload in run:
                accumulated.append(payload["text"])
                item = {
                    "start": payload["start"],
                    "end": payload["end"],
                    "text": payload["text"],
                    "partial_text": " ".join(part for part in accumulated if part),
                    "language": self._info_value(run.info, "language"),
                }
                if "words" in payload:
                    item["words"] = payload["words"]
                yield item

    @contextlib.contextmanager
    def _transcription(
        self,
        audio_bytes: bytes,
        language: str | None,
        model: str | None,
        word_timestamps: bool | None,
        vad: bool | None,
        temperature: float | None = None,
        initial_prompt: str | None = None,
    ) -> Iterator[_TranscriptionRun]:
        """Own the temp file for the whole duration of one transcription.

        Everything that reads the audio — including lazy iteration of the segment
        generator — happens inside this context, so the file cannot outlive the call
        on any path.
        """
        model_id, whisper = self._resolve_model(model)
        want_words = settings.whisper_word_timestamps if word_timestamps is None else bool(word_timestamps)
        vad_enabled = settings.whisper_vad_enabled if vad is None else bool(vad)

        with self._temporary_audio_file(audio_bytes) as audio_path:
            trim = self.vad.trim_file(audio_path) if vad_enabled else TrimResult()
            if trim.failed:
                # Detection could not run (missing module/asset, decoder mismatch,
                # detector exception). This is explicitly NOT silence: fall through
                # and transcribe the original audio so the request still works.
                logger.warning(
                    "Voice-activity detection failed for model '%s'; transcribing the "
                    "untrimmed audio",
                    model_id,
                )
            if trim.is_silence:
                # Detection ran and found no speech: nothing to decode.
                logger.info("VAD found no speech in %.2fs of audio", trim.original_duration)
                yield _TranscriptionRun(model_id, trim, None, None, want_words)
                return

            source: Any = trim.samples if trim.applied else audio_path
            kwargs: dict = {
                "beam_size": 5,
                # VAD is done explicitly above, so the decoder must not run it again.
                "vad_filter": False,
                "word_timestamps": want_words,
            }
            if language:
                kwargs["language"] = language
            if temperature is not None:
                # A scalar disables faster-whisper's sampled fallback ladder.
                kwargs["temperature"] = float(temperature)
            if initial_prompt:
                kwargs["initial_prompt"] = initial_prompt

            raw_segments, info = whisper.transcribe(source, **kwargs)
            logger.debug(
                "Transcribing with model '%s' (vad_applied=%s, vad_failed=%s, words=%s)",
                model_id,
                trim.applied,
                trim.failed,
                want_words,
            )
            yield _TranscriptionRun(model_id, trim, iter(raw_segments), info, want_words)

    def _build_result(
        self, run: _TranscriptionRun, segments: list[dict], text_parts: list[str]
    ) -> dict:
        info = run.info
        # ``duration`` is the duration of the audio the caller sent, not of the
        # VAD-trimmed signal handed to the decoder. ``speech_duration`` reports how
        # much of it was speech.
        duration = run.trim.original_duration
        if not duration:
            decoded = self._info_value(info, "duration")
            duration = decoded if isinstance(decoded, (int, float)) else 0.0
        language = self._info_value(info, "language")
        confidence = self._info_value(info, "language_probability")
        confidence_value = float(confidence) if isinstance(confidence, (int, float)) else None
        raw_probabilities = self._info_value(info, "all_language_probs") or []
        probabilities = [
            {"language": str(code), "probability": round(float(probability), 4)}
            for code, probability in list(raw_probabilities)[:LANGUAGE_PROBABILITY_LIMIT]
            if isinstance(probability, (int, float))
        ]
        threshold = settings.whisper_language_confidence_threshold
        return {
            "text": " ".join(text_parts),
            "language": language if isinstance(language, str) else None,
            "language_confidence": (
                round(confidence_value, 4) if confidence_value is not None else None
            ),
            "language_probabilities": probabilities or None,
            "needs_language_confirmation": bool(
                confidence_value is not None and confidence_value < threshold
            ),
            "segments": segments,
            "duration": round(float(duration or 0.0), 3),
            "model": run.model_id,
            "vad_applied": run.trim.applied,
            "vad_failed": run.trim.failed,
            "speech_duration": run.trim.speech_duration if run.trim.applied else None,
        }

    @staticmethod
    def _info_value(info: Any, name: str) -> Any:
        if info is None:
            return None
        return getattr(info, name, None)

    # ------------------------------------------------------------------
    # Temporary audio files
    # ------------------------------------------------------------------
    @staticmethod
    @contextlib.contextmanager
    def _temporary_audio_file(audio_bytes: bytes) -> Iterator[str]:
        """Write ``audio_bytes`` to a temp file, guaranteeing its removal.

        faster-whisper accepts a path (or a file-like object) and decodes lazily, so
        the file must exist for as long as the caller iterates. The context manager
        is therefore held by :meth:`_transcription` for the whole call: success,
        exception and generator close (client disconnect) all pass through the
        ``finally`` and unlink the file.
        """
        suffix = STTService._detect_suffix(audio_bytes)
        # Deliberately not managed by a ``with`` block: the file must outlive this
        # function and is owned by the ``finally`` below (see docstring).
        handle = tempfile.NamedTemporaryFile(prefix=TEMP_AUDIO_PREFIX, suffix=suffix, delete=False)  # noqa: SIM115
        path = handle.name
        try:
            try:
                handle.write(audio_bytes)
                handle.flush()
            finally:
                handle.close()
            yield path
        finally:
            # Covers a failed initial write as well as every consumer outcome.
            with contextlib.suppress(OSError):
                os.unlink(path)

    @staticmethod
    def _detect_suffix(audio_bytes: bytes) -> str:
        """Pick a file extension from the container magic bytes.

        faster-whisper can handle WAV, MP3, OGG, FLAC, etc. via ffmpeg/av; the
        suffix only has to be honest enough for the demuxer to try.
        """
        if audio_bytes[:4] == b"\x1a\x45\xdf\xa3":
            return ".webm"
        if audio_bytes[:4] == b"OggS":
            return ".ogg"
        if audio_bytes[:4] == b"RIFF":
            return ".wav"
        if audio_bytes[:4] == b"fLaC":
            return ".flac"
        if audio_bytes[:3] == b"ID3" or audio_bytes[:2] == b"\xff\xfb":
            return ".mp3"
        return ".wav"
