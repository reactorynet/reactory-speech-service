"""Chatterbox expressive engine adapter (SPEECH-005, Phase 4).

Chatterbox (Resemble AI, MIT) is the optional expressive engine.  It adds tone:
Kokoro can change pace and timbre but has no concept of emotion, so
``direction.style`` can only be reported as ignored there.

Binding constraints for this module:

* **Optional and lazy.**  Nothing here imports ``torch`` or ``chatterbox`` at
  module import time.  The backend is imported inside :meth:`ChatterboxEngine.load`,
  and the adapter module itself is only imported when ``CHATTERBOX_ENABLED`` is
  true.  The default ``speech-core`` image therefore neither grows nor gains a
  torch dependency.
* **No watermark in production audio.**  ``ChatterboxTTS`` applies a Perth
  watermark on every return path with no off switch.  ADR-0003 decision 4 requires
  clean production audio, so :func:`neutralise_watermarker` replaces
  ``model.watermarker`` with a no-op *before* any audio is returned.  This is not
  a configuration flag and there is no code path that leaves it on in production.
* **Exaggeration is bounded to 0.3-0.7.**  It co-varies with pace (the same
  sentence is roughly 30% shorter at 1.0 than at 0.0) and degrades diction above
  ~0.7.  No style may leave that range, and ``prosody.rate`` is never stacked on
  top of it.
* **Measured, not assumed.**  The compute-cost advisory is published in
  ``/api/tts/capabilities`` because the routing signal at RTF 0.13 is
  cost-per-second of audio, not ``native_rate``.  See
  ``docs/measurements/chatterbox-cpu.md``.
* **The model is never downloaded at container build time.**  Weights live in the
  models volume, with the licence and per-file SHA-256 recorded in
  ``manifest.json``.  A missing or unverifiable model is a clear
  ``EngineUnavailableError`` that the router turns into a Kokoro fallback, never a
  crash or a 500.
"""

from __future__ import annotations

import hashlib
import inspect
import io
import json
import logging
import re
import tempfile
import threading
from collections.abc import Iterator
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from app.config import settings
from app.core.contracts import (
    AudioChunk,
    AudioResult,
    DirectionPlan,
    EngineCapabilities,
    EngineRequest,
    VoiceDescriptor,
)
from app.engines.base import EngineUnavailableError, UnknownVoiceError
from app.services import voice_registry
from app.services.voice_registry import (
    all_canonical_languages,
    build_catalogue,
    canonical_language,
    engine_locale,
)

logger = logging.getLogger(__name__)

#: Engine id used in descriptors, routing and registry records.
CHATTERBOX_ENGINE = "chatterbox"

#: Native output sample rate of Chatterbox, in Hz (measured, not assumed).
CHATTERBOX_SAMPLE_RATE = 24000

#: Licence and attribution required by ADR-0001 and surfaced by the registry.
CHATTERBOX_LICENSE = "MIT"
CHATTERBOX_ATTRIBUTION = "Chatterbox (Resemble AI)"

#: Chatterbox 0.1.7 is English-only.
CHATTERBOX_LANGUAGES = ["en-US"]

#: Styles this adapter can render by varying ``exaggeration``.
#:
#: ``exaggeration`` controls *expressiveness intensity*, not valence: it cannot
#: make speech specifically warm, sad or empathetic.  Each mapping below is a
#: defensible intensity, not a claim to model the emotion.  ``whisper`` is absent
#: on purpose — there is no breathiness/whisper control, so it must be reported
#: as ignored rather than approximated.
STYLE_EXAGGERATION: dict[str, float] = {
    "narration": 0.30,
    "calm": 0.32,
    "serious": 0.36,
    "warm": 0.44,
    # 0.62 mangled short words and proper nouns: round-tripping our own output
    # through STT heard "I've been" for "Hi Vin" at 0.62, while 0.44 rendered
    # "Hi, Vin" correctly. A clone inheriting the old default would degrade the
    # names of the very person whose voice it reuses (SPEECH-006 known issue).
    "empathetic": 0.44,
    "cheerful": 0.66,
    "excited": 0.70,
}

#: The styles the ``standard`` variant advertises.  ``neutral`` means "no style
#: requested".  Turbo advertises only ``neutral``: see
#: :data:`VARIANT_SUPPORTS_EXAGGERATION`.
CHATTERBOX_STYLES: tuple[str, ...] = ("neutral", *STYLE_EXAGGERATION)
#: Hard bounds on ``exaggeration``.  Above ~0.7 diction degrades; below 0.3 the
#: output is effectively flat, so neither end is useful.
EXAGGERATION_MIN = 0.3
EXAGGERATION_MAX = 0.7

#: Engine default when no style is requested.
EXAGGERATION_NEUTRAL = 0.5

#: How far an explicit ``intensity`` may move a style's exaggeration, keeping the
#: result inside the hard bounds above.
INTENSITY_SPAN = 0.16

#: Paralinguistic tags Chatterbox understands.  They are declared so a caller can
#: use them, but the adapter never inserts them silently: adding a laugh to an
#: arbitrary sentence is not the caller's intent.
PARALINGUISTIC_TAGS: tuple[str, ...] = (
    "[laugh]",
    "[chuckle]",
    "[cough]",
    "[sigh]",
    "[gasp]",
    "[groan]",
    "[sniff]",
    "[clear_throat]",
)

#: Maximum characters per ``generate`` call.  Longer text is split at sentence
#: boundaries and concatenated, so a long narration still renders.
MAX_CHARS_PER_CALL = 600

#: Silence inserted between rendered chunks, in seconds.
CHUNK_GAP_SECONDS = 0.08

#: Leading bytes of a clone embedding written by the conditioning container's
#: own ``save`` (tensors only).  A blob without it is the legacy pickle format.
CONDS_EMBEDDING_MAGIC = b"CBXCONDS1\n"

#: Measured compute-cost advisory, published in ``/api/tts/capabilities``.
#:
#: The numbers are from ``docs/measurements/chatterbox-cpu.md`` on the reference
#: 8-core CPU node.  RTF 0.13 means 1 second of audio costs ~7.7 seconds of CPU:
#: this engine is for batch narration, not interactive turns.  Thread scaling is
#: flat because the bottleneck is sequential token decoding, so the advisory
#: deliberately does not recommend adding threads.
#: Capability notes shared by every deployment.  Written as explicit ``+``
#: concatenations rather than adjacent literals inside the ``capability_notes``
#: list: adjacent literals there are indistinguishable from a missing comma and
#: are flagged by static analysis (CodeQL ``py/implicit-string-concatenation-in-list``).
_CLONING_NOTE = (
    "Chatterbox supports zero-shot cloning and this adapter can render a stored speaker "
    + "embedding. Clone *creation* is still gated per deployment by the consent layer "
    + "(SPEECH-006, ADR-0003): /health reports whether cloning is configured."
)
_WATERMARK_NOTE = (
    "Production audio carries no Perth watermark: the adapter replaces the model "
    + "watermarker with a no-op (ADR-0003 decision 4)."
)
_EXAGGERATION_NOTE = (
    "exaggeration is bounded to 0.3-0.7; prosody.rate is never stacked on top of it "
    + "because the two co-vary."
)
_NO_EXAGGERATION_NOTE = "This variant ignores exaggeration, so no style is applied."
_STREAMING_NOTE = "Chatterbox renders a complete buffer; it does not stream incrementally."

COMPUTE_COST_ADVISORY: dict[str, Any] = {
    "measured_realtime_factor": 0.13,
    "speed_vs_realtime": "0.13x realtime (about 7.7x slower than realtime)",
    "load_seconds": 21.9,
    "resident_memory_mb": 3976,
    "sample_rate": CHATTERBOX_SAMPLE_RATE,
    "thread_scaling": (
        "flat: sequential token decoding is the bottleneck, so adding threads does not help"
    ),
    "routing_signal": "cost per second of synthesised audio, not native_rate",
    "suitable_for": ["batch narration", "pre-rendered content", "asynchronous jobs"],
    "not_suitable_for": ["interactive turn-taking", "live conversation", "low-latency streaming"],
    "measured_on": "8-core CPU reference node",
    "source": "docs/measurements/chatterbox-cpu.md",
}

#: Upstream weights repositories, per variant.  ``standard`` is ``ChatterboxTTS``
#: (350M); ``turbo`` is ``ChatterboxTurboTTS``.  There is no "Nano" class in
#: chatterbox-tts 0.1.7 — the 110M edge class described by the original ticket
#: does not exist in the released package.
VARIANT_REPOS: dict[str, str] = {
    "standard": "ResembleAI/chatterbox",
    "turbo": "ResembleAI/chatterbox-turbo",
}

#: Backend class locations, per variant.
VARIANT_BACKENDS: dict[str, tuple[str, str]] = {
    "standard": ("chatterbox.tts", "ChatterboxTTS"),
    "turbo": ("chatterbox.tts_turbo", "ChatterboxTurboTTS"),
}

#: Whether a variant can actually honour ``exaggeration`` — the source of truth
#: for the advertised style vocabulary.
#:
#: This is a per-variant fact read from the upstream source, **not** a check for
#: an ``exaggeration`` parameter in the signature.  ``ChatterboxTurboTTS.generate``
#: accepts ``exaggeration`` and then ignores it:
#:
#:     if cfg_weight > 0.0 or exaggeration > 0.0 or min_p > 0.0:
#:         logger.warning("CFG, min_p and exaggeration are not supported by Turbo
#:                         version and will be ignored.")
#:
#: and ``ChatterboxTurboTTS.from_local`` builds its config with
#: ``hp.emotion_adv = False``, so there is no emotion embedding to drive.  A
#: signature-name check therefore advertises controls that do nothing, which is
#: exactly the silent approximation this adapter exists to avoid.
VARIANT_SUPPORTS_EXAGGERATION: dict[str, bool] = {
    "standard": True,
    "turbo": False,
}

#: Why a variant cannot express tone, for the ``ignored`` warning and the
#: capability notes.
VARIANT_STYLE_CAVEAT: dict[str, str] = {
    "turbo": (
        "Chatterbox Turbo accepts but ignores exaggeration/cfg/min_p "
        "('not supported by Turbo version and will be ignored', hp.emotion_adv=False), "
        "so it cannot render tone; use CHATTERBOX_VARIANT=standard for expressive styles"
    ),
}

#: The files ``from_local`` actually reads, per variant.  Derived from the
#: chatterbox-tts 0.1.7 source, and pinned against the real upstream repository
#: layout by ``tests/data/chatterbox-upstream.json``.
#:
#: * ``tts.py::ChatterboxTTS.from_local`` reads ``ve.safetensors``,
#:   ``t3_cfg.safetensors``, ``s3gen.safetensors``, ``tokenizer.json`` (its
#:   ``EnTokenizer`` vocab) and the optional ``conds.pt``.
#: * ``tts_turbo.py::ChatterboxTurboTTS.from_local`` reads ``ve.safetensors``,
#:   ``t3_turbo_v1.safetensors``, ``s3gen_meanflow.safetensors``, the HuggingFace
#:   tokenizer files that ``AutoTokenizer.from_pretrained`` needs for a
#:   ``GPT2Tokenizer`` (``tokenizer_config.json`` + ``vocab.json`` +
#:   ``merges.txt``, plus ``special_tokens_map.json`` / ``added_tokens.json``)
#:   and the optional ``conds.pt``.
#:
#: There is **no** ``tokenizer.json`` in ``ResembleAI/chatterbox-turbo``.  The
#: standard repository holds nine redundant checkpoints (``*.pt`` duplicates,
#: v2/v3 multilingual T3 weights, extra vocoders) on top of these files, which is
#: why a whole-repository download is ~13.87 GB where ~3.19 GB is read.
VARIANT_FILES: dict[str, tuple[str, ...]] = {
    "standard": (
        "ve.safetensors",
        "t3_cfg.safetensors",
        "s3gen.safetensors",
        "tokenizer.json",
        "conds.pt",
    ),
    "turbo": (
        "ve.safetensors",
        "t3_turbo_v1.safetensors",
        "s3gen_meanflow.safetensors",
        "vocab.json",
        "merges.txt",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "conds.pt",
    ),
}

#: Whole-repository size, for the operator-facing comparison.
#:
#: Captured from the HuggingFace API (``?blobs=true``) on **2026-11-12**:
#: ``ResembleAI/chatterbox`` @ 5bb1f6ee = 18 files / 13,866,212,931 B;
#: ``ResembleAI/chatterbox-turbo`` @ 749d1c1a = 13 files / 4,044,184,736 B.
#: Informational only — nothing asserts these, so an upstream upload makes the
#: advisory stale, never the code wrong.  The file *layout* is pinned separately
#: by ``tests/data/chatterbox-upstream.json``.
VARIANT_REPO_BYTES: dict[str, int] = {
    "standard": 13_866_212_931,  # 13.87 GB across 18 files
    "turbo": 4_044_184_736,  # 4.04 GB across 13 files
}

#: Bytes actually read by ``from_local``, per variant.  Asserted against the
#: checked-in upstream listing, so a stale value fails a test.
VARIANT_REQUIRED_BYTES: dict[str, int] = {
    "standard": 3_191_966_992,  # 3.19 GB across 5 files
    "turbo": 2_987_680_596,  # 2.99 GB across 9 files
}

MANIFEST_NAME = "manifest.json"

#: Per-process memo of verification results, keyed by
#: ``(resolved root, variant, verification signature)``.  The value is
#: ``(ok, detail)`` where ``detail`` names the first failing file.
#:
#: Both outcomes are cached, not just success.  A truncated weight file would
#: otherwise re-hash ~3 GB on every request that routes to the engine, under the
#: engine lock, before falling back to Kokoro.  The key includes size and
#: ``mtime_ns`` of every recorded file plus the manifest's own mtime, so fixing
#: or replacing a file changes the key and forces a re-check.
_VERIFIED_MODELS: dict[tuple, tuple[bool, str | None]] = {}


def _human_bytes(value: int | None) -> str:
    """Format a byte count for operator-facing messages."""
    if not value:
        return "unknown size"
    if value >= 1024**3:
        return f"{value / 1024**3:.2f} GiB"
    return f"{value / 1024**2:.1f} MiB"


class ChatterboxError(EngineUnavailableError):
    """Base class for expressive-engine failures that trigger a fallback."""


class ChatterboxModelMissing(ChatterboxError):
    """The weights are absent, unverifiable or corrupt."""


class ChatterboxDependencyError(ChatterboxError):
    """The optional backend (torch / chatterbox-tts / resemble-perth) is broken."""


# ---------------------------------------------------------------------------
# Model store: download on demand, licence + SHA recorded, never at build time
# ---------------------------------------------------------------------------


class ChatterboxModelStore:
    """Owns the on-disk Chatterbox weights and their provenance manifest.

    The manifest records the licence, attribution, upstream repository and a
    SHA-256 for every weight file.  ``ensure_available`` refuses to load weights
    it cannot verify, and nothing here downloads anything implicitly: the
    operator runs ``python scripts/download_chatterbox.py`` (or
    ``python -m app.engines.chatterbox --download``) into the models volume.
    """

    def __init__(self, root: str | Path, *, variant: str = "standard") -> None:
        self.root = Path(root)
        self.variant = variant

    @property
    def manifest_path(self) -> Path:
        return self.root / MANIFEST_NAME

    @property
    def required_files(self) -> tuple[str, ...]:
        """The files ``from_local`` reads for this variant, in manifest order."""
        return VARIANT_FILES.get(self.variant, VARIANT_FILES["standard"])

    def weight_files(self) -> list[Path]:
        """The required weight files that are present, sorted by name.

        Deliberately scoped to :attr:`required_files` rather than "every file
        with a weight suffix".  A directory populated by a naive whole-repository
        download also contains the v2/v3 multilingual T3 checkpoints and the
        duplicate ``*.pt`` vocoders; hashing those would cost ~13.87 GB of I/O
        for files the engine never opens.
        """
        if not self.root.is_dir():
            return []
        present: list[Path] = []
        for name in sorted(self.required_files):
            path = self.root / name
            if path.is_file():
                present.append(path)
        return present

    def missing_files(self) -> list[str]:
        """Required files that are absent from the model directory."""
        return [name for name in self.required_files if not (self.root / name).is_file()]

    def is_present(self) -> bool:
        """Whether weights *and* a manifest are present."""
        return bool(self.weight_files()) and self.manifest_path.is_file()

    def read_manifest(self) -> dict[str, Any] | None:
        """The manifest, or ``None`` when absent or unreadable."""
        try:
            payload = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError):
            logger.warning(
                "Chatterbox manifest at %s is unreadable; treating the model as absent.",
                self.manifest_path,
                exc_info=True,
            )
            return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def build_manifest(self, *, repo_id: str | None = None, revision: str | None = None) -> dict:
        """Hash the required weights and build a complete manifest.

        Only :attr:`required_files` are recorded, so the manifest describes what
        the engine actually loads and the SHA-256 pass covers ~3.19 GB for
        ``standard`` instead of the ~13.87 GB in the whole repository.
        """
        files: dict[str, Any] = {}
        for path in self.weight_files():
            files[path.name] = {"sha256": self.sha256(path), "bytes": path.stat().st_size}
        allow_patterns = list(self.required_files)
        return {
            "engine": CHATTERBOX_ENGINE,
            "variant": self.variant,
            "repo_id": repo_id or VARIANT_REPOS.get(self.variant, VARIANT_REPOS["standard"]),
            "revision": revision,
            "license": CHATTERBOX_LICENSE,
            "attribution": CHATTERBOX_ATTRIBUTION,
            "recorded_at": datetime.now(UTC).isoformat(),
            "allow_patterns": allow_patterns,
            "download_bytes": sum(entry["bytes"] for entry in files.values()),
            "repo_total_bytes": VARIANT_REPO_BYTES.get(self.variant),
            "required_total_bytes": VARIANT_REQUIRED_BYTES.get(self.variant),
            "missing": self.missing_files(),
            "files": files,
        }

    def write_manifest(self, manifest: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    def verification_signature(self, manifest: dict[str, Any]) -> tuple:
        """Identity of the exact bytes this verification covers.

        Used to memoise the SHA-256 pass.  ``(name, size, mtime_ns)`` is enough
        to notice a truncated, replaced or rewritten file: any of those changes
        at least the size or the mtime.  A local attacker who can rewrite the
        weights *and* preserve size and mtime can also rewrite the manifest, so
        this does not weaken the threat model the manifest addresses (corruption
        and accidental replacement, not a hostile models volume).
        """
        parts: list[tuple[str, int, int]] = []
        for name in sorted((manifest.get("files") or {})):
            path = self.root / str(name)
            try:
                stat = path.stat()
            except OSError:
                return (str(self.manifest_path), "missing", str(name))
            parts.append((str(name), stat.st_size, stat.st_mtime_ns))
        manifest_stat = self.manifest_path.stat() if self.manifest_path.is_file() else None
        return (
            str(self.manifest_path),
            manifest_stat.st_mtime_ns if manifest_stat else 0,
            tuple(parts),
        )

    def manifest_problems(self, manifest: dict[str, Any]) -> list[str]:
        """Reasons this manifest cannot vouch for *this* store's model.

        Two independent checks, both of which the manager previously missed:

        * **variant identity.**  Both variants default to the same models
          directory.  Downloading turbo over a directory that already holds
          standard rewrites ``manifest.json`` with turbo's file list, while the
          standard files remain on disk.  Without this check the store would
          accept that directory, hash only turbo's entries, and load a standard
          ``t3_cfg.safetensors``/``s3gen.safetensors`` that no recorded hash
          covers.
        * **complete coverage.**  Every name in :attr:`required_files` must have
          a manifest entry.  Hashing "whatever the manifest happens to list" is
          not verification: the files the engine actually reads are the ones that
          have to be covered.
        """
        problems: list[str] = []
        declared = str(manifest.get("variant") or "")
        if declared != self.variant:
            problems.append(
                f"{MANIFEST_NAME} describes variant {declared or '<unset>'!r} but this store "
                f"requires {self.variant!r}; do not share one models directory between variants "
                "(or pass --force to the download command to overwrite it deliberately)"
            )
        files = manifest.get("files")
        if not isinstance(files, dict) or not files:
            problems.append(f"{MANIFEST_NAME} records no weight files")
            return problems
        missing_entries = [name for name in self.required_files if name not in files]
        if missing_entries:
            problems.append(
                f"{MANIFEST_NAME} has no SHA-256 entry for required file(s): "
                + ", ".join(missing_entries)
            )
        return problems

    def _check_files(self, manifest: dict[str, Any]) -> str | None:
        """Hash every required file once; return the first failure or ``None``."""
        files = manifest.get("files") or {}
        for name in self.required_files:
            entry = files.get(name) or {}
            path = self.root / name
            if not path.is_file():
                return f"recorded weight file '{name}' is missing"
            expected = str(entry.get("sha256") or "")
            actual = self.sha256(path)
            if not expected:
                return f"'{name}' has no recorded SHA-256"
            if actual != expected:
                return (
                    f"'{name}' failed SHA-256 verification (expected {expected}, got {actual}); "
                    "the file is corrupt or was replaced"
                )
        return None

    def verification_result(self, manifest: dict[str, Any] | None = None) -> tuple[bool, str | None]:
        """``(ok, detail)`` for one manifest, memoised on the file signature.

        ``detail`` names the first failing file.  Returning it means
        :meth:`ensure_available` can build its message without hashing anything a
        second time.
        """
        manifest = manifest if manifest is not None else self.read_manifest()
        if manifest is None:
            return False, f"{MANIFEST_NAME} is missing or unreadable"
        problems = self.manifest_problems(manifest)
        if problems:
            return False, "; ".join(problems)
        cache_key = (
            str(self.root.resolve()),
            self.variant,
            self.verification_signature(manifest),
        )
        cached = _VERIFIED_MODELS.get(cache_key)
        if cached is not None:
            return cached
        detail = self._check_files(manifest)
        result = (detail is None, detail)
        _VERIFIED_MODELS[cache_key] = result
        return result

    def verify(self, manifest: dict[str, Any] | None = None) -> bool:
        """Whether every required file is present and matches its SHA-256."""
        return self.verification_result(manifest)[0]

    def is_verified(self) -> bool:
        """Whether every recorded file is present and matches its SHA-256."""
        return self.verify()

    def ensure_available(self) -> dict[str, Any]:
        """Return the manifest or raise an actionable :class:`ChatterboxModelMissing`.

        The message names the directory and the exact command to run, because
        "model not found" on its own is a support incident: this engine is
        optional and its weights are never fetched at image build time.

        Also the engine's cheap viability check before the manager evicts
        anything: a missing or mismatched model is detected here, by ``stat`` and
        one memoised hashing pass, instead of after a healthy engine has already
        been unloaded.
        """
        if not self.root.is_dir():
            raise ChatterboxModelMissing(self._missing_message("the model directory does not exist"))
        manifest = self.read_manifest()
        if manifest is None:
            raise ChatterboxModelMissing(
                self._missing_message(
                    f"{MANIFEST_NAME} is missing or unreadable, so the licence, the variant and "
                    "the SHA-256 of each file cannot be verified"
                )
            )
        if str(manifest.get("license") or "") != CHATTERBOX_LICENSE:
            raise ChatterboxModelMissing(
                self._missing_message(
                    f"{MANIFEST_NAME} records licence {manifest.get('license')!r}; "
                    f"only {CHATTERBOX_LICENSE} is permitted (ADR-0001)"
                )
            )
        problems = self.manifest_problems(manifest)
        if problems:
            raise ChatterboxModelMissing(self._missing_message("; ".join(problems)))
        missing = self.missing_files()
        if missing:
            raise ChatterboxModelMissing(
                self._missing_message(
                    "required weight file(s) missing: "
                    + ", ".join(missing)
                    + " (a partial or whole-repository download is not enough; "
                    "run the download command to fetch exactly what from_local reads)"
                )
            )
        ok, detail = self.verification_result(manifest)
        if not ok:
            raise ChatterboxModelMissing(self._missing_message(detail or "weights did not verify"))
        return manifest

    def _missing_message(self, reason: str) -> str:
        command = (
            f"python scripts/download_chatterbox.py --variant {self.variant} "
            f"--models-dir {self.root}"
        )
        return (
            f"Chatterbox model is not usable at {self.root}: {reason}. The weights are "
            "downloaded on demand and never at container build time. Run "
            f"`{command}` on a machine with network access and mount the models volume, "
            "or set CHATTERBOX_ENABLED=false. Requests fall back to Kokoro until the "
            "model is available."
        )


def download_model(
    root: str | Path,
    *,
    variant: str = "standard",
    repo_id: str | None = None,
    revision: str | None = None,
    token: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Download exactly the weights ``from_local`` reads, and record provenance.

    Deliberately explicit and operator-invoked.  It is *not* called by the
    adapter, the Dockerfile or the test suite: a multi-gigabyte download during a
    container build or the first request is exactly what this ticket forbids.

    ``allow_patterns`` restricts the download to :data:`VARIANT_FILES`.  Fetching
    the whole repository would transfer 13.87 GB for ``standard`` (nine redundant
    checkpoints: duplicate ``*.pt`` vocoders, v2/v3 multilingual T3 weights) where
    ``from_local`` reads 3.19 GB.
    """
    store = ChatterboxModelStore(root, variant=variant)
    existing = store.read_manifest()
    if existing is not None and not force:
        declared = str(existing.get("variant") or "")
        if declared and declared != variant:
            raise ChatterboxModelMissing(
                store._missing_message(
                    f"this directory already holds the {declared!r} variant. Downloading "
                    f"{variant!r} over it would rewrite {MANIFEST_NAME} while leaving the other "
                    "variant's checkpoints on disk, so the store would load files no recorded "
                    "hash covers. Use a separate --models-dir per variant, or pass --force to "
                    "overwrite deliberately"
                )
            )
    store.root.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - operator environment
        raise ChatterboxDependencyError(
            "huggingface_hub is required to download Chatterbox weights. Install "
            "requirements-expressive.txt in the speech-expressive profile."
        ) from exc

    target = repo_id or VARIANT_REPOS.get(variant, VARIANT_REPOS["standard"])
    allow_patterns = list(store.required_files)
    logger.info(
        "Downloading %s from %s into %s (%d required file(s); the whole repository is %s)",
        variant,
        target,
        store.root,
        len(allow_patterns),
        _human_bytes(VARIANT_REPO_BYTES.get(variant)),
    )
    snapshot_download(
        repo_id=target,
        revision=revision,
        local_dir=str(store.root),
        token=token,
        allow_patterns=allow_patterns,
    )
    manifest = store.build_manifest(repo_id=target, revision=revision)
    if manifest["missing"]:
        raise ChatterboxModelMissing(
            store._missing_message(
                "the download finished but these required files are absent: "
                + ", ".join(manifest["missing"])
            )
        )
    store.write_manifest(manifest)
    logger.info(
        "Chatterbox weights recorded: %d file(s), %s downloaded (%s in the repository), "
        "licence %s",
        len(manifest["files"]),
        _human_bytes(manifest["download_bytes"]),
        _human_bytes(VARIANT_REPO_BYTES.get(variant)),
        CHATTERBOX_LICENSE,
    )
    return manifest


# ---------------------------------------------------------------------------
# Watermark neutralisation (ADR-0003 decision 4)
# ---------------------------------------------------------------------------


class _NoOpWatermarker:
    """Stand-in for ``PerthImplicitWatermarker`` that leaves audio untouched."""

    def apply_watermark(self, wav, *args, **kwargs):
        return wav

    def __getattr__(self, name: str):  # pragma: no cover - defensive compatibility
        def _noop(*args, **kwargs):
            return None

        return _noop

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return "<NoOpWatermarker (production audio carries no watermark)>"


def neutralise_watermarker(model) -> bool:
    """Replace ``model.watermarker`` with a no-op, returning whether one existed.

    Production audio must carry no watermark (ADR-0003 decision 4).  If the model
    exposes a watermarker we cannot replace, the adapter refuses to render rather
    than shipping a watermarked file: the caller sees a clear
    :class:`ChatterboxError` and the router falls back to Kokoro.
    """
    if not hasattr(model, "watermarker"):
        # No watermarker means there is nothing to apply, which satisfies the
        # requirement.  It does *not* mean the engine is untrustworthy.
        return False
    try:
        model.watermarker = _NoOpWatermarker()
    except Exception as exc:  # pragma: no cover - exotic backend
        raise ChatterboxError(
            "Could not neutralise the Chatterbox watermarker. Production audio must "
            "carry no watermark (ADR-0003 decision 4), so this adapter refuses to "
            "render rather than shipping watermarked audio."
        ) from exc
    return True


# ---------------------------------------------------------------------------
# Backend loading
# ---------------------------------------------------------------------------


def _import_backend(variant: str):
    """Import the optional backend classes lazily.

    ``torch`` and ``chatterbox-tts`` are optional dependencies of the
    ``speech-expressive`` profile.  Importing this module must never require
    them; importing them happens only here, on first load.
    """
    module_name, class_name = VARIANT_BACKENDS.get(
        variant, VARIANT_BACKENDS["standard"]
    )
    try:
        module = __import__(module_name, fromlist=[class_name])
        import torch  # noqa: F401  # presence check; the backend needs it at runtime
    except ImportError as exc:
        raise ChatterboxDependencyError(
            "Chatterbox is enabled but its optional backend is not installed "
            f"({exc}). Use the speech-expressive profile (requirements-expressive.txt) "
            "which pins chatterbox-tts==0.1.7, torch (CPU) and setuptools<81. The "
            "default speech-core image deliberately does not include torch."
        ) from exc
    return getattr(module, class_name)


def _instantiate_backend(backend, store: ChatterboxModelStore, device: str):
    """Construct the backend model from the verified local weights.

    The dependency trap this handles explicitly: ``resemble-perth`` calls
    ``pkg_resources``, which setuptools 81+ removed.  It swallows the ImportError
    and leaves ``PerthImplicitWatermarker = None``, which fails later as
    ``TypeError: 'NoneType' object is not callable`` inside the constructor.  That
    must degrade to a clear, actionable error and a Kokoro fallback, not an
    unhandled crash.
    """
    try:
        if callable(getattr(backend, "from_local", None)):
            return backend.from_local(str(store.root), device)
        signature = inspect.signature(backend.from_pretrained)
        kwargs: dict[str, Any] = {}
        if "device" in signature.parameters:
            kwargs["device"] = device
        if "cache_dir" in signature.parameters:
            kwargs["cache_dir"] = str(store.root)
        return backend.from_pretrained(**kwargs)
    except TypeError as exc:
        message = str(exc)
        if "NoneType" in message and "callable" in message:
            raise ChatterboxDependencyError(
                "Chatterbox failed to initialise because resemble-perth could not create "
                "its watermarker (PerthImplicitWatermarker is None). Install the pinned "
                "extras: `pip install -r requirements-expressive.txt` (setuptools<81). "
                "The adapter neutralises the watermarker after construction; this failure "
                "happens before that point."
            ) from exc
        raise ChatterboxDependencyError(
            f"Chatterbox could not be constructed from {store.root}: {exc}"
        ) from exc
    except Exception as exc:
        raise ChatterboxDependencyError(
            f"Chatterbox could not be constructed from {store.root}: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Text and style handling
# ---------------------------------------------------------------------------

_SSML_BREAK = re.compile(r"<\s*break[^>]*>", re.IGNORECASE)
_SSML_TAG = re.compile(r"</?[a-zA-Z][^>]*>")
_PHONEME_TAG = re.compile(
    r"<phoneme\s+ph=\"([^\"]*)\"\s*>(.*?)</phoneme>", re.IGNORECASE | re.DOTALL
)
#: The platform's inline phonetic markup: ``[word|/ipa/]`` or ``[word|respelling]``.
#: Distinguished from a paralinguistic tag by the pipe.
_INLINE_PHONETIC = re.compile(r"\[([^\|\]]+)\|([^\]]*)\]")


def sanitise_text(text: str) -> tuple[str, list[str]]:
    """Strip or translate markup Chatterbox does not understand.

    Chatterbox takes plain text plus its own paralinguistic tags.  It does not
    consume SSML or this service's inline phonetic markup:

    * ``<phoneme ph="...">word</phoneme>`` and ``[word|/ipa/]`` are *phonetic*
      annotations — the engine needs the grapheme, not the IPA, so the word is
      kept and the annotation is dropped (and reported).
    * ``<break/>`` becomes a punctuation pause; other XML/SSML tags are removed.
    * ``[laugh]``-style paralinguistic tags are Chatterbox's own vocabulary and
      are preserved.  A bracketed pair containing ``|`` is phonetic markup, not a
      tag, so it is never passed through verbatim.

    Every substitution is reported, so a caller is never told the annotation was
    honoured when it was translated.
    """
    notes: list[str] = []

    if _PHONEME_TAG.search(text):
        text = _PHONEME_TAG.sub(lambda match: match.group(2), text)
        notes.append(
            "replaced <phoneme> markup with its grapheme text; Chatterbox cannot consume IPA"
        )
    if _INLINE_PHONETIC.search(text):
        text = _INLINE_PHONETIC.sub(lambda match: match.group(1), text)
        notes.append(
            "replaced [word|pronunciation] phonetic markup with the plain word; "
            "Chatterbox cannot consume IPA"
        )
    if _SSML_BREAK.search(text):
        text = _SSML_BREAK.sub(", ", text)
        notes.append("translated <break> markup to punctuation pauses")
    if _SSML_TAG.search(text):
        text = _SSML_TAG.sub("", text)
        notes.append("stripped markup Chatterbox does not understand")
    return re.sub(r"\s{2,}", " ", text).strip(), notes


def exaggeration_for(style: str | None, intensity: float | None) -> float | None:
    """Map a style + intensity onto a bounded ``exaggeration`` value.

    ``None`` means the style is not expressible.  The result is always inside
    ``[EXAGGERATION_MIN, EXAGGERATION_MAX]``: the bounds are a hard product
    constraint, not a suggestion.
    """
    if style is None or style == "neutral":
        return None
    base = STYLE_EXAGGERATION.get(style)
    if base is None:
        return None
    if intensity is not None:
        base += (float(intensity) - 0.5) * 2 * INTENSITY_SPAN
    return round(max(EXAGGERATION_MIN, min(EXAGGERATION_MAX, base)), 3)


#: Where a hard split may fall, in order of preference.  Splitting mid-word
#: renders ``'exagge' | 'ration'`` as two utterances with an audible gap, which
#: unpunctuated transcript text hits constantly.
_SPLIT_POINTS = (" ", ",", ";", ":")


def _hard_split(piece: str, limit: int) -> tuple[str, str]:
    """Split ``piece`` at the last word boundary before ``limit``.

    Returns ``(head, tail)`` with ``head`` non-empty.  Falls back to a raw cut
    when the piece has no boundary to split on (a single very long token, a
    language without spaces).
    """
    window = piece[:limit]
    best = max((window.rfind(mark) for mark in _SPLIT_POINTS), default=-1)
    if best > 0:
        # Keep the separator with the head so the pause is not lost, and drop the
        # space so the next chunk does not start with whitespace.
        head = piece[: best + 1]
        tail = piece[best + 1 :].lstrip()
        return head.rstrip(), tail
    return piece[:limit], piece[limit:]


def _chunk_text(text: str, *, limit: int = MAX_CHARS_PER_CALL) -> list[str]:
    """Split text into chunks of at most ``limit`` characters.

    Boundaries are preferred in this order: sentence end, then the last space or
    comma before the limit, then — only if there is no boundary at all — a raw
    cut.  Every chunk is guaranteed to be at most ``limit`` characters.
    """
    text = text.strip()
    if len(text) <= limit:
        return [text] if text else []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip()
        if len(candidate) <= limit:
            current = candidate
            continue
        if current:
            chunks.append(current)
        while len(sentence) > limit:
            head, sentence = _hard_split(sentence, limit)
            chunks.append(head)
        current = sentence
    if current:
        chunks.append(current)
    return [chunk for chunk in chunks if chunk]


def _record_variant(record) -> str:
    """The variant a registry record belongs to (pack voices and clones)."""
    return str(record.style_spec.get("variant") or "standard").lower()


def _to_numpy(wav) -> np.ndarray:
    """Convert a torch tensor (or array-like) to a flat float32 numpy array."""
    value = wav
    for attribute in ("detach", "cpu", "float", "numpy"):
        method = getattr(value, attribute, None)
        if callable(method):
            value = method()
    return np.asarray(value, dtype=np.float32).reshape(-1)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class ChatterboxEngine:
    """Adapter for Chatterbox (MIT), the optional expressive engine.

    Registered only when ``CHATTERBOX_ENABLED`` is true.  It loads lazily on the
    first request that routes to it, so enabling the flag never slows startup and
    a missing model degrades to Kokoro instead of failing the process.
    """

    id = CHATTERBOX_ENGINE
    name = "Chatterbox (expressive, MIT)"
    license = CHATTERBOX_LICENSE
    attribution = CHATTERBOX_ATTRIBUTION
    max_chars = 10000

    #: ``EngineManager`` skips lazy engines during eager startup loading.
    lazy_load = True

    def __init__(
        self,
        *,
        model_store: ChatterboxModelStore | None = None,
        variant: str | None = None,
        device: str | None = None,
    ) -> None:
        self.variant = (variant or settings.chatterbox_variant or "standard").lower()
        self.device = device or settings.chatterbox_device or "cpu"
        self._store = model_store or ChatterboxModelStore(
            settings.chatterbox_model_dir_resolved, variant=self.variant
        )
        self._model = None
        self._ready = False
        self._sample_rate = CHATTERBOX_SAMPLE_RATE
        self._watermark_neutralised = False
        #: Resolved from :data:`VARIANT_SUPPORTS_EXAGGERATION` rather than from
        #: the backend signature, so the advertised styles are correct before the
        #: model is loaded and cannot claim a control the variant ignores.
        self._supports_exaggeration = VARIANT_SUPPORTS_EXAGGERATION.get(self.variant, False)
        #: Guards every use of the shared model object.  Chatterbox keeps the
        #: speaker conditioning on the model instance (``model.conds``), not in a
        #: per-call argument, so a render that changes it silently changes the
        #: voice of every other render using the same object.  A clone sets it to
        #: the subject; a pack-voice render reads whatever happens to be there.
        #: Both must therefore hold this lock for the whole model call, and a
        #: clone must always restore the previous conditioning.  An RLock because
        #: the clone backend re-enters it from inside a render.
        self._model_guard = threading.RLock()

    @property
    def model_guard(self) -> threading.RLock:
        """The lock guarding the shared model's mutable speaker conditioning."""
        return self._model_guard

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value) -> None:
        self._model = value

    @property
    def is_ready(self) -> bool:
        return self._ready

    def set_ready(self, ready: bool) -> None:
        self._ready = bool(ready)

    @property
    def model_store(self) -> ChatterboxModelStore:
        return self._store

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def supports_exaggeration(self) -> bool:
        """Whether this variant can actually honour ``exaggeration``."""
        return bool(self._supports_exaggeration)

    @property
    def style_caveat(self) -> str | None:
        """Why this variant cannot express tone, or ``None`` when it can."""
        if self.supports_exaggeration:
            return None
        return VARIANT_STYLE_CAVEAT.get(
            self.variant,
            f"Chatterbox variant '{self.variant}' has no exaggeration control",
        )

    async def initialize(self) -> None:
        """Load the model off the event loop (called on first use)."""
        import asyncio

        if self._ready and self._model is not None:
            return
        await asyncio.to_thread(self.load)

    def preflight(self) -> None:
        """Cheap viability check, run by the manager before it evicts anything.

        Only the model directory, the manifest and one memoised hashing pass; no
        torch import and no model construction.  Raising here means a request for
        an unservable engine fails *before* a healthy engine is unloaded, so the
        fallback does not have to reload Kokoro from cold.
        """
        self._store.ensure_available()

    def load(self) -> None:
        """Load the backend, with every failure mapped to a clear fallback error.

        Deliberately does **not** consult ``CHATTERBOX_ENABLED``.  Registration is
        the gate: ``TTSService`` only registers this engine when the flag is on,
        and an explicit ``TTSService(expressive=True)`` must remain loadable even
        on a process whose environment says otherwise.  A second check here made
        such an engine advertise itself in ``/capabilities`` and get style-routed
        while being unable to ever load.
        """
        # Weights first: on a deployment without the model this fails in
        # milliseconds with an actionable message, before importing torch.
        self._store.ensure_available()

        backend = _import_backend(self.variant)
        model = _instantiate_backend(backend, self._store, self.device)

        self._watermark_neutralised = neutralise_watermarker(model)
        if not self._watermark_neutralised:
            logger.warning(
                "Chatterbox backend exposes no watermarker attribute; there is nothing to "
                "neutralise, so production audio is clean by construction. chatterbox-tts "
                "0.1.7 does expose one, so verify the backend if this is unexpected."
            )

        # The table is the source of truth; the signature check can only
        # *downgrade* the capability, never invent one.  It guards against an
        # upstream release that drops the parameter while we still advertise it.
        if self._supports_exaggeration and not self._inspect_exaggeration(model):
            self._supports_exaggeration = False
            logger.warning(
                "Installed Chatterbox %s does not expose an exaggeration parameter; styles "
                "will be reported as ignored.",
                self.variant,
            )

        sample_rate = getattr(model, "sr", None)
        if isinstance(sample_rate, (int, float)) and sample_rate > 0:
            self._sample_rate = int(sample_rate)

        self._model = model
        self._ready = True
        logger.info(
            "Chatterbox %s loaded from %s (%d Hz, exaggeration=%s, watermark neutralised=%s)",
            self.variant,
            self._store.root,
            self._sample_rate,
            self._supports_exaggeration,
            self._watermark_neutralised,
        )

    @staticmethod
    def _inspect_exaggeration(model) -> bool:
        """Guard only: does the loaded backend *have* an exaggeration parameter?

        Never used to decide whether a variant is expressive — turbo has the
        parameter and ignores it.  See :data:`VARIANT_SUPPORTS_EXAGGERATION`.
        """
        generate = getattr(model, "generate", None)
        if not callable(generate):
            return False
        try:
            return "exaggeration" in inspect.signature(generate).parameters
        except (TypeError, ValueError):  # pragma: no cover - C-level callables
            return True

    def unload(self) -> None:
        """Release the model so another engine can take its memory."""
        self._model = None
        self._ready = False
        # Capability is a property of the variant, not of the loaded model.
        self._supports_exaggeration = VARIANT_SUPPORTS_EXAGGERATION.get(self.variant, False)
        self._watermark_neutralised = False

    # ------------------------------------------------------------------
    # Voices
    # ------------------------------------------------------------------

    def _records(self) -> list:
        """Pack voices for this variant plus **active** clones.

        Clones are contributed by the clone store through the registry hook and
        are listed only while active.  A disabled or expired clone disappears
        from the catalogue immediately, without a restart.
        """
        records = [
            record
            for record in voice_registry.VOICE_REGISTRY.for_engine(self.id, enabled_only=True)
            if _record_variant(record) == self.variant
        ]
        records.extend(
            record
            for record in voice_registry.reference_records(self.id)
            if _record_variant(record) == self.variant
        )
        return records

    @property
    def engine_voice_ids(self) -> list[str] | None:
        """Voice ids this deployment actually exposes, or ``None`` when unloaded."""
        if self._model is None:
            return None
        return [record.id for record in self._records()]

    def available_voice_ids(self) -> list[str]:
        return [record.id for record in self._records()]

    def voices(self) -> list[VoiceDescriptor]:
        return build_catalogue(self.available_voice_ids(), engine=self.id)

    def can_render_voice(self, voice_id: str) -> bool:
        """Whether *this deployed variant* can render ``voice_id``.

        The service uses this instead of comparing engine ids, so pinning the
        other Chatterbox variant's voice degrades to the deployed variant's own
        default voice (reported in ``ignored``) rather than returning a 422 that
        the flag-off path would not produce.
        """
        record = voice_registry.get_record(voice_id)
        if record is None or record.engine != self.id:
            return False
        declared = str(record.style_spec.get("variant") or "standard").lower()
        return declared == self.variant

    def get_voice(self, voice_id: str) -> VoiceDescriptor | None:
        if voice_id not in self.available_voice_ids():
            return None
        return build_catalogue([voice_id], engine=self.id)[0]

    def resolve_voice(self, voice_id: str | None) -> tuple[str, str]:
        """Resolve a caller-facing id to ``(engine voice key, locale)``.

        Chatterbox has no named voice pack: ``default`` means the model's own
        voice.  A ``reference`` record is a consent-gated clone (SPEECH-006): it
        resolves only while the clone store says it is active, and rendering it
        requires the stored embedding.  A clone that is not active is refused
        rather than silently rendered with the model's default speaker.

        A named voice is resolved through the single-id registry lookup only.
        :meth:`_records` lists every active clone, and each listing reads and
        verifies every clone's consent record, so it is consulted only when no
        voice was named and the default has to be chosen.
        """
        if voice_id is None:
            records = self._records()
            if not records:
                raise UnknownVoiceError(
                    "No Chatterbox voice is registered for variant "
                    f"'{self.variant}'. GET /api/tts/voices lists the available voices."
                )
            record = records[0]
        else:
            record = voice_registry.get_record(voice_id)
            if record is None or record.engine != self.id:
                raise UnknownVoiceError(
                    f"Voice '{voice_id}' is not a Chatterbox voice. "
                    "GET /api/tts/voices lists the voices available here."
                )
            if str(record.style_spec.get("variant") or "standard").lower() != self.variant:
                raise UnknownVoiceError(
                    f"Voice '{voice_id}' belongs to Chatterbox variant "
                    f"'{record.style_spec.get('variant')}', not the deployed '{self.variant}'."
                )
        if record.style_type == "reference":
            # Fail closed: a reference voice is renderable only while the clone
            # store says it is active.  With no store in this process the default
            # is inactive, never "assume active".
            if not voice_registry.reference_voice_is_active(record.id):
                raise UnknownVoiceError(
                    f"Voice '{record.id}' is a cloned reference voice and is not active "
                    "(disabled, taken down, expired, or unknown to this instance)."
                )
        return record.engine_voice_id, record.locale

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------

    def capabilities(self) -> EngineCapabilities:
        expressive = list(CHATTERBOX_STYLES) if self.supports_exaggeration else ["neutral"]
        others = [language for language in all_canonical_languages() if language not in CHATTERBOX_LANGUAGES]
        return EngineCapabilities(
            engine=self.id,
            languages=list(CHATTERBOX_LANGUAGES),
            unsupported_languages=others,
            language_support=[
                {
                    "language": language,
                    "locale": engine_locale(language),
                    "supported": language in CHATTERBOX_LANGUAGES,
                    "quality": "unverified" if language in CHATTERBOX_LANGUAGES else "unsupported",
                    "caveat": None
                    if language in CHATTERBOX_LANGUAGES
                    else "Chatterbox 0.1.7 is English-only",
                    "reason": "English-only model; round-trip quality was not measured for "
                    "SPEECH-005"
                    if language in CHATTERBOX_LANGUAGES
                    else "no English text front end for this language",
                    "measured": False,
                }
                for language in all_canonical_languages()
            ],
            native_rate=False,
            native_pitch=False,
            native_volume=False,
            styles=expressive,
            # SPEECH-006: the adapter exposes a clone backend that renders a stored
            # speaker embedding.  Whether THIS deployment may create clones is a
            # separate, fail-closed configuration question reported by /health.
            supports_cloning=True,
            supports_streaming=False,
            supports_ssml=False,
            sample_rates=[CHATTERBOX_SAMPLE_RATE],
            max_chars=self.max_chars,
            gpu_required=False,
            attribution=CHATTERBOX_ATTRIBUTION,
            variant=self.variant,
            compute_cost=dict(COMPUTE_COST_ADVISORY),
            capability_notes=[
                *([self.style_caveat] if self.style_caveat else []),
                _CLONING_NOTE,
                _WATERMARK_NOTE,
                _EXAGGERATION_NOTE if self.supports_exaggeration else _NO_EXAGGERATION_NOTE,
                _STREAMING_NOTE,
            ],
            supports_paralinguistic_tags=list(PARALINGUISTIC_TAGS),
        )

    def descriptor(self) -> dict[str, Any]:
        capabilities = self.capabilities()
        default_voice = self.available_voice_ids()
        return {
            "id": capabilities.engine,
            "engine": capabilities.engine,
            "name": self.name,
            "available": self.is_ready,
            "license": self.license,
            "attribution": self.attribution,
            "variant": self.variant,
            "languages": capabilities.languages,
            "unsupported_languages": capabilities.unsupported_languages,
            "language_support": capabilities.language_support,
            "voice_count": len(default_voice),
            "default_voice": default_voice[0] if default_voice else None,
            "native_rate": capabilities.native_rate,
            "native_pitch": capabilities.native_pitch,
            "native_volume": capabilities.native_volume,
            "styles": capabilities.styles,
            "supports_exaggeration": self.supports_exaggeration,
            "supports_cloning": capabilities.supports_cloning,
            "supports_streaming": capabilities.supports_streaming,
            "supports_ssml": capabilities.supports_ssml,
            "supports_paralinguistic_tags": capabilities.supports_paralinguistic_tags,
            "sample_rates": capabilities.sample_rates,
            "max_chars": capabilities.max_chars,
            "gpu_required": capabilities.gpu_required,
            "compute_cost": capabilities.compute_cost,
            "capability_notes": capabilities.capability_notes,
            "exaggeration_range": [EXAGGERATION_MIN, EXAGGERATION_MAX],
            "watermark": "neutralised" if self._watermark_neutralised else "not-loaded",
        }

    # ------------------------------------------------------------------
    # Direction lowering
    # ------------------------------------------------------------------

    def lower(self, direction) -> DirectionPlan:
        """Lower a provider-neutral direction onto Chatterbox's controls.

        Honest by construction: pacing, pitch and volume are reported as ignored
        rather than approximated, and ``whisper`` is refused because there is no
        breathiness control.  ``rate`` is never stacked on ``exaggeration``.
        """
        applied: dict[str, Any] = {}
        ignored: list[str] = []
        warnings: list[str] = []

        fields = set(direction.model_fields_set)
        prosody = direction.prosody
        prosody_fields = set(prosody.model_fields_set)
        style = direction.style if "style" in fields else None
        intensity = direction.intensity if "intensity" in fields else None

        if style not in (None, "neutral"):
            exaggeration = exaggeration_for(style, intensity)
            if exaggeration is None:
                ignored.append("style")
                warnings.append(
                    f"Chatterbox cannot express style '{style}': it has no control for it "
                    "(no whisper/breathiness or valence control); the style was not applied"
                )
                if intensity is not None:
                    ignored.append("intensity")
            elif not self.supports_exaggeration:
                ignored.append("style")
                if intensity is not None:
                    ignored.append("intensity")
                warnings.append(
                    f"Chatterbox variant '{self.variant}' cannot express style '{style}': "
                    f"{self.style_caveat}. The style was not applied"
                )
            else:
                applied["style"] = style
                applied["exaggeration"] = exaggeration
                if intensity is not None:
                    applied["intensity"] = intensity
                if style != "neutral":
                    warnings.append(
                        f"style '{style}' is rendered as exaggeration={exaggeration}. "
                        "exaggeration controls expressiveness intensity, not valence; the "
                        "engine cannot target a specific emotion."
                    )

        # ``intensity`` modifies a style; without an applied style it cannot do
        # anything.  Report it instead of dropping it silently — the same rule
        # ``lower_direction`` applies for every other engine.
        if intensity is not None and "style" not in applied:
            if "intensity" not in ignored:
                ignored.append("intensity")
            if style in (None, "neutral"):
                warnings.append(
                    "intensity is only meaningful with a style Chatterbox can express; "
                    "it was not applied"
                )

        if "rate" in prosody_fields and prosody.rate is not None:
            ignored.append("rate")
            warnings.append(
                "Chatterbox has no rate control and prosody.rate is not applied: "
                "exaggeration already co-varies with pace (the same sentence is roughly "
                "30% shorter at 1.0 than at 0.0), so stacking a rate multiplier would "
                "compound an unmeasured effect"
            )

        if "pitch" in prosody_fields and prosody.pitch not in (None, 0.0):
            ignored.append("pitch")
            warnings.append(
                f"Chatterbox has no pitch control; pitch {prosody.pitch:+g} semitones was "
                "not applied"
            )

        if "volume" in prosody_fields and prosody.volume not in (None, 0.0):
            ignored.append("volume")
            warnings.append(
                f"Chatterbox has no volume control; volume {prosody.volume:+g} dB was not "
                "applied"
            )

        if "pauses" in prosody_fields:
            if prosody.pauses == "natural":
                applied["pauses"] = "natural"
            else:
                ignored.append("pauses")
                warnings.append(
                    f"Chatterbox only pauses at punctuation; '{prosody.pauses}' pause "
                    "shaping requires silence insertion, which is not enabled"
                )

        return DirectionPlan(engine=self.id, applied=applied, ignored=ignored, warnings=warnings)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @staticmethod
    def _report_request_controls(req: EngineRequest, plan: DirectionPlan) -> DirectionPlan:
        """Report per-request controls this engine cannot honour.

        ``lower()`` only sees the provider-neutral :class:`Direction`.  These two
        controls travel on the request itself, so they are reported here rather
        than dropped silently:

        * ``custom_lexicon`` -- Chatterbox has no pronunciation dictionary.  The
          Kokoro path applies it through the G2P converter; here it cannot be
          honoured at all.
        * ``phonetic_preprocess`` -- the same pipeline pre-phonemises text for
          Kokoro.  Chatterbox consumes graphemes, so the switch has no effect.
        """
        ignored = list(plan.ignored)
        warnings = list(plan.warnings)
        if req.custom_lexicon:
            if "custom_lexicon" not in ignored:
                ignored.append("custom_lexicon")
            warnings.append(
                "custom_lexicon was not applied: Chatterbox has no pronunciation dictionary "
                f"and ignores the {len(req.custom_lexicon)} supplied override(s)"
            )
        # Only an explicit *request* for pre-phonemisation is reportable.  A
        # caller who sends ``false`` is asking for the default behaviour, so
        # reporting it as an ignored control would be misleading noise.
        if req.phonetic_preprocess:
            if "phonetic_preprocess" not in ignored:
                ignored.append("phonetic_preprocess")
            warnings.append(
                "phonetic_preprocess was not applied: Chatterbox consumes graphemes directly, "
                "so pre-phonemisation has no effect"
            )
        if ignored == plan.ignored and warnings == plan.warnings:
            return plan
        return plan.model_copy(update={"ignored": ignored, "warnings": warnings})

    def synthesize(self, req: EngineRequest) -> AudioResult:
        if not self._ready or self._model is None:
            raise ChatterboxError(
                f"Chatterbox engine is not loaded ({self._store.root}). "
                "Requests fall back to Kokoro while the model is unavailable."
            )
        if req.is_phonemes:
            raise ChatterboxError(
                "Chatterbox cannot render caller-supplied phonemes; it tokenises plain "
                "text. The request was routed to an engine that can."
            )

        plan = self.lower(req.direction)
        plan = self._report_request_controls(req, plan)
        _voice_key, locale = self.resolve_voice(req.voice)
        language = canonical_language(locale)
        if language is not None and language not in CHATTERBOX_LANGUAGES:
            raise ChatterboxError(
                f"Chatterbox 0.1.7 is English-only; it cannot render '{language}'."
            )

        text, text_notes = sanitise_text(req.text)
        plan = plan.model_copy(update={"warnings": [*plan.warnings, *text_notes]})
        if not text:
            raise ChatterboxError("No speakable text remained after stripping markup.")

        exaggeration = plan.applied.get("exaggeration", EXAGGERATION_NEUTRAL)
        # The whole model call is serialised.  ``model.conds`` is shared mutable
        # state: a pack-voice render reads whatever conditioning is currently set,
        # and a clone render sets it to the subject.  If the two overlapped, one
        # would speak in the other's voice.  The engine limiter allows one
        # Chatterbox render at a time, but ``synthesize`` is also reached
        # synchronously (stream, sentence spans) without taking a slot, so the
        # guarantee belongs here.
        with self._model_guard:
            # Defensive re-application: a backend may rebuild its watermarker, and
            # the requirement is that *no* returned audio carries one.
            if hasattr(self._model, "watermarker"):
                neutralise_watermarker(self._model)

            if req.clone_embedding is not None:
                # A clone is re-derived from its stored embedding; the reference
                # clip is gone by default, so there is deliberately no path that
                # reads one.
                samples = self._render_clone(
                    text, req.clone_embedding, exaggeration=float(exaggeration)
                )
            else:
                samples = self._render(text, exaggeration=float(exaggeration))
        return AudioResult(
            audio=self.samples_to_wav(samples, self._sample_rate),
            duration=len(samples) / self._sample_rate,
            sample_rate=self._sample_rate,
            engine=self.id,
            voice=req.voice,
            applied=plan.applied,
            ignored=plan.ignored,
            warnings=plan.warnings,
        )

    def synthesize_span(self, req: EngineRequest, text: str) -> AudioChunk:
        """Render one span; Chatterbox has no incremental streaming API."""
        result = self.synthesize(req.model_copy(update={"text": text}))
        return AudioChunk(
            audio=result.audio, duration=result.duration, sample_rate=result.sample_rate
        )

    def stream(self, req: EngineRequest) -> Iterator[AudioChunk]:
        """Yield the whole render as one chunk (declared ``supports_streaming=False``)."""
        result = self.synthesize(req)
        yield AudioChunk(
            audio=result.audio, duration=result.duration, sample_rate=result.sample_rate
        )

    def _render(self, text: str, *, exaggeration: float) -> np.ndarray:
        model = self._model
        if model is None:  # pragma: no cover - guarded by synthesize
            raise ChatterboxError("Chatterbox model is not loaded")

        chunks = _chunk_text(text)
        if not chunks:
            raise ChatterboxError("No speakable text remained after chunking.")

        rendered: list[np.ndarray] = []
        gap = np.zeros(int(self._sample_rate * CHUNK_GAP_SECONDS), dtype=np.float32)
        for index, chunk in enumerate(chunks):
            wav = self._generate(model, chunk, exaggeration)
            if index:
                rendered.append(gap)
            rendered.append(wav)
        return np.concatenate(rendered) if len(rendered) > 1 else rendered[0]

    def clone_backend(self) -> "ChatterboxCloneBackend | None":
        """The embedding/render bridge for cloning, or ``None`` when not loaded.

        Exposed rather than imported by the clone service so that module never
        touches torch or this adapter directly.
        """
        if self._model is None:
            return None
        return ChatterboxCloneBackend(self)

    def _render_clone(
        self, text: str, embedding: bytes, *, exaggeration: float
    ) -> np.ndarray:
        """Render from a stored embedding; the reference clip is not consulted."""
        backend = self.clone_backend()
        if backend is None:  # pragma: no cover - guarded by synthesize
            raise ChatterboxError("Chatterbox model is not loaded")
        chunks = _chunk_text(text)
        if not chunks:
            raise ChatterboxError("No speakable text remained after chunking.")
        rendered: list[np.ndarray] = []
        gap = np.zeros(int(self._sample_rate * CHUNK_GAP_SECONDS), dtype=np.float32)
        for index, chunk in enumerate(chunks):
            wav, _rate = backend.render(chunk, embedding, exaggeration=exaggeration)
            if wav.size == 0:
                raise ChatterboxError("Chatterbox returned empty audio for a cloned voice.")
            if index:
                rendered.append(gap)
            rendered.append(wav)
        return np.concatenate(rendered) if len(rendered) > 1 else rendered[0]

    def _generate(self, model, text: str, exaggeration: float) -> np.ndarray:
        kwargs: dict[str, Any] = {}
        if self.supports_exaggeration:
            kwargs["exaggeration"] = float(exaggeration)
        wav = model.generate(text, **kwargs)
        samples = _to_numpy(wav)
        if samples.size == 0:
            raise ChatterboxError("Chatterbox returned empty audio.")
        return samples

    @staticmethod
    def samples_to_wav(samples: np.ndarray, sample_rate: int) -> bytes:
        import io

        buffer = io.BytesIO()
        sf.write(buffer, samples, sample_rate, format="WAV", subtype="PCM_16")
        buffer.seek(0)
        return buffer.read()

    @staticmethod
    def split_sentences(text: str) -> list[str]:
        return [part for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


class ChatterboxCloneBackend:
    """Speaker-embedding extraction and rendering for a loaded Chatterbox model.

    The reference clip is deleted as soon as extraction completes (ADR-0003 #3),
    so both operations are expressed in terms of one opaque artefact: the
    model's speaker conditioning (``model.conds``), serialised with ``torch``.
    A clone is therefore *re-derived from the embedding only*; there is no path
    here that reads a reference clip at synthesis time.

    **The conditioning is borrowed, never taken.**  ``model.conds`` lives on the
    single shared model object, so leaving it set to the subject would make every
    later render in that process speak in the subject's voice — the exact harm
    ADR-0003 exists to prevent, pointed at the other voices.  Both methods
    therefore run under :attr:`ChatterboxEngine.model_guard` and restore the
    previous conditioning in ``finally``.

    Requires the ``speech-expressive`` profile (torch + chatterbox-tts).  The
    governance layer and its tests run without either: the backend is resolved
    at request time, so an unloaded engine makes cloning unavailable rather than
    breaking the service.
    """

    def __init__(self, engine: ChatterboxEngine) -> None:
        self._engine = engine

    def _model(self):
        model = self._engine.model
        if model is None:  # pragma: no cover - guarded by clone_backend()
            raise ChatterboxError("Chatterbox model is not loaded")
        return model

    @staticmethod
    def _torch():
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - expressive profile only
            raise ChatterboxDependencyError(
                "Voice cloning needs torch and chatterbox-tts (requirements-expressive.txt)."
            ) from exc
        return torch

    def extract_embedding(self, audio: bytes) -> bytes:
        """Build the speaker conditioning blob from a decodable reference clip.

        The clip is written to a **private** directory (mode 0700, created by
        ``TemporaryDirectory``) only because the upstream API takes a path, and
        the whole directory is removed in ``finally``.  The caller's own
        encrypted copy is deleted separately in the clone store.

        The conditioning is restored before returning: ``prepare_conditionals``
        leaves ``model.conds`` pointing at the subject, which would otherwise
        become the default speaker for every later pack-voice render.
        """
        # Fail with the actionable dependency error before touching the model.
        self._torch()
        model = self._model()
        data, sample_rate = sf.read(io.BytesIO(audio), dtype="float32", always_2d=True)
        mono = data.mean(axis=1) if data.ndim > 1 else data
        with tempfile.TemporaryDirectory(prefix="speech-clone-") as private_dir:
            clip_path = Path(private_dir) / "reference.wav"
            sf.write(str(clip_path), mono, int(sample_rate), format="WAV", subtype="PCM_16")
            prepare = getattr(model, "prepare_conditionals", None)
            if not callable(prepare):  # pragma: no cover - depends on upstream API
                raise ChatterboxError(
                    "The installed Chatterbox build cannot derive speaker conditionals "
                    "from a reference clip; cloning is unavailable on this build."
                )
            with self._engine.model_guard:
                previous = getattr(model, "conds", None)
                try:
                    prepare(str(clip_path), exaggeration=EXAGGERATION_NEUTRAL)
                    if getattr(model, "conds", None) is None:  # pragma: no cover - upstream API
                        raise ChatterboxError("Chatterbox produced no speaker conditioning.")
                    return self._serialise_conds(model.conds)
                finally:
                    # The subject's conditioning is a borrowed value, not this
                    # model's state.  Restore it even when extraction failed.
                    model.conds = previous

    # -- embedding serialisation -----------------------------------------
    #
    # The embedding is the conditioning container's *tensors*, written by the
    # container's own ``save`` and read back by its own ``load``.  In
    # chatterbox-tts that pair is ``Conditionals.save`` / ``Conditionals.load``:
    # a plain dict of tensors, reloaded with ``torch.load(weights_only=True)`` —
    # the same path the library uses for its shipped ``conds.pt``.  The first
    # SPEECH-006 release pickled the whole container and reloaded it with
    # ``weights_only=False``, which executes arbitrary pickle opcodes: anyone able
    # to write an embedding file (and holding the reference key) got code
    # execution on the next render.  No path here unpickles with
    # ``weights_only=False``.

    @staticmethod
    def _serialise_conds(conds) -> bytes:
        save = getattr(conds, "save", None)
        if not callable(save):
            raise ChatterboxError(
                "The installed Chatterbox build's speaker conditioning has no save(); "
                "it cannot be persisted without pickling arbitrary objects, so cloning "
                "is unavailable on this build."
            )
        buffer = io.BytesIO()
        save(buffer)
        return CONDS_EMBEDDING_MAGIC + buffer.getvalue()

    def _conds_class(self, model):
        """The conditioning container class for the loaded variant."""
        current = getattr(model, "conds", None)
        if current is not None and callable(getattr(type(current), "load", None)):
            return type(current)
        module_name, _class_name = VARIANT_BACKENDS.get(
            self._engine.variant, VARIANT_BACKENDS["standard"]
        )
        try:
            module = __import__(module_name, fromlist=["Conditionals"])
            return module.Conditionals
        except (ImportError, AttributeError) as exc:  # pragma: no cover - upstream API
            raise ChatterboxDependencyError(
                f"{module_name}.Conditionals is unavailable, so a stored clone embedding "
                "cannot be loaded."
            ) from exc

    def _deserialise_conds(self, torch, model, embedding: bytes):
        """Rebuild a conditioning container without unrestricted unpickling."""
        if embedding.startswith(CONDS_EMBEDDING_MAGIC):
            conds_class = self._conds_class(model)
            loaded = conds_class.load(
                io.BytesIO(embedding[len(CONDS_EMBEDDING_MAGIC) :]), map_location="cpu"
            )
        else:
            loaded = self._load_legacy_conds(torch, model, embedding)
        to_device = getattr(loaded, "to", None)
        if callable(to_device):
            loaded = to_device(self._engine.device)
        return loaded

    def _load_legacy_conds(self, torch, model, embedding: bytes):
        """Load an embedding written by the first SPEECH-006 release.

        Those blobs are a pickle of the whole container.  They are read with
        ``weights_only=True``, allow-listing exactly the container classes the
        loaded model already uses, so nothing outside that set can be built.  A
        blob that does not load that way is refused with a clear error; it is
        never retried with ``weights_only=False``.
        """
        current = getattr(model, "conds", None)
        allowed = [type(current)] if current is not None else []
        t3 = getattr(current, "t3", None)
        if t3 is not None:
            allowed.append(type(t3))
        serialization = getattr(torch, "serialization", None)
        safe_globals = getattr(serialization, "safe_globals", None)
        guard = safe_globals(allowed) if callable(safe_globals) and allowed else nullcontext()
        try:
            with guard:
                return torch.load(io.BytesIO(embedding), map_location="cpu", weights_only=True)
        except Exception as exc:
            raise ChatterboxError(
                "This clone's embedding was written in the legacy pickle format and could "
                "not be loaded safely (weights_only=True). Re-create the clone from a new "
                "reference clip with the consent flow."
            ) from exc

    def render(
        self, text: str, embedding: bytes, *, exaggeration: float
    ) -> tuple[np.ndarray, int]:
        """Render text with the stored conditioning, returning float32 samples.

        The previous conditioning is restored in ``finally`` so that rendering a
        clone cannot change the voice of a concurrent or later pack-voice render.
        """
        torch = self._torch()
        model = self._model()
        kwargs: dict[str, Any] = {}
        if self._engine.supports_exaggeration:
            kwargs["exaggeration"] = float(exaggeration)
        with self._engine.model_guard:
            previous = getattr(model, "conds", None)
            try:
                model.conds = self._deserialise_conds(torch, model, embedding)
                samples = _to_numpy(model.generate(text, **kwargs))
            finally:
                model.conds = previous
        return samples, int(self._engine.sample_rate)


__all__ = [
    "CHATTERBOX_ATTRIBUTION",
    "CHATTERBOX_ENGINE",
    "CHATTERBOX_LICENSE",
    "CHATTERBOX_SAMPLE_RATE",
    "CHATTERBOX_STYLES",
    "COMPUTE_COST_ADVISORY",
    "EXAGGERATION_MAX",
    "EXAGGERATION_MIN",
    "ChatterboxDependencyError",
    "ChatterboxEngine",
    "ChatterboxCloneBackend",
    "ChatterboxError",
    "ChatterboxModelMissing",
    "ChatterboxModelStore",
    "PARALINGUISTIC_TAGS",
    "STYLE_EXAGGERATION",
    "download_model",
    "exaggeration_for",
    "neutralise_watermarker",
    "sanitise_text",
]
