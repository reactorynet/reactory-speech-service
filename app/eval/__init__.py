"""Local, offline evaluation harness for the Reactory speech platform.

Phase 6 of the voice-expression roadmap (SPEECH-007). The harness exists so that
routing decisions ("which engine for empathetic French?") and upgrade decisions
("is version N+1 better than N?") are measurements rather than opinions.

Everything runs on the machine: a committed golden corpus, the engine adapters,
the local faster-whisper service for round-trip intelligibility, a dependency-free
speaker embedder and a data-driven routing policy. There is no network call and no
paid service anywhere in this package.

Layout
------

``corpus``      golden corpus loading and validation
``textnorm``    normalisation and language-aware tokenisation for WER/CER
``metrics``     WER/CER, duration bounds, RTF, audio inspection
``speaker``     speaker-embedding consistency of repeated renders
``engines``     engine discovery (built in, entry point or ``--register``)
``stt``         local round-trip transcription
``runner``      renders the matrix and measures it
``report``      report schema, table, baseline diff
``routing``     data-driven routing policy
``cli``         ``python -m app.eval run|report|compare|route``

Documentation: ``docs/architecture/evaluation-harness.md``.
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = ["__version__"]
