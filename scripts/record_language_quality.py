"""Record per-language round-trip quality from the committed golden corpus.

SPEECH-010 review finding F1: the completion report claimed "0 quality failures"
for the advertised languages, but that run used a hand-built 8-entry subset. The
committed corpus tells a different story — ``zh-narration-long`` scores CER 0.92
with ``zf_xiaoxiao`` — and the catalogue still advertised Mandarin as a clean
"yes". Selecting the sentences that pass is exactly the evidence standard the
ticket rejects.

This script is the fix: it measures **every** committed corpus entry for every
voice that ships, writes the result to ``app/eval/data/language-quality.json``,
and that file is what the catalogue table, ``GET /api/tts/capabilities`` and the
regression tests all read. There is no second, hand-maintained list of numbers.

Run it when the corpus or the models change::

    PYTHONPATH=. python scripts/record_language_quality.py

It needs the real models (``make download-models``) because a quality figure that
was not measured from audio is the thing this file exists to prevent. Without
models it exits non-zero rather than writing a guess.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from app.eval.corpus import load_corpus
from app.eval.runner import EvaluationRunner, RunnerConfig
from app.eval.stt import LocalTranscriber
from app.services.stt_service import STTService
from app.services.tts_service import TTSService

#: Written into every record so a reader can see how it was produced.
RECORD_COMMAND = (
    "PYTHONPATH=. python scripts/record_language_quality.py "
    "--voices af_heart,bf_emma,ef_dora,ff_siwis,hf_alpha,if_sara,pf_dora,"
    "zf_xiaoxiao,zm_yunjian,jf_alpha"
)

#: Quality bands. ``production`` means every measured entry round-tripped within
#: its corpus target; ``caveats`` means the language works but at least one
#: representative sentence did not. There is deliberately no "close enough" band:
#: the ticket's point is that a number either met its target or it did not.
PRODUCTION = "production"
CAVEATS = "caveats"
UNSUPPORTED = "unsupported"


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), 4)


def record(corpus_path: Path | None, voices: str) -> dict[str, Any]:
    corpus = load_corpus(corpus_path)

    tts = TTSService()
    asyncio.run(tts.initialize())
    stt = STTService()
    asyncio.run(stt.initialize())
    if not stt.available_models:
        raise SystemExit(
            "no faster-whisper model is mounted under models/; a round-trip quality "
            "record cannot be produced without one (run `make download-models`)"
        )

    engine = tts.engine_manager.get("kokoro")
    runner = EvaluationRunner(
        corpus,
        [engine],
        LocalTranscriber(stt),
        config=RunnerConfig(repeats=1, probe_rate=1.5),
        voices_spec=voices,
    )
    cells = runner.run()

    languages: dict[str, Any] = {}
    for cell in cells:
        target_language = cell.language
        row = languages.setdefault(
            target_language,
            {
                "quality": PRODUCTION,
                "voices": {},
                "entries_measured": 0,
                "entries_over_target": [],
                "worst_cer": 0.0,
                "worst_wer": 0.0,
                "worst_entry": None,
            },
        )
        voice_entries: list[dict[str, Any]] = []
        for item in cell.entries:
            entry = corpus.get(item["id"])
            over_target = "wer_threshold" in item["checks"]
            voice_entries.append(
                {
                    "id": item["id"],
                    "wer": _round(item["wer"]),
                    "cer": _round(item["cer"]),
                    "target": entry.max_wer,
                    "over_target": over_target,
                    "transcript": item["transcript"],
                }
            )
            row["entries_measured"] += 1
            for metric in ("cer", "wer"):
                value = item[metric]
                if value is not None and value > (row[f"worst_{metric}"] or 0.0):
                    row[f"worst_{metric}"] = _round(value)
                    row["worst_entry"] = item["id"]
            if over_target and item["id"] not in row["entries_over_target"]:
                row["entries_over_target"].append(item["id"])
        row["voices"][cell.voice] = voice_entries

    # A language whose text front end cannot phonemise every voice's probe is
    # unsupported; a language with a failing representative sentence is usable
    # with caveats. The classification is derived from measurement, never typed.
    for language, row in languages.items():
        if row["entries_over_target"]:
            row["quality"] = CAVEATS
    for row in runner.skipped:
        language = row["language"]
        if language in languages:
            continue
        languages[language] = {
            "quality": UNSUPPORTED,
            "reason": row["reason"],
            "voices": {},
            "entries_measured": 0,
            "entries_over_target": [],
            "worst_cer": None,
            "worst_wer": None,
            "worst_entry": None,
        }

    return {
        "schema_version": 1,
        "description": (
            "Per-language round-trip quality measured on the committed golden corpus "
            "through the local faster-whisper model. This file, not a hand-written "
            "table, is what docs/voices/kokoro-voice-catalogue.md and "
            "GET /api/tts/capabilities report. Regenerate with the command below; "
            "app/eval/tests assert the values still match a live run."
        ),
        "generated_by": RECORD_COMMAND,
        "corpus": {
            "digest": corpus.digest,
            "entries": len(corpus),
            "languages": corpus.languages,
        },
        "stt_model": str(stt.default_model),
        "languages": languages,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default=None, help="corpus path (default: committed)")
    parser.add_argument(
        "--voices",
        default=(
            "af_heart,bf_emma,ef_dora,ff_siwis,hf_alpha,if_sara,pf_dora,"
            "zf_xiaoxiao,zm_yunjian,jf_alpha"
        ),
        help="comma-separated voice ids, one or two per language family",
    )
    parser.add_argument(
        "--out",
        default="app/eval/data/language-quality.json",
        help="record path",
    )
    args = parser.parse_args(argv)

    payload = record(None if args.corpus is None else Path(args.corpus), args.voices)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"wrote {out}")
    for language, row in sorted(payload["languages"].items()):
        if row["quality"] == UNSUPPORTED:
            print(f"  {language:5} unsupported — {row['reason']}")
            continue
        print(
            f"  {language:5} {row['quality']:11} "
            f"worst_wer={row['worst_wer']} worst_cer={row['worst_cer']} "
            f"over_target={row['entries_over_target']}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover - script entry point
    sys.exit(main())
