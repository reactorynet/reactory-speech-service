# Evaluation harness and routing policy

Phase 6 of the [voice-expression roadmap](../roadmap/voice-expression-roadmap.md)
(`SPEECH-007`). The harness exists so that two questions have answers instead of
opinions:

1. **Routing** — which engine should render this request, given its language, style
   and tier?
2. **Upgrade** — is version N+1 better than version N?

Everything runs locally. There is no network call and no paid service anywhere in
`app/eval/`: a committed corpus, the engine adapters, the local faster-whisper
service for round-trip intelligibility, a dependency-free speaker embedder and a
JSON routing policy. Anyone can reproduce the numbers on a laptop.

```
python -m app.eval run      # measure the corpus, write a JSON report
python -m app.eval report   # render a comparison table
python -m app.eval compare  # diff a run against a baseline
python -m app.eval route    # ask the policy which engine to use
```

## What is measured

Per engine and voice, for every corpus sentence in that voice's language:

| Metric | Definition | Deterministic? |
|---|---|---|
| `wer` | Word error rate of the local STT transcript against the source, after NFKC normalisation, case folding and punctuation removal. For Japanese/Chinese the tokens are characters, so read `cer` as the primary number. | yes |
| `cer` | Character error rate of the transcript against the source. | yes |
| `duration_s` | Audio duration from the WAV frame count and header sample rate. | yes |
| `duration_ratio` | `duration_s` / nominal duration of the sentence at rate 1.0. A ratio near 1.4 means the engine speaks at about 2.4 words per second. | yes |
| `rtf` | Real-time factor: audio seconds per wall-clock synthesis second. Higher is faster. | machine-dependent |
| `ttfa_s` | Time to the first streamed chunk. Kokoro streams one sentence at a time, so this is first-sentence latency, not first-word latency. | machine-dependent |
| `peak_memory_mb` | Peak Python heap during one render (`tracemalloc`). Best effort: ONNX Runtime's native allocations are not attributed. `--memory-mode rss` reports the process high-water RSS instead, which is not per-render. | machine-dependent |
| `speaker_consistency_mean/min` | Cosine similarity between speaker embeddings of repeated renders of the same voice and text. 1.0 is a perfectly stable voice; `min` is the worst pair. | yes |
| `prosody_adherence` | `measured_rate / requested_rate`, where `measured_rate = base_duration / requested_duration`. 1.0 means a 1.5x request rendered exactly 1.5x faster; ~0.67 means the engine ignored the request. | yes |

Two gates are separated on purpose:

* **integrity** — the render is broken: silent, truncated, undecodable, wrong sample
  rate, wrong channel count, a synthesis exception, a failed probe. An integrity
  finding always fails the run (`exit 1`).
* **quality** — the render is well-formed but unintelligible (`wer` above the
  entry's `max_wer`). Recorded always; gates the exit code only with
  `--fail-on-wer`. That lets the harness be adopted now, while the multilingual
  bugs it just found are still open (see [current findings](#current-findings)).

### The speaker embedder, and why it is a proxy

`app/eval/speaker.py` implements a deterministic, dependency-free **log-mel
statistics embedder**: 25 ms Hamming frames at a 10 ms hop, a 26-band mel
filterbank, log magnitude, then mean and standard deviation across time, L2
normalised. It needs no download, no licence beyond numpy and no network — which is
the binding direction for this harness.

It measures **timbre and channel stability of repeated renders**. Two renders of the
same voice with the same text differ only in prosody; a different voice, a
resampled render or a re-encoded render does not. Cosine ≈ 1.0 therefore means "this
voice is stable", and a drop across versions means "this voice drifted".

It is **not** a biometric speaker-verification model. Absolute values are not
comparable to ECAPA/x-vector numbers and must never be used to claim two clips are
the same person. They are a regression signal for one engine/voice across runs.

A trained local model can replace it without touching the harness:
`--speaker-embedder module:factory`, where the factory returns an object with
`embed(samples, sample_rate) -> vector` and an `id`. The report records the embedder
id, and `compare` refuses to diff reports produced by different embedders.

## Determinism, variance and diffing

A report has no wall clock unless `--stamp` is passed, all floats are rounded to a
fixed precision and JSON keys are sorted, so two runs of the same corpus, engine and
settings differ only in the machine-dependent metrics.

Declared tolerances (`app/eval/report.py:TOLERANCES`, copied into every report):

| Metric | Tolerance |
|---|---|
| `wer`, `cer`, `wer_mean`, `cer_mean` | absolute 0.02 |
| `speaker_consistency_mean/min` | absolute 0.02 |
| `duration_s` | relative 5% |
| `duration_ratio` | relative 8% |
| `rtf` | relative 15% |
| `ttfa_s` | relative 20% |
| `peak_memory_mb` | relative 15% |
| `prosody_adherence` | relative 5% |

`compare` classifies each metric as `regression`, `improvement`, `unchanged`, `new`
or `missing`. Direction matters: higher is worse for error rates, memory and
latency; lower is worse for consistency and adherence. A duration shift beyond
tolerance is reported as `changed` rather than as an improvement or a regression,
because neither direction is inherently better for the same sentence — but it still
fails the gate, so a model that starts dropping or padding audio cannot slip through.

**The gate only checks reproducible metrics.** `--fail-on-regression` gates on
`wer`, `cer`, `duration_s`, `duration_ratio`, `speaker_consistency_*` and
`prosody_adherence` — the numbers a deterministic engine reproduces. `rtf`,
`ttfa_s` and `peak_memory_mb` are always reported and diffed, but two runs on a
busy laptop differ by tens of percent, so gating on them would fail releases for
scheduling noise. Ask for them explicitly with `--metrics all` (or a list, e.g.
`--metrics wer,cer`).

**Comparability.** `compare` refuses to diff reports whose corpus digest, engine
set, speaker embedder or Whisper model differ. Those are not regressions; they are
different experiments. Cross-machine comparison is meaningful for the quality
metrics (WER/CER/duration) and meaningless for `rtf`, `ttfa_s` and memory, which is
why their tolerances are relative and generous.

## The golden corpus

`app/eval/data/golden_corpus.json` — 22 entries, reviewed like source code and
identified by a SHA-256 digest in every report.

Each entry:

```json
{
  "id": "en-domain-vocabulary",
  "text": "Reactory, GraphQL, Kubernetes, ONNX and FQDN all appear in the platform vocabulary.",
  "language": "en",
  "use": "domain vocabulary",
  "tags": ["domain", "acronyms"],
  "max_wer": 0.45
}
```

Coverage: short greetings, long narration, domain vocabulary (Reactory, GraphQL,
Kubernetes, ONNX, FQDN), numbers, dates, abbreviations, identifiers, punctuation,
and five languages (English, French, Japanese, Chinese, Spanish). `use` is the
intended use; `max_wer` is the quality target used by `--fail-on-wer`.

`tests/test_eval_corpus.py` asserts that coverage, that every language has a shipped
voice, and that the loader rejects duplicate ids, missing fields and out-of-range
targets. A corpus edit changes the digest, so a baseline measured on the old corpus
is reported as not comparable instead of silently reused.

## Adding an engine

An engine is an adapter plus a corpus run. Three registration paths, no harness code
change:

1. **Built in** — add the id and class to `BUILTIN_ENGINES` in
   `app/eval/engines.py` (Kokoro is the only entry today).
2. **Entry point** — ship the adapter as a distribution advertising the
   `reactory.speech.eval_engines` group:

   ```toml
   [project.entry-points."reactory.speech.eval_engines"]
   chatterbox = "app.engines.chatterbox:ChatterboxEngine"
   ```

3. **Command line** — `--register app.engines.chatterbox:ChatterboxEngine`, or pass
   `--engines module:attr` directly.

The harness only calls `capabilities()`, `voices()`, `synthesize()` and optionally
`stream()`, so a new adapter is measured by:

```bash
python -m app.eval run --engines chatterbox --voices default --out reports/chatterbox.json
python -m app.eval compare reports/chatterbox.json app/eval/baselines/kokoro-v1.0.0.json
```

The second command refuses (exit 2) because the engine sets differ — that is the
intended behaviour: comparing a different engine is a new experiment, not a
regression. Use `--engines kokoro,chatterbox` for a like-for-like run.

## Routing policy

`app/eval/policies/routing.json` is the policy. It is data so a routing change is a
reviewable diff:

```json
{
  "default_engine": "kokoro",
  "fallback_chain": ["kokoro"],
  "rules": [
    {
      "id": "expressive-style",
      "when": { "style": ["empathetic", "excited", "cheerful", "whisper"] },
      "engine": "chatterbox",
      "require": { "supports_requested_style": true },
      "rationale": "Kokoro reports style as ignored; an empathetic request must not land there."
    }
  ]
}
```

* `when` predicates: `language`, `style`, `tier`, `rate_at_least`, `rate_at_most`.
  An absent predicate is a wildcard.
* `require` gates on the candidate engine: `native_rate`, `native_pitch`,
  `native_volume`, `supports_streaming`, `styles`, `supports_requested_style`,
  `max_wer`, and `beat` (`{metric, against, margin}` — a measured comparison against
  another engine).
* Rules are ordered; the first whose `when` matches **and** whose gates pass wins.
* A rule that matches but cannot be honoured is recorded as *skipped* with a reason
  and evaluation continues.
* When nothing is usable the request degrades to `default_engine`, then
  `fallback_chain`, then the first available engine. `degraded: true` records that a
  matching rule could not be honoured.

A **score gate never fires blind**: `max_wer` and `beat` fail when the measurement
is missing. Promoting an engine on measured quality with no measurement is exactly
the decision this phase exists to prevent.

Every `run` embeds the routing matrix — each corpus language × style × tier — in the
report, so the routing answer and the measurements behind it live in one artifact:

```bash
python -m app.eval route --language ja --style neutral --tier standard
python -m app.eval route --matrix
python -m app.eval route --language en --style empathetic --tier premium \
  --available kokoro,chatterbox --scores reports/run.json
```

## Baselines and cadence

Committed baseline: `app/eval/baselines/kokoro-v1.0.0.json`.

Regenerate it (models mounted, ~5 minutes on an 8-core laptop):

```bash
export MODELS_DIR=/path/to/models      # kokoro-v1.0.onnx, voices-v1.0.bin, whisper-base/
python -m app.eval run \
  --engines kokoro --voices default --stt-model base \
  --out reports/kokoro.json \
  --baseline-out app/eval/baselines/kokoro-v1.0.0.json
python -m app.eval compare reports/kokoro.json app/eval/baselines/kokoro-v1.0.0.json \
  --fail-on-regression
```

A fresh run against the committed baseline must produce `regressions=0` on the
gating metrics (verified when the baseline was committed; the only expected deltas
are `rtf`/`ttfa_s`/memory).

Promote a new baseline only when the diff is understood and reviewed: a baseline
update is a claim that the change is intended.

**Cadence.** A full corpus run is minutes, not seconds, and it needs the models, so
it is **not** a per-commit gate. The per-commit gate stays `pytest -m "not slow"` +
`ruff check app/ tests/`, which includes the harness's unit and falsification tests
against engine doubles. Run the full corpus:

* **on every release** (and before merging an engine, voice or model change);
* **weekly** on the reference node, to catch drift in dependencies;
* **whenever a routing rule or a baseline is changed**.

CI can run the harness without models by pointing it at a registered double, which
is how the falsification tests work; a full-model run belongs on a machine with the
models mounted.

## How the harness falsifies

The ticket's requirement is that a deliberately broken engine is *reported as a
failure, not a pass*. `tests/test_eval_harness.py` proves it end to end — real audio
bytes through the real pipeline, with only the acoustic model doubled:

| Breakage | Detected by |
|---|---|
| muted output (digital silence) | `silent_output`, `empty_transcript`, WER 1.0 |
| truncated output (20% of the samples) | `duration_out_of_bounds` and a higher WER |
| wrong sample rate (24 kHz samples under a 48 kHz header) | `unexpected_sample_rate`, plus a shorter measured duration |
| undecodable bytes | `undecodable_audio` |
| `synthesize` raises | `synthesis_error` |
| engine ignores the requested rate | `prosody_adherence ≈ 1/requested` and `ignored: ["rate"]` |
| unintelligible but well-formed render | `wer_threshold` (quality gate, `--fail-on-wer`) |

Two structural guarantees back that up:

* a run that measures **no** engine/voice cell raises instead of writing an empty
  report that says `passed`;
* a missing transcript is never WER 0 — an absent Whisper model exits 2 with the
  reason, and `--no-stt` records `stt: disabled` rather than a score.

## Current findings

The committed baseline is the evidence the ticket asked for, and it reproduces the
motivating bug:

| Engine | Voice | Language | WER | Duration ratio | Status |
|---|---|---|---|---|---|
| kokoro | af_heart | en | 0.096 | 1.44 | ok |
| kokoro | ef_dora | es | 0.098 | 1.18 | ok |
| kokoro | ff_siwis | fr | 0.146 | 1.29 | ok |
| kokoro | jf_alpha | ja | 2.69 | 2.58 | broken |
| kokoro | zf_xiaobei | zh | 0.96 | 1.33 | poor |

Japanese is not merely inaccurate: `ja-narration-long` raises
`IndexError: index 510 is out of bounds for axis 0 with size 510` inside Kokoro
(other long Japanese input is silently truncated: *"Phonemes are too long,
truncating to 510 phonemes"*), and the short greeting renders 13.9 s of nonsense for
21 characters. Chinese scores a
WER just under 1.0, i.e. the transcript is unrelated to the source. English,
Spanish and French are usable.

**This harness does not fix those bugs.** That is `SPEECH-010`'s job; this is the
evidence that will prove the fix, and the baseline that will show it as an
improvement rather than a claim.

## Out of scope

* Human MOS panels. The harness produces objective proxies; a later script can
  collect human ratings.
* A full corpus run on every commit (see [cadence](#baselines-and-cadence)).
* Any hosted metric service.
