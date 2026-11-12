# Measured: Chatterbox on CPU

**Date:** 2026-11-12 · **Machine:** Apple M1 Pro, 8 cores, 16 GB, native arm64
**Versions:** chatterbox-tts 0.1.7 · torch 2.15.0 · Python 3.14.3

This page exists because Phase 4 of the roadmap claimed *"Nano 110M for CPU/edge at roughly 3×
realtime on 8 cores"* and nobody had checked it. The figure came from upstream documentation, not
from this hardware. It is wrong by roughly an order of magnitude, and the roadmap has been
corrected.

Numbers here were produced by rendering through the library directly, then verifying intelligibility
with our own STT — not read from a README.

## Headline: 0.13 realtime factor

| Case | Words | Audio | Wall | **RTF** |
|---|---|---|---|---|
| short | 8 | 2.40 s | 17.9 s | **0.13** |
| medium | 32 | 10.00 s | 67.9 s | **0.15** |
| long | 80 | 21.76 s | 164.2 s | **0.13** |

`RTF = audio_seconds / wall_seconds`. **0.13 means one second of speech costs ~7.7 seconds of
compute** — the engine runs about 7.7× slower than realtime, not 3× faster.

Other figures: model load **21.9 s**; resident memory **3,976 MB**; sample rate **24,000 Hz** (the same
as Kokoro, so no resampling seam).

## The bottleneck is sequential decoding, not matmul

| Threads | 1 | 2 | 4 | 8 |
|---|---|---|---|---|
| RTF | 0.14 | 0.16 | 0.09 | 0.12 |

No scaling — the differences are noise. The logs show autoregressive T3 sampling at **2–5 tokens/s**,
one token at a time; each token depends on the previous one. **Adding cores cannot fix this.**
Closing the gap needs a smaller model or better decoding, not a larger machine.

## "Nano" is not available in the library

`ResembleAI/chatterbox-nano` exists on HuggingFace (t3 870 MB) but `chatterbox-tts` 0.1.7 exposes only:

| Class | Repo | t3 weights |
|---|---|---|
| `ChatterboxTTS` | `ResembleAI/chatterbox` | 2.13 GB |
| `ChatterboxTurboTTS` | `ResembleAI/chatterbox-turbo` | 1.92 GB |

There is no class for Nano. The S3Gen vocoder (~1.06 GB) is shared by every variant and dominates the
download, so no variant is small — the "Nano 110M" framing overstates the deployable win.

## Expressiveness works, with two caveats

`exaggeration` produces genuinely different renderings — waveform correlation with neutral is
approximately **−0.01**, not a filter:

| `exaggeration` | Duration | RMS |
|---|---|---|
| 0.0 | 10.32 s | 0.140 |
| 0.5 | 8.88 s | 0.144 |
| 1.0 | 7.32 s | 0.171 |

**Caveat 1 — it co-varies with pace.** The same sentence is 30% shorter at 1.0 than at 0.0. It is not
a pure emotion dial. Direction lowering must not apply `prosody.rate` on top without measuring the
interaction, and a caller asking for `style: "excited"` should expect the speaking rate to drift.

**Caveat 2 — high values cost diction.** Our own Whisper transcribed "Kokoro" as *"Cycocoro"* at 1.0
and *"Kakoro"* at 0.5. Usable band is roughly **0.3–0.7**.

## Watermarking is unconditional

`ChatterboxTTS.__init__` constructs `perth.PerthImplicitWatermarker()` and `generate()` calls
`apply_watermark()` on **every** return path. There is no flag, setting or environment variable to
disable it.

Measured with a fixed seed (`torch.manual_seed`): **max abs delta 0.110, rms delta 0.017** — real but
small, consistent with "imperceptible".

> Generation is stochastic. An unseeded comparison is meaningless: a first pass produced a max delta
> of 0.98, which was entirely sampling noise, not watermarking. Seed before comparing.

Replacing `model.watermarker` with a no-op object yields clean audio — verified. Neutralising it is
therefore a **required, explicit adapter step** to satisfy ADR-0003 decision 4, not an optional flag.
The engine does not "default on"; it always applies.

## Dependency fragility

Three separate breakages were needed to get a working import. All will recur in CI:

1. **`numba` (via `librosa`) imports `coverage.types`.** A namespace-package `coverage` shadow breaks
   it. Fix: install `coverage>=7.6`.
2. **`resemble-perth` imports `pkg_resources`**, removed in setuptools 81+. Perth swallows the
   `ImportError`, leaving `PerthImplicitWatermarker = None`, which fails later as
   `TypeError: 'NoneType' object is not callable` inside the model constructor. Fix: `setuptools<81`.
3. `chatterbox-tts` pulls `gradio`, `pre-commit` and `virtualenv` as runtime dependencies — demo and
   dev tooling in a service image.

## What this means for deployment

The capability is real; the CPU throughput claim is not. For sizing, at RTF 0.13:

- a 2–3 s assistant reply ≈ **15–25 s of one CPU core**
- a 30 s narration ≈ **4 minutes of one core**
- reaching ~1.0 RTF needs ~8× current throughput, **which more cores will not deliver**

So the expressive profile is appropriate for **asynchronous and batch narration**, not interactive
turn-taking, until throughput improves. `/capabilities` should carry a compute-cost advisory so
callers route short interactive replies to Kokoro and long narration to Chatterbox.

## Artefacts

Rendered WAVs (24 kHz mono 16-bit) are kept at
`REACTORY_DATA/profiles/reactor/personas/reactor/workspace/notes/samples/chatterbox/` —
`short`, `medium`, `long`, `exaggeration_{0.0,0.5,1.0}`, `wm_on`, `wm_off`.

## Not yet measured

**Chatterbox Turbo.** It is the one lever the evidence points at: fewer t3 weights and a different
sampling path (`inference_turbo`). If it lands near RTF 0.5–1.0 the phase stays viable for narration;
if it is also ~0.13 then CPU expressive TTS is not yet practical at this scale and the roadmap should
say so plainly. Nano likewise requires a library version that exposes it.
