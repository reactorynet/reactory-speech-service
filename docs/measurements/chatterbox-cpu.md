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

---

## Addendum — what the adapter does with these numbers (SPEECH-005)

This section was added by the SPEECH-005 implementation. The measurements above are the
committed record from the measurement branch and are unchanged.

### Variants the adapter exposes

| `CHATTERBOX_VARIANT` | Class | t3 weights | Expressive styles |
|---|---|---|---|
| `standard` | `ChatterboxTTS` | 2.13 GB | **yes** — `exaggeration` is honoured |
| `turbo` | `ChatterboxTurboTTS` | 1.92 GB | **no** — see below |

Turbo's `generate()` *accepts* `exaggeration` and then ignores it:

```
if cfg_weight > 0.0 or exaggeration > 0.0 or min_p > 0.0:
    logger.warning("CFG, min_p and exaggeration are not supported by Turbo version and
                    will be ignored.")
```

and `ChatterboxTurboTTS.from_local` builds its config with `hp.emotion_adv = False`, so there is
no emotion embedding to drive. The adapter therefore uses an explicit per-variant capability
table (`VARIANT_SUPPORTS_EXAGGERATION`), advertises `styles: ["neutral"]` for turbo, and reports
`style`/`intensity` in `direction_ignored`. A signature-name check would have advertised seven
styles that do nothing.

**Turbo remains unmeasured.** The RTF figures on this page are `ChatterboxTTS` only. Turbo's
different sampling path (`inference_turbo`, one-step decoder) may be faster; nobody has measured
it on this hardware, so the compute-cost advisory is deliberately the same conservative figure
for both variants.

### Download scope (corrected)

The whole HuggingFace repository is much larger than what `from_local` reads:

| Variant | Whole repository | Files `from_local` reads | Ratio |
|---|---|---|---|
| `standard` | 13,866,212,931 B / 18 files (13.87 GB) | 3,191,966,992 B / 5 files (3.19 GB) | 4.3× |
| `turbo` | 4,044,184,736 B / 13 files (4.04 GB) | 2,987,680,596 B / 9 files (2.99 GB) | 1.4× |

Captured 2026-11-12 from the HuggingFace API (`?blobs=true`), at `ResembleAI/chatterbox`
@ `5bb1f6ee` and `ResembleAI/chatterbox-turbo` @ `749d1c1a`. The per-file listing is checked in
at `tests/data/chatterbox-upstream.json`, so the required set is asserted against reality rather
than against a fake hub that materialises whatever it is asked for.

**Turbo has no `tokenizer.json`.** Its `tokenizer_config.json` declares `GPT2Tokenizer`, which
`AutoTokenizer.from_pretrained` loads from `vocab.json` + `merges.txt`. `tokenizer.json` exists only
in the standard repository, where `EnTokenizer` reads it. An earlier revision of the adapter
required it for both variants, which made `CHATTERBOX_VARIANT=turbo` impossible to download, verify
or load — the failure mode a permissive fake could not reveal.

The excess is nine redundant checkpoints in `ResembleAI/chatterbox`: duplicate `*.pt` vocoders,
the v2/v3 multilingual T3 weights, `t3_23lang`, and a second vocoder file. An earlier revision of
this ticket said "3.1 GB" while downloading all 13.87 GB and SHA-256ing every byte of it.

`scripts/download_chatterbox.py` now passes `allow_patterns` ( `VARIANT_FILES` ), records only
those files in the manifest, and stores `download_bytes`, `repo_total_bytes` and
`required_total_bytes` so the difference is auditable. File sizes were taken from the HuggingFace
API (`/api/models/<repo>?blobs=true`) on 2026-11-12.

### SHA-256 verification cost

`ensure_available()` hashes every recorded file. At 3.19 GB that is a measurable cost on every
cold load, on top of the 21.9 s model load:

| Quantity | Value |
|---|---|
| Bytes hashed (`standard`) | 3,191,966,992 B (3.19 GB) |
| Measured SHA-256 throughput | **331 MiB/s** (in-memory) |
| Cost per cold verification, real weights on disk | **9.66 s** |
| Cost once memoised (same process, unchanged files) | **1.0 ms** |

Because an idle engine is unloaded and reloaded, the result is memoised per process against a
signature of `(file, size, mtime_ns)` plus the manifest mtime, so a reload with demonstrably
unchanged files does not re-hash 3.19 GB. Any truncation, replacement or rewrite changes the size
or mtime and forces a full re-verification. This does not weaken the threat model the manifest
addresses (corruption and accidental replacement — a hostile models volume can rewrite the
manifest too).

### Verification cost

Measured on the implementation host with `hashlib.sha256` over a 256 MiB buffer:

```
$ python -c "import hashlib,os,time; d=os.urandom(256*1024*1024); t=time.perf_counter(); \
             hashlib.sha256(d).hexdigest(); e=time.perf_counter(); \
             print(f'{256/(e-t):.0f} MiB/s -> 3.19 GB in {3191.97/(256/(e-t)):.1f} s')"
```

Recorded output:

```
$ .venv/bin/python - <<'PY'
import hashlib, os, time
data = os.urandom(256*1024*1024)
t = time.perf_counter(); hashlib.sha256(data).hexdigest(); e = time.perf_counter()
mibs = 256/(e-t)
print(f"{mibs:.0f} MiB/s -> 3.19 GB (3191.97 MiB) in {3191.97/mibs:.1f} s")
PY
331 MiB/s -> 3.19 GB (3191.97 MiB) in 9.6 s
```

So a cold load costs **21.9 s of model loading plus ~9.6 s of verification** — 44% overhead. That
is why the result is memoised rather than recomputed on every reload. On a node whose models volume
is on spinning disk or a network mount the hashing throughput will be lower; budget accordingly, or
keep one engine instance resident so the memo is hit.

There is no environment variable to skip verification: skipping it is the failure
mode the manifest exists to prevent.

Both outcomes are memoised — success *and* failure — keyed on the size and
`mtime_ns` of every required file plus the manifest's own mtime. A truncated
weight file therefore costs one 9.66 s pass, not 9.66 s per request, and fixing
the file invalidates the key automatically.

### Verification integrity

`ensure_available()` refuses two states that a "hash whatever the manifest lists"
check would accept:

- **A manifest for a different variant.** Both variants default to
  `models/chatterbox`. Downloading `turbo` over a directory that already holds
  `standard` rewrites `manifest.json` while the standard checkpoints stay on
  disk; the store would then hash only turbo's entries and load a `t3_cfg` /
  `s3gen` that no recorded hash covers. The manifest's `variant` must match the
  store's, and `download_chatterbox.py` refuses to overwrite the other variant
  unless `--force` is passed.
- **A manifest with no entry for a required file.** Hashing the manifest's own
  list is not verification: every name in `VARIANT_FILES` must carry a SHA-256.

### Acceptance evidence — the style lowering is audibly different

The SPEECH-005 acceptance criterion asks for rendered WAVs or a measured prosody difference
between `style: "neutral"` and `style: "empathetic"`. The rendered artefacts from the measurement
run are kept at:

`REACTORY_DATA/profiles/reactor/personas/reactor/workspace/notes/samples/chatterbox/`
(`exaggeration_0.0.wav`, `exaggeration_0.5.wav`, `exaggeration_1.0.wav`).

Measured on those files (24 kHz mono 16-bit, same sentence and reference voice):

| File | `exaggeration` | Duration | RMS | Peak | Zero-crossing rate | Correlation with 0.5 |
|---|---|---|---|---|---|---|
| `exaggeration_0.0.wav` | 0.00 | 10.320 s | 0.1403 | 0.957 | 0.1208 | 0.007 |
| `exaggeration_0.5.wav` | 0.50 (neutral) | 8.880 s | 0.1440 | 1.000 | 0.1241 | 1.000 |
| `exaggeration_1.0.wav` | 1.00 | 7.320 s | 0.1708 | 1.000 | 0.1063 | 0.011 |

Near-zero waveform correlation between settings is the point: this is a different rendering, not a
gain or filter applied to one rendering.

The adapter maps `style: "neutral"` to `exaggeration=0.5` and `style: "empathetic"` to
`exaggeration=0.44` (see `STYLE_EXAGGERATION`). The 0.62 mapping this page originally discussed was
lowered after a round-trip measurement: our own STT heard **"I've been"** for **"Hi Vin"** at 0.62,
while 0.44 rendered **"Hi, Vin"** correctly. Short words and proper nouns degrade first, and a clone
inheriting the old default would mangle the names of the person whose voice it reuses.

0.44 sits below the measured 0.5 point, so the useful bound is the 0.0 → 0.5 slope:

- **0.5 → 0.44** (12% of that interval, reversed): duration ≈ +1.7%, RMS ≈ −0.3%
- **0.5 → 1.0:** duration −17.6%, RMS +18.6%
- **0.0 → 0.5:** duration −14.0%, RMS +2.6% — the whole scale moves pace, which is why
  `prosody.rate` is never stacked on top of it.

The real render has **not** been produced on this host: it is an Intel macOS machine and
PyTorch publishes no macOS x86_64 wheel for 2.6.0, so the real backend cannot be installed here.
The `slow` test `test_empathetic_audibly_differs_from_neutral_on_cpu` renders 0.5 and the mapped
empathetic value through the real model and asserts a duration/RMS delta when weights are mounted;
it skips otherwise.
