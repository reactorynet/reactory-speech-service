## Summary

Optional, flag-gated Chatterbox (MIT) expressive engine behind the existing TTSEngine contract. The default speech-core image, requirements.txt and all existing endpoint shapes are unchanged; the adapter module is imported only when CHATTERBOX_ENABLED=true and lazy-loads on first request. This revision removes the phantom tokenizer.json from the turbo required-file set (which made the documented CHATTERBOX_VARIANT=turbo deployment permanently undownloadable and unloadable), pins the upstream repository layout in a checked-in fixture so the same class of drift cannot recur silently, aligns the resident-memory advisory with the measurement page it cites, tightens the requirements test to the exact pins, refreshes the upstream size constants with a capture date, stops reporting phonetic_preprocess=false as an ignored control, strengthens the slow prosody assertion, and rebases the branch onto master with the docs reconciled additively.

- **Ticket / issue:** `REACTORY-302016408/20261110/4004237369`
- **Related PRs / commits:** 1 commit(s) on `feature/6af2b2498537a4b915289521` — latest: AI engineering run: REACTORY-302016408/20261110/4004237369
- **Reviewer focus:** Product deviation (recorded in the roadmap) still needs owner acknowledgement: the ticket specified a CPU Nano tier at ~3x realtime; the released library has neither, and measured throughput makes CPU expressive TTS batch-only.

## Type of change

- [x] Feature
- [ ] Bug fix
- [ ] Refactor (no functional change)
- [ ] Performance improvement
- [ ] Documentation
- [ ] Build / CI / tooling
- [ ] Dependency upgrade
- [ ] **Breaking change**

_Derived from the ticket `requestType` = `feature`._

## What changed

- `.env.example`
- `.github/workflows/ci.yml`
- `.gitignore`
- `Dockerfile.expressive`
- `Makefile`
- `app/config.py`
- `app/core/contracts.py`
- `app/data/voices.json`
- `app/engines/chatterbox.py`
- `app/engines/manager.py`
- `app/engines/router.py`
- `app/models/schemas.py`
- `app/routers/tts.py`
- `app/services/tts_service.py`
- `app/services/voice_registry.py`
- `docs/architecture/speech-platform.md`
- `docs/measurements/chatterbox-cpu.md`
- `docs/roadmap/voice-expression-roadmap.md`
- `requirements-expressive-macos.txt`
- `requirements-expressive.txt`
- `scripts/download_chatterbox.py`
- `tests/data/chatterbox-upstream.json`
- `tests/test_chatterbox_engine.py`
- `tests/test_engine_adapters.py`
- `tests/test_voice_catalogue.py`

## How was this tested?

### Automated

- [ ] Type check (`yarn typecheck` / `tsc --noEmit`)
- [x] Lint (`yarn lint`)
- [ ] Unit tests (`yarn test`) — suites touched: no test files changed
- [ ] Integration tests (`yarn test:integration`) — server only, and only where relevant
- [ ] Not applicable (docs/config only) — rationale:

### Manual verification

1. `make test` — not recorded
2. `make lint` — not recorded
3. `.venv/bin/coverage run --source=app -m pytest -q -m "not slow" && .venv/bin/coverage report -m --include=app/engines/chatterbox.py` — not recorded
4. `.venv/bin/pytest -q tests/test_chatterbox_engine.py -m "not slow"` — not recorded
5. `.venv/bin/pytest -q tests/test_chatterbox_engine.py -m slow -rs` — not recorded
6. `<mutation check> re-insert tokenizer.json into VARIANT_FILES['turbo'], run the upstream-layout tests, then restore` — not recorded
7. `<ad-hoc> download_model(variant='turbo') against a hub that creates only files present in tests/data/chatterbox-upstream.json` — not recorded
8. `<ad-hoc HTTP> /api/tts/synthesize/json + /api/tts/capabilities for standard and turbo` — not recorded

**Result:** 0 of 12 recorded command(s) passed, 12 did not.

## UI / UX evidence

_Not applicable — server-only change: the diff touches no client or UI files. Every box below is intentionally left unticked._

## Impact analysis

| Area | Impact |
| --- | --- |
| Server modules affected | None — no `src/modules` files changed |
| Client components / plugins affected | None — no client or plugin files changed |
| GraphQL schema / resolver changes | None — no schema or resolver files changed |
| Database migrations | None — no migration files changed |
| Config or environment variables | `.env.example` |
| Submodule version bumps | None — no submodule pointers changed |
| Cross-repo dependency | None — no dependency manifest changes |
| i18n key changes | None — no i18n files changed |

_Derived from the 25 file(s) in the branch diff against `master`._

## Breaking changes & rollback

**Is this a breaking change?** No — no public API, GraphQL schema, database migration or configuration contract change was detected in the diff.

**Rollback plan:**

- [x] Reverting this PR is sufficient
- [ ] Requires config revert — describe:
- [ ] Requires data / migration rollback — describe:
- [ ] Requires coordinated client + server release ordering — describe:

## Security considerations

- [x] No secrets, tokens or credentials committed (reviewed `git diff` and `.env*` files)
- [ ] Authorisation and permission checks reviewed for new endpoints, resolvers or menu entries
- [ ] User input validated at trust boundaries; query parameters and payloads bounded
- [ ] New or upgraded dependencies checked for advisories (`yarn audit` / `npm audit`)
- [ ] No sensitive data in logs, metrics or error responses
- [ ] Not applicable — rationale:

_Not applicable in this run — authorisation and permission checks: no authorisation code in the diff; user input validation: no resolver or endpoint changes in the diff; dependency advisories: no dependency manifest changed; sensitive data in logs: the diff adds no logging of request payloads or credentials._

## Documentation

- [ ] `reactory-docs` updated
- [ ] Inline JSDoc / API comments added for new public surface
- [ ] `.env.example` or config templates updated
- [ ] `CHANGELOG.md` updated (if this repo keeps one)
- [ ] No documentation required

_Not applicable in this run — documentation files changed: `docs/architecture/speech-platform.md`, `docs/measurements/chatterbox-cpu.md`, `docs/roadmap/voice-expression-roadmap.md`. No documentation box is ticked because the pipeline cannot evidence it._

## Author checklist

- [ ] Branch rebased on latest `master` / `main` and merges cleanly
- [ ] Commits are signed (`git log --show-signature`) and follow Conventional Commits
- [ ] No debug logging, commented-out code or unrelated drive-by changes
- [ ] Full diff reviewed line by line before requesting review
- [x] Ticket reference present in the title or description

_Only the ticket-reference box is ticked: it is the one author claim this run can evidence (the pull request title carries `REACTORY-302016408/20261110/4004237369`). The remaining claims are the author's to make._

## Reviewer checklist

- [ ] Behaviour matches the stated intent and the linked ticket
- [ ] Tests cover the changed paths and assert something meaningful
- [ ] Error handling and edge cases handled, not just the happy path
- [ ] No obvious performance regression (N+1 queries, unbounded loops, needless re-renders)
- [ ] Public API or contract changes have explicit sign-off
- [ ] Naming and structure consistent with surrounding code

### Risks

- Product deviation (recorded in the roadmap) still needs owner acknowledgement: the ticket specified a CPU Nano tier at ~3x realtime; the released library has neither, and measured throughput makes CPU expressive TTS batch-only.
- torch==2.6.0 is pinned by chatterbox-tts and carries known CVEs fixed only in later releases. The pin cannot move until chatterbox-tts widens its requirement; do not expose this profile to untrusted model inputs.
- Explicit engine='chatterbox' while the flag is off still returns 422 (the router's existing contract for an explicit override); style-based requests fall back to Kokoro, and with the flag on a missing model does fall back.
- A foreign voice that falls back is substituted with the target engine's variant-correct default and reported as 'voice' in direction_ignored; honest, but a timbre change.
- SHA-256 verification adds ~9.6 s to the first load, memoised per process by (file, size, mtime_ns) + manifest mtime.
- tests/data/chatterbox-upstream.json is a dated snapshot. If upstream adds or removes files the layout tests fail by design, which is the intended prompt to re-check VARIANT_FILES rather than a spurious failure.
- WebSocket streaming reports the plan computed before the render, so a mid-stream fallback can name the intended engine in its done frame; HTTP /synthesize reports the engine that actually rendered. Pre-existing streaming-reporting limitation.
- The rebase left the change set uncommitted in the worktree (commit → rebase → mixed reset), matching how the previous revisions were submitted; publication should squash it into the branch properly.

### Acceptance criteria

_Acceptance criteria: 7 met, 0 unmet, 1 partially met (of 8). Only a human can confirm these._

- **With the flag off: default image size, startup time and behaviour are unchanged; /capabilities lists only Kokoro.** — Met
  git diff --stat Dockerfile requirements.txt -> empty. Flag-off subprocess: adapter_imported False, torch_imported False, engines ['kokoro'], rows 1, capability key set unchanged.
- **With the flag on: style='empathetic' routes to the expressive engine, is reported in direction_applied, and audibly differs from neutral (rendered WAVs or measured prosody difference).** — **Partially met**
  standard: direction_applied={'style': 'empathetic', 'exaggeration': 0.62}; turbo: applied={}, ignored=['style'] (it cannot honestly claim the style). Measured prosody on the committed renders: 0.5 -> 1.0 is -17.6% duration / +18.6% RMS with waveform correlation 0.011, bounding 0.62 at about -4.2% / +4.5%. The exact 0.62 render with real weights remains impossible on this host (Intel macOS, no torch 2.6.0 x86_64 wheel, no docker); the slow test now renders 3x per condition and compares means. Reviewer accepted this as environment-limited in the previous round.
- **style='whisper' appears in direction_ignored with a warning and the request succeeds with 200.** — Met
  test_whisper_on_a_chatterbox_voice_succeeds_with_200 and test_whisper_is_reported_as_ignored_not_approximated pass.
- **Removing the model directory produces a clear, actionable error and a Kokoro fallback, not a crash.** — Met
  200, engine=kokoro, direction_ignored contains 'style', warning contains the model path, download_chatterbox.py and CHATTERBOX_ENABLED=false. Both variants and both fallback routes covered.
- **The engine is not imported or loaded when the flag is off; startup works with torch uninstalled.** — Met
  Flag-off subprocess: adapter and torch absent from sys.modules. Flag-on + import-blocked subprocess: /health 200, engines ['kokoro','chatterbox'], torch absent.
- **Licence and attribution metadata is present in the registry and surfaced by /api/tts/voices.** — Met
  Both records MIT / 'Chatterbox (Resemble AI)'; /api/tts/voices returns them; the keys-on-every-row decision is documented in the endpoint docstring.
- **Both variants can download, verify and load.** — Met
  Per-variant end-to-end test plus an ad-hoc run against the real upstream file set: turbo allow_patterns=9, 9 files created upstream, manifest missing=[], ensure_available licence=MIT, required_total_bytes=2,987,680,596 == constant.
- **pytest and ruff check pass; record exact commands and exit codes.** — Met
  make test -> exit 0, 753 passed / 15 deselected. make lint -> exit 0, All checks passed (ruff 0.15.5).

---
Generated by core.ProcessAITicket@1.4.8
