# Model support completion goal

Tracking date: 2026-09-16. This implements the user-approved priorities and
subsequent completion sequence. The historical assessment is not a claim that
all features or hardware are qualified.

## Goal and completion rule

Finish release coverage/setup and its outstanding acceptance gates, then deliver
core lifecycle reliability, gated streaming, measured preset/model alternatives,
pinned catalog maintenance and a first qualified non-English profile per app.
Every delivery has tests or a reproducible evidence report and a reviewed PR.
Do not mark quality gates passed from synthesis validity or ASR transcripts alone.
Do not promote hardware from runtime compatibility or one successful inference.

HIP remains pending and unqualified because hardware is unavailable. It is not
a release blocker for explicitly qualified combinations. Optional experiments
(cloning, delivery instructions, enrollment, additional native conversions) are
outside this goal; deferred and declined scope stays outside the active queue.

## Ordered joint delivery sequence

1. Reconcile acceptance status, evidence and supported versus experimental
   device claims. Audit fresh setup, activation and cache-failure rollback.
2. Finish available release checks. Obtain human listening ratings, real-room
   wake recordings and controlled resource/power measurements before closing
   their gates. Record unavailable evidence explicitly; continue independent work.
3. W10/S06: bounded requests, cancellation, request identity and owned playback
   pauses. Nested owners, client disconnects and crashes must not leak state,
   resume another owner's pause or let TTS retrigger wake actions.
4. S07: inspect the pinned API, then test incremental Supertonic output. Promote
   only with improved first-audio latency, bounded buffering, prompt cancellation,
   no seams/underruns and correct file export. Record a stop decision otherwise.
5. S08/S09: preserve legacy Supertonic voice identity while introducing only the
   named-voice/adapter fields needed by Kokoro. Pin all frontend/voice resources,
   provide preview and fresh installation proof, and compare speech quality.
6. W08: compare Moonshine Small/Medium with Tiny on identical corpora and declare
   quality/resource tradeoffs. Keep only variants that justify their footprint.
7. W07/S10: reproducible pinned catalog checks/imports, offline import and asset
   availability checks that never silently replace user pins.
8. W09/S10: one explicitly tested non-English profile per app, with language
   reaching inference and status, complete resources and language-specific evidence.
   Use benchmark and language requirements to choose the candidate before download.
9. W11/W22/S11: filters and transfer resume only when actual catalog/download
   scale warrants them. Record the decision; verified reuse must preserve hashes.

Each item ends with an implementation and acceptance result, or its specified
promote/defer/stop decision. Human-dependent gates remain open until evidence
arrives; missing evidence is not a pass. Core work can proceed while collecting
those inputs, without declaring the first release fully qualified.

## Current external inputs

- Human ratings of the existing English listening packs (intelligibility,
  pronunciation, missing words, naturalness and voice consistency).
- Independent real-room wake positives, human near-matches, distance and acoustic
  playback/TV/music recordings with labels and provenance.
- A controlled measurement window and readable energy counters for attributed
  resource/power comparisons. Shared-host GPU telemetry is not application power.
- Spanish is selected by the maintainer for the first non-English profile in
  both apps. Independent Spanish wake recordings and listening evaluation remain
  required before promotion. No blanket multilingual claim.

## Omaspeak acceptance status

| IDs | Status | Evidence / remaining acceptance |
|---|---|---|
| S01 | Partial | Versioned English corpus, matched seed output, independent transcripts and listening packs; human ratings and controlled resource gates remain open |
| S02/S03 | Implemented | Complete defaults, compatibility/family checks, stable CLI and scrolling; installed voices support cancellable Space-key preview |
| S04 | Implemented; acceptance mapped | [Installer audit](INSTALLER-PROTECTIONS.md) maps fresh setup, failed cache/probe, corruption, cancellation, lock and disk-budget checks; documented limits remain |
| S05 | Partial | CPU/GPU/NPU/Vulkan/CUDA evidence at differing depths; GB10 produced 36 valid CUDA WAVs with verified placement; HIP pending |
| S06 | Implemented; qualification limits recorded | [Bounded lifecycle](REQUEST-LIFECYCLE.md) adds admission, request cancellation, supervised synthesis and frozen config; [playback ownership](PLAYBACK-OWNERSHIP.md) covers speech/previews. CPU native evidence recorded; device timing and acoustic validation remain open |
| S07 | Evaluated; defer at current pin | [Native streaming probe](../benchmarks/results/2026-09-16-streaming/RESULTS.md) demonstrates early long-form PCM and exact export parity, but full-utterance retention fails the fixed buffering gate; keep offline output |
| S08 | Done | [Named selection](NAMED-VOICES.md) preserves all legacy IDs and supports config/IPC names |
| S09 | Done, evidence partially open | Kokoro 82M pinned ([KOKORO.md](KOKORO.md)); fresh-install + voice routing proven in-process; [paired latency recorded](../benchmarks/results/2026-09-16-kokoro/RESULTS.md); listening pack and human ratings still open |
| S10 | Tooling done; language queued | [Pinned URL health check](CATALOG-MAINTENANCE.md) (`setup model --check-urls [--url-prefix]`) with 18/18 healthy pinned URLs; offline-import diagnostics name expected/actual values; qualified non-English profile remains queued |
| S11 | Conditional | Search/filter and transfer resume only when catalog/download scale warrants them |

Current evidence: [defaults and gates](MODEL-DEFAULTS.md),
[installer protections](INSTALLER-PROTECTIONS.md),
[paired and NPU qualification](../benchmarks/results/2026-09-16-qualification/RESULTS.md),
[CUDA qualification](../benchmarks/results/2026-09-16-cuda/RESULTS.md).
Valid WAVs and independent transcription are useful evidence, but do not complete
human quality review. Uncontrolled timings do not pass a controlled regression gate.
