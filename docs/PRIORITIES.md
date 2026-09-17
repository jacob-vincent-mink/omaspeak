# Priorities and scope decisions

This is the proposed disposition of all next steps in the model assessment and
shared catalog contract. It does not claim to inventory every possible future
feature. Open issues and reviews were checked on 2026-09-15; neither repository
had additional open issues or review comments to incorporate. These are
recommendations for review, not decisions already accepted or declined by the
maintainer.

This document supersedes the earlier assessment's ordering. Source research
remains in [the assessment](MODEL-SUPPORT-PLAN.md), with common mechanics in
[the catalog contract](MODEL-CATALOG-CONTRACT.md). Merging this planning PR
does not itself authorize implementation of every optional capability.

## Product boundary

Omaspeak turns supplied text into local speech with a selected voice, then
plays it or writes audio. It owns synthesis, model/voice selection, cancellation
and daemon lifecycle. The consumer owns dialogue, text generation, reading UI
and the sequencing of roles. A voice option belongs here only when it improves
that existing text-to-speech operation without turning setup into a studio.

Every supported backend needs a complete downloadable default with a usable
preset voice. Keep Supertonic as the baseline while proving alternatives;
cloning or a user recording can never be required to finish ordinary setup.
Untested device combinations remain experimental, with no silent fallback
presented as successful accelerator support.

Priority meanings: **P0** completes the support contract; **P1** improves the
core service; **P2** is an optional experiment with a promotion gate. **Defer**
has no scheduled implementation until its trigger occurs. **Decline** excludes
the feature from the proposed product scope. Rows are in execution order and
use relative S/M/L effort, not calendar promises. They are independently
reviewable, subject to their dependencies.

## Ranked implementation backlog

| Rank / ID | Decision | Next step and owner | Dependency / effort | Completion or stop condition |
|---|---|---|---|---|
| 1 / S01 | P0: pursue | Establish representative synthesis/voice baseline and regression budgets | None / M | Notifications, replies, names/numbers and long text; listening checks, load/latency/memory and provenance recorded |
| 2 / S02 | P0: pursue | Complete catalog metadata, compatibility resolver and default-coverage checks | S01 / M | Both existing backends and all runtimes have complete default declarations; report missing hardware evidence explicitly |
| 3 / S03 | P0: pursue | Friendly compatible-model setup list, sizes/details, provider-family discovery and stable CLI output | S02 / M | Catalog and actual provider agree; no unsupported model offered as runnable; offline listing and narrow-terminal scrolling work |
| 4 / S04 | P0: pursue | Preserve download/activation rollback, license handling and caches; add missing locking/cancellation/disk preflight | S02 / M | Failed download, probe or cache compilation leaves active config/model intact; only fill real gaps in current implementation |
| 5 / S05 | P0: pursue | Qualify existing defaults across advertised device/runtime pairs, including HIP evidence | S01–S04 / L, hardware dependent | Release evidence separates supported from experimental; complete preset voice and resource checks for every promoted pair |
| 6 / S06 | P1: pursue | Core request lifecycle: prompt stop, bounded queues, request IDs and playback ownership | P0 / M, joint integration | Cancellation/crashes release playback and Omawake pause ownership; queued requests cannot leak state |
| 7 / S07 | P1: pursue | Stream Supertonic output if the pinned native API proves suitable | S01, S06 / M | Better first-audio latency, bounded chunks/backpressure, no seams/underruns, prompt stop and unchanged file-export correctness; retain offline path if not proven |
| 8 / S08 | P1: pursue | Generalize only what a second preset family needs: named voices, language, typed adapter/request and legacy-ID migration | S02 / M | Supertonic integers retain their identity; no unsupported option is silently ignored; avoid implementing future clone/design fields yet |
| 9 / S09 | P1: pursue | Add Kokoro GGUF with complete English frontend resources and voice preview | S01, S08 / M | Fresh setup produces good speech; compare pronunciation/latency and keep as optional unless it earns default status |
| 10 / S10 | P1: pursue | Extend qualified voice/language choices and catalog maintenance tooling | S09 / M–L | Only tested languages with their dictionaries/resources; reproducible pinned imports, offline import and asset health checks |
| 11 / S11 | P1: pursue when needed | Search/filter voices/models, group variants, resume/reuse verified downloads | Catalog growth / S–M | Controls solve actual list/download scale; no large studio UI or speculative capability filters |
| 12 / S12 | P2: bounded experiment | Qwen3 CustomVoice for per-request delivery instructions | S01, S08 / M–L | Clear expressive-speech benefit within declared latency/memory budgets; add one typed style/instruction path, not a generic effects surface |
| 13 / S13 | P2: bounded experiment | Qwen3 0.6B Base cloning with one saved local reference profile | S08; S01 evidence / L | Demonstrated voice fidelity and repeated-request performance; preset setup remains complete; stop if persistent complexity outweighs use |
| 14 / S14 | P2: bounded experiment | Intel Kokoro IR native adapter | S09, concrete Intel need / L | First CPU file synthesis, then GPU/NPU separately; promote only for measured benefit and complete resources; never replace working Supertonic on availability alone |

The first release stops after S01–S05. Core speech improvements S06–S11 can
ship separately. Streaming and cancellation outrank expressive models because
they improve every assistant reply. S12–S14 are independent optional
experiments, not mandatory release work. Hardware access may block individual
qualification rows without turning them into assumed support.

## Deferred and conditional work

| ID | Disposition | Item | Trigger / boundary |
|---|---|---|---|
| S15 | Defer | VoiceDesign model and voice-creation command | Presets and optional reference cloning demonstrably cannot satisfy assistant voice choice; separate authoring proposal before implementation |
| S16 | Defer | Qwen Base 1.7B | 0.6B fails a measured voice-quality requirement and larger variant earns its cost |
| S17 | Defer | PocketTTS cloning alternative | Need for smaller/faster cloning after S13; resolve exact model/voice terms and dependencies before a bounded comparison |
| S18 | Defer | Community Qwen OpenVINO pipelines | Advanced voices have proven users and measured Intel performance need; new autoregressive orchestration must justify long-term maintenance |
| S19 | Defer | OmniVoice, Chatterbox and other TTS families | Specific quality/language gap remains; assess only their TTS path and one candidate at a time |
| S20 | Defer | Dedicated provider editions for optional families | Measured package/dependency cost requires them; first keep one complete tested provider per supported runtime with a minimal family set |
| S21 | Defer | Persistent exported embeddings, multi-voice prompt-cache tuning | Repeated cloned-voice requests show a bottleneck; model-specific state must have versioned invalidation |
| S22 | External tooling | Generate wake corpora by calling ordinary `say`/file export | Omawake's maintainer scripts own labels, augmentation, evaluation and provenance; no dataset-generation subsystem here |
| S23 | Conditional maintenance | Project-owned conversions, resources or mirrors | Canonical compatible downloads are insufficient; maintain recipes/manifests in Git and immutable large artifacts elsewhere |
| S24 | Defer | Signed catalog refresh / dynamic remote catalogs | Bundled catalog updates fail a concrete availability need; prefer reviewed release updates first |

Cloning and delivery style remain plausible optional TTS inputs, not required
product commitments. Review can decline either S12 or S13 without compromising
defaults, setup, preset voices, playback or the core adapter design. Voice
design is deliberately outside the active queue.

## Recommended explicit declines

| ID | Decline in Omaspeak | Reason / destination |
|---|---|---|
| S25 | Dialogue engine, role sequencing and dedicated podcast/multi-speaker generation | Consumers can sequence ordinary text requests with selected voices; no VibeVoice dialogue-model work in this roadmap |
| S26 | Standalone forced alignment, captions or synchronized reading UI | Reading/export applications own alignment/presentation; a separate aligner dependency is not required to speak text |
| S27 | Audio-to-audio voice conversion, redubbing and recording editor | These are media-authoring workflows, not text-to-speech |
| S28 | Denoise, source separation, music/SFX/video generation | No direct contribution to the supplied-text speech contract |
| S29 | ASR, microphone capture/enrollment recorder, speaker recognition or authentication | Accept caller-provided text/reference files; other applications own recording and identity |
| S30 | LLM reasoning, full speech-to-speech assistant and full-duplex echo cancellation | The harness/audio integration owns conversation and echo management; expose cancellation/playback lifecycle only |
| S31 | Voice marketplace, gallery, multi-model arena or elaborate design studio | A small preset/profile selector and preview suffice for the service's purpose |
| S32 | All-model audio.cpp bundle, arbitrary task runner or unfiltered HF browser | Model availability is not a reason to broaden product scope or maintenance |
| S33 | Shared catalog microservice, generic plugin framework or new audio daemon | Existing process-isolated adapters and embedded data are enough |

Declines remove work from the implementation queue. Reopening one requires a
new explicit product proposal, not simply a model becoming available upstream.

## Small review and delivery units

1. **Coverage and setup:** S01–S04 with default checks and migration/rollback
   tests, followed by release qualification S05.
2. **Responsive playback:** S06/S07 with first-audio/stop evidence and cross-app
   pause ownership tests.
3. **Second preset family:** S08/S09 with a complete Kokoro download/resource
   recipe and side-by-side listening results; extend languages only afterward.
4. **Maintainability:** S10/S11 as the curated list grows.
5. **Optional experiments:** separate reports for S12/S13/S14; each concludes
   promote, defer or decline. Do not prebuild a voice-authoring framework for
   features that review may reject.

Documentation checks cover this update only. Implementation acceptance is in
the rows above and the companion assessment; no new native-model or hardware
validation is claimed here.
