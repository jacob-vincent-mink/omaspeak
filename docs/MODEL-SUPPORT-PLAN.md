# Model and voice support assessment and plan

Assessed 2026-09-15 against omaspeak `86064f6`, omawake `af25056`, and packaged
audio.cpp `e9ff20042ec85af960a720368c6927cda19ad65f`. This is an assessment and
implementation plan, not a provider/model rollout. No candidate models were
downloaded or benchmarked, and user configuration was not changed.

## Review order and scope

Read [PRIORITIES.md](PRIORITIES.md) first for the ranked backlog, dependencies,
release cuts, deferred experiments and recommended explicit declines. It
supersedes the ordering below. This assessment preserves the technical research;
a capability appearing here is not a commitment to implement it.

## Recommendation

Retain Supertonic 3 as the working default across existing backends. Generalize
the audio.cpp adapter and voice representation, add Kokoro for preset-voice
choice. Prioritize responsive playback and cancellation before advanced voice
features. Qwen3 CustomVoice and Base are optional experiments for delivery style
and cloning; defer VoiceDesign, PocketTTS and broader authoring features under
the scope decisions in PRIORITIES.md.

For OpenVINO, preserve the existing official Supertonic ONNX path. Intel's
Kokoro IR is the strongest next investigation; downloadable Qwen3 IR is useful
research input but does not plug into the Supertonic adapter. Each new OpenVINO
family needs its own text/audio preprocessing and generation orchestration.

Use the companion [catalog and setup contract](MODEL-CATALOG-CONTRACT.md).
The paired Omawake worktree covers wake models and evaluation integration.

## Findings in the current code

| Area | Current behavior | Required change |
|---|---|---|
| `src/catalog.rs` | Two Supertonic profiles, with pinned files, notices, voices and acceptance policy | Add adapter-specific profiles and per-variant capabilities; avoid extending the flat Supertonic field list forever |
| `scripts/build-default-audiocpp-provider.sh` | Compiles only `AUDIOCPP_MODELS=supertonic` | Add qualified families to release providers or distribute explicit complete provider editions |
| `src/audio_cpp.rs` | Rejects `model.family != supertonic`; integer voices map to M1..F5; offline TTS only | Pass selected family, task/mode, named voice and typed conditioning through the worker |
| `src/voices.rs` | Enumeration is Supertonic-specific | Introduce model-specific preset inventory and saved voice profiles |
| `src/backend.rs`, `src/engine.rs` | `generate(text, speed, voice)` assumes a narrow request | Introduce a synthesis request; retain current CLI/config compatibility |
| `src/main.rs` | Setup already has model selection and runtime validation, but defaults/activation are family-specific | Centralize resolution and share the compatibility view with CLI and setup |
| `src/supertonic.rs` | Direct OpenVINO pipeline knows Supertonic's four graphs and voice styles | Keep that adapter; a generic IR path is not sufficient for Kokoro or Qwen |

The pinned audio.cpp C ABI already provides registry family enumeration,
model task/mode checks, language and option introspection, speaker-reference
and style capability checks, reference PCM, named voice IDs, style setters and
stream events. Our binding currently uses only a subset. See the
[pinned header](https://github.com/0xShug0/audio.cpp/blob/e9ff20042ec85af960a720368c6927cda19ad65f/include/audiocpp.h).
Capability introspection helps validate requests but does not replace tested
per-family mappings or quality checks.

## Default coverage

| Backend / runtime | Default | Artifact authority | Status / rule |
|---|---|---|---|
| audio.cpp / CPU | Supertonic 3 original-precision GGUF, M1 | [audio.cpp conversion](https://huggingface.co/audio-cpp/audio.cpp-gguf), original [Supertone archive](https://huggingface.co/supertone-oss-archive/supertonic-3) | Keep existing pin and acceptance flow |
| audio.cpp / CUDA | Same GGUF + preset | Same | Keep; prior GB10 evidence exists |
| audio.cpp / Vulkan | Same GGUF + preset | Same | Keep; validate release/provider/device pair |
| audio.cpp / HIP | Same GGUF + preset | Same | Intended default; do not claim qualified/recommended without AMD evidence |
| Direct OpenVINO / CPU | Supertonic 3 official ONNX graphs + M1 style | Supertone archive | Keep |
| Direct OpenVINO / GPU | Same complete ONNX profile | Same | Keep with placement/cache probe |
| Direct OpenVINO / NPU | Same complete ONNX profile | Same | Keep with complete shape-cache/placement proof |

The existing GGUF download is 454,072,836 bytes. The ONNX profile includes four
graphs, text metadata and ten voice styles; use the manifest's sum rather than
only the largest weight file. NPU disk usage also includes compiled caches:
the prior report recorded about 781 MB for its ten blobs on that machine.

[Existing rc.3 evidence](../benchmarks/results/2026-09-15-rc3/RESULTS.md) includes
working CPU, Intel GPU/NPU and CUDA paths with Supertonic. Intel warm p50 was
1,658 ms for GGUF CPU and 405/135/64 ms for OpenVINO CPU/GPU/NPU on one sentence.
Formats and execution stacks differ; these are not isolated hardware speedups
or broad voice-quality results. Preserve this working baseline while expanding.

Every new backend must ship with a normal speak-text default that works without
a user's reference recording. Clone-only models can be optional capabilities,
not the only default. Supertonic's existing model-license acceptance remains
explicit; a free download is not equivalent to unrestricted license terms.

## Candidate list

| Priority | Model / artifact | Why it belongs | Integration and promotion conditions |
|---|---|---|---|
| P1 | Kokoro 82M Q8_0 GGUF, initially English `af_heart` | Many preset voices in a compact model; good candidate for assistant replies and reading text | Build `kokoro_tts`; named voices, matching language and phonemizer resources; listening/performance comparison against Supertonic |
| P1 | Qwen3-TTS 1.7B CustomVoice GGUF | Named speakers and instruction-based delivery for assistant responses | Build `qwen3_tts`; correct speaker option, `instruct`, generation limits and full speech-tokenizer assets |
| P2 | Qwen3-TTS 0.6B Base GGUF, then 1.7B if justified | User-supplied voice cloning and reusable assistant voices | Reference PCM/transcript handling, cached voice prompts, complete dependency manifest and latency/similarity evaluation |
| P2 | Qwen3-TTS 1.7B VoiceDesign GGUF | Create a voice from a description without a recording | Separate `vdes` task and package; save an approved preview as a reference for Base cloning |
| P2 | PocketTTS English Q8_0 + Alba embedding | Smaller cloning candidate and reusable preset | Resolve original-model access/terms and separate voice rights; pinned package includes `embeddings/alba.safetensors`; qualify API streaming rather than infer it |
| P2 spike | Intel Kokoro 82M INT8 IR | Canonical OpenVINO alternative with preset voices | New `kokoro-openvino` adapter, native text frontend, voice assets, runtime compatibility and per-device qualification |
| P3 research | Community Qwen3-TTS INT8 IR Base/CustomVoice/VoiceDesign | Potential Intel acceleration for advanced voices | New autoregressive pipeline, tokenizer/codec and speaker encoder handling; exact export provenance and CPU/GPU/NPU experiments required |

Sources and qualifications:

- [Kokoro GGUF files](https://huggingface.co/audio-cpp/audio.cpp-gguf/tree/main/Kokoro-82M-GGUF)
  list roughly 190 MB Q8 and 212 MB BF16. The
  [pinned audio.cpp Kokoro docs](https://github.com/0xShug0/audio.cpp/blob/e9ff20042ec85af960a720368c6927cda19ad65f/docs/models/kokoro_tts.md)
  describe 54 voices and embedded G2P tables. Several languages require eSpeak
  runtime/data; Japanese also requires MeCab/UniDic. Do not advertise all
  languages as ready merely because voice names exist. Original weights are
  at [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M).
- [Pinned Qwen3 integration docs](https://github.com/0xShug0/audio.cpp/blob/e9ff20042ec85af960a720368c6927cda19ad65f/docs/models/qwen3.md)
  distinguish Base cloning, CustomVoice TTS, and VoiceDesign `vdes`. They list
  offline TTS modes; upstream model streaming claims do not establish that our
  pinned C ABI integration can stream these variants. Check the package spec
  for the main model, speech tokenizer and sidecars together.
- [PocketTTS upstream](https://huggingface.co/kyutai/pocket-tts) currently displays
  CC-BY-4.0 and an access agreement, while the pinned audio.cpp package spec
  points at an ungated conversion. Resolve that discrepancy and the voice's
  own license before catalog promotion; do not infer permission from download
  reachability. It remains optional, not a required fresh-setup default.
- [Intel Kokoro IR](https://huggingface.co/OpenVINO/Kokoro-82M-int8-ov) specifies
  OpenVINO 2026.3.0+ and demonstrates Optimum/OVMS inference. OpenVINO 2026.4.0
  [announces NPU support](https://github.com/openvinotoolkit/openvino/releases/tag/2026.4.0).
  The pinned `kokoro-82m-openvino` catalog profile and its persistent GenAI
  worker are documented in the [integration evidence](KOKORO-OPENVINO-2026.4.md).
  The profile requires OpenVINO and GenAI 2026.4 or newer at activation.
- [Echo9Zulu Qwen3 Base IR](https://huggingface.co/Echo9Zulu/Qwen3-TTS-12Hz-Base-1.7B-INT8-OpenVINO)
  is a community conversion. Its card largely repeats the upstream Python
  workflow; inspect the actual graph contracts and converter before relying
  on it. Keep this outside the supported setup list until a complete pipeline
  works in our process-isolated architecture.

Do not add a model simply because it is large, new, or has many capability tags.
OmniVoice, Chatterbox and other expressive families can be later comparisons
if the first set leaves a concrete quality or language gap. A small useful
catalog is easier to qualify than the entire audio.cpp inventory.

## Voice capabilities mapped to use cases

| Capability | Natural product behavior | Model path / constraints |
|---|---|---|
| Preset voices | Choose a pleasant default for notifications, assistant replies or reading | Supertonic now; Kokoro next. Preview with the same sentence, normalized playback level and a stop control |
| Style control | “Calm”, “concise announcement”, “warm narrator” delivery for a request | Qwen CustomVoice instructions first; translate presets per model instead of promising universal emotion/pitch/speed support |
| Voice cloning | Save a voice from a user-selected recording, reuse it by name | Qwen Base; PocketTTS after qualification. Local reference audio/transcript and per-model prompt cache |
| Voice design | Describe an assistant's voice, generate previews, save one | Qwen VoiceDesign; saving text alone may not preserve identity, so retain preview audio/seed/provenance and clone from it |
| Streaming synthesis | Start speaking assistant output sooner and stop promptly | Supertonic is the first candidate; validate C ABI stream behavior, chunk boundaries and cancellation before promising latency |
| Multiple voices / dialogue | Read a conversation or a multi-role summary aloud | Initially sequence ordinary utterances with named profiles; investigate VibeVoice only if coherent long-form multi-speaker generation is needed |
| Alignment | Highlight words while reading or produce captions for exported speech | Optional post-generation aligner with known text; account for extra model/load cost |
| Voice conversion | Revoice an existing recording while retaining its timing | Possible later explicit conversion command, e.g. Chatterbox; weak fit for routine text notifications |
| Cloning/design/style for Omawake | Generate varied positives and confusing negatives for wake evaluation | Offline companion workflow with generator provenance; synthetic data supplements real recordings |

Music/SFX/video generation, source separation for media production and full
speech-to-speech conversation have weaker fit here. Keep Omaspeak a speech
service; the agent/harness owns conversation reasoning. Useful experiments can
be separate tools without expanding the ordinary `say` flow.

## Request and voice design

The full optional design is described here for review, not for up-front
implementation. Initially add only text, language, preset voice and rate; add
reference/style fields only if their corresponding experiments are promoted.

A later `SynthesisRequest` could include text, language, named voice/profile, speaking
rate, optional instruction/style, optional reference audio/transcript, seed and
bounded generation options. Resolve it into a typed family adapter. Keep
Supertonic's steps/language handling in its adapter rather than sending its
options to every model. Unsupported explicit options should produce an
actionable error, never silently disappear.

Add saved `VoiceProfile` records independent of model downloads: stable ID,
display name, kind (preset/reference/designed), compatible model/variant,
language, reference path/hash/transcript where needed, seed/design description,
and artifact provenance. A speaker embedding is model-specific; do not imply
it transfers across families. Retain original reference audio for regeneration
only as configured by the user. Deleting a profile also clears its owned prompt
caches. Use user-selected recordings with permission, and keep samples local.

Migrate existing numeric voice IDs through the selected Supertonic profile
(0 -> M1, etc.). Do not reinterpret `0` as a different speaker after a model
switch. Persist a compatible default and make the new voice visible in setup.

Illustrative future commands, not implemented by this plan; clone/style remain
optional and the design command is deferred:

```text
omaspeak voices list --model kokoro-82m-q8
omaspeak voices create my-voice --reference sample.wav --transcript "..."
omaspeak voices design narrator --description "A warm, measured adult voice"
omaspeak say "Your build finished." --voice narrator --style calm
```

IPC needs request IDs, cancellable streaming, backpressure and bounded PCM
chunks. Preserve hot sessions and worker isolation. Avoid loading every model
to populate setup: use catalog metadata, then probe only the selected model.
Voice previews and cloning must not leave old request conditioning in the next
request. Measure first-audio latency separately from total generation time.

## Technical implementation notes (subject to priorities)

1. **Catalog and provider discovery.** Implement the shared contract, default
   coverage and filtered setup. Expand the packaged provider only for families
   entering support; probe registry families on external providers. Preserve
   Supertonic config/IDs and atomic installation/activation.
2. **Preset model generalization.** Add named voices, typed requests and family
   adapters, then Kokoro. Package/pin necessary English phonemizer resources.
   Prove clean setup, existing Supertonic requests and voices still work,
   language mismatches fail clearly, and missing dependencies are visible.
3. **Style, clone and design.** Add Qwen CustomVoice, Base and VoiceDesign as
   separate profiles, complete package manifests, typed conditioning and saved
   voice profiles. Prove model/voice switching, cancellation, prompt caching,
   deterministic seed handling where supported and no cross-request leakage.
4. **Streaming.** First qualify Supertonic's actual pinned C ABI path. Measure
   time to first audio, underruns, chunk seams and stop latency with long text;
   preserve WAV export behavior. Coordinate playback ownership with Omawake;
   true barge-in is a separate echo-handling task.
5. **OpenVINO Kokoro spike.** First file synthesis from pinned Intel IR, then
   native text frontend/voice loading, CPU baseline, GPU and NPU separately.
   If an adapter is promoted, its initial default is a complete pinned Kokoro
   profile with `af_heart` and required resources. Retain Supertonic as the
   OpenVINO default until quality, placement and latency justify a switch.
6. **Optional candidates.** Resolve PocketTTS terms/assets, then compare it
   with Qwen cloning. Investigate Qwen IR only after the native baseline proves
   useful enough to justify a substantially more complex OpenVINO pipeline.

Qualification uses short notifications, long replies, numbers, acronyms, names,
punctuation, multilingual text and repeated voice changes. Compare listening
quality and pronunciation as well as ASR round-trip checks; measure speaker
similarity/style adherence for clone/design, cold/warm load, first-audio and
total latency, RSS/VRAM and actual device placement. Existing single-sentence
evidence is a baseline, not adequate for new model promotion.

No application tests were run for this documentation-only assessment. Model
availability and primary documentation were checked through web browsing;
source contracts were inspected in the local pinned audio.cpp checkout. New
asset hashes, native executions and hardware qualification remain explicit
implementation gates.
