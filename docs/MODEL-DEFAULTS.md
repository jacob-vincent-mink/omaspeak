# Model defaults: first implementation slice

Implements default selection and compatibility from approved S02 priorities
([planning PR](https://github.com/jacob-vincent-mink/omaspeak/pull/1)). The
catalog retains Supertonic 3 and its existing pinned GGUF/ONNX artifacts.

`setup all` without `--model` keeps the selected compatible catalog profile or
resolves the selected backend's default. Explicit `--model` takes precedence.
Switching between audio.cpp and OpenVINO now switches the model representation
as well; OpenVINO no longer retains a GGUF-only model configuration. Switching
between audio.cpp runtimes preserves the selected voice. Explicit model
activation clears foreign provider paths/options and chooses a compatible
runtime, while retaining valid same-provider configuration.

The model picker uses the same compatibility checks as default selection and
shows friendly names. Existing IDs, numeric voices and schema-1 installation
manifests remain compatible. The native probe and synthesis checks remain the
authority for actual runtime/device readiness.

## Default coverage

| Backend | Default | Runtime contract |
|---|---|---|
| audio.cpp | `supertonic-3-gguf`, M1 | CPU, CUDA, Vulkan, HIP |
| Direct OpenVINO | `supertonic-3-openvino`, M1 | CPU, GPU, NPU |

Each has a complete model/voice manifest and the existing explicit license
acceptance flow. Supported adapter/device combinations are not new hardware
qualification claims; HIP has no new evidence in this slice.

## Baseline and promotion gates (S01)

[Versioned English cases](../benchmarks/model-defaults-cases.json) cover short
notifications, replies, names/numbers, punctuation and long text. Exercise M1
and F1 to check voice switching. Reuse this exact corpus/hash for reference and
candidate builds, recording provider/model hashes, device, threads and runtime
cache state. Preserve
[existing rc.3 evidence](../benchmarks/results/2026-09-15-rc3/RESULTS.md) as prior
single-sentence evidence, not a fresh representative quality assessment.

Setup-only changes must retain exact asset/voice selection and valid output,
with no missing words, incorrect voice, non-finite samples, truncation or state
leakage. Require no more than a 10% increase in warm p95 synthesis time and peak
memory over three repeated matched runs before accepting a performance
regression. Record cold load separately. Listening/independent transcription
checks are required before model promotion; matching PCM is not generally a
valid gate for stochastic synthesis. Streaming later gets separate first-audio,
underrun and cancellation budgets; this slice adds no streaming claim.

The corpus establishes reproducible inputs; listening, multilingual coverage,
repeated performance/memory runs and additional device evidence remain S01/S05
work. Provider-family checks and installer locking/cancellation/disk preflight are
now documented in [installer protections](INSTALLER-PROTECTIONS.md).

## Setup list and voice auditions (S03)

The terminal list now tracks its height as well as its width, keeps the
highlighted row visible, and shows a position counter. Arrow keys move through
the list; terminal resizing recalculates the viewport. Noninteractive catalog
output is unchanged.

In `omaspeak setup model`, choose an installed model to open its voice list.
Press **Space** to play or stop a sample of the highlighted voice; **Enter**
chooses it and **Escape** cancels. Moving to another voice stops the previous
sample. Full setup previews the pending runtime selection. Models awaiting
installation show an install-first message; preview never downloads assets or
accepts a model license on the user's behalf.

A preview uses the verified installed assets, a fixed short English sentence,
and the selected runtime with fallback disabled. Preparation and playback run
outside the TUI and remain cancellable. Private temporary WAVs are removed on
completion, cancellation or error. A 120-second bound covers preparation and
playback. Preview does not submit to or stop the daemon, save configuration,
replace `last.wav`, or activate a voice. Playback uses `pw-play`, falling back
to `aplay` when the first executable cannot be started. Errors stay in the list.

[Preview smoke evidence](../benchmarks/results/voice-preview/RESULTS.md) records
real OpenVINO CPU synthesis through the TUI with a silent test player.
This is the list-navigation portion of S03 plus the requested preset audition.
The subsequent family checks and installer guards complement this flow.

## Subsequent implementation

See [provider and installer boundaries](INSTALLER-PROTECTIONS.md) for the next
implemented slice, its tests and remaining limits. See also the
[corpus measurements](../benchmarks/results/2026-09-16-model-corpus/RESULTS.md).

[Paired release-build and NPU evidence](../benchmarks/results/2026-09-16-qualification/RESULTS.md)
adds three comparison rounds, broader NPU synthesis, process-group memory and a
reproducible blinded listening pack. Human ratings and power qualification remain
open; this does not promote additional hardware or expressive model families.

Model setup now starts on an enabled compatible row when the previous model is
incompatible with a newly selected runtime. Disabled and out-of-range selections
are rejected before voice selection or installation. CLI catalog output is unchanged.

[GB10 CUDA corpus results](../benchmarks/results/2026-09-16-cuda/RESULTS.md)
add current-code ARM64/CUDA file synthesis, independent transcription, memory and
whole-GPU telemetry. HIP remains pending because no test hardware is available.

[Completion tracker](COMPLETION.md) records the approved remaining sequence,
current acceptance status and external evidence gates.
