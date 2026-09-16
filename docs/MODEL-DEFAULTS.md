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
work. S03 provider-family discovery and larger-list UI, and S04 installer
locking/cancellation/disk preflight, remain separate implementation slices.
