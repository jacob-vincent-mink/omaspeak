# Local streaming and Paradee survey — 2026-10-07

These are cold debug-build checks on the development machine, with installed local
models and a silent raw-PCM player. They prove early delivery, full captured PCM
matching the saved WAV, and process supervision; they are not listening tests or
comparative model benchmarks. Input was roughly 700 characters. NPU Supertonic
used prepared temporary compiled caches. No installed configuration was changed.

| Provider | First PCM (s) | Complete (s) | PCM equals WAV |
| --- | ---: | ---: | --- |
| audio.cpp Supertonic CPU | 7.48 | 19.82 | yes |
| OpenVINO Supertonic CPU | 4.59 | 13.25 | yes |
| OpenVINO Supertonic GPU | 4.91 | 12.60 | yes |
| OpenVINO Supertonic NPU | 6.18 | 7.54 | yes |
| OpenVINO Kokoro CPU | 4.81 | 11.56 | yes |
| OpenVINO Kokoro NPU | 3.17 | 6.36 | yes |

Machine-readable observations: [local-streaming.json](local-streaming.json).
Supertonic CPU/NPU streamed exports also matched fixed-seed offline generation;
GPU did not match that separate run. The GPU measurement used a pre-existing
FP16 precision experiment that is excluded from this independent audio branch;
it does not qualify the branch's default FP32 GPU profile. Kokoro uses bounded text segments, which
can change prosody relative to a single offline request. Kokoro GPU and GGML
accelerators were not hardware-qualified here.

## Paradee probes

Immutable Hugging Face revision: `8f34b01ef8adcb0bec470b89bf2fd62a7b2369e6`.
FP32 SHA256: `77b8bb28caf3dddda0cc61d16b700febc187ee6336c695d7865b1e77e452ee11`.
INT8 SHA256: `60e8f8a1bc7c546488154e9d99ecac6e9c50baf3f4b684c5b0de48ea03b698eb`.

Installed OpenVINO 2026.4 imported and generated finite audio from the FP32
model on CPU. INT8 CPU compilation failed on an unknown internal element type,
even after static input shaping/type inference. GPU required FP32 precision plus
expanding two linear Resize operations from rank 3 to rank 4; this prototype
compiled and generated finite audio. A seeded ONNX reference comparison had
matching output dimensions and maximum absolute sample difference 0.0010426.
These manual phoneme probes do not qualify pronunciation or perceptual quality.

The dynamic INT8 NPU probe crashed inside the installed Intel NPU compiler
(`libopenvino_intel_npu_compiler.so`, offset `0x13a7c3c`), after unbounded shape
diagnostics. System coredump metadata identifies compilation, not application
playback, as the failing phase. No NPU support is claimed.

The isolated ONNX Runtime reference environment is not an application dependency
or delivered runtime. Integration should use OpenVINO, with a qualified native
Paradee frontend. See the [survey](../../../docs/CLOUD-BACKENDS-AND-PARADEE.md).
Experimental probe sources live in `benchmarks/tools/paradee-openvino-probe.rs`
and `benchmarks/tools/paradee-graph-compatibility.py`; the latter references the
isolated `/tmp/paradee-survey` assets and requires developer-only Python ONNX tools.
Neither is a production model conversion or packaged runtime.

## Automated verification

The original combined-worktree checks included separate consumer-events/D-Bus
work. That work is excluded from this independent audio branch. See the
[cloud verification record](../2026-10-07-cloud/RESULTS.md) for branch checks.
