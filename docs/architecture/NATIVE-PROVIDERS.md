# Native provider migration

Omaspeak will remove ONNX Runtime before its first stable release. The Rust
application continues to own configuration, setup, discovery, daemon behavior,
audio output, and reporting. Inference is selected as one complete native
provider installation.

## Provider matrix

| Provider | Model form | Devices | Delivery |
| --- | --- | --- | --- |
| `audiocpp` | Supertonic 3 GGUF | CPU, CUDA, Vulkan; later HIP and Metal | A pinned, Supertonic-only CPU build is the default package provider. Setup can select a compatible complete external build for acceleration. |
| `openvino` | Official Supertonic ONNX graphs | Intel CPU, GPU, NPU | Optional external OpenVINO installation, loaded through its C API. |

Omaspeak does not assemble a runtime from unrelated core and provider
libraries. A provider path names one coherent installation. Setup discovers
and probes it without installing vendor software. Configuration is written
only after the probe and a real synthesis request succeed.

The first `audiocpp` implementation is pinned to upstream commit
`e9ff20042ec85af960a720368c6927cda19ad65f`, C ABI 0.1.0. It loads the published
C ABI dynamically and checks the ABI before creating any handle. It runs in an
isolated worker so a defective optional provider cannot take down the CLI or
daemon.

## Evidence for the decision

A local Supertonic-only CPU build of audio.cpp produced a 5.4 MiB unstripped
shared library (4.9 MiB stripped) with no ONNX Runtime dependency. The official
converted `supertonic-3-orig.gguf` is 454,072,836 bytes with SHA-256
`af814486a0bc9513fb36afabd9b1155ad14fb2c36a107ac6ffe62ea9adafb662`.

The file-only smoke phrase `Testing one, two, three.` synthesized at 44.1 kHz
in 719 ms on the Dell CPU with two threads. The 1.88-second result was
transcribed independently as `Testing 123.` No audio device was opened. This
one-shot measurement includes per-process initialization; persistent hot-run
measurements are an integration gate.

A persistent-session sweep on a 1.75-second phrase measured hot inference at
1,349 ms with one thread, 682 ms with two, 474 ms with four, and 351 ms with
eight. Twelve and sixteen threads were slower, so setup should suggest a
measured or topology-aware thread count rather than every logical CPU. The
312,784,196-byte F16 GGUF produced the same frame counts and independent
transcripts as the 454 MB original model. Its PCM correlation was 0.999989 for
M1 and 0.999731 for F1 at a fixed seed, while its eight-thread hot runs were
398-421 ms. Its SHA-256 is
`b312b57797d40ac5c09d915893dbdbaf6405b7dc043f544776c5c95712dff88c`.
F16 saves 31 percent of model bytes but needs a wider voice and language corpus
before becoming the default.

The existing direct OpenVINO implementation already runs all four official
Supertonic graphs without ONNX Runtime. It remains the Intel provider and keeps
NPU compilation in setup rather than first inference.

## Model and license boundaries

Omaspeak source remains MIT. audio.cpp is Apache-2.0 and must be accompanied by
its license and required notices when its library is packaged. Supertonic 3
weights, including converted GGUF files, remain under the upstream OpenRAIL-M
model license. Setup must show that license and record acceptance before a
download. A converted model is never described as MIT or Apache-2.0.

The default GGUF is downloaded from the audio.cpp model repository and pinned
by revision, byte size, and SHA-256. OpenVINO uses files downloaded from the
official Supertonic repository. No model archive is sourced from sherpa-onnx.

## Migration sequence

1. Add a supervised audio.cpp worker with bounded framed IPC, ABI discovery,
   model/session reuse, all ten Supertonic voice IDs, and file-only synthesis.
2. Add catalog metadata for the pinned GGUF and make CPU audio.cpp the default
   setup choice. Package its pinned Supertonic-only CPU library with Omaspeak.
3. Change runtime setup to choose a provider installation. Probe ABI, declared
   capabilities, requested device, model compatibility, and a short synthesis
   before saving.
4. Retain direct OpenVINO as an optional provider and verify CPU, Intel GPU, and
   Intel NPU. NPU setup must compile the complete shape plan into its cache.
5. Prove a complete audio.cpp CUDA build on GB10 and a Vulkan build on the Dell
   GPU. Record the library identity and actual device placement.
6. Remove `ort`, ORT plugin discovery, ORT configuration keys, and ORT release
   assets. Update packaging, setup, documentation, and tests in the same
   change; there is no legacy config migration before a stable release.

## Acceptance gates

- Default setup works from a clean user account using the packaged CPU provider
  and a model obtained by the guided downloader.
- `omaspeak say` works on demand with no systemd unit installed or running.
- A daemon reuses one model and session across requests; a failed optional
  worker produces an actionable error and no core dump notification.
- Voices M1-M5 and F1-F5 are selectable by stable name and numeric ID.
- File-only corpus comparison covers every voice and representative supported
  languages. It checks finite audio, sample rate, duration, transcript content,
  and gross acoustic drift against the current implementation.
- Cold load and hot synthesis latency, peak RSS, model bytes, and real-time
  factor are recorded for CPU, OpenVINO CPU, Intel GPU, Intel NPU, CUDA, and
  Vulkan where the hardware is available.
- Accelerated output must not show a material quality regression from the
  default CPU provider. Device placement must come from the selected runtime,
  not timing alone.
- The release binary has no load-time dependency on audio.cpp, OpenVINO, CUDA,
  or ONNX Runtime. The default installation includes one known-good CPU
  provider; optional providers are selected only when the user configures one.
