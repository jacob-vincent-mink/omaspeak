# Omaspeak 0.1.0

The new `kokoro-82m-openvino` catalog profile uses OpenVINO GenAI 2026.4 or
newer on Intel NPU, GPU, and CPU. It pins the INT8 IR, phonemizer data, and
`af_heart` and `am_michael` voice embeddings. A persistent Python worker keeps
the model loaded across synthesis requests. The backend currently supports
speed 1.0 and needs an environment with OpenVINO, OpenVINO GenAI, and NumPy.
See [Kokoro setup](docs/KOKORO-OPENVINO-2026.4.md).

On a Panther Lake NPU, catalog installation and activation, `benchmark`, and
the normal `say` request worker produced non-silent 24 kHz WAVs. A Kokoro WAV
also reached an Omawake Whisper wake detection with the isolated 2026.4 C
runtime. Existing Supertonic and audio.cpp profiles remain available.

If an existing configuration pins a versioned 2026.3 OpenVINO library path,
update it with `omaspeak config unset backend.openvino_library` after upgrading
the system runtime; the unversioned runtime will then be discovered.

## Omaspeak 0.0.3

This release delivers the model-support roadmap on the native-provider
architecture (first packaged as 0.0.1, setup hardened in 0.0.2):


- audio.cpp GGUF synthesis is the default and ships as a small CPU provider;
- complete external audio.cpp builds support CPU, CUDA, Vulkan, and HIP/ROCm;
- direct OpenVINO uses one downloadable official Supertonic 3 model for Intel
  CPU, GPU, and NPU;
- setup fetches the four official ONNX graphs, configuration, Unicode indexer,
  and ten voice styles as individually pinned files after OpenRAIL-M acceptance;
- the model license and complete provenance manifest are verified in staging
  before the model directory is atomically published;
- guided setup discovers exact provider paths and requires model-backed
  file-only proof before final activation;
- switching back to the packaged CPU provider clears prior accelerator paths
  without asking for another runtime directory;
- `omaspeak say` reads omitted text from a pipe and reports missing interactive
  text immediately instead of waiting on the terminal;
- interrupting `omaspeak say` stops and reaps playback, including playback
  owned by an already-running daemon;
- successful standalone config, runtime, model, and speaker changes
  automatically restart an already-active optional daemon exactly once;
- optional systemd units remain runtime-neutral instead of retaining loader
  paths from the backend selected when the unit was installed;
- setup detects PCI accelerator candidates in CUDA, Intel NPU, Intel GPU, then
  Vulkan order and recommends the best candidate with a complete provider, or
  packaged CPU, while reserving readiness for Apply proof;
- runtime inventory reports an audio.cpp device as unverified until that model
  proof succeeds, rather than presenting ABI discovery as a failed device;
- NPU model-cache compilation and a fresh-process cache-hit proof happen during
  setup, using OpenVINO's standard cache for the fixed 256-frame plan;
- ordinary setup does not install or start a systemd service;
- the Rust executable has no load-time dependency on an inference runtime;
- a second speech family, Kokoro 82M: Kokoro-capable packaged provider
  (supertonic;kokoro_tts with statically linked eSpeak-ng), 54 named voices
  plus legacy numeric IDs, and 24 kHz output;
- the eSpeak-ng data package is catalog-managed, not shipped in the core:
  pinned release asset, downloaded/verified/installed into the model
  directory, with the GPL-3.0 notice carried in MODEL-LICENSE;
- `setup model --check-urls`: pinned-catalog URL health checks for both
  speech families, plus offline-import diagnostics naming expected and
  observed values;
- a Spanish speech profile: `model.language` reaches the native worker and
  daemon status (smoke-scale evidence; listening gates stay open);
- `status --json` reports the same `backend.requests` shape with and without
  a running daemon;
- Supertonic streaming was evaluated and deferred at the current pin.

The pinned official model was exercised with file-only synthesis on default
CPU, OpenVINO CPU/iGPU/NPU, and CUDA on an NVIDIA GB10. Device placement,
setup-time NPU caching, cold and warm timing, and a one-sentence intelligibility
check are recorded in the
[rc.3 hardware evidence](benchmarks/results/2026-09-15-rc3/RESULTS.md), with
the follow-up Intel iGPU run in the
[Vulkan qualification](benchmarks/results/2026-09-15-vulkan/RESULTS.md).
 Device placement,
setup-time NPU caching, cold and warm timing, and a one-sentence intelligibility
check are recorded in the
[rc.3 hardware evidence](benchmarks/results/2026-09-15-rc3/RESULTS.md), with
the follow-up Intel iGPU run in the
[Vulkan qualification](benchmarks/results/2026-09-15-vulkan/RESULTS.md).
