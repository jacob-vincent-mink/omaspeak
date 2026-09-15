# Omaspeak 0.0.1

This release introduces the greenfield native-provider architecture:

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
- the Rust executable has no load-time dependency on an inference runtime.

The pinned official model was exercised with file-only synthesis on default
CPU, OpenVINO CPU/iGPU/NPU, and CUDA on an NVIDIA GB10. Device placement,
setup-time NPU caching, cold and warm timing, and a one-sentence intelligibility
check are recorded in the
[rc.3 hardware evidence](benchmarks/results/2026-09-15-rc3/RESULTS.md), with
the follow-up Intel iGPU run in the
[Vulkan qualification](benchmarks/results/2026-09-15-vulkan/RESULTS.md).
