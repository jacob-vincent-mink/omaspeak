# Omaspeak 0.0.1 release candidate

This candidate introduces the greenfield native-provider architecture:

- audio.cpp GGUF synthesis is the default and ships as a small CPU provider;
- complete external audio.cpp builds support CPU, CUDA, Vulkan, and HIP/ROCm;
- direct OpenVINO remains available for Intel CPU, GPU, and NPU;
- guided setup discovers exact provider paths and requires model-backed
  file-only proof before final activation;
- NPU model-cache compilation and a fresh-process cache-hit proof happen during
  setup, using OpenVINO's standard cache for the validated 256-frame plan;
- ordinary setup does not install or start a systemd service;
- the Rust executable has no load-time dependency on an inference runtime.

Fresh file-only proof on an Intel Core Ultra X7 358H verified every direct
OpenVINO graph on CPU, Arc B390 iGPU, and NPU. Hot synthesis real-time-factor
p50 was 0.0996, 0.0358, and 0.0241 respectively; the packaged audio.cpp CPU
baseline was 0.4314. The NPU imported ten setup-prepared cache blobs in a fresh
process before inference. Whisper Base.en recovered the exact input sentence
from all four outputs. See `benchmarks/results/2026-09-15-native-providers` for
the protocol, cold timings, limits, and machine-readable results.
