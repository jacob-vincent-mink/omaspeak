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

Accelerator performance and placement results will be published only after
fresh proof runs against this architecture.
