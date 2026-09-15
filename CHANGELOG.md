# Changelog

## Unreleased

- Made audio.cpp with Supertonic 3 GGUF the default inference provider.
- Added runtime-loaded complete audio.cpp providers for CPU, CUDA, Vulkan, and
  HIP/ROCm.
- Retained direct OpenVINO for Intel CPU, GPU, and NPU, including setup-time NPU
  cache preparation.
- Reworked guided and scriptable setup around exact provider discovery,
  ABI-only provisional setup, and model-backed file-only proof before final
  activation.
- Removed the obsolete split runtime/plugin configuration and packaging.
- Kept service installation behind the explicit `omaspeak setup systemd`
  command.
