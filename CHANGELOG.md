# Changelog

## Unreleased

- Made audio.cpp with Supertonic 3 GGUF the default inference provider.
- Added runtime-loaded complete audio.cpp providers for CPU, CUDA, Vulkan, and
  HIP/ROCm.
- Retained direct OpenVINO for Intel CPU, GPU, and NPU, including setup-time NPU
  cache preparation through OpenVINO's standard cache and a fresh-process
  cache-hit proof. The validated NPU plan covers output buckets through 256
  latent frames and uses shorter text chunks.
- Reworked guided and scriptable setup around exact provider discovery,
  ABI-only provisional setup, and model-backed file-only proof before final
  activation.
- Removed the obsolete split runtime/plugin configuration and packaging.
- Framed NPU child results so native OpenVINO diagnostics cannot corrupt setup
  evidence, and preserved both stdout and stderr in actionable failures.
- Added the exact PocketFFT-derived FFT notice to release archives and reduced
  the package-owned audio.cpp provider to the C ABI build surface.
- Kept service installation behind the explicit `omaspeak setup systemd`
  command.
