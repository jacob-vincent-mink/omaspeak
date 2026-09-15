# Changelog

## Unreleased

## 0.0.1-rc.3 - 2026-09-15

- Made audio.cpp with Supertonic 3 GGUF the default inference provider.
- Added runtime-loaded complete audio.cpp providers for CPU, CUDA, Vulkan, and
  HIP/ROCm.
- Retained direct OpenVINO for Intel CPU, GPU, and NPU, including setup-time NPU
  cache preparation through OpenVINO's standard cache and a fresh-process
  cache-hit proof. Replaced the former converted/mixed graph entries with one
  device-neutral model containing only the official archived Supertonic 3 ONNX,
  configuration, Unicode indexer, and ten voice-style JSON files.
- Replaced archive extraction with per-file pinned downloads and made the model
  manifest part of the staged atomic installation and subsequent verification.
- Reworked guided and scriptable setup around exact provider discovery,
  ABI-only provisional setup, and model-backed file-only proof before final
  activation.
- Made runtime evidence distinguish an ABI-only, unverified audio.cpp device
  from an inaccessible device, and preserved every explicit
  `OMASPEAK_LIBRARY_PATH` dependency directory in the supervised worker.
- Fixed desktop setup launcher quoting for spaces, field codes, shell-reserved
  characters, backslashes, and Unicode executable paths.
- Removed the obsolete split runtime/plugin configuration and packaging.
- Removed unused bzip2 and tar dependencies from model setup.
- Framed NPU child results so native OpenVINO diagnostics cannot corrupt setup
  evidence, and preserved both stdout and stderr in actionable failures.
- Added the exact PocketFFT-derived FFT notice to release archives and reduced
  the package-owned audio.cpp provider to the C ABI build surface.
- Kept service installation behind the explicit `omaspeak setup systemd`
  command.
