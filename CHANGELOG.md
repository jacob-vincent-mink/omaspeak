# Changelog

## Unreleased

## 0.0.1-rc.2 - 2026-09-15

- Updated rustls to 0.23.45 to address RUSTSEC-2026-0285. This release
  candidate supersedes 0.0.1-rc.1.

## 0.0.1-rc.1 - 2026-09-14

- Updated the packaged CPU core to ONNX Runtime 1.30.0 and switched CUDA setup
  to Microsoft's separately distributed CUDA Plugin EP.
- Added native Linux aarch64 release packaging alongside x86-64.
- Kept normal runtime commands strict while allowing setup to replace an
  invalid pre-release configuration after a successful apply.
- Prevented native build and test debris beside a developer binary from being
  treated as an installed runtime.

## 0.0.1-rc - 2026-09-14

- Added local Supertonic 3 speech synthesis with ten selectable speaker styles.
- Added on-demand synthesis, a persistent daemon, WAV output, and JSON benchmarks.
- Added guided and scriptable setup for models, runtimes, and diagnostics.
- Added one runtime-neutral executable with packaged CPU support and external
  direct OpenVINO and ONNX Runtime CUDA selection.
- Added setup-time compilation and cross-process verification of the fixed
  12-graph Intel NPU cache plan.
- Added Intel CPU, iGPU, NPU, and NVIDIA GB10 validation evidence.

Known limits: release artifacts target Linux x86-64 and aarch64 with glibc
2.35 or newer, and accelerator stacks are external.
