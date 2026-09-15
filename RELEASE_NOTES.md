# Omaspeak 0.0.1-rc.2

This release candidate provides local Supertonic 3 speech synthesis with ten
speaker styles, WAV output, on-demand execution, a hot-model daemon, JSON
benchmarks, and guided setup. Linux x86-64 and aarch64 archives include the
default CPU runtime. The same executable can use an externally installed direct
OpenVINO runtime or the standalone ONNX Runtime CUDA Plugin EP selected during
setup.

This release candidate supersedes 0.0.1-rc.1 and updates rustls to 0.23.45
to address RUSTSEC-2026-0285.

Intel NPU setup prepares and verifies a fixed 12-graph cache before activation;
normal synthesis requires those cache hits and will not compile an unprepared
shape during first use. Successful native compiler diagnostics are captured by
the setup worker, while actual failures retain their diagnostic detail.

The default model is downloaded only after explicit acceptance of its
OpenRAIL-M terms. Validated configurations include default and OpenVINO CPU
plus Intel iGPU and NPU. Accelerator support requires matching external runtime
libraries. See `INSTALL.md`, `RUNTIME.md`, and the benchmark reports for exact
setup and evidence.
