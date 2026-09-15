# Omaspeak 0.0.1-rc

This first preview provides local Supertonic 3 speech synthesis with ten
speaker styles, WAV output, on-demand execution, a hot-model daemon, JSON
benchmarks, and guided setup. The Linux x86-64 archive includes the default CPU
runtime. The same executable can use an externally installed direct OpenVINO
runtime or ONNX Runtime CUDA stack selected during setup.

Intel NPU setup prepares and verifies a fixed 12-graph cache before activation;
normal synthesis requires those cache hits and will not compile an unprepared
shape during first use.

The default model is downloaded only after explicit acceptance of its
OpenRAIL-M terms. Validated configurations include default and OpenVINO CPU,
Intel iGPU and NPU, and NVIDIA GB10 CUDA. Accelerator support requires matching
external runtime libraries. See `INSTALL.md`, `RUNTIME.md`, and the benchmark
reports for exact setup and evidence.
