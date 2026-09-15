# Omaspeak 0.0.1-rc.3

This candidate introduces the greenfield native-provider architecture:

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
- runtime inventory reports an audio.cpp device as unverified until that model
  proof succeeds, rather than presenting ABI discovery as a failed device;
- NPU model-cache compilation and a fresh-process cache-hit proof happen during
  setup, using OpenVINO's standard cache for the fixed 256-frame plan;
- ordinary setup does not install or start a systemd service;
- the Rust executable has no load-time dependency on an inference runtime.

Hardware performance and quality results from the former converted/mixed graph
catalog were withdrawn. Release evidence must be produced again with the pinned
official archive files before accelerator numbers are published.
