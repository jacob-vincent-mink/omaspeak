# Supertonic runtime hardware benchmark

This directory documents the direct OpenVINO harness and preserves dated
reports as historical evidence. The current release architecture is described
in the repository's `RUNTIME.md` and `ACCELERATOR_SETUP.md`; dated benchmark
reports do not define the live runtime contract.

The [rc.1 export/import NPU proof](RC1-EXPORT-IMPORT-NPU-2026-09-15.md)
validates the current deterministic cache mechanism on physical hardware. The
[pre-export/import NPU smoke](FINAL-HEAD-NPU-SMOKE-2026-09-14.md) and
[earlier setup/NPU smoke](CURRENT-SETUP-SMOKE-2026-09-14.md) predate the
0.0.1-rc.1 deterministic cache mechanism.

This harness compares one Supertonic model through Omaspeak's shipped runtime
paths: dynamically loaded ONNX Runtime on CPU and direct OpenVINO on CPU, Intel
GPU, and Intel NPU. Each lane gets a cold process with one measured synthesis
and a hot process with two warmups followed by ten measured syntheses. Audio is
written directly to WAV files; the harness never plays it.

Build the ordinary runtime-neutral binary, point the harness at installed
runtimes, and run it:

```bash
cargo build --release --locked

BINARY=target/release/omaspeak \
ORT_LIBRARY=/path/to/libonnxruntime.so.1.30.0 \
OPENVINO_LIBRARY=/opt/intel/openvino/runtime/lib/intel64/libopenvino_c.so \
OPENVINO_PLUGINS=/opt/intel/openvino/runtime/lib/intel64/plugins.xml \
MODEL_DIR="$HOME/.local/share/omaspeak/models/supertonic-3-npu" \
benchmarks/openvino-supertonic/run-hardware.sh
```

The default model is `supertonic-3-npu` for all lanes so CPU, GPU, and NPU
process the same weights. It uses the official FP32 vector estimator with the
other three INT8 graphs. Override `MODEL_NAME`, `MODEL_DIR`, and
`VECTOR_ESTIMATOR` together to measure another installed model.

The executable does not contain or install either runtime. `ORT_LIBRARY`
selects the external ONNX Runtime core for the default CPU lane.
`OPENVINO_LIBRARY` and `OPENVINO_PLUGINS` select an external OpenVINO install;
`OPENVINO_LIB` and `TBB_LIB` add its dependency directories to the loader path.
`RUNTIME_LIB` is retained as the directory default for `ORT_LIBRARY`.

Override `RESULT_ROOT`, `WORK_ROOT`, `NPU_BUSY_PATH`, `VOXTYPE`, `RUN_ID`,
`TEXT`, or the space-separated `BACKENDS` list when needed. Rendered configs,
benchmark JSON, process timing, stderr, exit status, runtime hashes, hardware
metadata, NPU busy-time snapshots, transcripts, normalized WER, and compact
audio metrics go under `results/<run-id>/`. WAVs and the OpenVINO compiled-model
cache stay under `WORK_ROOT` in `/tmp` by default.

Every benchmark JSON record must report the requested effective runtime,
`fallback_used=false`, and `placement_verified=true`. For direct OpenVINO this
means every compiled graph reported the requested physical device through
`EXECUTION_DEVICES`. The NPU lane also requires a positive kernel NPU busy-time
delta. Successful synthesis alone is not accepted as placement proof.

The harness transcribes every cold WAV with the local Voxtype model and requires
zero normalized word error for its fixed phrase. Use
`run-npu-accuracy.sh` for the broader fixed-phrase CPU-versus-NPU check.

The committed reports dated 2026-09-14 record earlier experiments that led to
the direct OpenVINO implementation and the NPU-specific model. Reports that say
`sherpa-onnx`, ONNX Runtime OpenVINO EP, or Piper are historical evidence; they
do not describe the current harness or release architecture.

See
[`DELL-XPS-DIRECT-OPENVINO-2026-09-14.md`](DELL-XPS-DIRECT-OPENVINO-2026-09-14.md)
for direct OpenVINO device placement and CPU-versus-NPU accuracy evidence.
