# Supertonic OpenVINO hardware benchmark

This harness compares the official Supertonic 3 int8 model using the default
ONNX Runtime CPU provider and the OpenVINO CPU, GPU, and NPU devices. Each
backend gets a cold process with one measured synthesis and a separate hot
process with two warmups followed by ten measured syntheses. OpenVINO CPU runs
all four graphs; GPU runs only the vector estimator at FP32; NPU runs the
duration predictor, text encoder, and vocoder. The remaining graphs in each
mixed lane run on ORT CPU.

Build against an OpenVINO-enabled shared sherpa-onnx stack, then run:

```bash
SHERPA_ONNX_LIB_DIR=/tmp/oma-native/runtime/lib \
  CARGO_TARGET_DIR=target-openvino \
  cargo build --release --locked --features openvino

benchmarks/openvino-supertonic/run-hardware.sh
```

The defaults use ORT 1.29 with Intel OpenVINO 2026.2.1, the version supported
by that ORT release, plus the required
[`zero-element tensor patch`](../../native/openvino/patches/onnxruntime-openvino-zero-element-tensors-v1.29.0.patch).
Stock ORT 1.29 aborts on Supertonic, and upstream sherpa-onnx lacks the
per-component routing key used by the accurate mixed placements. Follow the
pinned [`native build recipe`](../../native/openvino/README.md). Override
`BINARY`, `RUNTIME_LIB`, `OPENVINO_LIB`, `TBB_LIB`, `MODEL_DIR`, `RESULT_ROOT`,
`WORK_ROOT`, `PROFILE_ROOT`, `NPU_BUSY_PATH`, `VOXTYPE`, `RUN_ID`, `TEXT`, or the
space-separated `BACKENDS` list when needed. Rendered
configs, benchmark JSON, process timing, stderr, exit status, hardware metadata,
NPU busy-time snapshots, profile hashes, Voxtype transcripts, and normalized
WER go under `results/<run-id>/`.
Synthesized WAVs, generated provider caches, and raw ORT profiles stay below
`WORK_ROOT` (under `/tmp` by default) so large artifacts are not added to Git.
`profile-summary.json` retains provider-event counts, OpenVINO node timings, and
the size and SHA-256 of every external raw profile.
`audio-summary.json` retains WAV hashes, format, duration, RMS/peak levels, and
zero/clipping fractions without copying the WAVs into Git.

Some NPU compiler diagnostics are written to stdout. The byte-for-byte output
is retained as `*.benchmark.raw.log`; `*.benchmark.json` contains the extracted
JSON object. Missing output, a nonzero lane exit, invalid JSON, missing WAVs,
missing OpenVINO profile events, a nonpositive aggregate NPU busy-time delta,
or nonzero normalized WER causes the harness to exit nonzero. NPU profiling is
collected on the cold run;
the hot run omits profiling because this patched ORT stack has shown unstable
profile teardown on NPU.

The NPU template supplies the host-specific ORT 1.29 property as inline JSON:

```toml
load_config = '{"NPU":{"NPU_PLATFORM":"5010"}}'
```

Benchmark JSON always reports `placement_verified=false`. Treat NPU busy-time
deltas and OpenVINO entries in the ORT profile as separate placement evidence;
successful synthesis alone is insufficient.

`run-npu-accuracy.sh` synthesizes a fixed four-word phrase with incremental NPU
accuracy settings, transcribes each WAV with the local Voxtype Whisper model,
and computes normalized WER. It exits zero only when at least one variant has
zero WER; the normalizer treats `OMA speak` and `Omaspeak` as the same product
name. This is a narrow intelligibility regression check rather than a general
speech-quality score.

See [`DELL-XPS-OPENVINO-2026-09-14.md`](DELL-XPS-OPENVINO-2026-09-14.md) for
the passing four-lane hardware matrix and the failed full-device accuracy
experiments that determined the generated mixed-placement defaults.
