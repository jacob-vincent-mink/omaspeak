# Dell XPS direct OpenVINO validation — 2026-09-14

Omaspeak's runtime-neutral x86-64 release build synthesized directly to WAV on
a Dell XPS with an Intel Panther Lake Arc B390 iGPU (`8086:b080`) and Series 3
NPU (`8086:b03e`). The host ran Linux 7.2.3, OpenVINO 2026.3.1, and the packaged
ONNX Runtime 1.29.0 CPU library. No audio was played.

Every lane used the same `supertonic-3-npu` model: the official FP32 vector
estimator plus the three official INT8 graphs. The default lane used direct
ONNX Runtime CPU execution. The other lanes loaded the user-installed
`libopenvino_c.so` and `plugins.xml`; Omaspeak compiled and ran all four public
ONNX graphs directly on the requested OpenVINO device.

| Lane | Load | Cold synthesis | Cold load + synthesis | Hot p50 / p95 | Hot p50 RTF | WER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Default CPU | 396 ms | 349.65 ms | 745.65 ms | 362.05 / 364.50 ms | 0.1292 | 0.00 |
| OpenVINO CPU | 34 ms | 1,860.92 ms | 1,894.92 ms | 284.13 / 287.44 ms | 0.1014 | 0.00 |
| OpenVINO iGPU | 29 ms | 3,989.01 ms | 4,018.01 ms | 97.52 / 99.67 ms | 0.0348 | 0.00 |
| OpenVINO NPU | 35 ms | 5,517.58 ms | 5,552.58 ms | 38.43 / 59.12 ms | 0.0137 | 0.00 |

OpenVINO compiles lazily for concrete tensor shapes, so its compilation cost
appears in the first synthesis rather than the frontend load column. The hot
figures follow two warmups and summarize ten measured syntheses. Every lane
reported `fallback_used=false` and `placement_verified=true`. For each OpenVINO
lane, every compiled graph's `EXECUTION_DEVICES` property matched the requested
physical device.

The NPU kernel busy counter increased by 55,279 microseconds during the cold
lane and 429,861 microseconds during the hot lane. That independent counter
confirms physical NPU work in addition to OpenVINO's graph placement report.

A separate four-phrase accuracy run compared default CPU with direct OpenVINO
NPU. Both produced zero word errors over 25 normalized reference words, for an
NPU-minus-CPU WER delta of 0.00. The NPU busy counter increased by 233,397
microseconds. All eight accuracy WAVs were 44.1 kHz mono PCM and none clipped.

Compact proof artifacts are under
[`results/2026-09-14-direct-openvino`](results/2026-09-14-direct-openvino):
rendered configs, runtime and binary hashes, cold and hot benchmark JSON,
placement records, WER data, NPU counters, audio hashes and signal summaries,
and the passing harness result. Raw WAVs and compiled-model caches remain in
`/tmp` and are intentionally excluded from Git.
