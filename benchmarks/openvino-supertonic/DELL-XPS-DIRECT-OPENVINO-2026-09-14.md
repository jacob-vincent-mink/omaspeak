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

The load columns are therefore not equivalent initialization measurements.
The default ONNX Runtime lane creates its four sessions during engine load,
while the direct OpenVINO lane's load initializes only the frontend and runtime;
its first `generate` compiles the four graphs for concrete tensor shapes. Once
warmed, OpenVINO CPU is faster than the default CPU lane (284.13 versus 362.05
ms p50), the iGPU reaches 97.52 ms, and the NPU reaches 38.43 ms. The much larger
cold OpenVINO values are compilation rather than steady-state inference.

This first run measures empty-cache compilation and repeated synthesis within
the same process. The release-candidate setup-cache validation below measures
the final static NPU plan and new-process cache imports.

The NPU kernel busy counter increased by 55,279 microseconds during the cold
lane and 429,861 microseconds during the hot lane. That independent counter
confirms physical NPU work in addition to OpenVINO's graph placement report.

A separate four-phrase accuracy run compared default CPU with direct OpenVINO
NPU. Both produced zero word errors over 25 normalized reference words, for an
NPU-minus-CPU WER delta of 0.00. The NPU busy counter increased by 233,397
microseconds. All eight accuracy WAVs were 44.1 kHz mono PCM and none clipped.

## Release-candidate setup cache

The final setup path compiled the fixed NPU plan before first use: one duration
predictor, one text encoder, and vector-estimator/vocoder pairs for latent
buckets 32, 64, 128, 256, and 512. Setup took 36.82 seconds, created exactly 12
OpenVINO blobs totaling 872,531,694 bytes (832 MiB), and increased the kernel
NPU busy counter by 431,991 microseconds. A second isolated process imported
all 12 graphs from cache before setup published the directory.

Every later NPU process required `LOADED_FROM_CACHE` for each graph it touched.
The prepared manifest, file metadata, and SHA-256 list remained unchanged after
all validation. Eight synthesis cases covered all five latent buckets, voices
0 through 9, speeds from 0.6 through 4.0, and English, Japanese, Korean, and
Arabic. A deliberately slow request predicted 1,098 frames, exceeded the
prepared limit of 512, and failed with an actionable error without creating a
WAV or changing the cache.

| Lane | New-process load | First cached synthesis | Process total | Hot p50 / p95 | Hot p50 RTF |
| --- | ---: | ---: | ---: | ---: | ---: |
| Default CPU | 432 ms | 394.44 ms | 866.67 ms | 395.77 / 406.06 ms | 0.1215 |
| OpenVINO CPU | 44 ms | 1,979.46 ms | 2,111.49 ms | 324.34 / 326.84 ms | 0.0996 |
| OpenVINO iGPU | 50 ms | 10,357.92 ms | 10,729.64 ms | 131.63 / 195.03 ms | 0.0404 |
| OpenVINO NPU, setup cache | 350 ms | 1,818.85 ms | 2,289.02 ms | 100.46 / 111.72 ms | 0.0311 |

The NPU's first synthesis imports the prepared static graphs; it does not
compile a new shape. The NPU busy counter increased by 83,096 microseconds in
that new-process lane and 418,114 microseconds in the hot lane. OpenVINO CPU
and iGPU still use exact lazy shapes, so their cold rows include compilation.

Four accuracy phrases produced per-phrase WER values of 0.000, 0.125, 0.000,
and 0.000 on both default CPU and NPU, an NPU-minus-CPU delta of zero in every
case. All outputs were 44.1 kHz mono, none clipped, duration differences stayed
between 0.65% and 1.07%, and NPU-to-CPU RMS ratios stayed between 0.93 and 1.14.

Compact proof artifacts are under
[`results/2026-09-14-direct-openvino`](results/2026-09-14-direct-openvino):
rendered configs, runtime and binary hashes, cold and hot benchmark JSON,
placement records, WER data, NPU counters, audio hashes and signal summaries,
and the passing harness result. Raw WAVs and compiled-model caches remain in
`/tmp` and are intentionally excluded from Git.

The final setup-cache proof is under
[`results/2026-09-14-setup-cache`](results/2026-09-14-setup-cache). It contains
the setup transcript, 12-blob manifest and hashes, bucket observations,
over-limit failure, CPU/NPU accuracy comparisons, NPU busy counters, and final
cold/hot timing summary. Raw WAVs and the 832 MiB compiled cache are excluded.
