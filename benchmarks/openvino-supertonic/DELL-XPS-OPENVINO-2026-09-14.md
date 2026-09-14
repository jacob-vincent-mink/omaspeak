# Dell XPS Supertonic OpenVINO results

> A later diagnosis found the fully INT8 NPU failure and validated an official
> FP32 vector estimator with all-component NPU submission. See
> [`NPU-VECTOR-DIAGNOSIS-2026-09-14.md`](NPU-VECTOR-DIAGNOSIS-2026-09-14.md).

On 2026-09-14, Omaspeak synthesized `Omaspeak now supports Supertonic.` with
the official Supertonic 3 int8 model on a Dell XPS with an Intel Panther Lake
Arc B390 iGPU (`8086:b080`) and Series 3 NPU (`8086:b03e`). The host ran Linux
7.2.3. The native stack was ONNX Runtime 1.29, OpenVINO 2026.2.1, and
sherpa-onnx 1.13.8. This historical run used the predecessor combined native
builder and patches, which are no longer part of the Omaspeak source or release
architecture.

Each cold process loaded the model and synthesized once. Each independent hot
process loaded once, warmed up twice, then recorded ten syntheses. Every output
was a 44.1 kHz mono WAV of about 2.802 seconds. Times below are Omaspeak's
synthesis measurements; load time is reported separately.

| Lane | OpenVINO placement | Load (ms) | Cold synth (ms) | Hot p50 / p95 (ms) | Hot p50 RTF | WER |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Default CPU | none | 333 | 479.71 | 402.76 / 408.76 | 0.144 | 0.00 |
| OpenVINO CPU | all four components | 250 | 1,959.67 | 377.95 / 399.56 | 0.135 | 0.00 |
| OpenVINO GPU | vector estimator only, FP32 | 317 | 1,599.67 | 140.43 / 143.54 | 0.050 | 0.00 |
| OpenVINO NPU | duration predictor, text encoder, and vocoder | 463 | 1,827.86 | 367.74 / 380.89 | 0.131 | 0.00 |

Voxtype's local Whisper base.en int8 model transcribed every cold output as
either `OMASpeak now supports Supertonic.` or `OMA speak now supports
Supertonic.`; the product-name normalization gives zero errors over the
four-word reference. This is a narrow intelligibility check, not a perceptual
speech-quality score.

ORT profiles contained 156 OpenVINO provider events for CPU, 65 for GPU, and 7
for NPU. The GPU profile contains only its selected component. The mixed NPU
profile also contains 1,389 CPU provider events. The physical NPU busy counter
increased by 6,530 microseconds for the cold run and 69,641 microseconds for the
hot run. These observations establish activity for the selected components;
they do not show that the entire mixed model ran on an accelerator.

Full-device experiments defined the compatibility boundary. Full GPU placement
aborted in the Intel GPU plugin on the duration predictor's zero-shape layout.
GPU vector-estimator placement at the provider's default precision produced
degenerate speech; FP32 with the QDQ optimizer disabled passed. Full NPU
placement synthesized and incremented the hardware counter but failed every
tested accuracy configuration:

| Full-NPU configuration | Normalized WER |
| --- | ---: |
| generated defaults | 3.25 |
| `enable_qdq_optimizer=False` | 2.00 |
| plus `EXECUTION_MODE_HINT=ACCURACY` | 3.00 |
| plus `NPU_QDQ_OPTIMIZATION=NO` | 1.00 |
| plus `NPU_COMPILER_DYNAMIC_QUANTIZATION=NO` | 2.00 |

The passing matrix used Omaspeak's generated defaults: GPU selects
`vector_estimator`, `precision=FP32`, `enable_qdq_optimizer=False`, and
`disable_dynamic_shapes=True`; NPU selects
`duration_predictor,text_encoder,vocoder`, enables the QDQ optimizer, disables
dynamic shapes, and uses the host-specific inline `NPU_PLATFORM=5010` property.
The benchmark JSON intentionally retained `placement_verified=false`; profile
and hardware evidence are reported independently.

Run ID `2026-09-14-dell-xps-final` passed the historical harness validation.
The compact evidence remains tracked in this directory. The current harness
targets direct OpenVINO and produces a separate proof record.
