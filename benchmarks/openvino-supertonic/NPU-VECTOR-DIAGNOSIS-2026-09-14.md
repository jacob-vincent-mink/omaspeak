# Supertonic vector-estimator NPU diagnosis

On the same Dell XPS and native stack described in
[`DELL-XPS-OPENVINO-2026-09-14.md`](DELL-XPS-OPENVINO-2026-09-14.md), the fully
INT8 Supertonic vector estimator produced loud, clipped, unintelligible speech.
Disabling OpenVINO's QDQ optimizer, choosing the accuracy execution hint, and
disabling NPU QDQ and dynamic quantization optimizations did not restore the
reference transcription.

## Isolated failure

The INT8 vector graph contains 38 decomposed dynamic MatMul sequences:

```text
DynamicQuantizeLinear -> MatMulInteger -> Cast -> Mul(activation scale * weight scale)
```

Identical deterministic tensors with representative shapes (`[1,144,41]`
latent and `[1,256,34]` text) were submitted directly to OpenVINO CPU and NPU.
The stock graph's final outputs had correlation `0.687365` and RMSE `0.563107`.
At the first time-conditioning projection, CPU produced RMS `0.398253`, while
NPU produced exactly zero. Its activation and weight scales multiply to about
`4.69e-5`. Across the graph, 22 of the 38 rescaled dynamic MatMul outputs were
exactly zero on NPU.

A diagnostic graph replaced all 38 decomposed dynamic INT8 MatMuls with
mathematically equivalent float MatMuls using dequantized constant weights.
CPU and NPU then reached correlation `0.999988`, RMSE `0.003702`, and maximum
absolute difference `0.015065`. The rewritten CPU result also stayed close to
the stock INT8 CPU result: correlation `0.999690` and RMSE `0.019377`. This
isolates the failure to the dynamic INT8 MatMul representation or its NPU
lowering, rather than the surrounding attention and convolution blocks.

## Maintained model fix

Omaspeak uses Supertone's official FP32 `vector_estimator.onnx` instead of
shipping a locally rewritten model. The `supertonic-3-npu` catalog entry pins
the 256,534,781-byte file from immutable Supertone revision
`724fb5abbf5502583fb520898d45929e62f02c0b` with SHA256
`883ac868ea0275ef0e991524dc64f16b3c0376efd7c320af6b53f5b780d7c61c`.
The other three graphs remain from the pinned INT8 sherpa-onnx archive.
Installing that catalog entry through `omaspeak setup model --download` and
then synthesizing directly to WAV from the installed directory also passed:
the physical NPU busy counter increased by 405,166 microseconds, no component
fell back from the requested all-component placement, and hot synthesis reached
36.54 ms p50 and 60.13 ms p95.

Four independent cold, direct-to-WAV phrases ran with all four component
sessions submitted to OpenVINO NPU. Local Whisper transcription matched every
reference with WER `0.00`:

| Phrase | CPU synthesis | NPU cold synthesis | CPU WER | NPU WER |
| --- | ---: | ---: | ---: | ---: |
| `Omaspeak now supports Supertonic.` | 378 ms | 4,378 ms | 0.00 | 0.00 |
| `The quick brown fox jumps over the lazy dog.` | 516 ms | 5,253 ms | 0.00 | 0.00 |
| `Please open the kitchen window before the morning meeting.` | 626 ms | 5,126 ms | 0.00 | 0.00 |
| `Local speech synthesis should remain clear across several different sentences.` | 708 ms | 5,124 ms | 0.00 | 0.00 |

Compilation dominates those cold NPU processes. With one loaded engine, two
warmups, and ten measured syntheses of the first phrase, NPU synthesis reached
39.35 ms p50, 55.70 ms p95, and `0.0140` p50 real-time factor. The NPU busy
counter increased by 402,653 microseconds. Ten independently sampled outputs
had two ASR word errors out of 40 reference words; the corresponding CPU run
had three errors out of 40. This small sample establishes parity for the tested
phrases rather than general perceptual quality.

Four ORT profile files each contained OpenVINO execution-provider events. A
separate profiled run increased the physical NPU busy counter by 52,957
microseconds. ORT also recorded CPU events for shape and unpartitioned work, so
the evidence proves that every component session was submitted to the NPU and
that the physical NPU executed work; it does not claim every ONNX operation was
placed there.
