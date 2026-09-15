# Dell XPS native-provider result

This file-only run validates commit `62187cb` on an Intel Core Ultra X7 358H
with an Arc B390 iGPU and Intel Series 3 NPU. The host used OpenVINO 2026.3.1,
Intel NPU driver 1.38.0, and Intel compute runtime 26.31.39395.13. No playback
device was opened.

The test synthesized `The quick brown fox jumps over the lazy dog.` with voice
F3 and seed 42. A cold observation is one fresh `omaspeak say --no-play`
process after setup has populated the selected provider's persistent cache. A
hot observation keeps one engine loaded, performs one warmup, and measures five
file-only requests.

| Provider and device | Cold load | Cold synthesis | Cold total | Hot synthesis p50 / p95 | Hot RTF p50 / p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| packaged audio.cpp CPU | 1,342 ms | 1,366 ms | 2,708 ms | 1,380.7 / 1,393.8 ms | 0.4314 / 0.4355 |
| OpenVINO CPU | 42 ms | 1,555 ms | 1,597 ms | 279.8 / 280.3 ms | 0.0996 / 0.0998 |
| OpenVINO Arc B390 iGPU | 46 ms | 1,022 ms | 1,068 ms | 100.5 / 100.9 ms | 0.0358 / 0.0359 |
| OpenVINO NPU | 4,296 ms | 688 ms | 4,984 ms | 67.3 / 97.3 ms | 0.0241 / 0.0349 |

OpenVINO reported the requested device through `EXECUTION_DEVICES` for every
compiled graph. The NPU run also required ten nonempty setup-prepared cache
blobs and a successful fresh-child `LOADED_FROM_CACHE` check before model
activation. Its longer process load is therefore cached NPU model import and
initialization, rather than first-use compilation.

The default lane uses the original-precision Supertonic GGUF with eight
generation steps. The direct OpenVINO lanes use the same official ONNX graph
set and FP32 vector estimator with five steps, so the default lane is a product
baseline rather than a controlled runtime-only comparison. OpenVINO CPU, iGPU,
and NPU are the comparable device lanes.

Whisper Base.en transcribed the first deterministic output from all four lanes
as the exact normalized input sentence, for zero word errors in this
intelligibility smoke. The NPU output was 2.790 seconds versus 2.809 seconds for
OpenVINO CPU and iGPU, a 0.69% duration difference. This guards against a large
functional quality regression; it is not a listening test or a MOS claim.

The stripped packaged provider was 4,945,160 bytes with SHA-256
`f535626dd8628e71ec2cf863a9667a632018fb24fb4655ccde6b98ba17efbcdc`.
It used audio.cpp commit `e9ff20042ec85af960a720368c6927cda19ad65f` and the
454,072,836-byte GGUF with SHA-256
`af814486a0bc9513fb36afabd9b1155ad14fb2c36a107ac6ffe62ea9adafb662`.
The direct lanes used Supertonic revision
`724fb5abbf5502583fb520898d45929e62f02c0b`; the FP32 vector estimator SHA-256
was `883ac868ea0275ef0e991524dc64f16b3c0376efd7c320af6b53f5b780d7c61c`.

The complete machine-readable summary is in [metrics.json](metrics.json).
