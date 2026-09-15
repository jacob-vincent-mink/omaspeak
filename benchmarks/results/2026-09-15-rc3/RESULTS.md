# Omaspeak v0.0.1-rc.3 hardware evidence

The Linux release source at `620fbb7a792aa664c7fa056d0e6d6e36e5a0a4e6`
was exercised entirely with file output. No playback device was opened. The
local x86-64 binary SHA-256 was
`802a48b878d9bdeece5f969f80425e21dfd7650518d95f82c5ca4ab1a546178c`.

## Intel local results

The host was an Intel Core Ultra X7 358H with an Arc B390 integrated GPU and a
Series 3 NPU running OpenVINO 2026.3.1. Every backend synthesized the same text:
`Omaspeak release candidate three is running locally.` The default used the
packaged audio.cpp Supertonic GGUF provider. OpenVINO used the official pinned
Supertonic ONNX graphs.

“Fresh process” is one `say --no-play` invocation after setup with persistent
runtime caches retained. “Warm” is five measured requests through one loaded
engine after one unmeasured warmup request.

| Provider | Device | Fresh load | Fresh synthesis | Fresh total | Warm p50 / p95 synthesis | Warm p50 RTF | Placement |
|---|---|---:|---:|---:|---:|---:|---|
| audio.cpp GGUF | CPU | 1,287 ms | 1,597 ms | 2,884 ms | 1,658 / 1,694 ms | 0.442 | CPU worker verified |
| OpenVINO official ONNX | CPU | 42 ms | 1,322 ms | 1,364 ms | 405 / 407 ms | 0.103 | `EXECUTION_DEVICES` matched CPU for every graph |
| OpenVINO official ONNX | iGPU | 43 ms | 1,077 ms | 1,120 ms | 135 / 136 ms | 0.0344 | `EXECUTION_DEVICES` matched GPU for every graph |
| OpenVINO official ONNX | NPU | 194 ms | 710 ms | 904 ms | 64 / 92 ms | 0.0165 | `EXECUTION_DEVICES` matched NPU for every graph |

The OpenVINO paths use a different model representation and execution stack
from the packaged GGUF provider, so their timing difference is not accelerator
hardware alone. Within the official OpenVINO profile, iGPU and NPU provide the
expected warm throughput improvement over OpenVINO CPU. A qualification run
also observed a one-off 4.2-second iGPU first synthesis immediately after
setup; persistent-process performance settled to the table above.

NPU setup compiled and verified all ten fixed-shape blobs before activation:
780,748,228 bytes under fingerprint
`2831a873459475239af5047168e50539f1ed6fbc17d9a4299eca7a2858d35bc8`.
When a resident Voxtype process held an NPU inference session, OpenVINO
persisted only four blobs; releasing that workload allowed the next setup pass
to produce all ten. Omaspeak rejected the partial cache and now tells the user
to close other NPU workloads and retry. It never stops those applications.

An offline OpenVINO Whisper Base.en INT8 smoke check transcribed three outputs
exactly after case, punctuation, and number normalization. The iGPU output had
one proper-name substitution, `Omospeak` for `Omaspeak`: one token error across
28 normalized tokens (3.57%). This checks basic intelligibility for one
sentence. It is not a MOS, speaker-similarity, or corpus-level TTS evaluation.

## NVIDIA GB10 CUDA result

The same source commit built and passed all tests natively on aarch64, then
loaded a complete CUDA 13.0 audio.cpp provider on an NVIDIA GB10 with driver
580.173.02. All setup checks passed, fallback remained disabled, and
`nvidia-smi pmon` observed the exact Omaspeak worker PID using 548–554 MiB.
Independent whisper.cpp transcription exactly recovered `The quick brown fox
jumps over the lazy dog.` from the deterministic benchmark WAV.

| Cold load | Cold synthesis | Cold RTF | Hot p50 | Hot p95 | Hot p50 RTF |
|---:|---:|---:|---:|---:|---:|
| 3,491 ms | 237 ms | 0.0765 | 38.5 ms | 45.4 ms | 0.0124 |

SGLang was concurrently using about 92–95% of the GB10 SM capacity, so this is
functionality and placement evidence rather than an uncontended performance
ceiling. The complete evidence archive is retained locally as
`/tmp/omaspeak-rc3-cuda-evidence.tar.gz`, SHA-256
`0b9b93b8529597e01b4d8e336a2ec3387fce70609d5170a0ce0bd5e7d68538dd`.

Exact local report and WAV hashes are preserved in `metrics.json`.
