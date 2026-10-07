# Paradee family: audio.cpp pickup note

Local planning note, 2026-10-07. This is not an implemented audio.cpp family or
an upstream issue submission. Upstream checkout inspected:
`cf124a67cc55d8f65a9a15eec69edbff0fb212c8`; Omaspeak's packaged provider is pinned
to `e9ff20042ec85af960a720368c6927cda19ad65f`. Neither registers Paradee.

## Reference and working native route

- Model: https://huggingface.co/sahilmahendrakar/Paradee-8M-v1.0
- Immutable release: `8f34b01ef8adcb0bec470b89bf2fd62a7b2369e6`.
- License: Apache-2.0, verified official license SHA-256
  `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30`.
- 8.07M parameters, English/American single `af_heart` voice distilled from
  Kokoro. It is not a Kokoro graph/tensor layout and cannot use that loader.
- Official artifacts include separate PyTorch `text_side.pt`/`decoder.pt` and
  FP32/INT8 ONNX. No official GGUF artifact is available in this revision.
- Existing Omaspeak experimental `src/paradee.rs` runs the untouched FP32 ONNX
  graph using OpenVINO CPU and a native eSpeak NG→Misaki frontend. It does not
  ship or depend on ONNX Runtime/Python. Its catalog model is
  `paradee-8m-openvino`; default selection is unchanged.

The graph contract is `input_ids: int64[1,T]`, `speed: float32[1]`, mono
`waveform: float32[1,samples]`, 24 kHz. Pad phonemes with zero at both ends;
maximum 510 phonemes plus two pads. There is no speaker/style embedding input.
The output already includes the phase correction filter. Randomness in the graph
means comparisons need controlled seeds rather than assuming byte determinism.

## Native audio.cpp implementation checklist

1. Add a distinct registered family and architecture metadata, loader and GGUF
   conversion. Map the release's actual text-side/decoder tensors; do not invent
   compatibility with Kokoro checkpoints. Pin and hash reference weights/data.
2. Reproduce text encoding, duration prediction/expansion, stochastic decoder,
   waveform construction and phase correction filter. Verify shapes/operators
   against reference outputs before optimizing or quantizing.
3. Reuse audio.cpp's existing embedded eSpeak NG phonemizer/data with a
   **Paradee-specific American eSpeak→Misaki map**. The model author's browser
   map is https://github.com/sahilmahendrakar/paradee/blob/main/web/misaki.js .
   Raw IPA or merely sharing Kokoro's vocabulary is inadequate. Preserve
   punctuation and number/abbreviation handling; compare realistic names and
   tricky pronunciations. Retain GPL notices/source delivery for eSpeak data.
4. Expose only `af_heart`, English and qualified speed controls. Return 24 kHz
   mono PCM. Declare complete-segment streaming only until decoder frame
   streaming is actually implemented. Emit named chunk audio once; final merged
   results must not duplicate already emitted audio in clients.
5. Verify CPU end-to-end on a fixed speech corpus, seeded waveform comparisons,
   pronunciation/listening or independently measured ASR, footprint, cold/warm
   latency and peak memory. Then qualify individual accelerators and quantized
   formats. Catalog metadata must reflect measured support, not generic ggml
   device availability.
6. Once the family is ready, update Omaspeak's pinned provider, required-family
   build list, model artifacts/provenance and voice inventory. Retire the direct
   OpenVINO frontend only after parity and deployment checks; no ORT fallback.

## OpenVINO export findings to carry forward

The official FP32 graph compiles and produces finite audio on OpenVINO CPU
2026.4. Official INT8 import succeeds but CPU compilation fails on an internal
non-static element type; static input shape and ONNX type inference did not fix
that failure.

GPU requires forced FP32 plus conversion of the two rank-3 linear Resize
operations to equivalent rank-4 operations using Unsqueeze/Squeeze. That
prototype compiled and generated audio, but a seeded reference comparison had
maximum absolute waveform difference around 0.00104. It is not yet a qualified
export or supported Omaspeak placement.

The unbounded/dynamic INT8 graph crashed inside the Intel NPU compiler after
unbounded-shape diagnostics. Do not rerun it as a supported path. Investigate
bounded/static text-side and decoder exports, dynamic duration expansion and
phase-filter operators in isolated compiler processes. NPU remains unsupported.

A small Omaspeak native CPU probe with the eSpeak fallback frontend loaded in
about 0.64 seconds; a 648-character repeated passage emitted three completed
segments (first about 0.586 seconds, total about 1.68 seconds, 38.9 seconds audio).
These are debug-build, one-machine sanity measurements, not quality qualification
or performance promises. Omaspeak's `docs/PARADEE.md` and streaming benchmark
artifacts contain the current limits and reproducible graph probes.
