# Kokoro 82M on Panther Lake NPU — 2026-09-24

Intel Core Ultra Series 3 NPU (`8086:b03e`) was visible at `/dev/accel/accel0`.
The temporary Python environment used OpenVINO `2026.4.0` build
`22959-99c81491cc3` and OpenVINO GenAI `2026.4.0.0` build `3407-7ea2546852a`.
The model was Intel's `OpenVINO/Kokoro-82M-int8-ov` at immutable revision
`9c035d0bfab136f0c1c525a5841d3411d7220c66`.

OpenVINO reported `CPU`, `GPU`, and `NPU` as available. The
[`kokoro-openvino-spike.py`](../../../scripts/kokoro-openvino-spike.py) probe
constructed `Text2SpeechPipeline(model_dir, "NPU")`, supplied the pinned voice
embedding, generated speech, and saved 24 kHz mono WAVs. The initial uncached
NPU pipeline construction took 78.9 s and generated the short utterance in
2.49 s. Repeating with a prepared cache produced:

| Voice | Input | Pipeline load | Synthesis | Output | RMS / peak |
|---|---|---:|---:|---:|---:|
| `af_heart` | “Hello from Kokoro on the NPU.” | 3.42 s | 2.49 s, then 0.99 s | 61,200 frames, 2.55 s | 0.0506 / 0.315 |
| `am_michael` | “The build finished at ten forty-five, after three retries. Version two point six is ready.” | 3.73 s | 3.65 s | 161,400 frames, 6.73 s | 0.0422 / 0.390 |

The files are in [`samples/`](samples). Both WAVs have finite, non-silent audio
at 24 kHz mono. These are single-run timings, not latency percentiles. The
samples have not received a human listening rating. The probe targets the NPU
explicitly; it is separate from Omaspeak's application worker.

A second run from a fresh environment containing only `openvino==2026.4.0`,
`openvino-genai==2026.4.0.0`, NumPy, SoundFile, and Hugging Face Hub also
succeeded on NPU: `af_heart`, 61,200 frames, 3.56 s prepared-cache load and
2.66 s synthesis. The Optimum, Kokoro Python, Misaki, spaCy, and PyTorch
packages were absent from that environment.

The Optimum Intel path from the model card did **not** compile on this NPU.
The first failure cited unbounded graph dimensions; reshaping its inputs to a
static sentence length then failed because `aten::repeat_interleave/Tile` had a
nonconstant repeats input. OpenVINO GenAI 2026.4 succeeded with the same IR.
Release integration should use the GenAI Kokoro pipeline, not assume that
Optimum's raw model wrapper is NPU-ready.
