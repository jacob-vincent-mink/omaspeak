# Kokoro on OpenVINO 2026.4

OpenVINO [2026.4.0](https://github.com/openvinotoolkit/openvino/releases/tag/2026.4.0)
lists Kokoro-82M as supported on Intel CPU, GPU, and NPU. Intel's
[INT8 IR](https://huggingface.co/OpenVINO/Kokoro-82M-int8-ov) is a different
model representation from Omaspeak's `kokoro-82m-gguf`. On NPU it runs through
the [OpenVINO GenAI speech pipeline](https://docs.openvino.ai/2026/api/genai_api/_autosummary/openvino_genai.Text2SpeechPipeline.html),
which supplies phonemization, voice loading, and generation. The direct Omaspeak
OpenVINO provider still runs Supertonic.

## File-only device probe

The probe is an S14 experiment. It uses an isolated Python environment and a
local copy of Intel's model; it does not modify Omaspeak configuration.

```sh
uv venv /tmp/omaspeak-kokoro-ov
uv pip install --python /tmp/omaspeak-kokoro-ov/bin/python \
  'openvino==2026.4.0' 'openvino-genai==2026.4.0.0' \
  'numpy' 'soundfile' 'huggingface_hub'

/tmp/omaspeak-kokoro-ov/bin/hf download OpenVINO/Kokoro-82M-int8-ov \
  --revision 9c035d0bfab136f0c1c525a5841d3411d7220c66 \
  --local-dir /tmp/omaspeak-kokoro-ov-model

/tmp/omaspeak-kokoro-ov/bin/python scripts/kokoro-openvino-spike.py \
  --model-dir /tmp/omaspeak-kokoro-ov-model --device NPU \
  --output /tmp/kokoro-npu.wav 'Hello from Kokoro on the NPU.'
```

The script rejects a missing NPU and constructs the GenAI pipeline with the
explicit `NPU` device. It reports load and repeat synthesis time and writes a
24 kHz mono WAV. The [Panther Lake NPU results and listening files](../benchmarks/results/2026-09-24-kokoro-openvino-npu/RESULTS.md)
show successful `af_heart` and `am_michael` synthesis. The Optimum Intel model
wrapper failed NPU compilation on the same model, so this probe uses GenAI.

Before adding a supported catalog profile, pin the complete IR and voice file
set, prove fresh installation and Omaspeak worker integration, compare listening
quality and latency against the existing Kokoro GGUF and Supertonic profiles,
and verify device placement in the integrated worker. The NPU result proves
the upstream pipeline on this host, not an Omaspeak Kokoro provider.

The 2026.4 OpenVINO GenAI C headers expose Whisper but no text-to-speech
pipeline. Native integration therefore needs a small C++ bridge or an upstream
GenAI C API addition. The GenAI speech pipeline and its sample request C++17;
the successful Python probe did not compile an Omaspeak C++ bridge.

An `openvino-rs` change is only needed if the native adapter requires a C API
that its current bindings lack or a 2026.4 runtime compatibility defect is
reproduced. Omaspeak currently uses `openvino` and `openvino-sys` 0.11 with
runtime linking, so a new runtime can be selected independently of the Rust
crate version. This was checked with a temporary OpenVINO 2026.4.0 Python-wheel
runtime: Omaspeak's `openvino-rs` 0.11 binding loaded its C library, the runtime
probe reported 2026.4.0 and `CPU`, and a file-only Supertonic synthesis wrote
a 44.1 kHz WAV with 101,179 samples. This checks existing API compatibility,
not Kokoro support.
