# Kokoro on OpenVINO 2026.4

OpenVINO [2026.4.0](https://github.com/openvinotoolkit/openvino/releases/tag/2026.4.0)
lists Kokoro-82M as supported on Intel CPU, GPU, and NPU. Intel's
[INT8 IR](https://huggingface.co/OpenVINO/Kokoro-82M-int8-ov) is a different
model representation from Omaspeak's `kokoro-82m-gguf`. On NPU it runs through
the [OpenVINO GenAI speech pipeline](https://docs.openvino.ai/2026/api/genai_api/_autosummary/openvino_genai.Text2SpeechPipeline.html),
which supplies phonemization and generation. Omaspeak's `kokoro-genai` adapter
keeps a Python worker loaded and passes one of two pinned voice embeddings.

## Install the catalog profile

The profile requires Python with `openvino>=2026.4`, `openvino-genai>=2026.4`,
and NumPy. The tested environment used exact versions 2026.4.0 and 2026.4.0.0:

```sh
uv venv ~/.local/share/omaspeak/kokoro-python
uv pip install --python ~/.local/share/omaspeak/kokoro-python/bin/python \
  'openvino==2026.4.0' 'openvino-genai==2026.4.0.0' numpy

omaspeak config set backend.kind kokoro-genai
omaspeak config set backend.runtime openvino
omaspeak config set backend.device npu
omaspeak config set backend.options.python \
  "$HOME/.local/share/omaspeak/kokoro-python/bin/python"
omaspeak setup model --download kokoro-82m-openvino
omaspeak say --voice af_heart --out /tmp/kokoro.wav --no-play \
  'Hello from Kokoro on the NPU.'
```

The catalog pins the Intel INT8 model at revision
`9c035d0bfab136f0c1c525a5841d3411d7220c66`, including its phonemizer
data and two American English voice embeddings (`af_heart`, `am_michael`).
Activation proves actual synthesis on the selected device and rejects OpenVINO
or GenAI older than 2026.4. The current GenAI path supports speed `1.0`.
To install from an already downloaded revision, add
`--source /path/to/Kokoro-82M-int8-ov` to the setup command. The catalog
verifies every file's size and SHA-256 before activation.

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

The integrated profile was installed from the pinned source set and passed
Omaspeak's activation proof, benchmark command, and normal `say` request worker
on the Panther Lake NPU. Both generated WAVs were finite, non-silent 24 kHz
mono. A Kokoro `say` output was also transcribed and detected by Omawake's
Whisper verifier running against an isolated OpenVINO 2026.4 GenAI C build.
These are file-only checks; no microphone, speaker, or live daemon was used.

The 2026.4 OpenVINO GenAI C headers expose Whisper but no text-to-speech
pipeline, so the Omaspeak adapter uses the Python API. A future C++ bridge or
GenAI C API addition could remove the Python dependency. The upstream speech
pipeline and sample request C++17.

An `openvino-rs` change is only needed if the native adapter requires a C API
that its current bindings lack or a 2026.4 runtime compatibility defect is
reproduced. Omaspeak currently uses `openvino` and `openvino-sys` 0.11 with
runtime linking, so a new runtime can be selected independently of the Rust
crate version. This was checked with a temporary OpenVINO 2026.4.0 Python-wheel
runtime: Omaspeak's `openvino-rs` 0.11 binding loaded its C library, the runtime
probe reported 2026.4.0 and `CPU`, and a file-only Supertonic synthesis wrote
a 44.1 kHz WAV with 101,179 samples. This checks existing API compatibility,
not Kokoro support.
