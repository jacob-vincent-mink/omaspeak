# Kokoro on OpenVINO 2026.4

Omaspeak uses OpenVINO GenAI's C++ `Text2SpeechPipeline` through a small C ABI
bridge. The application downloads the pinned model with Rust and does not run
Python. The bridge is an optional provider; setup checks that it loads and can
see the selected device before downloading the model. Linux release archives
include the bridge in `lib/`; copy that directory beside the executable or to
`~/.local/lib/omaspeak`. OpenVINO and OpenVINO GenAI 2026.4 runtime libraries
must be installed separately with `ENABLE_MISAKI_CPP=ON`; a runtime built
without Misaki can detect NPU but cannot load Kokoro.

## Build the native provider

Install matching OpenVINO 2026.4 and OpenVINO GenAI 2026.4 C++ development
libraries. Build and place the bridge in Omaspeak's provider directory:

```sh
scripts/build-kokoro-openvino-bridge.sh \
  /path/to/openvino_genai/include \
  /path/to/openvino_genai/lib \
  "$HOME/.local/lib/omaspeak/libomaspeak_kokoro_openvino.so"
```

An extracted release archive can instead keep the bridge in `lib/` beside the
`omaspeak` executable. `backend.options.kokoro_library` or
`OMASPEAK_KOKORO_LIBRARY` can name an absolute path for a custom installation.
The GenAI library and its dependencies must also be loadable by the system
dynamic linker. No Python environment or module is required.
Omaspeak searches the executable bundle, its install prefix,
`~/.local/lib/omaspeak`, and system library directories for the bridge. It
passes a discovered OpenVINO CPU, GPU, or NPU plugin to the same GenAI core that
loads Kokoro, including distributions that put plugins under `/usr/lib/openvino`.

## Set up and run

```sh
omaspeak setup
# Select OpenVINO, a detected CPU/GPU/NPU, and Kokoro OpenVINO, then Apply.
omaspeak say --voice af_heart --out /tmp/kokoro.wav --no-play \
  'Hello from Kokoro on OpenVINO.'
```

The catalog pins Intel's INT8 IR at revision
`9c035d0bfab136f0c1c525a5841d3411d7220c66`, including phonemizer data
and two American English voice embeddings. Activation proves actual synthesis
on the selected device. Current GenAI synthesis supports speed `1.0`.
To install from an already downloaded revision, add
`--source /path/to/Kokoro-82M-int8-ov` to `omaspeak setup model --download`.
The catalog verifies every file's size and SHA-256 before activation.

The [earlier Panther Lake NPU results](../benchmarks/results/2026-09-24-kokoro-openvino-npu/RESULTS.md)
were measured with the former Python experiment. They are historical performance
evidence, not validation of this native bridge.

OpenVINO GenAI 2026.4 exposes text to speech through its C++ API. Its C API
does not expose that pipeline, which is why the bridge is necessary. Omaspeak's
Supertonic OpenVINO path continues to use `openvino-rs` directly.
