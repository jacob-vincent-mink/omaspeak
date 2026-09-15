# Accelerator setup

Accelerators use software already installed on the machine. Omaspeak discovers
or records exact native paths and validates the selection; it does not install
drivers, SDKs, or vendor runtimes.

## audio.cpp CUDA, Vulkan, and HIP/ROCm

Build or install one complete audio.cpp shared library with the required backend
enabled. Keep its dependent libraries in the same provider tree or in that
installation's normal runtime search path. Select it with:

```bash
omaspeak setup runtime --runtime cuda --device gpu --device-id 0 --dir /path/to/provider --apply
omaspeak setup runtime --runtime vulkan --device gpu --device-id 0 --dir /path/to/provider --apply
omaspeak setup runtime --runtime hip --device gpu --device-id 0 --dir /path/to/provider --apply
```

`--device-id` selects a zero-based device for these accelerator runtimes. The
guided setup asks for the same value. The exact backend and device ID are sent
to audio.cpp when Omaspeak creates the worker session.

To build a compatible provider from the revision used by Omaspeak releases:

```bash
git clone https://github.com/0xShug0/audio.cpp /tmp/audio.cpp
git -C /tmp/audio.cpp checkout e9ff20042ec85af960a720368c6927cda19ad65f

# Choose exactly one of these backend flags.
backend_flag=-DENGINE_ENABLE_CUDA=ON       # NVIDIA CUDA 12 or newer
# backend_flag=-DENGINE_ENABLE_VULKAN=ON   # Vulkan SDK and loader
# backend_flag=-DENGINE_ENABLE_HIP=ON       # AMD ROCm/HIP

cmake -S /tmp/audio.cpp -B /tmp/audio.cpp-build \
  -DCMAKE_BUILD_TYPE=Release \
  -DAUDIOCPP_BUILD_C_API=ON \
  -DAUDIOCPP_DEPLOYMENT_BUILD=OFF \
  -DAUDIOCPP_MODEL_SET=custom \
  -DAUDIOCPP_MODELS=supertonic \
  -DENGINE_BUILD_EXAMPLES=OFF \
  -DENGINE_BUILD_TESTS=OFF \
  -DENGINE_BUILD_EXTENDED_TESTS=OFF \
  -DENGINE_BUILD_MODEL_TESTS=OFF \
  "$backend_flag"
cmake --build /tmp/audio.cpp-build --parallel --target audiocpp
```

For a local CUDA build, add `-DCMAKE_CUDA_ARCHITECTURES=native` to reduce build
time. Omit it when building a provider for other GPU generations. audio.cpp's
[Linux build guide](https://github.com/0xShug0/audio.cpp/blob/e9ff20042ec85af960a720368c6927cda19ad65f/docs/build/linux.md)
documents toolkit selection and architecture values. The resulting provider is
under `/tmp/audio.cpp-build/bin`; select that directory with the matching
`omaspeak setup runtime` command above.

ABI loading alone is not placement proof. With an installed GGUF model, setup
creates a real Supertonic session and writes a file-only synthesis before it
saves the candidate. Without a model it reports an ABI-only result and tells
you which setup action remains.

## Direct OpenVINO

Install an OpenVINO runtime that exposes `libopenvino_c` and `plugins.xml`, plus
the Intel driver for the selected device. Then:

```bash
omaspeak setup runtime --runtime openvino --device cpu --dir /opt/intel/openvino --apply
omaspeak setup runtime --runtime openvino --device gpu --dir /opt/intel/openvino --apply
omaspeak setup runtime --runtime openvino --device npu --dir /opt/intel/openvino --apply
```

The directory can be an official archive root such as `/opt/intel/openvino_2026`
or a system prefix such as `/usr`; setup locates the runtime library and plugin
manifest beneath the prefix and stores their exact paths. Intel documents the
[Linux OpenVINO archive installation](https://docs.openvino.ai/2026/get-started/install-openvino/install-openvino-archive-linux.html),
[GPU driver setup](https://docs.openvino.ai/2026/get-started/install-openvino/configurations/configurations-intel-gpu.html),
and [NPU driver setup](https://docs.openvino.ai/2026/get-started/install-openvino/configurations/configurations-intel-npu.html).
Omaspeak only validates those components; it never installs them.

Direct OpenVINO consumes the official archived Supertonic graph and metadata
files. Normal setup downloads every required file directly from the pinned
catalog revision after license acceptance:

```bash
omaspeak setup model --model supertonic-3-openvino \
  --accept-license OpenRAIL-M
```

The catalog pins `supertone-oss-archive/supertonic-3` revision
`aafc6e32416a594460b32413efc49d7fe4ce6d46`: four ONNX graphs, `tts.json`,
`unicode_indexer.json`, and ten voice-style JSON files. `--source PATH` is an
offline override for a directory with that exact layout and content.

For NPU use `supertonic-3-openvino`, whose file set contains the official
FP32 graphs. Setup verifies each file hash and compiles the fixed
shape plan through OpenVINO's standard `CACHE_DIR` mechanism, then starts a
fresh child and requires every graph to report a cache hit. The prepared latent
buckets stop at 256 frames (about 17.8 seconds of 44.1 kHz output). Omaspeak
uses shorter 160-character chunks, or 80 for Japanese and Korean, on NPU and
rejects a predicted chunk over that duration with the supported limit in the
error. You can inspect or repeat the explicit preparation step with:

```bash
omaspeak setup cache
omaspeak setup cache --prepare
```

Inference on NPU fails with an actionable error if the prepared cache is absent
or no longer matches the model/runtime identity. Accelerator validation is
machine-specific; `omaspeak setup runtime --json`, `omaspeak setup check`, and
`omaspeak benchmark` provide the evidence to retain for a proof run.

Keep validation file-only:

```bash
omaspeak setup check
omaspeak say --no-play --out /tmp/omaspeak-accelerator.wav \
  "Accelerator placement proof."
omaspeak benchmark --text "Warm accelerator benchmark." \
  --out-dir /tmp/omaspeak-benchmark
```

`benchmark` writes WAV files and JSON; it does not open a playback device.
