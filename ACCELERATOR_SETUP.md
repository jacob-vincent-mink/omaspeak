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

Direct OpenVINO consumes the official Supertonic graph files. Omaspeak does not
publish a synthetic multi-file archive. Supply a directory whose files came
from the official pinned Supertone revision:

```bash
omaspeak setup model --model supertonic-3-int8 \
  --archive /path/to/official/files --accept-license OpenRAIL-M
```

For NPU use `supertonic-3-npu`, whose pinned file set includes the official
FP32 vector estimator. Setup verifies each file hash and compiles the fixed
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
