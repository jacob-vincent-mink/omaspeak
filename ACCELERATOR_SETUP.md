# Accelerator setup

Omaspeak 0.0.1-rc ships one runtime-neutral executable and a small default CPU
ONNX Runtime. Acceleration is optional. Setup discovers and validates native
libraries already installed by the user; it never downloads or installs an
OpenVINO, CUDA, or driver package.

Run `omaspeak setup runtime --json` at any point to see every supported
runtime/device pair, the files that were discovered, and exact remediation.
Only an explicit `--apply` writes a runtime selection.

## Default CPU

The release archive and `-bin-rc` package need no inference runtime package:

```bash
omaspeak setup all --accept-license OpenRAIL-M
omaspeak say "Default CPU is ready" --no-play --out cpu-check.wav
```

The release-provided `libonnxruntime.so.1.29.0` is found beside the executable
or under `/usr/lib/omaspeak` when installed by a package.

## Intel CPU, integrated GPU, and NPU with OpenVINO

Omaspeak uses OpenVINO directly for Intel devices. It needs
`libopenvino_c.so`, `plugins.xml`, the device plugin, and the matching driver.
On Arch Linux and Omarchy, install only the device stack you plan to use:

```bash
# OpenVINO CPU
sudo pacman -S --needed openvino

# Add Intel integrated GPU support
sudo pacman -S --needed openvino-intel-gpu-plugin

# Add Intel NPU support (includes the compiler and NPU driver dependencies)
sudo pacman -S --needed openvino-intel-npu-plugin
```

Log out and back in if group membership or device permissions changed. Confirm
that the selected device node is visible:

```bash
ls -l /dev/dri/renderD*   # integrated GPU
ls -l /dev/accel/accel*  # NPU
```

The Arch packages place the C library at `/usr/lib/libopenvino_c.so` and the
plugin catalog at `/usr/lib/openvino/plugins.xml`, so `/usr` is the setup root:

```bash
# Intel integrated GPU
omaspeak setup runtime --runtime openvino --device gpu --dir /usr --apply
omaspeak setup model --download supertonic-3-int8 --accept-license OpenRAIL-M

# Intel NPU
omaspeak setup runtime --runtime openvino --device npu --dir /usr --apply
omaspeak setup model --download supertonic-3-npu --accept-license OpenRAIL-M
omaspeak setup cache --prepare
```

For Intel's archive distribution, pass the extracted OpenVINO root instead.
Setup recognizes `runtime/lib/intel64` and
`runtime/lib/intel64/Release` automatically.

As soon as an NPU runtime and compatible installed model are both selected,
setup compiles a fixed, reusable shape plan before activating the second
selection: text length 320 and latent lengths 32, 64, 128, 256, and 512. It
publishes exactly 12 compiled blobs below
`${XDG_CACHE_HOME:-$HOME/.cache}/omaspeak/openvino/npu/static-v1/<fingerprint>`
and then starts a second isolated process that must load every graph from that
cache. The tested OpenVINO 2026.3.1 cache occupies about 833 MiB. Normal NPU
synthesis refuses to compile an unprepared shape and directs the user back to
`omaspeak setup cache --prepare`. Inputs beyond the prepared range fail safely;
split unusually long text into shorter requests.

Verify placement without playing audio:

```bash
omaspeak setup check
omaspeak say "Intel accelerator check" --no-play --out intel-check.wav
```

The [Dell XPS OpenVINO report](benchmarks/openvino-supertonic/DELL-XPS-DIRECT-OPENVINO-2026-09-14.md)
records CPU, integrated GPU, and NPU placement, timing, and output accuracy.

## NVIDIA GPU with CUDA

Omaspeak's CUDA backend needs an ONNX Runtime 1.29 GPU archive whose CUDA major
matches the installed NVIDIA stack. ONNX Runtime 1.29 supports CUDA 12 with
cuDNN 9 and CUDA 13 with cuDNN 9. Install the NVIDIA driver, CUDA, and cuDNN
using the distribution or NVIDIA instructions, then download one official ORT
archive:

```bash
# CUDA 13, Linux x86-64
curl -fLO https://github.com/microsoft/onnxruntime/releases/download/v1.29.0/onnxruntime-linux-x64-gpu_cuda13-1.29.0.tgz
printf '%s  %s\n' \
  844c64acfc43ab9423215c26493055ea229268e28283146cc644ecef0bdae048 \
  onnxruntime-linux-x64-gpu_cuda13-1.29.0.tgz | sha256sum -c -
tar -xzf onnxruntime-linux-x64-gpu_cuda13-1.29.0.tgz

# CUDA 12 alternative: SHA-256
# 4ca594a0da83927befbd73fe020d7f569be151d70bb4fe9741ad405f4882e2ad
```

Point setup at the extracted ORT root:

```bash
ort_root="$PWD/onnxruntime-linux-x64-gpu_cuda13-1.29.0"
omaspeak setup runtime --runtime cuda --device gpu --dir "$ort_root" --apply
omaspeak setup check
omaspeak say "CUDA check" --no-play --out cuda-check.wav
```

The provider's dependencies must also be visible to the dynamic loader. A
system CUDA/cuDNN installation normally handles that. For an isolated install,
persist every required directory and rerun the probe:

```bash
omaspeak config set backend.library_dirs \
  "$ort_root/lib:/usr/local/cuda/lib64:/absolute/path/to/cudnn/lib"
omaspeak setup check
```

Use `backend.device_id` for a specific visible CUDA ordinal and string-valued
`backend.options.*` for ONNX Runtime CUDA provider options. The
[GB10 CUDA report](benchmarks/cuda-gb10-2026-09-14.md) records provider probing,
Nsight kernel placement, no-play synthesis, accuracy, and timings.

## Services

Runtime and model setup leave the user service absent. Test on demand first.
Run `omaspeak setup systemd` only when you want an always-running daemon.

Official runtime references:

- [OpenVINO Linux installation](https://docs.openvino.ai/2026/get-started/install-openvino/install-openvino-linux.html)
- [OpenVINO NPU device requirements](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html)
- [ONNX Runtime CUDA compatibility](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
