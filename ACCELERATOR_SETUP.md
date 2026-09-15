# Accelerator setup

Omaspeak 0.0.1-rc.1 ships one runtime-neutral executable and a small default CPU
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

The release-provided `libonnxruntime.so.1.30.0` is found beside the executable
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

Omaspeak continues to use the ONNX Runtime 1.30.0 CPU core shipped in its
release package, then registers Microsoft's separately released CUDA Plugin EP.
Install the NVIDIA driver, matching CUDA toolkit, and cuDNN using the
distribution or NVIDIA instructions. Download the plugin archive matching the
machine architecture and installed CUDA major; the archive contains the
provider DSO and no second ONNX Runtime core.

```bash
# CUDA 13, Linux x86-64
curl -fLO https://github.com/microsoft/onnxruntime/releases/download/plugin-ep-cuda/v0.1.0/cuda_ep_cuda13_0.1.0_linux-x64.tar.gz
printf '%s  %s\n' \
  5fa5cc5b19843809707818302771e4d16b740df069ca63908b28245e5f6b8398 \
  cuda_ep_cuda13_0.1.0_linux-x64.tar.gz | sha256sum -c -
mkdir -p "$HOME/.local/share/omaspeak/runtimes/cuda13"
tar -C "$HOME/.local/share/omaspeak/runtimes/cuda13" \
  -xzf cuda_ep_cuda13_0.1.0_linux-x64.tar.gz

```

Other official v0.1.0 Linux assets:

| CUDA | Architecture | Archive | SHA-256 |
|---|---|---|---|
| 12 | x86-64 | [`cuda_ep_cuda12_0.1.0_linux-x64.tar.gz`](https://github.com/microsoft/onnxruntime/releases/download/plugin-ep-cuda/v0.1.0/cuda_ep_cuda12_0.1.0_linux-x64.tar.gz) | `dc34a4450e1b352671235205fb7d865c56ae61c7f8631df33ae2a369d4d1dcab` |
| 13 | aarch64 | [`cuda_ep_cuda13_0.1.0_linux-aarch64.tar.gz`](https://github.com/microsoft/onnxruntime/releases/download/plugin-ep-cuda/v0.1.0/cuda_ep_cuda13_0.1.0_linux-aarch64.tar.gz) | `d02f9d438df1ad2cfc770e9eb93094710c1713d6a23f6bdc0f532ed83eb4b5f4` |

Point setup at the extracted plugin directory. Setup probes those files in
place and saves their paths only after a successful probe; it does not copy or
install the plugin, toolkit, driver, or cuDNN.

```bash
cuda_plugin="$HOME/.local/share/omaspeak/runtimes/cuda13"
omaspeak setup runtime --runtime cuda --device gpu --dir "$cuda_plugin" --apply
omaspeak setup check
omaspeak say "CUDA check" --no-play --out cuda-check.wav
```

The provider's dependencies must also be visible to the dynamic loader. A
system CUDA/cuDNN installation normally handles that. For an isolated install,
persist every required directory and rerun the probe:

```bash
omaspeak config set backend.library_dirs \
  "$cuda_plugin:/usr/local/cuda/lib64:/absolute/path/to/cudnn/lib"
omaspeak setup check
```

Use `backend.device_id` for a specific visible CUDA ordinal and string-valued
`backend.options.*` for ONNX Runtime CUDA provider options.

## Services

Runtime and model setup leave the user service absent. Test on demand first.
Run `omaspeak setup systemd` only when you want an always-running daemon.

Official runtime references:

- [OpenVINO Linux installation](https://docs.openvino.ai/2026/get-started/install-openvino/install-openvino-linux.html)
- [OpenVINO NPU device requirements](https://docs.openvino.ai/2026/openvino-workflow/running-inference/inference-devices-and-modes/npu-device.html)
- [CUDA Plugin EP v0.1.0](https://github.com/microsoft/onnxruntime/releases/tag/plugin-ep-cuda/v0.1.0)
- [CUDA Plugin EP quick start](https://github.com/microsoft/onnxruntime/blob/main/docs/cuda_plugin_ep/QUICK_START.md)
