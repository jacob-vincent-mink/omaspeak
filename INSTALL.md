# Installing Omaspeak

Omaspeak supports Linux x86-64 and aarch64. Each release archive contains one
runtime-neutral executable and a bundled CPU ONNX Runtime. OpenVINO and CUDA
remain external runtime choices configured after installation.

Linux release CI produces x86-64 and aarch64 archives. There are no prebuilt
artifacts for macOS or Windows. CUDA has been validated on aarch64 NVIDIA GB10
hardware.

## Release archive

Download `omaspeak-0.0.1-rc-linux-x86_64.tar.xz` and `SHA256SUMS.txt` from the
[v0.0.1-rc release](https://github.com/jacob-vincent-mink/omaspeak/releases/tag/v0.0.1-rc),
then verify and unpack it:

```bash
sha256sum --check --ignore-missing SHA256SUMS.txt
tar -xJf omaspeak-0.0.1-rc-linux-x86_64.tar.xz
cd omaspeak-0.0.1-rc-linux-x86_64
./omaspeak --version
```

You can run Omaspeak from the unpacked directory. To install it for one user
while preserving runtime discovery:

```bash
install -Dm755 omaspeak "$HOME/.local/bin/omaspeak"
mkdir -p "$HOME/.local/lib/omaspeak"
cp -a lib/. "$HOME/.local/lib/omaspeak/"
```

Ensure `$HOME/.local/bin` is on `PATH`, then install the default model with the
guided setup or one command:

```bash
omaspeak setup

# Scriptable equivalent; review and accept the model's OpenRAIL-M terms.
omaspeak setup all --accept-license OpenRAIL-M
omaspeak say "Installation complete" --no-play --out test.wav
```

Setup installs a desktop launcher. It does not install, enable, or start a
systemd service. `say` starts the configured engine on demand when no daemon is
listening.

Distribution packages may place the disabled vendor unit from
`packaging/systemd/omaspeak.service` under `/usr/lib/systemd/user`. Installing
that file does not enable or start the daemon.

## OpenVINO or CUDA

Install the vendor runtime, then point setup at its root or library directory.
For CUDA, use the official standalone CUDA Plugin EP directory; the packaged
ONNX Runtime core remains selected.

```bash
omaspeak setup runtime --runtime openvino --device npu \
  --dir /opt/intel/openvino --apply

omaspeak setup runtime --runtime cuda --device gpu \
  --dir /opt/omaspeak-cuda-runtime --apply
```

The probe must pass before configuration is saved. Omaspeak does not download
or install accelerator runtimes. Use `omaspeak setup runtime --json` for exact
library, device, and remediation details.

See [ACCELERATOR_SETUP.md](ACCELERATOR_SETUP.md) for tested Arch/Omarchy Intel
iGPU and NPU packages, setup-time NPU cache compilation, and official CUDA
Plugin EP downloads and checksums.

## Build from source

Install a Rust toolchain and run:

```bash
git clone https://github.com/jacob-vincent-mink/omaspeak.git
cd omaspeak
cargo build --release --locked
cargo test --locked
```

The source-built executable is runtime-neutral and does not contain the CPU
library shipped in the release archive. Supply a compatible ONNX Runtime
through setup or copy the release `lib/` directory beside the executable.
