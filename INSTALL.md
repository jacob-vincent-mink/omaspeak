# Installation

## Release archive

Extract the release archive without separating the executable from its `lib/`
directory:

```bash
sha256sum -c SHA256SUMS.txt
tar -xf omaspeak-VERSION-linux-ARCH.tar.xz
install -Dm755 omaspeak-VERSION-linux-ARCH/omaspeak ~/.local/bin/omaspeak
mkdir -p ~/.local/lib/omaspeak
cp -a omaspeak-VERSION-linux-ARCH/lib/. ~/.local/lib/omaspeak/
```

When the executable is installed in `~/.local/bin`, Omaspeak discovers the
package provider in `~/.local/lib/omaspeak`. It also supports a `lib/` directory
beside the executable, which makes the extracted archive runnable in place.

Run the keyboard-driven setup:

```bash
omaspeak setup
```

To inspect the available models and detected runtimes first:

```bash
omaspeak setup model --list
omaspeak setup runtime
```

Or install the default model and launcher without a TUI:

```bash
omaspeak setup all --model supertonic-3-gguf --accept-license OpenRAIL-M
omaspeak say --no-play --out /tmp/proof.wav "Installation proof"
```

Setup downloads models only after explicit model selection and license
acceptance. It never installs a vendor runtime and never installs a systemd
unit as part of ordinary or Full setup.
The runtime screen reports detected hardware and considers CUDA GPU, Intel NPU,
Intel GPU through OpenVINO, then a Vulkan-capable GPU. It recommends the first
detected accelerator with a complete provider, or the packaged CPU fallback;
when no provider is available, it highlights the best hardware candidate and
the missing provider. This is advisory: Apply must still prove the chosen device
before saving it.

## Source build

```bash
cargo build --release --locked
```

This builds the one runtime-neutral Rust executable. It does not build a native
provider. Build the release-equivalent CPU provider separately with the pinned
script shown in the README, or configure a compatible complete audio.cpp or
OpenVINO installation with the full selection, for example
`omaspeak setup runtime --runtime openvino --device npu --dir /opt/intel/openvino --apply`.
See [Accelerator setup](ACCELERATOR_SETUP.md) for CUDA, Vulkan, HIP, and each
OpenVINO device.

For a source-tree provider, configure its directory before model installation:

```bash
omaspeak setup runtime --runtime default --device cpu \
  --dir /tmp/audio.cpp-build/bin --apply
omaspeak setup model --model supertonic-3-gguf --accept-license OpenRAIL-M
```

`--model` is the readable alias for `--download` in model setup. `--source`
accepts a pinned local file for a one-file model or the catalog directory layout
for a multi-file model. It does not bypass catalog size or SHA-256 checks.
Installation and activation are separate
commit points: if provider proof fails after verified files are installed,
Omaspeak retains those files, leaves the active config unchanged, and tells you
to configure the provider and retry with `omaspeak setup model --set MODEL`.

## Files

- Config: `${XDG_CONFIG_HOME:-~/.config}/omaspeak/config.toml`
- Models: `${XDG_DATA_HOME:-~/.local/share}/omaspeak/models/`
- Cache: `${XDG_CACHE_HOME:-~/.cache}/omaspeak/`
- Daemon state/socket: `${XDG_RUNTIME_DIR}/omaspeak/`
- Optional user service: `${XDG_CONFIG_HOME:-~/.config}/systemd/user/omaspeak.service`

`omaspeak setup menu` explicitly installs the desktop settings launcher.
`omaspeak setup systemd` explicitly installs the user service.
Once that service is active, later CLI config, runtime, model, and speaker
changes use `try-restart` so the daemon adopts the saved configuration. An
inactive or uninstalled service is left untouched.

## Audio devices

Run `omaspeak setup audio` to select and test the application’s audio device.
Pinned routing requires PipeWire’s `pw-dump` and `pw-record` (Omawake) or
`pw-play` (Omaspeak), supplied by `pipewire` and `pipewire-audio` on Arch.
See [Audio device selection](docs/AUDIO-DEVICES.md) for configuration,
service restart behavior, discovery JSON, and disconnect recovery.
