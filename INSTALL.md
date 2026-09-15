# Installation

## Release archive

Extract the release archive without separating the executable from its `lib/`
directory:

```bash
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

Or install the default model and launcher without a TUI:

```bash
omaspeak setup all --model supertonic-3-gguf --accept-license OpenRAIL-M
omaspeak say --no-play --out /tmp/proof.wav "Installation proof"
```

Setup downloads models only after explicit model selection and license
acceptance. It never installs a vendor runtime and never installs a systemd
unit as part of ordinary or Full setup.

## Source build

```bash
cargo build --release --locked
```

This builds the one runtime-neutral Rust executable. It does not build a native
provider. Build the release-equivalent CPU provider separately with the pinned
script shown in the README, or configure a compatible complete audio.cpp or
OpenVINO installation with `omaspeak setup runtime --dir ...`.

## Files

- Config: `${XDG_CONFIG_HOME:-~/.config}/omaspeak/config.toml`
- Models: `${XDG_DATA_HOME:-~/.local/share}/omaspeak/models/`
- Cache: `${XDG_CACHE_HOME:-~/.cache}/omaspeak/`
- Daemon state/socket: `${XDG_RUNTIME_DIR}/omaspeak/`
- Optional user service: `${XDG_CONFIG_HOME:-~/.config}/systemd/user/omaspeak.service`

`omaspeak setup menu` explicitly installs the desktop settings launcher.
`omaspeak setup systemd` explicitly installs the user service.
