# Omaspeak

Omaspeak is a local text-to-speech CLI and hot-model daemon written in Rust. sherpa-onnx is the first registered backend and currently supports Piper/VITS models, with model-reported sample rates, WAV output, optional PipeWire/ALSA playback, stdin, JSON status, and config mutation.

## Build

```bash
cargo build --release
cargo test
```

## Setup

One-command network install (downloads and activates the default `en_US-lessac-medium` model, writes the config, installs the desktop launcher and the systemd user service, then runs the checks):

```bash
omaspeak setup all
```

Useful setup subcommands:

```bash
omaspeak setup model --list                 # list catalog models and install status
omaspeak setup model --download en_US-lessac-medium   # download, verify, and install a model
omaspeak setup model --verify en_US-lessac-medium     # re-verify an installed model
omaspeak setup model --download en_US-lessac-medium --archive /path/to/vits-piper-en_US-lessac-medium.tar.bz2
omaspeak setup check                        # verify config, backend, model, engine, audio, launcher, service
omaspeak setup runtime                      # show registered backends, compiled runtime capabilities, and models
omaspeak setup systemd --status             # show systemd user service status
omaspeak setup systemd --uninstall          # remove the systemd user service
omaspeak setup menu --status                # show desktop launcher status
omaspeak setup menu --uninstall             # remove the desktop launcher
```

`--archive` installs from an already-downloaded pinned archive instead of fetching it; it is still size and SHA256 verified before extraction. Downloaded archives and installed model assets are size + SHA256 verified against the pinned catalog, and installs are atomic: a verified extraction is staged, then renamed into place, with the previous model restored on failure.

## Synthesis

Standalone synthesis loads a model for the request:

```bash
omaspeak say "Hello Omarchy" --no-play --out hello.wav
printf '%s' 'Text from stdin' | omaspeak say --no-play
```

The daemon loads once and serializes requests through one inference engine:

```bash
omaspeak daemon
omaspeak status --json
omaspeak say "The hot model serves this request."
omaspeak stop
```

`speed` is constrained to 0.25 through 4.0, text is size-bounded, speaker IDs are checked against model metadata, and empty or non-finite synthesis is rejected.

## Backends

Omaspeak uses the same runtime/device matrix as Omawake: `default` with `auto|cpu`, `cuda` with `auto|gpu`, and OpenVINO with individual `cpu|gpu|npu` or `AUTO`, `HETERO`, and `MULTI` device lists. The CPU release fails closed for unavailable providers unless `fallback = "cpu"`; a fallback is warned and reported. OpenVINO and CUDA require provider-specific builds.

See [DEMO.md](DEMO.md) for cold, hot, concurrent, and fallback results.
