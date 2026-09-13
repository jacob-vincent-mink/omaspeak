# Omaspeak

Omaspeak is a local text-to-speech CLI and hot-model daemon written in Rust. It embeds sherpa-onnx and currently supports Piper/VITS models, model-reported sample rates, WAV output, optional PipeWire/ALSA playback, stdin, JSON status, and config mutation.

## Build

```bash
cargo build --release
cargo test
```

Copy `config.example.toml` to `${XDG_CONFIG_HOME:-$HOME/.config}/omaspeak/config.toml`. Download and extract [Piper `en_US-lessac-medium`](https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models) beneath `${XDG_DATA_HOME:-$HOME/.local/share}/omaspeak/models/`, or set `model.directory` to its absolute path.

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
