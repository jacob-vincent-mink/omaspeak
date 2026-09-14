# Omaspeak

Omaspeak is a local text-to-speech CLI and hot-model daemon written in Rust. sherpa-onnx is the first registered backend and currently supports Piper/VITS and Supertonic models, with model-reported sample rates, WAV output, optional PipeWire/ALSA playback, stdin, JSON status, and config mutation.

## Build

```bash
cargo build --release
cargo test
```

The default build uses sherpa-onnx's implicit static CPU runtime. An OpenVINO
build must link to a separately built shared sherpa-onnx stack whose ONNX
Runtime includes the OpenVINO Execution Provider:

```bash
export SHERPA_ONNX_LIB_DIR=/absolute/path/to/openvino-enabled-sherpa-onnx/lib
cargo build --release --features openvino
```

Supertonic currently also requires the ORT 1.29 zero-element tensor patch in
[`native/openvino/patches`](native/openvino/patches).
Stock ORT 1.29 OpenVINO EP aborts when Supertonic passes a zero-element tensor
across a provider partition boundary. The pinned, reproducible native build is
documented in [`native/openvino`](native/openvino/README.md). It also carries
the sherpa patch for selecting OpenVINO independently for each Supertonic
component.

At runtime, make the matching ONNX Runtime, sherpa-onnx, and OpenVINO shared
libraries discoverable (for example with their setup script or
`LD_LIBRARY_PATH`). Intel's OpenVINO runtime alone is insufficient: ONNX
Runtime itself must have been built with the OpenVINO EP. The upstream
sherpa-onnx shared release is CPU-only, so it can exercise shared linking but
cannot establish OpenVINO placement.

## Setup

One-command network install (downloads and activates the default `en_US-lessac-medium` model, writes the config, installs the desktop launcher and the systemd user service, then runs the checks):

```bash
omaspeak setup all
```

Useful setup subcommands:

```bash
omaspeak setup model --list                 # list catalog models and install status
omaspeak setup model --download en_US-lessac-medium   # download, verify, and install a model
omaspeak setup model --download supertonic-3-int8     # multilingual int8 evaluation model
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

Piper remains the default CPU model. The catalog also pins the official
Supertonic 3 int8 archive and verifies each of its four ONNX graphs, TTS
metadata, Unicode indexer, and voice-style bundle. Installing it activates all
required filenames automatically:

```bash
omaspeak setup model --download supertonic-3-int8
omaspeak config set model.language en
omaspeak config set model.steps 5
```

Supertonic supports `en`, `ko`, `ja`, `ar`, `bg`, `cs`, `da`, `de`, `el`,
`es`, `et`, `fi`, `fr`, `hi`, `hr`, `hu`, `id`, `it`, `lt`, `lv`, `nl`,
`pl`, `pt`, `ro`, `ru`, `sk`, `sl`, `sv`, `tr`, `uk`, and `vi`. The configured
voice remains the speaker index from `voice.bin`; `model.steps` controls its
denoising iterations.

## Synthesis

Standalone synthesis loads a model for the request:

```bash
omaspeak say "Hello Omarchy" --no-play --out hello.wav
printf '%s' 'Text from stdin' | omaspeak say --no-play
```

For repeatable file-only measurements, load the engine once and write every
warmup and measured synthesis without playback:

```bash
omaspeak benchmark --text "Hello Omarchy" --out-dir benchmark \
  --warmup 2 --iterations 10 > benchmark.json
```

The JSON includes model load time, every output path and synthesis/wall timing,
audio duration, real-time factor, and p50/p95 summaries.

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

With `runtime = "openvino"`, Omaspeak passes sherpa an absolute
`openvino:/.../provider.config` provider string. If `provider_config` is set,
the path must name an existing regular file; relative paths resolve beside the
Omaspeak config file, and the file contents are used unchanged. Otherwise
Omaspeak atomically writes a mode-0600 config below
`$XDG_STATE_HOME/omaspeak/cache/openvino/<device>/`, with uppercase
`device_type`, an isolated compiled-model `cache_dir`, and
`enable_qdq_optimizer=True` for `NPU`. String-valued `[backend.options]` entries
override generated defaults after syntax validation. `device_type` must match
the selected device and `cache_dir` remains application-managed. Session-level
keys such as `ProfilingFilePrefix` pass through. Omaspeak defaults
`disable_dynamic_shapes=True` on NPU because the official Supertonic int8
graphs require concrete input shapes; `[backend.options]` can override it.
Host-specific OpenVINO hardware properties belong in ONNX Runtime's inline
`load_config` JSON; Omaspeak does not guess or generate them. For example:

```toml
[backend.options]
load_config = '{"NPU":{"NPU_PLATFORM":"5010"}}'
```

The current full-model Supertonic NPU path does not pass speech-accuracy
validation. On exact `NPU`, Omaspeak therefore defaults Supertonic to the
largest validated mixed placement: `duration_predictor,text_encoder,vocoder`
on OpenVINO NPU and `vector_estimator` on ORT CPU. The reserved
`SherpaOnnx.SupertonicComponents` backend option can override that allowlist for
experiments. Hardware evaluation must include output-accuracy checks in
addition to successful execution and device activity, so the catalog does not
yet mark Supertonic as generally NPU-capable.
On exact `GPU`, the validated default places only `vector_estimator` on the
OpenVINO GPU and runs the other components on ORT CPU; it also sets `FP32`,
disables the QDQ optimizer, and disables dynamic shapes. Other GPU component
sets either failed compatibility checks or produced inaccurate audio on the
tested Intel stack. On exact `CPU`, all four components use OpenVINO and dynamic
shapes are disabled. All generated defaults remain overridable through
`[backend.options]` for controlled experiments.

The feature and provider config make OpenVINO available to the runtime; they do
not prove that a model was placed on a particular device. Status continues to
report accelerated placement as unverified until independent provider or
device evidence is collected.

See [DEMO.md](DEMO.md) for cold, hot, concurrent, and fallback results.
