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

Run setup in a terminal for the guided interface:

```bash
omaspeak setup
```

Use the arrow keys and Enter to choose `Full setup`, `Runtime`, `Model`, or
`Check`. Full setup walks through the runtime, compatible device, and model,
then shows a summary before it changes the config, downloads model assets, or
installs the desktop launcher and enables and starts the systemd user service. The runtime and model
flows can also be opened directly:

```bash
omaspeak setup runtime   # choose a compiled runtime, then a compatible device
omaspeak setup model     # browse catalog metadata and install/activate a model
```

The model picker marks the active model, models already installed, and models
available to download. It includes download sizes, model families, backend
names, language information, and Intel NPU validation. Unavailable runtimes
remain visible with their build requirement, and an exact OpenVINO NPU setup
only offers catalog models validated for that device.

For scripts or redirected input/output, setup remains noninteractive. This
one-command network install downloads and activates the default
`en_US-lessac-medium` model, writes the config, installs the desktop launcher
and systemd user service, then runs the checks:

```bash
omaspeak setup all
```

Useful setup subcommands:

```bash
omaspeak setup model --list                 # list catalog models and install status
omaspeak setup model --download en_US-lessac-medium   # download, verify, and install a model
omaspeak setup model --download supertonic-3-int8     # multilingual int8 evaluation model
omaspeak setup model --download supertonic-3-npu      # validated Intel NPU model mix
omaspeak setup model --verify en_US-lessac-medium     # re-verify an installed model
omaspeak setup model --download en_US-lessac-medium --archive /path/to/vits-piper-en_US-lessac-medium.tar.bz2
omaspeak setup check                        # verify config, backend, model, engine, audio, launcher, service
omaspeak setup runtime --json               # machine-readable runtimes, devices, capabilities, and models
omaspeak setup systemd --status             # show systemd user service status
omaspeak setup systemd --uninstall          # remove the systemd user service
omaspeak setup menu --status                # show desktop launcher status
omaspeak setup menu --uninstall             # remove the desktop launcher
```

`--archive` installs from an already-downloaded pinned archive instead of fetching it; it is still size and SHA256 verified before extraction. A model can also declare checksum-pinned supplemental assets; these are downloaded even with `--archive` unless they are already in Omaspeak's verified download cache. Downloaded archives, supplements, and installed model assets are size + SHA256 verified against the pinned catalog. Installs are atomic: extraction and supplements are prepared in a staging directory, then renamed into place, with the previous model restored on failure.

Piper remains the default CPU model. The catalog also pins the official
Supertonic 3 int8 archive and verifies each of its four ONNX graphs, TTS
metadata, Unicode indexer, and voice-style bundle. Installing it activates all
required filenames automatically:

```bash
omaspeak setup model --download supertonic-3-int8
omaspeak config set model.language en
omaspeak config set model.steps 5
```

For Intel NPU use, `supertonic-3-npu` combines the same pinned INT8 archive
with Supertone's official FP32 vector estimator. Its download is about 368 MiB
and its installed model is about 309 MiB. This avoids a numerical failure in
the INT8 vector graph on the tested Panther Lake NPU, where 22 of 38 dynamic
INT8 MatMul rescaling outputs became zero. Omaspeak verifies the supplemental
model's 256,534,781-byte payload against SHA256
`883ac868ea0275ef0e991524dc64f16b3c0376efd7c320af6b53f5b780d7c61c`
from Supertone's immutable `724fb5abbf5502583fb520898d45929e62f02c0b`
revision, and removes the superseded INT8 vector graph before activating the
model.

```bash
omaspeak setup model --download supertonic-3-npu
omaspeak config set backend.runtime openvino
omaspeak config set backend.device npu
```

For this catalog entry, the generated NPU provider config submits all four
Supertonic components to OpenVINO. An explicit
`backend.options.SherpaOnnx.SupertonicComponents` value still overrides that
default. The original `supertonic-3-int8` catalog entry and custom Supertonic
models retain the validated mixed placement.

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

The fully INT8 `supertonic-3-int8` vector estimator does not pass NPU
speech-accuracy validation. On exact `NPU`, that entry and custom Supertonic
models therefore default to the largest validated mixed placement:
`duration_predictor,text_encoder,vocoder` on OpenVINO NPU and
`vector_estimator` on ORT CPU. The NPU-specific `supertonic-3-npu` catalog entry
uses the official FP32 vector estimator and submits all four components to
OpenVINO; it is marked `npu_capable=true`. The reserved
`SherpaOnnx.SupertonicComponents` backend option can override either default.
Hardware evaluation must include output-accuracy checks in addition to
successful execution and device activity.
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
