<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/omaspeak-mark.svg">
    <source media="(prefers-color-scheme: light)" srcset="assets/omaspeak-mark-on-light.svg">
    <img alt="Omaspeak: a stylized speaker with sound waves" src="assets/omaspeak-mark-on-light.svg" width="160">
  </picture>
</p>

# Omaspeak

Omaspeak is a local Supertonic text-to-speech CLI and hot-model daemon written
in Rust, with model-reported sample rates, selectable voices, WAV output,
optional PipeWire/ALSA playback, stdin, JSON status, and config mutation.

## Install

The 0.0.1-rc.2 release supports Linux x86-64 and aarch64 with glibc 2.35 or newer. It ships one
executable plus a ready-to-use CPU runtime. Download the archive and
`SHA256SUMS.txt` from the
[GitHub release](https://github.com/jacob-vincent-mink/omaspeak/releases/tag/v0.0.1-rc.2),
then verify, unpack, and configure it:

```bash
sha256sum --check --ignore-missing SHA256SUMS.txt
tar -xJf omaspeak-0.0.1-rc.2-linux-x86_64.tar.xz
cd omaspeak-0.0.1-rc.2-linux-x86_64
./omaspeak setup all --accept-license OpenRAIL-M
./omaspeak say "Installation complete" --no-play --out test.wav
```

Move the application to its final location before setup because a desktop
launcher or an explicitly requested service records the executable path. See
[INSTALL.md](INSTALL.md) for a per-user installation, source builds, and
external OpenVINO or CUDA setup.
[ACCELERATOR_SETUP.md](ACCELERATOR_SETUP.md) gives complete Intel iGPU/NPU and
NVIDIA CUDA recipes.

## Build

```bash
cargo build --release
cargo test
```

Run the source build as `target/release/omaspeak`. Building requires stable
Rust. The release archive supplies the native CPU library; a source checkout
requires a compatible external ONNX Runtime or a copied release `lib/`
directory.

Omaspeak does not link ONNX Runtime or OpenVINO into the executable. Every
build supports CPU, OpenVINO, and CUDA. Build the same runtime-neutral
executable for every deployment:

```bash
cargo build --release
```

The normal Linux release archive includes a working CPU default under `lib/`:
the official ONNX Runtime 1.30.0 CPU library. An unpacked release therefore
needs only a Supertonic model for CPU inference. OpenVINO setup loads a
user-supplied OpenVINO runtime directly. CUDA setup keeps the packaged ORT core
and points the same executable at Microsoft's standalone CUDA Plugin EP plus
its vendor runtime. Omaspeak validates each selected runtime before it creates
the TTS engine. The release does not bundle acceleration libraries.

The Rust Supertonic frontend handles text, voice styles, diffusion, and audio
assembly for both ONNX Runtime and direct OpenVINO execution. The
[current GB10 report](benchmarks/cuda-gb10-ort130-2026-09-15.md) validates the
official ORT 1.30 CUDA Plugin EP with direct WAV output and physical GPU
telemetry. An earlier ORT 1.29 run is retained as historical evidence. The
[Dell XPS OpenVINO report](benchmarks/openvino-supertonic/DELL-XPS-DIRECT-OPENVINO-2026-09-14.md)
records direct CPU, iGPU, and NPU execution and accuracy, and the
[rc.2 NPU setup proof](benchmarks/openvino-supertonic/RC2-SETUP-CACHE-2026-09-15.md)
validates clean setup-time compiled-model export and second-process import.

## Setup

Run setup in a terminal for the guided interface:

```bash
omaspeak setup
```

Use the arrow keys and Enter to choose `Full setup`, `Runtime`, `Model`, or
`Check`. Full setup walks through the runtime, compatible device, and model,
then shows a summary before it changes the config, downloads model assets, or
installs the desktop launcher. It does not install, enable, or start a service.
Full setup safely restarts an already active service after Apply so it loads
the new configuration; an inactive service remains inactive. Focused runtime
and model setup saves the selection without managing the service:

```bash
omaspeak setup runtime   # choose a runtime/device and optionally point at its native bundle
omaspeak setup runtime --dir /opt/omaspeak-runtime  # preview and probe a flat bundle or SDK root
omaspeak setup runtime --dir /opt/omaspeak-runtime --apply  # save only after the probe passes
omaspeak setup model     # browse catalog metadata and install/activate a model
```

Pre-release configuration files use the current schema only. If setup finds an
invalid file, it starts from current defaults and tells you that the next
successful apply will replace the file. Inventory, checks, and cancelled setup
leave the original bytes untouched. Runtime commands and `omaspeak config`
remain strict, so a malformed or unknown field cannot silently affect speech.

The model picker marks the active model, models already installed, models that
need license acceptance, and models that must be supplied by the user. It includes
download sizes, the Supertonic model family, backend names, license status,
language information, and Intel NPU validation. OpenVINO and CUDA remain
selectable even before their external native stack is configured, with setup
guidance for supplying it. An exact OpenVINO NPU setup only offers catalog
models validated for that device.

For scripts or redirected input/output, setup remains noninteractive. This
one-command network install explicitly accepts the default Supertonic 3
model's OpenRAIL-M terms, downloads and activates the model, writes the config,
installs the desktop launcher, and then runs the checks. It does not install or
start a service:

```bash
omaspeak setup all --accept-license OpenRAIL-M
```

If the daemon is already active, `setup all` safely restarts it after the model
and configuration changes succeed. It never starts an inactive service.

Useful setup subcommands:

```bash
omaspeak setup model --list                 # catalog models, license status, and install status
omaspeak setup model --download supertonic-3-int8 --accept-license OpenRAIL-M
omaspeak setup model --download supertonic-3-npu --accept-license OpenRAIL-M
omaspeak setup model --verify supertonic-3-int8       # re-verify an installed model
omaspeak setup cache --prepare             # compile and verify the selected NPU cache plan
omaspeak setup check                        # verify config, model metadata, runtime/device, audio, launcher; report optional service
omaspeak setup systemd                      # explicitly install, enable, and start the systemd user service
omaspeak setup runtime --json               # read-only runtime/device inventory with paths, evidence, and remediation
omaspeak setup runtime --dir /path/to/sdk   # preview and probe exact libraries without changing config
omaspeak setup runtime --dir /path/to/sdk --apply  # save a successfully probed candidate
omaspeak setup systemd --status             # show systemd user service status
omaspeak setup systemd --uninstall          # remove the systemd user service
omaspeak setup menu --status                # show desktop launcher status
omaspeak setup menu --uninstall             # remove the desktop launcher
```

`--archive` installs from an already-downloaded pinned archive instead of
fetching it; it is still size and SHA256 verified before extraction. Supertonic
catalog installs require `--accept-license OpenRAIL-M`, including installs from
a local archive, and the guided TUI presents a separate acceptance step before
any download. The exact pinned license is written as `MODEL-LICENSE` beside the
weights. `.omaspeak-model.json` records the immutable source revision, archive
and supplement hashes, whether the source was a catalog download or
user-supplied archive, and the acceptance time. A model can also declare
checksum-pinned supplemental assets; these are downloaded even with `--archive`
unless they are already in Omaspeak's verified download cache. Downloaded
archives, supplements, and installed model assets are size + SHA256 verified
against the pinned catalog. Installs are atomic: extraction and supplements are
prepared in a staging directory, then renamed into place, with the previous
model restored on failure.

Native libraries can be supplied with the TOML path list
`backend.library_dirs` or the colon-separated `OMASPEAK_LIBRARY_PATH` overlay.
For deterministic selection, set `backend.onnxruntime_library` and, for CUDA,
`backend.provider_library`. Their environment equivalents are
`OMASPEAK_ONNXRUNTIME_LIBRARY` and `OMASPEAK_PROVIDER_LIBRARY`. Configured exact
paths take precedence over exact
environment paths. Directory search order is configured directories, the
`OMASPEAK_LIBRARY_PATH` overlay, package directories, the ambient loader path,
then system libraries. This makes the bundled CPU stack the automatic default
while a configured external CUDA plugin wins deterministically. Direct OpenVINO
libraries use their separately configured paths.
Relative TOML paths resolve beside the config file; environment overrides must
be absolute. Omaspeak checks `lib/` beside the executable and
`../lib/omaspeak` for package libraries. Before
an engine-loading command on Linux, it validates the selected files and their
dependencies and re-executes once with effective directories prepended to
`LD_LIBRARY_PATH`. Config and setup discovery commands do not re-exec.

`setup runtime --dir` recognizes libraries directly in the chosen directory
and common SDK layouts under `lib`, `lib64`, `runtime/lib/intel64`, and
`runtime/lib/intel64/Release`. It previews and probes the resolved candidate in
an isolated process. Add `--apply` to save it. Failed probes and previews leave
the config byte-for-byte unchanged.

`omaspeak setup runtime --json` reports every supported runtime/device and
distinguishes discovery, configuration, loadability, device access, and
readiness. It includes exact paths, provenance, probe evidence, errors, and
remediation without changing the config. An explicit `omaspeak setup systemd` writes only that
app-owned effective path into the unit; it never copies the caller's ambient
`LD_LIBRARY_PATH`.

The packaged runtime contract is summarized in [RUNTIME.md](RUNTIME.md).

Supertonic 3 int8 is the default model. The catalog pins its official archive
and verifies each of its four ONNX graphs, TTS
metadata, Unicode indexer, and voice-style bundle. Installing it activates all
required filenames automatically:

```bash
omaspeak setup model --download supertonic-3-int8 --accept-license OpenRAIL-M
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
omaspeak setup model --download supertonic-3-npu --accept-license OpenRAIL-M
omaspeak config set backend.runtime openvino
omaspeak config set backend.device npu
```

The direct OpenVINO backend submits all four Supertonic graphs to the selected
device. Use the `supertonic-3-npu` catalog entry on NPU; its FP32 vector
estimator avoids the accuracy failure seen with the fully INT8 model.

Supertonic supports `en`, `ko`, `ja`, `ar`, `bg`, `cs`, `da`, `de`, `el`,
`es`, `et`, `fi`, `fr`, `hi`, `hr`, `hu`, `id`, `it`, `lt`, `lv`, `nl`,
`pl`, `pt`, `ro`, `ru`, `sk`, `sl`, `sv`, `tr`, `uk`, and `vi`. The configured
voice is a Supertonic speaker-style index from `voice.bin`; `model.steps`
controls its denoising iterations. The bundled style file exposes `M1` through
`M5` as IDs `0` through `4`, followed by `F1` through `F5` as IDs `5` through
`9`.

List every voice exposed by the active model and mark the configured default:

```bash
omaspeak voices
omaspeak voices --json
```

Guided model setup asks for a default voice after the model choice. It can also
be changed directly, while `say --voice` overrides it for one request by name
or numeric ID:

```bash
omaspeak config set model.voice 7
omaspeak say --voice F3 "Testing another speaker" --no-play --out voice-f3.wav
```

## Synthesis

Standalone synthesis loads a model for the request:

```bash
omaspeak say "Hello from Omaspeak" --no-play --out hello.wav
printf '%s' 'Text from stdin' | omaspeak say --no-play
```

For repeatable file-only measurements, load the engine once and write every
warmup and measured synthesis without playback:

```bash
omaspeak benchmark --text "Hello from Omaspeak" --out-dir benchmark \
  --warmup 2 --iterations 10 > benchmark.json

# Benchmark a specific speaker without changing the configured default.
omaspeak benchmark --text "Hello from Omaspeak" --voice F5 --out-dir voice-f5 \
  --warmup 2 --iterations 10 > voice-4.json
```

The JSON includes model load time, every output path and synthesis/wall timing,
audio duration, real-time factor, and p50/p95 summaries.

When no daemon is listening, `say` loads the model for that request and exits.
For repeated synthesis, start the daemon in one terminal; it loads once and
serializes requests through one inference engine:

```bash
omaspeak daemon
```

Use a second terminal for control and synthesis commands:

```bash
omaspeak status --json
omaspeak say "The hot model serves this request."
omaspeak stop
```

`speed` is constrained to 0.25 through 4.0, text is size-bounded, speaker IDs are checked against model metadata, and empty or non-finite synthesis is rejected.

## Backends

Omaspeak supports `default` with `auto|cpu`, `cuda` with `auto|gpu`, and
OpenVINO with `auto|cpu|gpu|npu`.
Runtime validation fails closed for missing or incompatible external libraries
unless `fallback = "cpu"`; a fallback is warned and reported. Every release
binary exposes CPU, OpenVINO, and CUDA from the same link-free executable.

With `runtime = "cuda"`, Omaspeak verifies its exact ONNX Runtime 1.30.0 core,
registers the selected standalone CUDA Plugin EP DSO, and checks that the
requested NVIDIA device is accessible. `backend.device_id` selects its logical
CUDA ordinal (for example, `0` for the first visible GPU). Inventory evidence
also reports the runtime's hardware ID for each enumerated device.
String-valued `[backend.options]` entries are forwarded to ONNX
Runtime's CUDA execution-provider option map, which supports settings such as
`cudnn_conv_algo_search`, `gpu_mem_limit`, `arena_extend_strategy`, and
`do_copy_in_default_stream`. Omaspeak defaults the cuDNN search to `HEURISTIC`
to avoid ONNX Runtime's expensive exhaustive search during cold loads.
`backend.options.device_id` is reserved; use the
typed `backend.device_id` setting instead. Relative paths resolve beside the
Omaspeak config file. Supertonic submits all four component graphs to CUDA.
Setup accepts a directory containing only `libonnxruntime_providers_cuda.so`;
it reuses the packaged ORT core and never downloads or installs accelerator
software.

Provider and model option maps can also be managed without editing TOML:

```bash
omaspeak config set backend.options.gpu_mem_limit 4294967296
omaspeak config set backend.options.cudnn_conv_algo_search HEURISTIC
omaspeak config unset backend.options.gpu_mem_limit
```

With `runtime = "openvino"`, Omaspeak loads `libopenvino_c.so` at runtime and
runs Supertonic's four public ONNX graphs through OpenVINO directly. Set exact
runtime files in the config when automatic directory discovery is unsuitable:

```toml
[backend]
openvino_library = "/opt/intel/openvino/runtime/lib/intel64/libopenvino_c.so"
openvino_plugins = "/opt/intel/openvino/runtime/lib/intel64/plugins.xml"
```

`omaspeak setup runtime --dir` accepts an OpenVINO installation root and scans
its common `runtime/lib/intel64[/Release]` layouts for the C API library and
plugin catalog. The backend verifies that the selected physical device is
available, specializes each graph to the request's concrete tensor shapes, and
checks `EXECUTION_DEVICES` after compilation. OpenVINO's compiled-model cache
lives below `$XDG_CACHE_HOME/omaspeak/openvino/`. NPU setup prepares and
verifies a fixed 12-blob shape plan before activation; normal NPU synthesis
refuses an absent or mismatched cache instead of compiling on the first
request. See [ACCELERATOR_SETUP.md](ACCELERATOR_SETUP.md).

Entries in `[backend.options]` are passed through as OpenVINO properties for
the selected device. `CACHE_DIR` and `INFERENCE_NUM_THREADS` are managed by
Omaspeak; set `backend.threads` for the CPU thread count. For example:

```toml
[backend.options]
PERFORMANCE_HINT = "LATENCY"
NPU_PLATFORM = "5010"
```

The fully INT8 `supertonic-3-int8` vector estimator did not pass NPU speech
accuracy validation. The NPU-specific `supertonic-3-npu` catalog entry uses the
official FP32 vector estimator and is the validated NPU choice. The Dell XPS
validation checks both output accuracy and physical device use: CPU and NPU
each achieved 0% WER over the same 25-word suite, while OpenVINO placement and
the NPU busy counter independently confirmed NPU execution.

See [DEMO.md](DEMO.md) for a current usage walkthrough. Direct OpenVINO device
placement and accuracy evidence is in the
[Dell XPS direct OpenVINO report](benchmarks/openvino-supertonic/DELL-XPS-DIRECT-OPENVINO-2026-09-14.md)
and [rc.2 NPU setup proof](benchmarks/openvino-supertonic/RC2-SETUP-CACHE-2026-09-15.md).
The [2026-09-13 predecessor demo](benchmarks/historical/2026-09-13-piper.md)
and [earlier GB10 CUDA report](benchmarks/cuda-gb10-2026-09-14.md) are retained
only as historical evidence.

## License

Omaspeak source is licensed under the [MIT License](LICENSE). Bundled runtime
components and downloaded models remain under their own licenses; see
[third-party notices](THIRD_PARTY_NOTICES.md). Model terms are shown before
download and are not covered by Omaspeak's MIT license.
