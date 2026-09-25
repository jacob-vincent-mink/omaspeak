<p align="center">
  <img src="assets/omaspeak-mark.svg" width="128" alt="Omaspeak speaker mark">
</p>

# Omaspeak

Omaspeak is a local-first Linux text-to-speech CLI and optional daemon. The default
release uses Supertonic 3 GGUF through a small, package-owned audio.cpp CPU
provider. Optional CUDA, Vulkan, HIP/ROCm, and direct OpenVINO providers are
selected at setup time from software already installed by the user.

Omaspeak integrates audio.cpp as a library: it dynamically loads audio.cpp's
public C ABI and never invokes or depends on the audio.cpp CLI. Native inference
runs in a hidden supervised worker made by re-executing the Omaspeak binary.
That worker isolates native failures and keeps one model session warm across
requests.

## Install and run the default provider

Download the archive for your architecture from Releases, extract it, and put
`omaspeak` and its `lib/` directory together in one installation prefix. Then:

```bash
omaspeak setup
```

The first page shows the detected hardware recommendation, runtime, model,
speaker, and output device. Press Left or Right to select **Use recommended
settings** or **Customize**, then Enter to continue. A separate **Accept setup**
page starts model download and any NPU cache compilation only after acceptance.
The same recommendation is available
as `omaspeak setup --recommended --accept-license OpenRAIL-M` when the selected
model requires that license and is not already installed. It detects accelerator hardware and considers CUDA GPU, Intel NPU,
Intel GPU through OpenVINO, then a Vulkan-capable GPU. It recommends the first
candidate with a complete provider, or the packaged CPU fallback. If no provider
is present, it highlights the best hardware candidate and explains what is
missing. Hardware and provider detection are shown separately; only Apply runs
the isolated and model-backed checks that establish readiness. For an unattended
default install:

```bash
omaspeak setup all --model supertonic-3-gguf --accept-license OpenRAIL-M
omaspeak say "Testing one, two, three"
printf '%s' 'Piped input works too.' | omaspeak say --no-play --out /tmp/test.wav
```

When the text argument is omitted, `say` reads redirected or piped standard
input. An interactive invocation with no text exits immediately with usage
guidance. Pressing Ctrl-C cancels playback. When a daemon handles the request,
it watches the client connection and stops its player as soon as the client
exits.

`setup all` installs the verified model and desktop settings launcher. It does
not install or start a service. `say` starts inference on demand when no daemon
is running.

To repeat a full TUI Apply without changing your active configuration, run
`python3 scripts/verify-setup-tui.py --binary ~/.local/bin/omaspeak --model-cache ~/.local/share/omaspeak/models/supertonic-3-openvino`.
The script verifies the cached model first, then uses isolated XDG directories
to check the final Accept page, model proof, NPU cache preparation when
applicable, config, and launcher.

Browse the same runtime and model catalog without changing the machine:

```bash
omaspeak setup runtime
omaspeak setup model --list
```

The Supertonic 3 GGUF conversion is supplied by the audio.cpp model repository
at revision `09fe073ba154561f4474162e8bd4ab233a848eca`. The archived upstream
model source is `supertone-oss-archive/supertonic-3` at revision
`aafc6e32416a594460b32413efc49d7fe4ce6d46`. Setup verifies every downloaded
byte, shows the OpenRAIL-M terms, and records the conversion and upstream
provenance beside the model.

## Optional native providers

Omaspeak never downloads or installs CUDA, ROCm, Vulkan drivers, or OpenVINO.
Install the desired stack yourself, obtain a complete compatible provider, and
point setup at its directory. The directory must contain the complete
`libaudiocpp.so` build and any libraries it needs; Omaspeak does not combine a
core from one build with plugins from another.

```bash
# NVIDIA GPU: complete CUDA-enabled audio.cpp provider
omaspeak setup runtime --runtime cuda --device gpu \
  --device-id 0 \
  --dir /opt/audiocpp-cuda --apply

# Vulkan GPU: complete Vulkan-enabled audio.cpp provider
omaspeak setup runtime --runtime vulkan --device gpu \
  --device-id 0 \
  --dir /opt/audiocpp-vulkan --apply

# AMD GPU: complete HIP/ROCm-enabled audio.cpp provider
omaspeak setup runtime --runtime hip --device gpu \
  --device-id 0 \
  --dir /opt/audiocpp-hip --apply
```

If the model is installed, Apply runs a real file-only synthesis before saving.
If it is not installed, setup saves only an ABI-validated provider selection
and states that Full setup or model installation must complete the proof. Model
activation also performs that synthesis before the configuration is saved.

Direct OpenVINO uses Omaspeak's own Supertonic pre/post-processing and the
OpenVINO C API. Point setup at an OpenVINO installation containing
`libopenvino_c` and `plugins.xml`:

```bash
omaspeak setup runtime --runtime openvino --device gpu \
  --dir /opt/intel/openvino --apply
omaspeak setup runtime --runtime openvino --device npu \
  --dir /opt/intel/openvino --apply
```

Install the direct OpenVINO model from the catalog after reviewing its license:

```bash
omaspeak setup model --model supertonic-3-openvino \
  --accept-license OpenRAIL-M
```

Omaspeak downloads the four official ONNX graphs, `tts.json`,
`unicode_indexer.json`, and all ten official voice-style JSON files directly
from `supertone-oss-archive/supertonic-3` at revision
`aafc6e32416a594460b32413efc49d7fe4ce6d46`. Every URL, byte size, and SHA-256
is pinned. The files and a canonical provenance/license-acceptance manifest are
published together in one atomic install. Use `--source /path/to/files` only as
an offline override with the same directory layout and pinned bytes.

Use `supertonic-3-openvino` for Intel CPU, GPU, and NPU. NPU activation compiles
and verifies the fixed shape cache during setup; first inference refuses to
compile a missing cache.
`kokoro-82m-openvino` uses OpenVINO GenAI 2026.4 or newer on Intel CPU, GPU,
and NPU. It currently includes `af_heart` and `am_michael` voices and needs an
OpenVINO GenAI Python environment. See the [setup and NPU evidence](docs/KOKORO-OPENVINO-2026.4.md).
Omaspeak does not claim CUDA, Vulkan, HIP, GPU, or NPU placement until a
model-backed provider proof succeeds on that machine.

Inspect what setup found with:

```bash
omaspeak setup runtime
omaspeak setup runtime --json
omaspeak setup check
```

The report shows the exact provider path, search directories, capabilities,
device result, detected PCI hardware, recommended configuration, model-proof
state, and remediation. Detection does not install a vendor runtime or claim
that an unproved device is ready.
The [rc.3 hardware results](benchmarks/results/2026-09-15-rc3/RESULTS.md)
compare default CPU, OpenVINO CPU/iGPU/NPU, and CUDA using file-only output.

## Speakers and daemon mode

Supertonic provides ten stable speaker presets: `M1`–`M5` and `F1`–`F5`.

```bash
omaspeak voices
omaspeak say --voice F3 "A different speaker"
omaspeak config set model.voice F3
```

Preset names also work in `model.voice` and IPC. Existing numeric IDs retain
their identity; see [named voices](docs/NAMED-VOICES.md).

Daemon installation is explicit:

```bash
omaspeak setup systemd          # install and start the user unit
omaspeak setup systemd --no-start # install and enable it without starting it
omaspeak setup systemd --status
omaspeak setup systemd --uninstall
```

Without that unit, speech commands still run inference on demand. If a daemon socket is
stale or refuses a connection, `say` falls back to local inference.
When the optional unit is installed and active, successful `config set`,
`config unset`, runtime Apply, and model or speaker activation restart it with
`systemctl --user try-restart`. An installed but inactive unit remains inactive,
and Full setup performs only its existing single transactional restart. The
unit does not embed runtime library paths; workers resolve the saved backend
configuration after each restart.

## Build from source

The Rust binary has no load-time dependency on audio.cpp, OpenVINO, CUDA,
Vulkan, or ROCm:

```bash
cargo build --release --locked
```

A source build does not silently install or bundle a provider. To build the
same pinned CPU provider used by releases:

```bash
git clone https://github.com/0xShug0/audio.cpp /tmp/audio.cpp
git -C /tmp/audio.cpp checkout e9ff20042ec85af960a720368c6927cda19ad65f
./scripts/build-default-audiocpp-provider.sh /tmp/audio.cpp /tmp/audio.cpp-build
```

Copy `libaudiocpp.so.0.1.0` plus `libaudiocpp.so.0` and `libaudiocpp.so`
SONAME links beside `omaspeak` in `lib/`, or select the build directory through
setup before installing the model:

```bash
omaspeak setup runtime --runtime default --device cpu \
  --dir /tmp/audio.cpp-build/bin --apply
omaspeak setup model --model supertonic-3-gguf --accept-license OpenRAIL-M
```

Runtime setup records an ABI-only selection when the model is absent. Model
setup then runs the file-only synthesis proof and saves activation. If model
installation succeeds but that proof fails, the installed files are retained,
the active config remains unchanged, and the error gives the exact `--set`
command to retry after runtime setup.

See [INSTALL.md](INSTALL.md), [ACCELERATOR_SETUP.md](ACCELERATOR_SETUP.md),
and [RUNTIME.md](RUNTIME.md) for the full contracts.

## License

Omaspeak source is MIT licensed. The packaged audio.cpp provider is Apache-2.0,
and the release includes notices for all code retained in it, including the
BSD-3-Clause PocketFFT-derived FFT. Supertonic model weights are OpenRAIL-M and
are not included in the source or release archive. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Audio devices

Run `omaspeak setup audio` to select and test the application’s audio device.
Pinned routing requires PipeWire’s `pw-dump` and `pw-record` (Omawake) or
`pw-play` (Omaspeak), supplied by `pipewire` and `pipewire-audio` on Arch.
See [Audio device selection](docs/AUDIO-DEVICES.md) for configuration,
service restart behavior, discovery JSON, and disconnect recovery.
