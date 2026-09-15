<p align="center">
  <img src="assets/omaspeak-mark.svg" width="128" alt="Omaspeak speaker mark">
</p>

# Omaspeak

Omaspeak is a local-first text-to-speech CLI and optional daemon. The default
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

The guided terminal uses arrow keys and Enter to choose a runtime, model, and
speaker. For an unattended default install:

```bash
omaspeak setup all --model supertonic-3-gguf --accept-license OpenRAIL-M
omaspeak say "Testing one, two, three"
printf '%s' 'Piped input works too.' | omaspeak say --no-play --out /tmp/test.wav
```

`setup all` installs the verified model and desktop settings launcher. It does
not install or start a service. `say` starts inference on demand when no daemon
is running.

The Supertonic 3 GGUF artifact is supplied by the audio.cpp model repository
at revision `09fe073ba154561f4474162e8bd4ab233a848eca`. Its original model source
is Supertone's `Supertone/supertonic-3` repository at revision
`724fb5abbf5502583fb520898d45929e62f02c0b`. Setup verifies the artifact's
pinned byte size and SHA-256, shows the OpenRAIL-M terms, and records acceptance
and both provenance identities beside the model.

## Optional native providers

Omaspeak never downloads or installs CUDA, ROCm, Vulkan drivers, or OpenVINO.
Install the desired stack yourself, obtain a complete compatible provider, and
point setup at its directory. The directory must contain the complete
`libaudiocpp.so` build and any libraries it needs; Omaspeak does not combine a
core from one build with plugins from another.

```bash
# NVIDIA GPU: complete CUDA-enabled audio.cpp provider
omaspeak setup runtime --runtime cuda --device gpu \
  --dir /opt/audiocpp-cuda --apply

# Vulkan GPU: complete Vulkan-enabled audio.cpp provider
omaspeak setup runtime --runtime vulkan --device gpu \
  --dir /opt/audiocpp-vulkan --apply

# AMD GPU: complete HIP/ROCm-enabled audio.cpp provider
omaspeak setup runtime --runtime hip --device gpu \
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

The direct OpenVINO model entries are deliberately user-supplied. Download the
official files from the pinned Supertone revision into one directory, then let
Omaspeak verify each file while installing it:

```bash
omaspeak setup model --download supertonic-3-int8 \
  --archive /path/to/official-supertonic-files \
  --accept-license OpenRAIL-M
```

Use `supertonic-3-npu` for NPU. NPU activation compiles and verifies the fixed
shape cache during setup; first inference refuses to compile a missing cache.
Omaspeak does not claim CUDA, Vulkan, HIP, GPU, or NPU placement until a
model-backed provider proof succeeds on that machine.

Inspect what setup found with:

```bash
omaspeak setup runtime
omaspeak setup runtime --json
omaspeak setup check
```

The report shows the exact provider path, search directories, capabilities,
device result, model-proof state, and remediation.

## Speakers and daemon mode

Supertonic provides ten stable speaker presets: `M1`–`M5` and `F1`–`F5`.

```bash
omaspeak voices
omaspeak say --voice F3 "A different speaker"
omaspeak config set model.voice 7
```

Daemon installation is explicit:

```bash
omaspeak setup systemd          # install and start the user unit
omaspeak setup systemd --status
omaspeak setup systemd --uninstall
```

Without that unit, every command still works on demand. If a daemon socket is
stale or refuses a connection, `say` falls back to local inference.

## Build from source

The Rust binary has no load-time dependency on audio.cpp, OpenVINO, CUDA,
Vulkan, or ROCm:

```bash
cargo build --release --locked
```

A source build does not silently install or bundle a provider. To build the
same pinned CPU provider used by releases:

```bash
git clone --recursive https://github.com/0xShug0/audio.cpp /tmp/audio.cpp
git -C /tmp/audio.cpp checkout e9ff20042ec85af960a720368c6927cda19ad65f
git -C /tmp/audio.cpp submodule update --init --recursive
./scripts/build-default-audiocpp-provider.sh /tmp/audio.cpp /tmp/audio.cpp-build
```

Copy `libaudiocpp.so.0.1.0` plus `libaudiocpp.so.0` and `libaudiocpp.so`
SONAME links beside `omaspeak` in `lib/`, or select the build directory through
setup. See [INSTALL.md](INSTALL.md), [ACCELERATOR_SETUP.md](ACCELERATOR_SETUP.md),
and [RUNTIME.md](RUNTIME.md) for the full contracts.

## License

Omaspeak source is MIT licensed. The packaged audio.cpp provider is Apache-2.0
and retains all statically linked third-party licenses. Supertonic model weights
are OpenRAIL-M and are not included in the source or release archive. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
