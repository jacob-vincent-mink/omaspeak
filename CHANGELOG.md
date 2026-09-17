# Changelog

## 0.0.1-rc.4 - 2026-09-17

- Pin Kokoro 82M as a second speech family: 54 named voices plus legacy
  numeric IDs, 24 kHz output, fresh-install proof with activation rollback,
  and paired latency evidence against the NPU Supertonic path.
- Added pinned catalog URL health checks (`setup model --check-urls`) and
  sharper offline-import diagnostics with expected/observed values.
- Proved the Spanish speech profile end to end: `model.language` reaches the
  native worker and daemon status, with Spanish synthesis recorded on the
  OpenVINO/NPU path.
- Supertonic streaming stays deferred at the current pin (unbounded session
  retention failed the buffering gate); the offline path is unchanged.
- Prove wake ownership through speech: speech and previews cannot retrigger
  wake actions, and playback cancellation releases ownership.

- Add persistent audio-device selection, PipeWire discovery, Audio setup and
  model-free device tests, dynamic schema choices, and routing diagnostics.
- Prevent pinned audio routes from falling back to another device.

## 0.0.1 - 2026-09-15

- Promoted the validated rc.3 native-provider architecture to the first stable
  release.
- Cancel daemon-backed and on-demand playback when `omaspeak say` is
  interrupted, and terminate and reap the player instead of leaving speech
  running after its client exits.
- Restart an already-active optional daemon after successful standalone config,
  runtime, model, or speaker changes while leaving inactive services alone.
- Keep optional systemd units runtime-neutral so a later backend change cannot
  inherit loader paths from the runtime that was active during installation.
- Detect PCI accelerator candidates during setup and recommend the highest
  priority candidate with a complete provider, or packaged CPU, without
  conflating detection with the provider and model proof required for readiness.
- Kept runtime transitions provider-specific: the packaged CPU provider is
  reused when leaving OpenVINO, while accelerator selections still require a
  matching complete external provider.
- Made `omaspeak say` fail immediately with usage guidance when no text is
  supplied from an interactive terminal, while preserving piped input.

## 0.0.1-rc.3 - 2026-09-15

- Made audio.cpp with Supertonic 3 GGUF the default inference provider.
- Added runtime-loaded complete audio.cpp providers for CPU, CUDA, Vulkan, and
  HIP/ROCm.
- Retained direct OpenVINO for Intel CPU, GPU, and NPU, including setup-time NPU
  cache preparation through OpenVINO's standard cache and a fresh-process
  cache-hit proof. Replaced the former converted/mixed graph entries with one
  device-neutral model containing only the official archived Supertonic 3 ONNX,
  configuration, Unicode indexer, and ten voice-style JSON files.
- Replaced archive extraction with per-file pinned downloads and made the model
  manifest part of the staged atomic installation and subsequent verification.
- Reworked guided and scriptable setup around exact provider discovery,
  ABI-only provisional setup, and model-backed file-only proof before final
  activation.
- Made runtime evidence distinguish an ABI-only, unverified audio.cpp device
  from an inaccessible device, and preserved every explicit
  `OMASPEAK_LIBRARY_PATH` dependency directory in the supervised worker.
- Fixed desktop setup launcher quoting for spaces, field codes, shell-reserved
  characters, backslashes, and Unicode executable paths.
- Removed the obsolete split runtime/plugin configuration and packaging.
- Removed unused bzip2 and tar dependencies from model setup.
- Framed NPU child results so native OpenVINO diagnostics cannot corrupt setup
  evidence, and preserved both stdout and stderr in actionable failures.
- Added the exact PocketFFT-derived FFT notice to release archives and reduced
  the package-owned audio.cpp provider to the C ABI build surface.
- Kept service installation behind the explicit `omaspeak setup systemd`
  command.
