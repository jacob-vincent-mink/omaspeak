# Native runtime contract

Omaspeak is one Rust executable with runtime-loaded native providers.

The default release includes one stripped CPU-only `libaudiocpp.so.0.1.0` built
from audio.cpp commit `e9ff20042ec85af960a720368c6927cda19ad65f`, with SONAME
`libaudiocpp.so.0` and the normal SONAME links. The executable dynamically loads
the public audio.cpp C ABI. It does not invoke the audio.cpp CLI. Omaspeak
re-executes itself as a hidden supervised worker to isolate native faults and
reuse a loaded session.

CUDA, Vulkan, and HIP/ROCm selections point to a complete external audio.cpp
provider. Omaspeak passes the exact backend and device ID requested in config.
It never assembles plugins into a differently built provider and never installs
vendor software.

OpenVINO is a separate direct provider. Omaspeak loads the external OpenVINO C
API and plugins manifest, owns the Supertonic pre/post-processing, and keeps
NPU compilation in setup. NPU persistence uses OpenVINO's standard `CACHE_DIR`
mechanism, followed by a fresh-process cache-hit proof before activation.

Search order is:

1. exact configured library;
2. configured application-owned library directories;
3. `OMASPEAK_LIBRARY_PATH` directories;
4. package directories beside the executable and under its prefix.

The ambient system loader remains responsible for a provider's dependencies.
Runtime inventory reports exact paths and missing configured directories. The
release executable itself must have no load-time dependency on any inference
runtime or accelerator library.

Runtime JSON distinguishes discovery from execution proof. For audio.cpp,
`device_accessible` is `null` after an ABI-only probe because no model session
has exercised the device yet. It becomes `true` only after setup completes a
model-backed file-only synthesis. `model_inference_verified` records the same
proof explicitly.
