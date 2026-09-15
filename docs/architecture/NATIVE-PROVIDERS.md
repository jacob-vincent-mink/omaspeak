# Native providers

Omaspeak owns setup, configuration, daemon behavior, model provenance, audio
output, and reporting. Inference is selected as one coherent native provider.

| Provider | Model form | Devices | Delivery |
| --- | --- | --- | --- |
| audio.cpp | Supertonic 3 GGUF | CPU, CUDA, Vulkan, HIP/ROCm | Pinned CPU provider in releases; complete external providers for accelerators |
| Direct OpenVINO | Official Supertonic graphs | Intel CPU, GPU, NPU | User-installed OpenVINO C runtime and user-supplied, hash-verified official model files |

The audio.cpp integration uses its public C ABI directly. Omaspeak does not run
or require the audio.cpp CLI. A hidden Omaspeak worker process loads the native
library, validates the ABI, creates the requested backend/device session, and
keeps it warm. Framed bounded IPC separates native crashes from the CLI and
daemon.

Setup never installs a vendor runtime. A runtime-only selection with no model
can establish ABI compatibility, but it must report that model proof remains.
Any model activation and any provider change with an installed model must pass
a real file-only synthesis before configuration is saved. Direct OpenVINO NPU
also prepares and verifies all static shape blobs during setup.

The release builds audio.cpp commit
`e9ff20042ec85af960a720368c6927cda19ad65f` with only the Supertonic model and
CPU backend. The release contract stages the stripped versioned library and
SONAME links, includes every applicable license, and rejects native runtime
load dependencies from the Rust executable.
