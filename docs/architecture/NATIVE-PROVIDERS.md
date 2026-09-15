# Native providers

Omaspeak owns setup, configuration, daemon behavior, model provenance, audio
output, and reporting. Inference is selected as one coherent native provider.

| Provider | Model form | Devices | Delivery |
| --- | --- | --- | --- |
| audio.cpp | Supertonic 3 GGUF | CPU, CUDA, Vulkan, HIP/ROCm | Pinned CPU provider in releases; complete external providers for accelerators |
| Direct OpenVINO | Official Supertonic graphs | Intel CPU, GPU, NPU | User-installed OpenVINO C runtime; setup downloads hash-pinned official files after license acceptance |

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

The direct catalog has one device-neutral `supertonic-3-openvino` entry. Its 16
files come directly from `supertone-oss-archive/supertonic-3` revision
`aafc6e32416a594460b32413efc49d7fe4ce6d46`. Setup downloads each file into a
separate verified cache, writes the complete provenance and license acceptance
manifest inside the model staging directory, verifies that manifest, and then
publishes the directory with one rename. Missing or tampered manifests make the
installation unready.

The release builds audio.cpp commit
`e9ff20042ec85af960a720368c6927cda19ad65f` with only the Supertonic model and
CPU backend. The release contract stages the stripped versioned library and
SONAME links, includes every applicable license, and rejects native runtime
load dependencies from the Rust executable.
