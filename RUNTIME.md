Omaspeak ships one executable and CPU ONNX Runtime 1.30.0 under lib/.
The Rust Supertonic frontend uses this runtime automatically without configured
native paths. The release includes no sherpa, Piper, eSpeak or accelerator DSOs.

`omaspeak setup runtime --json` is a read-only inventory of every supported
runtime/device. It distinguishes supported, discovered, configured, loadable,
device_accessible and ready, and reports exact paths, source, evidence, errors
and remediation. Runtime/device readiness does not prove model placement.

OpenVINO requires an external libopenvino_c.so, plugins.xml and Intel plugins.
CUDA reuses that core and requires Microsoft's standalone CUDA Plugin EP plus
matching NVIDIA dependencies.
Setup never downloads, builds, copies or installs native runtimes.

`omaspeak setup runtime --runtime openvino --device npu --dir /absolute/runtime`
previews and probes a candidate in an isolated process. Add `--apply` to save
after success. Guided setup offers an Apply/Cancel review. Failures and
cancellation leave config unchanged; ambient LD_LIBRARY_PATH is never saved.

Install models separately with explicit OpenRAIL-M acceptance. Only
`omaspeak setup systemd` installs or starts the optional user service.

See `ACCELERATOR_SETUP.md` for tested Arch/Omarchy Intel packages, NPU
setup-time cache preparation, and the official standalone CUDA Plugin EP.
