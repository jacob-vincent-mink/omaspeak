# Historical pre-export/import OpenVINO NPU smoke — 2026-09-14

This report predates Omaspeak's deterministic compiled-model export/import
cache. It remains useful inference and placement evidence, but it does not
validate the 0.0.1-rc.1 setup-time cache contract.

Omaspeak commit `61ac63478e8e741fa84665d3d6cb9054a1d321d5` was
built in release mode and validated on the Dell XPS against the system
OpenVINO 2026.3.1 installation. The executable SHA-256 was
`502ab2ed6ef150cd183e1f150675e3eef5a6a060ac566dbf6196a28d2bafe5b6`.
All output was written directly to WAV files; no audio was played.

The read-only inventory returned all eight supported runtime/device rows. Its
OpenVINO/NPU row loaded the configured external runtime, enumerated CPU, GPU,
and NPU, selected NPU, and reported ready. An explicit
`setup runtime --runtime openvino --device npu --dir /usr` preview left the
configuration SHA-256 unchanged. Repeating the command with `--apply` passed
the isolated probe and persisted the canonical library and `plugins.xml`
paths.

The direct-file benchmark used one warmup and one measured synthesis of “The
quick brown fox jumps over the lazy dog.” It reported
`effective_runtime=openvino`, `fallback_used=false`, and
`placement_verified=true`; every compiled graph's OpenVINO
`EXECUTION_DEVICES` value matched NPU. Model loading took 63 ms. The measured
synthesis took 74.84 ms for 3.260 seconds of 44.1 kHz mono PCM audio, an RTF of
0.02295. The kernel NPU busy counter increased by 126,757 microseconds across
the run.

Compact JSON, hashes, NPU counters, and the self-verifying SHA-256 manifest are
in
[`final-head-smoke-results-2026-09-14`](final-head-smoke-results-2026-09-14/).
The WAV files remain outside the repository under
`/tmp/omaspeak-61ac634-dell-proof-host/audio`.
