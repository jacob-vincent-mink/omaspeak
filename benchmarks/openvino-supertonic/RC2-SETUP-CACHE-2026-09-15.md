# RC2 OpenVINO NPU setup-cache validation — 2026-09-15

Omaspeak commit `391bdcc2d354f10b8cbcae191d69c7e0d1f3c43f` was built in
release mode and used to prepare the complete Supertonic static-shape cache on
the Dell XPS Intel NPU, followed by direct WAV synthesis. Audio playback was
disabled.

The setup command compiled and published all 12 expected blobs under cache
fingerprint
`b323436d952b827eadbfe1bb0dee755f246a4dc86137831328f6d4803ded8e48`,
then verified them from an isolated process before reporting `ready=true`. The
physical NPU busy counter increased by 454,779 microseconds. The command exited
zero, and its captured stderr was empty.

A later process selected speaker `F3`, imported the prepared graphs, and wrote
"testing one two three" directly to a WAV file. It produced 75,085 samples at
44.1 kHz (1.703 seconds of audio), loaded the model in 161 milliseconds, and
synthesized in 600 milliseconds. It reported
`effective_runtime="openvino"`, `requested_device="npu"`,
`placement_verified=true`, and `fallback_used=false`. The physical NPU busy
counter increased by another 57,678 microseconds, and synthesis stderr was
empty.

This proof specifically covers the rc.2 setup-output boundary. OpenVINO native
compiler diagnostics are drained concurrently so the child cannot block, but
successful diagnostics are not presented as application errors. If the child
fails or returns malformed evidence, Omaspeak includes the captured diagnostic
in the error.

The executable SHA-256 was
`2894d75b42144f0ae3f0d7ba98d97560e0351b66e720ca24c97426cbbd34eb7f`.
The JSON progress, final cache state, synthesis result, NPU counters, empty
stderr files, WAV checksum, source identity, and checksums are retained in
[`rc2-setup-cache-results-2026-09-15`](rc2-setup-cache-results-2026-09-15/).
The 833 MiB compiled cache remains off-repository.

The earlier [direct OpenVINO validation](DELL-XPS-DIRECT-OPENVINO-2026-09-14.md)
records CPU, iGPU, and NPU synthesis plus transcription-based accuracy. The
[rc.1 export/import proof](RC1-EXPORT-IMPORT-NPU-2026-09-15.md) records later
NPU synthesis from the prepared cache. Rc.2 changes the setup diagnostic
boundary and TLS dependency resolution; it does not change the Supertonic
graphs or frontend math.
