# RC1 OpenVINO NPU export/import validation — 2026-09-15

Omaspeak commit `e634c3c547e9c8bd62ebe3ae61e458679fedbeec` was
built in release mode and exercised on the Dell XPS Intel Series 3 NPU. The
host's OpenVINO 2026.3.1 runtime exposed `CPU`, `GPU`, and `NPU`; Omaspeak's
isolated probe selected `NPU` with `fallback = "error"`. All synthesis wrote
WAV files directly, with no audio playback.

This run validates the deterministic compiled-model mechanism added after the
earlier direct OpenVINO report. Omaspeak explicitly exports each compiled model
with the OpenVINO C API, publishes the cache only after all 12 expected blobs
exist, and starts a second process that imports every blob before marking the
cache ready. It does not rely on OpenVINO's nondeterministic implicit
`CACHE_DIR` population.

## Setup and first use

After the application version changed to rc.1, `say` refused the obsolete
cache fingerprint and directed the user to `setup cache --prepare`; it did not
compile during inference. Running that explicit setup command took about 36.2
seconds, increased the physical NPU busy counter by 443,852 microseconds, and
published exactly 12 blobs totaling 872,482,542 bytes under fingerprint
`a6f065d02b689d50f5a4df30d76c4dcc4dcfa8fe9e5fa8964104b8d0705724a5`.
The command's required second-process import verification passed before setup
reported `ready = true`.

A later process used named speaker `F3` and imported the prepared models. It
produced 187,154 mono 44.1 kHz PCM samples, or 4.244 seconds of audio. Model
load took 166 ms and synthesis took 799 ms. The kernel NPU busy counter rose
from 463,474,005 to 463,555,612 microseconds, a physical-device delta of
81,607 microseconds. The output WAV SHA-256 was
`061add3041dab111585864bd289efe8f95e2615e098de206be1de119656930c3`.

The compiled-model transport change does not alter the Supertonic graphs,
weights, shapes, or frontend math. The earlier
[direct OpenVINO validation](DELL-XPS-DIRECT-OPENVINO-2026-09-14.md) compares
default CPU and NPU across four phrases and records the same per-phrase WER for
both lanes, with zero NPU-minus-CPU WER delta. This rc.1 run adds proof that the
new explicit export/import cache preserves functional NPU synthesis and keeps
compilation in setup.

## Artifacts

The x86-64 release executable SHA-256 was
`9a1394a06851b457f837030fd2586557c262ff9fef624d7923157ea4de1a02d0`.
The cache manifest SHA-256 was
`2a6b4c5192c6f4f6fe97b241c280abae6d9174bcee74f3ccd9d2ebc0b355943b`.
Compact evidence is tracked in
[`rc1-export-import-results-2026-09-15`](rc1-export-import-results-2026-09-15/):
the runtime inventory, cache status, synthesis JSON, empty stderr, NPU counters,
environment summary, and checksums. The 833 MiB compiled cache and WAV remain
off-repository.
