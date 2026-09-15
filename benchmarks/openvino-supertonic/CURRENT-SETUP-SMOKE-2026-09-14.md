# Historical setup and NPU synthesis smoke — 2026-09-14

This report predates Omaspeak's deterministic compiled-model export/import
cache. It is retained as historical evidence and does not validate the
0.0.1-rc.1 setup-time cache contract.

The post-stabilization Omaspeak tree was rebuilt in release mode and exercised
against the Dell XPS OpenVINO 2026.3.1 installation. The executable SHA-256 was
`0aa0793d30a7f34c838c67b0437ae581dfcdbed7777a800b202505d3b1a2ccb8`.
No audio was played.

`setup runtime --json` returned the complete eight-row runtime/device matrix.
The OpenVINO/NPU row was discovered from the system, loadable, device-accessible,
and ready. Its isolated probe enumerated `CPU`, `GPU`, and `NPU`, selected `NPU`,
and reported OpenVINO 2026.3.1.

The explicit candidate command was:

```console
omaspeak --config /tmp/omaspeak-current-hardware-20260914/dell/config.toml \
  setup runtime --runtime openvino --device npu --dir /usr
```

The preview reported `applied=false` and did not change the config SHA-256.
Repeating it with `--apply` reported `applied=true` and saved the canonical
OpenVINO library and `plugins.xml` paths. A deliberately invalid explicit
runtime directory was then attempted with `--apply`; it exited 1 and the config
SHA-256 remained unchanged. This verifies both preview and rejected-apply
atomicity on the current executable.

A direct file benchmark performed one warmup and one measured synthesis of
“The quick brown fox jumps over the lazy dog.” The result reported
`effective_runtime=openvino`, `requested_device=npu`, `fallback_used=false`, and
`placement_verified=true`; every compiled graph's OpenVINO `EXECUTION_DEVICES`
matched NPU. The measured synthesis took 72.52 ms for 3.260 seconds of 44.1 kHz
mono PCM audio (RTF 0.02224). The independent kernel NPU busy counter increased
by 125,129 microseconds across the run.

The read-only `setup check` runtime, device, and engine rows passed. Its overall
exit was 1 because this older local model installation predates the now-required
`MODEL-LICENSE` file. The same model's complete inference graph loaded and
synthesized successfully in the direct benchmark.

Compact machine-readable evidence is in
[`current-setup-smoke-results-2026-09-14`](current-setup-smoke-results-2026-09-14).
Raw WAVs remain under `/tmp/omaspeak-current-hardware-20260914/dell`.
