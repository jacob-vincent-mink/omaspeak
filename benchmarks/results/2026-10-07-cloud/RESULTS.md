# Cloud adapters and combined Paradee verification — 2026-10-07

Implemented TTS: ElevenLabs HTTP streaming, OpenAI-compatible speech,
Cartesia bytes (API version 2026-08-14), Deepgram Aura HTTP speech.
Implemented ASR: Deepgram prerecorded HTTP and OpenAI-compatible file
transcription, with bounded local energy endpointing and process supervision.
These ASR adapters are utterance-based; WebSocket realtime providers remain
future work. Protocol references and setup are linked from each repository's
`docs/CLOUD.md`.

No paid API requests or microphone capture were used. Loopback HTTP fixtures
with fake credentials verified routes, auth, request bodies, fragmented PCM,
first-chunk playback, HTTP errors/redirects, body redaction and output preservation.
A stalled stream cancellation closed HTTP, stopped playback, kept the daemon
responsive and preserved the destination. ASR fixtures verified bounded WAV
uploads, whole-phrase matching, silence exclusion, final-only responses and
error redaction. Process fixtures verified nonblocking live acceptance, bounded
queues, no duplicate detections and immediate disposal of stalled workers.
Vendor account/voice availability, billing, perceptual quality, false-wake rates
and real-room latency are not qualified by these fixtures.

## Checks

The initial combined-worktree checks passed 772 tests including private-bus
tests. The independent audio branches omit the pre-existing consumer-events/
D-Bus implementation, its zbus/Tokio dependencies, the Omawake ABI relaxation,
and the Omaspeak GPU precision experiment. Independent branch verification passed 368 Omaspeak tests and 400 Omawake
tests, strict Clippy (`--all-targets -- -D warnings`), formatting and diff checks.

- The only added direct Rust dependency is `url`, already in both lockfiles as
  a transitive dependency. There is no new transitive dependency or inference
  runtime. ONNX Runtime remains excluded from delivery.

## Paradee through the combined CLI

[paradee-cli.json](paradee-cli.json) records actual OpenVINO CPU FP32 execution
with separately built native eSpeak NG 1.52.0, isolated configuration and a
silent raw-PCM capture player. File output returned finite native audio. A
long request delivered its first PCM at 1.35 seconds and finished at 2.88
seconds; captured PCM matched the saved WAV. These are cold debug-build checks,
not a comparative performance benchmark or a perceptual qualification.

The experimental profile is CPU-only. The initial integration uses the native
American eSpeak-to-Misaki fallback, with pronunciation limits documented in
[PARADEE.md](../../../docs/PARADEE.md). The qualified model assets and compiled
execution placement are pinned/verified by the catalog and provider. See the
[separate native probe results](../2026-10-07-paradee/RESULTS.md) and the
[audio.cpp pickup note](../../../docs/AUDIOCPP-PARADEE-FOLLOWUP.md).
