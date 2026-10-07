# Incremental speech playback

Normal `say` requests now play generated audio as it becomes available, both
with the daemon and with on-demand inference. `--no-play` still produces a
complete WAV without opening an audio output or acquiring a wake pause.
The public request/reply protocol remains version 1.

| Provider | Incremental generation | Boundary |
| --- | --- | --- |
| audio.cpp Supertonic, pinned provider | Native pull events through the C ABI | The provider's existing text chunks |
| Direct OpenVINO Supertonic CPU/GPU/NPU | Emit existing frontend chunks as each vocoder call finishes | Existing device-specific chunks and inter-chunk silence |
| audio.cpp Kokoro | Separate bounded synthesis requests on the warm worker | At most 240 Unicode characters per text segment |
| Kokoro OpenVINO CPU/GPU/NPU | Separate bounded calls to the existing native bridge | At most 240 Unicode characters per text segment |
| Older audio.cpp provider without the optional streaming symbols | Bounded text segments using its offline API | At most 240 characters |
| Paradee OpenVINO CPU FP32 | Completed native frontend/phoneme segments | At most 510 phonemes; CPU-only experimental profile |
| ElevenLabs / OpenAI-compatible / Cartesia / Deepgram | HTTP response PCM as it arrives | Network audio chunks |
| Other implementations of `TtsBackend` | Default complete-buffer fallback | Override `generate_stream` to deliver earlier audio |

Kokoro's current APIs return a complete waveform for each segment. Segmenting
text enables early playback of longer input but can change prosody at boundaries.
This is not sample-by-sample generation. Short single-chunk requests still wait
for that chunk. No language, voice, speed or accelerator restrictions are relaxed.

`TtsBackend::generate_stream` receives a fallible sink of ordered mono float32
PCM. `Engine::synthesize_stream` validates samples, writes the staged WAV, and
hands at most 4096 samples per call to playback. Direct Supertonic uses the same
seeded RNG, shape selection, and silence as complete generation. audio.cpp reads
named audio from each native event, frees the event, and discards the merged
final result to avoid playing the same waveform twice. Older offline providers
remain compatible because streaming symbols are loaded as an optional group.

The supervised synthesis worker owns the raw PCM player. The default output uses
`pw-play`; `aplay` is used only if `pw-play` cannot be found before any audio is
sent. A pinned PipeWire target uses its configured properties and never reroutes.
Playback failure is reported without replaying already-heard speech.

Small IPC frames and OS pipes provide backpressure. Omaspeak does not queue the
entire streamed waveform in Rust. Native providers may retain their own merged
result internally; this change does not promise bounded memory inside audio.cpp.
A closed player pipe stops further generation. The worker owns an acknowledged
Omawake pause hold before writing the first audio, keeps it through player drain,
and passes the descriptor to the player for parent-death cleanup.

Cancellation, disconnect, timeout, or daemon shutdown terminates the worker's
process group, including native inference and the player. The next request
loads a fresh worker from the frozen configuration. On-demand speech uses the
same worker supervision and handles SIGINT/SIGTERM. Successful requests publish
the staged WAV only after synthesis and playback have finished. Failure or
cancellation removes the staged file and preserves the previous destination.
Already played audio cannot be retracted.

`synthesis_milliseconds` includes streaming delivery/backpressure and excludes
final player drain. A successful response follows player drain and WAV publication.

See [verification](../benchmarks/results/2026-10-07-streaming/RESULTS.md).
