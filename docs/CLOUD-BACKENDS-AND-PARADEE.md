# Cloud backend shortlist and Paradee assessment

Assessed 2026-10-07 against the current worktree and pinned audio.cpp
`e9ff20042ec85af960a720368c6927cda19ad65f`. Cloud APIs were surveyed from primary
vendor documentation; no paid API requests, cost comparison, or cloud quality
benchmarks were performed. The HTTP TTS adapters and bounded Deepgram/OpenAI-compatible ASR adapters are
now implemented; see [cloud configuration](CLOUD.md). The remaining realtime
ASR and demand-driven vendors remain candidates. Local streaming is implemented separately in [STREAMING.md](STREAMING.md).

## Omaspeak cloud choices

| Priority | Backend | Why include it | Initial integration |
| --- | --- | --- | --- |
| First | ElevenLabs | Requested provider; named voices and low-latency models fit replies and notifications | HTTP audio streaming with a configurable model and provider voice ID; request PCM at an explicitly supported sample rate |
| First | OpenAI-compatible speech endpoint | One transport can support OpenAI and explicitly qualified self-hosted speech servers | `/v1/audio/speech`, WAV/PCM, configurable base URL, model and voice; compatibility must be tested per server |
| Next | Cartesia | A focused alternative with streamed output and voice/speed controls | `/tts/bytes`; pin the API version and model, currently documented as `2026-08-14` and `sonic-3.6` |
| Next when sharing a Deepgram account matters | Deepgram Aura | Both synthesis and recognition with one provider | `/v1/speak`, initially an Aura-2 preset and explicitly requested linear PCM; WebSocket text streaming later |
| Demand-driven | Google Cloud, Azure Speech, Amazon Polly | Useful for existing vendor accounts, regional deployment, SSML and language requirements | Provider-specific adapters and credential handling; avoid implementing three additional authentication stacks before demand exists |

Sources: [ElevenLabs HTTP streaming](https://elevenlabs.io/docs/api-reference/text-to-speech/stream),
[ElevenLabs models](https://elevenlabs.io/docs/overview/capabilities/text-to-speech),
[official OpenAI text-to-speech documentation](https://developers.openai.com/api/docs/guides/text-to-speech),
[Cartesia bytes](https://docs.cartesia.ai/api-reference/tts/bytes),
[Deepgram request](https://developers.deepgram.com/reference/text-to-speech/speak-request),
[Google Chirp 3 HD](https://docs.cloud.google.com/text-to-speech/docs/chirp3-hd),
[Polly synthesis](https://docs.aws.amazon.com/polly/latest/APIReference/API_SynthesizeSpeech.html).
Azure remains demand-driven; this assessment does not qualify a specific Azure
voice, region, transport or plan.

For ElevenLabs, `eleven_flash_v2_5` is a reasonable first low-latency profile;
newer expressive models should be configurable and benchmarked. Vendor inference
latency excludes network and playback overhead. Prefer full-text HTTP audio
streaming for ordinary `say`. WebSockets are useful later for text arriving
incrementally from a consumer, which the current request protocol does not yet
accept. [ElevenLabs explains this distinction](https://elevenlabs.io/docs/eleven-api/concepts/audio-streaming).

The new PCM sink is the integration point. Cloud setup must resolve a usable
model and voice without requiring a local model directory or native-library
probe. Cloud placement must be reported as remote rather than CPU/GPU. Provider
voice identifiers need provider-specific resolution before the current integer
voice index reaches generation. Unsupported language/speed/style requests must
fail clearly. Keep credentials in environment/credential references, with
redacted diagnostics and typed endpoint/model/voice settings. Bound HTTP/audio
buffers and deadlines; preserve cancellation, staging, pause ownership and the
rule against replaying a request after partial audio delivery. Streaming an API
response into a complete vector would lose the latency benefit.

## Omawake cloud choices

Omawake currently performs local VAD, utterance transcription and local phrase
matching. A cloud backend would replace transcription, not the action policy.

| Priority | Backend | Fit |
| --- | --- | --- |
| First | Deepgram streaming ASR | Explicit final results and endpointing map well onto phrase detection |
| Next | ElevenLabs Scribe v2 Realtime | Useful shared account with Omaspeak; explicit committed transcripts and manual/VAD commit choices |
| Alternative | Soniox | Consider when target multilingual phrases justify its token/finalization protocol |
| Optional interoperability | OpenAI-compatible file transcription | Upload bounded VAD-completed clips; simpler transport but waits for utterance completion. Realtime transcription is a separate protocol, not implied by file-endpoint compatibility |
| Separate experiment | Azure keyword verification or Picovoice Porcupine | Azure combines local keywords with cloud verification/STT; Porcupine is on-device inference with vendor credentials, not cloud ASR |

Sources: [Deepgram live audio](https://developers.deepgram.com/reference/speech-to-text/listen-streaming),
[endpointing and final results](https://developers.deepgram.com/docs/understand-endpointing-interim-results/),
[Scribe realtime protocol](https://elevenlabs.io/docs/api-reference/speech-to-text/v-1-speech-to-text-realtime),
[Soniox transcription](https://soniox.com/docs/stt/rt/real-time-transcription),
[official OpenAI realtime transcription](https://developers.openai.com/api/docs/guides/realtime-transcription),
[Azure keyword recognition](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/keyword-recognition-overview),
[Porcupine](https://github.com/Picovoice/porcupine).

Keep remote ASR opt-in and local detection as the default. VAD only removes
silence: uploading every VAD segment sends unrelated speech too. A local keyword
candidate followed by cloud verification is a separate mode with a different
accuracy/latency tradeoff. Retain local normalization, whole-phrase matching,
cooldowns and direct argument-vector actions; never run actions on provisional
transcripts. Accumulate finalized segments until a completed utterance and map
provider timestamps to the capture timeline. Network work belongs in supervised
bounded workers, not the capture callback. Reconnects must discard stale utterance
state and never duplicate a detection. Qualification should measure false wakes,
missed phrases, end-to-detection latency and uploaded/billed audio on real phrase
recordings, rather than general ASR leaderboards.

## Paradee: available now versus future work

[Paradee-8M-v1.0](https://huggingface.co/sahilmahendrakar/Paradee-8M-v1.0)
is an Apache-2.0 English/American, single-voice distillation of Kokoro's `af_heart`.
The release supplies INT8 ONNX (~9 MB) and FP32 ONNX (~37 MB), not GGUF.
The tested immutable revision is `8f34b01ef8adcb0bec470b89bf2fd62a7b2369e6`.

| Route | Evidence today | What remains |
| --- | --- | --- |
| Upstream ONNX Runtime CPU | Official Python package explicitly selects CPU. Both released graphs produced finite audio in an isolated ONNX Runtime 1.30.0 probe | Reference probe only. Do not add or ship ONNX Runtime: retain the existing audio.cpp/OpenVINO delivery architecture |
| OpenVINO CPU, FP32 | Unmodified official graph imported, compiled and inferred on installed OpenVINO 2026.4 | Experimental native adapter and pinned catalog are now implemented: see [Paradee](PARADEE.md). Native eSpeak-to-Misaki fallback is exercised; perceptual/pronunciation qualification remains |
| OpenVINO CPU, INT8 | Import succeeds but compile fails because an internal tensor has no static element type. Static input shape and ONNX type inference did not fix it | Investigate dynamic quantization lowering or produce a separately qualified OpenVINO quantized export; use FP32 for initial development |
| OpenVINO GPU | Official graph fails: default precision has a mixed FP16/FP32 Add; forcing FP32 exposes unsupported 3D linear interpolation. A prototype wrapping the two linear Resize operations in Unsqueeze/Squeeze to 4D compiled and inferred | Pin a conversion pipeline, validate numerics, pronunciation, device placement and benefit. One seeded ORT comparison had max absolute sample difference ~0.00104; this is not a quality qualification |
| OpenVINO NPU | Released dynamic INT8 graph failed with unbounded-dimension diagnostics and SIGSEGV inside `libopenvino_intel_npu_compiler.so` | Separate/re-export text-side and decoder graphs with bounded/static duration and audio shapes; qualify operators and phase filter, and investigate the compiler crash in isolation. No NPU support claim |
| Existing audio.cpp / GGML runtimes | Pinned provider has no Paradee family; current upstream source tree also showed no Paradee paths in this survey | Architecture-aware conversion, tensor mapping, duration expansion, decoder and phase-filter implementation. Existing Kokoro support does not load Paradee weights |
| Existing Kokoro OpenVINO GenAI adapter | Requires the specific Kokoro pipeline/artifact interface | Paradee's phoneme-ID graph needs a different adapter; changing model paths cannot provide support |
| Hugging Face Inference Providers | Model card reports no deployed Inference Provider | Dedicated hosting requires a custom handler/server; Hub hosting is not a ready TTS API |
| Custom hosted endpoint | Upstream CPU inference can be wrapped in a service | An OpenAI-compatible server plus Omaspeak's future remote adapter; deployment and latency qualification remain separate |

The input contract is `input_ids: int64[1,T]` and `speed: float32[1]`; output is
mono `waveform: float32[1,samples]` at 24 kHz. Include pad ID 0 at both ends and
split at 510 phonemes instead of silently truncating. There is no voice/style
input: expose only `af_heart`, English and supported speed. The graph contains
the phase correction filter and generates a complete waveform for each supplied
phoneme chunk. Initial streaming should therefore deliver completed sentence/
phoneme segments through the new sink; frame-level streaming would need new work.

The important frontend requirement is Misaki spelling. Upstream Python uses
Misaki English G2P plus its eSpeak fallback. The author's
[eSpeak-to-Misaki conversion](https://github.com/sahilmahendrakar/paradee/blob/main/web/misaki.js)
is a useful native-port reference, but sharing Kokoro's vocabulary alone is not
sufficient. Package/pin frontend data, normalize numbers/abbreviations, verify
phoneme equivalence and pronunciation, and count the complete installed footprint
rather than claiming the 9 MB weight file is the entire application dependency.
Model speed/quality figures are the author's measurements, not Omaspeak results.

Recommendation: implement and qualify a native OpenVINO CPU FP32 Paradee adapter
first, keep the existing default, then pursue the GPU conversion. Do not offer
Paradee as a runnable setup profile before its native G2P and complete synthesis
proof work. INT8 CPU, NPU and GGML backends are subsequent projects.
