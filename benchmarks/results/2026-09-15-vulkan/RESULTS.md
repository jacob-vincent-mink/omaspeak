# Intel Vulkan validation

These measurements use the published `v0.0.1-rc.3` binary, from source
`376fc5b90582668d4213a6a08982670b9a124e80`. Binary SHA-256:
`9f1a60ef94917544d4584924b54261c4f8fce79a81b3e4d610ab9f6f01466eb4`.
They are the Vulkan baseline for the follow-up release, not measurements of
its later setup and service changes.

The local host is an Intel Core Ultra X7 358H with an Arc B390 integrated GPU.
The combined audio.cpp provider was built from pinned revision
`e9ff20042ec85af960a720368c6927cda19ad65f` with Vulkan enabled and
`AUDIOCPP_MODELS=moonshine_asr,supertonic`. The provider enumerated
`Vulkan:0 Intel Arc B390 (PTL) [IGPU]`. Worker command lines confirmed the
exact provider, model, `vulkan` backend, and device `0`; setup also passed
model-backed validation with fallback disabled. These establish provider
selection and successful execution; hardware utilization counters were not
collected.

All runs used isolated configuration directories and files. No microphone,
audio playback, or wake-word actions were used. SHA-256 hashes of the local
raw reports and their summarized metrics are in [metrics.json](metrics.json).

## Same-model comparison

Both devices used Supertonic 3 GGUF, voice F3, and the text
“Vulkan synthesis on the Intel graphics processor.” The model SHA-256 was
`af814486a0bc9513fb36afabd9b1155ad14fb2c36a107ac6ffe62ea9adafb662`.

| Device | Benchmark model load | Warm synthesis p50 / p95 | Warm p50 RTF |
|---|---:|---:|---:|
| CPU | 1,474 ms | 1,945 / 1,947 ms | 0.5313 |
| Vulkan iGPU | 1,699 ms | 163 / 166 ms | 0.0446 |

Each benchmark measured five iterations after warmup. A separate fresh Vulkan
`say --no-play --out` run took 1,748 ms to load and 327 ms to synthesize.
Model load is a fresh process measurement with potentially warm OS/driver
caches, not a first-boot or first-ever shader compilation measurement.

The resulting Vulkan WAV was mono 44.1 kHz with 161,384 samples (3.6595 s).
Over their shared sample range, CPU and Vulkan outputs had correlation 0.9999856, mean absolute sample error
0.0001654, RMS amplitudes 0.056371 and 0.056347, and the same peak amplitude
0.422363. Their durations differed by 107 samples (2.43 ms).

This is a numerical quality sanity check for one sentence and voice. It does
not substitute for listening tests or broad language/voice intelligibility
validation. Vulkan was substantially faster for this tested synthesis workload;
it need not be faster for every model or sentence.

## Updated-build smoke test

The release build from `217d1fc` (including cancellation and setup/service
updates) also synthesized the same F3 sentence through the isolated Vulkan
configuration without playback. It produced 161,384 samples at 44.1 kHz,
with 1,720 ms model load and 179 ms synthesis. Binary SHA-256:
`d29f05e461d9bf57edf2ad9cf511d3db97fa976f883a53e05fadb0d1b02bf820`.
Output WAV SHA-256:
`432ef233871a8dda727447f2d48247c86c9523d2a0b3ca5cb6075216f9ed477b`.
This confirms the updated build still performs file-only Vulkan synthesis;
it is a smoke test, not another percentile benchmark.
